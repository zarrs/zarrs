//! Planned array read operations.
//!
//! Each intersecting chunk is planned as a task that decodes into its region of the output, and the
//! tasks of an operation are executed by one drive, with separate I/O and compute concurrency limits.
//!
//! Planning only needs owned state ([`ChunkPlanner`]), so tasks that decode into an owned output
//! are `'static`, and asynchronous reads can run them on Rayon.

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use zarrs_chunk_grid::ChunkGrid;
use zarrs_chunk_key_encoding::ChunkKeyEncoding;
use zarrs_codec::plan::{
    ByteSource, DecodeRequest, Generator, Join, Outcome, PlanIndexer, ReadBytes, SharedOutput,
    SourceKey, SyncIo, TargetViews, Task, TaskTarget, TaskValue,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayToBytesCodecTraits, CodecError, CodecOptions, CowBytes,
    InvalidNumberOfElementsError, Resources, copy_fill_value_into,
};
use zarrs_data_type::DataType;
use zarrs_storage::byte_range::ByteRange;
use zarrs_storage::{
    Bytes, ReadableStorage, ReadableStorageTraits, StorageError, StorageHandle, StoreKey,
};

use super::super::array_bytes_internal::{
    merge_chunks, optional_innermost, optional_nesting_depth,
};
use super::super::codec::CodecChainBound;
use crate::array::{
    Array, ArrayBytes, ArrayError, ArrayIndicesTinyVec, ArraySubset, ArraySubsetTraits, ChunkShape,
    ChunkShapeTraits, Indexer, subset_indices_at,
};
use crate::node::{NodePath, data_key};

mod partial_decoder;

pub(super) use partial_decoder::{array_to_codec_error, static_local_subchunk_grids};

#[cfg(feature = "async")]
pub(super) use async_planned::async_read_ranges;

pub(super) use zarrs_codec::plan::drive_blocking_one;

/// The chunks of an operation, which resolve the source keys of tasks to store keys.
pub(super) enum ChunkKeys {
    /// One chunk, with source key 0.
    One(StoreKey),
    /// The chunks of a region of the chunk grid. The source key of a chunk is its index in the region (in C order),
    /// and its store key is created when it is read.
    Grid {
        path: NodePath,
        encoding: ChunkKeyEncoding,
        chunks: ArraySubset,
    },
    /// Chunks by their indices. The source key of a chunk is its position in `indices`.
    List {
        path: NodePath,
        encoding: ChunkKeyEncoding,
        indices: Vec<ArrayIndicesTinyVec>,
    },
}

impl ChunkKeys {
    /// The store key of the chunk with source `key`.
    pub(super) fn store_key(&self, key: SourceKey) -> Result<Cow<'_, StoreKey>, StorageError> {
        match self {
            Self::One(store_key) => Ok(Cow::Borrowed(store_key)),
            Self::Grid {
                path,
                encoding,
                chunks,
            } => {
                let chunk_indices = subset_indices_at(chunks, key.0 as u64);
                let chunk_key = encoding
                    .encode(&chunk_indices)
                    .map_err(|err| StorageError::Other(err.to_string()))?;
                data_key(path, &chunk_key)
                    .map(Cow::Owned)
                    .map_err(StorageError::from)
            }
            Self::List {
                path,
                encoding,
                indices,
            } => {
                let chunk_indices = indices.get(key.0).ok_or_else(|| {
                    StorageError::Other(format!("no chunk with source key {}", key.0))
                })?;
                let chunk_key = encoding
                    .encode(chunk_indices)
                    .map_err(|err| StorageError::Other(err.to_string()))?;
                data_key(path, &chunk_key)
                    .map(Cow::Owned)
                    .map_err(StorageError::from)
            }
        }
    }
}

/// The synchronous I/O backend of the chunks of an array.
struct ChunkIo {
    storage: ReadableStorage,
    keys: ChunkKeys,
}

impl SyncIo for ChunkIo {
    fn read(
        &self,
        key: SourceKey,
        ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        read_ranges(&*self.storage, &*self.keys.store_key(key)?, ranges)
    }

    fn read_coalescing(&self) -> Option<zarrs_storage::ReadCoalescing> {
        self.storage.read_coalescing()
    }
}

/// Read `ranges` of the value of `key` in `storage`.
pub(super) fn read_ranges(
    storage: &dyn ReadableStorageTraits,
    key: &StoreKey,
    ranges: &[ByteRange],
) -> Result<Option<ReadBytes>, StorageError> {
    match ranges {
        [ByteRange::FromStart(0, None)] => Ok(storage.get(key)?.map(|bytes| smallvec![bytes])),
        [range] => Ok(storage
            .get_partial(key, *range)?
            .map(|bytes| smallvec![bytes])),
        ranges => {
            let Some(values) = storage.get_partial_many(key, Box::new(ranges.iter().copied()))?
            else {
                return Ok(None);
            };
            let mut bytes = SmallVec::with_capacity(ranges.len());
            for value in values {
                bytes.push(value?);
            }
            Ok(Some(bytes))
        }
    }
}

/// Convert an error of planning a chunk to a [`CodecError`].
///
/// Other errors (e.g. of chunk indices or subsets) are not expected for chunks within the array.
#[allow(clippy::wildcard_enum_match_arm)]
pub(super) fn plan_error(err: ArrayError) -> CodecError {
    match err {
        ArrayError::CodecError(err) => err,
        err => CodecError::Other(err.to_string()),
    }
}

/// The owned state for planning the decoding of the chunks of an array.
#[derive(Clone)]
pub(super) struct ChunkPlanner {
    chunk_grid: ChunkGrid,
    pub(super) codecs: Arc<CodecChainBound>,
    pub(super) data_type: DataType,
    pub(super) options: CodecOptions,
    /// Whether chunks of codecs that plan encode tasks of their own (e.g. a task per block of subchunks of a shard)
    /// are encoded by those tasks, rather than in one compute step.
    pub(super) encode_subtasks: bool,
}

impl ChunkPlanner {
    pub(super) fn chunk_subset(&self, chunk_indices: &[u64]) -> Result<ArraySubset, ArrayError> {
        super::chunk_grid_subset(&self.chunk_grid, chunk_indices)
    }

    /// Plan the decoding of `indexer` of the chunk with source `key` and `chunk_shape` into `target`.
    fn plan_chunk<'a>(
        &self,
        key: SourceKey,
        chunk_shape: &[NonZeroU64],
        indexer: PlanIndexer<'a>,
        target: TaskTarget<'a>,
    ) -> Result<Task<'a>, CodecError> {
        self.codecs.clone().plan_decode(
            ByteSource::external(key),
            chunk_shape,
            DecodeRequest { indexer, target },
            &self.options,
        )
    }

    /// Plan the decoding of `array_subset` into `views` (with the shape of the subset), with a task per chunk in `chunks`.
    ///
    /// The source keys of the chunks are their indices in `chunks`.
    fn plan_into<'a>(
        &self,
        array_subset: ArraySubset,
        chunks: ArraySubset,
        views: TargetViews<'a>,
    ) -> Result<Task<'a>, CodecError> {
        let output_start = views.subset().start().to_vec();
        let chunks_len = chunks.num_elements_usize();
        let planner = self.clone();
        let tasks = Generator::indexed(chunks_len, move |i| {
            let chunk_indices = subset_indices_at(&chunks, i as u64);
            let chunk_subset = planner.chunk_subset(&chunk_indices).map_err(plan_error)?;
            // The overlap of the chunk with the subset, relative to the chunk and positioned in the output
            let dimensionality = chunk_subset.dimensionality();
            let mut overlap_start = Vec::with_capacity(dimensionality);
            let mut overlap_shape = Vec::with_capacity(dimensionality);
            let mut output_subset_start = Vec::with_capacity(dimensionality);
            for (dim, ((&chunk_start, &chunk_len), (&subset_start, &subset_len))) in
                std::iter::zip(chunk_subset.start(), chunk_subset.shape())
                    .zip(std::iter::zip(array_subset.start(), array_subset.shape()))
                    .enumerate()
            {
                let start = chunk_start.max(subset_start);
                let end = (chunk_start + chunk_len).min(subset_start + subset_len);
                overlap_start.push(start - chunk_start);
                overlap_shape.push(end.saturating_sub(start));
                output_subset_start.push(start - subset_start + output_start[dim]);
            }
            let output_subset =
                ArraySubset::new_with_start_shape(output_subset_start, overlap_shape.clone())?;
            // SAFETY: the overlaps of chunks with the subset are disjoint, and the views are only subdivided
            let chunk_views = unsafe { views.subdivide(output_subset) }?;
            planner
                .plan_chunk(
                    SourceKey(i),
                    subset_chunk_shape(&chunk_subset).map_err(plan_error)?,
                    PlanIndexer::Subset(ArraySubset::new_with_start_shape(
                        overlap_start,
                        overlap_shape,
                    )?),
                    TaskTarget::Views(chunk_views),
                )
                .map(Some)
        });
        if chunks_len == 1 {
            // A single chunk is planned directly, without a generator
            return Ok(tasks
                .plan_index(0)?
                .unwrap_or_else(|| Task::compute(|_| Ok(Outcome::Done))));
        }
        Ok(Task::compute(move |_| {
            Ok(Outcome::Generate {
                children: tasks,
                join: None,
            })
        }))
    }

    /// Plan the decoding of `array_subset` as the values of tasks, one per chunk in `chunks`, and merge them.
    fn plan_values(
        &self,
        array_subset: ArraySubset,
        chunks: ArraySubset,
    ) -> Result<Task<'static>, ArrayError> {
        let overlaps = chunks
            .indices()
            .iter()
            .map(|chunk_indices| {
                Ok(self
                    .chunk_subset(&chunk_indices)?
                    .overlap(&array_subset)?
                    .relative_to(array_subset.start())?)
            })
            .collect::<Result<Vec<_>, ArrayError>>()?;
        let output_shape = array_subset.shape().to_vec();
        let planner = self.clone();
        let tasks = Generator::indexed(chunks.num_elements_usize(), move |i| {
            let chunk_indices = subset_indices_at(&chunks, i as u64);
            let chunk_subset = planner.chunk_subset(&chunk_indices).map_err(plan_error)?;
            let overlap = chunk_subset.overlap(&array_subset)?;
            planner
                .plan_chunk(
                    SourceKey(i),
                    subset_chunk_shape(&chunk_subset).map_err(plan_error)?,
                    PlanIndexer::Subset(overlap.relative_to(chunk_subset.start())?),
                    TaskTarget::Value,
                )
                .map(Some)
        });
        let data_type = self.data_type.clone();
        Ok(Task::compute(move |_| {
            Ok(Outcome::Generate {
                children: tasks,
                join: Some(Join::new(move |values, _ctx| {
                    let chunks = std::iter::zip(values, overlaps)
                        .map(|(value, overlap)| Ok((array_value(value)?, overlap)))
                        .collect::<Result<Vec<_>, CodecError>>()?;
                    Ok(Outcome::Value(TaskValue::Array(merge_chunks(
                        chunks,
                        &output_shape,
                        &data_type,
                    )?)))
                })),
            })
        }))
    }
}

/// A planned read of an array subset: the task to drive (if any), and how to finish the read.
struct SubsetPlan {
    task: Option<Task<'static>>,
    finish: SubsetFinish,
}

/// How a [`SubsetPlan`] produces the elements of the subset.
enum SubsetFinish {
    /// The fill value of an empty subset.
    Empty(ArrayBytes<'static>),
    /// The value of the task.
    Value,
    /// The output the task decodes into.
    Into(Arc<SharedOutput>),
}

impl SubsetFinish {
    /// The elements of the subset, from the `value` of the task of the plan.
    fn finish(self, value: Option<TaskValue<'static>>) -> Result<ArrayBytes<'static>, ArrayError> {
        match self {
            Self::Empty(bytes) => Ok(bytes),
            Self::Value => Ok(array_value(value.unwrap_or(TaskValue::None))?.into_owned()),
            Self::Into(output) => {
                // SAFETY: the chunks intersecting the subset cover it, and each task writes all elements of its region
                unsafe { output.into_array_bytes() }.map_err(|_| {
                    CodecError::Other("a chunk output view outlived its task".to_string()).into()
                })
            }
        }
    }
}

impl<TStorage: ?Sized> Array<TStorage> {
    /// The owned state for planning the decoding of the chunks of the array with `options`.
    pub(super) fn chunk_planner(&self, options: &CodecOptions) -> ChunkPlanner {
        ChunkPlanner {
            chunk_grid: self.chunk_grid().clone(),
            codecs: self.codecs_bound().clone(),
            data_type: self.data_type().clone(),
            options: *options,
            encode_subtasks: true,
        }
    }

    /// The keys of the chunks of the region `chunks` of the chunk grid.
    pub(super) fn grid_keys(&self, chunks: ArraySubset) -> ChunkKeys {
        ChunkKeys::Grid {
            path: self.path().clone(),
            encoding: self.chunk_key_encoding().clone(),
            chunks,
        }
    }

    /// The keys of the chunks selected by `chunks`, in the order of `chunks.iter_indices()`.
    pub(super) fn indexer_keys(&self, chunks: &dyn Indexer) -> ChunkKeys {
        match chunks.as_array_subset() {
            Some(subset) => self.grid_keys(subset.to_array_subset()),
            None => ChunkKeys::List {
                path: self.path().clone(),
                encoding: self.chunk_key_encoding().clone(),
                indices: chunks.iter_indices().collect(),
            },
        }
    }

    /// The key of the chunk at `chunk_indices`.
    /// Plan reading the encoded subchunk at `subchunk_indices` of the shard at `chunk_indices`, and the keys of the
    /// shard.
    ///
    /// The array must have a `sharding_indexed` array-to-bytes codec and no bytes-to-bytes codecs.
    pub(super) fn plan_encoded_subchunk(
        &self,
        chunk_indices: &[u64],
        subchunk_indices: &[u64],
    ) -> Result<(Task<'static>, ChunkKeys), ArrayError> {
        let codecs = self.codecs_bound();
        let not_addressable = || {
            ArrayError::CodecError(CodecError::Other(
                "encoded subchunks are only addressable in arrays with a sharding_indexed array-to-bytes codec and no bytes-to-bytes codecs"
                    .to_string(),
            ))
        };
        if !codecs.bytes_to_bytes_codecs().is_empty() {
            return Err(not_addressable());
        }
        let sharding = codecs
            .array_to_bytes_codec()
            .as_any()
            .downcast_ref::<crate::array::codec::ShardingCodecBound>()
            .ok_or_else(not_addressable)?
            .arc();
        let mut shape = self.chunk_shape(chunk_indices)?;
        for codec in codecs.array_to_array_codecs() {
            shape = codec.encoded_shape(&shape)?;
        }
        let task = sharding.plan_retrieve_subchunk_encoded(
            ByteSource::external(SourceKey(0)),
            &shape,
            subchunk_indices,
            self.codec_options(),
        )?;
        Ok((task, self.one_key(chunk_indices)?))
    }

    pub(super) fn one_key(&self, chunk_indices: &[u64]) -> Result<ChunkKeys, ArrayError> {
        Ok(ChunkKeys::One(self.chunk_key(chunk_indices)?))
    }

    /// The chunks of `array_subset`.
    pub(super) fn planned_chunks(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<ArraySubset, ArrayError> {
        let invalid = || {
            ArrayError::InvalidArraySubset(array_subset.to_array_subset(), self.shape().to_vec())
        };
        if array_subset.dimensionality() != self.dimensionality() {
            return Err(invalid());
        }
        self.chunks_in_array_subset(array_subset)?
            .ok_or_else(invalid)
    }

    /// Plan reading `array_subset`, and the I/O backend of its chunks.
    fn plan_array_subset(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<(SubsetPlan, ChunkKeys), ArrayError> {
        let chunks = self.planned_chunks(array_subset)?;
        let data_type = self.data_type();
        let array_subset = array_subset.to_array_subset();
        if array_subset.num_elements() == 0 {
            let bytes = ArrayBytes::new_fill_value(data_type, 0, self.fill_value())
                .map_err(CodecError::from)?;
            let plan = SubsetPlan {
                task: None,
                finish: SubsetFinish::Empty(bytes),
            };
            return Ok((plan, self.grid_keys(chunks)));
        }
        if chunks.num_elements() == 1 {
            // The elements of a single chunk are its value, which may be decoded without a copy
            let (task, keys) =
                self.plan_one_chunk(chunks.start(), &array_subset, TaskTarget::Value, options)?;
            let plan = SubsetPlan {
                task: Some(task),
                finish: SubsetFinish::Value,
            };
            return Ok((plan, keys));
        }
        let planner = self.chunk_planner(options);
        let keys = self.grid_keys(chunks.clone());
        let Some(data_type_size) = optional_innermost(data_type).fixed_size() else {
            // Variable-length data is decoded as values, then merged
            let plan = SubsetPlan {
                task: Some(planner.plan_values(array_subset, chunks)?),
                finish: SubsetFinish::Value,
            };
            return Ok((plan, keys));
        };
        // Fixed-length data decodes directly into an output allocated once
        let (output, views) = SharedOutput::new(
            data_type_size,
            array_subset.shape(),
            optional_nesting_depth(data_type),
        );
        let plan = SubsetPlan {
            task: Some(planner.plan_into(array_subset, chunks, views)?),
            finish: SubsetFinish::Into(output),
        };
        Ok((plan, keys))
    }

    /// Plan reading `array_subset` into `output_target`, and the I/O backend of its chunks.
    fn plan_array_subset_into<'a>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: &'a ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(Task<'a>, ChunkKeys), ArrayError> {
        let chunks = self.planned_chunks(array_subset)?;
        if !self.data_type().is_fixed() {
            return Err(ArrayError::CodecError(CodecError::Other(
                "retrieve_array_subset_into does not support variable-length data types"
                    .to_string(),
            )));
        }
        if output_target.num_elements() != array_subset.num_elements() {
            return Err(ArrayError::CodecError(
                InvalidNumberOfElementsError::new(
                    output_target.num_elements(),
                    array_subset.num_elements(),
                )
                .into(),
            ));
        }
        let views = whole_views(output_target)?;
        if chunks.num_elements() == 1 {
            // The elements of one chunk decode into all of the output in order
            self.plan_one_chunk(
                chunks.start(),
                array_subset,
                TaskTarget::Views(views),
                options,
            )
        } else if views.subset().shape() == &*array_subset.shape() {
            let keys = self.grid_keys(chunks.clone());
            let task = self.chunk_planner(options).plan_into(
                array_subset.to_array_subset(),
                chunks,
                views,
            )?;
            Ok((task, keys))
        } else {
            Err(ArrayError::CodecError(CodecError::Other(
                "the output target shape must match the array subset shape".to_string(),
            )))
        }
    }

    /// Plan reading the elements of `indexer` of the chunk at `chunk_indices` with `chunk_shape` into `target`.
    fn plan_partial_chunk<'a>(
        &self,
        chunk_indices: &[u64],
        chunk_shape: &[NonZeroU64],
        indexer: PlanIndexer<'a>,
        target: TaskTarget<'a>,
        options: &CodecOptions,
    ) -> Result<(Task<'a>, ChunkKeys), ArrayError> {
        let task =
            self.chunk_planner(options)
                .plan_chunk(SourceKey(0), chunk_shape, indexer, target)?;
        Ok((task, self.one_key(chunk_indices)?))
    }

    /// Plan reading the chunk at `chunk_indices` if it exists: the value of the task is the decoded chunk, or
    /// [`TaskValue::None`] if the chunk does not exist.
    #[cfg(feature = "async")]
    fn plan_chunk_if_exists(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<(Task<'static>, ChunkKeys), ArrayError> {
        let shape: SmallVec<[NonZeroU64; 4]> = self.chunk_shape(chunk_indices)?.as_slice().into();
        let codecs = self.codecs_bound().clone();
        let source = ByteSource::external(SourceKey(0));
        let hint = zarrs_codec::plan::whole_chunk_hint(&*codecs, &source, &shape);
        let options = *options;
        let task = Task::read(
            source,
            [ByteRange::FromStart(0, None)],
            move |mut reads, ctx| {
                let Some(encoded) = reads.pop().flatten() else {
                    return Ok(Outcome::Value(TaskValue::None));
                };
                // The chunk is decoded in one compute step: chunk caches fill each chunk of an operation in a
                // separate drive, so subchunks are not planned as tasks of their own
                let bytes = codecs.decode(encoded, &shape, &options, ctx.resources())?;
                Ok(Outcome::Value(TaskValue::Array(bytes)))
            },
        )
        .with_hint(hint);
        Ok((task, self.one_key(chunk_indices)?))
    }

    /// Plan reading the elements of `array_subset` of the chunk at `chunk_indices` into `target`.
    fn plan_one_chunk<'a>(
        &self,
        chunk_indices: &[u64],
        array_subset: &dyn ArraySubsetTraits,
        target: TaskTarget<'a>,
        options: &CodecOptions,
    ) -> Result<(Task<'a>, ChunkKeys), ArrayError> {
        let chunk_subset = self.chunk_subset(chunk_indices)?;
        let task = self.codecs_bound().clone().plan_decode(
            ByteSource::external(SourceKey(0)),
            subset_chunk_shape(&chunk_subset)?,
            DecodeRequest {
                indexer: PlanIndexer::Subset(array_subset.relative_to(chunk_subset.start())?),
                target,
            },
            options,
        )?;
        Ok((task, self.one_key(chunk_indices)?))
    }

    /// The indices of the chunk that is all of `array_subset`, if its codecs plan no tasks of their own, so it can be
    /// read with one read and one decode.
    fn whole_single_chunk(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<Option<Vec<u64>>, ArrayError> {
        let chunks = self.planned_chunks(array_subset)?;
        if chunks.num_elements() != 1 || self.codecs_bound().plans_subtasks() {
            return Ok(None);
        }
        let chunk_indices = chunks.start();
        let chunk_subset = self.chunk_subset(chunk_indices)?;
        Ok((chunk_subset.start() == &*array_subset.start()
            && chunk_subset.shape() == &*array_subset.shape())
            .then(|| chunk_indices.to_vec()))
    }

    /// The shape of the chunk at `chunk_indices`, and an array subset of all of its elements.
    pub(crate) fn whole_chunk_indexer(
        &self,
        chunk_indices: &[u64],
    ) -> Result<(ChunkShape, ArraySubset), ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let subset = ArraySubset::new_with_shape(bytemuck::must_cast_slice(&chunk_shape).to_vec());
        Ok((chunk_shape, subset))
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits + 'static> Array<TStorage> {
    /// The I/O backend of the chunks of `keys`.
    fn chunk_io(&self, keys: ChunkKeys) -> Result<ChunkIo, ArrayError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        Ok(ChunkIo {
            storage: self
                .storage_transformers()
                .create_readable_transformer(storage_handle)?,
            keys,
        })
    }

    /// Read the whole encoded chunk at `chunk_indices` under the I/O limit of `resources`.
    ///
    /// This is the plan of a whole chunk whose codecs plan no tasks of their own: one read, then one decode.
    fn read_chunk_planned(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<Bytes>, ArrayError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage = self
            .storage_transformers()
            .create_readable_transformer(storage_handle)?;
        let key = self.chunk_key(chunk_indices)?;
        let _permit = resources.io_permit_blocking();
        Ok(storage.get(&key)?)
    }

    /// Retrieve the encoded chunks selected by `chunks` (validated) with a planned read.
    pub(crate) fn retrieve_encoded_chunks_planned(
        &self,
        chunks: &dyn Indexer,
        resources: &Resources,
    ) -> Result<Vec<Option<Bytes>>, ArrayError> {
        let num_chunks = usize::try_from(chunks.len()).unwrap();
        if num_chunks == 0 {
            return Ok(vec![]);
        }
        let io = self.chunk_io(self.indexer_keys(chunks))?;
        encoded_chunks(drive_blocking_one(
            plan_encoded_chunks(num_chunks),
            &io,
            resources,
        )?)
    }

    /// Retrieve the encoded subchunk at `subchunk_indices` of the shard at `chunk_indices` with a planned read.
    pub(crate) fn retrieve_encoded_subchunk_planned(
        &self,
        chunk_indices: &[u64],
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<Bytes>, ArrayError> {
        let (task, keys) = self.plan_encoded_subchunk(chunk_indices, subchunk_indices)?;
        encoded_subchunk(drive_blocking_one(task, &self.chunk_io(keys)?, resources)?)
    }

    /// Retrieve the chunk at `chunk_indices` with a planned read, or [`None`] if it does not exist.
    ///
    /// The chunk is decoded in one compute step: chunk caches fill each chunk of an operation separately, so the
    /// subchunks of a shard are not planned as tasks of their own.
    pub(crate) fn retrieve_chunk_if_exists_planned(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<ArrayBytes<'static>>, ArrayError> {
        let shape = self.chunk_shape(chunk_indices)?;
        self.read_chunk_planned(chunk_indices, resources)?
            .map(|encoded| {
                Ok(self
                    .codecs_bound()
                    .decode(CowBytes::Shared(encoded), &shape, options, resources)?
                    .into_owned())
            })
            .transpose()
    }

    /// Retrieve the chunk at `chunk_indices` with a planned read, and its shape.
    pub(crate) fn retrieve_chunk_planned(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(ArrayBytes<'static>, ChunkShape), ArrayError> {
        let codecs = self.codecs_bound();
        if codecs.plans_subtasks() {
            let (shape, indexer) = self.whole_chunk_indexer(chunk_indices)?;
            let bytes = self.retrieve_partial_chunk_planned(
                chunk_indices,
                &shape,
                &indexer,
                options,
                resources,
            )?;
            return Ok((bytes, shape));
        }
        let shape = self.chunk_shape(chunk_indices)?;
        let bytes = match self.read_chunk_planned(chunk_indices, resources)? {
            Some(encoded) => {
                let bytes = codecs
                    .decode(CowBytes::Shared(encoded), &shape, options, resources)?
                    .into_owned();
                bytes.validate(shape.num_elements_u64(), self.data_type())?;
                bytes
            }
            None => ArrayBytes::new_fill_value(
                self.data_type(),
                shape.num_elements_u64(),
                self.fill_value(),
            )
            .map_err(CodecError::from)?,
        };
        Ok((bytes, shape))
    }

    /// Retrieve the chunk at `chunk_indices` into `output_target` with a planned read.
    pub(crate) fn retrieve_chunk_into_planned(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let (chunk_shape, indexer) = self.whole_chunk_indexer(chunk_indices)?;
        if self.codecs_bound().plans_subtasks() {
            return self.retrieve_partial_chunk_into_planned(
                chunk_indices,
                &chunk_shape,
                &indexer,
                output_target,
                options,
                resources,
            );
        }
        check_target_len(&indexer, &output_target)?;
        match self.read_chunk_planned(chunk_indices, resources)? {
            Some(encoded) => self.codecs_bound().decode_into(
                CowBytes::Shared(encoded).into(),
                &chunk_shape,
                output_target,
                options,
                resources,
            )?,
            None => copy_fill_value_into(self.data_type(), self.fill_value(), output_target)?,
        }
        Ok(())
    }

    /// Retrieve `array_subset` with a planned read.
    pub(crate) fn retrieve_array_subset_planned(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        if let Some(chunk_indices) = self.whole_single_chunk(array_subset)? {
            return Ok(self
                .retrieve_chunk_planned(&chunk_indices, options, resources)?
                .0);
        }
        let (SubsetPlan { task, finish }, keys) = self.plan_array_subset(array_subset, options)?;
        let value = match task {
            Some(task) => Some(drive_blocking_one(task, &self.chunk_io(keys)?, resources)?),
            None => None,
        };
        let bytes = finish.finish(value)?;
        bytes.validate(array_subset.num_elements(), self.data_type())?;
        Ok(bytes)
    }

    /// Retrieve `array_subset` into `output_target` with a planned read.
    pub(crate) fn retrieve_array_subset_into_planned(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        if let Some(chunk_indices) = self.whole_single_chunk(array_subset)? {
            return self.retrieve_chunk_into_planned(
                &chunk_indices,
                output_target,
                options,
                resources,
            );
        }
        let (task, keys) = self.plan_array_subset_into(array_subset, &output_target, options)?;
        drive_blocking_one(task, &self.chunk_io(keys)?, resources)?;
        Ok(())
    }

    /// Retrieve the elements of `indexer` of the chunk at `chunk_indices` with `chunk_shape` with a planned read.
    pub(crate) fn retrieve_partial_chunk_planned(
        &self,
        chunk_indices: &[u64],
        chunk_shape: &[NonZeroU64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        let (task, keys) = self.plan_partial_chunk(
            chunk_indices,
            chunk_shape,
            PlanIndexer::Borrowed(indexer),
            TaskTarget::Value,
            options,
        )?;
        let bytes =
            array_value(drive_blocking_one(task, &self.chunk_io(keys)?, resources)?)?.into_owned();
        bytes.validate(indexer.len(), self.data_type())?;
        Ok(bytes)
    }

    /// Retrieve the elements of `indexer` of the chunk at `chunk_indices` with `chunk_shape` into `output_target` with a planned read.
    pub(crate) fn retrieve_partial_chunk_into_planned(
        &self,
        chunk_indices: &[u64],
        chunk_shape: &[NonZeroU64],
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        check_target_len(indexer, &output_target)?;
        let views = whole_views(&output_target)?;
        let (task, keys) = self.plan_partial_chunk(
            chunk_indices,
            chunk_shape,
            PlanIndexer::Borrowed(indexer),
            TaskTarget::Views(views),
            options,
        )?;
        drive_blocking_one(task, &self.chunk_io(keys)?, resources)?;
        Ok(())
    }
}

/// Views of all of `output_target`.
/// Plan reading the encoded chunks with source keys `0..num_chunks`.
///
/// The value of the task is the encoded chunks, or [`None`] for chunks that do not exist (see [`encoded_chunks`]).
pub(super) fn plan_encoded_chunks(num_chunks: usize) -> Task<'static> {
    let tasks = Generator::indexed(num_chunks, |i| {
        Ok(Some(Task::read(
            ByteSource::external(SourceKey(i)),
            [ByteRange::FromStart(0, None)],
            |mut reads, _ctx| Ok(Outcome::Value(TaskValue::Bytes(reads.pop().flatten()))),
        )))
    });
    Task::compute(move |_| {
        Ok(Outcome::Generate {
            children: tasks,
            join: Some(Join::new(|values, _ctx| {
                let chunks: Vec<Option<Bytes>> = values
                    .into_iter()
                    .map(|value| value.into_bytes().flatten().map(CowBytes::into_bytes))
                    .collect();
                Ok(Outcome::Value(TaskValue::Any(Box::new(chunks))))
            })),
        })
    })
}

/// The encoded chunks of the value of a task planned by [`plan_encoded_chunks`].
pub(super) fn encoded_chunks(value: TaskValue<'_>) -> Result<Vec<Option<Bytes>>, ArrayError> {
    value
        .downcast()
        .ok_or_else(|| CodecError::Other("expected encoded chunks".to_string()).into())
}

/// The encoded subchunk that is the value of a task planned by [`Array::plan_encoded_subchunk`].
pub(super) fn encoded_subchunk(value: TaskValue<'_>) -> Result<Option<Bytes>, ArrayError> {
    let bytes = value
        .into_bytes()
        .ok_or_else(|| CodecError::Other("expected an encoded subchunk".to_string()))?;
    Ok(bytes.map(CowBytes::into_bytes))
}

pub(super) fn whole_views<'a>(
    output_target: &'a ArrayBytesDecodeIntoTarget<'_>,
) -> Result<TargetViews<'a>, ArrayError> {
    // SAFETY: the whole views are subdivided once, and the parent views are not used while the tasks run
    Ok(unsafe { TargetViews::from_target(output_target) }?)
}

pub(super) fn check_target_len(
    indexer: &dyn Indexer,
    output_target: &ArrayBytesDecodeIntoTarget<'_>,
) -> Result<(), ArrayError> {
    if indexer.len() == output_target.num_elements() {
        Ok(())
    } else {
        Err(ArrayError::CodecError(
            InvalidNumberOfElementsError::new(indexer.len(), output_target.num_elements()).into(),
        ))
    }
}

/// The decoded chunk of a task of [`Array::plan_chunk_if_exists`].
#[cfg(feature = "async")]
fn chunk_if_exists(value: TaskValue<'static>) -> Result<Option<ArrayBytes<'static>>, ArrayError> {
    match value {
        TaskValue::None => Ok(None),
        value @ (TaskValue::Array(_)
        | TaskValue::Bytes(_)
        | TaskValue::Prepared(_)
        | TaskValue::Any(_)) => Ok(Some(array_value(value)?)),
    }
}

pub(super) fn array_value(value: TaskValue<'_>) -> Result<ArrayBytes<'_>, CodecError> {
    value
        .into_array()
        .ok_or_else(|| CodecError::Other("expected decoded chunk array bytes".to_string()))
}

/// The shape of a chunk with `chunk_subset`.
pub(super) fn subset_chunk_shape(chunk_subset: &ArraySubset) -> Result<&[NonZeroU64], ArrayError> {
    bytemuck::checked::try_cast_slice(chunk_subset.shape()).map_err(|_| {
        ArrayError::CodecError(CodecError::Other(
            "a chunk has a zero-sized dimension".to_string(),
        ))
    })
}

#[cfg(feature = "async")]
mod async_planned {
    use zarrs_codec::plan::{AsyncIo, drive_async_one};
    use zarrs_storage::{AsyncReadableStorage, AsyncReadableStorageTraits};

    use super::{
        Array, ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayError, ArraySubset, ArraySubsetTraits,
        ByteRange, Bytes, ChunkKeys, CodecOptions, Indexer, NonZeroU64, PlanIndexer, ReadBytes,
        Resources, SmallVec, SourceKey, StorageError, StorageHandle, StoreKey, SubsetPlan, Task,
        TaskTarget, TaskValue, array_value, check_target_len, chunk_if_exists, encoded_chunks,
        encoded_subchunk, plan_encoded_chunks, smallvec, whole_views,
    };

    /// The asynchronous I/O backend of the chunks of an array.
    pub(super) struct AsyncChunkIo {
        pub(super) storage: AsyncReadableStorage,
        pub(super) keys: ChunkKeys,
    }

    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    impl AsyncIo for AsyncChunkIo {
        async fn read(
            &self,
            key: SourceKey,
            ranges: &[ByteRange],
        ) -> Result<Option<ReadBytes>, StorageError> {
            async_read_ranges(&*self.storage, &*self.keys.store_key(key)?, ranges).await
        }

        fn read_coalescing(&self) -> Option<zarrs_storage::ReadCoalescing> {
            self.storage.read_coalescing()
        }
    }

    /// Read `ranges` of the value of `key` in `storage`.
    pub(in super::super) async fn async_read_ranges(
        storage: &dyn AsyncReadableStorageTraits,
        key: &StoreKey,
        ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        use futures::StreamExt;
        match ranges {
            [ByteRange::FromStart(0, None)] => {
                Ok(storage.get(key).await?.map(|bytes| smallvec![bytes]))
            }
            [range] => Ok(storage
                .get_partial(key, *range)
                .await?
                .map(|bytes| smallvec![bytes])),
            ranges => {
                let Some(mut values) = storage
                    .get_partial_many(key, Box::new(ranges.iter().copied()))
                    .await?
                else {
                    return Ok(None);
                };
                let mut bytes = SmallVec::with_capacity(ranges.len());
                while let Some(value) = values.next().await {
                    bytes.push(value?);
                }
                Ok(Some(bytes))
            }
        }
    }

    /// Drive a task that does not borrow, with compute offloaded to Rayon where supported.
    pub(super) async fn drive_owned(
        task: Task<'static>,
        io: &AsyncChunkIo,
        resources: &Resources,
    ) -> Result<TaskValue<'static>, ArrayError> {
        #[cfg(not(target_arch = "wasm32"))]
        return Ok(zarrs_codec::plan::drive_async_offload_one(task, io, resources).await?);
        #[cfg(target_arch = "wasm32")]
        return Ok(drive_async_one(task, io, resources).await?);
    }

    impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> Array<TStorage> {
        /// The I/O backend of the chunks of `keys`.
        pub(super) async fn async_chunk_io(
            &self,
            keys: ChunkKeys,
        ) -> Result<AsyncChunkIo, ArrayError> {
            let storage_handle = std::sync::Arc::new(StorageHandle::new(self.storage.clone()));
            Ok(AsyncChunkIo {
                storage: self
                    .storage_transformers()
                    .create_async_readable_transformer(storage_handle)
                    .await?,
                keys,
            })
        }

        /// Retrieve the encoded chunks selected by `chunks` (validated) with a planned read.
        pub(crate) async fn async_retrieve_encoded_chunks_planned(
            &self,
            chunks: &dyn Indexer,
            resources: &Resources,
        ) -> Result<Vec<Option<Bytes>>, ArrayError> {
            let num_chunks = usize::try_from(chunks.len()).unwrap();
            if num_chunks == 0 {
                return Ok(vec![]);
            }
            let io = self.async_chunk_io(self.indexer_keys(chunks)).await?;
            encoded_chunks(drive_async_one(plan_encoded_chunks(num_chunks), &io, resources).await?)
        }

        /// Retrieve the encoded subchunk at `subchunk_indices` of the shard at `chunk_indices` with a planned read.
        pub(crate) async fn async_retrieve_encoded_subchunk_planned(
            &self,
            chunk_indices: &[u64],
            subchunk_indices: &[u64],
            resources: &Resources,
        ) -> Result<Option<Bytes>, ArrayError> {
            let (task, keys) = self.plan_encoded_subchunk(chunk_indices, subchunk_indices)?;
            let io = self.async_chunk_io(keys).await?;
            encoded_subchunk(drive_async_one(task, &io, resources).await?)
        }

        /// Retrieve the chunk at `chunk_indices` with a planned read, or [`None`] if it does not exist.
        ///
        /// Compute is offloaded to Rayon, since the tasks decode into an owned output.
        pub(crate) async fn async_retrieve_chunk_if_exists_planned(
            &self,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Option<ArrayBytes<'static>>, ArrayError> {
            let (task, keys) = self.plan_chunk_if_exists(chunk_indices, options)?;
            let io = self.async_chunk_io(keys).await?;
            chunk_if_exists(drive_owned(task, &io, resources).await?)
        }

        /// Retrieve `array_subset` with a planned read.
        ///
        /// Compute is offloaded to Rayon, since the tasks decode into an owned output.
        pub(crate) async fn async_retrieve_array_subset_planned(
            &self,
            array_subset: &dyn ArraySubsetTraits,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<ArrayBytes<'static>, ArrayError> {
            let (SubsetPlan { task, finish }, keys) =
                self.plan_array_subset(array_subset, options)?;
            let value = match task {
                Some(task) => {
                    let io = self.async_chunk_io(keys).await?;
                    Some(drive_owned(task, &io, resources).await?)
                }
                None => None,
            };
            let bytes = finish.finish(value)?;
            bytes.validate(array_subset.num_elements(), self.data_type())?;
            Ok(bytes)
        }

        /// Retrieve `array_subset` into `output_target` with a planned read.
        ///
        /// Compute runs on the polling task, since the tasks borrow the output.
        pub(crate) async fn async_retrieve_array_subset_into_planned(
            &self,
            array_subset: &dyn ArraySubsetTraits,
            output_target: ArrayBytesDecodeIntoTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            let (task, keys) =
                self.plan_array_subset_into(array_subset, &output_target, options)?;
            let io = self.async_chunk_io(keys).await?;
            drive_async_one(task, &io, resources).await?;
            Ok(())
        }

        /// Retrieve the chunk at `chunk_indices` with a planned read, with compute offloaded to Rayon.
        pub(crate) async fn async_retrieve_chunk_planned(
            &self,
            chunk_indices: &[u64],
            chunk_shape: &[NonZeroU64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<ArrayBytes<'static>, ArrayError> {
            let indexer =
                ArraySubset::new_with_shape(bytemuck::must_cast_slice(chunk_shape).to_vec());
            let num_elements = indexer.num_elements();
            let (task, keys) = self.plan_partial_chunk(
                chunk_indices,
                chunk_shape,
                PlanIndexer::Subset(indexer),
                TaskTarget::Value,
                options,
            )?;
            let io = self.async_chunk_io(keys).await?;
            let bytes = array_value(drive_owned(task, &io, resources).await?)?.into_owned();
            bytes.validate(num_elements, self.data_type())?;
            Ok(bytes)
        }

        /// Retrieve the elements of `indexer` of the chunk at `chunk_indices` with `chunk_shape` with a planned read.
        ///
        /// Compute is offloaded to Rayon if `indexer` is an array subset (which the tasks can own).
        pub(crate) async fn async_retrieve_partial_chunk_planned(
            &self,
            chunk_indices: &[u64],
            chunk_shape: &[NonZeroU64],
            indexer: &dyn Indexer,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<ArrayBytes<'static>, ArrayError> {
            let value = if let Some(subset) = indexer.as_array_subset() {
                let (task, keys) = self.plan_partial_chunk(
                    chunk_indices,
                    chunk_shape,
                    PlanIndexer::Subset(subset.to_array_subset()),
                    TaskTarget::Value,
                    options,
                )?;
                let io = self.async_chunk_io(keys).await?;
                array_value(drive_owned(task, &io, resources).await?)?.into_owned()
            } else {
                let (task, keys) = self.plan_partial_chunk(
                    chunk_indices,
                    chunk_shape,
                    PlanIndexer::Borrowed(indexer),
                    TaskTarget::Value,
                    options,
                )?;
                let io = self.async_chunk_io(keys).await?;
                array_value(drive_async_one(task, &io, resources).await?)?.into_owned()
            };
            value.validate(indexer.len(), self.data_type())?;
            Ok(value)
        }

        /// Retrieve the elements of `indexer` of the chunk at `chunk_indices` with `chunk_shape` into `output_target` with a planned read.
        pub(crate) async fn async_retrieve_partial_chunk_into_planned(
            &self,
            chunk_indices: &[u64],
            chunk_shape: &[NonZeroU64],
            indexer: &dyn Indexer,
            output_target: ArrayBytesDecodeIntoTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            check_target_len(indexer, &output_target)?;
            let views = whole_views(&output_target)?;
            let (task, keys) = self.plan_partial_chunk(
                chunk_indices,
                chunk_shape,
                PlanIndexer::Borrowed(indexer),
                TaskTarget::Views(views),
                options,
            )?;
            let io = self.async_chunk_io(keys).await?;
            drive_async_one(task, &io, resources).await?;
            Ok(())
        }
    }
}
