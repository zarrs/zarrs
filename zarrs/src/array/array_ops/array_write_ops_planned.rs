//! Planned array write operations.
//!
//! Each chunk written is a task that encodes the chunk (after reading, decoding and updating it, for a partial
//! write) and then writes or erases it. The tasks of an operation are executed by one drive, with separate I/O and
//! compute concurrency limits.

use std::num::NonZeroU64;
use std::sync::Arc;

use smallvec::SmallVec;
use zarrs_codec::plan::{
    Generator, MultipartId, MultipartOp, MultipartWriters, Outcome, PlanIndexer, ReadBytes,
    SourceKey, SyncIo, Task, TaskValue, WriteOp, encode_and_write, encode_hint,
};
use zarrs_codec::{ArrayToBytesCodecTraits, CodecOptions, MultipartWrite, Resources};
use zarrs_plugin::{MaybeSend, MaybeSync};
use zarrs_storage::byte_range::ByteRange;
use zarrs_storage::{
    Bytes, ReadableWritableStorageTraits, StorageError, StoreWriter, WritableStorageTraits,
};

use super::super::codec::CodecChainBound;
use super::OpStorage;
use super::array_read_ops_planned::{
    ChunkKeys, ChunkPlanner, drive_blocking_one, plan_error, read_ranges, subset_chunk_shape,
};
mod partial_encoder;

use crate::array::{
    Array, ArrayBytes, ArrayError, ArraySubset, ArraySubsetTraits, Indexer, subset_indices_at,
};

/// Perform the write `op` to the chunks of `keys` in `storage`.
fn write_op(
    storage: &(impl WritableStorageTraits + ?Sized),
    keys: &ChunkKeys,
    op: WriteOp<'_>,
) -> Result<(), StorageError> {
    match op {
        WriteOp::Set(key, bytes) => storage.set(&*keys.store_key(key)?, bytes),
        WriteOp::SetPartial(key, offset_values) => {
            storage.set_partial_many(&*keys.store_key(key)?, Box::new(offset_values.into_iter()))
        }
        WriteOp::Erase(key) => storage.erase(&*keys.store_key(key)?),
        WriteOp::Multipart(..) => unreachable!("multipart writes are performed by the backend"),
    }
}

/// Perform the step `op` of the multipart write `id` to the chunks of `keys` in `storage`, with the writers of
/// `multipart`.
///
/// Multipart writes are only supported to the storage of the array (without storage transformers), since a writer
/// borrows the storage.
fn write_multipart<'s, TStorage: ?Sized + WritableStorageTraits, TTransformed: ?Sized>(
    storage: &OpStorage<'s, TStorage, TTransformed>,
    keys: &ChunkKeys,
    multipart: &MultipartWriters<dyn StoreWriter + 's>,
    id: MultipartId,
    op: MultipartOp<'_>,
) -> Result<(), StorageError> {
    multipart.write(id, op, |key, size_hint| match storage {
        OpStorage::Array(storage) => storage.set_multipart(&*keys.store_key(key)?, size_hint),
        OpStorage::Transformed(_) => Err(multipart_transformed_error()),
    })
}

/// Resolve the multipart write setting `multipart_write` for storage that writes values in parts if `supports_multipart`,
/// which the chunk I/O backends can only begin multipart writes with without storage transformers.
fn resolve_multipart_write(
    multipart_write: MultipartWrite,
    without_transformers: bool,
    supports_multipart: bool,
) -> MultipartWrite {
    match multipart_write {
        _ if !without_transformers => MultipartWrite::Never,
        MultipartWrite::Auto if supports_multipart => MultipartWrite::Always,
        MultipartWrite::Auto | MultipartWrite::Never => MultipartWrite::Never,
        MultipartWrite::Always => MultipartWrite::Always,
    }
}

/// The error of a multipart write through storage transformers.
fn multipart_transformed_error() -> StorageError {
    StorageError::Unsupported(
        "multipart writes through storage transformers are not supported".to_string(),
    )
}

/// The synchronous I/O backend for writing the chunks of an array.
struct ChunkWriteIo<'s, TStorage: ?Sized> {
    storage: OpStorage<'s, TStorage, dyn WritableStorageTraits>,
    keys: ChunkKeys,
    multipart: MultipartWriters<dyn StoreWriter + 's>,
}

impl<TStorage: ?Sized + WritableStorageTraits> SyncIo for ChunkWriteIo<'_, TStorage> {
    fn read(
        &self,
        _key: SourceKey,
        _ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        Err(StorageError::Other(
            "the chunks of a write are not readable".to_string(),
        ))
    }

    fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
        if let WriteOp::Multipart(id, op) = op {
            return write_multipart(&self.storage, &self.keys, &self.multipart, id, op);
        }
        with_op_storage!(self.storage, |storage| write_op(storage, &self.keys, op))
    }
}

/// The synchronous I/O backend for reading and writing the chunks of an array.
struct ChunkReadWriteIo<'s, TStorage: ?Sized> {
    storage: OpStorage<'s, TStorage, dyn ReadableWritableStorageTraits>,
    keys: ChunkKeys,
    multipart: MultipartWriters<dyn StoreWriter + 's>,
}

/// The synchronous I/O backend for reading and writing chunks that owns its storage.
type OwnedChunkReadWriteIo = ChunkReadWriteIo<'static, dyn ReadableWritableStorageTraits>;

impl<TStorage: ?Sized + ReadableWritableStorageTraits> ChunkReadWriteIo<'_, TStorage> {
    /// Returns true if the chunk with source `key` exists.
    fn exists(&self, key: SourceKey) -> Result<bool, StorageError> {
        let key = self.keys.store_key(key)?;
        with_op_storage!(self.storage, |storage| Ok(storage
            .size_key(&key)?
            .is_some()))
    }
}

impl<TStorage: ?Sized + ReadableWritableStorageTraits> SyncIo for ChunkReadWriteIo<'_, TStorage> {
    fn read(
        &self,
        key: SourceKey,
        ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        let key = self.keys.store_key(key)?;
        with_op_storage!(self.storage, |storage| read_ranges(storage, &key, ranges))
    }

    fn read_coalescing(&self) -> Option<zarrs_storage::ReadCoalescing> {
        with_op_storage!(self.storage, |storage| storage.read_coalescing())
    }

    fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
        if let WriteOp::Multipart(id, op) = op {
            return write_multipart(&self.storage, &self.keys, &self.multipart, id, op);
        }
        with_op_storage!(self.storage, |storage| write_op(storage, &self.keys, op))
    }
}

impl ChunkPlanner {
    /// Plan storing `bytes`, the chunk of `shape` with source `key`.
    fn plan_store<'a>(
        &self,
        key: SourceKey,
        shape: &[NonZeroU64],
        bytes: ArrayBytes<'a>,
    ) -> Task<'a> {
        Self::plan_store_with(self.codecs.clone(), self.options, key, shape, bytes)
    }

    /// Plan storing `bytes`, the chunk of `shape` with source `key`, with `codecs`.
    fn plan_store_with<'a>(
        codecs: Arc<CodecChainBound>,
        options: CodecOptions,
        key: SourceKey,
        shape: &[NonZeroU64],
        bytes: ArrayBytes<'a>,
    ) -> Task<'a> {
        let hint = encode_hint(&bytes);
        let codecs: Arc<dyn ArrayToBytesCodecTraits> = codecs;
        let shape: SmallVec<[NonZeroU64; 4]> = shape.into();
        Task::compute(move |ctx| encode_and_write(codecs, key, &shape, bytes, &options, ctx))
            .with_hint(hint)
    }
}

/// Plan storing the elements of `array_subset` in the chunks of `chunks`, which it intersects.
///
/// `bytes` holds the elements of `array_subset`. The elements of each chunk are extracted (copied) from them in its
/// task, so the tasks borrow `bytes` only as long as `B` does.
/// Chunks covered by `array_subset` are encoded and written, and other chunks are partially encoded (`set_partial` is
/// true if the storage supports partial writes).
fn plan_store_subset<'a, 'b, B>(
    planner: ChunkPlanner,
    chunks: ArraySubset,
    array_subset: ArraySubset,
    bytes: B,
    set_partial: bool,
) -> Task<'a>
where
    B: std::ops::Deref<Target = ArrayBytes<'b>> + MaybeSend + MaybeSync + 'a,
{
    let tasks = Generator::indexed(chunks.num_elements_usize(), move |i| {
        let chunk_indices = subset_indices_at(&chunks, i as u64);
        let chunk_subset = planner.chunk_subset(&chunk_indices).map_err(plan_error)?;
        let overlap = array_subset.overlap(&chunk_subset)?;
        // The extracted elements are owned, so this does not copy
        let chunk_bytes = bytes
            .extract_array_subset(
                &overlap.relative_to(array_subset.start())?,
                array_subset.shape(),
                &planner.data_type,
            )?
            .into_owned();
        let shape = subset_chunk_shape(&chunk_subset).map_err(plan_error)?;
        Ok(Some(if overlap.shape() == chunk_subset.shape() {
            planner.plan_store(SourceKey(i), shape, chunk_bytes)
        } else {
            planner.codecs.clone().plan_partial_encode(
                SourceKey(i),
                shape,
                PlanIndexer::Subset(overlap.relative_to(chunk_subset.start())?),
                chunk_bytes,
                None,
                set_partial,
                &planner.options,
            )?
        }))
    });
    Task::compute(move |_| {
        Ok(Outcome::Generate {
            children: tasks,
            join: None,
        })
    })
}

/// The number of chunks erased by a task of [`plan_erase_chunks`].
const ERASE_BLOCK: usize = 32;

/// Plan erasing the chunks with source keys `0..num_chunks`.
///
/// A task erases a block of chunks, since an erase has no compute to interleave with.
fn plan_erase_chunks(num_chunks: usize) -> Task<'static> {
    let tasks = Generator::indexed(num_chunks.div_ceil(ERASE_BLOCK), move |block| {
        let start = block * ERASE_BLOCK;
        let end = (start + ERASE_BLOCK).min(num_chunks);
        Ok(Some(Task::compute(move |_ctx| {
            Ok(Outcome::Write {
                ops: (start..end).map(|i| WriteOp::Erase(SourceKey(i))).collect(),
                then: TaskValue::None,
            })
        })))
    });
    Task::compute(move |_| {
        Ok(Outcome::Generate {
            children: tasks,
            join: None,
        })
    })
}

impl<TStorage: ?Sized + WritableStorageTraits + 'static> Array<TStorage> {
    /// Erase the chunks selected by `chunks` (validated) with a planned write.
    pub(crate) fn erase_chunks_planned(
        &self,
        chunks: &dyn Indexer,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let num_chunks = usize::try_from(chunks.len()).unwrap();
        if num_chunks > 0 {
            let io = self.chunk_write_io(self.indexer_keys(chunks))?;
            drive_blocking_one(plan_erase_chunks(num_chunks), &io, resources)?;
        }
        Ok(())
    }

    /// `options` with the multipart write setting resolved for the storage of the array (see [`MultipartWrite`]).
    fn write_options(&self, options: &CodecOptions) -> CodecOptions {
        let without_transformers = self.storage_transformers().is_empty();
        options
            .with_multipart_write(resolve_multipart_write(
                options.multipart_write(),
                without_transformers,
                self.storage.supports_multipart(),
            ))
            .with_multipart_patch(without_transformers && self.storage.supports_multipart_patch())
    }

    /// The I/O backend for writing the chunks of `keys`.
    fn chunk_write_io(&self, keys: ChunkKeys) -> Result<ChunkWriteIo<'_, TStorage>, ArrayError> {
        let storage = if self.storage_transformers().is_empty() {
            OpStorage::Array(&*self.storage)
        } else {
            OpStorage::Transformed(self.writable_storage()?)
        };
        Ok(ChunkWriteIo {
            storage,
            keys,
            multipart: MultipartWriters::new(),
        })
    }

    /// Store `chunk_bytes` (validated) at `chunk_indices` with a planned write.
    pub(crate) fn store_chunk_planned(
        &self,
        chunk_indices: &[u64],
        chunk_bytes: ArrayBytes<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let shape = self.chunk_shape(chunk_indices)?;
        let task = ChunkPlanner::plan_store_with(
            self.codecs_bound(),
            self.write_options(options),
            SourceKey(0),
            &shape,
            chunk_bytes,
        );
        let io = self.chunk_write_io(self.one_key(chunk_indices)?)?;
        drive_blocking_one(task, &io, resources)?;
        Ok(())
    }

    /// Store `chunks_bytes` (validated), the elements of `array_subset`, at the chunks of `chunks` (which `array_subset`
    /// covers exactly) with a planned write.
    pub(crate) fn store_chunks_planned(
        &self,
        chunks: &dyn ArraySubsetTraits,
        array_subset: &ArraySubset,
        chunks_bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunks = chunks.to_array_subset();
        let keys = self.grid_keys(chunks.clone());
        let task = plan_store_subset(
            self.chunk_planner(&self.write_options(options)),
            chunks,
            array_subset.clone(),
            chunks_bytes,
            false,
        );
        drive_blocking_one(task, &self.chunk_write_io(keys)?, resources)?;
        Ok(())
    }
}

impl<TStorage: ?Sized + ReadableWritableStorageTraits + 'static> Array<TStorage> {
    /// The I/O backend for reading and writing the chunks of `keys`.
    fn chunk_read_write_io(
        &self,
        keys: ChunkKeys,
    ) -> Result<ChunkReadWriteIo<'_, TStorage>, ArrayError> {
        let storage = if self.storage_transformers().is_empty() {
            OpStorage::Array(&*self.storage)
        } else {
            OpStorage::Transformed(self.readable_writable_storage()?)
        };
        Ok(ChunkReadWriteIo {
            storage,
            keys,
            multipart: MultipartWriters::new(),
        })
    }

    /// The I/O backend for reading and writing the chunks of `keys`, which owns its storage.
    fn owned_chunk_read_write_io(
        &self,
        keys: ChunkKeys,
    ) -> Result<OwnedChunkReadWriteIo, ArrayError> {
        Ok(ChunkReadWriteIo {
            storage: OpStorage::Transformed(self.readable_writable_storage()?),
            keys,
            multipart: MultipartWriters::new(),
        })
    }

    /// Store `encoded` at the subchunk at `subchunk_indices` of the subchunk grid at `level`, or erase the subchunk if
    /// [`None`], with a planned write.
    pub(crate) fn store_encoded_subchunk_planned(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        encoded: Option<Bytes>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let (chunk_indices, coordinates) =
            self.encoded_subchunk_address(level, subchunk_indices)?;
        let task = self.codecs_bound().clone().plan_store_encoded_subchunk(
            SourceKey(0),
            &self.chunk_shape(&chunk_indices)?,
            level,
            &coordinates,
            encoded,
            self.storage.supports_set_partial(),
            self.codec_options(),
        )?;
        let io = self.chunk_read_write_io(self.one_key(&chunk_indices)?)?;
        drive_blocking_one(task, &io, resources)?;
        Ok(())
    }

    /// Store `subset_bytes` (validated), the elements of `array_subset`, which intersects the chunks of `chunks`, with a
    /// planned write.
    ///
    /// Chunks covered by `array_subset` are encoded and written, and other chunks are partially encoded.
    pub(crate) fn store_array_subset_planned(
        &self,
        chunks: &ArraySubset,
        array_subset: &ArraySubset,
        subset_bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let keys = self.grid_keys(chunks.clone());
        let task = plan_store_subset(
            self.chunk_planner(&self.write_options(options)),
            chunks.clone(),
            array_subset.clone(),
            subset_bytes,
            self.storage.supports_set_partial(),
        );
        drive_blocking_one(task, &self.chunk_read_write_io(keys)?, resources)?;
        Ok(())
    }

    /// Partially encode `indexer_bytes` (validated), the elements of `indexer`, in the chunk at `chunk_indices` with a
    /// planned write.
    pub(crate) fn store_partial_chunk_planned(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        indexer_bytes: ArrayBytes<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let task = self.codecs_bound().plan_partial_encode(
            SourceKey(0),
            &self.chunk_shape(chunk_indices)?,
            PlanIndexer::Borrowed(indexer),
            indexer_bytes,
            None,
            self.storage.supports_set_partial(),
            options,
        )?;
        let io = self.chunk_read_write_io(self.one_key(chunk_indices)?)?;
        drive_blocking_one(task, &io, resources)?;
        Ok(())
    }
}

/// Returns true if `bytes` are not borrowed, so [`ArrayBytes::into_owned`] does not copy them.
#[cfg(feature = "async")]
fn is_shared(bytes: &ArrayBytes<'_>) -> bool {
    use zarrs_codec::CowBytes;
    match bytes {
        ArrayBytes::Fixed(bytes) => matches!(bytes, CowBytes::Shared(_)),
        ArrayBytes::Variable(bytes) => matches!(bytes.bytes(), CowBytes::Shared(_)),
        ArrayBytes::Optional(bytes) => {
            is_shared(bytes.data()) && matches!(bytes.mask(), CowBytes::Shared(_))
        }
    }
}

#[cfg(feature = "async")]
pub(super) mod async_planned {
    use zarrs_codec::plan::{AsyncIo, drive_async_offload_one, drive_async_one};
    use zarrs_storage::{
        AsyncReadableWritableStorageTraits, AsyncStoreWriter, AsyncWritableStorageTraits,
    };

    use super::super::array_read_ops_planned::async_read_ranges;
    use super::{
        Arc, Array, ArrayBytes, ArrayError, ArraySubset, ArrayToBytesCodecTraits, ByteRange, Bytes,
        ChunkKeys, ChunkPlanner, CodecOptions, Indexer, MultipartId, MultipartOp, MultipartWriters,
        OpStorage, PlanIndexer, ReadBytes, Resources, SourceKey, StorageError, WriteOp, is_shared,
        multipart_transformed_error, plan_erase_chunks, plan_store_subset, resolve_multipart_write,
    };

    /// Perform the write `op` to the chunks of `keys` in `storage`.
    async fn write_op(
        storage: &(impl AsyncWritableStorageTraits + ?Sized),
        keys: &ChunkKeys,
        op: WriteOp<'_>,
    ) -> Result<(), StorageError> {
        match op {
            WriteOp::Set(key, bytes) => storage.set(&*keys.store_key(key)?, bytes).await,
            WriteOp::SetPartial(key, offset_values) => {
                storage
                    .set_partial_many(&*keys.store_key(key)?, Box::new(offset_values.into_iter()))
                    .await
            }
            WriteOp::Erase(key) => storage.erase(&*keys.store_key(key)?).await,
            WriteOp::Multipart(..) => {
                unreachable!("multipart writes are performed by the backend")
            }
        }
    }

    /// Perform the step `op` of the multipart write `id` to the chunks of `keys` in `storage`, with the writers of
    /// `multipart`.
    async fn write_multipart<
        's,
        TStorage: ?Sized + AsyncWritableStorageTraits,
        TTransformed: ?Sized,
    >(
        storage: &OpStorage<'s, TStorage, TTransformed>,
        keys: &ChunkKeys,
        multipart: &MultipartWriters<dyn AsyncStoreWriter + 's>,
        id: MultipartId,
        op: MultipartOp<'_>,
    ) -> Result<(), StorageError> {
        multipart
            .write_async(id, op, |key, size_hint| async move {
                match storage {
                    OpStorage::Array(storage) => {
                        storage
                            .set_multipart(&*keys.store_key(key)?, size_hint)
                            .await
                    }
                    OpStorage::Transformed(_) => Err(multipart_transformed_error()),
                }
            })
            .await
    }

    /// Return `result`, the result of a drive with the multipart writes `multipart`, aborting the writes in progress if it
    /// failed (an object store may otherwise keep the parts of an unfinished upload).
    async fn abort_on_error<T>(
        multipart: &MultipartWriters<dyn AsyncStoreWriter + '_>,
        result: Result<T, zarrs_codec::CodecError>,
    ) -> Result<T, zarrs_codec::CodecError> {
        if result.is_err() {
            // The error of the drive is reported rather than that of aborting
            let _ = multipart.abort_all_async().await;
        }
        result
    }

    /// The asynchronous I/O backend for writing the chunks of an array.
    pub(super) struct AsyncChunkWriteIo<'s, TStorage: ?Sized> {
        storage: OpStorage<'s, TStorage, dyn AsyncWritableStorageTraits>,
        keys: ChunkKeys,
        multipart: MultipartWriters<dyn AsyncStoreWriter + 's>,
    }

    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    impl<TStorage: ?Sized + AsyncWritableStorageTraits> AsyncIo for AsyncChunkWriteIo<'_, TStorage> {
        async fn read(
            &self,
            _key: SourceKey,
            _ranges: &[ByteRange],
        ) -> Result<Option<ReadBytes>, StorageError> {
            Err(StorageError::Other(
                "the chunks of a write are not readable".to_string(),
            ))
        }

        async fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
            if let WriteOp::Multipart(id, op) = op {
                return write_multipart(&self.storage, &self.keys, &self.multipart, id, op).await;
            }
            with_op_storage!(self.storage, |storage| write_op(storage, &self.keys, op)
                .await)
        }
    }

    /// The asynchronous I/O backend for reading and writing the chunks of an array.
    pub(super) struct AsyncChunkReadWriteIo<'s, TStorage: ?Sized> {
        storage: OpStorage<'s, TStorage, dyn AsyncReadableWritableStorageTraits>,
        keys: ChunkKeys,
        multipart: MultipartWriters<dyn AsyncStoreWriter + 's>,
    }

    /// The asynchronous I/O backend for reading and writing chunks that owns its storage.
    pub(super) type OwnedAsyncChunkReadWriteIo =
        AsyncChunkReadWriteIo<'static, dyn AsyncReadableWritableStorageTraits>;

    impl<TStorage: ?Sized + AsyncReadableWritableStorageTraits> AsyncChunkReadWriteIo<'_, TStorage> {
        /// Returns true if the chunk with source `key` exists.
        pub(super) async fn exists(&self, key: SourceKey) -> Result<bool, StorageError> {
            let key = self.keys.store_key(key)?;
            with_op_storage!(self.storage, |storage| Ok(storage
                .size_key(&key)
                .await?
                .is_some()))
        }
    }

    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    impl<TStorage: ?Sized + AsyncReadableWritableStorageTraits> AsyncIo
        for AsyncChunkReadWriteIo<'_, TStorage>
    {
        async fn read(
            &self,
            key: SourceKey,
            ranges: &[ByteRange],
        ) -> Result<Option<ReadBytes>, StorageError> {
            let key = self.keys.store_key(key)?;
            with_op_storage!(self.storage, |storage| {
                async_read_ranges(storage, &key, ranges).await
            })
        }

        fn read_coalescing(&self) -> Option<zarrs_storage::ReadCoalescing> {
            with_op_storage!(self.storage, |storage| storage.read_coalescing())
        }

        async fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
            if let WriteOp::Multipart(id, op) = op {
                return write_multipart(&self.storage, &self.keys, &self.multipart, id, op).await;
            }
            with_op_storage!(self.storage, |storage| write_op(storage, &self.keys, op)
                .await)
        }
    }

    impl<TStorage: ?Sized + AsyncWritableStorageTraits + 'static> Array<TStorage> {
        /// `options` with the multipart write setting resolved for the storage of the array (see
        /// [`MultipartWrite`](zarrs_codec::MultipartWrite)).
        fn async_write_options(&self, options: &CodecOptions) -> CodecOptions {
            let without_transformers = self.storage_transformers().is_empty();
            options
                .with_multipart_write(resolve_multipart_write(
                    options.multipart_write(),
                    without_transformers,
                    self.storage.supports_multipart(),
                ))
                .with_multipart_patch(
                    without_transformers && self.storage.supports_multipart_patch(),
                )
        }

        /// The I/O backend for writing the chunks of `keys`.
        pub(super) async fn async_chunk_write_io(
            &self,
            keys: ChunkKeys,
        ) -> Result<AsyncChunkWriteIo<'_, TStorage>, ArrayError> {
            let storage = if self.storage_transformers().is_empty() {
                OpStorage::Array(&*self.storage)
            } else {
                OpStorage::Transformed(self.async_writable_storage().await?)
            };
            Ok(AsyncChunkWriteIo {
                storage,
                keys,
                multipart: MultipartWriters::new(),
            })
        }

        /// Erase the chunks selected by `chunks` (validated) with a planned write.
        pub(crate) async fn async_erase_chunks_planned(
            &self,
            chunks: &dyn Indexer,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            let num_chunks = usize::try_from(chunks.len()).unwrap();
            if num_chunks > 0 {
                let io = self.async_chunk_write_io(self.indexer_keys(chunks)).await?;
                drive_async_one(plan_erase_chunks(num_chunks), &io, resources).await?;
            }
            Ok(())
        }

        /// Store `chunk_bytes` (validated) at `chunk_indices` with a planned write.
        ///
        /// Compute is offloaded to Rayon if `chunk_bytes` are not borrowed, and otherwise runs on the polling task.
        pub(crate) async fn async_store_chunk_planned(
            &self,
            chunk_indices: &[u64],
            chunk_bytes: ArrayBytes<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            let shape = self.chunk_shape(chunk_indices)?;
            let io = self
                .async_chunk_write_io(self.one_key(chunk_indices)?)
                .await?;
            let options = self.async_write_options(options);
            if is_shared(&chunk_bytes) {
                let task = ChunkPlanner::plan_store_with(
                    self.codecs_bound(),
                    options,
                    SourceKey(0),
                    &shape,
                    chunk_bytes.into_owned(),
                );
                abort_on_error(
                    &io.multipart,
                    drive_async_offload_one(task, &io, resources).await,
                )
                .await?;
            } else {
                let task = ChunkPlanner::plan_store_with(
                    self.codecs_bound(),
                    options,
                    SourceKey(0),
                    &shape,
                    chunk_bytes,
                );
                abort_on_error(&io.multipart, drive_async_one(task, &io, resources).await).await?;
            }
            Ok(())
        }

        /// Store `chunks_bytes` (validated), the elements of `array_subset`, at the chunks of `chunks` (which
        /// `array_subset` covers exactly) with a planned write.
        ///
        /// Compute is offloaded to Rayon if `chunks_bytes` are not borrowed, and otherwise runs on the polling task.
        pub(crate) async fn async_store_chunks_planned(
            &self,
            chunks: &ArraySubset,
            array_subset: &ArraySubset,
            chunks_bytes: ArrayBytes<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            let planner = self.chunk_planner(&self.async_write_options(options));
            let io = self
                .async_chunk_write_io(self.grid_keys(chunks.clone()))
                .await?;
            if is_shared(&chunks_bytes) {
                let bytes = Arc::new(chunks_bytes.into_owned());
                let task =
                    plan_store_subset(planner, chunks.clone(), array_subset.clone(), bytes, false);
                abort_on_error(
                    &io.multipart,
                    drive_async_offload_one(task, &io, resources).await,
                )
                .await?;
                Ok(())
            } else {
                let task = plan_store_subset(
                    planner,
                    chunks.clone(),
                    array_subset.clone(),
                    &chunks_bytes,
                    false,
                );
                abort_on_error(&io.multipart, drive_async_one(task, &io, resources).await).await?;
                Ok(())
            }
        }
    }

    impl<TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static> Array<TStorage> {
        /// The I/O backend for reading and writing the chunks of `keys`.
        pub(super) async fn async_chunk_read_write_io(
            &self,
            keys: ChunkKeys,
        ) -> Result<AsyncChunkReadWriteIo<'_, TStorage>, ArrayError> {
            let storage = if self.storage_transformers().is_empty() {
                OpStorage::Array(&*self.storage)
            } else {
                OpStorage::Transformed(self.async_readable_writable_storage().await?)
            };
            Ok(AsyncChunkReadWriteIo {
                storage,
                keys,
                multipart: MultipartWriters::new(),
            })
        }

        /// The I/O backend for reading and writing the chunks of `keys`, which owns its storage.
        pub(super) async fn async_owned_chunk_read_write_io(
            &self,
            keys: ChunkKeys,
        ) -> Result<OwnedAsyncChunkReadWriteIo, ArrayError> {
            Ok(AsyncChunkReadWriteIo {
                storage: OpStorage::Transformed(self.async_readable_writable_storage().await?),
                keys,
                multipart: MultipartWriters::new(),
            })
        }

        /// Store `encoded` at the subchunk at `subchunk_indices` of the subchunk grid at `level`, or erase the
        /// subchunk if [`None`], with a planned write.
        pub(crate) async fn async_store_encoded_subchunk_planned(
            &self,
            level: usize,
            subchunk_indices: &[u64],
            encoded: Option<Bytes>,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            let (chunk_indices, coordinates) =
                self.encoded_subchunk_address(level, subchunk_indices)?;
            let task = self.codecs_bound().clone().plan_store_encoded_subchunk(
                SourceKey(0),
                &self.chunk_shape(&chunk_indices)?,
                level,
                &coordinates,
                encoded,
                self.storage.supports_set_partial(),
                self.codec_options(),
            )?;
            let io = self
                .async_chunk_read_write_io(self.one_key(&chunk_indices)?)
                .await?;
            drive_async_offload_one(task, &io, resources).await?;
            Ok(())
        }

        /// Store `subset_bytes` (validated), the elements of `array_subset`, which intersects the chunks of `chunks`,
        /// with a planned write.
        ///
        /// Compute is offloaded to Rayon if `subset_bytes` are not borrowed, and otherwise runs on the polling task.
        pub(crate) async fn async_store_array_subset_planned(
            &self,
            chunks: &ArraySubset,
            array_subset: &ArraySubset,
            subset_bytes: ArrayBytes<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            let planner = self.chunk_planner(&self.async_write_options(options));
            let set_partial = self.storage.supports_set_partial();
            let io = self
                .async_chunk_read_write_io(self.grid_keys(chunks.clone()))
                .await?;
            if is_shared(&subset_bytes) {
                let bytes = Arc::new(subset_bytes.into_owned());
                let task = plan_store_subset(
                    planner,
                    chunks.clone(),
                    array_subset.clone(),
                    bytes,
                    set_partial,
                );
                abort_on_error(
                    &io.multipart,
                    drive_async_offload_one(task, &io, resources).await,
                )
                .await?;
                Ok(())
            } else {
                let task = plan_store_subset(
                    planner,
                    chunks.clone(),
                    array_subset.clone(),
                    &subset_bytes,
                    set_partial,
                );
                abort_on_error(&io.multipart, drive_async_one(task, &io, resources).await).await?;
                Ok(())
            }
        }

        /// Partially encode `indexer_bytes` (validated), the elements of `indexer`, in the chunk at `chunk_indices`
        /// with a planned write.
        pub(crate) async fn async_store_partial_chunk_planned(
            &self,
            chunk_indices: &[u64],
            indexer: &dyn Indexer,
            indexer_bytes: ArrayBytes<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            let shape = self.chunk_shape(chunk_indices)?;
            let set_partial = self.storage.supports_set_partial();
            let io = self
                .async_chunk_read_write_io(self.one_key(chunk_indices)?)
                .await?;
            if let Some(subset) = indexer.as_array_subset()
                && is_shared(&indexer_bytes)
            {
                // The task does not borrow, so its compute is offloaded to Rayon
                let task = self.codecs_bound().plan_partial_encode(
                    SourceKey(0),
                    &shape,
                    PlanIndexer::Subset(subset.to_array_subset()),
                    indexer_bytes.into_owned(),
                    None,
                    set_partial,
                    options,
                )?;
                drive_async_offload_one(task, &io, resources).await?;
                return Ok(());
            }
            let task = self.codecs_bound().plan_partial_encode(
                SourceKey(0),
                &shape,
                PlanIndexer::Borrowed(indexer),
                indexer_bytes,
                None,
                set_partial,
                options,
            )?;
            drive_async_one(task, &io, resources).await?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use smallvec::smallvec;
    use zarrs_storage::ReadableStorageTraits;
    use zarrs_storage::store::MemoryStore;

    use super::*;
    use crate::array::{ArrayBuilder, data_type};

    /// A task that writes the chunk with source key 0 in two parts, as a multipart write.
    fn multipart_task<'a>() -> Task<'a> {
        Task::compute(|_| {
            let id = MultipartId::unique();
            Ok(Outcome::Write {
                ops: smallvec![
                    WriteOp::Multipart(id, MultipartOp::Begin(SourceKey(0), Some(3))),
                    WriteOp::Multipart(
                        id,
                        MultipartOp::Parts(vec![vec![1u8].into(), vec![2u8].into()])
                    ),
                    WriteOp::Multipart(id, MultipartOp::Parts(vec![vec![3u8].into()])),
                    WriteOp::Multipart(id, MultipartOp::Finish),
                ],
                then: TaskValue::None,
            })
        })
    }

    #[test]
    fn chunk_io_multipart_writes() -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build(store.clone(), "/array")?;
        let resources = Resources::default();
        let key = array.chunk_key(&[1])?;

        let io = array.chunk_write_io(array.one_key(&[1])?)?;
        drive_blocking_one(multipart_task(), &io, &resources)?;
        assert!(io.multipart.is_empty());
        assert_eq!(store.get(&key)?.unwrap(), vec![1, 2, 3]);

        store.erase(&key)?;
        let io = array.chunk_read_write_io(array.one_key(&[1])?)?;
        drive_blocking_one(multipart_task(), &io, &resources)?;
        assert_eq!(store.get(&key)?.unwrap(), vec![1, 2, 3]);

        // A writer borrows the storage, so a backend that owns its storage does not support multipart writes
        let io = array.owned_chunk_read_write_io(array.one_key(&[1])?)?;
        assert!(drive_blocking_one(multipart_task(), &io, &resources).is_err());
        Ok(())
    }

    /// An asynchronous store whose multipart writes fail, counting the writers aborted.
    #[cfg(feature = "async")]
    #[derive(Default)]
    struct FailingMultipartStore {
        inner: zarrs_storage::store::AsyncMemoryStore,
        aborted: std::sync::atomic::AtomicUsize,
    }

    #[cfg(feature = "async")]
    struct FailingWriter<'s>(&'s std::sync::atomic::AtomicUsize);

    #[cfg(feature = "async")]
    #[async_trait::async_trait]
    impl zarrs_storage::AsyncStoreWriter for FailingWriter<'_> {
        async fn write(&mut self, _part: zarrs_codec::CowBytes<'_>) -> Result<(), StorageError> {
            Err(StorageError::Other("write failed".to_string()))
        }

        async fn finish(self: Box<Self>) -> Result<(), StorageError> {
            Ok(())
        }

        async fn abort(self: Box<Self>) -> Result<(), StorageError> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    #[cfg(feature = "async")]
    #[async_trait::async_trait]
    impl zarrs_storage::AsyncWritableStorageTraits for FailingMultipartStore {
        async fn set(
            &self,
            key: &zarrs_storage::StoreKey,
            value: zarrs_codec::CowBytes<'_>,
        ) -> Result<(), StorageError> {
            self.inner.set(key, value).await
        }

        async fn set_partial_many<'a>(
            &'a self,
            key: &zarrs_storage::StoreKey,
            offset_values: zarrs_storage::OffsetBytesIterator<'a>,
        ) -> Result<(), StorageError> {
            self.inner.set_partial_many(key, offset_values).await
        }

        async fn erase(&self, key: &zarrs_storage::StoreKey) -> Result<(), StorageError> {
            self.inner.erase(key).await
        }

        async fn erase_prefix(
            &self,
            prefix: &zarrs_storage::StorePrefix,
        ) -> Result<(), StorageError> {
            self.inner.erase_prefix(prefix).await
        }

        fn supports_set_partial(&self) -> bool {
            false
        }

        async fn set_multipart<'s>(
            &'s self,
            _key: &zarrs_storage::StoreKey,
            _size_hint: Option<u64>,
        ) -> Result<Box<dyn zarrs_storage::AsyncStoreWriter + 's>, StorageError> {
            Ok(Box::new(FailingWriter(&self.aborted)))
        }

        fn supports_multipart(&self) -> bool {
            true
        }
    }

    /// The multipart writes of a failed drive are aborted.
    #[cfg(feature = "async")]
    #[test]
    fn async_failed_multipart_writes_are_aborted() -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(FailingMultipartStore::default());
        let mut builder = ArrayBuilder::new(vec![64, 64], vec![64, 64], data_type::uint16(), 0u16);
        builder.subchunk_shape(vec![8, 8]);
        let array = builder
            .build_arc(store.clone(), "/array")?
            .with_codec_options(CodecOptions::default().with_multipart_part_size(1024));
        let data: Vec<u16> = (0..64 * 64).collect();
        let result = futures::executor::block_on(array.async_store_chunk(
            &[0, 0],
            data,
            &Resources::default(),
        ));
        assert!(result.is_err());
        assert_eq!(store.aborted.load(std::sync::atomic::Ordering::SeqCst), 1);
        Ok(())
    }

    #[cfg(feature = "async")]
    #[test]
    fn async_chunk_io_multipart_writes() -> Result<(), Box<dyn std::error::Error>> {
        use zarrs_codec::plan::drive_async_one;
        use zarrs_storage::AsyncReadableStorageTraits;
        use zarrs_storage::store::AsyncMemoryStore;

        futures::executor::block_on(async {
            let store = Arc::new(AsyncMemoryStore::default());
            let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
                .build_arc(store.clone(), "/array")?;
            let resources = Resources::default();
            let key = array.chunk_key(&[1])?;

            let io = array.async_chunk_write_io(array.one_key(&[1])?).await?;
            drive_async_one(multipart_task(), &io, &resources).await?;
            assert_eq!(store.get(&key).await?.unwrap(), vec![1, 2, 3]);

            let io = array
                .async_chunk_read_write_io(array.one_key(&[0])?)
                .await?;
            drive_async_one(multipart_task(), &io, &resources).await?;
            assert_eq!(
                store.get(&array.chunk_key(&[0])?).await?.unwrap(),
                vec![1, 2, 3]
            );

            let io = array
                .async_owned_chunk_read_write_io(array.one_key(&[1])?)
                .await?;
            assert!(
                drive_async_one(multipart_task(), &io, &resources)
                    .await
                    .is_err()
            );
            Ok(())
        })
    }
}
