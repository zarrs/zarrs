//! Planned array write operations.
//!
//! Each chunk written is a task that encodes the chunk (after reading, decoding and updating it, for a partial
//! write) and then writes or erases it. The tasks of an operation are executed by one drive, with separate I/O and
//! compute concurrency limits.

use std::num::NonZeroU64;
use std::sync::Arc;

use smallvec::SmallVec;
use zarrs_codec::plan::{
    Generator, Outcome, PlanIndexer, ReadBytes, SourceKey, SyncIo, Task, TaskValue, WriteOp,
    encode_and_write, encode_hint,
};
use zarrs_codec::{ArrayToBytesCodecTraits, CodecOptions, Resources};
use zarrs_plugin::{MaybeSend, MaybeSync};
use zarrs_storage::byte_range::ByteRange;
use zarrs_storage::{
    ReadableWritableStorage, ReadableWritableStorageTraits, StorageError, StorageHandle,
    WritableStorage, WritableStorageTraits,
};

use super::array_read_ops_planned::{
    ChunkKeys, ChunkPlanner, drive_blocking_one, plan_error, read_ranges, subset_chunk_shape,
};
mod partial_encoder;

use crate::array::{
    Array, ArrayBytes, ArrayError, ArraySubset, ArraySubsetTraits, Indexer, subset_indices_at,
};

/// Perform the write `op` to the chunks of `keys` in `storage`.
fn write_op(
    storage: &dyn WritableStorageTraits,
    keys: &ChunkKeys,
    op: WriteOp<'_>,
) -> Result<(), StorageError> {
    match op {
        WriteOp::Set(key, bytes) => storage.set(&*keys.store_key(key)?, bytes),
        WriteOp::SetPartial(key, offset_values) => {
            storage.set_partial_many(&*keys.store_key(key)?, Box::new(offset_values.into_iter()))
        }
        WriteOp::Erase(key) => storage.erase(&*keys.store_key(key)?),
    }
}

/// The synchronous I/O backend for writing the chunks of an array.
struct ChunkWriteIo {
    storage: WritableStorage,
    keys: ChunkKeys,
}

impl SyncIo for ChunkWriteIo {
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
        write_op(&*self.storage, &self.keys, op)
    }
}

/// The synchronous I/O backend for reading and writing the chunks of an array.
struct ChunkReadWriteIo {
    storage: ReadableWritableStorage,
    keys: ChunkKeys,
}

impl SyncIo for ChunkReadWriteIo {
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

    fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
        write_op(&*self.storage, &self.keys, op)
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
        let hint = encode_hint(&bytes);
        let codecs: Arc<dyn ArrayToBytesCodecTraits> = self.codecs.clone();
        let shape: SmallVec<[NonZeroU64; 4]> = shape.into();
        let options = self.options;
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

    /// The I/O backend for writing the chunks of `keys`.
    fn chunk_write_io(&self, keys: ChunkKeys) -> Result<ChunkWriteIo, ArrayError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        Ok(ChunkWriteIo {
            storage: self
                .storage_transformers()
                .create_writable_transformer(storage_handle)?,
            keys,
        })
    }

    /// Store `chunk_bytes` (validated) at `chunk_indices` with a planned write.
    ///
    /// A chunk of codecs that plan no encode tasks of their own is encoded and written directly, under the I/O limit
    /// of `resources`.
    pub(crate) fn store_chunk_planned(
        &self,
        chunk_indices: &[u64],
        chunk_bytes: ArrayBytes<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let shape = self.chunk_shape(chunk_indices)?;
        let codecs = self.codecs_bound();
        if codecs.plans_encode_subtasks() {
            let task = self
                .chunk_planner(options)
                .plan_store(SourceKey(0), &shape, chunk_bytes);
            let io = self.chunk_write_io(self.one_key(chunk_indices)?)?;
            drive_blocking_one(task, &io, resources)?;
            return Ok(());
        }
        let encoded =
            if !options.store_empty_chunks() && chunk_bytes.is_fill_value(self.fill_value()) {
                None
            } else {
                Some(codecs.encode(chunk_bytes, &shape, options, resources)?)
            };
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage = self
            .storage_transformers()
            .create_writable_transformer(storage_handle)?;
        let key = self.chunk_key(chunk_indices)?;
        let _permit = resources.io_permit_blocking();
        match encoded {
            Some(encoded) => storage.set(&key, encoded)?,
            None => storage.erase(&key)?,
        }
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
            self.chunk_planner(options),
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
    fn chunk_read_write_io(&self, keys: ChunkKeys) -> Result<ChunkReadWriteIo, ArrayError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        Ok(ChunkReadWriteIo {
            storage: self
                .storage_transformers()
                .create_readable_writable_transformer(storage_handle)?,
            keys,
        })
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
            self.chunk_planner(options),
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
        AsyncReadableWritableStorage, AsyncReadableWritableStorageTraits, AsyncWritableStorage,
        AsyncWritableStorageTraits,
    };

    use super::super::array_read_ops_planned::async_read_ranges;
    use super::{
        Arc, Array, ArrayBytes, ArrayError, ArraySubset, ArrayToBytesCodecTraits, ByteRange,
        ChunkKeys, CodecOptions, Indexer, PlanIndexer, ReadBytes, Resources, SourceKey,
        StorageError, StorageHandle, WriteOp, is_shared, plan_erase_chunks, plan_store_subset,
    };

    /// Perform the write `op` to the chunks of `keys` in `storage`.
    async fn write_op(
        storage: &dyn AsyncWritableStorageTraits,
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
        }
    }

    /// The asynchronous I/O backend for writing the chunks of an array.
    struct AsyncChunkWriteIo {
        storage: AsyncWritableStorage,
        keys: ChunkKeys,
    }

    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    impl AsyncIo for AsyncChunkWriteIo {
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
            write_op(&*self.storage, &self.keys, op).await
        }
    }

    /// The asynchronous I/O backend for reading and writing the chunks of an array.
    pub(super) struct AsyncChunkReadWriteIo {
        pub(super) storage: AsyncReadableWritableStorage,
        pub(super) keys: ChunkKeys,
    }

    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    impl AsyncIo for AsyncChunkReadWriteIo {
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

        async fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
            write_op(&*self.storage, &self.keys, op).await
        }
    }

    impl<TStorage: ?Sized + AsyncWritableStorageTraits + 'static> Array<TStorage> {
        /// The I/O backend for writing the chunks of `keys`.
        async fn async_chunk_write_io(
            &self,
            keys: ChunkKeys,
        ) -> Result<AsyncChunkWriteIo, ArrayError> {
            let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
            Ok(AsyncChunkWriteIo {
                storage: self
                    .storage_transformers()
                    .create_async_writable_transformer(storage_handle)
                    .await?,
                keys,
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
        /// A chunk of codecs that plan encode tasks of their own (e.g. a shard) is encoded by those tasks offloaded to
        /// Rayon if `chunk_bytes` are not borrowed. Otherwise, the chunk is encoded on the polling task and written
        /// directly, under the I/O limit of `resources`.
        pub(crate) async fn async_store_chunk_planned(
            &self,
            chunk_indices: &[u64],
            chunk_bytes: ArrayBytes<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), ArrayError> {
            let shape = self.chunk_shape(chunk_indices)?;
            let codecs = self.codecs_bound();
            if codecs.plans_encode_subtasks() && is_shared(&chunk_bytes) {
                // The encode tasks are offloaded to Rayon
                let task = self.chunk_planner(options).plan_store(
                    SourceKey(0),
                    &shape,
                    chunk_bytes.into_owned(),
                );
                let io = self
                    .async_chunk_write_io(self.one_key(chunk_indices)?)
                    .await?;
                drive_async_offload_one(task, &io, resources).await?;
                return Ok(());
            }
            let encoded =
                if !options.store_empty_chunks() && chunk_bytes.is_fill_value(self.fill_value()) {
                    None
                } else {
                    Some(codecs.encode(chunk_bytes, &shape, options, resources)?)
                };
            let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
            let storage = self
                .storage_transformers()
                .create_async_writable_transformer(storage_handle)
                .await?;
            let key = self.chunk_key(chunk_indices)?;
            let _permit = resources.io_permit().await;
            match encoded {
                Some(encoded) => storage.set(&key, encoded).await?,
                None => storage.erase(&key).await?,
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
            let planner = self.chunk_planner(options);
            let io = self
                .async_chunk_write_io(self.grid_keys(chunks.clone()))
                .await?;
            if is_shared(&chunks_bytes) {
                let bytes = Arc::new(chunks_bytes.into_owned());
                let task =
                    plan_store_subset(planner, chunks.clone(), array_subset.clone(), bytes, false);
                drive_async_offload_one(task, &io, resources).await?;
                Ok(())
            } else {
                let task = plan_store_subset(
                    planner,
                    chunks.clone(),
                    array_subset.clone(),
                    &chunks_bytes,
                    false,
                );
                drive_async_one(task, &io, resources).await?;
                Ok(())
            }
        }
    }

    impl<TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static> Array<TStorage> {
        /// The I/O backend for reading and writing the chunks of `keys`.
        pub(super) async fn async_chunk_read_write_io(
            &self,
            keys: ChunkKeys,
        ) -> Result<AsyncChunkReadWriteIo, ArrayError> {
            let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
            Ok(AsyncChunkReadWriteIo {
                storage: self
                    .storage_transformers()
                    .create_async_readable_writable_transformer(storage_handle)
                    .await?,
                keys,
            })
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
            let planner = self.chunk_planner(options);
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
                drive_async_offload_one(task, &io, resources).await?;
                Ok(())
            } else {
                let task = plan_store_subset(
                    planner,
                    chunks.clone(),
                    array_subset.clone(),
                    &subset_bytes,
                    set_partial,
                );
                drive_async_one(task, &io, resources).await?;
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
