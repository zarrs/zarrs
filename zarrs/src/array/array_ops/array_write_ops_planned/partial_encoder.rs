//! Planned partial encoders of a chunk.
//!
//! If the codecs plan partial encodes (and the storage supports partial writes), a partial encoder keeps the state of
//! its chunk (e.g. the shard index) between partial encodes, so that each partial encode only reads what it updates.
//! Otherwise, each partial encode reads, decodes, updates, encodes and writes the chunk.

use std::sync::{Arc, Mutex};

use zarrs_codec::plan::{
    ByteSource, DecodeRequest, PartialEncodeState, PlanIndexer, SourceKey, SyncIo, TaskTarget,
    WriteOp, local_subchunk_grids, partial_encode_state,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderSubchunkingTraits,
    ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, CodecError,
    CodecOptions, Resources,
};
use zarrs_data_type::DataType;
use zarrs_storage::{ReadableWritableStorageTraits, StorageError};

use super::super::array_read_ops_planned::{
    array_to_codec_error, array_value, check_target_len, drive_sync, static_local_subchunk_grids,
    whole_views,
};
use super::{ChunkReadWriteIo, plan_update};
use crate::array::{
    Array, ArrayBytes, ArrayError, ChunkGrid, ChunkShape, CodecChainBound, Indexer,
};

/// The chunk of a planned partial encoder, shared by its synchronous and asynchronous variants.
struct PartialChunk {
    codecs: Arc<CodecChainBound>,
    shape: ChunkShape,
    /// Whether partial encodes are planned by the codecs, rather than updates of the whole chunk.
    partial_encodes: bool,
    /// Whether a chunk updated in full is encoded by encode tasks planned by the codecs (see [`plan_update`]).
    encode_subtasks: bool,
}

impl PartialChunk {
    /// Plan the decoding of the elements of `indexer` of the chunk into `target`.
    fn plan_decode<'a>(
        &self,
        indexer: &'a dyn Indexer,
        target: TaskTarget<'a>,
        options: &CodecOptions,
    ) -> Result<zarrs_codec::plan::Task<'a>, CodecError> {
        self.codecs.clone().plan_decode(
            ByteSource::external(SourceKey(0)),
            &self.shape,
            DecodeRequest {
                indexer: PlanIndexer::Borrowed(indexer),
                target,
            },
            options,
        )
    }

    /// Plan resolving the chunk-local subchunk grids of the chunk.
    fn plan_local_subchunk_grids(
        &self,
        options: &CodecOptions,
    ) -> Result<zarrs_codec::plan::Task<'static>, CodecError> {
        self.codecs.clone().plan_local_subchunk_grids(
            ByteSource::external(SourceKey(0)),
            &self.shape,
            options,
        )
    }

    /// Plan a partial encode of the elements of `indexer` to `bytes`, given the `state` of the chunk.
    fn plan_partial_encode<'a>(
        &self,
        indexer: &'a dyn Indexer,
        bytes: &ArrayBytes<'a>,
        state: Option<PartialEncodeState>,
        options: &CodecOptions,
    ) -> Result<zarrs_codec::plan::Task<'a>, CodecError> {
        if !self.partial_encodes {
            return Ok(plan_update(
                self.codecs.clone(),
                SourceKey(0),
                &self.shape,
                PlanIndexer::Borrowed(indexer),
                bytes.clone(),
                options,
                self.encode_subtasks,
            ));
        }
        Ok(self
            .codecs
            .clone()
            .plan_partial_encode(
                SourceKey(0),
                &self.shape,
                PlanIndexer::Borrowed(indexer),
                bytes.clone(),
                state,
                options,
            )?
            .expect("the codecs plan partial encodes"))
    }
}

/// A planned partial encoder of a chunk, driven synchronously.
///
/// Partial decodes read the current chunk (they do not use the state of the partial encoder).
pub(crate) struct PlannedArrayPartialEncoder {
    chunk: PartialChunk,
    io: ChunkReadWriteIo,
    /// The state of the chunk after the last partial encode, or [`None`] if it is unknown.
    ///
    /// The lock is held during a partial encode, so partial encodes of the chunk are serialised.
    state: Mutex<Option<PartialEncodeState>>,
}

impl ArrayPartialDecoderSubchunkingTraits for PlannedArrayPartialEncoder {
    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        if let Some(grids) = static_local_subchunk_grids(&self.chunk.codecs, &self.chunk.shape)? {
            return Ok(grids);
        }
        let task = self.chunk.plan_local_subchunk_grids(options)?;
        local_subchunk_grids(drive_sync(task, &self.io, resources)?)
    }
}

impl ArrayPartialDecoderTraits for PlannedArrayPartialEncoder {
    fn data_type(&self) -> &DataType {
        self.chunk.codecs.data_type()
    }

    fn exists(&self) -> Result<bool, StorageError> {
        Ok(self
            .io
            .storage
            .size_key(&*self.io.keys.store_key(SourceKey(0))?)?
            .is_some())
    }

    fn size_held(&self) -> usize {
        0
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let task = self
            .chunk
            .plan_decode(indexer, TaskTarget::Value, options)?;
        let bytes = array_value(drive_sync(task, &self.io, resources)?)?.into_owned();
        bytes.validate(indexer.len(), self.data_type())?;
        Ok(bytes)
    }

    fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        check_target_len(indexer, &output_target).map_err(array_to_codec_error)?;
        let views = whole_views(&output_target).map_err(array_to_codec_error)?;
        let task = self
            .chunk
            .plan_decode(indexer, TaskTarget::Views(views), options)?;
        drive_sync(task, &self.io, resources)?;
        Ok(())
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

impl ArrayPartialEncoderTraits for PlannedArrayPartialEncoder {
    fn erase(&self) -> Result<(), CodecError> {
        let mut state = self.state.lock().unwrap();
        *state = None;
        Ok(self.io.write(WriteOp::Erase(SourceKey(0)))?)
    }

    fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        let mut state = self.state.lock().unwrap();
        // The state is unknown if the partial encode fails
        let task = self
            .chunk
            .plan_partial_encode(indexer, bytes, state.take(), options)?;
        *state = partial_encode_state(drive_sync(task, &self.io, resources)?);
        Ok(())
    }

    fn supports_partial_encode(&self) -> bool {
        true
    }
}

impl<TStorage: ?Sized + ReadableWritableStorageTraits + 'static> Array<TStorage> {
    /// A planned partial encoder of the chunk at `chunk_indices`.
    pub(crate) fn planned_partial_encoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, ArrayError> {
        Ok(Arc::new(PlannedArrayPartialEncoder {
            chunk: PartialChunk {
                codecs: self.codecs_bound().clone(),
                shape: self.chunk_shape(chunk_indices)?,
                partial_encodes: self.partial_encodes(),
                encode_subtasks: true,
            },
            io: self.chunk_read_write_io(self.one_key(chunk_indices)?)?,
            state: Mutex::new(None),
        }))
    }
}

#[cfg(feature = "async")]
mod async_planned {
    use std::sync::Arc;

    use zarrs_codec::plan::{
        AsyncIo, PartialEncodeState, SourceKey, TaskTarget, WriteOp, drive_async_one,
        local_subchunk_grids, partial_encode_state,
    };
    use zarrs_codec::{
        ArrayBytesDecodeIntoTarget, ArrayCodecTraits, AsyncArrayPartialDecoderSubchunkingTraits,
        AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, CodecError, CodecOptions,
        Resources,
    };
    use zarrs_data_type::DataType;
    use zarrs_storage::{AsyncReadableWritableStorageTraits, StorageError};

    use super::super::super::array_read_ops_planned::{
        array_to_codec_error, array_value, check_target_len, static_local_subchunk_grids,
        whole_views,
    };
    use super::super::async_planned::AsyncChunkReadWriteIo;
    use super::PartialChunk;
    use crate::array::{Array, ArrayBytes, ArrayError, ChunkGrid, Indexer};

    /// A planned partial encoder of a chunk, driven asynchronously.
    ///
    /// Compute runs on the polling task, since the tasks borrow the indexer and elements of a partial encode.
    pub(crate) struct AsyncPlannedArrayPartialEncoder {
        chunk: PartialChunk,
        io: AsyncChunkReadWriteIo,
        /// The state of the chunk after the last partial encode, or [`None`] if it is unknown.
        ///
        /// The lock is held during a partial encode, so partial encodes of the chunk are serialised.
        state: async_lock::Mutex<Option<PartialEncodeState>>,
    }

    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    impl AsyncArrayPartialDecoderSubchunkingTraits for AsyncPlannedArrayPartialEncoder {
        async fn local_subchunk_grids(
            &self,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
            if let Some(grids) = static_local_subchunk_grids(&self.chunk.codecs, &self.chunk.shape)?
            {
                return Ok(grids);
            }
            let task = self.chunk.plan_local_subchunk_grids(options)?;
            local_subchunk_grids(drive_async_one(task, &self.io, resources).await?)
        }
    }

    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    impl AsyncArrayPartialDecoderTraits for AsyncPlannedArrayPartialEncoder {
        fn data_type(&self) -> &DataType {
            self.chunk.codecs.data_type()
        }

        async fn exists(&self) -> Result<bool, StorageError> {
            Ok(self
                .io
                .storage
                .size_key(&*self.io.keys.store_key(SourceKey(0))?)
                .await?
                .is_some())
        }

        fn size_held(&self) -> usize {
            0
        }

        async fn partial_decode<'a>(
            &'a self,
            indexer: &dyn Indexer,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<ArrayBytes<'a>, CodecError> {
            let task = self
                .chunk
                .plan_decode(indexer, TaskTarget::Value, options)?;
            let bytes =
                array_value(drive_async_one(task, &self.io, resources).await?)?.into_owned();
            bytes.validate(indexer.len(), self.data_type())?;
            Ok(bytes)
        }

        async fn partial_decode_into(
            &self,
            indexer: &dyn Indexer,
            output_target: ArrayBytesDecodeIntoTarget<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), CodecError> {
            check_target_len(indexer, &output_target).map_err(array_to_codec_error)?;
            let views = whole_views(&output_target).map_err(array_to_codec_error)?;
            let task = self
                .chunk
                .plan_decode(indexer, TaskTarget::Views(views), options)?;
            drive_async_one(task, &self.io, resources).await?;
            Ok(())
        }

        fn supports_partial_decode(&self) -> bool {
            true
        }
    }

    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    impl AsyncArrayPartialEncoderTraits for AsyncPlannedArrayPartialEncoder {
        async fn erase(&self) -> Result<(), CodecError> {
            let mut state = self.state.lock().await;
            *state = None;
            Ok(self.io.write(WriteOp::Erase(SourceKey(0))).await?)
        }

        async fn partial_encode(
            &self,
            indexer: &dyn Indexer,
            bytes: &ArrayBytes<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), CodecError> {
            let mut state = self.state.lock().await;
            // The state is unknown if the partial encode fails
            let task = self
                .chunk
                .plan_partial_encode(indexer, bytes, state.take(), options)?;
            *state = partial_encode_state(drive_async_one(task, &self.io, resources).await?);
            Ok(())
        }

        fn supports_partial_encode(&self) -> bool {
            true
        }
    }

    impl<TStorage: ?Sized + AsyncReadableWritableStorageTraits + 'static> Array<TStorage> {
        /// A planned partial encoder of the chunk at `chunk_indices`.
        pub(crate) async fn async_planned_partial_encoder(
            &self,
            chunk_indices: &[u64],
        ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, ArrayError> {
            Ok(Arc::new(AsyncPlannedArrayPartialEncoder {
                chunk: PartialChunk {
                    codecs: self.codecs_bound().clone(),
                    shape: self.chunk_shape(chunk_indices)?,
                    partial_encodes: self.async_partial_encodes(),
                    // The tasks of a chunk would run serially on the polling task
                    encode_subtasks: false,
                },
                io: self
                    .async_chunk_read_write_io(self.one_key(chunk_indices)?)
                    .await?,
                state: async_lock::Mutex::new(None),
            }))
        }
    }
}
