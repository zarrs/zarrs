//! Partial decoders of a chunk that plan each request with a decoder prepared once, and drive it.

use std::sync::Arc;

use zarrs_codec::plan::{
    ByteSource, DecodeRequest, MissingDecoder, PlanIndexer, PrepareDecode, PreparedDecoder,
    ReadBytes, SourceKey, SyncIo, Task, TaskTarget, drive_memory_one, local_subchunk_grids,
    prepared,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderSubchunkingTraits,
    ArrayPartialDecoderTraits, ArrayToBytesCodecSubchunkingTraits, ArrayToBytesCodecTraits,
    ChunkGridDecoded, ChunkGridDecodedRef, CodecError, CodecOptions, CowBytes, Resources,
};
use zarrs_data_type::DataType;
use zarrs_storage::byte_range::ByteRange;
use zarrs_storage::{Bytes, ReadableStorageTraits, StorageError};

use super::{
    Array, ArrayBytes, ArrayError, ChunkKeys, CodecChainBound, Indexer, OwnedChunkIo, array_value,
    check_target_len, drive_blocking_one, whole_views,
};
use crate::array::chunk_grid::RegularBoundedChunkGrid;
use crate::array::{ChunkGrid, ChunkShape};

/// The state of a planned partial decoder shared by its synchronous and asynchronous variants.
struct PlannedChunk {
    decoder: Arc<dyn PreparedDecoder>,
    codecs: Arc<CodecChainBound>,
    shape: ChunkShape,
}

impl PlannedChunk {
    /// The request for the elements of `indexer` as a value.
    fn value_request<'a>(indexer: &'a dyn Indexer) -> DecodeRequest<'a> {
        DecodeRequest {
            indexer: PlanIndexer::Borrowed(indexer),
            target: TaskTarget::Value,
        }
    }

    /// The chunk-local subchunk grids of the chunk, outermost first, if they are independent of the chunk bytes.
    ///
    /// Returns [`None`] if a codec has chunk-local subchunk grids that depend on the bytes of the chunk.
    fn static_local_subchunk_grids(&self) -> Result<Option<Vec<Option<ChunkGrid>>>, CodecError> {
        static_local_subchunk_grids(&self.codecs, &self.shape)
    }

    /// Plan resolving the chunk-local subchunk grids of the chunk from its encoded bytes `source`.
    fn plan_local_subchunk_grids<'a>(
        &self,
        source: ByteSource<'a>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        self.codecs
            .clone()
            .plan_local_subchunk_grids(source, &self.shape, options)
    }

    fn data_type(&self) -> &DataType {
        self.codecs.data_type()
    }

    /// Every request is planned to read and decode only what it needs, or extracted from the decoded chunk if the
    /// codecs cannot decode part of it.
    fn supports_partial_decode(&self) -> bool {
        true
    }

    /// The chunk-local subchunk grids of the chunk with encoded bytes `source`, driven synchronously with `io`.
    fn local_subchunk_grids(
        &self,
        source: ByteSource<'_>,
        io: &dyn SyncIo,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        if let Some(grids) = self.static_local_subchunk_grids()? {
            return Ok(grids);
        }
        let task = self.plan_local_subchunk_grids(source, options)?;
        local_subchunk_grids(drive_blocking_one(task, io, resources)?)
    }

    /// Decode the elements `indexer`, driven synchronously with `io`.
    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        io: &dyn SyncIo,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, CodecError> {
        let task = self
            .decoder
            .clone()
            .plan_decode(Self::value_request(indexer), options)?;
        let bytes = array_value(drive_blocking_one(task, io, resources)?)?.into_owned();
        bytes.validate(indexer.len(), self.data_type())?;
        Ok(bytes)
    }

    /// Decode the elements `indexer` into `output_target`, driven synchronously with `io`.
    fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        io: &dyn SyncIo,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        check_target_len(indexer, &output_target).map_err(array_to_codec_error)?;
        let views = whole_views(&output_target).map_err(array_to_codec_error)?;
        let task = self.decoder.clone().plan_decode(
            DecodeRequest {
                indexer: PlanIndexer::Borrowed(indexer),
                target: TaskTarget::Views(views),
            },
            options,
        )?;
        drive_blocking_one(task, io, resources)?;
        Ok(())
    }
}

/// A partial decoder of a chunk that plans each request with a decoder prepared once (e.g. reading a shard index), and
/// drives it with the synchronous driver.
pub(crate) struct PlannedArrayPartialDecoder {
    chunk: PlannedChunk,
    io: OwnedChunkIo,
}

impl ArrayPartialDecoderSubchunkingTraits for PlannedArrayPartialDecoder {
    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.chunk.local_subchunk_grids(
            ByteSource::external(SourceKey(0)),
            &self.io,
            options,
            resources,
        )
    }
}

impl ArrayPartialDecoderTraits for PlannedArrayPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.chunk.data_type()
    }

    fn exists(&self) -> Result<bool, StorageError> {
        match self.chunk.decoder.exists() {
            Some(exists) => Ok(exists),
            None => self.io.exists(SourceKey(0)),
        }
    }

    fn size_held(&self) -> usize {
        self.chunk.decoder.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        self.chunk
            .partial_decode(indexer, &self.io, options, resources)
    }

    fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        self.chunk
            .partial_decode_into(indexer, output_target, &self.io, options, resources)
    }

    fn supports_partial_decode(&self) -> bool {
        self.chunk.supports_partial_decode()
    }
}

/// The chunk-local subchunk grids of a chunk of `shape` with `codecs`, outermost first, if they are independent of the
/// chunk bytes.
///
/// Returns [`None`] if a codec has chunk-local subchunk grids that depend on the bytes of the chunk.
pub(in crate::array::array_ops) fn static_local_subchunk_grids(
    codecs: &CodecChainBound,
    shape: &ChunkShape,
) -> Result<Option<Vec<Option<ChunkGrid>>>, CodecError> {
    let shape_u64: &[u64] = bytemuck::must_cast_slice(shape);
    let chunk_grid = ChunkGrid::new(RegularBoundedChunkGrid::new(
        shape_u64.to_vec(),
        shape.clone(),
    )?);
    Ok(codecs
        .decoded_subchunk_grids(ChunkGridDecodedRef::Array(&chunk_grid))
        .map_err(|err| CodecError::Other(err.to_string()))?
        .into_iter()
        .map(|grid| match grid {
            ChunkGridDecoded::Array(grid) => Some(Some(grid)),
            ChunkGridDecoded::None => Some(None),
            ChunkGridDecoded::ChunkLocal => None,
        })
        .collect())
}

#[allow(clippy::wildcard_enum_match_arm)]
pub(in crate::array::array_ops) fn array_to_codec_error(err: ArrayError) -> CodecError {
    match err {
        ArrayError::CodecError(err) => err,
        err => CodecError::Other(err.to_string()),
    }
}

impl<TStorage: ?Sized> Array<TStorage> {
    /// Prepare the decoding of the chunk at `chunk_indices`.
    fn prepare_chunk(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<(PrepareDecode, Arc<CodecChainBound>, ChunkShape, ChunkKeys), ArrayError> {
        let shape = self.chunk_shape(chunk_indices)?;
        let keys = self.one_key(chunk_indices)?;
        let codecs = self.codecs_bound().clone();
        let prepare =
            codecs
                .clone()
                .prepare_decode(ByteSource::external(SourceKey(0)), &shape, options)?;
        Ok((prepare, codecs, shape, keys))
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits + 'static> Array<TStorage> {
    /// A partial decoder of the chunk at `chunk_indices` that plans each request, prepared with a planned read.
    pub(crate) fn planned_partial_decoder(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        let (prepare, codecs, shape, keys) = self.prepare_chunk(chunk_indices, options)?;
        let io = self.owned_chunk_io(keys)?;
        let decoder = match prepare {
            PrepareDecode::Ready(decoder) => decoder,
            PrepareDecode::Plan(task) => prepared(drive_blocking_one(task, &io, resources)?)?,
        };
        Ok(Arc::new(PlannedArrayPartialDecoder {
            chunk: PlannedChunk {
                decoder,
                codecs,
                shape,
            },
            io,
        }))
    }
}

/// The I/O backend of a partial decoder of a chunk in memory, which only reads a chunk that does not exist.
struct MemoryChunkIo {
    exists: bool,
}

impl SyncIo for MemoryChunkIo {
    fn read(
        &self,
        _key: SourceKey,
        _ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        if self.exists {
            Err(StorageError::Other(
                "a task with an in-memory source requested I/O".to_string(),
            ))
        } else {
            Ok(None)
        }
    }
}

/// A partial decoder of an encoded chunk in memory (e.g. in a chunk cache) that plans each request with a decoder
/// prepared once.
pub(crate) struct PlannedMemoryPartialDecoder {
    chunk: PlannedChunk,
    encoded: Option<Bytes>,
    io: MemoryChunkIo,
}

impl ArrayPartialDecoderSubchunkingTraits for PlannedMemoryPartialDecoder {
    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        let source = match &self.encoded {
            Some(encoded) => ByteSource::memory(CowBytes::Shared(encoded.clone())),
            None => ByteSource::external(SourceKey(0)),
        };
        self.chunk
            .local_subchunk_grids(source, &self.io, options, resources)
    }
}

impl ArrayPartialDecoderTraits for PlannedMemoryPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.chunk.data_type()
    }

    fn exists(&self) -> Result<bool, StorageError> {
        Ok(self.encoded.is_some())
    }

    fn size_held(&self) -> usize {
        self.chunk.decoder.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        self.chunk
            .partial_decode(indexer, &self.io, options, resources)
    }

    fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        self.chunk
            .partial_decode_into(indexer, output_target, &self.io, options, resources)
    }

    fn supports_partial_decode(&self) -> bool {
        self.chunk.supports_partial_decode()
    }
}

impl<TStorage: ?Sized> Array<TStorage> {
    /// A partial decoder of the `encoded` chunk at `chunk_indices` (or a missing chunk if [`None`]) that plans each
    /// request.
    pub(crate) fn planned_memory_partial_decoder(
        &self,
        encoded: Option<Bytes>,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        let shape = self.chunk_shape(chunk_indices)?;
        let codecs = self.codecs_bound().clone();
        let decoder: Arc<dyn PreparedDecoder> = if let Some(encoded) = &encoded {
            let source = ByteSource::memory(CowBytes::Shared(encoded.clone()));
            match codecs.clone().prepare_decode(source, &shape, options)? {
                PrepareDecode::Ready(decoder) => decoder,
                PrepareDecode::Plan(task) => prepared(drive_memory_one(task, resources)?)?,
            }
        } else {
            Arc::new(MissingDecoder::new(codecs.clone().into_dyn(), &shape))
        };
        Ok(Arc::new(PlannedMemoryPartialDecoder {
            chunk: PlannedChunk {
                decoder,
                codecs,
                shape,
            },
            io: MemoryChunkIo {
                exists: encoded.is_some(),
            },
            encoded,
        }))
    }
}

#[cfg(feature = "async")]
mod async_partial_decoder {
    use std::sync::Arc;

    use zarrs_codec::plan::{
        ByteSource, DecodeRequest, PlanIndexer, PrepareDecode, TaskTarget, drive_async_one,
        local_subchunk_grids, prepared,
    };
    use zarrs_codec::{
        ArrayBytesDecodeIntoTarget, AsyncArrayPartialDecoderSubchunkingTraits,
        AsyncArrayPartialDecoderTraits, CodecError, CodecOptions, Resources,
    };
    use zarrs_data_type::DataType;
    use zarrs_storage::{AsyncReadableStorageTraits, StorageError};

    use zarrs_codec::plan::drive_async_offload_one;

    use super::super::async_planned::OwnedAsyncChunkIo;
    use super::{
        Array, ArrayBytes, ArrayError, Indexer, PlannedChunk, SourceKey, array_to_codec_error,
        array_value, check_target_len, whole_views,
    };
    use crate::array::ChunkGrid;

    /// A partial decoder of a chunk that plans each request with a decoder prepared once (e.g. reading a shard index),
    /// and drives it with the asynchronous driver.
    pub(crate) struct AsyncPlannedArrayPartialDecoder {
        chunk: PlannedChunk,
        io: OwnedAsyncChunkIo,
    }

    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    impl AsyncArrayPartialDecoderSubchunkingTraits for AsyncPlannedArrayPartialDecoder {
        async fn local_subchunk_grids(
            &self,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
            if let Some(grids) = self.chunk.static_local_subchunk_grids()? {
                return Ok(grids);
            }
            let task = self
                .chunk
                .plan_local_subchunk_grids(ByteSource::external(SourceKey(0)), options)?;
            local_subchunk_grids(drive_async_one(task, &self.io, resources).await?)
        }
    }

    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    impl AsyncArrayPartialDecoderTraits for AsyncPlannedArrayPartialDecoder {
        fn data_type(&self) -> &DataType {
            self.chunk.data_type()
        }

        async fn exists(&self) -> Result<bool, StorageError> {
            match self.chunk.decoder.exists() {
                Some(exists) => Ok(exists),
                None => self.io.exists(SourceKey(0)).await,
            }
        }

        fn size_held(&self) -> usize {
            self.chunk.decoder.size_held()
        }

        async fn partial_decode<'a>(
            &'a self,
            indexer: &dyn Indexer,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<ArrayBytes<'a>, CodecError> {
            let bytes = if let Some(subset) = indexer.as_array_subset() {
                // The tasks own the request, so their compute can be offloaded to Rayon
                let task = self.chunk.decoder.clone().plan_decode(
                    DecodeRequest {
                        indexer: PlanIndexer::Subset(subset.to_array_subset()),
                        target: TaskTarget::Value,
                    },
                    options,
                )?;
                drive_async_offload_one(task, &self.io, resources).await?
            } else {
                let task = self
                    .chunk
                    .decoder
                    .clone()
                    .plan_decode(PlannedChunk::value_request(indexer), options)?;
                zarrs_codec::plan::drive_async_one(task, &self.io, resources).await?
            };
            let bytes = array_value(bytes)?.into_owned();
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
            let task = self.chunk.decoder.clone().plan_decode(
                DecodeRequest {
                    indexer: PlanIndexer::Borrowed(indexer),
                    target: TaskTarget::Views(views),
                },
                options,
            )?;
            zarrs_codec::plan::drive_async_one(task, &self.io, resources).await?;
            Ok(())
        }

        fn supports_partial_decode(&self) -> bool {
            self.chunk.supports_partial_decode()
        }
    }

    impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> Array<TStorage> {
        /// A partial decoder of the chunk at `chunk_indices` that plans each request, prepared with a planned read.
        pub(crate) async fn async_planned_partial_decoder(
            &self,
            chunk_indices: &[u64],
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
            let (prepare, codecs, shape, keys) = self.prepare_chunk(chunk_indices, options)?;
            let io = self.async_owned_chunk_io(keys).await?;
            let decoder = match prepare {
                PrepareDecode::Ready(decoder) => decoder,
                PrepareDecode::Plan(task) => {
                    prepared(drive_async_offload_one(task, &io, resources).await?)?
                }
            };
            Ok(Arc::new(AsyncPlannedArrayPartialDecoder {
                chunk: PlannedChunk {
                    decoder,
                    codecs,
                    shape,
                },
                io,
            }))
        }
    }
}
