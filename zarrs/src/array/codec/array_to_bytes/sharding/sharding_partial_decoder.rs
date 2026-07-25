#[cfg(feature = "async")]
mod futures;
mod rayon;

use std::num::NonZeroU64;
use std::sync::Arc;

use ambisync::ambisync;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::ArraySubset;
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderTraits,
    ArrayToBytesCodecTraits, ByteIntervalPartialDecoder, BytesPartialDecoderTraits, CodecError,
    CodecOptions, InvalidNumberOfElementsError, decode_into_array_bytes_target,
};
#[cfg(feature = "async")]
use zarrs_codec::{
    AsyncArrayPartialDecoderTraits, AsyncByteIntervalPartialDecoder, AsyncBytesPartialDecoderTraits,
};
use zarrs_plugin::ExtensionAliasesV3;
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteLength, ByteOffset, ByteRange};

use super::{ShardingCodecOptions, ShardingIndexLocation, nested_local_subchunk_grids};
use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesOffsets, ArrayBytesRaw, ArrayIndices,
    ChunkGrid, ChunkShape, ChunkShapeTraits, CodecChainBound, DataType, DataTypeSize, Indexer,
    IndexerError, ravel_indices,
};

/// Partial decoder for the sharding codec.
#[ambisync(
    sync(
        types(
            AsyncShardingPartialDecoder => ShardingPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
pub struct AsyncShardingPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    shard_shape: ChunkShape,
    subchunk_shape: ChunkShape,
    inner_codecs: Arc<CodecChainBound>,
    shard_index: Option<Vec<u64>>,
    #[expect(dead_code)] // TODO: Remove when sharding-specific options are added
    sharding_options: ShardingCodecOptions,
}

#[ambisync(
    sync(
        fns(
            "{}",
            decode_shard_index_async_partial_decoder => decode_shard_index_partial_decoder,
        ),
        types(
            AsyncShardingPartialDecoder => ShardingPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
impl AsyncShardingPartialDecoder {
    /// Create a new partial decoder for the sharding codec.
    #[expect(clippy::too_many_arguments)]
    pub async fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shard_shape: ChunkShape,
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChainBound>,
        index_codecs: &CodecChainBound,
        index_location: ShardingIndexLocation,
        options: &CodecOptions,
        sharding_options: ShardingCodecOptions,
    ) -> Result<Self, CodecError> {
        let shard_index = super::decode_shard_index_async_partial_decoder(
            &*input_handle,
            index_codecs,
            index_location,
            &shard_shape,
            &subchunk_shape,
            options,
        )
        .await?;

        Ok(Self {
            input_handle,
            shard_shape,
            subchunk_shape,
            inner_codecs,
            shard_index,
            sharding_options,
        })
    }

    /// Retrieve the byte range of an encoded subchunk.
    ///
    /// The `chunk_indices` are relative to the start of the shard.
    pub fn subchunk_byte_range(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ByteRange>, CodecError> {
        super::subchunk_byte_range(
            self.shard_index.as_deref(),
            &self.shard_shape,
            &self.subchunk_shape,
            chunk_indices,
        )
    }

    /// Retrieve the encoded bytes of a subchunk.
    ///
    /// The `chunk_indices` are relative to the start of the shard.
    pub async fn retrieve_subchunk_encoded(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayBytesRaw<'_>>, CodecError> {
        let byte_range = self.subchunk_byte_range(chunk_indices)?;
        if let Some(byte_range) = byte_range {
            self.input_handle
                .partial_decode(byte_range, &CodecOptions::default())
                .await
        } else {
            Ok(None)
        }
    }
}

#[ambisync(
    sync(
        name = "partial_decode",
        types(AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits),
    ),
    async(feature = "async"),
)]
pub(crate) async fn partial_decode_async(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    if indexer.dimensionality() != shard_shape.len() {
        return Err(IndexerError::new_incompatible_dimensionality(
            indexer.dimensionality(),
            shard_shape.len(),
        )
        .into());
    }

    let data_type = inner_codecs.data_type();
    if data_type.is_optional() {
        return Err(CodecError::UnsupportedDataType(
            data_type.clone(),
            super::ShardingCodec::aliases_v3().default_name.to_string(),
        ));
    }

    match data_type.size() {
        DataTypeSize::Fixed(_data_type_size) => {
            if let Some(subset) = indexer.as_array_subset() {
                ambisync::alt!(
                    sync => {
                        let array_shape = subset.shape();
                        let array_subset_size = subset.num_elements_usize() * _data_type_size;
                        let mut output = vec![0; array_subset_size];
                        let output_slice = UnsafeCellSlice::new(output.as_mut_slice());
                        let mut output_view = unsafe {
                            ArrayBytesFixedDisjointView::new(
                                output_slice,
                                _data_type_size,
                                &array_shape,
                                ArraySubset::new_with_shape(array_shape.to_vec()),
                            )?
                        };
                        rayon::partial_decode_fixed_array_subset_into(
                            input_handle,
                            shard_shape,
                            subchunk_shape,
                            inner_codecs,
                            shard_index,
                            subset,
                            options,
                            &mut output_view,
                        )?;
                        Ok(ArrayBytes::from(output))
                    },
                    async => futures::partial_decode_fixed_array_subset(
                        input_handle,
                        shard_shape,
                        subchunk_shape,
                        inner_codecs,
                        shard_index,
                        subset,
                        options,
                    ).await,
                )
            } else {
                ambisync::alt!(
                    sync => partial_decode_fixed_indexer(
                        input_handle,
                        shard_shape,
                        subchunk_shape,
                        inner_codecs,
                        shard_index,
                        indexer,
                        options,
                    ),
                    async => partial_decode_fixed_indexer_async(
                        input_handle,
                        shard_shape,
                        subchunk_shape,
                        inner_codecs,
                        shard_index,
                        indexer,
                        options,
                    ).await,
                )
            }
        }
        DataTypeSize::Variable => {
            if let Some(subset) = indexer.as_array_subset() {
                ambisync::alt!(
                    sync => rayon::partial_decode_variable_array_subset(
                        input_handle,
                        shard_shape,
                        subchunk_shape,
                        inner_codecs,
                        shard_index,
                        subset,
                        options,
                    ),
                    async => futures::partial_decode_variable_array_subset(
                        input_handle,
                        data_type,
                        inner_codecs.fill_value(),
                        shard_shape,
                        subchunk_shape,
                        inner_codecs,
                        shard_index,
                        subset,
                        options,
                    ).await,
                )
            } else {
                ambisync::alt!(
                    sync => partial_decode_variable_indexer(
                        input_handle,
                        shard_shape,
                        subchunk_shape,
                        inner_codecs,
                        shard_index,
                        indexer,
                        options,
                    ),
                    async => partial_decode_variable_indexer_async(
                        input_handle,
                        shard_shape,
                        subchunk_shape,
                        inner_codecs,
                        shard_index,
                        indexer,
                        options,
                    ).await,
                )
            }
        }
    }
}

#[ambisync(
    sync(
        name = "get_subchunk_partial_decoder",
        fns(async_partial_decoder => partial_decoder),
        types(
            AsyncArrayPartialDecoderTraits => ArrayPartialDecoderTraits,
            AsyncByteIntervalPartialDecoder => ByteIntervalPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
async fn get_subchunk_partial_decoder_async(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    options: &CodecOptions,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
    inner_codecs
        .clone()
        .async_partial_decoder(
            Arc::new(AsyncByteIntervalPartialDecoder::new(
                input_handle.clone(),
                byte_offset,
                byte_length,
            )),
            subchunk_shape,
            options,
        )
        .await
        .map_err(|error| {
            if let CodecError::InvalidByteRangeError(_) = error {
                CodecError::Other(
                    "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                        .to_string(),
                )
            } else {
                error
            }
        })
}

#[ambisync(
    sync(
        name = "partial_decode_fixed_indexer",
        fns(get_subchunk_partial_decoder_async => get_subchunk_partial_decoder),
        types(
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
async fn partial_decode_fixed_indexer_async(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let data_type_size = data_type.fixed_size().expect("called on fixed data type");
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, indexer);
    };
    let chunks_per_shard =
        super::calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let options = &options;

    let output_len = usize::try_from(indexer.len() * data_type_size as u64).unwrap();
    let mut output: Vec<u8> = Vec::with_capacity(output_len);

    #[cfg(not(target_arch = "wasm32"))]
    let subchunk_partial_decoders = ambisync::alt!(
        sync => moka::sync::Cache::new(chunks_per_shard.iter().product()),
        async => moka::future::Cache::new(chunks_per_shard.iter().product()),
    );
    #[cfg(target_arch = "wasm32")]
    let subchunk_partial_decoders = quick_cache::sync::Cache::new(
        usize::try_from(chunks_per_shard.iter().product::<u64>()).unwrap(),
    );

    for indices in indexer.iter_indices() {
        if indices.len() != shard_shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                indices.len(),
                shard_shape.len(),
            )
            .into());
        }
        let chunk_index: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&index, &chunk_size)| index / chunk_size)
            .collect();
        let chunk_index_1d = ravel_indices(&chunk_index, &chunks_per_shard)
            .ok_or_else(|| IndexerError::new_oob(chunk_index, chunks_per_shard.clone()))?;

        let shard_index_idx = usize::try_from(chunk_index_1d).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        #[cfg(not(target_arch = "wasm32"))]
        let inner_partial_decoder = ambisync::alt!(
            sync => subchunk_partial_decoders
                .entry(chunk_index_1d)
                .or_try_insert_with(|| get_subchunk_partial_decoder_async(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                ))
                .map_err(Arc::unwrap_or_clone)?
                .into_value(),
            async => subchunk_partial_decoders
                .entry(chunk_index_1d)
                .or_try_insert_with(get_subchunk_partial_decoder_async(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                ))
                .await
                .map_err(Arc::unwrap_or_clone)?
                .into_value(),
        );
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder = ambisync::alt!(
            sync => subchunk_partial_decoders.get_or_insert_with(&chunk_index_1d, || {
                get_subchunk_partial_decoder_async(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                )
            })?,
            async => subchunk_partial_decoders
                .get_or_insert_async(&chunk_index_1d, async {
                    get_subchunk_partial_decoder_async(
                        input_handle,
                        subchunk_shape,
                        inner_codecs,
                        options,
                        offset,
                        size,
                    )
                    .await
                })
                .await?,
        );

        let indices_in_subchunk: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&index, &chunk_size)| index - (index / chunk_size) * chunk_size.get())
            .collect();

        let element_bytes = inner_partial_decoder
            .partial_decode(&[indices_in_subchunk], options)
            .await?
            .into_fixed()
            .expect("fixed data");
        output.extend_from_slice(&element_bytes);
    }

    debug_assert_eq!(output.len(), output_len);
    Ok(output.into())
}

#[ambisync(
    sync(
        name = "partial_decode_variable_indexer",
        fns(get_subchunk_partial_decoder_async => get_subchunk_partial_decoder),
        types(
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
async fn partial_decode_variable_indexer_async(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    inner_codecs: &Arc<CodecChainBound>,
    shard_index: Option<&[u64]>,
    indexer: &dyn Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = inner_codecs.data_type();
    let fill_value = inner_codecs.fill_value();
    let Some(shard_index) = shard_index else {
        return super::partial_decode_empty_shard(data_type, fill_value, indexer);
    };
    let chunks_per_shard =
        super::calculate_chunks_per_shard(shard_shape, subchunk_shape)?.to_array_shape();
    let options = &options;

    let offsets_len = usize::try_from(indexer.len() + 1).unwrap();
    let mut bytes: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = Vec::with_capacity(offsets_len);
    offsets.push(0);

    #[cfg(not(target_arch = "wasm32"))]
    let subchunk_partial_decoders = ambisync::alt!(
        sync => moka::sync::Cache::new(chunks_per_shard.iter().product()),
        async => moka::future::Cache::new(chunks_per_shard.iter().product()),
    );
    #[cfg(target_arch = "wasm32")]
    let subchunk_partial_decoders = quick_cache::sync::Cache::new(
        usize::try_from(chunks_per_shard.iter().product::<u64>()).unwrap(),
    );

    for indices in indexer.iter_indices() {
        if indices.len() != shard_shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                indices.len(),
                shard_shape.len(),
            )
            .into());
        }
        let chunk_index: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&index, &chunk_size)| index / chunk_size)
            .collect();
        let chunk_index_1d = ravel_indices(&chunk_index, &chunks_per_shard)
            .ok_or_else(|| IndexerError::new_oob(chunk_index, chunks_per_shard.clone()))?;

        let shard_index_idx = usize::try_from(chunk_index_1d).unwrap();
        let offset = shard_index[shard_index_idx * 2];
        let size = shard_index[shard_index_idx * 2 + 1];

        #[cfg(not(target_arch = "wasm32"))]
        let inner_partial_decoder = ambisync::alt!(
            sync => subchunk_partial_decoders
                .entry(chunk_index_1d)
                .or_try_insert_with(|| get_subchunk_partial_decoder_async(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                ))
                .map_err(Arc::unwrap_or_clone)?
                .into_value(),
            async => subchunk_partial_decoders
                .entry(chunk_index_1d)
                .or_try_insert_with(get_subchunk_partial_decoder_async(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                ))
                .await
                .map_err(Arc::unwrap_or_clone)?
                .into_value(),
        );
        #[cfg(target_arch = "wasm32")]
        let inner_partial_decoder = ambisync::alt!(
            sync => subchunk_partial_decoders.get_or_insert_with(&chunk_index_1d, || {
                get_subchunk_partial_decoder_async(
                    input_handle,
                    subchunk_shape,
                    inner_codecs,
                    options,
                    offset,
                    size,
                )
            })?,
            async => subchunk_partial_decoders
                .get_or_insert_async(&chunk_index_1d, async {
                    get_subchunk_partial_decoder_async(
                        input_handle,
                        subchunk_shape,
                        inner_codecs,
                        options,
                        offset,
                        size,
                    )
                    .await
                })
                .await?,
        );

        let indices_in_subchunk: ArrayIndices = indices
            .iter()
            .zip(subchunk_shape)
            .map(|(&index, &chunk_size)| index - (index / chunk_size) * chunk_size.get())
            .collect();

        let (element_bytes, element_offsets) = inner_partial_decoder
            .partial_decode(&[indices_in_subchunk], options)
            .await?
            .into_variable()?
            .into_parts();
        debug_assert_eq!(element_offsets.len(), 2);
        bytes.extend_from_slice(&element_bytes);
        offsets.push(bytes.len());
    }

    Ok(ArrayBytes::new_vlen(
        bytes,
        ArrayBytesOffsets::new(offsets)?,
    )?)
}

#[ambisync(
    sync(
        fns("{}", partial_decode_async => partial_decode),
        types(
            AsyncArrayPartialDecoderTraits => ArrayPartialDecoderTraits,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
            AsyncShardingPartialDecoder => ShardingPartialDecoder,
        ),
    ),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32")),
    ),
)]
impl AsyncArrayPartialDecoderTraits for AsyncShardingPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.inner_codecs.data_type()
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
            + self.shard_index.as_ref().map_or(0, Vec::len) * size_of::<u64>()
    }

    async fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        partial_decode_async(
            &self.input_handle,
            &self.shard_shape,
            &self.subchunk_shape,
            &self.inner_codecs,
            self.shard_index.as_deref(),
            indexer,
            options,
        )
        .await
    }

    async fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        let shard_shape = bytemuck::must_cast_slice(&self.shard_shape).to_vec();
        let subchunk_grid = ChunkGrid::new(
            RegularChunkGrid::new(shard_shape, self.subchunk_shape.clone())
                .map_err(|err| CodecError::Other(err.to_string()))?,
        );
        nested_local_subchunk_grids(subchunk_grid, &self.inner_codecs)
    }

    async fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if indexer.len() != output_target.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                indexer.len(),
                output_target.num_elements(),
            )
            .into());
        }
        if let DataTypeSize::Fixed(_) = self.inner_codecs.data_type().size()
            && let Some(subset) = indexer.as_array_subset()
            && let ArrayBytesDecodeIntoTarget::Fixed(output_view) = output_target
        {
            ambisync::alt!(
                sync => rayon::partial_decode_fixed_array_subset_into(
                    &self.input_handle,
                    &self.shard_shape,
                    &self.subchunk_shape,
                    &self.inner_codecs,
                    self.shard_index.as_deref(),
                    subset,
                    options,
                    output_view,
                ),
                async => futures::partial_decode_fixed_array_subset_into(
                    &self.input_handle,
                    &self.shard_shape,
                    &self.subchunk_shape,
                    &self.inner_codecs,
                    self.shard_index.as_deref(),
                    subset,
                    options,
                    output_view,
                ).await,
            )
        } else {
            let decoded_value = self.partial_decode(indexer, options).await?;
            decode_into_array_bytes_target(&decoded_value, output_target)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}
