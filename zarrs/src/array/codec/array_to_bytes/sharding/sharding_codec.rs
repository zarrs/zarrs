use std::{
    borrow::Cow,
    num::NonZeroU64,
    sync::{Arc, atomic::AtomicUsize},
};

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_plugin::ExtensionIdentifier;

#[cfg(feature = "async")]
use super::sharding_partial_decoder_async::AsyncShardingPartialDecoder;
use super::{
    ShardingCodecConfiguration, ShardingCodecConfigurationV1, ShardingIndexLocation,
    calculate_chunks_per_shard, compute_index_encoded_size, decode_shard_index,
    sharding_index_shape, sharding_partial_decoder_sync::ShardingPartialDecoder,
    sharding_partial_encoder,
};
#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};
use crate::metadata::Configuration;
use crate::{
    array::{
        ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesRaw, BytesRepresentation, ChunkShape,
        ChunkShapeTraits, DataType, DataTypeSize, FillValue,
        array_bytes::merge_chunks_vlen,
        chunk_shape_to_array_shape,
        codec::{
            ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderTraits,
            ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, BytesPartialDecoderTraits,
            BytesPartialEncoderTraits, CodecChain, CodecError, CodecMetadataOptions, CodecOptions,
            CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
            RecommendedConcurrency,
        },
        concurrency::calc_concurrency_outer_inner,
        data_type::DataTypeExt,
        transmute_to_bytes_vec, unravel_index,
    },
    array_subset::ArraySubset,
    plugin::PluginCreateError,
};

/// A `sharding` codec implementation.
#[derive(Clone, Debug)]
pub struct ShardingCodec {
    /// An array of integers specifying the shape of the inner chunks in a shard along each dimension of the outer array.
    pub(crate) subchunk_shape: ChunkShape,
    /// The codecs used to encode and decode inner chunks.
    pub(crate) inner_codecs: Arc<CodecChain>,
    /// The codecs used to encode and decode the shard index.
    pub(crate) index_codecs: Arc<CodecChain>,
    /// Specifies whether the shard index is located at the beginning or end of the file.
    pub(crate) index_location: ShardingIndexLocation,
}

impl ShardingCodec {
    /// Create a new `sharding` codec.
    #[must_use]
    pub fn new(
        subchunk_shape: ChunkShape,
        inner_codecs: Arc<CodecChain>,
        index_codecs: Arc<CodecChain>,
        index_location: ShardingIndexLocation,
    ) -> Self {
        Self {
            subchunk_shape,
            inner_codecs,
            index_codecs,
            index_location,
        }
    }

    /// Create a new `sharding` codec from configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if there is a configuration issue.
    pub fn new_with_configuration(
        configuration: &ShardingCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            ShardingCodecConfiguration::V1(configuration) => {
                let inner_codecs = Arc::new(CodecChain::from_metadata(&configuration.codecs)?);
                let index_codecs =
                    Arc::new(CodecChain::from_metadata(&configuration.index_codecs)?);
                Ok(Self::new(
                    configuration.chunk_shape.clone(),
                    inner_codecs,
                    index_codecs,
                    configuration.index_location,
                ))
            }
            _ => Err(PluginCreateError::Other(
                "this sharding_indexed codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for ShardingCodec {
    fn identifier(&self) -> &'static str {
        Self::IDENTIFIER
    }

    fn configuration(&self, _name: &str, options: &CodecMetadataOptions) -> Option<Configuration> {
        let configuration = ShardingCodecConfiguration::V1(ShardingCodecConfigurationV1 {
            chunk_shape: self.subchunk_shape.clone(),
            codecs: self.inner_codecs.create_metadatas(options),
            index_codecs: self.index_codecs.create_metadatas(options),
            index_location: self.index_location,
        });
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: true,
        }
    }
}

impl ArrayCodecTraits for ShardingCodec {
    fn recommended_concurrency(
        &self,
        shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        let chunks_per_shard = calculate_chunks_per_shard(shape, self.subchunk_shape.as_slice())?;
        let num_elements = chunks_per_shard.num_elements_nonzero_usize();
        Ok(RecommendedConcurrency::new_maximum(num_elements.into()))
    }

    fn partial_decode_granularity(&self, _shape: &[NonZeroU64]) -> ChunkShape {
        self.subchunk_shape.clone()
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for ShardingCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        let num_elements = shape.iter().map(|d| d.get()).product::<u64>();
        bytes.validate(num_elements, data_type)?;

        // Get chunk bytes representation, and choose implementation based on whether the size is unbounded or not
        let chunk_bytes_representation = self.inner_codecs.encoded_representation(
            &self.subchunk_shape,
            data_type,
            fill_value,
        )?;

        let bytes = match chunk_bytes_representation {
            BytesRepresentation::BoundedSize(size) | BytesRepresentation::FixedSize(size) => self
                .encode_bounded(
                    &bytes,
                    data_type,
                    fill_value,
                    shape,
                    &self.subchunk_shape,
                    size,
                    options,
                ),
            BytesRepresentation::UnboundedSize => self.encode_unbounded(
                &bytes,
                data_type,
                fill_value,
                shape,
                &self.subchunk_shape,
                options,
            ),
        }?;
        Ok(ArrayBytesRaw::from(bytes))
    }

    #[allow(clippy::too_many_lines)]
    fn decode<'a>(
        &self,
        encoded_shard: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let chunks_per_shard = calculate_chunks_per_shard(shape, &self.subchunk_shape)?;
        let num_chunks = chunks_per_shard
            .as_slice()
            .iter()
            .map(|i| usize::try_from(i.get()).unwrap())
            .product::<usize>();

        let shard_index =
            self.decode_index(&encoded_shard, chunks_per_shard.as_slice(), options)?;

        // Calc self/internal concurrent limits
        let (shard_concurrent_limit, concurrency_limit_inner_chunks) = calc_concurrency_outer_inner(
            options.concurrent_target(),
            &self.recommended_concurrency(shape, data_type)?,
            &self
                .inner_codecs
                .recommended_concurrency(&self.subchunk_shape, data_type)?,
        );
        let options = options.with_concurrent_target(concurrency_limit_inner_chunks);

        if data_type.is_optional() {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                Self::IDENTIFIER.to_string(),
            ));
        }

        let shard_shape_u64 = bytemuck::must_cast_slice(shape);
        match data_type.size() {
            DataTypeSize::Variable => {
                let decode_inner_chunk = |chunk_index: usize| {
                    let chunk_subset = self
                        .chunk_index_to_subset(chunk_index as u64, chunks_per_shard.as_slice())
                        .expect("inbounds chunk");

                    // Read the offset/size
                    let offset = shard_index[chunk_index * 2];
                    let size = shard_index[chunk_index * 2 + 1];
                    let chunk_bytes = if offset == u64::MAX && size == u64::MAX {
                        ArrayBytes::new_fill_value(
                            data_type,
                            self.subchunk_shape.num_elements_u64(),
                            fill_value,
                        )?
                    } else if usize::try_from(offset + size).unwrap() > encoded_shard.len() {
                        return Err(CodecError::Other(
                            "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                                .to_string(),
                        ));
                    } else {
                        let offset: usize = offset.try_into().unwrap();
                        let size: usize = size.try_into().unwrap();
                        let encoded_chunk = &encoded_shard[offset..offset + size];
                        self.inner_codecs.decode(
                            Cow::Borrowed(encoded_chunk),
                            &self.subchunk_shape,
                            data_type,
                            fill_value,
                            &options,
                        )?
                    };
                    Ok((chunk_bytes, chunk_subset))
                };

                // Decode the inner chunks
                let chunk_bytes_and_subsets = crate::iter_concurrent_limit!(
                    shard_concurrent_limit,
                    (0..num_chunks),
                    map,
                    decode_inner_chunk
                )
                .collect::<Result<Vec<_>, _>>()?;

                // Convert into an array
                merge_chunks_vlen(chunk_bytes_and_subsets, shard_shape_u64)
            }
            DataTypeSize::Fixed(data_type_size) => {
                // Allocate an array for the output
                let num_elements = shape.iter().map(|d| d.get()).product::<u64>();
                let size_output = usize::try_from(num_elements).unwrap() * data_type_size;
                if size_output == 0 {
                    return Ok(ArrayBytes::new_flen(vec![]));
                }
                let mut decoded_shard = Vec::<u8>::with_capacity(size_output);

                {
                    let output =
                        UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut decoded_shard);
                    let decode_chunk = |chunk_index: usize| {
                        let chunk_subset = self
                            .chunk_index_to_subset(chunk_index as u64, chunks_per_shard.as_slice())
                            .expect("inbounds chunk");
                        let mut output_view_inner_chunk = unsafe {
                            // SAFETY: chunks represent disjoint array subsets
                            ArrayBytesFixedDisjointView::new(
                                output,
                                data_type_size,
                                shard_shape_u64,
                                chunk_subset,
                            )?
                        };

                        // Read the offset/size
                        let offset = shard_index[chunk_index * 2];
                        let size = shard_index[chunk_index * 2 + 1];
                        if offset == u64::MAX && size == u64::MAX {
                            output_view_inner_chunk.fill(fill_value.as_ne_bytes())?;
                        } else if usize::try_from(offset + size).unwrap() > encoded_shard.len() {
                            return Err(CodecError::Other(
                                "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                                    .to_string(),
                            ));
                        } else {
                            let offset: usize = offset.try_into().unwrap();
                            let size: usize = size.try_into().unwrap();
                            let encoded_chunk = &encoded_shard[offset..offset + size];
                            let decoded_chunk = self.inner_codecs.decode(
                                Cow::Borrowed(encoded_chunk),
                                &self.subchunk_shape,
                                data_type,
                                fill_value,
                                &options,
                            )?;
                            output_view_inner_chunk
                                .copy_from_slice(&decoded_chunk.into_fixed()?)?;
                        }

                        Ok::<_, CodecError>(())
                    };

                    crate::iter_concurrent_limit!(
                        shard_concurrent_limit,
                        (0..num_chunks),
                        try_for_each,
                        decode_chunk
                    )?;
                }
                unsafe { decoded_shard.set_len(decoded_shard.capacity()) };
                Ok(ArrayBytes::from(decoded_shard))
            }
        }
    }

    fn compact<'a>(
        &self,
        bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        _data_type: &DataType,
        _fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytesRaw<'a>>, CodecError> {
        // Calculate chunks per shard
        let chunks_per_shard = calculate_chunks_per_shard(shape, self.subchunk_shape.as_slice())?;

        // Decode the shard index
        let shard_index = self.decode_index(&bytes, chunks_per_shard.as_slice(), options)?;

        // Get index metadata
        let index_shape = sharding_index_shape(chunks_per_shard.as_slice());
        let index_encoded_size =
            compute_index_encoded_size(self.index_codecs.as_ref(), &index_shape)?;

        // Check if compaction is needed (no-op optimization)
        let mut needs_compaction = false;
        let mut chunks_size = 0;
        for &[offset, size] in shard_index.as_chunks::<2>().0 {
            if offset != u64::MAX && size != u64::MAX {
                chunks_size += size;
            }
        }
        if chunks_size != bytes.len() as u64 - index_encoded_size {
            needs_compaction = true;
        }

        if !needs_compaction {
            return Ok(None); // No compaction needed
        }

        // Calculate compact size
        let data_size: usize = shard_index
            .as_chunks::<2>()
            .0
            .iter()
            .filter(|chunk| chunk[0] != u64::MAX)
            .map(|chunk| usize::try_from(chunk[1]).unwrap())
            .sum();

        let compact_size = data_size + usize::try_from(index_encoded_size).unwrap();

        // Build compacted shard
        let mut compact_shard = vec![0u8; compact_size];
        let mut new_index = vec![u64::MAX; shard_index.len()];

        let mut write_offset = match self.index_location {
            ShardingIndexLocation::Start => usize::try_from(index_encoded_size).unwrap(),
            ShardingIndexLocation::End => 0,
        };

        for (i, &[old_offset, size]) in shard_index.as_chunks::<2>().0.iter().enumerate() {
            if old_offset != u64::MAX && size != u64::MAX {
                let old_offset_usize = usize::try_from(old_offset).unwrap();
                let size_usize = usize::try_from(size).unwrap();

                // Validate bounds
                if old_offset_usize + size_usize > bytes.len() {
                    return Err(CodecError::Other(
                        "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                            .to_string(),
                    ));
                }

                // Copy chunk data
                compact_shard[write_offset..write_offset + size_usize]
                    .copy_from_slice(&bytes[old_offset_usize..old_offset_usize + size_usize]);

                // Update index
                new_index[i * 2] = u64::try_from(write_offset).unwrap();
                new_index[i * 2 + 1] = size;

                write_offset += size_usize;
            }
        }

        // Encode and write index
        let index_bytes = transmute_to_bytes_vec(new_index);
        let encoded_index = self.index_codecs.encode(
            ArrayBytes::from(index_bytes),
            &index_shape,
            &crate::array::data_type::uint64(),
            &FillValue::from(u64::MAX),
            options,
        )?;

        match self.index_location {
            ShardingIndexLocation::Start => {
                compact_shard[..encoded_index.len()].copy_from_slice(&encoded_index);
            }
            ShardingIndexLocation::End => {
                let index_start = compact_size - encoded_index.len();
                compact_shard[index_start..].copy_from_slice(&encoded_index);
            }
        }

        Ok(Some(Cow::Owned(compact_shard)))
    }

    #[allow(clippy::too_many_lines)]
    fn decode_into(
        &self,
        encoded_shard: ArrayBytesRaw<'_>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        // Sharding currently only supports non-optional data
        let output_view = match output_target {
            ArrayBytesDecodeIntoTarget::Fixed(data) => data,
            ArrayBytesDecodeIntoTarget::Optional(..) => {
                return Err(CodecError::UnsupportedDataType(
                    data_type.clone(),
                    Self::IDENTIFIER.to_string(),
                ));
            }
        };
        let chunks_per_shard = calculate_chunks_per_shard(shape, &self.subchunk_shape)?;
        let num_chunks = chunks_per_shard
            .as_slice()
            .iter()
            .map(|i| usize::try_from(i.get()).unwrap())
            .product::<usize>();

        let shard_index =
            self.decode_index(&encoded_shard, chunks_per_shard.as_slice(), options)?;

        // Calc self/internal concurrent limits
        let (shard_concurrent_limit, concurrency_limit_inner_chunks) = calc_concurrency_outer_inner(
            options.concurrent_target(),
            &self.recommended_concurrency(shape, data_type)?,
            &self
                .inner_codecs
                .recommended_concurrency(&self.subchunk_shape, data_type)?,
        );
        let options = options.with_concurrent_target(concurrency_limit_inner_chunks);

        let decode_chunk = |chunk_index: usize| {
            let chunk_subset = self
                .chunk_index_to_subset(chunk_index as u64, chunks_per_shard.as_slice())
                .expect("inbounds chunk");

            let output_subset_chunk = ArraySubset::new_with_start_shape(
                std::iter::zip(output_view.subset().start(), chunk_subset.start())
                    .map(|(o, s)| o + s)
                    .collect(),
                chunk_subset.shape().to_vec(),
            )
            .unwrap();
            let mut output_view_inner_chunk = unsafe {
                // SAFETY: inner chunks represent disjoint array subsets
                output_view.subdivide(output_subset_chunk)?
            };

            // Read the offset/size
            let offset = shard_index[chunk_index * 2];
            let size = shard_index[chunk_index * 2 + 1];
            if offset == u64::MAX && size == u64::MAX {
                output_view_inner_chunk.fill(fill_value.as_ne_bytes())?;
            } else if usize::try_from(offset + size).unwrap() > encoded_shard.len() {
                return Err(CodecError::Other(
                    "The shard index references out-of-bounds bytes. The chunk may be corrupted."
                        .to_string(),
                ));
            } else {
                let offset: usize = offset.try_into().unwrap();
                let size: usize = size.try_into().unwrap();
                let encoded_chunk = &encoded_shard[offset..offset + size];
                self.inner_codecs.decode_into(
                    Cow::Borrowed(encoded_chunk),
                    &self.subchunk_shape,
                    data_type,
                    fill_value,
                    ArrayBytesDecodeIntoTarget::Fixed(&mut output_view_inner_chunk),
                    &options,
                )?;
            }

            Ok::<_, CodecError>(())
        };

        crate::iter_concurrent_limit!(
            shard_concurrent_limit,
            (0..num_chunks),
            try_for_each,
            decode_chunk
        )?;

        Ok(())
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(ShardingPartialDecoder::new(
            input_handle,
            data_type.clone(),
            fill_value.clone(),
            ChunkShape::from(shape.to_vec()),
            self.subchunk_shape.clone(),
            self.inner_codecs.clone(),
            &self.index_codecs,
            self.index_location,
            options,
        )?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            AsyncShardingPartialDecoder::new(
                input_handle,
                data_type.clone(),
                fill_value.clone(),
                ChunkShape::from(shape.to_vec()),
                self.subchunk_shape.clone(),
                self.inner_codecs.clone(),
                &self.index_codecs,
                self.index_location,
                options,
            )
            .await?,
        ))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(
            sharding_partial_encoder::ShardingPartialEncoder::new(
                input_output_handle,
                data_type.clone(),
                fill_value.clone(),
                ChunkShape::from(shape.to_vec()),
                self.subchunk_shape.clone(),
                self.inner_codecs.clone(),
                self.index_codecs.clone(),
                self.index_location,
                options,
            )?,
        ))
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
    ) -> Result<BytesRepresentation, CodecError> {
        // Get the maximum size of encoded chunks
        let chunk_bytes_representation = self
            .inner_codecs
            .encoded_representation(shape, data_type, fill_value)?;

        match chunk_bytes_representation {
            BytesRepresentation::BoundedSize(size) | BytesRepresentation::FixedSize(size) => {
                let chunks_per_shard =
                    calculate_chunks_per_shard(shape, self.subchunk_shape.as_slice())?;
                let index_encoded_size = compute_index_encoded_size(
                    self.index_codecs.as_ref(),
                    &sharding_index_shape(chunks_per_shard.as_slice()),
                )?;
                let shard_size = Self::encoded_shard_bounded_size(
                    index_encoded_size,
                    size,
                    chunks_per_shard.as_slice(),
                );
                Ok(BytesRepresentation::BoundedSize(shard_size))
            }
            BytesRepresentation::UnboundedSize => Ok(BytesRepresentation::UnboundedSize),
        }
    }
}

impl ShardingCodec {
    /// Convert a linearised chunk index to an array subset.
    /// Returns [`None`] if `chunk_index` is out-of-bounds of `chunks_per_shard`.
    fn chunk_index_to_subset(
        &self,
        chunk_index: u64,
        chunks_per_shard: &[NonZeroU64],
    ) -> Option<ArraySubset> {
        let chunks_per_shard = chunk_shape_to_array_shape(chunks_per_shard);
        let chunk_indices = unravel_index(chunk_index, chunks_per_shard.as_slice())?;
        let chunk_start = std::iter::zip(&chunk_indices, self.subchunk_shape.as_slice())
            .map(|(i, c)| i * c.get())
            .collect::<Vec<_>>();
        let shape = self.subchunk_shape.as_slice();
        let ranges = shape
            .iter()
            .zip(&chunk_start)
            .map(|(&sh, &st)| st..(st + sh.get()));
        Some(ArraySubset::from(ranges))
    }

    /// Computed the bounded size of an encoded shard from
    ///  - the chunk bytes representation, and
    ///  - the number of chunks per shard.
    ///
    /// Equal to `num chunks * max chunk size + index size`
    fn encoded_shard_bounded_size(
        index_encoded_size: u64,
        chunk_encoded_size: u64,
        chunks_per_shard: &[NonZeroU64],
    ) -> u64 {
        let num_chunks = chunks_per_shard.iter().map(|i| i.get()).product::<u64>();
        num_chunks * chunk_encoded_size + index_encoded_size
    }

    /// Preallocate shard, encode and write chunks (in parallel), then truncate shard
    #[allow(clippy::too_many_lines)]
    #[expect(clippy::too_many_arguments)]
    fn encode_bounded(
        &self,
        decoded_value: &ArrayBytes,
        data_type: &DataType,
        fill_value: &FillValue,
        shard_shape: &[NonZeroU64],
        subchunk_shape: &[NonZeroU64],
        chunk_size_bounded: u64,
        options: &CodecOptions,
    ) -> Result<Vec<u8>, CodecError> {
        decoded_value.validate(shard_shape.num_elements_u64(), data_type)?;

        // Calculate maximum possible shard size
        let chunks_per_shard = calculate_chunks_per_shard(shard_shape, subchunk_shape)?;
        let index_shape = sharding_index_shape(chunks_per_shard.as_slice());
        let index_encoded_size =
            compute_index_encoded_size(self.index_codecs.as_ref(), &index_shape)?;
        let shard_size_bounded = Self::encoded_shard_bounded_size(
            index_encoded_size,
            chunk_size_bounded,
            chunks_per_shard.as_slice(),
        );

        let shard_size_bounded = usize::try_from(shard_size_bounded).unwrap();
        let index_encoded_size = usize::try_from(index_encoded_size).unwrap();

        // Allocate an array for the shard
        let mut shard = Vec::with_capacity(shard_size_bounded);

        // Allocate the decoded shard index
        let mut shard_index = vec![u64::MAX; index_shape.num_elements_usize()];
        let encoded_shard_offset: AtomicUsize = match self.index_location {
            ShardingIndexLocation::Start => index_encoded_size.into(),
            ShardingIndexLocation::End => 0.into(),
        };

        // Calc self/internal concurrent limits
        let (shard_concurrent_limit, concurrency_limit_inner_chunks) = calc_concurrency_outer_inner(
            options.concurrent_target(),
            &self.recommended_concurrency(shard_shape, data_type)?,
            &self
                .inner_codecs
                .recommended_concurrency(subchunk_shape, data_type)?,
        );
        let options = options.with_concurrent_target(concurrency_limit_inner_chunks);

        // Encode the shards and update the shard index
        {
            let shard_slice = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut shard);
            let shard_index_slice = UnsafeCellSlice::new(&mut shard_index);
            let n_chunks = chunks_per_shard
                .as_slice()
                .iter()
                .map(|i| usize::try_from(i.get()).unwrap())
                .product::<usize>();
            crate::iter_concurrent_limit!(
                shard_concurrent_limit,
                (0..n_chunks),
                try_for_each,
                |chunk_index: usize| {
                    let chunk_subset = self
                        .chunk_index_to_subset(chunk_index as u64, chunks_per_shard.as_slice())
                        .expect("inbounds chunk");
                    let bytes = decoded_value.extract_array_subset(
                        &chunk_subset,
                        bytemuck::must_cast_slice(shard_shape),
                        data_type,
                    )?;
                    if !bytes.is_fill_value(fill_value) {
                        let chunk_encoded = self.inner_codecs.encode(
                            bytes,
                            subchunk_shape,
                            data_type,
                            fill_value,
                            &options,
                        )?;

                        let chunk_offset = encoded_shard_offset
                            .fetch_add(chunk_encoded.len(), std::sync::atomic::Ordering::Relaxed);
                        if chunk_offset + chunk_encoded.len() > shard_size_bounded {
                            // This is a dev error, indicates the codec bounded size is not correct
                            return Err(CodecError::from(
                                "Sharding did not allocate a large enough buffer",
                            ));
                        }

                        unsafe {
                            let shard_index_unsafe =
                                shard_index_slice.index_mut(chunk_index * 2..chunk_index * 2 + 2);
                            shard_index_unsafe[0] = u64::try_from(chunk_offset).unwrap();
                            shard_index_unsafe[1] = u64::try_from(chunk_encoded.len()).unwrap();

                            shard_slice
                                .index_mut(chunk_offset..chunk_offset + chunk_encoded.len())
                                .copy_from_slice(&chunk_encoded);
                        }
                    }
                    Ok(())
                }
            )?;
        }

        // Truncate shard
        let shard_length = encoded_shard_offset.load(std::sync::atomic::Ordering::Relaxed)
            + match self.index_location {
                ShardingIndexLocation::Start => 0,
                ShardingIndexLocation::End => index_encoded_size,
            };

        // Encode and write array index
        let shard_index_bytes: ArrayBytesRaw = transmute_to_bytes_vec(shard_index).into();
        let encoded_array_index = self.index_codecs.encode(
            shard_index_bytes.into(),
            &index_shape,
            &crate::array::data_type::uint64(),
            &FillValue::from(u64::MAX),
            &options,
        )?;
        {
            // SAFETY: `shard_slice` is not read from until it has been written
            let shard_slice = unsafe { crate::vec_spare_capacity_to_mut_slice(&mut shard) };
            match self.index_location {
                ShardingIndexLocation::Start => {
                    shard_slice[..encoded_array_index.len()].copy_from_slice(&encoded_array_index);
                }
                ShardingIndexLocation::End => {
                    shard_slice[shard_length - encoded_array_index.len()..shard_length]
                        .copy_from_slice(&encoded_array_index);
                }
            }
        }
        // SAFETY: all elements have been initialised
        unsafe { shard.set_len(shard_length) };
        Ok(shard)
    }

    /// Encode inner chunks (in parallel), then allocate shard, then write to shard (in parallel)
    // TODO: Collecting chunks then allocating shard can use a lot of memory, have a low memory variant
    // TODO: Also benchmark performance with just performing an alloc like 1x decoded size and writing directly into it, growing if needed
    #[allow(clippy::too_many_lines)]
    fn encode_unbounded(
        &self,
        decoded_value: &ArrayBytes,
        data_type: &DataType,
        fill_value: &FillValue,
        shard_shape: &[NonZeroU64],
        subchunk_shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Vec<u8>, CodecError> {
        decoded_value.validate(shard_shape.num_elements_u64(), data_type)?;

        let chunks_per_shard = calculate_chunks_per_shard(shard_shape, subchunk_shape)?;
        let index_shape = sharding_index_shape(chunks_per_shard.as_slice());
        let index_encoded_size =
            compute_index_encoded_size(self.index_codecs.as_ref(), &index_shape)?;
        let index_encoded_size = usize::try_from(index_encoded_size).unwrap();

        // Find chunks that are not entirely the fill value and collect their decoded bytes
        let n_chunks = chunks_per_shard
            .as_slice()
            .iter()
            .map(|i| usize::try_from(i.get()).unwrap())
            .product::<usize>();

        // Calc self/internal concurrent limits
        let (shard_concurrent_limit, concurrency_limit_inner_chunks) = calc_concurrency_outer_inner(
            options.concurrent_target(),
            &self.recommended_concurrency(shard_shape, data_type)?,
            &self
                .inner_codecs
                .recommended_concurrency(subchunk_shape, data_type)?,
        );
        let options_inner = options.with_concurrent_target(concurrency_limit_inner_chunks);

        let encode_chunk = |chunk_index| {
            let chunk_subset = self
                .chunk_index_to_subset(chunk_index as u64, chunks_per_shard.as_slice())
                .expect("inbounds chunk");

            let bytes = decoded_value.extract_array_subset(
                &chunk_subset,
                bytemuck::must_cast_slice(shard_shape),
                data_type,
            );
            let bytes = match bytes {
                Ok(bytes) => bytes,
                Err(err) => return Some(Err(err)),
            };

            let is_fill_value = bytes.is_fill_value(fill_value);
            if is_fill_value {
                None
            } else {
                let encoded_chunk = self.inner_codecs.encode(
                    bytes,
                    subchunk_shape,
                    data_type,
                    fill_value,
                    &options_inner,
                );
                match encoded_chunk {
                    Ok(encoded_chunk) => Some(Ok((chunk_index, encoded_chunk.to_vec()))),
                    Err(err) => Some(Err(err)),
                }
            }
        };

        #[cfg(not(target_arch = "wasm32"))]
        let iterator = (0..n_chunks).into_par_iter();
        #[cfg(target_arch = "wasm32")]
        let iterator = 0..n_chunks;

        let encoded_chunks: Vec<(usize, Vec<u8>)> = crate::iter_concurrent_limit!(
            shard_concurrent_limit,
            iterator,
            filter_map,
            encode_chunk
        )
        .collect::<Result<Vec<_>, _>>()?;

        // Allocate the shard
        let encoded_chunk_length = encoded_chunks
            .iter()
            .map(|(_, bytes)| bytes.len())
            .sum::<usize>();
        let shard_length = encoded_chunk_length + index_encoded_size;
        let mut shard = Vec::with_capacity(shard_length);

        // Allocate the decoded shard index
        let mut shard_index = vec![u64::MAX; index_shape.num_elements_usize()];
        let encoded_shard_offset: AtomicUsize = match self.index_location {
            ShardingIndexLocation::Start => index_encoded_size.into(),
            ShardingIndexLocation::End => 0.into(),
        };

        // Write shard and update shard index
        if !encoded_chunks.is_empty() {
            let shard_slice = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut shard);
            let shard_index_slice = UnsafeCellSlice::new(&mut shard_index);
            crate::iter_concurrent_limit!(
                options.concurrent_target(),
                encoded_chunks,
                for_each,
                |(chunk_index, chunk_encoded): (usize, Vec<u8>)| {
                    let chunk_offset = encoded_shard_offset
                        .fetch_add(chunk_encoded.len(), std::sync::atomic::Ordering::Relaxed);
                    unsafe {
                        let shard_index_unsafe =
                            shard_index_slice.index_mut(chunk_index * 2..chunk_index * 2 + 2);
                        shard_index_unsafe[0] = u64::try_from(chunk_offset).unwrap();
                        shard_index_unsafe[1] = u64::try_from(chunk_encoded.len()).unwrap();

                        shard_slice
                            .index_mut(chunk_offset..chunk_offset + chunk_encoded.len())
                            .copy_from_slice(&chunk_encoded);
                    }
                }
            );
        }

        // Write shard index
        let encoded_array_index = self.index_codecs.encode(
            ArrayBytes::from(transmute_to_bytes_vec(shard_index)),
            &index_shape,
            &crate::array::data_type::uint64(),
            &FillValue::from(u64::MAX),
            options,
        )?;
        {
            // SAFETY: `shard_slice` is not read from until it has been written
            let shard_slice = unsafe { crate::vec_spare_capacity_to_mut_slice(&mut shard) };
            match self.index_location {
                ShardingIndexLocation::Start => {
                    shard_slice[..encoded_array_index.len()].copy_from_slice(&encoded_array_index);
                }
                ShardingIndexLocation::End => {
                    shard_slice[shard_length - encoded_array_index.len()..]
                        .copy_from_slice(&encoded_array_index);
                }
            }
        }
        // SAFETY: all elements have been initialised
        unsafe { shard.set_len(shard_length) };
        Ok(shard)
    }

    fn decode_index(
        &self,
        encoded_shard: &[u8],
        chunks_per_shard: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Vec<u64>, CodecError> {
        // Get index array representation and encoded size
        let index_shape = sharding_index_shape(chunks_per_shard);
        let index_encoded_size =
            compute_index_encoded_size(self.index_codecs.as_ref(), &index_shape)?;

        // Get encoded shard index
        if (encoded_shard.len() as u64) < index_encoded_size {
            return Err(CodecError::Other(
                "The encoded shard is smaller than the expected size of its index.".to_string(),
            ));
        }

        let encoded_shard_index = match self.index_location {
            ShardingIndexLocation::Start => {
                &encoded_shard[..index_encoded_size.try_into().unwrap()]
            }
            ShardingIndexLocation::End => {
                let encoded_shard_offset =
                    usize::try_from(encoded_shard.len() as u64 - index_encoded_size).unwrap();
                &encoded_shard[encoded_shard_offset..]
            }
        };

        // Decode the shard index
        decode_shard_index(
            encoded_shard_index,
            &index_shape,
            self.index_codecs.as_ref(),
            options,
        )
    }
}
