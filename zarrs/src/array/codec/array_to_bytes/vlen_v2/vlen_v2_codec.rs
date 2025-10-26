use std::sync::Arc;

use itertools::Itertools;
use zarrs_metadata::Configuration;

use crate::array::{
    codec::{
        ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits,
        BytesPartialDecoderTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
        PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    },
    ArrayBytes, BytesRepresentation, ChunkRepresentation, DataTypeSize, RawBytes, RawBytesOffsets,
};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};

/// The `vlen_v2` codec implementation.
#[derive(Debug, Clone, Default)]
pub struct VlenV2Codec {}

impl VlenV2Codec {
    /// Create a new `vlen_v2` codec.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl CodecTraits for VlenV2Codec {
    fn identifier(&self) -> &str {
        zarrs_registry::codec::VLEN_V2
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(Configuration::default())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false, // NOTE: It is effectively a full decode when separating offsets/bytes
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl ArrayCodecTraits for VlenV2Codec {
    fn recommended_concurrency(
        &self,
        _decoded_representation: &ChunkRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for VlenV2Codec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        bytes.validate(
            decoded_representation.num_elements(),
            decoded_representation.data_type(),
        )?;
        let (bytes, offsets) = bytes.into_variable()?;

        let num_elements = decoded_representation.num_elements();
        debug_assert_eq!(1 + num_elements, offsets.len() as u64);

        let mut data: Vec<u8> = Vec::with_capacity(offsets.len() * size_of::<u32>() + bytes.len());
        // Number of elements
        let num_elements = u32::try_from(num_elements).map_err(|_| {
            CodecError::Other("num_elements exceeds u32::MAX in vlen codec".to_string())
        })?;
        data.extend_from_slice(num_elements.to_le_bytes().as_slice());
        // Interleaved length (u32, little endian) and element bytes
        for (&curr, &next) in offsets.iter().tuple_windows() {
            let element_bytes = &bytes[curr..next];
            let element_bytes_len = u32::try_from(element_bytes.len()).unwrap();
            data.extend_from_slice(&element_bytes_len.to_le_bytes());
            data.extend_from_slice(element_bytes);
        }

        Ok(data.into())
    }

    fn decode<'a>(
        &self,
        bytes: RawBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let num_elements = decoded_representation.num_elements_usize();
        let (bytes, offsets) = super::get_interleaved_bytes_and_offsets(num_elements, &bytes)?;
        let offsets = RawBytesOffsets::new(offsets)?;
        let array_bytes = ArrayBytes::new_vlen(bytes, offsets)?;
        Ok(array_bytes)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            super::vlen_v2_partial_decoder::VlenV2PartialDecoder::new(
                input_handle,
                decoded_representation.clone(),
            ),
        ))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            super::vlen_v2_partial_decoder::AsyncVlenV2PartialDecoder::new(
                input_handle,
                decoded_representation.clone(),
            ),
        ))
    }

    fn encoded_representation(
        &self,
        decoded_representation: &ChunkRepresentation,
    ) -> Result<BytesRepresentation, CodecError> {
        match decoded_representation.data_type().size() {
            DataTypeSize::Variable => Ok(BytesRepresentation::UnboundedSize),
            DataTypeSize::Fixed(_) => Err(CodecError::UnsupportedDataType(
                decoded_representation.data_type().clone(),
                zarrs_registry::codec::VLEN_V2.to_string(),
            )),
        }
    }
}
