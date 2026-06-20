use std::num::NonZeroU64;
use std::sync::Arc;

use itertools::Itertools;

use crate::array::{
    ArrayBytes, ArrayBytesOffsets, ArrayBytesRaw, BytesRepresentation, DataType, DataTypeSize,
    FillValue,
};
use zarrs_codec::{
    ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits,
    BytesPartialDecoderTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    UnboundArrayToBytesCodecTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};
use zarrs_metadata::Configuration;
use zarrs_plugin::{ExtensionAliasesV3, ZarrVersion};

/// The `vlen_v2` codec implementation.
#[derive(Debug, Clone, Default)]
pub struct VlenV2Codec {}

/// The `vlen_v2` codec implementation bound to a data type and fill value.
#[derive(Debug, Clone)]
struct VlenV2CodecBound {
    data_type: DataType,
    fill_value: FillValue,
}

impl VlenV2Codec {
    /// Create a new `vlen_v2` codec.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl CodecTraits for VlenV2Codec {
    fn configuration(
        &self,
        _version: ZarrVersion,
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

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl UnboundArrayToBytesCodecTraits for VlenV2Codec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self as Arc<dyn UnboundArrayToBytesCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecError> {
        if data_type.is_optional() || !matches!(data_type.size(), DataTypeSize::Variable) {
            return Err(CodecError::UnsupportedDataType(
                data_type,
                Self::aliases_v3().default_name.to_string(),
            ));
        }
        Ok(Arc::new(VlenV2CodecBound {
            data_type,
            fill_value,
        }))
    }
}

impl ArrayCodecTraits for VlenV2CodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for VlenV2CodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        let num_elements = shape.iter().map(|d| d.get()).product::<u64>();
        bytes.validate(num_elements, &self.data_type)?;
        let (bytes, offsets) = bytes.into_variable()?.into_parts();

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
        bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let num_elements = shape.iter().map(|d| d.get()).product::<u64>();
        let num_elements_usize = usize::try_from(num_elements).unwrap();
        let (bytes, offsets) =
            super::get_interleaved_bytes_and_offsets(num_elements_usize, &bytes)?;
        let offsets = ArrayBytesOffsets::new(offsets)?;
        let array_bytes = ArrayBytes::new_vlen(bytes, offsets)?;
        Ok(array_bytes)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            super::vlen_v2_partial_decoder::VlenV2PartialDecoder::new(
                input_handle,
                shape.to_vec(),
                self.data_type.clone(),
                self.fill_value.clone(),
            ),
        ))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            super::vlen_v2_partial_decoder::AsyncVlenV2PartialDecoder::new(
                input_handle,
                shape.to_vec(),
                self.data_type.clone(),
                self.fill_value.clone(),
            ),
        ))
    }

    fn encoded_representation(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        match self.data_type.size() {
            DataTypeSize::Variable => Ok(BytesRepresentation::UnboundedSize),
            DataTypeSize::Fixed(_) => Err(CodecError::UnsupportedDataType(
                self.data_type.clone(),
                VlenV2Codec::aliases_v3().default_name.to_string(),
            )),
        }
    }
}
