use std::num::NonZeroU64;
use std::sync::Arc;

use super::{VlenCodecConfiguration, VlenCodecConfigurationV0_1, vlen_partial_decoder};
use crate::array::codec::BytesCodec;
use crate::array::{
    ArrayBytes, ArrayBytesOffsets, ArrayBytesRaw, BytesRepresentation, CodecChain, DataType,
    DataTypeSize, Endianness, FillValue, transmute_to_bytes_vec,
};
use crate::metadata::Configuration;
use crate::metadata_ext::codec::vlen::{VlenIndexDataType, VlenIndexLocation};
use crate::plugin::PluginCreateError;
use zarrs_codec::{
    ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits,
    BytesPartialDecoderTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};
use zarrs_plugin::{ExtensionAliasesV3, ZarrVersion};

/// A `vlen` codec implementation.
#[derive(Debug, Clone)]
pub struct VlenCodec {
    index_codecs: Arc<CodecChain>,
    data_codecs: Arc<CodecChain>,
    index_data_type: VlenIndexDataType,
    index_location: VlenIndexLocation,
}

impl Default for VlenCodec {
    fn default() -> Self {
        let index_codecs = Arc::new(CodecChain::new(
            vec![],
            Arc::new(BytesCodec::new(Some(Endianness::Little))),
            vec![],
        ));
        let data_codecs = Arc::new(CodecChain::new(
            vec![],
            Arc::new(BytesCodec::new(None)),
            vec![],
        ));
        Self {
            index_codecs,
            data_codecs,
            index_data_type: VlenIndexDataType::UInt64,
            index_location: VlenIndexLocation::Start,
        }
    }
}

impl VlenCodec {
    /// Create a new `vlen` codec.
    #[must_use]
    pub fn new(
        index_codecs: Arc<CodecChain>,
        data_codecs: Arc<CodecChain>,
        index_data_type: VlenIndexDataType,
        index_location: VlenIndexLocation,
    ) -> Self {
        Self {
            index_codecs,
            data_codecs,
            index_data_type,
            index_location,
        }
    }

    /// Set the index location.
    #[must_use]
    pub fn with_index_location(mut self, index_location: VlenIndexLocation) -> Self {
        self.index_location = index_location;
        self
    }

    /// Create a new `vlen` codec from configuration.
    ///
    /// # Errors
    /// Returns a [`PluginCreateError`] if the codecs cannot be constructed from the codec metadata.
    pub fn new_with_configuration(
        configuration: &VlenCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            VlenCodecConfiguration::V0_1(configuration) => {
                let index_codecs =
                    Arc::new(CodecChain::from_metadata(&configuration.index_codecs)?);
                let data_codecs = Arc::new(CodecChain::from_metadata(&configuration.data_codecs)?);
                Ok(Self::new(
                    index_codecs,
                    data_codecs,
                    configuration.index_data_type,
                    configuration.index_location,
                ))
            }
            VlenCodecConfiguration::V0(configuration) => {
                let index_codecs =
                    Arc::new(CodecChain::from_metadata(&configuration.index_codecs)?);
                let data_codecs = Arc::new(CodecChain::from_metadata(&configuration.data_codecs)?);
                Ok(Self::new(
                    index_codecs,
                    data_codecs,
                    configuration.index_data_type,
                    VlenIndexLocation::Start,
                ))
            }
            _ => Err(PluginCreateError::Other(
                "this vlen codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for VlenCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = VlenCodecConfiguration::V0_1(VlenCodecConfigurationV0_1 {
            index_codecs: self.index_codecs.create_metadatas(options),
            data_codecs: self.data_codecs.create_metadatas(options),
            index_data_type: self.index_data_type,
            index_location: self.index_location,
        });
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false, // TODO: could read offsets first, ideally cached, then grab values as needed
            partial_decode: false, // TODO
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl ArrayCodecTraits for VlenCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for VlenCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        let num_elements = shape.iter().map(|d| d.get()).product::<u64>();
        bytes.validate(num_elements, data_type)?;
        let (data, offsets) = bytes.into_variable()?.into_parts();
        assert_eq!(offsets.len(), usize::try_from(num_elements).unwrap() + 1);

        // Encode offsets
        let num_offsets =
            NonZeroU64::try_from(usize::try_from(num_elements).unwrap() as u64 + 1).unwrap();
        let offsets = match self.index_data_type {
            VlenIndexDataType::UInt32 => {
                let offsets = offsets
                    .iter()
                    .map(|offset| u32::try_from(*offset))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| {
                        CodecError::Other(
                            "index offsets are too large for a uint32 index_data_type".to_string(),
                        )
                    })?;
                let offsets = transmute_to_bytes_vec(offsets);
                let index_shape = vec![num_offsets];
                self.index_codecs.encode(
                    offsets.into(),
                    &index_shape,
                    &crate::array::data_type::uint32(),
                    &FillValue::from(0u32),
                    options,
                )?
            }
            VlenIndexDataType::UInt64 => {
                let offsets = offsets
                    .iter()
                    .map(|offset| u64::try_from(*offset).unwrap())
                    .collect::<Vec<u64>>();
                let offsets = transmute_to_bytes_vec(offsets);
                let index_shape = vec![num_offsets];
                self.index_codecs.encode(
                    offsets.into(),
                    &index_shape,
                    &crate::array::data_type::uint64(),
                    &FillValue::from(0u64),
                    options,
                )?
            }
        };

        // Encode data
        let data = if let Ok(data_len) = NonZeroU64::try_from(data.len() as u64) {
            let data_shape = vec![data_len];
            self.data_codecs.encode(
                data.into(),
                &data_shape,
                &crate::array::data_type::uint8(),
                &FillValue::from(0u8),
                options,
            )?
        } else {
            vec![].into()
        };

        // Pack encoded offsets length, encoded offsets, and encoded data
        let mut bytes = Vec::with_capacity(size_of::<u64>() + offsets.len() + data.len());

        match self.index_location {
            VlenIndexLocation::Start => {
                bytes.extend_from_slice(&u64::try_from(offsets.len()).unwrap().to_le_bytes()); // offsets length as u64 little endian
                bytes.extend_from_slice(&offsets);
                bytes.extend_from_slice(&data);
            }
            VlenIndexLocation::End => {
                bytes.extend_from_slice(&data);
                bytes.extend_from_slice(&offsets);
                bytes.extend_from_slice(&u64::try_from(offsets.len()).unwrap().to_le_bytes());
            }
        }

        Ok(bytes.into())
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        _data_type: &DataType,
        _fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let (bytes, offsets) = super::get_vlen_bytes_and_offsets(
            &bytes,
            shape,
            self.index_data_type,
            &self.index_codecs,
            &self.data_codecs,
            self.index_location,
            options,
        )?;
        let offsets = ArrayBytesOffsets::new(offsets)?;
        let array_bytes = ArrayBytes::new_vlen(bytes, offsets)?;
        Ok(array_bytes)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(vlen_partial_decoder::VlenPartialDecoder::new(
            input_handle,
            shape.to_vec(),
            data_type.clone(),
            fill_value.clone(),
            self.index_codecs.clone(),
            self.data_codecs.clone(),
            self.index_data_type,
            self.index_location,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            vlen_partial_decoder::AsyncVlenPartialDecoder::new(
                input_handle,
                shape.to_vec(),
                data_type.clone(),
                fill_value.clone(),
                self.index_codecs.clone(),
                self.data_codecs.clone(),
                self.index_data_type,
                self.index_location,
            ),
        ))
    }

    fn encoded_representation(
        &self,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
    ) -> Result<BytesRepresentation, CodecError> {
        if data_type.is_optional() {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                Self::aliases_v3().default_name.to_string(),
            ));
        }

        match data_type.size() {
            DataTypeSize::Variable => Ok(BytesRepresentation::UnboundedSize),
            DataTypeSize::Fixed(_) => Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                Self::aliases_v3().default_name.to_string(),
            )),
        }
    }
}
