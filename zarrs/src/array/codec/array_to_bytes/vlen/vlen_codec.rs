use std::{num::NonZeroU64, sync::Arc};

use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::vlen::{VlenIndexDataType, VlenIndexLocation};

use crate::{
    array::{
        codec::{
            ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, BytesCodec,
            BytesPartialDecoderTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
            PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
        },
        transmute_to_bytes_vec, ArrayBytes, ArrayBytesRaw, BytesRepresentation,
        ChunkRepresentation, CodecChain, DataType, DataTypeSize, Endianness, RawBytesOffsets,
    },
    plugin::PluginCreateError,
};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};

use super::{vlen_partial_decoder, VlenCodecConfiguration, VlenCodecConfigurationV0_1};

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
        let index_codecs = Arc::new(CodecChain::new_named(
            vec![],
            Arc::new(BytesCodec::new(Some(Endianness::Little))).into(),
            vec![],
        ));
        let data_codecs = Arc::new(CodecChain::new_named(
            vec![],
            Arc::new(BytesCodec::new(None)).into(),
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
    fn identifier(&self) -> &str {
        zarrs_registry::codec::VLEN
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = VlenCodecConfiguration::V0_1(VlenCodecConfigurationV0_1 {
            index_codecs: self.index_codecs.create_metadatas(),
            data_codecs: self.data_codecs.create_metadatas(),
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
impl ArrayToBytesCodecTraits for VlenCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        bytes.validate(
            decoded_representation.num_elements(),
            decoded_representation.data_type(),
        )?;
        let (data, offsets) = bytes.into_variable()?;
        assert_eq!(
            offsets.len(),
            decoded_representation.num_elements_usize() + 1
        );

        // Encode offsets
        let num_offsets =
            NonZeroU64::try_from(decoded_representation.num_elements_usize() as u64 + 1).unwrap();
        let offsets = match self.index_data_type {
            // VlenIndexDataType::UInt8 => {
            //     let offsets = offsets
            //         .iter()
            //         .map(|offset| u8::try_from(*offset))
            //         .collect::<Result<Vec<_>, _>>()
            //         .map_err(|_| {
            //             CodecError::Other(
            //                 "index offsets are too large for a uint8 index_data_type".to_string(),
            //             )
            //         })?;
            //     let offsets = transmute_to_bytes_vec(offsets);
            //     let index_chunk_rep = ChunkRepresentation::new(
            //         vec![num_offsets],
            //         DataType::UInt8,
            //         0u8,
            //     )
            //     .unwrap();
            //     self.index_codecs
            //         .encode(offsets.into(), &index_chunk_rep, options)?
            // }
            // VlenIndexDataType::UInt16 => {
            //     let offsets = offsets
            //         .iter()
            //         .map(|offset| u16::try_from(*offset))
            //         .collect::<Result<Vec<_>, _>>()
            //         .map_err(|_| {
            //             CodecError::Other(
            //                 "index offsets are too large for a uint16 index_data_type".to_string(),
            //             )
            //         })?;
            //     let offsets = transmute_to_bytes_vec(offsets);
            //     let index_chunk_rep = ChunkRepresentation::new(
            //         vec![num_offsets],
            //         DataType::UInt16,
            //         0u16,
            //     )
            //     .unwrap();
            //     self.index_codecs
            //         .encode(offsets.into(), &index_chunk_rep, options)?
            // }
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
                let index_chunk_rep =
                    ChunkRepresentation::new(vec![num_offsets], DataType::UInt32, 0u32).unwrap();
                self.index_codecs
                    .encode(offsets.into(), &index_chunk_rep, options)?
            }
            VlenIndexDataType::UInt64 => {
                let offsets = offsets
                    .iter()
                    .map(|offset| u64::try_from(*offset).unwrap())
                    .collect::<Vec<u64>>();
                let offsets = transmute_to_bytes_vec(offsets);
                let index_chunk_rep =
                    ChunkRepresentation::new(vec![num_offsets], DataType::UInt64, 0u64).unwrap();
                self.index_codecs
                    .encode(offsets.into(), &index_chunk_rep, options)?
            }
        };

        // Encode data
        let data = if let Ok(data_len) = NonZeroU64::try_from(data.len() as u64) {
            self.data_codecs.encode(
                data.into(),
                &ChunkRepresentation::new(vec![data_len], DataType::UInt8, 0u8).unwrap(),
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
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let num_elements = decoded_representation.num_elements_usize();
        let index_shape = vec![NonZeroU64::try_from(num_elements as u64 + 1).unwrap()];
        let index_chunk_rep = match self.index_data_type {
            // VlenIndexDataType::UInt8 => {
            //     ChunkRepresentation::new(index_shape, DataType::UInt8, 0u8)
            // }
            // VlenIndexDataType::UInt16 => {
            //     ChunkRepresentation::new(index_shape, DataType::UInt16, 0u16)
            // }
            VlenIndexDataType::UInt32 => {
                ChunkRepresentation::new(index_shape, DataType::UInt32, 0u32)
            }
            VlenIndexDataType::UInt64 => {
                ChunkRepresentation::new(index_shape, DataType::UInt64, 0u64)
            }
        }
        .unwrap();
        let (bytes, offsets) = super::get_vlen_bytes_and_offsets(
            &index_chunk_rep,
            &bytes,
            &self.index_codecs,
            &self.data_codecs,
            self.index_location,
            options,
        )?;
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
        Ok(Arc::new(vlen_partial_decoder::VlenPartialDecoder::new(
            input_handle,
            decoded_representation.clone(),
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
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            vlen_partial_decoder::AsyncVlenPartialDecoder::new(
                input_handle,
                decoded_representation.clone(),
                self.index_codecs.clone(),
                self.data_codecs.clone(),
                self.index_data_type,
                self.index_location,
            ),
        ))
    }

    fn encoded_representation(
        &self,
        decoded_representation: &ChunkRepresentation,
    ) -> Result<BytesRepresentation, CodecError> {
        if decoded_representation.data_type().is_optional() {
            return Err(CodecError::UnsupportedDataType(
                decoded_representation.data_type().clone(),
                zarrs_registry::codec::VLEN.to_string(),
            ));
        }

        match decoded_representation.data_type().size() {
            DataTypeSize::Variable => Ok(BytesRepresentation::UnboundedSize),
            DataTypeSize::Fixed(_) => Err(CodecError::UnsupportedDataType(
                decoded_representation.data_type().clone(),
                zarrs_registry::codec::VLEN.to_string(),
            )),
        }
    }
}
