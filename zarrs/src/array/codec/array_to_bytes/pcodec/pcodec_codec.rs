use std::{borrow::Cow, sync::Arc};

use pco::{ChunkConfig, DeltaSpec, ModeSpec, PagingSpec, standalone::guarantee::file_size};
use zarrs_plugin::PluginCreateError;

use super::{
    PcodecCodecConfiguration, PcodecCodecConfigurationV1, PcodecCompressionLevel,
    PcodecDeltaEncodingOrder,
};
use crate::array::{
    BytesRepresentation, ChunkShapeTraits, DataType, FillValue,
    codec::{
        ArrayBytes, ArrayBytesRaw, ArrayCodecTraits, ArrayToBytesCodecTraits, CodecError,
        CodecMetadataOptions, CodecOptions, CodecTraits, PartialDecoderCapability,
        PartialEncoderCapability, RecommendedConcurrency,
    },
    convert_from_bytes_slice, transmute_to_bytes_vec,
};
use crate::metadata::Configuration;
use crate::metadata_ext::codec::pcodec::{
    PcodecDeltaSpecConfiguration, PcodecModeSpecConfiguration, PcodecPagingSpecConfiguration,
};
use std::num::NonZeroU64;
use zarrs_data_type::PcodecElementType;
use zarrs_plugin::ExtensionIdentifier;

/// A `pcodec` codec implementation.
#[derive(Debug, Clone)]
pub struct PcodecCodec {
    chunk_config: ChunkConfig,
}

fn mode_spec_config_to_pco(mode_spec: PcodecModeSpecConfiguration) -> ModeSpec {
    match mode_spec {
        PcodecModeSpecConfiguration::Auto => ModeSpec::Auto,
        PcodecModeSpecConfiguration::Classic => ModeSpec::Classic,
    }
}

fn mode_spec_pco_to_config(mode_spec: &ModeSpec) -> PcodecModeSpecConfiguration {
    #[allow(clippy::wildcard_enum_match_arm)]
    match mode_spec {
        ModeSpec::Auto => PcodecModeSpecConfiguration::Auto,
        ModeSpec::Classic => PcodecModeSpecConfiguration::Classic,
        _ => unreachable!("Mode spec is not supported"),
    }
}

fn configuration_to_chunk_config(configuration: &PcodecCodecConfigurationV1) -> ChunkConfig {
    let mode_spec = mode_spec_config_to_pco(configuration.mode_spec);
    let delta_spec = match configuration.delta_spec {
        PcodecDeltaSpecConfiguration::Auto => DeltaSpec::Auto,
        PcodecDeltaSpecConfiguration::None => DeltaSpec::None,
        PcodecDeltaSpecConfiguration::TryConsecutive => DeltaSpec::TryConsecutive(
            configuration
                .delta_encoding_order
                .map_or(0, |o| o.as_usize()),
        ),
        PcodecDeltaSpecConfiguration::TryLookback => DeltaSpec::TryLookback,
    };
    let paging_spec = match configuration.paging_spec {
        PcodecPagingSpecConfiguration::EqualPagesUpTo => {
            PagingSpec::EqualPagesUpTo(configuration.equal_pages_up_to)
        }
    };
    ChunkConfig::default()
        .with_compression_level(configuration.level.as_usize())
        .with_mode_spec(mode_spec)
        .with_delta_spec(delta_spec)
        .with_paging_spec(paging_spec)
}

impl PcodecCodec {
    /// Create a new `pcodec` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &PcodecCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            PcodecCodecConfiguration::V1(configuration) => {
                let chunk_config = configuration_to_chunk_config(configuration);
                Ok(Self { chunk_config })
            }
            _ => Err(PluginCreateError::Other(
                "this pcodec codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for PcodecCodec {
    fn identifier(&self) -> &'static str {
        Self::IDENTIFIER
    }

    fn configuration(&self, _name: &str, _options: &CodecMetadataOptions) -> Option<Configuration> {
        let mode_spec = mode_spec_pco_to_config(&self.chunk_config.mode_spec);
        let (delta_spec, delta_encoding_order) = match self.chunk_config.delta_spec {
            DeltaSpec::Auto => (PcodecDeltaSpecConfiguration::Auto, None),
            DeltaSpec::None => (PcodecDeltaSpecConfiguration::None, None),
            DeltaSpec::TryConsecutive(delta_encoding_order) => (
                PcodecDeltaSpecConfiguration::TryConsecutive,
                Some(PcodecDeltaEncodingOrder::try_from(delta_encoding_order).expect("valid")),
            ),
            DeltaSpec::TryLookback => (PcodecDeltaSpecConfiguration::TryLookback, None),
            _ => unimplemented!("unsupported pcodec delta spec"),
        };
        let (paging_spec, equal_pages_up_to) = match self.chunk_config.paging_spec {
            PagingSpec::EqualPagesUpTo(equal_pages_up_to) => (
                PcodecPagingSpecConfiguration::EqualPagesUpTo,
                equal_pages_up_to,
            ),
            PagingSpec::Exact(_) => unimplemented!("pcodec exact paging spec not supported"),
            _ => unimplemented!("unsupported pcodec paging spec"),
        };

        let configuration = PcodecCodecConfiguration::V1(PcodecCodecConfigurationV1 {
            level: PcodecCompressionLevel::try_from(self.chunk_config.compression_level)
                .expect("validated on creation"),
            mode_spec,
            delta_spec,
            paging_spec,
            delta_encoding_order,
            equal_pages_up_to,
        });

        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl ArrayCodecTraits for PcodecCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // pcodec does not support parallel decode
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for PcodecCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        // Use codec_pcodec() from DataTypeExtension trait to get element type
        let pcodec = data_type.codec_pcodec().ok_or_else(|| {
            CodecError::UnsupportedDataType(data_type.clone(), Self::IDENTIFIER.to_string())
        })?;
        let element_type = pcodec.pcodec_element_type();

        let bytes = bytes.into_fixed()?;
        macro_rules! pcodec_encode {
            ( $t:ty ) => {
                pco::standalone::simple_compress(
                    &convert_from_bytes_slice::<$t>(&bytes),
                    &self.chunk_config,
                )
                .map(Cow::Owned)
                .map_err(|err| CodecError::Other(err.to_string()))
            };
        }

        match element_type {
            PcodecElementType::U16 => pcodec_encode!(u16),
            PcodecElementType::U32 => pcodec_encode!(u32),
            PcodecElementType::U64 => pcodec_encode!(u64),
            PcodecElementType::I16 => pcodec_encode!(i16),
            PcodecElementType::I32 => pcodec_encode!(i32),
            PcodecElementType::I64 => pcodec_encode!(i64),
            PcodecElementType::F16 => pcodec_encode!(half::f16),
            PcodecElementType::F32 => pcodec_encode!(f32),
            PcodecElementType::F64 => pcodec_encode!(f64),
            _ => Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                Self::IDENTIFIER.to_string(),
            )),
        }
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytesRaw<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        // Use codec_pcodec() from DataTypeExtension trait to get element type
        let pcodec = data_type.codec_pcodec().ok_or_else(|| {
            CodecError::UnsupportedDataType(data_type.clone(), Self::IDENTIFIER.to_string())
        })?;
        let element_type = pcodec.pcodec_element_type();

        macro_rules! pcodec_decode {
            ( $t:ty ) => {
                pco::standalone::simple_decompress(&bytes)
                    .map(|bytes| Cow::Owned(transmute_to_bytes_vec::<$t>(bytes)))
                    .map_err(|err| CodecError::Other(err.to_string()))
            };
        }

        let bytes = match element_type {
            PcodecElementType::U16 => pcodec_decode!(u16),
            PcodecElementType::U32 => pcodec_decode!(u32),
            PcodecElementType::U64 => pcodec_decode!(u64),
            PcodecElementType::I16 => pcodec_decode!(i16),
            PcodecElementType::I32 => pcodec_decode!(i32),
            PcodecElementType::I64 => pcodec_decode!(i64),
            PcodecElementType::F16 => pcodec_decode!(half::f16),
            PcodecElementType::F32 => pcodec_decode!(f32),
            PcodecElementType::F64 => pcodec_decode!(f64),
            _ => {
                return Err(CodecError::UnsupportedDataType(
                    data_type.clone(),
                    Self::IDENTIFIER.to_string(),
                ));
            }
        }?;
        Ok(ArrayBytes::from(bytes))
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
    ) -> Result<BytesRepresentation, CodecError> {
        // Use codec_pcodec() from DataTypeExtension trait to get element type info
        let pcodec = data_type.codec_pcodec().ok_or_else(|| {
            CodecError::UnsupportedDataType(data_type.clone(), Self::IDENTIFIER.to_string())
        })?;
        let element_type = pcodec.pcodec_element_type();

        let num_elements = shape.num_elements_usize() * pcodec.pcodec_elements_per_element();

        let size = match element_type {
            PcodecElementType::U16 | PcodecElementType::I16 | PcodecElementType::F16 => {
                file_size::<u16>(num_elements, &self.chunk_config.paging_spec)
                    .map_err(|err| CodecError::from(err.to_string()))?
            }
            PcodecElementType::U32 | PcodecElementType::I32 | PcodecElementType::F32 => {
                file_size::<u32>(num_elements, &self.chunk_config.paging_spec)
                    .map_err(|err| CodecError::from(err.to_string()))?
            }
            PcodecElementType::U64 | PcodecElementType::I64 | PcodecElementType::F64 => {
                file_size::<u64>(num_elements, &self.chunk_config.paging_spec)
                    .map_err(|err| CodecError::from(err.to_string()))?
            }
            _ => {
                return Err(CodecError::UnsupportedDataType(
                    data_type.clone(),
                    Self::IDENTIFIER.to_string(),
                ));
            }
        };
        Ok(BytesRepresentation::BoundedSize(
            u64::try_from(size).unwrap(),
        ))
    }
}
