use std::{borrow::Cow, sync::Arc};

use pco::{standalone::guarantee::file_size, ChunkConfig, DeltaSpec, ModeSpec, PagingSpec};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::pcodec::{
    PcodecDeltaSpecConfiguration, PcodecModeSpecConfiguration, PcodecPagingSpecConfiguration,
};
use zarrs_plugin::PluginCreateError;
use zarrs_registry::codec::PCODEC;

use crate::array::{
    codec::{
        ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits,
        BytesPartialDecoderTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
        RawBytes, RecommendedConcurrency,
    },
    convert_from_bytes_slice, transmute_to_bytes_vec, BytesRepresentation, ChunkRepresentation,
    DataType,
};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};

use super::{
    pcodec_partial_decoder, PcodecCodecConfiguration, PcodecCodecConfigurationV1,
    PcodecCompressionLevel, PcodecDeltaEncodingOrder,
};

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
    fn identifier(&self) -> &str {
        PCODEC
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
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

    fn partial_decoder_should_cache_input(&self) -> bool {
        false
    }

    fn partial_decoder_decodes_all(&self) -> bool {
        true
    }
}

impl ArrayCodecTraits for PcodecCodec {
    fn recommended_concurrency(
        &self,
        _decoded_representation: &ChunkRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // pcodec does not support parallel decode
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(all(feature = "async", not(target_arch = "wasm32")), async_trait::async_trait)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for PcodecCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        let data_type = decoded_representation.data_type();
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

        match data_type {
            DataType::UInt16 => {
                pcodec_encode!(u16)
            }
            DataType::UInt32 => {
                pcodec_encode!(u32)
            }
            DataType::UInt64 => {
                pcodec_encode!(u64)
            }
            DataType::Int16 => {
                pcodec_encode!(i16)
            }
            DataType::Int32 => {
                pcodec_encode!(i32)
            }
            DataType::Int64
            | DataType::NumpyDateTime64 {
                unit: _,
                scale_factor: _,
            }
            | DataType::NumpyTimeDelta64 {
                unit: _,
                scale_factor: _,
            } => {
                pcodec_encode!(i64)
            }
            DataType::Float16 | DataType::ComplexFloat16 => {
                pcodec_encode!(half::f16)
            }
            DataType::Float32 | DataType::Complex64 | DataType::ComplexFloat32 => {
                pcodec_encode!(f32)
            }
            DataType::Float64 | DataType::Complex128 | DataType::ComplexFloat64 => {
                pcodec_encode!(f64)
            }
            super::unsupported_dtypes!() => Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                PCODEC.to_string(),
            )),
        }
    }

    fn decode<'a>(
        &self,
        bytes: RawBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let data_type = decoded_representation.data_type();
        macro_rules! pcodec_decode {
            ( $t:ty ) => {
                pco::standalone::simple_decompress(&bytes)
                    .map(|bytes| Cow::Owned(transmute_to_bytes_vec::<$t>(bytes)))
                    .map_err(|err| CodecError::Other(err.to_string()))
            };
        }

        let bytes = match data_type {
            DataType::UInt16 => {
                pcodec_decode!(u16)
            }
            DataType::UInt32 => {
                pcodec_decode!(u32)
            }
            DataType::UInt64 => {
                pcodec_decode!(u64)
            }
            DataType::Int16 => {
                pcodec_decode!(i16)
            }
            DataType::Int32 => {
                pcodec_decode!(i32)
            }
            DataType::Int64
            | DataType::NumpyDateTime64 {
                unit: _,
                scale_factor: _,
            }
            | DataType::NumpyTimeDelta64 {
                unit: _,
                scale_factor: _,
            } => {
                pcodec_decode!(i64)
            }
            DataType::Float16 | DataType::ComplexFloat16 => {
                pcodec_decode!(half::f16)
            }
            DataType::Float32 | DataType::Complex64 | DataType::ComplexFloat32 => {
                pcodec_decode!(f32)
            }
            DataType::Float64 | DataType::Complex128 | DataType::ComplexFloat64 => {
                pcodec_decode!(f64)
            }
            super::unsupported_dtypes!() => Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                PCODEC.to_string(),
            )),
        }?;
        Ok(ArrayBytes::from(bytes))
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(pcodec_partial_decoder::PcodecPartialDecoder::new(
            input_handle,
            decoded_representation.clone(),
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
            pcodec_partial_decoder::AsyncPCodecPartialDecoder::new(
                input_handle,
                decoded_representation.clone(),
            ),
        ))
    }

    fn encoded_representation(
        &self,
        decoded_representation: &ChunkRepresentation,
    ) -> Result<BytesRepresentation, CodecError> {
        let data_type = decoded_representation.data_type();
        let mut num_elements = decoded_representation.num_elements_usize();
        if data_type == &DataType::Complex64 || data_type == &DataType::Complex128 {
            num_elements *= 2;
        }

        let size = match data_type {
            DataType::UInt16 | DataType::Int16 | DataType::Float16 | DataType::ComplexFloat16 => {
                Ok(
                    file_size::<u16>(num_elements, &self.chunk_config.paging_spec)
                        .map_err(|err| CodecError::from(err.to_string()))?,
                )
            }
            DataType::UInt32
            | DataType::Int32
            | DataType::Float32
            | DataType::Complex64
            | DataType::ComplexFloat32 => Ok(file_size::<u32>(
                num_elements,
                &self.chunk_config.paging_spec,
            )
            .map_err(|err| CodecError::from(err.to_string()))?),
            DataType::UInt64
            | DataType::Int64
            | DataType::Float64
            | DataType::Complex128
            | DataType::ComplexFloat64
            | DataType::NumpyDateTime64 {
                unit: _,
                scale_factor: _,
            }
            | DataType::NumpyTimeDelta64 {
                unit: _,
                scale_factor: _,
            } => Ok(
                file_size::<u64>(num_elements, &self.chunk_config.paging_spec)
                    .map_err(|err| CodecError::from(err.to_string()))?,
            ),
            super::unsupported_dtypes!() => Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                PCODEC.to_string(),
            )),
        }?;
        Ok(BytesRepresentation::BoundedSize(
            u64::try_from(size).unwrap(),
        ))
    }
}
