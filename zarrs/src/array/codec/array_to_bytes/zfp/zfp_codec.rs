use std::{borrow::Cow, sync::Arc};

use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::{
    zfp::ZfpMode,
    zfpy::{ZfpyCodecConfiguration, ZfpyCodecConfigurationMode},
};
use zarrs_plugin::PluginCreateError;
use zarrs_registry::codec::ZFP;
use zfp_sys::{
    zfp_compress,
    zfp_stream_maximum_size,
    zfp_stream_rewind,
    zfp_stream_set_bit_stream,
    zfp_write_header,
    // zfp_exec_policy_zfp_exec_omp, zfp_stream_set_execution
};

use crate::array::{
    codec::{
        ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits,
        BytesPartialDecoderTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
        RawBytes, RecommendedConcurrency,
    },
    BytesRepresentation, ChunkRepresentation, DataType,
};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};

use super::{
    promote_before_zfp_encoding, zarr_to_zfp_data_type, zfp_bitstream::ZfpBitstream, zfp_decode,
    zfp_field::ZfpField, zfp_partial_decoder, zfp_stream::ZfpStream, ZfpCodecConfiguration,
    ZfpCodecConfigurationV1,
};

/// A `zfp` codec implementation.
#[derive(Clone, Copy, Debug)]
pub struct ZfpCodec {
    mode: ZfpMode,
    write_header: bool,
}

impl ZfpCodec {
    /// Create a new `zfp` codec in expert mode.
    #[must_use]
    pub const fn new_expert(minbits: u32, maxbits: u32, maxprec: u32, minexp: i32) -> Self {
        Self {
            mode: ZfpMode::Expert {
                minbits,
                maxbits,
                maxprec,
                minexp,
            },
            write_header: false,
        }
    }

    /// Create a new `zfp` codec in fixed rate mode.
    #[must_use]
    pub const fn new_fixed_rate(rate: f64) -> Self {
        Self {
            mode: ZfpMode::FixedRate { rate },
            write_header: false,
        }
    }

    /// Create a new `zfp` codec in fixed precision mode.
    #[must_use]
    pub const fn new_fixed_precision(precision: u32) -> Self {
        Self {
            mode: ZfpMode::FixedPrecision { precision },
            write_header: false,
        }
    }

    /// Create a new `zfp` codec in fixed accuracy mode.
    #[must_use]
    pub const fn new_fixed_accuracy(tolerance: f64) -> Self {
        Self {
            mode: ZfpMode::FixedAccuracy { tolerance },
            write_header: false,
        }
    }

    /// Create a new `zfp` codec in reversible mode.
    #[must_use]
    pub const fn new_reversible() -> Self {
        Self {
            mode: ZfpMode::Reversible,
            write_header: false,
        }
    }

    /// Create a new `zfp` codec a `zfpy` codec configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration_zfpy(
        configuration: &ZfpyCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        // zfpy writes a redundant header
        match configuration {
            ZfpyCodecConfiguration::Numcodecs(configuration) => match configuration.mode {
                ZfpyCodecConfigurationMode::FixedRate { rate } => Ok(Self {
                    mode: ZfpMode::FixedRate { rate },
                    write_header: true,
                }),
                ZfpyCodecConfigurationMode::FixedPrecision { precision } => Ok(Self {
                    mode: ZfpMode::FixedPrecision { precision },
                    write_header: true,
                }),
                ZfpyCodecConfigurationMode::FixedAccuracy { tolerance } => Ok(Self {
                    mode: ZfpMode::FixedAccuracy { tolerance },
                    write_header: true,
                }),
                ZfpyCodecConfigurationMode::Reversible => Ok(Self {
                    mode: ZfpMode::Reversible,
                    write_header: true,
                }),
            },
            _ => Err(PluginCreateError::Other(
                "this zfpy codec configuration variant is unsupported".to_string(),
            ))?,
        }
    }

    /// Create a new `zfp` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &ZfpCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        let configuration = match configuration {
            ZfpCodecConfiguration::V1(configuration) => configuration.clone(),
            _ => Err(PluginCreateError::Other(
                "this zfp codec configuration variant is unsupported".to_string(),
            ))?,
        };

        Ok(match configuration.mode {
            ZfpMode::Expert {
                minbits,
                maxbits,
                maxprec,
                minexp,
            } => Self::new_expert(minbits, maxbits, maxprec, minexp),
            ZfpMode::FixedRate { rate } => Self::new_fixed_rate(rate),
            ZfpMode::FixedPrecision { precision } => Self::new_fixed_precision(precision),
            ZfpMode::FixedAccuracy { tolerance } => Self::new_fixed_accuracy(tolerance),
            ZfpMode::Reversible => Self::new_reversible(),
        })
    }
}

impl CodecTraits for ZfpCodec {
    fn identifier(&self) -> &str {
        ZFP
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        // ZfpyCodecConfigurationNumcodecs is forward compatible with ZfpCodecConfigurationV1
        Some(ZfpCodecConfiguration::V1(ZfpCodecConfigurationV1 { mode: self.mode }).into())
    }

    fn partial_decoder_should_cache_input(&self) -> bool {
        false
    }

    fn partial_decoder_decodes_all(&self) -> bool {
        true
    }
}

impl ArrayCodecTraits for ZfpCodec {
    fn recommended_concurrency(
        &self,
        _decoded_representation: &ChunkRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        // TODO: zfp supports multi thread, when is it optimal to kick in?
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(feature = "async", async_trait::async_trait)]
impl ArrayToBytesCodecTraits for ZfpCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        let bytes = bytes.into_fixed()?;
        let mut bytes_promoted = promote_before_zfp_encoding(&bytes, decoded_representation)?;
        let zfp_type = bytes_promoted.zfp_type();
        let field = ZfpField::new(
            &mut bytes_promoted,
            &decoded_representation
                .shape()
                .iter()
                .map(|u| usize::try_from(u.get()).unwrap())
                .collect::<Vec<usize>>(),
        )
        .ok_or_else(|| CodecError::from("failed to create zfp field"))?;
        let stream = ZfpStream::new(&self.mode, zfp_type)
            .ok_or_else(|| CodecError::from("failed to create zfp stream"))?;

        let bufsize = unsafe {
            // SAFETY: zfp stream and field are valid
            zfp_stream_maximum_size(stream.as_zfp_stream(), field.as_zfp_field())
        };
        let mut encoded_value: Vec<u8> = vec![0; bufsize];

        let bitstream = ZfpBitstream::new(&mut encoded_value)
            .ok_or_else(|| CodecError::from("failed to create zfp field"))?;
        unsafe {
            // SAFETY: zfp stream and bitstream are valid
            zfp_stream_set_bit_stream(stream.as_zfp_stream(), bitstream.as_bitstream());
            zfp_stream_rewind(stream.as_zfp_stream()); // needed?
        }
        if self.write_header {
            unsafe {
                // SAFETY: zfp stream and field are valid
                zfp_write_header(
                    stream.as_zfp_stream(),
                    field.as_zfp_field(),
                    zfp_sys::ZFP_HEADER_FULL,
                );
            };
        }

        // FIXME
        // if parallel {
        //     // Number of threads is set automatically
        //     unsafe {
        //         zfp_stream_set_execution(zfp.as_zfp_stream(), zfp_exec_policy_zfp_exec_omp);
        //     }
        // }

        // Compress array
        let size = unsafe {
            // SAFETY: zfp stream and field are valid
            zfp_compress(stream.as_zfp_stream(), field.as_zfp_field())
        };

        if size == 0 {
            Err(CodecError::from("zfp compression failed"))
        } else {
            encoded_value.truncate(size);
            Ok(Cow::Owned(encoded_value))
        }
    }

    fn decode<'a>(
        &self,
        bytes: RawBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        zfp_decode(
            &self.mode,
            self.write_header,
            &mut bytes.to_vec(), // FIXME: Does zfp **really** need the encoded value as mutable?
            decoded_representation,
            false, // FIXME
        )
        .map(ArrayBytes::from)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(zfp_partial_decoder::ZfpPartialDecoder::new(
            input_handle,
            decoded_representation,
            self.mode,
            self.write_header,
        )?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(zfp_partial_decoder::AsyncZfpPartialDecoder::new(
            input_handle,
            decoded_representation,
            self.mode,
            self.write_header,
        )?))
    }

    fn encoded_representation(
        &self,
        decoded_representation: &ChunkRepresentation,
    ) -> Result<BytesRepresentation, CodecError> {
        let data_type = decoded_representation.data_type();
        let zfp_type = zarr_to_zfp_data_type(decoded_representation.data_type())
            .ok_or_else(|| CodecError::from("data type {} is unsupported for zfp codec"))?;

        let bufsize = {
            let field = unsafe {
                // SAFETY: zfp_stream_maximum_size does not use the data in the field, so it can be empty
                ZfpField::new_empty(
                    zfp_type,
                    &decoded_representation
                        .shape()
                        .iter()
                        .map(|u| usize::try_from(u.get()).unwrap())
                        .collect::<Vec<usize>>(),
                )
            }
            .ok_or_else(|| CodecError::from("failed to create zfp field"))?;

            let stream = ZfpStream::new(&self.mode, zfp_type)
                .ok_or_else(|| CodecError::from("failed to create zfp stream"))?;

            unsafe {
                // SAFETY: zfp stream and field are valid
                zfp_stream_maximum_size(stream.as_zfp_stream(), field.as_zfp_field())
            }
        };

        match data_type {
            DataType::Int8
            | DataType::UInt8
            | DataType::Int16
            | DataType::UInt16
            | DataType::Int32
            | DataType::UInt32
            | DataType::Int64
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::NumpyDateTime64 {
                unit: _,
                scale_factor: _,
            }
            | DataType::NumpyTimeDelta64 {
                unit: _,
                scale_factor: _,
            } => Ok(BytesRepresentation::BoundedSize(bufsize as u64)),
            super::unsupported_dtypes!() => Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                ZFP.to_string(),
            )),
        }
    }
}
