use std::sync::Arc;

use zarrs_plugin::{ExtensionAliasesV3, PluginCreateError, ZarrVersion};

use super::{
    FixedScaleOffsetCodecConfiguration, FixedScaleOffsetCodecConfigurationNumcodecsF64,
    FixedScaleOffsetDataTypeExt, FixedScaleOffsetElementType,
};
use crate::array::{DataType, FillValue};
use crate::convert::data_type_metadata_v2_to_v3;
use std::num::NonZeroU64;
use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecTraits, CodecError, CodecMetadataOptions,
    CodecOptions, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
    RecommendedConcurrency,
};
use zarrs_metadata::Configuration;
use zarrs_metadata::v2::DataTypeMetadataV2;

/// A `fixedscaleoffset` codec implementation.
#[derive(Clone, Debug)]
pub struct FixedScaleOffsetCodec {
    offset: f64,
    scale: f64,
    dtype_str: String,
    astype_str: Option<String>,
    dtype: DataType,
    astype: Option<DataType>,
}

/// Add a byteorder to a numpy data type string (e.g. `f8` -> `<f8`, `i1` -> `|i1`) if it does not have one.
fn add_byteorder_to_dtype(dtype: &str) -> String {
    if dtype.starts_with(['<', '>', '|']) {
        dtype.to_string()
    } else if matches!(dtype, "b1" | "i1" | "u1") {
        // Single byte data types are not applicable to byte ordering
        format!("|{dtype}")
    } else {
        format!("<{dtype}")
    }
}

impl FixedScaleOffsetCodec {
    fn new(
        offset: f64,
        scale: f64,
        dtype_str: &str,
        astype_str: Option<&str>,
    ) -> Result<Self, PluginCreateError> {
        // Add a byteorder to the data type name, byteorder may be omitted
        // FixedScaleOffsets permits `dtype` / `astype` with and without a byteoder character, but it is irrelevant
        let dtype = add_byteorder_to_dtype(dtype_str);
        let astype = astype_str.map(add_byteorder_to_dtype);

        // Get the data type metadata
        let dtype = DataTypeMetadataV2::Simple(dtype);
        let astype = astype
            .as_ref()
            .map(|dtype| DataTypeMetadataV2::Simple(dtype.clone()));

        // Convert to a V3 data type
        let dtype_err = |_| {
            PluginCreateError::Other(
                "fixedscaleoffset cannot interpret Zarr V2 data type as V3 equivalent".to_string(),
            )
        };
        let dtype =
            DataType::from_metadata(&data_type_metadata_v2_to_v3(&dtype).map_err(dtype_err)?)?;
        let astype = if let Some(astype) = astype {
            Some(DataType::from_metadata(
                &data_type_metadata_v2_to_v3(&astype).map_err(dtype_err)?,
            )?)
        } else {
            None
        };

        Ok(Self {
            offset,
            scale,
            dtype,
            astype,
            dtype_str: dtype_str.to_string(),
            astype_str: astype_str.map(str::to_string),
        })
    }

    /// Create a new `fixedscaleoffset` codec from a configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &FixedScaleOffsetCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            FixedScaleOffsetCodecConfiguration::NumcodecsF64(configuration) => Self::new(
                configuration.offset,
                configuration.scale,
                &configuration.dtype,
                configuration.astype.as_deref(),
            ),
            FixedScaleOffsetCodecConfiguration::Numcodecs(configuration) => Self::new(
                f64::from(configuration.offset),
                f64::from(configuration.scale),
                &configuration.dtype,
                configuration.astype.as_deref(),
            ),
            _ => Err(PluginCreateError::Other(
                "this fixedscaleoffset codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for FixedScaleOffsetCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = FixedScaleOffsetCodecConfiguration::NumcodecsF64(
            FixedScaleOffsetCodecConfigurationNumcodecsF64 {
                offset: self.offset,
                scale: self.scale,
                dtype: self.dtype_str.clone(),
                astype: self.astype_str.clone(),
            },
        );
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        // NOTE: the default array-to-array partial decoder supports partial read/decode
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false, // TODO
        }
    }
}

impl ArrayCodecTraits for FixedScaleOffsetCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

fn get_element_type(
    data_type: &DataType,
) -> Result<FixedScaleOffsetElementType, zarrs_data_type::DataTypeCodecError> {
    let fso = data_type.codec_fixedscaleoffset()?;
    Ok(fso.fixedscaleoffset_element_type())
}

/// Bind `$T` to the Rust type of the `$element_type` elements, and optionally `$F` to the float type that `numpy` computes in for them (`f32` for `float32`, otherwise `f64`).
macro_rules! with_element_type {
    ($element_type:expr, $T:ident $(, $F:ident)?, $body:block) => {
        match $element_type {
            FixedScaleOffsetElementType::I8 => {
                type $T = i8;
                $(type $F = f64;)?
                $body
            }
            FixedScaleOffsetElementType::I16 => {
                type $T = i16;
                $(type $F = f64;)?
                $body
            }
            FixedScaleOffsetElementType::I32 => {
                type $T = i32;
                $(type $F = f64;)?
                $body
            }
            FixedScaleOffsetElementType::I64 => {
                type $T = i64;
                $(type $F = f64;)?
                $body
            }
            FixedScaleOffsetElementType::U8 => {
                type $T = u8;
                $(type $F = f64;)?
                $body
            }
            FixedScaleOffsetElementType::U16 => {
                type $T = u16;
                $(type $F = f64;)?
                $body
            }
            FixedScaleOffsetElementType::U32 => {
                type $T = u32;
                $(type $F = f64;)?
                $body
            }
            FixedScaleOffsetElementType::U64 => {
                type $T = u64;
                $(type $F = f64;)?
                $body
            }
            FixedScaleOffsetElementType::F32 => {
                type $T = f32;
                $(type $F = f32;)?
                $body
            }
            FixedScaleOffsetElementType::F64 => {
                type $T = f64;
                $(type $F = f64;)?
                $body
            }
        }
    };
}

/// Transform elements of `from` to elements of `to` with `$transform`, a function of an element `x` and `offset` and `scale` (`$offset` and `$scale` as the float type `x` is computed in).
///
/// As in `numpy`, elements are computed in [`f32`] if `from` is `float32`, otherwise [`f64`].
/// Integer elements of `to` are rounded to the nearest integer (ties to even) and saturated to the range of the type.
macro_rules! transform {
    ($bytes:expr, $from:expr, $to:expr, $offset:expr, $scale:expr, |$x:ident, $o:ident, $s:ident| $transform:expr) => {{
        let bytes: &[u8] = $bytes;
        let to_integer = !matches!(
            $to,
            FixedScaleOffsetElementType::F32 | FixedScaleOffsetElementType::F64
        );
        with_element_type!($from, T, F, {
            let ($o, $s) = ($offset as F, $scale as F);
            let elements = bytes.as_chunks::<{ size_of::<T>() }>().0;
            with_element_type!($to, U, {
                let mut out = Vec::with_capacity(elements.len() * size_of::<U>());
                for element in elements {
                    let $x = <T>::from_ne_bytes(*element) as F;
                    let value = $transform;
                    let value = if to_integer {
                        value.round_ties_even()
                    } else {
                        value
                    };
                    out.extend_from_slice(&(value as U).to_ne_bytes());
                }
                out
            })
        })
    }};
}

/// Encode elements of `element_type` as elements of `encoded_element_type`: `round((x - offset) * scale)`.
///
/// As in `numcodecs`, the transform is computed in [`f32`] for `float32` elements and otherwise [`f64`], and rounded to the nearest integer with ties to even (`numpy.around`).
/// Values out of the range of `encoded_element_type` are saturated (unspecified in `numcodecs`).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::unnecessary_cast
)]
fn do_encode(
    bytes: ArrayBytes<'_>,
    element_type: FixedScaleOffsetElementType,
    offset: f64,
    scale: f64,
    encoded_element_type: FixedScaleOffsetElementType,
) -> Result<ArrayBytes<'_>, CodecError> {
    let bytes = bytes.into_fixed()?;
    let encoded = transform!(
        &bytes,
        element_type,
        encoded_element_type,
        offset,
        scale,
        |x, offset, scale| ((x - offset) * scale).round_ties_even()
    );
    Ok(encoded.into())
}

/// Decode elements of `encoded_element_type` as elements of `element_type`: `x / scale + offset`.
///
/// As in `numcodecs`, the transform is computed in [`f32`] for `float32` encoded elements and otherwise [`f64`].
/// `numcodecs` truncates decoded integer elements, whereas they are rounded to the nearest integer (ties to even) here, so they may differ by one (the transform is lossy).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::unnecessary_cast
)]
fn do_decode(
    bytes: ArrayBytes<'_>,
    element_type: FixedScaleOffsetElementType,
    offset: f64,
    scale: f64,
    encoded_element_type: FixedScaleOffsetElementType,
) -> Result<ArrayBytes<'_>, CodecError> {
    let bytes = bytes.into_fixed()?;
    let decoded = transform!(
        &bytes,
        encoded_element_type,
        element_type,
        offset,
        scale,
        |x, offset, scale| x / scale + offset
    );
    Ok(decoded.into())
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for FixedScaleOffsetCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        if self.dtype != *data_type {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                FixedScaleOffsetCodec::aliases_v3().default_name.to_string(),
            ));
        }

        do_encode(
            bytes,
            get_element_type(data_type)?,
            self.offset,
            self.scale,
            get_element_type(self.astype.as_ref().unwrap_or(data_type))?,
        )
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        if self.dtype != *data_type {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                FixedScaleOffsetCodec::aliases_v3().default_name.to_string(),
            ));
        }

        do_decode(
            bytes,
            get_element_type(data_type)?,
            self.offset,
            self.scale,
            get_element_type(self.astype.as_ref().unwrap_or(data_type))?,
        )
    }

    fn encoded_fill_value(
        &self,
        data_type: &DataType,
        fill_value: &FillValue,
    ) -> Result<FillValue, CodecError> {
        let bytes = ArrayBytes::new_fill_value(data_type, 1, fill_value)?;
        let encoded = do_encode(
            bytes,
            get_element_type(data_type)?,
            self.offset,
            self.scale,
            get_element_type(self.astype.as_ref().unwrap_or(data_type))?,
        )?;
        Ok(FillValue::new(encoded.into_fixed()?.into_owned()))
    }

    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        // Check if the data type is supported by checking the trait
        get_element_type(decoded_data_type)?;

        if let Some(astype) = &self.astype {
            Ok(astype.clone())
        } else {
            Ok(decoded_data_type.clone())
        }
    }
}
