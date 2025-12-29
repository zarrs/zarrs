use std::sync::Arc;

use zarrs_plugin::PluginCreateError;

use super::{FixedScaleOffsetCodecConfiguration, FixedScaleOffsetCodecConfigurationNumcodecs};
use crate::array::NamedDataType;
use crate::array::{
    DataType, FillValue,
    codec::{
        ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecTraits, CodecError, CodecMetadataOptions,
        CodecOptions, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
        RecommendedConcurrency,
    },
};
use crate::convert::data_type_metadata_v2_to_v3;
use crate::metadata::{Configuration, v2::DataTypeMetadataV2};
use std::num::NonZeroU64;
use zarrs_data_type::{DataTypeExtension, FixedScaleOffsetElementType, FixedScaleOffsetFloatType};
use zarrs_plugin::ExtensionIdentifier;

/// A `fixedscaleoffset` codec implementation.
#[derive(Clone, Debug)]
pub struct FixedScaleOffsetCodec {
    offset: f32,
    scale: f32,
    dtype_str: String,
    astype_str: Option<String>,
    dtype: NamedDataType,
    astype: Option<NamedDataType>,
}

fn add_byteoder_to_dtype(dtype: &str) -> String {
    if dtype == "u1" {
        "|u1".to_string()
    } else if !(dtype.starts_with('<') | dtype.starts_with('>')) {
        format!("<{dtype}")
    } else {
        dtype.to_string()
    }
}

impl FixedScaleOffsetCodec {
    /// Create a new `fixedscaleoffset` codec from a configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &FixedScaleOffsetCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            FixedScaleOffsetCodecConfiguration::Numcodecs(configuration) => {
                // Add a byteorder to the data type name, byteorder may be omitted
                // FixedScaleOffsets permits `dtype` / `astype` with and without a byteoder character, but it is irrelevant
                let dtype = add_byteoder_to_dtype(&configuration.dtype);
                let astype = configuration
                    .astype
                    .as_ref()
                    .map(|astype| add_byteoder_to_dtype(astype));

                // Get the data type metadata
                let dtype = DataTypeMetadataV2::Simple(dtype);
                let astype = astype
                    .as_ref()
                    .map(|dtype| DataTypeMetadataV2::Simple(dtype.clone()));

                // Convert to a V3 data type
                let dtype_err = |_| {
                    PluginCreateError::Other(
                        "fixedscaleoffset cannot interpret Zarr V2 data type as V3 equivalent"
                            .to_string(),
                    )
                };
                let dtype = NamedDataType::try_from(
                    &data_type_metadata_v2_to_v3(&dtype).map_err(dtype_err)?,
                )?;
                let astype = if let Some(astype) = astype {
                    Some(NamedDataType::try_from(
                        &data_type_metadata_v2_to_v3(&astype).map_err(dtype_err)?,
                    )?)
                } else {
                    None
                };

                Ok(Self {
                    offset: configuration.offset,
                    scale: configuration.scale,
                    dtype,
                    astype,
                    dtype_str: configuration.dtype.clone(),
                    astype_str: configuration.astype.clone(),
                })
            }
            _ => Err(PluginCreateError::Other(
                "this fixedscaleoffset codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for FixedScaleOffsetCodec {
    fn identifier(&self) -> &'static str {
        Self::IDENTIFIER
    }

    fn configuration(&self, _name: &str, _options: &CodecMetadataOptions) -> Option<Configuration> {
        let configuration = FixedScaleOffsetCodecConfiguration::Numcodecs(
            FixedScaleOffsetCodecConfigurationNumcodecs {
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

fn get_element_type(data_type: &DataType) -> Result<FixedScaleOffsetElementType, CodecError> {
    let fso = data_type.codec_fixedscaleoffset().ok_or_else(|| {
        CodecError::UnsupportedDataType(
            data_type.clone(),
            FixedScaleOffsetCodec::IDENTIFIER.to_string(),
        )
    })?;
    fso.fixedscaleoffset_element_type().ok_or_else(|| {
        CodecError::UnsupportedDataType(
            data_type.clone(),
            FixedScaleOffsetCodec::IDENTIFIER.to_string(),
        )
    })
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss
)]
fn scale_array(
    bytes: &mut [u8],
    data_type: &DataType,
    offset: f32,
    scale: f32,
) -> Result<(), CodecError> {
    let element_type = get_element_type(data_type)?;
    let float_type = element_type.intermediate_float();

    macro_rules! scale_impl {
        ($ty:ty, $float:ty) => {{
            for chunk in bytes.chunks_exact_mut(std::mem::size_of::<$ty>()) {
                let element = <$ty>::from_ne_bytes(chunk.try_into().unwrap());
                let element =
                    ((element as $float - offset as $float) * scale as $float).round() as $ty;
                chunk.copy_from_slice(&element.to_ne_bytes());
            }
        }};
    }

    match (element_type, float_type) {
        (FixedScaleOffsetElementType::I8, FixedScaleOffsetFloatType::F32) => scale_impl!(i8, f32),
        (FixedScaleOffsetElementType::I16, FixedScaleOffsetFloatType::F32) => scale_impl!(i16, f32),
        (FixedScaleOffsetElementType::I32, FixedScaleOffsetFloatType::F64) => scale_impl!(i32, f64),
        (FixedScaleOffsetElementType::I64, FixedScaleOffsetFloatType::F64) => scale_impl!(i64, f64),
        (FixedScaleOffsetElementType::U8, FixedScaleOffsetFloatType::F32) => scale_impl!(u8, f32),
        (FixedScaleOffsetElementType::U16, FixedScaleOffsetFloatType::F32) => scale_impl!(u16, f32),
        (FixedScaleOffsetElementType::U32, FixedScaleOffsetFloatType::F64) => scale_impl!(u32, f64),
        (FixedScaleOffsetElementType::U64, FixedScaleOffsetFloatType::F64) => scale_impl!(u64, f64),
        (FixedScaleOffsetElementType::F32, FixedScaleOffsetFloatType::F32) => scale_impl!(f32, f32),
        (FixedScaleOffsetElementType::F64, FixedScaleOffsetFloatType::F64) => scale_impl!(f64, f64),
        _ => {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                FixedScaleOffsetCodec::IDENTIFIER.to_string(),
            ));
        }
    }
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss
)]
fn unscale_array(
    bytes: &mut [u8],
    data_type: &DataType,
    offset: f32,
    scale: f32,
) -> Result<(), CodecError> {
    let element_type = get_element_type(data_type)?;
    let float_type = element_type.intermediate_float();

    macro_rules! unscale_impl {
        ($ty:ty, $float:ty) => {{
            for chunk in bytes.chunks_exact_mut(std::mem::size_of::<$ty>()) {
                let element = <$ty>::from_ne_bytes(chunk.try_into().unwrap());
                let element = ((element as $float / scale as $float) + offset as $float) as $ty;
                chunk.copy_from_slice(&element.to_ne_bytes());
            }
        }};
    }

    match (element_type, float_type) {
        (FixedScaleOffsetElementType::I8, FixedScaleOffsetFloatType::F32) => unscale_impl!(i8, f32),
        (FixedScaleOffsetElementType::I16, FixedScaleOffsetFloatType::F32) => {
            unscale_impl!(i16, f32);
        }
        (FixedScaleOffsetElementType::I32, FixedScaleOffsetFloatType::F64) => {
            unscale_impl!(i32, f64);
        }
        (FixedScaleOffsetElementType::I64, FixedScaleOffsetFloatType::F64) => {
            unscale_impl!(i64, f64);
        }
        (FixedScaleOffsetElementType::U8, FixedScaleOffsetFloatType::F32) => unscale_impl!(u8, f32),
        (FixedScaleOffsetElementType::U16, FixedScaleOffsetFloatType::F32) => {
            unscale_impl!(u16, f32);
        }
        (FixedScaleOffsetElementType::U32, FixedScaleOffsetFloatType::F64) => {
            unscale_impl!(u32, f64);
        }
        (FixedScaleOffsetElementType::U64, FixedScaleOffsetFloatType::F64) => {
            unscale_impl!(u64, f64);
        }
        (FixedScaleOffsetElementType::F32, FixedScaleOffsetFloatType::F32) => {
            unscale_impl!(f32, f32);
        }
        (FixedScaleOffsetElementType::F64, FixedScaleOffsetFloatType::F64) => {
            unscale_impl!(f64, f64);
        }
        _ => {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                FixedScaleOffsetCodec::IDENTIFIER.to_string(),
            ));
        }
    }
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss
)]
fn cast_array(
    bytes: &[u8],
    data_type: &DataType,
    as_type: &DataType,
) -> Result<Vec<u8>, CodecError> {
    let from_type = get_element_type(data_type)?;
    let to_type = get_element_type(as_type)?;

    // First cast to f32
    let elements: Vec<f32> = match from_type {
        FixedScaleOffsetElementType::I8 => bytes
            .chunks_exact(1)
            .map(|c| i8::from_ne_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        FixedScaleOffsetElementType::I16 => bytes
            .chunks_exact(2)
            .map(|c| i16::from_ne_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        FixedScaleOffsetElementType::I32 => bytes
            .chunks_exact(4)
            .map(|c| i32::from_ne_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        FixedScaleOffsetElementType::I64 => bytes
            .chunks_exact(8)
            .map(|c| i64::from_ne_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        FixedScaleOffsetElementType::U8 => bytes
            .chunks_exact(1)
            .map(|c| u8::from_ne_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        FixedScaleOffsetElementType::U16 => bytes
            .chunks_exact(2)
            .map(|c| u16::from_ne_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        FixedScaleOffsetElementType::U32 => bytes
            .chunks_exact(4)
            .map(|c| u32::from_ne_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        FixedScaleOffsetElementType::U64 => bytes
            .chunks_exact(8)
            .map(|c| u64::from_ne_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        FixedScaleOffsetElementType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_ne_bytes(c.try_into().unwrap()))
            .collect(),
        FixedScaleOffsetElementType::F64 => bytes
            .chunks_exact(8)
            .map(|c| f64::from_ne_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        _ => {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                FixedScaleOffsetCodec::IDENTIFIER.to_string(),
            ));
        }
    };

    // Then cast from f32 to target type
    let result: Vec<u8> = match to_type {
        FixedScaleOffsetElementType::I8 => elements
            .into_iter()
            .flat_map(|e| (e as i8).to_ne_bytes())
            .collect(),
        FixedScaleOffsetElementType::I16 => elements
            .into_iter()
            .flat_map(|e| (e as i16).to_ne_bytes())
            .collect(),
        FixedScaleOffsetElementType::I32 => elements
            .into_iter()
            .flat_map(|e| (e as i32).to_ne_bytes())
            .collect(),
        FixedScaleOffsetElementType::I64 => elements
            .into_iter()
            .flat_map(|e| (e as i64).to_ne_bytes())
            .collect(),
        FixedScaleOffsetElementType::U8 => elements
            .into_iter()
            .flat_map(|e| (e as u8).to_ne_bytes())
            .collect(),
        FixedScaleOffsetElementType::U16 => elements
            .into_iter()
            .flat_map(|e| (e as u16).to_ne_bytes())
            .collect(),
        FixedScaleOffsetElementType::U32 => elements
            .into_iter()
            .flat_map(|e| (e as u32).to_ne_bytes())
            .collect(),
        FixedScaleOffsetElementType::U64 => elements
            .into_iter()
            .flat_map(|e| (e as u64).to_ne_bytes())
            .collect(),
        FixedScaleOffsetElementType::F32 => {
            elements.into_iter().flat_map(f32::to_ne_bytes).collect()
        }
        FixedScaleOffsetElementType::F64 => elements
            .into_iter()
            .flat_map(|e| (e as f64).to_ne_bytes())
            .collect(),
        _ => {
            return Err(CodecError::UnsupportedDataType(
                as_type.clone(),
                FixedScaleOffsetCodec::IDENTIFIER.to_string(),
            ));
        }
    };

    Ok(result)
}

fn do_encode<'a>(
    bytes: ArrayBytes<'a>,
    data_type: &DataType,
    offset: f32,
    scale: f32,
    astype: Option<&DataType>,
) -> Result<ArrayBytes<'a>, CodecError> {
    let mut bytes = bytes.into_fixed()?.into_owned();
    scale_array(&mut bytes, data_type, offset, scale)?;
    if let Some(astype) = astype {
        Ok(cast_array(&bytes, data_type, astype)?.into())
    } else {
        Ok(bytes.into())
    }
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
        if self.dtype.data_type() != data_type {
            return Err(CodecError::Other(format!(
                "fixedscaleoffset got {} as input, but metadata expects {}",
                data_type,
                self.dtype.data_type()
            )));
        }

        do_encode(
            bytes,
            data_type,
            self.offset,
            self.scale,
            self.astype.as_ref().map(NamedDataType::data_type),
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
        if self.dtype.data_type() != data_type {
            return Err(CodecError::Other(format!(
                "fixedscaleoffset got {} as input, but metadata expects {}",
                data_type,
                self.dtype.data_type()
            )));
        }

        let bytes = bytes.into_fixed()?.into_owned();
        let mut bytes = if let Some(astype) = &self.astype {
            cast_array(&bytes, astype, data_type)?
        } else {
            bytes
        };
        unscale_array(&mut bytes, data_type, self.offset, self.scale)?;
        Ok(bytes.into())
    }

    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        // Check if the data type is supported by checking the trait
        get_element_type(decoded_data_type)?;

        if let Some(astype) = &self.astype {
            Ok(astype.data_type().clone())
        } else {
            Ok(decoded_data_type.clone())
        }
    }
}

zarrs_plugin::impl_extension_aliases!(FixedScaleOffsetCodec, "fixedscaleoffset",
    v3: "numcodecs.fixedscaleoffset", [],
    v2: "fixedscaleoffset", []
);
