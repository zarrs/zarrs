use std::ops::Deref;

use base64::{prelude::BASE64_STANDARD, Engine};
use zarrs_data_type::{DataTypeFillValueMetadataError, FillValue};
use zarrs_metadata::{
    v3::{FillValueMetadataV3, MetadataV3},
    Configuration,
};

use crate::array::{
    data_type::{complex_subfloat_hex_string_to_fill_value, subfloat_hex_string_to_fill_value},
    DataType,
};

/// A named data type.
#[derive(Debug)]
pub struct NamedDataType {
    name: String,
    data_type: DataType,
}

impl NamedDataType {
    /// Create a new [`NamedDataType`].
    #[must_use]
    pub fn new(name: String, data_type: DataType) -> Self {
        Self { name, data_type }
    }

    /// The name of the data type.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create the data type metadata.
    #[must_use]
    pub fn metadata(&self) -> MetadataV3 {
        let configuration = self.configuration();
        if let Some(configuration) = configuration {
            MetadataV3::new_with_configuration(self.name(), configuration.clone())
        } else {
            MetadataV3::new(self.name())
        }
    }

    /// Create the data type configuration.
    #[must_use]
    pub fn configuration(&self) -> Option<Configuration> {
        self.data_type().metadata().into()
    }

    /// The underlying data type extension.
    #[must_use]
    pub const fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Create a fill value from metadata.
    ///
    /// # Errors
    /// Returns [`DataTypeFillValueMetadataError`] if the fill value is incompatible with the data type.
    #[allow(clippy::too_many_lines)]
    pub fn fill_value_from_metadata(
        &self,
        fill_value: &FillValueMetadataV3,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        use FillValue as FV;
        let name = self.name();
        let err0 = || DataTypeFillValueMetadataError::new(name.to_string(), fill_value.clone());
        let err = |_| DataTypeFillValueMetadataError::new(name.to_string(), fill_value.clone());
        match self.data_type() {
            DataType::Bool => Ok(FV::from(fill_value.as_bool().ok_or_else(err0)?)),
            DataType::Int2 => {
                let int = fill_value.as_i64().ok_or_else(err0)?;
                let int = i8::try_from(int).map_err(err)?;
                if (-2..2).contains(&int) {
                    Ok(FV::from(int))
                } else {
                    Err(err0())
                }
            }
            DataType::Int4 => {
                let int = fill_value.as_i64().ok_or_else(err0)?;
                let int = i8::try_from(int).map_err(err)?;
                if (-8..8).contains(&int) {
                    Ok(FV::from(int))
                } else {
                    Err(err0())
                }
            }
            DataType::Int8 => {
                let int = fill_value.as_i64().ok_or_else(err0)?;
                let int = i8::try_from(int).map_err(err)?;
                Ok(FV::from(int))
            }
            DataType::Int16 => {
                let int = fill_value.as_i64().ok_or_else(err0)?;
                let int = i16::try_from(int).map_err(err)?;
                Ok(FV::from(int))
            }
            DataType::Int32 => {
                let int = fill_value.as_i64().ok_or_else(err0)?;
                let int = i32::try_from(int).map_err(err)?;
                Ok(FV::from(int))
            }
            DataType::Int64 => {
                let int = fill_value.as_i64().ok_or_else(err0)?;
                Ok(FV::from(int))
            }
            DataType::UInt2 => {
                let int = fill_value.as_u64().ok_or_else(err0)?;
                let int = u8::try_from(int).map_err(err)?;
                if (0..4).contains(&int) {
                    Ok(FV::from(int))
                } else {
                    Err(err0())
                }
            }
            DataType::UInt4 => {
                let int = fill_value.as_u64().ok_or_else(err0)?;
                let int = u8::try_from(int).map_err(err)?;
                if (0..16).contains(&int) {
                    Ok(FV::from(int))
                } else {
                    Err(err0())
                }
            }
            DataType::UInt8 => {
                let int = fill_value.as_u64().ok_or_else(err0)?;
                let int = u8::try_from(int).map_err(err)?;
                Ok(FV::from(int))
            }
            DataType::UInt16 => {
                let int = fill_value.as_u64().ok_or_else(err0)?;
                let int = u16::try_from(int).map_err(err)?;
                Ok(FV::from(int))
            }
            DataType::UInt32 => {
                let int = fill_value.as_u64().ok_or_else(err0)?;
                let int = u32::try_from(int).map_err(err)?;
                Ok(FV::from(int))
            }
            DataType::UInt64 => {
                let int = fill_value.as_u64().ok_or_else(err0)?;
                Ok(FV::from(int))
            }
            DataType::Float8E4M3 => {
                #[cfg(feature = "float8")]
                {
                    subfloat_hex_string_to_fill_value(fill_value)
                        .or_else(|| {
                            let number = float8::F8E4M3::from_f64(fill_value.as_f64()?);
                            Some(FV::from(number.to_bits()))
                        })
                        .ok_or_else(err0)
                }
                #[cfg(not(feature = "float8"))]
                subfloat_hex_string_to_fill_value(fill_value).ok_or_else(err0)
            }
            DataType::Float8E5M2 => {
                #[cfg(feature = "float8")]
                {
                    subfloat_hex_string_to_fill_value(fill_value)
                        .or_else(|| {
                            let number = float8::F8E5M2::from_f64(fill_value.as_f64()?);
                            Some(FV::from(number.to_bits()))
                        })
                        .ok_or_else(err0)
                }
                #[cfg(not(feature = "float8"))]
                subfloat_hex_string_to_fill_value(fill_value).ok_or_else(err0)
            }
            DataType::Float4E2M1FN
            | DataType::Float6E2M3FN
            | DataType::Float6E3M2FN
            | DataType::Float8E3M4
            | DataType::Float8E4M3B11FNUZ
            | DataType::Float8E4M3FNUZ
            | DataType::Float8E5M2FNUZ
            | DataType::Float8E8M0FNU => {
                // FIXME: Support normal floating point fill value metadata for these data types.
                subfloat_hex_string_to_fill_value(fill_value).ok_or_else(err0)
            }
            DataType::BFloat16 => Ok(FV::from(fill_value.as_bf16().ok_or_else(err0)?)),
            DataType::Float16 => Ok(FV::from(fill_value.as_f16().ok_or_else(err0)?)),
            DataType::Float32 => Ok(FV::from(fill_value.as_f32().ok_or_else(err0)?)),
            DataType::Float64 => Ok(FV::from(fill_value.as_f64().ok_or_else(err0)?)),
            DataType::ComplexBFloat16 => {
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = re.as_bf16().ok_or_else(err0)?;
                    let im = im.as_bf16().ok_or_else(err0)?;
                    Ok(FV::from(num::complex::Complex::<half::bf16>::new(re, im)))
                } else {
                    Err(err0())?
                }
            }
            DataType::ComplexFloat16 => {
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = re.as_f16().ok_or_else(err0)?;
                    let im = im.as_f16().ok_or_else(err0)?;
                    Ok(FV::from(num::complex::Complex::<half::f16>::new(re, im)))
                } else {
                    Err(err0())?
                }
            }
            DataType::Complex64 | DataType::ComplexFloat32 => {
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = re.as_f32().ok_or_else(err0)?;
                    let im = im.as_f32().ok_or_else(err0)?;
                    Ok(FV::from(num::complex::Complex32::new(re, im)))
                } else {
                    Err(err0())?
                }
            }
            DataType::Complex128 | DataType::ComplexFloat64 => {
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = re.as_f64().ok_or_else(err0)?;
                    let im = im.as_f64().ok_or_else(err0)?;
                    Ok(FV::from(num::complex::Complex64::new(re, im)))
                } else {
                    Err(err0())?
                }
            }
            DataType::ComplexFloat8E4M3 => {
                #[cfg(feature = "float8")]
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = subfloat_hex_string_to_fill_value(re)
                        .or_else(|| {
                            let number = float8::F8E4M3::from_f64(re.as_f64()?);
                            Some(FV::from(number.to_bits()))
                        })
                        .ok_or_else(err0)?;
                    let im = subfloat_hex_string_to_fill_value(im)
                        .or_else(|| {
                            let number = float8::F8E4M3::from_f64(im.as_f64()?);
                            Some(FV::from(number.to_bits()))
                        })
                        .ok_or_else(err0)?;
                    Ok(FV::from(num::complex::Complex::new(re, im)))
                } else {
                    Err(err0())?
                }
                #[cfg(not(feature = "float8"))]
                complex_subfloat_hex_string_to_fill_value(fill_value).ok_or_else(err0)
            }
            DataType::ComplexFloat8E5M2 => {
                #[cfg(feature = "float8")]
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = subfloat_hex_string_to_fill_value(re)
                        .or_else(|| {
                            let number = float8::F8E5M2::from_f64(re.as_f64()?);
                            Some(FV::from(number.to_bits()))
                        })
                        .ok_or_else(err0)?;
                    let im = subfloat_hex_string_to_fill_value(im)
                        .or_else(|| {
                            let number = float8::F8E5M2::from_f64(im.as_f64()?);
                            Some(FV::from(number.to_bits()))
                        })
                        .ok_or_else(err0)?;
                    Ok(FV::from(num::complex::Complex::new(re, im)))
                } else {
                    Err(err0())?
                }
                #[cfg(not(feature = "float8"))]
                complex_subfloat_hex_string_to_fill_value(fill_value).ok_or_else(err0)
            }
            DataType::ComplexFloat4E2M1FN
            | DataType::ComplexFloat6E2M3FN
            | DataType::ComplexFloat6E3M2FN
            | DataType::ComplexFloat8E3M4
            | DataType::ComplexFloat8E4M3B11FNUZ
            | DataType::ComplexFloat8E4M3FNUZ
            | DataType::ComplexFloat8E5M2FNUZ
            | DataType::ComplexFloat8E8M0FNU => {
                // FIXME: Support normal floating point fill value metadata for these data types.
                complex_subfloat_hex_string_to_fill_value(fill_value).ok_or_else(err0)
            }
            DataType::RawBits(size) => {
                let bytes = fill_value.as_bytes().ok_or_else(err0)?;
                if bytes.len() == *size {
                    Ok(FV::from(bytes))
                } else {
                    Err(err0())?
                }
            }
            DataType::Bytes => {
                if let Some(bytes) = fill_value.as_bytes() {
                    return Ok(FV::from(bytes));
                } else if let Some(string) = fill_value.as_str() {
                    if name == "variable_length_bytes" {
                        // NOTE: zarr-python uses base64 encoded strings fill values for the `variable_length_bytes` data type.
                        // TODO: `zarrs` needs a NamedDataType API similar to NamedCodec to preserve names and handle this case specifically
                        return Ok(FV::from(
                            BASE64_STANDARD.decode(string).map_err(|_| err0())?,
                        ));
                    }
                    // Do not permit strings for the `bytes` data type
                }
                Err(err0())?
            }
            DataType::String => Ok(FV::from(fill_value.as_str().ok_or_else(err0)?)),
            DataType::NumpyDateTime64 {
                unit: _,
                scale_factor: _,
            }
            | DataType::NumpyTimeDelta64 {
                unit: _,
                scale_factor: _,
            } => {
                if let Some("NaT") = fill_value.as_str() {
                    Ok(FV::from(i64::MIN))
                } else if let Some(i) = fill_value.as_i64() {
                    Ok(FV::from(i))
                } else {
                    Err(err0())?
                }
            }
            DataType::Extension(ext) => ext.fill_value(fill_value),
        }
    }
}

impl Clone for NamedDataType {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            data_type: self.data_type.clone(),
        }
    }
}

impl From<DataType> for NamedDataType {
    fn from(data_type: DataType) -> Self {
        NamedDataType::new(data_type.name(), data_type)
    }
}

impl Deref for NamedDataType {
    type Target = DataType;

    fn deref(&self) -> &Self::Target {
        &self.data_type
    }
}
