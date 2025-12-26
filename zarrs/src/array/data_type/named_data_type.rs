use std::ops::Deref;

use base64::{Engine, prelude::BASE64_STANDARD};
use zarrs_data_type::DataTypePlugin;
use zarrs_data_type::{DataTypeFillValueMetadataError, FillValue};
use zarrs_metadata::{
    ConfigurationSerialize,
    v3::{FillValueMetadataV3, MetadataV3},
};
use zarrs_plugin::{PluginCreateError, PluginMetadataInvalidError, PluginUnsupportedError};
use zarrs_registry::ExtensionAliasesDataTypeV3;

use crate::{
    array::{
        DataType, DataTypeOptional,
        data_type::{complex_subfloat_hex_string_to_fill_value, subfloat_hex_string_to_fill_value},
    },
    config::global_config,
};

/// A named data type.
#[derive(Debug, Clone, PartialEq, Eq)]
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

    /// Create a new [`NamedDataType`] with the default name for the data type.
    #[must_use]
    pub fn new_default_name(data_type: DataType, aliases: &ExtensionAliasesDataTypeV3) -> Self {
        let name = aliases.default_name(data_type.identifier()).to_string();
        Self { name, data_type }
    }

    /// Create a [`NamedDataType`] from metadata.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if the metadata is invalid or not associated with a registered data type plugin.
    #[allow(clippy::too_many_lines)]
    pub fn from_metadata(
        metadata: &MetadataV3,
        data_type_aliases: &ExtensionAliasesDataTypeV3,
    ) -> Result<Self, PluginCreateError> {
        if !metadata.must_understand() {
            return Err(PluginCreateError::Other(
                r#"data type must not have `"must_understand": false`"#.to_string(),
            ));
        }

        let identifier = data_type_aliases.identifier(metadata.name());
        if metadata.name() != identifier {
            log::info!(
                "Using data type alias `{}` for `{}`",
                metadata.name(),
                identifier
            );
        }

        let name = metadata.name().to_string();
        if let Some(configuration) = metadata.configuration() {
            match identifier {
                zarrs_registry::data_type::NUMPY_DATETIME64 => {
                    use crate::metadata_ext::data_type::numpy_datetime64::NumpyDateTime64DataTypeConfigurationV1;
                    let NumpyDateTime64DataTypeConfigurationV1 { unit, scale_factor } =
                        NumpyDateTime64DataTypeConfigurationV1::try_from_configuration(
                            configuration.clone(),
                        )
                        .map_err(|_| {
                            PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                                zarrs_registry::data_type::NUMPY_DATETIME64,
                                "data_type",
                                metadata.to_string(),
                            ))
                        })?;
                    return Ok(Self::new(
                        name,
                        DataType::NumpyDateTime64 { unit, scale_factor },
                    ));
                }
                zarrs_registry::data_type::NUMPY_TIMEDELTA64 => {
                    use crate::metadata_ext::data_type::numpy_timedelta64::NumpyTimeDelta64DataTypeConfigurationV1;
                    let NumpyTimeDelta64DataTypeConfigurationV1 { unit, scale_factor } =
                        NumpyTimeDelta64DataTypeConfigurationV1::try_from_configuration(
                            configuration.clone(),
                        )
                        .map_err(|_| {
                            PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                                zarrs_registry::data_type::NUMPY_TIMEDELTA64,
                                "data_type",
                                metadata.to_string(),
                            ))
                        })?;
                    return Ok(Self::new(
                        name,
                        DataType::NumpyTimeDelta64 { unit, scale_factor },
                    ));
                }
                zarrs_registry::data_type::OPTIONAL => {
                    use crate::metadata_ext::data_type::optional::OptionalDataTypeConfigurationV1;
                    let OptionalDataTypeConfigurationV1 {
                        name: inner_name,
                        configuration,
                    } = OptionalDataTypeConfigurationV1::try_from_configuration(
                        configuration.clone(),
                    )
                    .map_err(|_| {
                        PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                            zarrs_registry::data_type::OPTIONAL,
                            "data_type",
                            metadata.to_string(),
                        ))
                    })?;

                    // Create metadata for the inner data type
                    let inner_metadata = if configuration.is_empty() {
                        MetadataV3::new(inner_name)
                    } else {
                        MetadataV3::new_with_configuration(inner_name, configuration)
                    };

                    // Recursively parse the inner data type
                    let inner_data_type = Self::from_metadata(&inner_metadata, data_type_aliases)?;
                    let data_type = DataType::Optional(DataTypeOptional::new(inner_data_type));
                    return Ok(Self::new(name, data_type));
                }
                _ => {}
            }
        }

        if metadata.configuration_is_none_or_empty() {
            use zarrs_registry::data_type as dt;
            // Data types with no configuration
            match identifier {
                dt::BOOL => return Ok(Self::new(name, DataType::Bool)),
                dt::INT2 => return Ok(Self::new(name, DataType::Int2)),
                dt::INT4 => return Ok(Self::new(name, DataType::Int4)),
                dt::INT8 => return Ok(Self::new(name, DataType::Int8)),
                dt::INT16 => return Ok(Self::new(name, DataType::Int16)),
                dt::INT32 => return Ok(Self::new(name, DataType::Int32)),
                dt::INT64 => return Ok(Self::new(name, DataType::Int64)),
                dt::UINT2 => return Ok(Self::new(name, DataType::UInt2)),
                dt::UINT4 => return Ok(Self::new(name, DataType::UInt4)),
                dt::UINT8 => return Ok(Self::new(name, DataType::UInt8)),
                dt::UINT16 => return Ok(Self::new(name, DataType::UInt16)),
                dt::UINT32 => return Ok(Self::new(name, DataType::UInt32)),
                dt::UINT64 => return Ok(Self::new(name, DataType::UInt64)),
                dt::FLOAT4_E2M1FN => {
                    return Ok(Self::new(name, DataType::Float4E2M1FN));
                }
                dt::FLOAT6_E2M3FN => {
                    return Ok(Self::new(name, DataType::Float6E2M3FN));
                }
                dt::FLOAT6_E3M2FN => {
                    return Ok(Self::new(name, DataType::Float6E3M2FN));
                }
                dt::FLOAT8_E3M4 => {
                    return Ok(Self::new(name, DataType::Float8E3M4));
                }
                dt::FLOAT8_E4M3 => {
                    return Ok(Self::new(name, DataType::Float8E4M3));
                }
                dt::FLOAT8_E4M3B11FNUZ => {
                    return Ok(Self::new(name, DataType::Float8E4M3B11FNUZ));
                }
                dt::FLOAT8_E4M3FNUZ => {
                    return Ok(Self::new(name, DataType::Float8E4M3FNUZ));
                }
                dt::FLOAT8_E5M2 => {
                    return Ok(Self::new(name, DataType::Float8E5M2));
                }
                dt::FLOAT8_E5M2FNUZ => {
                    return Ok(Self::new(name, DataType::Float8E5M2FNUZ));
                }
                dt::FLOAT8_E8M0FNU => {
                    return Ok(Self::new(name, DataType::Float8E8M0FNU));
                }
                dt::BFLOAT16 => {
                    return Ok(Self::new(name, DataType::BFloat16));
                }
                dt::FLOAT16 => {
                    return Ok(Self::new(name, DataType::Float16));
                }
                dt::FLOAT32 => {
                    return Ok(Self::new(name, DataType::Float32));
                }
                dt::FLOAT64 => {
                    return Ok(Self::new(name, DataType::Float64));
                }
                dt::COMPLEX_BFLOAT16 => {
                    return Ok(Self::new(name, DataType::ComplexBFloat16));
                }
                dt::COMPLEX_FLOAT16 => {
                    return Ok(Self::new(name, DataType::ComplexFloat16));
                }
                dt::COMPLEX_FLOAT32 => {
                    return Ok(Self::new(name, DataType::ComplexFloat32));
                }
                dt::COMPLEX_FLOAT64 => {
                    return Ok(Self::new(name, DataType::ComplexFloat64));
                }
                dt::COMPLEX64 => {
                    return Ok(Self::new(name, DataType::Complex64));
                }
                dt::COMPLEX128 => {
                    return Ok(Self::new(name, DataType::Complex128));
                }
                dt::COMPLEX_FLOAT4_E2M1FN => {
                    return Ok(Self::new(name, DataType::ComplexFloat4E2M1FN));
                }
                dt::COMPLEX_FLOAT6_E2M3FN => {
                    return Ok(Self::new(name, DataType::ComplexFloat6E2M3FN));
                }
                dt::COMPLEX_FLOAT6_E3M2FN => {
                    return Ok(Self::new(name, DataType::ComplexFloat6E3M2FN));
                }
                dt::COMPLEX_FLOAT8_E3M4 => {
                    return Ok(Self::new(name, DataType::ComplexFloat8E3M4));
                }
                dt::COMPLEX_FLOAT8_E4M3 => {
                    return Ok(Self::new(name, DataType::ComplexFloat8E4M3));
                }
                dt::COMPLEX_FLOAT8_E4M3B11FNUZ => {
                    return Ok(Self::new(name, DataType::ComplexFloat8E4M3B11FNUZ));
                }
                dt::COMPLEX_FLOAT8_E4M3FNUZ => {
                    return Ok(Self::new(name, DataType::ComplexFloat8E4M3FNUZ));
                }
                dt::COMPLEX_FLOAT8_E5M2 => {
                    return Ok(Self::new(name, DataType::ComplexFloat8E5M2));
                }
                dt::COMPLEX_FLOAT8_E5M2FNUZ => {
                    return Ok(Self::new(name, DataType::ComplexFloat8E5M2FNUZ));
                }
                dt::COMPLEX_FLOAT8_E8M0FNU => {
                    return Ok(Self::new(name, DataType::ComplexFloat8E8M0FNU));
                }
                dt::STRING => return Ok(Self::new(name, DataType::String)),
                dt::BYTES => return Ok(Self::new(name, DataType::Bytes)),
                _ => {
                    if name.starts_with('r') && name.len() > 1 {
                        if let Ok(size_bits) = metadata.name()[1..].parse::<usize>() {
                            if size_bits % 8 == 0 {
                                let size_bytes = size_bits / 8;
                                return Ok(Self::new(name, DataType::RawBits(size_bytes)));
                            }
                            return Err(PluginUnsupportedError::new(
                                name.clone(),
                                "data type".to_string(),
                            )
                            .into());
                        }
                    }
                }
            }
        }

        // Try an extension
        for plugin in inventory::iter::<DataTypePlugin> {
            if plugin.match_name(identifier) {
                return plugin.create(metadata).map(|dt| {
                    NamedDataType::new(metadata.name().to_string(), DataType::Extension(dt))
                });
            }
        }

        // The data type is not supported
        Err(
            PluginUnsupportedError::new(metadata.name().to_string(), "data type".to_string())
                .into(),
        )
    }

    /// The name of the data type.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The underlying data type extension.
    #[must_use]
    pub const fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Wrap this data type in an [`Optional`](DataType::Optional).
    ///
    /// Can be chained to create nested optional types.
    ///
    /// # Examples
    /// ```
    /// # use zarrs::array::{DataType, DataTypeOptional};
    /// // Single level optional
    /// let aliases = zarrs::registry::ExtensionAliasesDataTypeV3::default();
    /// let opt_u8 = DataType::UInt8.into_named(&aliases).into_optional();
    /// # assert_eq!(opt_u8.data_type(), &DataType::Optional(DataTypeOptional::new(DataType::UInt8.into_named(&aliases))));
    ///
    /// // Nested optional
    /// let opt_opt_u8 = DataType::UInt8.into_named(&aliases).into_optional().into_optional();
    /// ```
    #[must_use]
    pub fn into_optional(self) -> Self {
        let data_type = DataType::Optional(DataTypeOptional::new(self));
        let name = global_config()
            .codec_aliases_v3()
            .default_name(data_type.identifier())
            .to_string();
        Self::new(name, data_type)
    }

    /// Create the data type metadata.
    #[must_use]
    pub fn metadata(&self) -> MetadataV3 {
        let configuration = self.data_type.configuration();
        if configuration.is_empty() {
            MetadataV3::new(self.name.clone())
        } else {
            MetadataV3::new_with_configuration(self.name.clone(), configuration)
        }
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
                    // Permit bytes for any data type alias of `bytes`
                    Ok(FV::from(bytes))
                } else if let Some(string) = fill_value.as_str() {
                    Ok(FV::from(
                        BASE64_STANDARD.decode(string).map_err(|_| err0())?,
                    ))
                } else {
                    Err(err0())?
                }
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
            DataType::Optional(opt) => {
                if fill_value.is_null() {
                    // Null fill value for optional type: single 0x00 byte
                    Ok(FillValue::new_optional_null())
                } else if let Some([inner]) = fill_value.as_array() {
                    // Wrapped value [inner_metadata] -> Some(inner)
                    let inner_fv = opt.fill_value_from_metadata(inner)?;
                    Ok(inner_fv.into_optional())
                } else {
                    // Invalid format for optional
                    Err(err0())
                }
            }
            DataType::Extension(ext) => ext.fill_value(fill_value).map_err(|_| {
                DataTypeFillValueMetadataError::new(name.to_string(), fill_value.clone())
            }),
        }
    }
}

impl Deref for NamedDataType {
    type Target = DataType;

    fn deref(&self) -> &Self::Target {
        &self.data_type
    }
}

impl From<NamedDataType> for DataType {
    fn from(value: NamedDataType) -> Self {
        value.data_type
    }
}
