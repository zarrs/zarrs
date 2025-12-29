use std::ops::Deref;

use base64::{Engine, prelude::BASE64_STANDARD};
use zarrs_data_type::DataTypePlugin;
use zarrs_data_type::{DataTypeFillValueMetadataError, FillValue};
use zarrs_metadata::{
    ConfigurationSerialize,
    v3::{FillValueMetadataV3, MetadataV3},
};
use zarrs_plugin::{
    ExtensionIdentifier, PluginCreateError, PluginMetadataInvalidError, PluginUnsupportedError,
    ZarrVersions,
};

use crate::array::{
    DataType,
    data_type::{
        BFloat16DataType, BoolDataType, BytesDataType, Complex64DataType, Complex128DataType,
        ComplexBFloat16DataType, ComplexFloat4E2M1FNDataType, ComplexFloat6E2M3FNDataType,
        ComplexFloat6E3M2FNDataType, ComplexFloat8E3M4DataType, ComplexFloat8E4M3B11FNUZDataType,
        ComplexFloat8E4M3DataType, ComplexFloat8E4M3FNUZDataType, ComplexFloat8E5M2DataType,
        ComplexFloat8E5M2FNUZDataType, ComplexFloat8E8M0FNUDataType, ComplexFloat16DataType,
        ComplexFloat32DataType, ComplexFloat64DataType, Float4E2M1FNDataType, Float6E2M3FNDataType,
        Float6E3M2FNDataType, Float8E3M4DataType, Float8E4M3B11FNUZDataType, Float8E4M3DataType,
        Float8E4M3FNUZDataType, Float8E5M2DataType, Float8E5M2FNUZDataType, Float8E8M0FNUDataType,
        Float16DataType, Float32DataType, Float64DataType, Int2DataType, Int4DataType,
        Int8DataType, Int16DataType, Int32DataType, Int64DataType, NumpyDateTime64DataType,
        NumpyTimeDelta64DataType, OptionalDataType, RawBitsDataType, StringDataType, UInt2DataType,
        UInt4DataType, UInt8DataType, UInt16DataType, UInt32DataType, UInt64DataType,
        subfloat_hex_string_to_fill_value,
    },
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
    pub fn new_default_name(data_type: DataType) -> Self {
        let name = data_type.default_name().into_owned();
        Self { name, data_type }
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
    /// # use zarrs::array::{DataType, data_type::OptionalDataType};
    /// // Single level optional
    /// let opt_u8 = DataType::UInt8.into_optional();
    /// # assert_eq!(opt_u8, DataType::Optional(OptionalDataType::new(DataType::UInt8.into_named())));
    ///
    /// // Nested optional
    /// let opt_opt_u8 = DataType::UInt8.into_optional().into_optional();
    /// ```
    #[must_use]
    pub fn into_optional(self) -> Self {
        let data_type = DataType::Optional(OptionalDataType::new(self));
        let name = data_type.default_name().into_owned();
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
    pub fn fill_value_from_metadata(
        &self,
        fill_value: &FillValueMetadataV3,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        use zarrs_data_type::DataTypeExtension;
        let name = self.name();
        let err0 = || DataTypeFillValueMetadataError::new(name.to_string(), fill_value.clone());
        match self.data_type() {
            // Delegate to marker types for standard numeric types
            DataType::Bool => BoolDataType.fill_value(fill_value),
            DataType::Int2 => Int2DataType.fill_value(fill_value),
            DataType::Int4 => Int4DataType.fill_value(fill_value),
            DataType::Int8 => Int8DataType.fill_value(fill_value),
            DataType::Int16 => Int16DataType.fill_value(fill_value),
            DataType::Int32 => Int32DataType.fill_value(fill_value),
            DataType::Int64 => Int64DataType.fill_value(fill_value),
            DataType::UInt2 => UInt2DataType.fill_value(fill_value),
            DataType::UInt4 => UInt4DataType.fill_value(fill_value),
            DataType::UInt8 => UInt8DataType.fill_value(fill_value),
            DataType::UInt16 => UInt16DataType.fill_value(fill_value),
            DataType::UInt32 => UInt32DataType.fill_value(fill_value),
            DataType::UInt64 => UInt64DataType.fill_value(fill_value),
            DataType::BFloat16 => BFloat16DataType.fill_value(fill_value),
            DataType::Float16 => Float16DataType.fill_value(fill_value),
            DataType::Float32 => Float32DataType.fill_value(fill_value),
            DataType::Float64 => Float64DataType.fill_value(fill_value),
            DataType::ComplexBFloat16 => ComplexBFloat16DataType.fill_value(fill_value),
            DataType::ComplexFloat16 => ComplexFloat16DataType.fill_value(fill_value),
            DataType::ComplexFloat32 => ComplexFloat32DataType.fill_value(fill_value),
            DataType::ComplexFloat64 => ComplexFloat64DataType.fill_value(fill_value),
            DataType::Complex64 => Complex64DataType.fill_value(fill_value),
            DataType::Complex128 => Complex128DataType.fill_value(fill_value),
            DataType::String => StringDataType.fill_value(fill_value),
            DataType::NumpyDateTime64 { .. } => NumpyDateTime64DataType::STATIC.fill_value(fill_value),
            DataType::NumpyTimeDelta64 { .. } => NumpyTimeDelta64DataType::STATIC.fill_value(fill_value),
            // Float8E4M3 and Float8E5M2 support float values when float8 feature is enabled
            DataType::Float8E4M3 => {
                #[cfg(feature = "float8")]
                {
                    subfloat_hex_string_to_fill_value(fill_value)
                        .or_else(|| {
                            let number = float8::F8E4M3::from_f64(fill_value.as_f64()?);
                            Some(FillValue::from(number.to_bits()))
                        })
                        .ok_or_else(err0)
                }
                #[cfg(not(feature = "float8"))]
                Float8E4M3DataType.fill_value(fill_value)
            }
            DataType::Float8E5M2 => {
                #[cfg(feature = "float8")]
                {
                    subfloat_hex_string_to_fill_value(fill_value)
                        .or_else(|| {
                            let number = float8::F8E5M2::from_f64(fill_value.as_f64()?);
                            Some(FillValue::from(number.to_bits()))
                        })
                        .ok_or_else(err0)
                }
                #[cfg(not(feature = "float8"))]
                Float8E5M2DataType.fill_value(fill_value)
            }
            // Other subfloats use marker types directly
            DataType::Float4E2M1FN => Float4E2M1FNDataType.fill_value(fill_value),
            DataType::Float6E2M3FN => Float6E2M3FNDataType.fill_value(fill_value),
            DataType::Float6E3M2FN => Float6E3M2FNDataType.fill_value(fill_value),
            DataType::Float8E3M4 => Float8E3M4DataType.fill_value(fill_value),
            DataType::Float8E4M3B11FNUZ => Float8E4M3B11FNUZDataType.fill_value(fill_value),
            DataType::Float8E4M3FNUZ => Float8E4M3FNUZDataType.fill_value(fill_value),
            DataType::Float8E5M2FNUZ => Float8E5M2FNUZDataType.fill_value(fill_value),
            DataType::Float8E8M0FNU => Float8E8M0FNUDataType.fill_value(fill_value),
            // ComplexFloat8E4M3 and ComplexFloat8E5M2 support float values when float8 feature is enabled
            DataType::ComplexFloat8E4M3 => {
                #[cfg(feature = "float8")]
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = subfloat_hex_string_to_fill_value(re)
                        .or_else(|| {
                            let number = float8::F8E4M3::from_f64(re.as_f64()?);
                            Some(FillValue::from(number.to_bits()))
                        })
                        .ok_or_else(err0)?;
                    let im = subfloat_hex_string_to_fill_value(im)
                        .or_else(|| {
                            let number = float8::F8E4M3::from_f64(im.as_f64()?);
                            Some(FillValue::from(number.to_bits()))
                        })
                        .ok_or_else(err0)?;
                    Ok(FillValue::from(num::complex::Complex::new(re, im)))
                } else {
                    Err(err0())?
                }
                #[cfg(not(feature = "float8"))]
                ComplexFloat8E4M3DataType.fill_value(fill_value)
            }
            DataType::ComplexFloat8E5M2 => {
                #[cfg(feature = "float8")]
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = subfloat_hex_string_to_fill_value(re)
                        .or_else(|| {
                            let number = float8::F8E5M2::from_f64(re.as_f64()?);
                            Some(FillValue::from(number.to_bits()))
                        })
                        .ok_or_else(err0)?;
                    let im = subfloat_hex_string_to_fill_value(im)
                        .or_else(|| {
                            let number = float8::F8E5M2::from_f64(im.as_f64()?);
                            Some(FillValue::from(number.to_bits()))
                        })
                        .ok_or_else(err0)?;
                    Ok(FillValue::from(num::complex::Complex::new(re, im)))
                } else {
                    Err(err0())?
                }
                #[cfg(not(feature = "float8"))]
                ComplexFloat8E5M2DataType.fill_value(fill_value)
            }
            // Other complex subfloats use marker types directly
            DataType::ComplexFloat4E2M1FN => ComplexFloat4E2M1FNDataType.fill_value(fill_value),
            DataType::ComplexFloat6E2M3FN => ComplexFloat6E2M3FNDataType.fill_value(fill_value),
            DataType::ComplexFloat6E3M2FN => ComplexFloat6E3M2FNDataType.fill_value(fill_value),
            DataType::ComplexFloat8E3M4 => ComplexFloat8E3M4DataType.fill_value(fill_value),
            DataType::ComplexFloat8E4M3B11FNUZ => ComplexFloat8E4M3B11FNUZDataType.fill_value(fill_value),
            DataType::ComplexFloat8E4M3FNUZ => ComplexFloat8E4M3FNUZDataType.fill_value(fill_value),
            DataType::ComplexFloat8E5M2FNUZ => ComplexFloat8E5M2FNUZDataType.fill_value(fill_value),
            DataType::ComplexFloat8E8M0FNU => ComplexFloat8E8M0FNUDataType.fill_value(fill_value),
            // RawBits uses array representation (legacy behavior)
            DataType::RawBits(size) => {
                let bytes = fill_value.as_bytes().ok_or_else(err0)?;
                if bytes.len() == *size {
                    Ok(FillValue::from(bytes))
                } else {
                    Err(err0())?
                }
            }
            // Bytes supports both array and base64 (legacy behavior)
            DataType::Bytes => {
                if let Some(bytes) = fill_value.as_bytes() {
                    // Permit bytes for any data type alias of `bytes`
                    Ok(FillValue::from(bytes))
                } else if let Some(string) = fill_value.as_str() {
                    Ok(FillValue::from(
                        BASE64_STANDARD.decode(string).map_err(|_| err0())?,
                    ))
                } else {
                    Err(err0())?
                }
            }
            // Optional has special recursive handling
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

impl TryFrom<&MetadataV3> for NamedDataType {
    type Error = PluginCreateError;

    /// Create a [`NamedDataType`] from metadata.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if the metadata is invalid or not associated with a registered data type plugin.
    #[allow(clippy::too_many_lines)]
    fn try_from(metadata: &MetadataV3) -> Result<Self, Self::Error> {
        if !metadata.must_understand() {
            return Err(PluginCreateError::Other(
                r#"data type must not have `"must_understand": false`"#.to_string(),
            ));
        }

        let name = metadata.name();

        // Handle data types with configuration
        if let Some(configuration) = metadata.configuration() {
            if NumpyDateTime64DataType::matches_name(name, ZarrVersions::V3) {
                use crate::metadata_ext::data_type::numpy_datetime64::NumpyDateTime64DataTypeConfigurationV1;
                let NumpyDateTime64DataTypeConfigurationV1 { unit, scale_factor } =
                    NumpyDateTime64DataTypeConfigurationV1::try_from_configuration(
                        configuration.clone(),
                    )
                    .map_err(|_| {
                        PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                            NumpyDateTime64DataType::IDENTIFIER,
                            "data_type",
                            metadata.to_string(),
                        ))
                    })?;
                return Ok(Self::new(
                    name.to_string(),
                    DataType::NumpyDateTime64 { unit, scale_factor },
                ));
            }

            if NumpyTimeDelta64DataType::matches_name(name, ZarrVersions::V3) {
                use crate::metadata_ext::data_type::numpy_timedelta64::NumpyTimeDelta64DataTypeConfigurationV1;
                let NumpyTimeDelta64DataTypeConfigurationV1 { unit, scale_factor } =
                    NumpyTimeDelta64DataTypeConfigurationV1::try_from_configuration(
                        configuration.clone(),
                    )
                    .map_err(|_| {
                        PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                            NumpyTimeDelta64DataType::IDENTIFIER,
                            "data_type",
                            metadata.to_string(),
                        ))
                    })?;
                return Ok(Self::new(
                    name.to_string(),
                    DataType::NumpyTimeDelta64 { unit, scale_factor },
                ));
            }

            if OptionalDataType::matches_name(name, ZarrVersions::V3) {
                use crate::metadata_ext::data_type::optional::OptionalDataTypeConfigurationV1;
                let OptionalDataTypeConfigurationV1 {
                    name: inner_name,
                    configuration,
                } = OptionalDataTypeConfigurationV1::try_from_configuration(configuration.clone())
                    .map_err(|_| {
                        PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                            OptionalDataType::IDENTIFIER,
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
                let inner_data_type = Self::try_from(&inner_metadata)?;
                let data_type = DataType::Optional(OptionalDataType::new(inner_data_type));
                return Ok(Self::new(name.to_string(), data_type));
            }
        }

        // Handle data types with no configuration
        if metadata.configuration_is_none_or_empty() {
            // Boolean
            if BoolDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Bool));
            }

            // Signed integers
            if Int2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Int2));
            }
            if Int4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Int4));
            }
            if Int8DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Int8));
            }
            if Int16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Int16));
            }
            if Int32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Int32));
            }
            if Int64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Int64));
            }

            // Unsigned integers
            if UInt2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::UInt2));
            }
            if UInt4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::UInt4));
            }
            if UInt8DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::UInt8));
            }
            if UInt16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::UInt16));
            }
            if UInt32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::UInt32));
            }
            if UInt64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::UInt64));
            }

            // Subfloats
            if Float4E2M1FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float4E2M1FN));
            }
            if Float6E2M3FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float6E2M3FN));
            }
            if Float6E3M2FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float6E3M2FN));
            }
            if Float8E3M4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E3M4));
            }
            if Float8E4M3DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E4M3));
            }
            if Float8E4M3B11FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E4M3B11FNUZ));
            }
            if Float8E4M3FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E4M3FNUZ));
            }
            if Float8E5M2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E5M2));
            }
            if Float8E5M2FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E5M2FNUZ));
            }
            if Float8E8M0FNUDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E8M0FNU));
            }

            // Standard floats
            if BFloat16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::BFloat16));
            }
            if Float16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float16));
            }
            if Float32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float32));
            }
            if Float64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float64));
            }

            // Complex subfloats
            if ComplexFloat4E2M1FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat4E2M1FN));
            }
            if ComplexFloat6E2M3FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat6E2M3FN));
            }
            if ComplexFloat6E3M2FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat6E3M2FN));
            }
            if ComplexFloat8E3M4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat8E3M4));
            }
            if ComplexFloat8E4M3DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat8E4M3));
            }
            if ComplexFloat8E4M3B11FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    DataType::ComplexFloat8E4M3B11FNUZ,
                ));
            }
            if ComplexFloat8E4M3FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat8E4M3FNUZ));
            }
            if ComplexFloat8E5M2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat8E5M2));
            }
            if ComplexFloat8E5M2FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat8E5M2FNUZ));
            }
            if ComplexFloat8E8M0FNUDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat8E8M0FNU));
            }

            // Complex floats
            if ComplexBFloat16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexBFloat16));
            }
            if ComplexFloat16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat16));
            }
            if ComplexFloat32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat32));
            }
            if ComplexFloat64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat64));
            }
            if Complex64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Complex64));
            }
            if Complex128DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Complex128));
            }

            // Variable-length types
            if StringDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::String));
            }
            if BytesDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Bytes));
            }

            // RawBits (r8, r16, r24, etc.)
            if RawBitsDataType::matches_name(name, ZarrVersions::V3) {
                if let Ok(size_bits) = name[1..].parse::<usize>() {
                    if size_bits % 8 == 0 {
                        let size_bytes = size_bits / 8;
                        return Ok(Self::new(name.to_string(), DataType::RawBits(size_bytes)));
                    }
                    return Err(PluginUnsupportedError::new(
                        name.to_string(),
                        "data type".to_string(),
                    )
                    .into());
                }
            }
        }

        // Try an extension plugin
        for plugin in inventory::iter::<DataTypePlugin> {
            if plugin.match_name(name, ZarrVersions::V3) {
                return plugin.create(metadata).map(|dt| {
                    NamedDataType::new(metadata.name().to_string(), DataType::Extension(dt))
                });
            }
        }

        // The data type is not supported
        Err(PluginUnsupportedError::new(name.to_string(), "data type".to_string()).into())
    }
}
