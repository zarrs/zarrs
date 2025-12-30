use std::ops::Deref;
use std::sync::Arc;

use zarrs_data_type::{DataTypeFillValueMetadataError, DataTypePlugin, FillValue};
use zarrs_metadata::{
    ConfigurationSerialize,
    v3::{FillValueMetadataV3, MetadataV3},
};
use zarrs_plugin::{
    ExtensionIdentifier, PluginCreateError, PluginMetadataInvalidError, PluginUnsupportedError,
    ZarrVersions,
};

use crate::array::{DataType, data_type};

/// A named data type.
#[derive(Debug, Clone)]
pub struct NamedDataType {
    name: String,
    data_type: DataType,
}

impl PartialEq for NamedDataType {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.data_type.eq(other.data_type.as_ref())
    }
}

impl Eq for NamedDataType {}

impl NamedDataType {
    /// Create a new [`NamedDataType`].
    #[must_use]
    pub fn new(name: String, data_type: DataType) -> Self {
        Self { name, data_type }
    }

    /// Create a new [`NamedDataType`] with the default name for the data type.
    ///
    /// Uses the instance `default_name` if it provides one, otherwise uses the type-level registered default name.
    #[must_use]
    pub fn new_default_name(data_type: DataType) -> Self {
        let name = data_type.default_name(ZarrVersions::V3);
        if let Some(name) = name {
            Self::new(name.into_owned(), data_type)
        } else {
            for plugin in inventory::iter::<DataTypePlugin> {
                if plugin.identifier() == data_type.identifier() {
                    let default_name = plugin.default_name(ZarrVersions::V3);
                    return Self::new(default_name.into_owned(), data_type);
                }
            }
            Self::new(data_type.identifier().to_string(), data_type)
        }
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

    /// Wrap this data type in an optional type.
    ///
    /// Can be chained to create nested optional types.
    ///
    /// # Examples
    /// ```
    /// # use zarrs::array::{data_type, DataTypeExt};
    /// // Single level optional
    /// let opt_u8 = data_type::uint8().to_named().into_optional();
    /// assert_eq!(opt_u8.identifier(), "zarrs.optional");
    ///
    /// // Nested optional
    /// let opt_opt_u8 = opt_u8.into_optional();
    /// ```
    #[must_use]
    pub fn into_optional(self) -> Self {
        let data_type = data_type::optional(self);
        Self::new_default_name(data_type)
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
        // Delegate to the trait method - each data type implementation handles its own fill value parsing
        self.data_type.fill_value(fill_value)
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
            if data_type::NumpyDateTime64DataType::matches_name(name, ZarrVersions::V3) {
                use crate::metadata_ext::data_type::numpy_datetime64::NumpyDateTime64DataTypeConfigurationV1;
                let NumpyDateTime64DataTypeConfigurationV1 { unit, scale_factor } =
                    NumpyDateTime64DataTypeConfigurationV1::try_from_configuration(
                        configuration.clone(),
                    )
                    .map_err(|_| {
                        PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                            data_type::NumpyDateTime64DataType::IDENTIFIER,
                            "data_type",
                            metadata.to_string(),
                        ))
                    })?;
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::NumpyDateTime64DataType::new(unit, scale_factor)),
                ));
            }

            if data_type::NumpyTimeDelta64DataType::matches_name(name, ZarrVersions::V3) {
                use crate::metadata_ext::data_type::numpy_timedelta64::NumpyTimeDelta64DataTypeConfigurationV1;
                let NumpyTimeDelta64DataTypeConfigurationV1 { unit, scale_factor } =
                    NumpyTimeDelta64DataTypeConfigurationV1::try_from_configuration(
                        configuration.clone(),
                    )
                    .map_err(|_| {
                        PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                            data_type::NumpyTimeDelta64DataType::IDENTIFIER,
                            "data_type",
                            metadata.to_string(),
                        ))
                    })?;
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::NumpyTimeDelta64DataType::new(unit, scale_factor)),
                ));
            }

            if data_type::OptionalDataType::matches_name(name, ZarrVersions::V3) {
                use crate::metadata_ext::data_type::optional::OptionalDataTypeConfigurationV1;
                let OptionalDataTypeConfigurationV1 {
                    name: inner_name,
                    configuration,
                } = OptionalDataTypeConfigurationV1::try_from_configuration(configuration.clone())
                    .map_err(|_| {
                        PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                            data_type::OptionalDataType::IDENTIFIER,
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
                let data_type = data_type::optional(inner_data_type);
                return Ok(Self::new(name.to_string(), data_type));
            }
        }

        // Handle data types with no configuration
        if metadata.configuration_is_none_or_empty() {
            // Boolean
            if data_type::BoolDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::BoolDataType),
                ));
            }

            // Signed integers
            if data_type::Int2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Int2DataType),
                ));
            }
            if data_type::Int4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Int4DataType),
                ));
            }
            if data_type::Int8DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Int8DataType),
                ));
            }
            if data_type::Int16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Int16DataType),
                ));
            }
            if data_type::Int32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Int32DataType),
                ));
            }
            if data_type::Int64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Int64DataType),
                ));
            }

            // Unsigned integers
            if data_type::UInt2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::UInt2DataType),
                ));
            }
            if data_type::UInt4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::UInt4DataType),
                ));
            }
            if data_type::UInt8DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::UInt8DataType),
                ));
            }
            if data_type::UInt16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::UInt16DataType),
                ));
            }
            if data_type::UInt32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::UInt32DataType),
                ));
            }
            if data_type::UInt64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::UInt64DataType),
                ));
            }

            // Subfloats
            if data_type::Float4E2M1FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float4E2M1FNDataType),
                ));
            }
            if data_type::Float6E2M3FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float6E2M3FNDataType),
                ));
            }
            if data_type::Float6E3M2FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float6E3M2FNDataType),
                ));
            }
            if data_type::Float8E3M4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float8E3M4DataType),
                ));
            }
            if data_type::Float8E4M3DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float8E4M3DataType),
                ));
            }
            if data_type::Float8E4M3B11FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float8E4M3B11FNUZDataType),
                ));
            }
            if data_type::Float8E4M3FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float8E4M3FNUZDataType),
                ));
            }
            if data_type::Float8E5M2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float8E5M2DataType),
                ));
            }
            if data_type::Float8E5M2FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float8E5M2FNUZDataType),
                ));
            }
            if data_type::Float8E8M0FNUDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float8E8M0FNUDataType),
                ));
            }

            // Standard floats
            if data_type::BFloat16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::BFloat16DataType),
                ));
            }
            if data_type::Float16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float16DataType),
                ));
            }
            if data_type::Float32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float32DataType),
                ));
            }
            if data_type::Float64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Float64DataType),
                ));
            }

            // Complex subfloats
            if data_type::ComplexFloat4E2M1FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat4E2M1FNDataType),
                ));
            }
            if data_type::ComplexFloat6E2M3FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat6E2M3FNDataType),
                ));
            }
            if data_type::ComplexFloat6E3M2FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat6E3M2FNDataType),
                ));
            }
            if data_type::ComplexFloat8E3M4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat8E3M4DataType),
                ));
            }
            if data_type::ComplexFloat8E4M3DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat8E4M3DataType),
                ));
            }
            if data_type::ComplexFloat8E4M3B11FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat8E4M3B11FNUZDataType),
                ));
            }
            if data_type::ComplexFloat8E4M3FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat8E4M3FNUZDataType),
                ));
            }
            if data_type::ComplexFloat8E5M2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat8E5M2DataType),
                ));
            }
            if data_type::ComplexFloat8E5M2FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat8E5M2FNUZDataType),
                ));
            }
            if data_type::ComplexFloat8E8M0FNUDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat8E8M0FNUDataType),
                ));
            }

            // Complex floats
            if data_type::ComplexBFloat16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexBFloat16DataType),
                ));
            }
            if data_type::ComplexFloat16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat16DataType),
                ));
            }
            if data_type::ComplexFloat32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat32DataType),
                ));
            }
            if data_type::ComplexFloat64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::ComplexFloat64DataType),
                ));
            }
            if data_type::Complex64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Complex64DataType),
                ));
            }
            if data_type::Complex128DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::Complex128DataType),
                ));
            }

            // Variable-length types
            if data_type::StringDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::StringDataType),
                ));
            }
            if data_type::BytesDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(data_type::BytesDataType),
                ));
            }

            // RawBits (r8, r16, r24, etc.)
            if data_type::RawBitsDataType::matches_name(name, ZarrVersions::V3)
                && let Ok(size_bits) = name[1..].parse::<usize>()
            {
                if size_bits % 8 == 0 {
                    let size_bytes = size_bits / 8;
                    return Ok(Self::new(
                        name.to_string(),
                        Arc::new(data_type::RawBitsDataType::new(size_bytes)),
                    ));
                }
                return Err(
                    PluginUnsupportedError::new(name.to_string(), "data type".to_string()).into(),
                );
            }
        }

        // Try an extension plugin
        for plugin in inventory::iter::<DataTypePlugin> {
            if plugin.match_name(name, ZarrVersions::V3) {
                return plugin
                    .create(metadata)
                    .map(|dt| NamedDataType::new(metadata.name().to_string(), dt));
            }
        }

        // The data type is not supported
        Err(PluginUnsupportedError::new(name.to_string(), "data type".to_string()).into())
    }
}
