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

use crate::array::{
    DataType, DataTypeExt,
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
        UInt4DataType, UInt8DataType, UInt16DataType, UInt32DataType, UInt64DataType, data_types,
    },
};

/// A named data type.
#[derive(Debug, Clone)]
pub struct NamedDataType {
    name: String,
    data_type: DataType,
}

impl PartialEq for NamedDataType {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.data_type.data_type_eq(other.data_type.as_ref())
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

    /// Wrap this data type in an optional type.
    ///
    /// Can be chained to create nested optional types.
    ///
    /// # Examples
    /// ```
    /// # use zarrs::array::{data_types, DataTypeExt};
    /// // Single level optional
    /// let opt_u8 = data_types::uint8().to_named().into_optional();
    /// assert_eq!(opt_u8.identifier(), "zarrs.optional");
    ///
    /// // Nested optional
    /// let opt_opt_u8 = opt_u8.into_optional();
    /// ```
    #[must_use]
    pub fn into_optional(self) -> Self {
        let data_type = data_types::optional(self);
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
                    Arc::new(NumpyDateTime64DataType::new(unit, scale_factor)),
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
                    Arc::new(NumpyTimeDelta64DataType::new(unit, scale_factor)),
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
                let data_type = data_types::optional(inner_data_type);
                return Ok(Self::new(name.to_string(), data_type));
            }
        }

        // Handle data types with no configuration
        if metadata.configuration_is_none_or_empty() {
            // Boolean
            if BoolDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(BoolDataType)));
            }

            // Signed integers
            if Int2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Int2DataType)));
            }
            if Int4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Int4DataType)));
            }
            if Int8DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Int8DataType)));
            }
            if Int16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Int16DataType)));
            }
            if Int32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Int32DataType)));
            }
            if Int64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Int64DataType)));
            }

            // Unsigned integers
            if UInt2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(UInt2DataType)));
            }
            if UInt4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(UInt4DataType)));
            }
            if UInt8DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(UInt8DataType)));
            }
            if UInt16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(UInt16DataType)));
            }
            if UInt32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(UInt32DataType)));
            }
            if UInt64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(UInt64DataType)));
            }

            // Subfloats
            if Float4E2M1FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Float4E2M1FNDataType)));
            }
            if Float6E2M3FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Float6E2M3FNDataType)));
            }
            if Float6E3M2FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Float6E3M2FNDataType)));
            }
            if Float8E3M4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Float8E3M4DataType)));
            }
            if Float8E4M3DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Float8E4M3DataType)));
            }
            if Float8E4M3B11FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(Float8E4M3B11FNUZDataType),
                ));
            }
            if Float8E4M3FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(Float8E4M3FNUZDataType),
                ));
            }
            if Float8E5M2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Float8E5M2DataType)));
            }
            if Float8E5M2FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(Float8E5M2FNUZDataType),
                ));
            }
            if Float8E8M0FNUDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Float8E8M0FNUDataType)));
            }

            // Standard floats
            if BFloat16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(BFloat16DataType)));
            }
            if Float16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Float16DataType)));
            }
            if Float32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Float32DataType)));
            }
            if Float64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Float64DataType)));
            }

            // Complex subfloats
            if ComplexFloat4E2M1FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat4E2M1FNDataType),
                ));
            }
            if ComplexFloat6E2M3FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat6E2M3FNDataType),
                ));
            }
            if ComplexFloat6E3M2FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat6E3M2FNDataType),
                ));
            }
            if ComplexFloat8E3M4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat8E3M4DataType),
                ));
            }
            if ComplexFloat8E4M3DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat8E4M3DataType),
                ));
            }
            if ComplexFloat8E4M3B11FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat8E4M3B11FNUZDataType),
                ));
            }
            if ComplexFloat8E4M3FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat8E4M3FNUZDataType),
                ));
            }
            if ComplexFloat8E5M2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat8E5M2DataType),
                ));
            }
            if ComplexFloat8E5M2FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat8E5M2FNUZDataType),
                ));
            }
            if ComplexFloat8E8M0FNUDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat8E8M0FNUDataType),
                ));
            }

            // Complex floats
            if ComplexBFloat16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexBFloat16DataType),
                ));
            }
            if ComplexFloat16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat16DataType),
                ));
            }
            if ComplexFloat32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat32DataType),
                ));
            }
            if ComplexFloat64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    Arc::new(ComplexFloat64DataType),
                ));
            }
            if Complex64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Complex64DataType)));
            }
            if Complex128DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(Complex128DataType)));
            }

            // Variable-length types
            if StringDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(StringDataType)));
            }
            if BytesDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), Arc::new(BytesDataType)));
            }

            // RawBits (r8, r16, r24, etc.)
            if RawBitsDataType::matches_name(name, ZarrVersions::V3)
                && let Ok(size_bits) = name[1..].parse::<usize>()
            {
                if size_bits % 8 == 0 {
                    let size_bytes = size_bits / 8;
                    return Ok(Self::new(
                        name.to_string(),
                        Arc::new(RawBitsDataType::new(size_bytes)),
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
