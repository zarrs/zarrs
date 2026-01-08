use zarrs_data_type::{DataType, DataTypeFillValueMetadataError, FillValue};
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::{Configuration, ConfigurationSerialize};
use zarrs_metadata_ext::data_type::optional::OptionalDataTypeConfigurationV1;
use zarrs_plugin::{ExtensionIdentifier, PluginCreateError, PluginMetadataInvalidError};

use crate::array::NamedDataType;

/// The `optional` data type.
///
/// This wraps the inner [`NamedDataType`] and provides methods specific to optional types,
/// such as checking if a fill value represents null and extracting inner fill value bytes.
///
/// Use the explicit accessor methods (e.g., [`inner()`](Self::inner), [`data_type()`](Self::data_type))
/// to access the wrapped data type's properties.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OptionalDataType(Box<NamedDataType>);

inventory::submit! {
    zarrs_data_type::DataTypePlugin::new(
        <OptionalDataType as ExtensionIdentifier>::IDENTIFIER,
        <OptionalDataType as ExtensionIdentifier>::matches_name,
        <OptionalDataType as ExtensionIdentifier>::default_name,
        |metadata: &MetadataV3| -> Result<DataType, PluginCreateError> {
            let configuration = metadata.configuration().ok_or_else(|| {
                PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                    OptionalDataType::IDENTIFIER,
                    "data_type",
                    "missing configuration".to_string(),
                ))
            })?;
            let config = OptionalDataTypeConfigurationV1::try_from_configuration(configuration.clone())
                .map_err(|_| {
                    PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                        OptionalDataType::IDENTIFIER,
                        "data_type",
                        metadata.to_string(),
                    ))
                })?;

            // Create metadata for the inner data type
            let inner_metadata = if config.configuration.is_empty() {
                MetadataV3::new(config.name)
            } else {
                MetadataV3::new_with_configuration(config.name, config.configuration)
            };

            // Recursively parse the inner data type
            let inner_data_type = NamedDataType::try_from(&inner_metadata)?;
            Ok(std::sync::Arc::new(OptionalDataType::new(inner_data_type)).into())
        },
    )
}
zarrs_plugin::impl_extension_aliases!(OptionalDataType, "optional",
  v3: "zarrs.optional", [],
  v2: "zarrs.optional", []
);

impl OptionalDataType {
    /// Create a new optional data type wrapper.
    #[must_use]
    pub fn new(inner: NamedDataType) -> Self {
        Self(Box::new(inner))
    }

    /// Check if the fill value represents null (last byte is `0x00`).
    #[must_use]
    pub fn is_fill_value_null(&self, fill_value: &FillValue) -> bool {
        fill_value.as_ne_bytes().last() == Some(&0)
    }

    /// Get the inner fill value bytes (without optional suffix).
    ///
    /// For optional data types, returns all bytes except the last suffix byte.
    #[must_use]
    pub fn fill_value_inner_bytes<'a>(&self, fill_value: &'a FillValue) -> &'a [u8] {
        let bytes = fill_value.as_ne_bytes();
        if bytes.is_empty() {
            &[]
        } else {
            &bytes[..bytes.len() - 1]
        }
    }

    /// Returns the configuration for this optional data type.
    #[must_use]
    pub fn configuration(&self) -> Configuration {
        Configuration::from(OptionalDataTypeConfigurationV1 {
            name: self.0.name().to_string(),
            configuration: self.0.configuration(),
        })
    }
}

impl OptionalDataType {
    /// Returns a reference to the inner named data type.
    #[must_use]
    pub fn inner(&self) -> &NamedDataType {
        &self.0
    }

    /// Returns the inner named data type, consuming self.
    #[must_use]
    pub fn into_inner(self) -> NamedDataType {
        *self.0
    }

    // Delegate methods from NamedDataType to avoid needing Deref

    /// The name of the inner data type.
    #[must_use]
    pub fn name(&self) -> &str {
        self.0.name()
    }

    /// The underlying inner data type.
    #[must_use]
    pub fn data_type(&self) -> &crate::array::DataType {
        self.0.data_type()
    }

    /// The size of the inner data type.
    #[must_use]
    pub fn inner_size(&self) -> zarrs_metadata::DataTypeSize {
        self.0.size()
    }

    /// Returns true if the inner data type is fixed size.
    #[must_use]
    pub fn is_fixed(&self) -> bool {
        use crate::array::data_type::DataTypeExt;
        self.0.data_type().is_fixed()
    }

    /// Returns the fixed size of the inner data type if it's fixed size.
    #[must_use]
    pub fn fixed_size(&self) -> Option<usize> {
        use crate::array::data_type::DataTypeExt;
        self.0.data_type().fixed_size()
    }

    /// Create a fill value from metadata for the inner data type.
    ///
    /// # Errors
    /// Returns [`DataTypeFillValueMetadataError`] if the fill value is incompatible with the inner data type.
    pub fn fill_value_from_metadata(
        &self,
        fill_value: &zarrs_metadata::v3::FillValueMetadataV3,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        self.0.fill_value_from_metadata(fill_value)
    }
}

// DataTypeTraits implementation for OptionalDataType
impl zarrs_data_type::DataTypeTraits for OptionalDataType {
    fn identifier(&self) -> &'static str {
        <Self as ExtensionIdentifier>::IDENTIFIER
    }

    fn configuration(&self) -> Configuration {
        // Use the existing configuration() method
        OptionalDataType::configuration(self)
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        // Optional type size is the inner type's size
        // The mask/suffix byte is stored separately in the data representation
        self.inner_size()
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::v3::FillValueMetadataV3,
    ) -> Result<FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        let err = || {
            zarrs_data_type::DataTypeFillValueMetadataError::new(
                self.identifier().to_string(),
                fill_value_metadata.clone(),
            )
        };

        if fill_value_metadata.is_null() {
            // Null fill value for optional type: single 0x00 byte
            Ok(FillValue::new_optional_null())
        } else if let Some([inner]) = fill_value_metadata.as_array() {
            // Wrapped value [inner_metadata] -> Some(inner)
            let inner_fv = self.0.fill_value_from_metadata(inner)?;
            Ok(inner_fv.into_optional())
        } else {
            Err(err())
        }
    }

    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<zarrs_metadata::v3::FillValueMetadataV3, zarrs_data_type::DataTypeFillValueError>
    {
        if self.is_fill_value_null(fill_value) {
            Ok(zarrs_metadata::v3::FillValueMetadataV3::Null)
        } else {
            // Extract inner bytes and convert to metadata
            let inner_bytes = self.fill_value_inner_bytes(fill_value);
            let inner_fv = FillValue::new(inner_bytes.to_vec());
            let inner_metadata = self.0.data_type().metadata_fill_value(&inner_fv)?;
            Ok(zarrs_metadata::v3::FillValueMetadataV3::from(vec![
                inner_metadata,
            ]))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
