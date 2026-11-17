use std::borrow::Cow;

use crate::{DataType, DataTypeFillValueMetadataError, FillValue, OptionalDataTypeTraits};
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::{Configuration, ConfigurationSerialize};
use zarrs_plugin::{ExtensionName, PluginCreateError, ZarrVersion};

/// The `optional` data type configuration (version 1.0).
///
/// Matches `zarrs_metadata_ext::data_type::optional::OptionalDataTypeConfigurationV1`.
#[derive(
    Clone, Eq, PartialEq, Debug, serde::Serialize, serde::Deserialize, derive_more::Display,
)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
struct OptionalDataTypeConfigurationV1 {
    /// The inner data type name.
    pub name: String,
    /// The configuration of the inner data type.
    ///
    /// Optional in metadata, defaulting to an empty configuration.
    #[serde(default)]
    pub configuration: Configuration,
}

impl ConfigurationSerialize for OptionalDataTypeConfigurationV1 {}

/// The `optional` data type.
///
/// This wraps the inner [`DataType`] and provides methods specific to optional types,
/// such as checking if a fill value represents null and extracting inner fill value bytes.
///
/// Use the explicit accessor methods (e.g., [`inner()`](Self::inner), [`data_type()`](Self::data_type))
/// to access the wrapped data type's properties.
#[derive(Clone, Debug)]
pub struct OptionalDataType(DataType);

impl PartialEq for OptionalDataType {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for OptionalDataType {}

zarrs_plugin::impl_extension_aliases!(OptionalDataType,
  v3: "zarrs.optional", [],
  v2: "zarrs.optional", []
);

impl crate::DataTypeTraitsV3 for OptionalDataType {
    fn create(metadata: &MetadataV3) -> Result<DataType, PluginCreateError> {
        let config: OptionalDataTypeConfigurationV1 = metadata.to_typed_configuration()?;

        // Create metadata for the inner data type
        let inner_metadata = if config.configuration.is_empty() {
            MetadataV3::new(config.name)
        } else {
            MetadataV3::new_with_configuration(config.name, config.configuration)
        };

        // Recursively parse the inner data type
        let inner_data_type = DataType::from_metadata(&inner_metadata)?;
        Ok(std::sync::Arc::new(OptionalDataType::new(inner_data_type)).into())
    }
}

// Register as V3-only data type.
inventory::submit! {
    crate::DataTypePluginV3::new::<OptionalDataType>()
}

impl OptionalDataType {
    /// Create a new optional data type wrapper.
    #[must_use]
    pub fn new(inner: DataType) -> Self {
        Self(inner)
    }

    /// Check if the fill value represents null (last byte is `0x00`).
    #[must_use]
    pub fn is_fill_value_null(&self, fill_value: &FillValue) -> bool {
        OptionalDataTypeTraits::is_fill_value_null(self, fill_value)
    }

    /// Get the inner fill value bytes (without optional suffix).
    ///
    /// For optional data types, returns all bytes except the last suffix byte.
    #[must_use]
    pub fn fill_value_inner_bytes<'a>(&self, fill_value: &'a FillValue) -> &'a [u8] {
        OptionalDataTypeTraits::fill_value_inner_bytes(self, fill_value)
    }

    /// Returns the configuration for this optional data type.
    #[must_use]
    pub fn configuration(&self, version: ZarrVersion) -> Configuration {
        let inner_name = self
            .0
            .name(version)
            .map_or_else(String::new, Cow::into_owned);
        Configuration::from(OptionalDataTypeConfigurationV1 {
            name: inner_name,
            configuration: self.0.configuration(version),
        })
    }
}

impl OptionalDataType {
    /// Returns a reference to the inner data type.
    #[must_use]
    pub fn inner(&self) -> &DataType {
        &self.0
    }

    /// Returns the inner data type, consuming self.
    #[must_use]
    pub fn into_inner(self) -> DataType {
        self.0
    }

    // Delegate methods from DataType to avoid needing Deref

    /// The underlying inner data type.
    #[must_use]
    pub fn data_type(&self) -> &crate::DataType {
        &self.0
    }

    /// Create a fill value from metadata for the inner data type.
    ///
    /// # Errors
    /// Returns [`DataTypeFillValueMetadataError`] if the fill value is incompatible with the inner data type.
    pub fn fill_value_from_metadata(
        &self,
        fill_value: &zarrs_metadata::FillValueMetadata,
        version: zarrs_plugin::ZarrVersion,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        self.0.fill_value(fill_value, version)
    }
}

// DataTypeTraits implementation for OptionalDataType
impl crate::DataTypeTraits for OptionalDataType {
    fn into_data_type(self: std::sync::Arc<Self>) -> DataType {
        DataType::Optional(self)
    }

    fn configuration(&self, version: ZarrVersion) -> Configuration {
        // Use the existing configuration() method
        OptionalDataType::configuration(self, version)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::FillValueMetadata,
        version: zarrs_plugin::ZarrVersion,
    ) -> Result<FillValue, crate::DataTypeFillValueMetadataError> {
        if fill_value_metadata.is_null() {
            // Null fill value for optional type: single 0x00 byte
            Ok(FillValue::new_optional_null())
        } else if let Some([inner_fv_metadata]) = fill_value_metadata.as_array() {
            // Wrapped value [inner_metadata] -> Some(inner)
            // Propagate the same version to the inner fill value
            let inner_fv = self.0.fill_value(inner_fv_metadata, version)?;
            Ok(inner_fv.into_optional())
        } else {
            Err(DataTypeFillValueMetadataError)
        }
    }

    /// The fill value must be `[0]` (null) or a valid fill value of the inner data type with a `1` suffix.
    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<zarrs_metadata::FillValueMetadata, crate::DataTypeFillValueError> {
        match fill_value.as_ne_bytes().split_last() {
            Some((0, [])) => Ok(zarrs_metadata::FillValueMetadata::Null),
            Some((1, inner_bytes)) => {
                let inner_fv = FillValue::new(inner_bytes.to_vec());
                let inner_metadata = self.0.metadata_fill_value(&inner_fv)?;
                Ok(zarrs_metadata::FillValueMetadata::from(vec![
                    inner_metadata,
                ]))
            }
            _ => Err(crate::DataTypeFillValueError),
        }
    }

    fn eq(&self, other: &dyn crate::DataTypeTraits) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|other| self == other)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl OptionalDataTypeTraits for OptionalDataType {
    fn data_type(&self) -> &DataType {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::borrow::Cow;

    use zarrs_metadata::FillValueMetadata;

    use super::*;
    use crate::{
        DataTypeFillValueError, DataTypeTraits, FixedDataTypeTraits, VariableDataTypeTraits,
    };

    #[derive(Debug)]
    struct UnnamedA;
    #[derive(Debug)]
    struct UnnamedB;
    #[derive(Debug)]
    struct TaggedEq(u8);

    impl ExtensionName for UnnamedA {
        fn name(&self, _version: ZarrVersion) -> Option<Cow<'static, str>> {
            None
        }
    }

    impl ExtensionName for UnnamedB {
        fn name(&self, _version: ZarrVersion) -> Option<Cow<'static, str>> {
            None
        }
    }

    impl ExtensionName for TaggedEq {
        fn name(&self, _version: ZarrVersion) -> Option<Cow<'static, str>> {
            Some(Cow::Borrowed("test.tagged_eq"))
        }
    }

    impl FixedDataTypeTraits for UnnamedA {
        fn size(&self) -> usize {
            1
        }
    }

    impl VariableDataTypeTraits for UnnamedB {}

    impl FixedDataTypeTraits for TaggedEq {
        fn size(&self) -> usize {
            1
        }
    }

    impl DataTypeTraits for UnnamedA {
        fn into_data_type(self: std::sync::Arc<Self>) -> DataType {
            DataType::Fixed(self)
        }

        fn configuration(&self, _version: ZarrVersion) -> Configuration {
            Configuration::default()
        }

        fn fill_value(
            &self,
            _metadata: &FillValueMetadata,
            _version: ZarrVersion,
        ) -> Result<FillValue, DataTypeFillValueMetadataError> {
            Ok(FillValue::from(0u8))
        }

        fn metadata_fill_value(
            &self,
            _fill_value: &FillValue,
        ) -> Result<FillValueMetadata, DataTypeFillValueError> {
            Ok(FillValueMetadata::from(0))
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl DataTypeTraits for UnnamedB {
        fn into_data_type(self: std::sync::Arc<Self>) -> DataType {
            DataType::Variable(self)
        }

        fn configuration(&self, _version: ZarrVersion) -> Configuration {
            Configuration::default()
        }

        fn fill_value(
            &self,
            _metadata: &FillValueMetadata,
            _version: ZarrVersion,
        ) -> Result<FillValue, DataTypeFillValueMetadataError> {
            Ok(FillValue::new(vec![]))
        }

        fn metadata_fill_value(
            &self,
            _fill_value: &FillValue,
        ) -> Result<FillValueMetadata, DataTypeFillValueError> {
            Ok(FillValueMetadata::Null)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl DataTypeTraits for TaggedEq {
        fn into_data_type(self: std::sync::Arc<Self>) -> DataType {
            DataType::Fixed(self)
        }

        fn configuration(&self, _version: ZarrVersion) -> Configuration {
            Configuration::default()
        }

        fn fill_value(
            &self,
            _metadata: &FillValueMetadata,
            _version: ZarrVersion,
        ) -> Result<FillValue, DataTypeFillValueMetadataError> {
            Ok(FillValue::from(0u8))
        }

        fn metadata_fill_value(
            &self,
            _fill_value: &FillValue,
        ) -> Result<FillValueMetadata, DataTypeFillValueError> {
            Ok(FillValueMetadata::from(0))
        }

        fn eq(&self, other: &dyn DataTypeTraits) -> bool {
            other
                .as_any()
                .downcast_ref::<Self>()
                .is_some_and(|other| self.0 == other.0)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn optional_data_type_equality_delegates_to_inner_type() {
        let a = DataType::new(UnnamedA);
        let b = DataType::new(UnnamedB);
        assert_ne!(a, b);
        assert_ne!(a.to_optional(), b.to_optional());
        assert_ne!(a.to_optional().to_optional(), b.to_optional().to_optional());
        assert_eq!(a.to_optional(), DataType::new(UnnamedA).to_optional());

        let t1 = DataType::new(TaggedEq(1));
        let t2 = DataType::new(TaggedEq(2));
        assert_ne!(t1, t2);
        assert_ne!(t1.to_optional(), t2.to_optional());
        assert_eq!(t1.to_optional(), DataType::new(TaggedEq(1)).to_optional());
    }
}
