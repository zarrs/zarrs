//! The `string` data type.

use zarrs_plugin::ExtensionIdentifier;

use super::macros::register_data_type_plugin;

/// The `string` data type.
#[derive(Debug, Clone, Copy)]
pub struct StringDataType;
register_data_type_plugin!(StringDataType);
zarrs_plugin::impl_extension_aliases!(StringDataType, "string",
    v3: "string", [],
    v2: "|O", ["|O"]
);

impl zarrs_data_type::DataTypeExtension for StringDataType {
    fn identifier(&self) -> &'static str {
        <Self as ExtensionIdentifier>::IDENTIFIER
    }

    fn configuration(&self) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Variable
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::v3::FillValueMetadataV3,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        let err = || {
            zarrs_data_type::DataTypeFillValueMetadataError::new(
                self.identifier().to_string(),
                fill_value_metadata.clone(),
            )
        };
        let s = fill_value_metadata.as_str().ok_or_else(err)?;
        Ok(zarrs_data_type::FillValue::from(s.as_bytes().to_vec()))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<zarrs_metadata::v3::FillValueMetadataV3, zarrs_data_type::DataTypeFillValueError>
    {
        let error = || {
            zarrs_data_type::DataTypeFillValueError::new(
                self.identifier().to_string(),
                fill_value.clone(),
            )
        };
        let s = std::str::from_utf8(fill_value.as_ne_bytes()).map_err(|_| error())?;
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(s))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
