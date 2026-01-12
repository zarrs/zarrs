//! The `string` data type.

use super::macros::register_data_type_plugin;

/// The `string` data type.
#[derive(Debug, Clone, Copy)]
pub struct StringDataType;
register_data_type_plugin!(StringDataType);
zarrs_plugin::impl_extension_aliases!(StringDataType,
    v3: "string", [],
    v2: "|O", ["|O"]
);

impl zarrs_data_type::DataTypeTraits for StringDataType {
    fn configuration(&self, _version: zarrs_plugin::ZarrVersions) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Variable
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::FillValueMetadata,
        version: zarrs_plugin::ZarrVersions,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        let s = if let Some(s) = fill_value_metadata.as_str() {
            s.to_string()
        } else if matches!(version, zarrs_plugin::ZarrVersions::V2) {
            // V2: null -> empty string, 0 -> empty string (zarr-python compatibility)
            if fill_value_metadata.is_null() || fill_value_metadata.as_u64() == Some(0) {
                String::new()
            } else {
                return Err(zarrs_data_type::DataTypeFillValueMetadataError);
            }
        } else {
            return Err(zarrs_data_type::DataTypeFillValueMetadataError);
        };
        Ok(zarrs_data_type::FillValue::from(s.as_bytes().to_vec()))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<zarrs_metadata::FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
        let s = std::str::from_utf8(fill_value.as_ne_bytes())
            .map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        Ok(zarrs_metadata::FillValueMetadata::from(s))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
