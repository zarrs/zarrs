//! The `bytes` data type.

use zarrs_plugin::ExtensionIdentifier;

use super::macros::register_data_type_plugin;

/// The `bytes` data type.
#[derive(Debug, Clone, Copy)]
pub struct BytesDataType;
zarrs_plugin::impl_extension_aliases!(BytesDataType, "bytes",
    v3: "bytes", ["binary", "variable_length_bytes"],
    v2: "|VX", ["|VX"]
);
register_data_type_plugin!(BytesDataType);

impl zarrs_data_type::DataTypeExtension for BytesDataType {
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
        use base64::{Engine, prelude::BASE64_STANDARD};
        let err = || {
            zarrs_data_type::DataTypeFillValueMetadataError::new(
                self.identifier().to_string(),
                fill_value_metadata.clone(),
            )
        };
        // Bytes fill value can be base64-encoded string or array of bytes
        if let Some(s) = fill_value_metadata.as_str() {
            let bytes = BASE64_STANDARD.decode(s).map_err(|_| err())?;
            Ok(zarrs_data_type::FillValue::from(bytes))
        } else if let Some(arr) = fill_value_metadata.as_array() {
            let bytes: Result<Vec<u8>, _> = arr
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|u| u8::try_from(u).ok())
                        .ok_or_else(err)
                })
                .collect();
            Ok(zarrs_data_type::FillValue::from(bytes?))
        } else {
            Err(err())
        }
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<zarrs_metadata::v3::FillValueMetadataV3, zarrs_data_type::DataTypeFillValueError>
    {
        // Return as array of bytes for consistency
        // Note: base64 encoding may be preferred per zarr spec - see comments in test
        let bytes = fill_value.as_ne_bytes();
        let arr: Vec<zarrs_metadata::v3::FillValueMetadataV3> = bytes
            .iter()
            .map(|&b| zarrs_metadata::v3::FillValueMetadataV3::from(b))
            .collect();
        Ok(zarrs_metadata::v3::FillValueMetadataV3::Array(arr))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
