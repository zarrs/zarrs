//! The `bytes` data type.

use zarrs_metadata::FillValueMetadata;

use super::macros::register_data_type_plugin;

/// The `bytes` data type.
#[derive(Debug, Clone, Copy)]
pub struct BytesDataType;
register_data_type_plugin!(BytesDataType);
zarrs_plugin::impl_extension_aliases!(BytesDataType,
    v3: "bytes", ["binary", "variable_length_bytes"],
    v2: "|VX", ["|VX"]
);

impl zarrs_data_type::DataTypeTraits for BytesDataType {
    fn configuration(&self, _version: zarrs_plugin::ZarrVersion) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Variable
    }

    fn fill_value(
        &self,
        fill_value_metadata: &FillValueMetadata,
        version: zarrs_plugin::ZarrVersion,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        use base64::Engine;
        use base64::prelude::BASE64_STANDARD;
        // Bytes fill value can be base64-encoded string or array of bytes
        if let Some(s) = fill_value_metadata.as_str() {
            let bytes = BASE64_STANDARD
                .decode(s)
                .map_err(|_| zarrs_data_type::DataTypeFillValueMetadataError)?;
            Ok(zarrs_data_type::FillValue::from(bytes))
        } else if let Some(arr) = fill_value_metadata.as_array() {
            let bytes: Result<Vec<u8>, _> = arr
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|u| u8::try_from(u).ok())
                        .ok_or(zarrs_data_type::DataTypeFillValueMetadataError)
                })
                .collect();
            Ok(zarrs_data_type::FillValue::from(bytes?))
        } else if matches!(version, zarrs_plugin::ZarrVersion::V2) && fill_value_metadata.is_null()
        {
            // V2: null -> empty bytes
            Ok(zarrs_data_type::FillValue::from(Vec::<u8>::new()))
        } else {
            Err(zarrs_data_type::DataTypeFillValueMetadataError)
        }
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
        // Return as array of bytes for consistency
        // Note: base64 encoding may be preferred per zarr spec - see comments in test
        let bytes = fill_value.as_ne_bytes();
        let arr: Vec<FillValueMetadata> =
            bytes.iter().map(|&b| FillValueMetadata::from(b)).collect();
        Ok(FillValueMetadata::Array(arr))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn compatible_element_types(&self) -> &'static [std::any::TypeId] {
        const TYPES: [std::any::TypeId; 2] = [
            std::any::TypeId::of::<Vec<u8>>(),
            std::any::TypeId::of::<&[u8]>(),
        ];
        &TYPES
    }
}
