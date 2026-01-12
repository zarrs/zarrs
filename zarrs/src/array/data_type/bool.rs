//! The `bool` data type.

use zarrs_metadata::FillValueMetadata;

use super::macros::register_data_type_plugin;

/// The `bool` data type.
#[derive(Debug, Clone, Copy)]
pub struct BoolDataType;
register_data_type_plugin!(BoolDataType);
zarrs_plugin::impl_extension_aliases!(BoolDataType,
    v3: "bool", [],
    v2: "|b1", ["|b1"]
);

impl zarrs_data_type::DataTypeTraits for BoolDataType {
    fn configuration(&self, _version: zarrs_plugin::ZarrVersion) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(1)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &FillValueMetadata,
        version: zarrs_plugin::ZarrVersion,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        // V2 compatibility: 0/1 integers are accepted as false/true, null -> false
        if let Some(b) = fill_value_metadata.as_bool() {
            Ok(zarrs_data_type::FillValue::from(b))
        } else if matches!(version, zarrs_plugin::ZarrVersion::V2) {
            // V2: accept 0/1 as false/true, null -> false
            if fill_value_metadata.is_null() {
                Ok(zarrs_data_type::FillValue::from(false))
            } else {
                match fill_value_metadata.as_u64() {
                    Some(0) => Ok(zarrs_data_type::FillValue::from(false)),
                    Some(1) => Ok(zarrs_data_type::FillValue::from(true)),
                    _ => Err(zarrs_data_type::DataTypeFillValueMetadataError),
                }
            }
        } else {
            Err(zarrs_data_type::DataTypeFillValueMetadataError)
        }
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
        let bytes: [u8; 1] = fill_value
            .as_ne_bytes()
            .try_into()
            .map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        Ok(FillValueMetadata::from(bytes[0] != 0))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

crate::array::codec::impl_bytes_codec_passthrough!(BoolDataType);
crate::array::codec::impl_packbits_codec!(BoolDataType, 1, unsigned, 1);
