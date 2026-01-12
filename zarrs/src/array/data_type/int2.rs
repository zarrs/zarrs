//! The `int2` data type.

use super::macros::register_data_type_plugin;

/// The `int2` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int2DataType;
register_data_type_plugin!(Int2DataType);
zarrs_plugin::impl_extension_aliases!(Int2DataType, v3: "int2");

impl zarrs_data_type::DataTypeTraits for Int2DataType {
    fn configuration(&self, _version: zarrs_plugin::ZarrVersions) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(1)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::FillValueMetadata,
        _version: zarrs_plugin::ZarrVersions,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        let int = fill_value_metadata
            .as_i64()
            .ok_or(zarrs_data_type::DataTypeFillValueMetadataError)?;
        // int2 range: -2 to 1
        if !(-2..=1).contains(&int) {
            return Err(zarrs_data_type::DataTypeFillValueMetadataError);
        }
        #[expect(clippy::cast_possible_truncation)]
        Ok(zarrs_data_type::FillValue::from(int as i8))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<zarrs_metadata::FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
        let bytes: [u8; 1] = fill_value
            .as_ne_bytes()
            .try_into()
            .map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = i8::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

crate::array::codec::impl_packbits_codec!(Int2DataType, 2, signed, 1);
crate::array::codec::impl_bytes_codec_passthrough!(Int2DataType);
