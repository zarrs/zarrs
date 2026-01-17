//! The `int4` data type.

use super::macros::register_data_type_plugin;

/// The `int4` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int4DataType;
register_data_type_plugin!(Int4DataType);
zarrs_plugin::impl_extension_aliases!(Int4DataType, v3: "int4");

impl zarrs_data_type::DataTypeTraits for Int4DataType {
    fn configuration(&self, _version: zarrs_plugin::ZarrVersion) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(1)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::FillValueMetadata,
        _version: zarrs_plugin::ZarrVersion,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        let int = fill_value_metadata
            .as_i64()
            .ok_or(zarrs_data_type::DataTypeFillValueMetadataError)?;
        // int4 range: -8 to 7
        if !(-8..=7).contains(&int) {
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

zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits!(Int4DataType, 4, signed, 1);
zarrs_data_type::codec_traits::impl_bytes_data_type_traits!(Int4DataType, 1);
