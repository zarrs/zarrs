//! The `uint4` data type.

use zarrs_data_type::DataTypeFillValueMetadataError;

use super::macros::register_data_type_plugin;

/// The `uint4` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt4DataType;
register_data_type_plugin!(UInt4DataType);
zarrs_plugin::impl_extension_aliases!(UInt4DataType, v3: "uint4");

impl zarrs_data_type::DataTypeTraits for UInt4DataType {
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
    ) -> Result<zarrs_data_type::FillValue, DataTypeFillValueMetadataError> {
        let int = fill_value_metadata
            .as_u64()
            .ok_or(DataTypeFillValueMetadataError)?;
        // uint4 range: 0 to 15
        if int > 15 {
            return Err(DataTypeFillValueMetadataError);
        }
        #[expect(clippy::cast_possible_truncation)]
        Ok(zarrs_data_type::FillValue::from(int as u8))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<zarrs_metadata::FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
        let bytes: [u8; 1] = fill_value
            .as_ne_bytes()
            .try_into()
            .map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = u8::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

crate::array::codec::impl_packbits_codec!(UInt4DataType, 4, unsigned, 1);
crate::array::codec::impl_bytes_codec_passthrough!(UInt4DataType);
