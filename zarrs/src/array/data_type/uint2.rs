//! The `uint2` data type.

use super::macros::{impl_bytes_codec_passthrough, impl_packbits_codec, register_data_type_plugin};

/// The `uint2` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt2DataType;
register_data_type_plugin!(UInt2DataType);
zarrs_plugin::impl_extension_aliases!(UInt2DataType, "uint2");

impl zarrs_data_type::DataTypeExtension for UInt2DataType {
    fn identifier(&self) -> &'static str {
        <Self as zarrs_plugin::ExtensionIdentifier>::IDENTIFIER
    }

    fn configuration(&self) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(1)
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
        let int = fill_value_metadata.as_u64().ok_or_else(err)?;
        // uint2 range: 0 to 3
        if int > 3 {
            return Err(err());
        }
        #[expect(clippy::cast_possible_truncation)]
        Ok(zarrs_data_type::FillValue::from(int as u8))
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
        let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = u8::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl_packbits_codec!(UInt2DataType, 2, unsigned, 1);
impl_bytes_codec_passthrough!(UInt2DataType);
