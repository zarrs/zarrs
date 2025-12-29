//! The `uint4` data type.

use super::macros::{impl_bytes_codec_passthrough, impl_packbits_codec, register_data_type_plugin};

/// The `uint4` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt4DataType;
register_data_type_plugin!(UInt4DataType);
zarrs_plugin::impl_extension_aliases!(UInt4DataType, "uint4");

impl zarrs_data_type::DataTypeExtension for UInt4DataType {
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
        // uint4 range: 0 to 15
        if int > 15 {
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

    fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
        Some(self)
    }

    fn codec_packbits(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionPackBitsCodec> {
        Some(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl_packbits_codec!(UInt4DataType, 4, unsigned, 1);
impl_bytes_codec_passthrough!(UInt4DataType);
