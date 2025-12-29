//! The `bool` data type.

use zarrs_plugin::ExtensionIdentifier;

use super::macros::{impl_bytes_codec_passthrough, impl_packbits_codec, register_data_type_plugin};

/// The `bool` data type.
#[derive(Debug, Clone, Copy)]
pub struct BoolDataType;
zarrs_plugin::impl_extension_aliases!(BoolDataType, "bool",
    v3: "bool", [],
    v2: "|b1", ["|b1"]
);
register_data_type_plugin!(BoolDataType);

impl zarrs_data_type::DataTypeExtension for BoolDataType {
    fn identifier(&self) -> &'static str {
        <Self as ExtensionIdentifier>::IDENTIFIER
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
        let b = fill_value_metadata.as_bool().ok_or_else(err)?;
        Ok(zarrs_data_type::FillValue::from(u8::from(b)))
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
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(bytes[0] != 0))
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

impl_bytes_codec_passthrough!(BoolDataType);
impl_packbits_codec!(BoolDataType, 1, unsigned, 1);
