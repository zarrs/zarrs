//! The `r*` (raw bits) data type.

use super::macros::impl_bytes_codec_passthrough;
use zarrs_plugin::{ExtensionIdentifier, PluginCreateError, PluginMetadataInvalidError, Regex};

/// The `r*` data type.
///
/// The size is stored as the number of bytes (e.g., `r8` = 1 byte, `r16` = 2 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawBitsDataType {
    /// Size in bytes.
    size_bytes: usize,
}
zarrs_plugin::impl_extension_aliases!(RawBitsDataType, "r*",
    v3: "r*", [], [Regex::new(r"^r\d+$").unwrap()],
    v2: "r*", [], [Regex::new(r"^r\d+$").unwrap(), Regex::new(r"^\|V\d+$").unwrap()]
);

impl RawBitsDataType {
    /// Static instance for use in trait implementations (defaults to 1 byte).
    pub const STATIC: Self = Self { size_bytes: 1 };

    /// Create a new `RawBitsDataType` with the given size in bytes.
    #[must_use]
    pub const fn new(size_bytes: usize) -> Self {
        Self { size_bytes }
    }

    /// Returns the size in bytes.
    #[must_use]
    pub const fn size_bytes(&self) -> usize {
        self.size_bytes
    }
}

impl zarrs_data_type::DataTypeExtension for RawBitsDataType {
    fn identifier(&self) -> &'static str {
        <Self as ExtensionIdentifier>::IDENTIFIER
    }

    fn metadata_name(
        &self,
        _zarr_version: zarrs_plugin::ZarrVersions,
    ) -> std::borrow::Cow<'static, str> {
        // Return "r{bits}" where bits = size_bytes * 8
        std::borrow::Cow::Owned(format!("r{}", self.size_bytes * 8))
    }

    fn configuration(&self) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(self.size_bytes)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::v3::FillValueMetadataV3,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        use base64::{Engine, prelude::BASE64_STANDARD};
        // Use metadata_name for better error messages (e.g., "r16" instead of "r*")
        let name = self.metadata_name(zarrs_plugin::ZarrVersions::V3);
        let err = || {
            zarrs_data_type::DataTypeFillValueMetadataError::new(
                name.to_string(),
                fill_value_metadata.clone(),
            )
        };
        // RawBits fill value can be base64-encoded string or array of bytes
        if let Some(s) = fill_value_metadata.as_str() {
            let bytes = BASE64_STANDARD.decode(s).map_err(|_| err())?;
            if bytes.len() != self.size_bytes {
                return Err(err());
            }
            Ok(zarrs_data_type::FillValue::from(bytes))
        } else if let Some(arr) = fill_value_metadata.as_array() {
            if arr.len() != self.size_bytes {
                return Err(err());
            }
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
        // Return as array of bytes (not base64 encoded) for consistency
        let bytes = fill_value.as_ne_bytes();
        let arr: Vec<zarrs_metadata::v3::FillValueMetadataV3> = bytes
            .iter()
            .map(|&b| zarrs_metadata::v3::FillValueMetadataV3::from(b))
            .collect();
        Ok(zarrs_metadata::v3::FillValueMetadataV3::Array(arr))
    }

    fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
        Some(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl_bytes_codec_passthrough!(RawBitsDataType);

// Custom plugin registration for RawBitsDataType (size parsed from name)
inventory::submit! {
    zarrs_data_type::DataTypePlugin::new(
        <RawBitsDataType as ExtensionIdentifier>::IDENTIFIER,
        <RawBitsDataType as ExtensionIdentifier>::matches_name,
        <RawBitsDataType as ExtensionIdentifier>::default_name,
        |metadata: &zarrs_metadata::v3::MetadataV3| -> Result<std::sync::Arc<dyn zarrs_data_type::DataTypeExtension>, PluginCreateError> {
            let name = metadata.name();
            // Parse size from name (e.g., "r8" -> 1 byte, "r16" -> 2 bytes)
            // Also handle V2 format like "|V2"
            let size_bits = if let Some(stripped) = name.strip_prefix('r') {
                stripped.parse::<usize>().map_err(|_| {
                    PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                        RawBitsDataType::IDENTIFIER,
                        "data_type",
                        format!("invalid raw bits name: {name}"),
                    ))
                })?
            } else if let Some(stripped) = name.strip_prefix("|V") {
                // V2 format: |V{bytes}
                let size_bytes = stripped.parse::<usize>().map_err(|_| {
                    PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                        RawBitsDataType::IDENTIFIER,
                        "data_type",
                        format!("invalid raw bits name: {name}"),
                    ))
                })?;
                size_bytes * 8
            } else {
                return Err(PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                    RawBitsDataType::IDENTIFIER,
                    "data_type",
                    format!("invalid raw bits name: {name}"),
                )));
            };
            if size_bits % 8 != 0 {
                return Err(PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                    RawBitsDataType::IDENTIFIER,
                    "data_type",
                    format!("raw bits size must be a multiple of 8: {size_bits}"),
                )));
            }
            let size_bytes = size_bits / 8;
            Ok(std::sync::Arc::new(RawBitsDataType::new(size_bytes)))
        },
    )
}
