//! The `r*` (raw bits) data type.

use std::borrow::Cow;

use zarrs_data_type::DataType;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::{
    ExtensionAliases, PluginConfigurationInvalidError, PluginCreateError, Regex, ZarrVersions,
};

/// The `r*` data type.
///
/// The size is stored as the number of bytes (e.g., `r8` = 1 byte, `r16` = 2 bytes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawBitsDataType {
    /// Size in bytes.
    size_bytes: usize,
}

// Manual implementations instead of impl_extension_aliases! because RawBitsDataType
// needs instance-specific names based on size_bytes (e.g., "r16", "r32", "|V2")
use std::sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
use zarrs_plugin::{ExtensionAliasesConfig, ZarrVersion2, ZarrVersion3};

// Register V3 plugin.
inventory::submit! {
    zarrs_data_type::DataTypePluginV3::new::<RawBitsDataType>(create_rawbits_datatype_v3)
}

// Register V2 plugin.
inventory::submit! {
    zarrs_data_type::DataTypePluginV2::new::<RawBitsDataType>(create_rawbits_datatype_v2)
}

fn create_rawbits_datatype_v3(metadata: &MetadataV3) -> Result<DataType, PluginCreateError> {
    let name = metadata.name();
    // Parse size from name (e.g., "r8" -> 1 byte, "r16" -> 2 bytes)
    let size_bytes = if let Some(stripped) = name.strip_prefix('r') {
        let size_bits = stripped
            .parse::<usize>()
            .map_err(|_| PluginCreateError::NameInvalid {
                name: name.to_string(),
            })?;
        if size_bits % 8 != 0 {
            return Err(PluginConfigurationInvalidError::new(format!(
                "raw bits size must be a multiple of 8: {size_bits}"
            ))
            .into());
        }
        size_bits / 8
    } else {
        return Err(PluginCreateError::NameInvalid {
            name: name.to_string(),
        });
    };
    Ok(std::sync::Arc::new(RawBitsDataType::new(size_bytes)).into())
}

fn create_rawbits_datatype_v2(
    metadata: &zarrs_metadata::v2::DataTypeMetadataV2,
) -> Result<DataType, PluginCreateError> {
    let size_bytes = match metadata {
        zarrs_metadata::v2::DataTypeMetadataV2::Simple(name) => {
            if let Some(stripped) = name.strip_prefix("|V") {
                // V2 format: |V{bytes}
                stripped
                    .parse::<usize>()
                    .map_err(|_| PluginCreateError::NameInvalid { name: name.clone() })?
            } else {
                return Err(PluginCreateError::NameInvalid { name: name.clone() });
            }
        }
        zarrs_metadata::v2::DataTypeMetadataV2::Structured(_) => {
            return Err(PluginCreateError::Other(
                "raw bits does not support structured types".into(),
            ));
        }
    };
    Ok(std::sync::Arc::new(RawBitsDataType::new(size_bytes)).into())
}

static RAWBITSDATATYPE_ALIASES_V3: LazyLock<RwLock<ExtensionAliasesConfig>> = LazyLock::new(|| {
    RwLock::new(ExtensionAliasesConfig::new(
        "r*",
        vec![],
        vec![Regex::new(r"^r\d+$").unwrap()],
    ))
});

static RAWBITSDATATYPE_ALIASES_V2: LazyLock<RwLock<ExtensionAliasesConfig>> = LazyLock::new(|| {
    RwLock::new(ExtensionAliasesConfig::new(
        "r*",
        vec![],
        vec![Regex::new(r"^\|V\d+$").unwrap()],
    ))
});

impl ExtensionAliases<ZarrVersion3> for RawBitsDataType {
    fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
        RAWBITSDATATYPE_ALIASES_V3.read().unwrap()
    }

    fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
        RAWBITSDATATYPE_ALIASES_V3.write().unwrap()
    }
}

impl ExtensionAliases<ZarrVersion2> for RawBitsDataType {
    fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
        RAWBITSDATATYPE_ALIASES_V2.read().unwrap()
    }

    fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
        RAWBITSDATATYPE_ALIASES_V2.write().unwrap()
    }
}

// Instance-specific ExtensionName implementation
impl zarrs_plugin::ExtensionName for RawBitsDataType {
    fn name(&self, version: ZarrVersions) -> Option<Cow<'static, str>> {
        Some(match version {
            ZarrVersions::V3 => {
                // Return "r{bits}" where bits = size_bytes * 8
                Cow::Owned(format!("r{}", self.size_bytes * 8))
            }
            ZarrVersions::V2 => {
                // Return "|V{bytes}"
                Cow::Owned(format!("|V{}", self.size_bytes))
            }
        })
    }
}

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

impl zarrs_data_type::DataTypeTraits for RawBitsDataType {
    fn configuration(&self, _version: zarrs_plugin::ZarrVersions) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(self.size_bytes)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::FillValueMetadata,
        _version: zarrs_plugin::ZarrVersions,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        use base64::Engine;
        use base64::prelude::BASE64_STANDARD;
        // RawBits fill value can be base64-encoded string or array of bytes
        if let Some(s) = fill_value_metadata.as_str() {
            let bytes = BASE64_STANDARD
                .decode(s)
                .map_err(|_| zarrs_data_type::DataTypeFillValueMetadataError)?;
            if bytes.len() != self.size_bytes {
                return Err(zarrs_data_type::DataTypeFillValueMetadataError);
            }
            Ok(zarrs_data_type::FillValue::from(bytes))
        } else if let Some(arr) = fill_value_metadata.as_array() {
            if arr.len() != self.size_bytes {
                return Err(zarrs_data_type::DataTypeFillValueMetadataError);
            }
            let bytes: Result<Vec<u8>, _> = arr
                .iter()
                .map(|v| {
                    v.as_u64()
                        .and_then(|u| u8::try_from(u).ok())
                        .ok_or(zarrs_data_type::DataTypeFillValueMetadataError)
                })
                .collect();
            Ok(zarrs_data_type::FillValue::from(bytes?))
        } else {
            Err(zarrs_data_type::DataTypeFillValueMetadataError)
        }
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<zarrs_metadata::FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
        // Return as array of bytes (not base64 encoded) for consistency
        let bytes = fill_value.as_ne_bytes();
        let arr: Vec<zarrs_metadata::FillValueMetadata> = bytes
            .iter()
            .map(|&b| zarrs_metadata::FillValueMetadata::from(b))
            .collect();
        Ok(zarrs_metadata::FillValueMetadata::Array(arr))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

crate::array::codec::impl_bytes_codec_passthrough!(RawBitsDataType);
