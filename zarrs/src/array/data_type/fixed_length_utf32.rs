//! The `fixed_length_utf32` data type.
//!
//! See <https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/fixed-length-utf32>.

use std::borrow::Cow;
use std::sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

use zarrs_codec::CodecError;
use zarrs_data_type::DataType;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::{
    ExtensionAliases, ExtensionAliasesConfig, PluginCreateError, Regex, ZarrVersion, ZarrVersion2,
    ZarrVersion3,
};

use zarrs_metadata_ext::data_type::fixed_length_utf32::FixedLengthUtf32DataTypeConfigurationV1;

/// The `fixed_length_utf32` data type.
///
/// Represents fixed-length UTF-32 encoded strings where each array element
/// contains exactly `length_bytes` bytes (must be divisible by 4).
///
/// Each Unicode code point is encoded as 4 bytes (UTF-32), so the number of
/// characters is `length_bytes / 4`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedLengthUtf32DataType {
    /// Length in bytes (must be divisible by 4).
    length_bytes: usize,
}

// Manual implementations instead of impl_extension_aliases! because FixedLengthUtf32DataType
// needs instance-specific names based on length_bytes for V2 (e.g., "<U20", ">U5")
static FIXEDLENGTHUTF32DATATYPE_ALIASES_V3: LazyLock<RwLock<ExtensionAliasesConfig>> =
    LazyLock::new(|| {
        RwLock::new(ExtensionAliasesConfig::new(
            "fixed_length_utf32",
            vec![],
            vec![], // No regex needed for V3
        ))
    });

static FIXEDLENGTHUTF32DATATYPE_ALIASES_V2: LazyLock<RwLock<ExtensionAliasesConfig>> =
    LazyLock::new(|| {
        RwLock::new(ExtensionAliasesConfig::new(
            "<U*", // Placeholder, actual matching done via regex
            vec![],
            vec![Regex::new(r"^[<>]U\d+$").unwrap()], // Matches <U20, >U5, etc.
        ))
    });

impl ExtensionAliases<ZarrVersion3> for FixedLengthUtf32DataType {
    fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
        FIXEDLENGTHUTF32DATATYPE_ALIASES_V3.read().unwrap()
    }

    fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
        FIXEDLENGTHUTF32DATATYPE_ALIASES_V3.write().unwrap()
    }
}

impl ExtensionAliases<ZarrVersion2> for FixedLengthUtf32DataType {
    fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
        FIXEDLENGTHUTF32DATATYPE_ALIASES_V2.read().unwrap()
    }

    fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
        FIXEDLENGTHUTF32DATATYPE_ALIASES_V2.write().unwrap()
    }
}

// Instance-specific ExtensionName implementation
impl zarrs_plugin::ExtensionName for FixedLengthUtf32DataType {
    fn name(&self, version: ZarrVersion) -> Option<Cow<'static, str>> {
        match version {
            ZarrVersion::V3 => {
                let aliases = FIXEDLENGTHUTF32DATATYPE_ALIASES_V3.read().unwrap();
                (!aliases.default_name.is_empty()).then(|| aliases.default_name.clone())
            }
            ZarrVersion::V2 => {
                // Return "<U{num_chars}" (always little-endian for V2 name)
                // The actual endianness is handled by the bytes codec via data_type_metadata_v2_to_endianness
                let num_chars = self.length_bytes / 4;
                Some(Cow::Owned(format!("<U{num_chars}")))
            }
        }
    }
}

impl zarrs_data_type::DataTypeTraitsV3 for FixedLengthUtf32DataType {
    fn create(metadata: &MetadataV3) -> Result<DataType, PluginCreateError> {
        let config: FixedLengthUtf32DataTypeConfigurationV1 = metadata.to_typed_configuration()?;

        // Validate: must be divisible by 4
        if !config.length_bytes.is_multiple_of(4) {
            return Err(PluginCreateError::Other(format!(
                "fixed_length_utf32 length_bytes must be divisible by 4: {}",
                config.length_bytes
            )));
        }

        // Validate: max value (2147483644 = i32::MAX - 3, rounded down to multiple of 4)
        if config.length_bytes > 2_147_483_644 {
            return Err(PluginCreateError::Other(format!(
                "fixed_length_utf32 length_bytes exceeds maximum: {}",
                config.length_bytes
            )));
        }

        Ok(std::sync::Arc::new(FixedLengthUtf32DataType::new(config.length_bytes as usize)).into())
    }
}

impl zarrs_data_type::DataTypeTraitsV2 for FixedLengthUtf32DataType {
    fn create(
        metadata: &zarrs_metadata::v2::DataTypeMetadataV2,
    ) -> Result<DataType, PluginCreateError> {
        match metadata {
            zarrs_metadata::v2::DataTypeMetadataV2::Simple(name) => {
                // Parse "<U{n}" or ">U{n}"
                let rest = if let Some(rest) = name.strip_prefix('<') {
                    rest
                } else if let Some(rest) = name.strip_prefix('>') {
                    rest
                } else {
                    return Err(PluginCreateError::NameInvalid { name: name.clone() });
                };

                let num_chars_str = rest
                    .strip_prefix('U')
                    .ok_or_else(|| PluginCreateError::NameInvalid { name: name.clone() })?;

                let num_chars: usize = num_chars_str
                    .parse()
                    .map_err(|_| PluginCreateError::NameInvalid { name: name.clone() })?;

                let length_bytes = num_chars * 4;

                Ok(std::sync::Arc::new(FixedLengthUtf32DataType::new(length_bytes)).into())
            }
            zarrs_metadata::v2::DataTypeMetadataV2::Structured(_) => Err(PluginCreateError::Other(
                "fixed_length_utf32 does not support structured types".into(),
            )),
        }
    }
}

// Register V3 plugin
inventory::submit! {
    zarrs_data_type::DataTypePluginV3::new::<FixedLengthUtf32DataType>()
}

// Register V2 plugin
inventory::submit! {
    zarrs_data_type::DataTypePluginV2::new::<FixedLengthUtf32DataType>()
}

impl FixedLengthUtf32DataType {
    /// Static instance for use in trait implementations (defaults to 4 bytes = 1 character).
    pub const STATIC: Self = Self { length_bytes: 4 };

    /// Create a new `fixed_length_utf32` data type with the given length in bytes.
    ///
    /// # Panics
    /// Panics if `length_bytes` is not divisible by 4.
    #[must_use]
    pub const fn new(length_bytes: usize) -> Self {
        assert!(
            length_bytes.is_multiple_of(4),
            "length_bytes must be divisible by 4"
        );
        Self { length_bytes }
    }

    /// Returns the length in bytes.
    #[must_use]
    pub const fn length_bytes(&self) -> usize {
        self.length_bytes
    }

    /// Returns the number of UTF-32 characters (code points).
    #[must_use]
    pub const fn num_chars(&self) -> usize {
        self.length_bytes / 4
    }
}

impl zarrs_data_type::DataTypeTraits for FixedLengthUtf32DataType {
    #[allow(clippy::cast_possible_truncation)]
    fn configuration(&self, version: zarrs_plugin::ZarrVersion) -> zarrs_metadata::Configuration {
        match version {
            zarrs_plugin::ZarrVersion::V3 => {
                // length_bytes is validated to be <= 2_147_483_644 which fits in u32
                zarrs_metadata::Configuration::from(FixedLengthUtf32DataTypeConfigurationV1 {
                    length_bytes: self.length_bytes as u32,
                })
            }
            zarrs_plugin::ZarrVersion::V2 => {
                // V2 doesn't use configuration - everything is in the name
                zarrs_metadata::Configuration::default()
            }
        }
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(self.length_bytes)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::FillValueMetadata,
        version: zarrs_plugin::ZarrVersion,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        // Handle string fill value
        let s = if let Some(s) = fill_value_metadata.as_str() {
            s.to_string()
        } else if matches!(version, zarrs_plugin::ZarrVersion::V2) {
            // V2: null/0 -> empty string
            if fill_value_metadata.is_null() || fill_value_metadata.as_u64() == Some(0) {
                String::new()
            } else {
                return Err(zarrs_data_type::DataTypeFillValueMetadataError);
            }
        } else {
            return Err(zarrs_data_type::DataTypeFillValueMetadataError);
        };

        // Encode string as UTF-32 and pad to length_bytes
        let utf32_chars: Vec<u32> = s.chars().map(|c| c as u32).collect();
        let utf32_bytes_needed = utf32_chars.len() * 4;

        if utf32_bytes_needed > self.length_bytes {
            return Err(zarrs_data_type::DataTypeFillValueMetadataError);
        }

        // Create byte buffer with UTF-32 encoding (native endian in memory)
        let mut bytes = Vec::with_capacity(self.length_bytes);
        for code_point in utf32_chars {
            bytes.extend_from_slice(&code_point.to_ne_bytes());
        }
        // Pad with zeros
        bytes.resize(self.length_bytes, 0);

        Ok(zarrs_data_type::FillValue::from(bytes))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<zarrs_metadata::FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
        let bytes = fill_value.as_ne_bytes();

        // Decode UTF-32 from bytes
        let mut chars = Vec::new();
        for chunk in bytes.chunks_exact(4) {
            let code_point = u32::from_ne_bytes(chunk.try_into().unwrap());
            if code_point == 0 {
                break; // Stop at null terminator
            }
            if let Some(c) = char::from_u32(code_point) {
                chars.push(c);
            } else {
                return Err(zarrs_data_type::DataTypeFillValueError);
            }
        }

        let s: String = chars.into_iter().collect();
        Ok(zarrs_metadata::FillValueMetadata::from(s.as_str()))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl crate::array::codec::BytesCodecDataTypeTraits for FixedLengthUtf32DataType {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, CodecError> {
        let endianness = endianness.ok_or(CodecError::from(
            "endianness must be specified for fixed_length_utf32",
        ))?;

        if endianness == zarrs_metadata::Endianness::native() {
            Ok(bytes)
        } else {
            // Swap endianness per 4-byte character (UTF-32 code point)
            let mut result = bytes.into_owned();
            for chunk in result.as_chunks_mut::<4>().0 {
                chunk.reverse();
            }
            Ok(std::borrow::Cow::Owned(result))
        }
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, CodecError> {
        self.encode(bytes, endianness) // Symmetric operation
    }
}

// Register with bytes codec
zarrs_codec::register_data_type_extension_codec!(
    FixedLengthUtf32DataType,
    crate::array::codec::BytesPlugin,
    crate::array::codec::BytesCodecDataTypeTraits
);
