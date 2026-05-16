//! The `fixed_length_utf32` data type.
//!
//! See <https://github.com/zarr-developers/zarr-extensions/tree/main/data-types/fixed_length_utf32>.

use std::any::TypeId;
use std::borrow::Cow;
use std::sync::Arc;

use zarrs_data_type::DataType;
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::{Configuration, DataTypeSize, FillValueMetadata};
use zarrs_plugin::{PluginCreateError, Regex, ZarrVersion};

use zarrs_metadata_ext::data_type::fixed_length_utf32::FixedLengthUTF32DataTypeConfigurationV1;

/// The `fixed_length_utf32` data type.
///
/// Represents fixed-length UTF-32 strings stored as native-endian UTF-32 code units,
/// padded with U+0000 to fill `length_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedLengthUTF32DataType {
    /// Total size of each element in bytes (must be a multiple of 4, at least 4).
    length_bytes: u32,
}

// FixedLengthUTF32DataType uses instance-specific names based on length_bytes.
// We use a V3-only approach with configuration-based matching.

impl zarrs_data_type::DataTypeTraitsV3 for FixedLengthUTF32DataType {
    fn create(metadata: &MetadataV3) -> Result<DataType, PluginCreateError> {
        let config: FixedLengthUTF32DataTypeConfigurationV1 = metadata.to_typed_configuration()?;
        Ok(Arc::new(FixedLengthUTF32DataType::new(config.length_bytes)?).into())
    }
}

impl zarrs_data_type::DataTypeTraitsV2 for FixedLengthUTF32DataType {
    fn create(
        metadata: &zarrs_metadata::v2::DataTypeMetadataV2,
    ) -> Result<DataType, PluginCreateError> {
        let name = match metadata {
            zarrs_metadata::v2::DataTypeMetadataV2::Simple(name) => name.as_str(),
            zarrs_metadata::v2::DataTypeMetadataV2::Structured(_) => {
                return Err(PluginCreateError::Other(
                    "fixed_length_utf32 does not support structured types".into(),
                ));
            }
        };

        // Parse <U{N} or >U{N} (NumPy fixed Unicode dtypes)
        // The N is the number of code points, so length_bytes = N * 4
        let (n_str, _) = if let Some(stripped) = name.strip_prefix('<') {
            if let Some(n_str) = stripped.strip_prefix('U') {
                (n_str, true)
            } else {
                return Err(PluginCreateError::NameInvalid {
                    name: name.to_string(),
                });
            }
        } else if let Some(stripped) = name.strip_prefix('>') {
            if let Some(n_str) = stripped.strip_prefix('U') {
                (n_str, true)
            } else {
                return Err(PluginCreateError::NameInvalid {
                    name: name.to_string(),
                });
            }
        } else {
            return Err(PluginCreateError::NameInvalid {
                name: name.to_string(),
            });
        };

        let n: u32 = n_str.parse().map_err(|_| PluginCreateError::NameInvalid {
            name: name.to_string(),
        })?;

        let length_bytes = n
            .checked_mul(4)
            .ok_or_else(|| PluginCreateError::Other("length_bytes overflow".to_string()))?;

        Ok(Arc::new(FixedLengthUTF32DataType::new(length_bytes)?).into())
    }
}

// Register V3 plugin.
inventory::submit! {
    zarrs_data_type::DataTypePluginV3::new::<FixedLengthUTF32DataType>()
}

// Register V2 plugin.
inventory::submit! {
    zarrs_data_type::DataTypePluginV2::new::<FixedLengthUTF32DataType>()
}

impl zarrs_plugin::ExtensionAliases<zarrs_plugin::ZarrVersion3> for FixedLengthUTF32DataType {
    fn aliases() -> std::sync::RwLockReadGuard<'static, zarrs_plugin::ExtensionAliasesConfig> {
        FIXEDLENGTHUTF32DATATYPE_ALIASES_V3.read().unwrap()
    }

    fn aliases_mut() -> std::sync::RwLockWriteGuard<'static, zarrs_plugin::ExtensionAliasesConfig> {
        FIXEDLENGTHUTF32DATATYPE_ALIASES_V3.write().unwrap()
    }
}

impl zarrs_plugin::ExtensionAliases<zarrs_plugin::ZarrVersion2> for FixedLengthUTF32DataType {
    fn aliases() -> std::sync::RwLockReadGuard<'static, zarrs_plugin::ExtensionAliasesConfig> {
        FIXEDLENGTHUTF32DATATYPE_ALIASES_V2.read().unwrap()
    }

    fn aliases_mut() -> std::sync::RwLockWriteGuard<'static, zarrs_plugin::ExtensionAliasesConfig> {
        FIXEDLENGTHUTF32DATATYPE_ALIASES_V2.write().unwrap()
    }
}

static FIXEDLENGTHUTF32DATATYPE_ALIASES_V3: std::sync::LazyLock<
    std::sync::RwLock<zarrs_plugin::ExtensionAliasesConfig>,
> = std::sync::LazyLock::new(|| {
    std::sync::RwLock::new(zarrs_plugin::ExtensionAliasesConfig::new(
        "fixed_length_utf32",
        vec![],
        vec![],
    ))
});

static FIXEDLENGTHUTF32DATATYPE_ALIASES_V2: std::sync::LazyLock<
    std::sync::RwLock<zarrs_plugin::ExtensionAliasesConfig>,
> = std::sync::LazyLock::new(|| {
    std::sync::RwLock::new(zarrs_plugin::ExtensionAliasesConfig::new(
        "", // V2 name is instance-specific (<U12, >U12 etc)
        vec![],
        vec![Regex::new(r"^[<>]U\d+$").unwrap()],
    ))
});

impl zarrs_plugin::ExtensionName for FixedLengthUTF32DataType {
    fn name(&self, version: ZarrVersion) -> Option<Cow<'static, str>> {
        Some(match version {
            ZarrVersion::V3 => Cow::Borrowed("fixed_length_utf32"),
            ZarrVersion::V2 => {
                // Return <U{N} for V2 (NumPy format)
                Cow::Owned(format!("<U{}", self.capacity_code_points()))
            }
        })
    }
}

impl FixedLengthUTF32DataType {
    /// Create a new `FixedLengthUTF32DataType` with the given size in bytes.
    ///
    /// `length_bytes` must be at least 4 and a multiple of 4.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if `length_bytes` is invalid.
    pub fn new(length_bytes: u32) -> Result<Self, PluginCreateError> {
        if length_bytes < 4 || !length_bytes.is_multiple_of(4) {
            return Err(PluginCreateError::Other(format!(
                "length_bytes must be at least 4 and a multiple of 4, got {length_bytes}"
            )));
        }
        Ok(Self { length_bytes })
    }

    /// Returns the size of each element in bytes.
    #[must_use]
    pub const fn length_bytes(&self) -> u32 {
        self.length_bytes
    }

    /// Returns the number of code points each element can hold (length_bytes / 4).
    #[must_use]
    pub const fn capacity_code_points(&self) -> u32 {
        self.length_bytes / 4
    }
}

impl zarrs_data_type::DataTypeTraits for FixedLengthUTF32DataType {
    fn configuration(&self, _version: ZarrVersion) -> Configuration {
        Configuration::from(FixedLengthUTF32DataTypeConfigurationV1 {
            length_bytes: self.length_bytes,
        })
    }

    fn size(&self) -> DataTypeSize {
        DataTypeSize::Fixed(self.length_bytes as usize)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &FillValueMetadata,
        _version: ZarrVersion,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        // Fill value is a JSON string
        let Some(s) = fill_value_metadata.as_str() else {
            return Err(zarrs_data_type::DataTypeFillValueMetadataError);
        };

        let chars: Vec<char> = s.chars().collect();
        let capacity = self.capacity_code_points() as usize;

        // Check that the string fits
        if chars.len() > capacity {
            return Err(zarrs_data_type::DataTypeFillValueMetadataError);
        }

        // Encode as native-endian UTF-32, padded with U+0000
        let mut bytes = Vec::with_capacity(self.length_bytes as usize);
        for &ch in &chars {
            bytes.extend_from_slice(&(ch as u32).to_ne_bytes());
        }
        // Pad with U+0000
        for _ in chars.len()..capacity {
            bytes.extend_from_slice(&0u32.to_ne_bytes());
        }

        Ok(zarrs_data_type::FillValue::new(bytes))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
        let bytes = fill_value.as_ne_bytes();
        if bytes.len() != self.length_bytes as usize {
            return Err(zarrs_data_type::DataTypeFillValueError);
        }

        // Decode all UTF-32 code units (native-endian)
        let capacity = self.capacity_code_points() as usize;
        let mut chars = Vec::with_capacity(capacity);

        for i in 0..capacity {
            let start = i * 4;
            let mut code_unit_bytes = [0u8; 4];
            code_unit_bytes.copy_from_slice(&bytes[start..start + 4]);
            let code_unit = u32::from_ne_bytes(code_unit_bytes);

            // U+0000 is a valid Unicode scalar value
            match char::from_u32(code_unit) {
                Some(ch) => chars.push(ch),
                None => return Err(zarrs_data_type::DataTypeFillValueError),
            }
        }

        // Trim trailing U+0000 (padding)
        while chars.last() == Some(&'\0') {
            chars.pop();
        }

        let s: String = chars.into_iter().collect();
        Ok(FillValueMetadata::from(s))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn compatible_element_types(&self) -> &'static [TypeId] {
        // &[char] and Vec<char> are always compatible (within capacity)
        // [char; N] and &[char; N] require exact capacity match, checked at runtime
        const TYPES: [TypeId; 3] = [
            TypeId::of::<&[char]>(),
            TypeId::of::<Vec<char>>(),
            TypeId::of::<char>(),
        ];
        &TYPES
    }
}

zarrs_data_type::codec_traits::impl_bytes_data_type_traits!(FixedLengthUTF32DataType, 4);

#[cfg(test)]
mod tests {
    use super::*;
    use zarrs_metadata::v3::MetadataV3;
    use zarrs_plugin::ExtensionName;

    fn data_type_metadata(data_type: &DataType) -> MetadataV3 {
        let name = data_type
            .name_v3()
            .map_or_else(String::new, std::borrow::Cow::into_owned);
        let configuration = data_type.configuration_v3();
        if configuration.is_empty() {
            MetadataV3::new(name)
        } else {
            MetadataV3::new_with_configuration(name, configuration)
        }
    }

    #[test]
    fn from_metadata_v3() {
        let json = r#"{"name":"fixed_length_utf32","configuration":{"length_bytes":16}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        let data_type = DataType::from_metadata(&metadata).unwrap();
        assert!(data_type.is::<FixedLengthUTF32DataType>());
        assert_eq!(data_type.fixed_size(), Some(16));
        assert_eq!(
            json,
            serde_json::to_string(&data_type_metadata(&data_type)).unwrap()
        );
    }

    #[test]
    fn from_metadata_v3_missing_length_bytes() {
        let json = r#"{"name":"fixed_length_utf32","configuration":{}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        assert!(DataType::from_metadata(&metadata).is_err());
    }

    #[test]
    fn from_metadata_v3_zero_length_bytes() {
        let json = r#"{"name":"fixed_length_utf32","configuration":{"length_bytes":0}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        assert!(DataType::from_metadata(&metadata).is_err());
    }

    #[test]
    fn from_metadata_v3_non_multiple_of_4() {
        let json = r#"{"name":"fixed_length_utf32","configuration":{"length_bytes":6}}"#;
        let metadata: MetadataV3 = serde_json::from_str(json).unwrap();
        assert!(DataType::from_metadata(&metadata).is_err());
    }

    #[test]
    fn v2_name_parsing_lt_u12() {
        // Verify that the V2 plugin correctly parses <U12 format
        let dt = FixedLengthUTF32DataType::new(48).unwrap();
        // V3 name should be fixed_length_utf32
        assert_eq!(
            dt.name(ZarrVersion::V3).unwrap().as_ref(),
            "fixed_length_utf32"
        );
        // V2 name should be <U12
        assert_eq!(dt.name(ZarrVersion::V2).unwrap().as_ref(), "<U12");
    }

    #[test]
    fn v2_name_parsing_gt_u12() {
        let dt = FixedLengthUTF32DataType::new(48).unwrap();
        // V2 name is always <U{N} (we use little-endian as default)
        assert_eq!(dt.name(ZarrVersion::V2).unwrap().as_ref(), "<U12");
    }

    #[test]
    fn fill_value_empty() {
        let data_type = FixedLengthUTF32DataType::new(8).unwrap();
        let dt = DataType::new(data_type);
        let metadata = FillValueMetadata::from("");
        let fill_value = dt.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.as_ne_bytes(), [0u8; 8]);

        // Round-trip
        let round_trip = dt.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(round_trip, FillValueMetadata::from(""));
    }

    #[test]
    fn fill_value_exact() {
        let data_type = FixedLengthUTF32DataType::new(8).unwrap();
        let dt = DataType::new(data_type);
        // "ab" = 2 code points = 8 bytes
        let metadata = FillValueMetadata::from("ab");
        let fill_value = dt.fill_value_v3(&metadata).unwrap();
        let expected: Vec<u8> = [b'a' as u32, b'b' as u32]
            .iter()
            .flat_map(|&c| c.to_ne_bytes())
            .collect();
        assert_eq!(fill_value.as_ne_bytes(), expected);

        let round_trip = dt.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(round_trip, FillValueMetadata::from("ab"));
    }

    #[test]
    fn fill_value_shorter() {
        let data_type = FixedLengthUTF32DataType::new(12).unwrap();
        let dt = DataType::new(data_type);
        // "a" = 1 code point, padded to 3 code points (12 bytes)
        let metadata = FillValueMetadata::from("a");
        let fill_value = dt.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.size(), 12);
        // First 4 bytes = 'a' in UTF-32, rest = 0
        let mut expected = vec![0u8; 12];
        expected[..4].copy_from_slice(&(b'a' as u32).to_ne_bytes());
        assert_eq!(fill_value.as_ne_bytes(), expected);

        let round_trip = dt.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(round_trip, FillValueMetadata::from("a"));
    }

    #[test]
    fn fill_value_overlong() {
        let data_type = FixedLengthUTF32DataType::new(8).unwrap();
        let dt = DataType::new(data_type);
        // "abc" = 3 code points = 12 bytes > 8 byte capacity
        let metadata = FillValueMetadata::from("abc");
        assert!(dt.fill_value_v3(&metadata).is_err());
    }

    #[test]
    fn fill_value_with_interior_null() {
        let data_type = FixedLengthUTF32DataType::new(12).unwrap();
        let dt = DataType::new(data_type);
        // "a\0b" = 3 code points = 12 bytes, interior U+0000
        let metadata = FillValueMetadata::from("a\0b");
        let fill_value = dt.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.size(), 12);

        // Round-trip should preserve interior null
        let round_trip = dt.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(round_trip, FillValueMetadata::from("a\0b"));
    }

    #[test]
    fn fill_value_trailing_padding_trim() {
        let data_type = FixedLengthUTF32DataType::new(16).unwrap();
        let dt = DataType::new(data_type);
        // "ab" = 2 code points, padded to 4 code points (16 bytes)
        let metadata = FillValueMetadata::from("ab");
        let fill_value = dt.fill_value_v3(&metadata).unwrap();

        let round_trip = dt.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(round_trip, FillValueMetadata::from("ab")); // trailing padding trimmed
    }

    #[test]
    fn fill_value_non_ascii() {
        let data_type = FixedLengthUTF32DataType::new(8).unwrap();
        let dt = DataType::new(data_type);
        // '🎉' (U+1F389) is a single code point = 4 bytes
        let metadata = FillValueMetadata::from("🎉");
        let fill_value = dt.fill_value_v3(&metadata).unwrap();
        assert_eq!(fill_value.size(), 8);

        let round_trip = dt.metadata_fill_value(&fill_value).unwrap();
        assert_eq!(round_trip, FillValueMetadata::from("🎉"));
    }

    #[test]
    fn bytes_codec_little_endian() {
        use std::borrow::Cow;
        use zarrs_data_type::codec_traits::bytes::BytesDataTypeTraits;

        let data_type = FixedLengthUTF32DataType::new(12).unwrap();
        // "abc" = 3 code points in native endian
        let chars = ['a', 'b', 'c'];
        let bytes: Vec<u8> = chars
            .iter()
            .flat_map(|&c| (c as u32).to_ne_bytes())
            .collect();
        let bytes_cow: std::borrow::Cow<'_, [u8]> = Cow::Owned(bytes.clone());

        // Encode to little endian (same as native on LE)
        let encoded = data_type
            .encode(bytes_cow.clone(), Some(zarrs_metadata::Endianness::Little))
            .unwrap();

        // On LE, little endian should be a no-op
        if cfg!(target_endian = "little") {
            assert_eq!(encoded.as_ref(), &bytes);
        }
    }

    #[test]
    fn bytes_codec_big_endian_swap_per_code_unit() {
        use std::borrow::Cow;
        use zarrs_data_type::codec_traits::bytes::BytesDataTypeTraits;

        let data_type = FixedLengthUTF32DataType::new(12).unwrap();
        // "abc" = 3 code points
        let chars = ['a', 'b', 'c'];
        let native_bytes: Vec<u8> = chars
            .iter()
            .flat_map(|&c| (c as u32).to_ne_bytes())
            .collect();

        // Encode to big endian
        let encoded = data_type
            .encode(
                Cow::Owned(native_bytes.clone()),
                Some(zarrs_metadata::Endianness::Big),
            )
            .unwrap();

        // Each 4-byte code unit should be byte-swapped
        for (i, &ch) in chars.iter().enumerate() {
            let start = i * 4;
            let expected = (ch as u32).to_be_bytes();
            assert_eq!(&encoded[start..start + 4], &expected[..]);
        }

        // Decode back
        let decoded = data_type
            .decode(encoded, Some(zarrs_metadata::Endianness::Big))
            .unwrap();
        assert_eq!(decoded.as_ref(), &native_bytes);
    }
}
