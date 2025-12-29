//! Special data type markers and implementations (bool, string, bytes, rawbits, optional, numpy time types).

use std::num::NonZeroU32;

use zarrs_metadata::ConfigurationSerialize;
use zarrs_plugin::{ExtensionIdentifier, PluginCreateError, PluginMetadataInvalidError, Regex};

use crate::metadata_ext::data_type::{
    NumpyTimeUnit, numpy_datetime64::NumpyDateTime64DataTypeConfigurationV1,
    numpy_timedelta64::NumpyTimeDelta64DataTypeConfigurationV1,
};

use super::macros::register_data_type_plugin;

// Boolean - V2: |b1

/// The `bool` data type.
#[derive(Debug, Clone, Copy)]
pub struct BoolDataType;
zarrs_plugin::impl_extension_aliases!(BoolDataType, "bool",
    v3: "bool", [],
    v2: "|b1", ["|b1"]
);

// Variable-length types - V2: |O for string, |V\d+ for bytes

/// The `string` data type.
#[derive(Debug, Clone, Copy)]
pub struct StringDataType;
zarrs_plugin::impl_extension_aliases!(StringDataType, "string",
    v3: "string", [],
    v2: "|O", ["|O"]
);

/// The `bytes` data type.
#[derive(Debug, Clone, Copy)]
pub struct BytesDataType;
zarrs_plugin::impl_extension_aliases!(BytesDataType, "bytes",
    v3: "bytes", ["binary", "variable_length_bytes"],
    v2: "|VX", ["|VX"]
);

// RawBits - special handling for regex-based matching

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

// NumPy time types

/// The `numpy.datetime64` data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumpyDateTime64DataType {
    /// The `NumPy` temporal unit.
    pub unit: NumpyTimeUnit,
    /// The `NumPy` temporal scale factor.
    pub scale_factor: NonZeroU32,
}
zarrs_plugin::impl_extension_aliases!(NumpyDateTime64DataType, "numpy.datetime64");

impl NumpyDateTime64DataType {
    /// Static instance for use in trait implementations (bytes codec only).
    /// Uses `NumpyTimeUnit::Generic` and scale factor 1 as defaults.
    pub const STATIC: Self = Self {
        unit: NumpyTimeUnit::Generic,
        scale_factor: NonZeroU32::new(1).unwrap(),
    };

    /// Create a new `numpy.datetime64` data type.
    #[must_use]
    pub const fn new(unit: NumpyTimeUnit, scale_factor: NonZeroU32) -> Self {
        Self { unit, scale_factor }
    }
}

/// The `numpy.timedelta64` data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumpyTimeDelta64DataType {
    /// The `NumPy` temporal unit.
    pub unit: NumpyTimeUnit,
    /// The `NumPy` temporal scale factor.
    pub scale_factor: NonZeroU32,
}
zarrs_plugin::impl_extension_aliases!(NumpyTimeDelta64DataType, "numpy.timedelta64");

impl NumpyTimeDelta64DataType {
    /// Static instance for use in trait implementations (bytes codec only).
    /// Uses `NumpyTimeUnit::Generic` and scale factor 1 as defaults.
    pub const STATIC: Self = Self {
        unit: NumpyTimeUnit::Generic,
        scale_factor: NonZeroU32::new(1).unwrap(),
    };

    /// Create a new `numpy.timedelta64` data type.
    #[must_use]
    pub const fn new(unit: NumpyTimeUnit, scale_factor: NonZeroU32) -> Self {
        Self { unit, scale_factor }
    }
}

// ============================================================================
// DataTypeExtension implementations for special types
// ============================================================================

// Bool - fixed size 1 byte
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
}

impl zarrs_data_type::DataTypeExtensionBytesCodec for BoolDataType {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        _endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        _endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        Ok(bytes)
    }
}

// String - variable length
impl zarrs_data_type::DataTypeExtension for StringDataType {
    fn identifier(&self) -> &'static str {
        <Self as ExtensionIdentifier>::IDENTIFIER
    }

    fn configuration(&self) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Variable
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
        let s = fill_value_metadata.as_str().ok_or_else(err)?;
        Ok(zarrs_data_type::FillValue::from(s.as_bytes().to_vec()))
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
        let s = std::str::from_utf8(fill_value.as_ne_bytes()).map_err(|_| error())?;
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(s))
    }

    fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
        Some(self)
    }
}

impl zarrs_data_type::DataTypeExtensionBytesCodec for StringDataType {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        _endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        _endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        Ok(bytes)
    }
}

// Bytes - variable length
impl zarrs_data_type::DataTypeExtension for BytesDataType {
    fn identifier(&self) -> &'static str {
        <Self as ExtensionIdentifier>::IDENTIFIER
    }

    fn configuration(&self) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Variable
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::v3::FillValueMetadataV3,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        use base64::{Engine, prelude::BASE64_STANDARD};
        let err = || {
            zarrs_data_type::DataTypeFillValueMetadataError::new(
                self.identifier().to_string(),
                fill_value_metadata.clone(),
            )
        };
        // Bytes fill value is base64-encoded
        let s = fill_value_metadata.as_str().ok_or_else(err)?;
        let bytes = BASE64_STANDARD.decode(s).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(bytes))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<zarrs_metadata::v3::FillValueMetadataV3, zarrs_data_type::DataTypeFillValueError>
    {
        use base64::{Engine, prelude::BASE64_STANDARD};
        let encoded = BASE64_STANDARD.encode(fill_value.as_ne_bytes());
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(encoded))
    }

    fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
        Some(self)
    }
}

impl zarrs_data_type::DataTypeExtensionBytesCodec for BytesDataType {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        _endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        _endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        Ok(bytes)
    }
}

// RawBits - Note: RawBits has a configurable size parsed from the name (e.g., r8, r16, r24)
// The plugin registration uses the marker type, but actual instances need size from metadata.
// For the plugin, we register it so V2->V3 name matching works.
impl zarrs_data_type::DataTypeExtension for RawBitsDataType {
    fn identifier(&self) -> &'static str {
        <Self as ExtensionIdentifier>::IDENTIFIER
    }

    fn metadata_name(&self, _zarr_version: zarrs_plugin::ZarrVersions) -> std::borrow::Cow<'static, str> {
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
        let err = || {
            zarrs_data_type::DataTypeFillValueMetadataError::new(
                self.identifier().to_string(),
                fill_value_metadata.clone(),
            )
        };
        // RawBits fill value is base64-encoded
        let s = fill_value_metadata.as_str().ok_or_else(err)?;
        let bytes = BASE64_STANDARD.decode(s).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(bytes))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<zarrs_metadata::v3::FillValueMetadataV3, zarrs_data_type::DataTypeFillValueError>
    {
        use base64::{Engine, prelude::BASE64_STANDARD};
        let encoded = BASE64_STANDARD.encode(fill_value.as_ne_bytes());
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(encoded))
    }

    fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
        Some(self)
    }
}

impl zarrs_data_type::DataTypeExtensionBytesCodec for RawBitsDataType {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        _endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        _endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        Ok(bytes)
    }
}

// NumpyDateTime64 - 8 bytes (i64)
impl zarrs_data_type::DataTypeExtension for NumpyDateTime64DataType {
    fn identifier(&self) -> &'static str {
        <Self as ExtensionIdentifier>::IDENTIFIER
    }

    fn configuration(&self) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::from(NumpyDateTime64DataTypeConfigurationV1 {
            unit: self.unit,
            scale_factor: self.scale_factor,
        })
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(8)
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
        // Handle "NaT" (Not a Time) as i64::MIN
        if let Some("NaT") = fill_value_metadata.as_str() {
            return Ok(zarrs_data_type::FillValue::from(i64::MIN));
        }
        // Otherwise expect an integer
        let i = fill_value_metadata.as_i64().ok_or_else(err)?;
        Ok(zarrs_data_type::FillValue::from(i))
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
        let bytes: [u8; 8] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let i = i64::from_ne_bytes(bytes);
        if i == i64::MIN {
            Ok(zarrs_metadata::v3::FillValueMetadataV3::from("NaT"))
        } else {
            Ok(zarrs_metadata::v3::FillValueMetadataV3::from(i))
        }
    }

    fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
        Some(self)
    }

    fn codec_bitround(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBitroundCodec> {
        Some(self)
    }

    fn codec_pcodec(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionPcodecCodec> {
        Some(self)
    }

    fn codec_zfp(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionZfpCodec> {
        Some(self)
    }
}

impl zarrs_data_type::DataTypeExtensionBytesCodec for NumpyDateTime64DataType {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        let endianness = endianness
            .ok_or(zarrs_data_type::DataTypeExtensionBytesCodecError::EndiannessNotSpecified)?;
        if endianness == zarrs_metadata::Endianness::native() {
            Ok(bytes)
        } else {
            let mut result = bytes.into_owned();
            for chunk in result.chunks_exact_mut(8) {
                chunk.reverse();
            }
            Ok(std::borrow::Cow::Owned(result))
        }
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        self.encode(bytes, endianness)
    }
}

// NumpyTimeDelta64 - 8 bytes (i64)
impl zarrs_data_type::DataTypeExtension for NumpyTimeDelta64DataType {
    fn identifier(&self) -> &'static str {
        <Self as ExtensionIdentifier>::IDENTIFIER
    }

    fn configuration(&self) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::from(NumpyTimeDelta64DataTypeConfigurationV1 {
            unit: self.unit,
            scale_factor: self.scale_factor,
        })
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(8)
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
        // Handle "NaT" (Not a Time) as i64::MIN
        if let Some("NaT") = fill_value_metadata.as_str() {
            return Ok(zarrs_data_type::FillValue::from(i64::MIN));
        }
        // Otherwise expect an integer
        let i = fill_value_metadata.as_i64().ok_or_else(err)?;
        Ok(zarrs_data_type::FillValue::from(i))
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
        let bytes: [u8; 8] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let i = i64::from_ne_bytes(bytes);
        if i == i64::MIN {
            Ok(zarrs_metadata::v3::FillValueMetadataV3::from("NaT"))
        } else {
            Ok(zarrs_metadata::v3::FillValueMetadataV3::from(i))
        }
    }

    fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
        Some(self)
    }

    fn codec_bitround(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBitroundCodec> {
        Some(self)
    }

    fn codec_pcodec(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionPcodecCodec> {
        Some(self)
    }

    fn codec_zfp(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionZfpCodec> {
        Some(self)
    }
}

impl zarrs_data_type::DataTypeExtensionBytesCodec for NumpyTimeDelta64DataType {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        let endianness = endianness
            .ok_or(zarrs_data_type::DataTypeExtensionBytesCodecError::EndiannessNotSpecified)?;
        if endianness == zarrs_metadata::Endianness::native() {
            Ok(bytes)
        } else {
            let mut result = bytes.into_owned();
            for chunk in result.chunks_exact_mut(8) {
                chunk.reverse();
            }
            Ok(std::borrow::Cow::Owned(result))
        }
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        self.encode(bytes, endianness)
    }
}

// Pcodec implementations for NumpyDateTime64 and NumpyTimeDelta64
// They behave like 64-bit signed integers
impl zarrs_data_type::DataTypeExtensionPcodecCodec for NumpyDateTime64DataType {
    fn pcodec_element_type(&self) -> Option<zarrs_data_type::PcodecElementType> {
        Some(zarrs_data_type::PcodecElementType::I64)
    }
}

impl zarrs_data_type::DataTypeExtensionPcodecCodec for NumpyTimeDelta64DataType {
    fn pcodec_element_type(&self) -> Option<zarrs_data_type::PcodecElementType> {
        Some(zarrs_data_type::PcodecElementType::I64)
    }
}

// Bitround implementations for NumpyDateTime64 and NumpyTimeDelta64
// They behave like 64-bit integers (same as Int64/UInt64)
impl zarrs_data_type::DataTypeExtensionBitroundCodec for NumpyDateTime64DataType {
    fn mantissa_bits(&self) -> Option<u32> {
        None // Integer type - rounds from MSB
    }

    fn component_size(&self) -> usize {
        8
    }

    fn round(&self, bytes: &mut [u8], keepbits: u32) {
        zarrs_data_type::round_bytes_int64(bytes, keepbits);
    }
}

impl zarrs_data_type::DataTypeExtensionBitroundCodec for NumpyTimeDelta64DataType {
    fn mantissa_bits(&self) -> Option<u32> {
        None // Integer type - rounds from MSB
    }

    fn component_size(&self) -> usize {
        8
    }

    fn round(&self, bytes: &mut [u8], keepbits: u32) {
        zarrs_data_type::round_bytes_int64(bytes, keepbits);
    }
}

// ZFP codec implementations for NumpyDateTime64 and NumpyTimeDelta64
// They behave like 64-bit signed integers (no promotion needed)
impl zarrs_data_type::DataTypeExtensionZfpCodec for NumpyDateTime64DataType {
    fn zfp_type(&self) -> Option<zarrs_data_type::ZfpType> {
        Some(zarrs_data_type::ZfpType::Int64)
    }

    fn zfp_promotion(&self) -> zarrs_data_type::ZfpPromotion {
        zarrs_data_type::ZfpPromotion::None
    }
}

impl zarrs_data_type::DataTypeExtensionZfpCodec for NumpyTimeDelta64DataType {
    fn zfp_type(&self) -> Option<zarrs_data_type::ZfpType> {
        Some(zarrs_data_type::ZfpType::Int64)
    }

    fn zfp_promotion(&self) -> zarrs_data_type::ZfpPromotion {
        zarrs_data_type::ZfpPromotion::None
    }
}

// Plugin registrations
register_data_type_plugin!(BoolDataType);
register_data_type_plugin!(StringDataType);
register_data_type_plugin!(BytesDataType);

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

// Custom plugin registration for NumpyDateTime64DataType (has configuration)
inventory::submit! {
    zarrs_data_type::DataTypePlugin::new(
        <NumpyDateTime64DataType as ExtensionIdentifier>::IDENTIFIER,
        <NumpyDateTime64DataType as ExtensionIdentifier>::matches_name,
        <NumpyDateTime64DataType as ExtensionIdentifier>::default_name,
        |metadata: &zarrs_metadata::v3::MetadataV3| -> Result<std::sync::Arc<dyn zarrs_data_type::DataTypeExtension>, PluginCreateError> {
            let configuration = metadata.configuration().ok_or_else(|| {
                PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                    NumpyDateTime64DataType::IDENTIFIER,
                    "data_type",
                    "missing configuration".to_string(),
                ))
            })?;
            let config = NumpyDateTime64DataTypeConfigurationV1::try_from_configuration(configuration.clone())
                .map_err(|_| {
                    PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                        NumpyDateTime64DataType::IDENTIFIER,
                        "data_type",
                        metadata.to_string(),
                    ))
                })?;
            Ok(std::sync::Arc::new(NumpyDateTime64DataType::new(config.unit, config.scale_factor)))
        },
    )
}

// Custom plugin registration for NumpyTimeDelta64DataType (has configuration)
inventory::submit! {
    zarrs_data_type::DataTypePlugin::new(
        <NumpyTimeDelta64DataType as ExtensionIdentifier>::IDENTIFIER,
        <NumpyTimeDelta64DataType as ExtensionIdentifier>::matches_name,
        <NumpyTimeDelta64DataType as ExtensionIdentifier>::default_name,
        |metadata: &zarrs_metadata::v3::MetadataV3| -> Result<std::sync::Arc<dyn zarrs_data_type::DataTypeExtension>, PluginCreateError> {
            let configuration = metadata.configuration().ok_or_else(|| {
                PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                    NumpyTimeDelta64DataType::IDENTIFIER,
                    "data_type",
                    "missing configuration".to_string(),
                ))
            })?;
            let config = NumpyTimeDelta64DataTypeConfigurationV1::try_from_configuration(configuration.clone())
                .map_err(|_| {
                    PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                        NumpyTimeDelta64DataType::IDENTIFIER,
                        "data_type",
                        metadata.to_string(),
                    ))
                })?;
            Ok(std::sync::Arc::new(NumpyTimeDelta64DataType::new(config.unit, config.scale_factor)))
        },
    )
}
