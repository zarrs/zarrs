//! Special data type markers and implementations (bool, string, bytes, rawbits, optional, numpy time types).

use zarrs_plugin::{ExtensionIdentifier, Regex};

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
#[derive(Debug, Clone, Copy)]
pub struct RawBitsDataType;
zarrs_plugin::impl_extension_aliases!(RawBitsDataType, "r*",
    v3: "r*", [], [Regex::new(r"^r\d+$").unwrap()],
    v2: "r*", [], [Regex::new(r"^r\d+$").unwrap(), Regex::new(r"^\|V\d+$").unwrap()]
);

// NumPy time types

/// The `numpy.datetime64` data type.
#[derive(Debug, Clone, Copy)]
pub struct NumpyDateTime64DataType;
zarrs_plugin::impl_extension_aliases!(NumpyDateTime64DataType, "numpy.datetime64");

/// The `numpy.timedelta64` data type.
#[derive(Debug, Clone, Copy)]
pub struct NumpyTimeDelta64DataType;
zarrs_plugin::impl_extension_aliases!(NumpyTimeDelta64DataType, "numpy.timedelta64");

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

    fn codec_bytes(
        &self,
    ) -> Result<
        &dyn zarrs_data_type::DataTypeExtensionBytesCodec,
        zarrs_data_type::DataTypeExtensionError,
    > {
        Ok(self)
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

    fn codec_bytes(
        &self,
    ) -> Result<
        &dyn zarrs_data_type::DataTypeExtensionBytesCodec,
        zarrs_data_type::DataTypeExtensionError,
    > {
        Ok(self)
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

    fn codec_bytes(
        &self,
    ) -> Result<
        &dyn zarrs_data_type::DataTypeExtensionBytesCodec,
        zarrs_data_type::DataTypeExtensionError,
    > {
        Ok(self)
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

    fn configuration(&self) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        // RawBits size is determined by the name (r8 = 1 byte, r16 = 2 bytes, etc.)
        // The marker type doesn't know the size; actual size comes from NamedDataType
        zarrs_metadata::DataTypeSize::Fixed(1)
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

    fn codec_bytes(
        &self,
    ) -> Result<
        &dyn zarrs_data_type::DataTypeExtensionBytesCodec,
        zarrs_data_type::DataTypeExtensionError,
    > {
        Ok(self)
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

// Plugin registrations
register_data_type_plugin!(BoolDataType);
register_data_type_plugin!(StringDataType);
register_data_type_plugin!(BytesDataType);
register_data_type_plugin!(RawBitsDataType);
