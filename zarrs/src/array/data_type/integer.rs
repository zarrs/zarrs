//! Integer data type markers and implementations.

use super::macros::{impl_data_type_extension_numeric, register_data_type_plugin};

// Signed integers - V2: |i1, <i2, <i4, <i8 (and > variants)

/// The `int2` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int2DataType;
zarrs_plugin::impl_extension_aliases!(Int2DataType, "int2");

/// The `int4` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int4DataType;
zarrs_plugin::impl_extension_aliases!(Int4DataType, "int4");

/// The `int8` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int8DataType;
zarrs_plugin::impl_extension_aliases!(Int8DataType, "int8",
    v3: "int8", [],
    v2: "|i1", ["|i1"]
);

/// The `int16` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int16DataType;
zarrs_plugin::impl_extension_aliases!(Int16DataType, "int16",
    v3: "int16", [],
    v2: "<i2", ["<i2", ">i2"]
);

/// The `int32` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int32DataType;
zarrs_plugin::impl_extension_aliases!(Int32DataType, "int32",
    v3: "int32", [],
    v2: "<i4", ["<i4", ">i4"]
);

/// The `int64` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int64DataType;
zarrs_plugin::impl_extension_aliases!(Int64DataType, "int64",
    v3: "int64", [],
    v2: "<i8", ["<i8", ">i8"]
);

// Unsigned integers - V2: |u1, <u2, <u4, <u8 (and > variants)

/// The `uint2` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt2DataType;
zarrs_plugin::impl_extension_aliases!(UInt2DataType, "uint2");

/// The `uint4` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt4DataType;
zarrs_plugin::impl_extension_aliases!(UInt4DataType, "uint4");

/// The `uint8` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt8DataType;
zarrs_plugin::impl_extension_aliases!(UInt8DataType, "uint8",
    v3: "uint8", [],
    v2: "|u1", ["|u1"]
);

/// The `uint16` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt16DataType;
zarrs_plugin::impl_extension_aliases!(UInt16DataType, "uint16",
    v3: "uint16", [],
    v2: "<u2", ["<u2", ">u2"]
);

/// The `uint32` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt32DataType;
zarrs_plugin::impl_extension_aliases!(UInt32DataType, "uint32",
    v3: "uint32", [],
    v2: "<u4", ["<u4", ">u4"]
);

/// The `uint64` data type.
#[derive(Debug, Clone, Copy)]
pub struct UInt64DataType;
zarrs_plugin::impl_extension_aliases!(UInt64DataType, "uint64",
    v3: "uint64", [],
    v2: "<u8", ["<u8", ">u8"]
);

// DataTypeExtension implementations for sub-byte integers (Int2, Int4, UInt2, UInt4)
// These require special handling for packing/range validation

impl zarrs_data_type::DataTypeExtension for Int2DataType {
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
        let int = fill_value_metadata.as_i64().ok_or_else(err)?;
        // int2 range: -2 to 1
        if !(-2..=1).contains(&int) {
            return Err(err());
        }
        #[expect(clippy::cast_possible_truncation)]
        Ok(zarrs_data_type::FillValue::from(int as i8))
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
        let number = i8::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
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

impl zarrs_data_type::DataTypeExtensionBytesCodec for Int2DataType {
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

impl zarrs_data_type::DataTypeExtension for Int4DataType {
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
        let int = fill_value_metadata.as_i64().ok_or_else(err)?;
        // int4 range: -8 to 7
        if !(-8..=7).contains(&int) {
            return Err(err());
        }
        #[expect(clippy::cast_possible_truncation)]
        Ok(zarrs_data_type::FillValue::from(int as i8))
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
        let number = i8::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
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

impl zarrs_data_type::DataTypeExtensionBytesCodec for Int4DataType {
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

    fn codec_bytes(
        &self,
    ) -> Result<
        &dyn zarrs_data_type::DataTypeExtensionBytesCodec,
        zarrs_data_type::DataTypeExtensionError,
    > {
        Ok(self)
    }
}

impl zarrs_data_type::DataTypeExtensionBytesCodec for UInt2DataType {
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

    fn codec_bytes(
        &self,
    ) -> Result<
        &dyn zarrs_data_type::DataTypeExtensionBytesCodec,
        zarrs_data_type::DataTypeExtensionError,
    > {
        Ok(self)
    }
}

impl zarrs_data_type::DataTypeExtensionBytesCodec for UInt4DataType {
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

// DataTypeExtension implementations for standard integers using macro
impl_data_type_extension_numeric!(Int8DataType, 1, i8);
impl_data_type_extension_numeric!(Int16DataType, 2, i16);
impl_data_type_extension_numeric!(Int32DataType, 4, i32);
impl_data_type_extension_numeric!(Int64DataType, 8, i64);
impl_data_type_extension_numeric!(UInt8DataType, 1, u8);
impl_data_type_extension_numeric!(UInt16DataType, 2, u16);
impl_data_type_extension_numeric!(UInt32DataType, 4, u32);
impl_data_type_extension_numeric!(UInt64DataType, 8, u64);

// Plugin registrations
register_data_type_plugin!(Int2DataType);
register_data_type_plugin!(Int4DataType);
register_data_type_plugin!(Int8DataType);
register_data_type_plugin!(Int16DataType);
register_data_type_plugin!(Int32DataType);
register_data_type_plugin!(Int64DataType);
register_data_type_plugin!(UInt2DataType);
register_data_type_plugin!(UInt4DataType);
register_data_type_plugin!(UInt8DataType);
register_data_type_plugin!(UInt16DataType);
register_data_type_plugin!(UInt32DataType);
register_data_type_plugin!(UInt64DataType);
