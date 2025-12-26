use std::collections::HashMap;

use regex::Regex;

use crate::{data_type, ExtensionAliases, ExtensionTypeDataType, ZarrVersion2, ZarrVersion3};

/// Aliases for Zarr V3 *data type* extensions.
pub type ExtensionAliasesDataTypeV3 = ExtensionAliases<ZarrVersion3, ExtensionTypeDataType>;

/// Aliases for Zarr V2 *data type* extensions.
pub type ExtensionAliasesDataTypeV2 = ExtensionAliases<ZarrVersion2, ExtensionTypeDataType>;

impl Default for ExtensionAliasesDataTypeV3 {
    #[rustfmt::skip]
    fn default() -> Self {
        Self::new(
            // The default serialised `name`s
            HashMap::from([
                (data_type::OPTIONAL, "zarrs.optional".into()),
            ]),
            // `name` aliases (string match)
            HashMap::from([
                ("zarrs.optional".into(), data_type::OPTIONAL),
                ("binary".into(), data_type::BYTES), // ZEP0007 uses binary, zarr-python uses bytes
                ("variable_length_bytes".into(), data_type::BYTES), // zarr-python 3.1 uses variable_length_bytes
            ]),
            // `name` aliases (regex match)
            vec![
                (Regex::new(r"^r\d+$").expect("valid"), data_type::RAWBITS),
            ],
        )
    }
}

impl Default for ExtensionAliasesDataTypeV2 {
    #[rustfmt::skip]
    fn default() -> Self {
        Self::new(
            // The default serialised `name`s
            HashMap::from([]),
            // `name` aliases (string match)
            HashMap::from([
                ("|b1".into(), data_type::BOOL),
                ("|i1".into(), data_type::INT8),
                ("<i2".into(), data_type::INT16),
                (">i2".into(), data_type::INT16),
                ("<i4".into(), data_type::INT32),
                (">i4".into(), data_type::INT32),
                ("<i8".into(), data_type::INT64),
                (">i8".into(), data_type::INT64),
                ("|u1".into(), data_type::UINT8),
                ("<u2".into(), data_type::UINT16),
                (">u2".into(), data_type::UINT16),
                ("<u4".into(), data_type::UINT32),
                (">u4".into(), data_type::UINT32),
                ("<u8".into(), data_type::UINT64),
                (">u8".into(), data_type::UINT64),
                ("<f2".into(), data_type::FLOAT16),
                (">f2".into(), data_type::FLOAT16),
                ("<f4".into(), data_type::FLOAT32),
                (">f4".into(), data_type::FLOAT32),
                ("<f8".into(), data_type::FLOAT64),
                (">f8".into(), data_type::FLOAT64),
                ("<c8".into(), data_type::COMPLEX64),
                (">c8".into(), data_type::COMPLEX64),
                ("<c16".into(), data_type::COMPLEX128),
                (">c16".into(), data_type::COMPLEX128),
                ("|O".into(), data_type::STRING),
                ("|VX".into(), data_type::BYTES),
                // ("|mX".into(), "timedelta"),
                // ("|MX".into(), "datetime"),
                // ("|SX".into(), "fixedcharstring"),
                // ("|UX".into(), "fixedunicodestring"),
            ]),
            // `name` aliases (regex match)
            vec![
                (Regex::new(r"^\|V\d+$").expect("valid"), data_type::BYTES),
                // (Regex::new(r"^\|m\d+$").expect("valid"), "timedelta"),
                // (Regex::new(r"^\|M\d+$").expect("valid"), "datetime"),
                // (Regex::new(r"^\|S\d+$").expect("valid"), "fixedcharstring"),
                // (Regex::new(r"^\|U\d+$").expect("valid"), "fixedunicodestring"),
            ],
        )
    }
}
