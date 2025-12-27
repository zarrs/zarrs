//! Marker types for data types that implement [`ExtensionIdentifier`].
//!
//! Each marker type provides per-data-type runtime-configurable aliases for both
//! Zarr V2 and V3 formats.

use std::sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

use zarrs_plugin::{
    ExtensionAliases, ExtensionAliasesConfig, ExtensionIdentifier, Regex, ZarrVersion2,
    ZarrVersion3,
};

/// Macro to define a data type marker with `ExtensionIdentifier` implementation.
macro_rules! data_type_marker {
    // Simple case: V3 identifier == V3 default name, V2 uses same identifier (no special aliases)
    ($marker:ident, $identifier:expr, $static_v3:ident) => {
        #[doc = concat!("Marker type for the `", stringify!($marker), "` data type.")]
        #[derive(Debug, Clone, Copy)]
        pub struct $marker;

        static $static_v3: LazyLock<RwLock<ExtensionAliasesConfig>> =
            LazyLock::new(|| RwLock::new(ExtensionAliasesConfig::new($identifier, vec![], vec![])));

        paste::paste! {
            static [<$static_v3 _V2>]: LazyLock<RwLock<ExtensionAliasesConfig>> =
                LazyLock::new(|| RwLock::new(ExtensionAliasesConfig::new($identifier, vec![], vec![])));
        }

        impl ExtensionAliases<ZarrVersion3> for $marker {
            fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
                $static_v3.read().unwrap()
            }

            fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
                $static_v3.write().unwrap()
            }
        }

        paste::paste! {
            impl ExtensionAliases<ZarrVersion2> for $marker {
                fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
                    [<$static_v3 _V2>].read().unwrap()
                }

                fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
                    [<$static_v3 _V2>].write().unwrap()
                }
            }
        }

        impl ExtensionIdentifier for $marker {
            const IDENTIFIER: &'static str = $identifier;
        }
    };

    // Case with V3 custom aliases only (V2 uses identifier)
    ($marker:ident, $identifier:expr, $static_v3:ident, $v3_default:expr, $v3_aliases:expr) => {
        #[doc = concat!("Marker type for the `", stringify!($marker), "` data type.")]
        #[derive(Debug, Clone, Copy)]
        pub struct $marker;

        static $static_v3: LazyLock<RwLock<ExtensionAliasesConfig>> = LazyLock::new(|| {
            RwLock::new(ExtensionAliasesConfig::new(
                $v3_default,
                $v3_aliases,
                vec![],
            ))
        });

        paste::paste! {
            static [<$static_v3 _V2>]: LazyLock<RwLock<ExtensionAliasesConfig>> =
                LazyLock::new(|| RwLock::new(ExtensionAliasesConfig::new($identifier, vec![], vec![])));
        }

        impl ExtensionAliases<ZarrVersion3> for $marker {
            fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
                $static_v3.read().unwrap()
            }

            fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
                $static_v3.write().unwrap()
            }
        }

        paste::paste! {
            impl ExtensionAliases<ZarrVersion2> for $marker {
                fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
                    [<$static_v3 _V2>].read().unwrap()
                }

                fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
                    [<$static_v3 _V2>].write().unwrap()
                }
            }
        }

        impl ExtensionIdentifier for $marker {
            const IDENTIFIER: &'static str = $identifier;
        }
    };
}

/// Macro to define a data type marker with V2 NumPy-style aliases.
macro_rules! data_type_marker_v2 {
    // V3 uses identifier, V2 has custom default name and aliases
    ($marker:ident, $identifier:expr, $static_v3:ident, $v2_default:expr, $v2_aliases:expr) => {
        #[doc = concat!("Marker type for the `", stringify!($marker), "` data type.")]
        #[derive(Debug, Clone, Copy)]
        pub struct $marker;

        static $static_v3: LazyLock<RwLock<ExtensionAliasesConfig>> =
            LazyLock::new(|| RwLock::new(ExtensionAliasesConfig::new($identifier, vec![], vec![])));

        paste::paste! {
            static [<$static_v3 _V2>]: LazyLock<RwLock<ExtensionAliasesConfig>> = LazyLock::new(|| {
                RwLock::new(ExtensionAliasesConfig::new(
                    $v2_default,
                    $v2_aliases,
                    vec![],
                ))
            });
        }

        impl ExtensionAliases<ZarrVersion3> for $marker {
            fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
                $static_v3.read().unwrap()
            }

            fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
                $static_v3.write().unwrap()
            }
        }

        paste::paste! {
            impl ExtensionAliases<ZarrVersion2> for $marker {
                fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
                    [<$static_v3 _V2>].read().unwrap()
                }

                fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
                    [<$static_v3 _V2>].write().unwrap()
                }
            }
        }

        impl ExtensionIdentifier for $marker {
            const IDENTIFIER: &'static str = $identifier;
        }
    };

    // V3 has custom aliases, V2 has custom aliases
    ($marker:ident, $identifier:expr, $static_v3:ident,
     $v3_default:expr, $v3_aliases:expr,
     $v2_default:expr, $v2_aliases:expr) => {
        #[doc = concat!("Marker type for the `", stringify!($marker), "` data type.")]
        #[derive(Debug, Clone, Copy)]
        pub struct $marker;

        static $static_v3: LazyLock<RwLock<ExtensionAliasesConfig>> = LazyLock::new(|| {
            RwLock::new(ExtensionAliasesConfig::new(
                $v3_default,
                $v3_aliases,
                vec![],
            ))
        });

        paste::paste! {
            static [<$static_v3 _V2>]: LazyLock<RwLock<ExtensionAliasesConfig>> = LazyLock::new(|| {
                RwLock::new(ExtensionAliasesConfig::new(
                    $v2_default,
                    $v2_aliases,
                    vec![],
                ))
            });
        }

        impl ExtensionAliases<ZarrVersion3> for $marker {
            fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
                $static_v3.read().unwrap()
            }

            fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
                $static_v3.write().unwrap()
            }
        }

        paste::paste! {
            impl ExtensionAliases<ZarrVersion2> for $marker {
                fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
                    [<$static_v3 _V2>].read().unwrap()
                }

                fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
                    [<$static_v3 _V2>].write().unwrap()
                }
            }
        }

        impl ExtensionIdentifier for $marker {
            const IDENTIFIER: &'static str = $identifier;
        }
    };

    // V3 uses identifier, V2 has custom default name, aliases, and regex
    ($marker:ident, $identifier:expr, $static_v3:ident, $v2_default:expr, $v2_aliases:expr, $v2_regex:expr) => {
        #[doc = concat!("Marker type for the `", stringify!($marker), "` data type.")]
        #[derive(Debug, Clone, Copy)]
        pub struct $marker;

        static $static_v3: LazyLock<RwLock<ExtensionAliasesConfig>> =
            LazyLock::new(|| RwLock::new(ExtensionAliasesConfig::new($identifier)));

        paste::paste! {
            static [<$static_v3 _V2>]: LazyLock<RwLock<ExtensionAliasesConfig>> = LazyLock::new(|| {
                RwLock::new(ExtensionAliasesConfig::new(
                    $identifier,
                    $v2_default,
                    $v2_aliases,
                    $v2_regex,
                ))
            });
        }

        impl ExtensionAliases<ZarrVersion3> for $marker {
            fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
                $static_v3.read().unwrap()
            }

            fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
                $static_v3.write().unwrap()
            }
        }

        paste::paste! {
            impl ExtensionAliases<ZarrVersion2> for $marker {
                fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
                    [<$static_v3 _V2>].read().unwrap()
                }

                fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
                    [<$static_v3 _V2>].write().unwrap()
                }
            }
        }

        impl ExtensionIdentifier for $marker {
            const IDENTIFIER: &'static str = $identifier;
        }
    };
}

// Boolean - V2: |b1
data_type_marker_v2!(
    BoolDataType,
    "bool",
    BOOL_ALIASES,
    "|b1",
    vec!["|b1".into()]
);

// Signed integers - V2: |i1, <i2, <i4, <i8 (and > variants)
data_type_marker!(Int2DataType, "int2", INT2_ALIASES); // No V2 equivalent
data_type_marker!(Int4DataType, "int4", INT4_ALIASES); // No V2 equivalent
data_type_marker_v2!(
    Int8DataType,
    "int8",
    INT8_ALIASES,
    "|i1",
    vec!["|i1".into()]
);
data_type_marker_v2!(
    Int16DataType,
    "int16",
    INT16_ALIASES,
    "<i2",
    vec!["<i2".into(), ">i2".into()]
);
data_type_marker_v2!(
    Int32DataType,
    "int32",
    INT32_ALIASES,
    "<i4",
    vec!["<i4".into(), ">i4".into()]
);
data_type_marker_v2!(
    Int64DataType,
    "int64",
    INT64_ALIASES,
    "<i8",
    vec!["<i8".into(), ">i8".into()]
);

// Unsigned integers - V2: |u1, <u2, <u4, <u8 (and > variants)
data_type_marker!(UInt2DataType, "uint2", UINT2_ALIASES); // No V2 equivalent
data_type_marker!(UInt4DataType, "uint4", UINT4_ALIASES); // No V2 equivalent
data_type_marker_v2!(
    UInt8DataType,
    "uint8",
    UINT8_ALIASES,
    "|u1",
    vec!["|u1".into()]
);
data_type_marker_v2!(
    UInt16DataType,
    "uint16",
    UINT16_ALIASES,
    "<u2",
    vec!["<u2".into(), ">u2".into()]
);
data_type_marker_v2!(
    UInt32DataType,
    "uint32",
    UINT32_ALIASES,
    "<u4",
    vec!["<u4".into(), ">u4".into()]
);
data_type_marker_v2!(
    UInt64DataType,
    "uint64",
    UINT64_ALIASES,
    "<u8",
    vec!["<u8".into(), ">u8".into()]
);

// Subfloats - No V2 equivalents
data_type_marker!(Float4E2M1FNDataType, "float4_e2m1fn", FLOAT4_E2M1FN_ALIASES);
data_type_marker!(Float6E2M3FNDataType, "float6_e2m3fn", FLOAT6_E2M3FN_ALIASES);
data_type_marker!(Float6E3M2FNDataType, "float6_e3m2fn", FLOAT6_E3M2FN_ALIASES);
data_type_marker!(Float8E3M4DataType, "float8_e3m4", FLOAT8_E3M4_ALIASES);
data_type_marker!(Float8E4M3DataType, "float8_e4m3", FLOAT8_E4M3_ALIASES);
data_type_marker!(
    Float8E4M3B11FNUZDataType,
    "float8_e4m3b11fnuz",
    FLOAT8_E4M3B11FNUZ_ALIASES
);
data_type_marker!(
    Float8E4M3FNUZDataType,
    "float8_e4m3fnuz",
    FLOAT8_E4M3FNUZ_ALIASES
);
data_type_marker!(Float8E5M2DataType, "float8_e5m2", FLOAT8_E5M2_ALIASES);
data_type_marker!(
    Float8E5M2FNUZDataType,
    "float8_e5m2fnuz",
    FLOAT8_E5M2FNUZ_ALIASES
);
data_type_marker!(
    Float8E8M0FNUDataType,
    "float8_e8m0fnu",
    FLOAT8_E8M0FNU_ALIASES
);

// Standard floats - V2: <f2, <f4, <f8 (and > variants), no bfloat16
data_type_marker!(BFloat16DataType, "bfloat16", BFLOAT16_ALIASES); // No V2 equivalent
data_type_marker_v2!(
    Float16DataType,
    "float16",
    FLOAT16_ALIASES,
    "<f2",
    vec!["<f2".into(), ">f2".into()]
);
data_type_marker_v2!(
    Float32DataType,
    "float32",
    FLOAT32_ALIASES,
    "<f4",
    vec!["<f4".into(), ">f4".into()]
);
data_type_marker_v2!(
    Float64DataType,
    "float64",
    FLOAT64_ALIASES,
    "<f8",
    vec!["<f8".into(), ">f8".into()]
);

// Complex subfloats - No V2 equivalents
data_type_marker!(
    ComplexFloat4E2M1FNDataType,
    "complex_float4_e2m1fn",
    COMPLEX_FLOAT4_E2M1FN_ALIASES
);
data_type_marker!(
    ComplexFloat6E2M3FNDataType,
    "complex_float6_e2m3fn",
    COMPLEX_FLOAT6_E2M3FN_ALIASES
);
data_type_marker!(
    ComplexFloat6E3M2FNDataType,
    "complex_float6_e3m2fn",
    COMPLEX_FLOAT6_E3M2FN_ALIASES
);
data_type_marker!(
    ComplexFloat8E3M4DataType,
    "complex_float8_e3m4",
    COMPLEX_FLOAT8_E3M4_ALIASES
);
data_type_marker!(
    ComplexFloat8E4M3DataType,
    "complex_float8_e4m3",
    COMPLEX_FLOAT8_E4M3_ALIASES
);
data_type_marker!(
    ComplexFloat8E4M3B11FNUZDataType,
    "complex_float8_e4m3b11fnuz",
    COMPLEX_FLOAT8_E4M3B11FNUZ_ALIASES
);
data_type_marker!(
    ComplexFloat8E4M3FNUZDataType,
    "complex_float8_e4m3fnuz",
    COMPLEX_FLOAT8_E4M3FNUZ_ALIASES
);
data_type_marker!(
    ComplexFloat8E5M2DataType,
    "complex_float8_e5m2",
    COMPLEX_FLOAT8_E5M2_ALIASES
);
data_type_marker!(
    ComplexFloat8E5M2FNUZDataType,
    "complex_float8_e5m2fnuz",
    COMPLEX_FLOAT8_E5M2FNUZ_ALIASES
);
data_type_marker!(
    ComplexFloat8E8M0FNUDataType,
    "complex_float8_e8m0fnu",
    COMPLEX_FLOAT8_E8M0FNU_ALIASES
);

// Complex floats - V2: <c8, <c16 (and > variants)
data_type_marker!(
    ComplexBFloat16DataType,
    "complex_bfloat16",
    COMPLEX_BFLOAT16_ALIASES
); // No V2 equivalent
data_type_marker!(
    ComplexFloat16DataType,
    "complex_float16",
    COMPLEX_FLOAT16_ALIASES
); // No V2 equivalent
data_type_marker!(
    ComplexFloat32DataType,
    "complex_float32",
    COMPLEX_FLOAT32_ALIASES
); // No V2 equivalent
data_type_marker!(
    ComplexFloat64DataType,
    "complex_float64",
    COMPLEX_FLOAT64_ALIASES
); // No V2 equivalent
data_type_marker_v2!(
    Complex64DataType,
    "complex64",
    COMPLEX64_ALIASES,
    "<c8",
    vec!["<c8".into(), ">c8".into()]
);
data_type_marker_v2!(
    Complex128DataType,
    "complex128",
    COMPLEX128_ALIASES,
    "<c16",
    vec!["<c16".into(), ">c16".into()]
);

// Variable-length types - V2: |O for string, |V\d+ for bytes
data_type_marker_v2!(
    StringDataType,
    "string",
    STRING_ALIASES,
    "|O",
    vec!["|O".into()]
);
data_type_marker_v2!(
    BytesDataType,
    "bytes",
    BYTES_ALIASES,
    "bytes",
    vec!["binary".into(), "variable_length_bytes".into()],
    "|VX",
    vec!["|VX".into()]
);

// Special types

/// Marker type for the `RawBitsDataType` data type.
#[derive(Debug, Clone, Copy)]
pub struct RawBitsDataType;

static RAWBITS_ALIASES_V3: LazyLock<RwLock<ExtensionAliasesConfig>> = LazyLock::new(|| {
    RwLock::new(ExtensionAliasesConfig::new(
        "r*",
        vec![],
        vec![Regex::new(r"^r\d+$").unwrap()],
    ))
});

static RAWBITS_ALIASES_V2: LazyLock<RwLock<ExtensionAliasesConfig>> = LazyLock::new(|| {
    RwLock::new(ExtensionAliasesConfig::new(
        "r*",
        vec![],
        vec![
            Regex::new(r"^r\d+$").unwrap(),
            Regex::new(r"^\|V\d+$").unwrap(),
        ],
    ))
});

impl ExtensionAliases<ZarrVersion3> for RawBitsDataType {
    fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
        RAWBITS_ALIASES_V3.read().unwrap()
    }

    fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
        RAWBITS_ALIASES_V3.write().unwrap()
    }
}

impl ExtensionAliases<ZarrVersion2> for RawBitsDataType {
    fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
        RAWBITS_ALIASES_V2.read().unwrap()
    }

    fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
        RAWBITS_ALIASES_V2.write().unwrap()
    }
}

impl ExtensionIdentifier for RawBitsDataType {
    const IDENTIFIER: &'static str = "r*";
}
data_type_marker!(
    OptionalDataType,
    "optional",
    OPTIONAL_ALIASES,
    "zarrs.optional",
    vec!["zarrs.optional".into()]
);
data_type_marker!(
    NumpyDateTime64DataType,
    "numpy.datetime64",
    NUMPY_DATETIME64_ALIASES
);
data_type_marker!(
    NumpyTimeDelta64DataType,
    "numpy.timedelta64",
    NUMPY_TIMEDELTA64_ALIASES
);
