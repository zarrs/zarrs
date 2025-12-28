//! Complex data type markers and implementations.

use super::macros::{
    impl_complex_data_type, impl_complex_subfloat_data_type, register_data_type_plugin,
};

// Complex floats - V2: <c8, <c16 (and > variants)

/// The `complex_bfloat16` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexBFloat16DataType;
zarrs_plugin::impl_extension_aliases!(ComplexBFloat16DataType, "complex_bfloat16");

/// The `complex_float16` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat16DataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat16DataType, "complex_float16");

/// The `complex_float32` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat32DataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat32DataType, "complex_float32");

/// The `complex_float64` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat64DataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat64DataType, "complex_float64");

/// The `complex64` data type.
#[derive(Debug, Clone, Copy)]
pub struct Complex64DataType;
zarrs_plugin::impl_extension_aliases!(Complex64DataType, "complex64",
    v3: "complex64", [],
    v2: "<c8", ["<c8", ">c8"]
);

/// The `complex128` data type.
#[derive(Debug, Clone, Copy)]
pub struct Complex128DataType;
zarrs_plugin::impl_extension_aliases!(Complex128DataType, "complex128",
    v3: "complex128", [],
    v2: "<c16", ["<c16", ">c16"]
);

// Complex subfloats - No V2 equivalents

/// The `complex_float4_e2m1fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat4E2M1FNDataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat4E2M1FNDataType, "complex_float4_e2m1fn");

/// The `complex_float6_e2m3fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat6E2M3FNDataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat6E2M3FNDataType, "complex_float6_e2m3fn");

/// The `complex_float6_e3m2fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat6E3M2FNDataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat6E3M2FNDataType, "complex_float6_e3m2fn");

/// The `complex_float8_e3m4` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E3M4DataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E3M4DataType, "complex_float8_e3m4");

/// The `complex_float8_e4m3` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E4M3DataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E4M3DataType, "complex_float8_e4m3");

/// The `complex_float8_e4m3b11fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E4M3B11FNUZDataType;
zarrs_plugin::impl_extension_aliases!(
    ComplexFloat8E4M3B11FNUZDataType,
    "complex_float8_e4m3b11fnuz"
);

/// The `complex_float8_e4m3fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E4M3FNUZDataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E4M3FNUZDataType, "complex_float8_e4m3fnuz");

/// The `complex_float8_e5m2` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E5M2DataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E5M2DataType, "complex_float8_e5m2");

/// The `complex_float8_e5m2fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E5M2FNUZDataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E5M2FNUZDataType, "complex_float8_e5m2fnuz");

/// The `complex_float8_e8m0fnu` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat8E8M0FNUDataType;
zarrs_plugin::impl_extension_aliases!(ComplexFloat8E8M0FNUDataType, "complex_float8_e8m0fnu");

// DataTypeExtension implementations for standard complex floats
impl_complex_data_type!(ComplexBFloat16DataType, 4, bf16);
impl_complex_data_type!(ComplexFloat16DataType, 4, f16);
impl_complex_data_type!(ComplexFloat32DataType, 8, f32);
impl_complex_data_type!(ComplexFloat64DataType, 16, f64);
impl_complex_data_type!(Complex64DataType, 8, f32);
impl_complex_data_type!(Complex128DataType, 16, f64);

// DataTypeExtension implementations for complex subfloats
impl_complex_subfloat_data_type!(ComplexFloat4E2M1FNDataType);
impl_complex_subfloat_data_type!(ComplexFloat6E2M3FNDataType);
impl_complex_subfloat_data_type!(ComplexFloat6E3M2FNDataType);
impl_complex_subfloat_data_type!(ComplexFloat8E3M4DataType);
impl_complex_subfloat_data_type!(ComplexFloat8E4M3DataType);
impl_complex_subfloat_data_type!(ComplexFloat8E4M3B11FNUZDataType);
impl_complex_subfloat_data_type!(ComplexFloat8E4M3FNUZDataType);
impl_complex_subfloat_data_type!(ComplexFloat8E5M2DataType);
impl_complex_subfloat_data_type!(ComplexFloat8E5M2FNUZDataType);
impl_complex_subfloat_data_type!(ComplexFloat8E8M0FNUDataType);

// Plugin registrations
register_data_type_plugin!(ComplexBFloat16DataType);
register_data_type_plugin!(ComplexFloat16DataType);
register_data_type_plugin!(ComplexFloat32DataType);
register_data_type_plugin!(ComplexFloat64DataType);
register_data_type_plugin!(Complex64DataType);
register_data_type_plugin!(Complex128DataType);
register_data_type_plugin!(ComplexFloat4E2M1FNDataType);
register_data_type_plugin!(ComplexFloat6E2M3FNDataType);
register_data_type_plugin!(ComplexFloat6E3M2FNDataType);
register_data_type_plugin!(ComplexFloat8E3M4DataType);
register_data_type_plugin!(ComplexFloat8E4M3DataType);
register_data_type_plugin!(ComplexFloat8E4M3B11FNUZDataType);
register_data_type_plugin!(ComplexFloat8E4M3FNUZDataType);
register_data_type_plugin!(ComplexFloat8E5M2DataType);
register_data_type_plugin!(ComplexFloat8E5M2FNUZDataType);
register_data_type_plugin!(ComplexFloat8E8M0FNUDataType);
