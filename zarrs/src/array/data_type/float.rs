//! Float data type markers and implementations.

use super::macros::{impl_data_type_extension_numeric, register_data_type_plugin};

// Standard floats - V2: <f2, <f4, <f8 (and > variants), no bfloat16

/// The `bfloat16` data type.
#[derive(Debug, Clone, Copy)]
pub struct BFloat16DataType;
zarrs_plugin::impl_extension_aliases!(BFloat16DataType, "bfloat16");

/// The `float16` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float16DataType;
zarrs_plugin::impl_extension_aliases!(Float16DataType, "float16",
    v3: "float16", [],
    v2: "<f2", ["<f2", ">f2"]
);

/// The `float32` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float32DataType;
zarrs_plugin::impl_extension_aliases!(Float32DataType, "float32",
    v3: "float32", [],
    v2: "<f4", ["<f4", ">f4"]
);

/// The `float64` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float64DataType;
zarrs_plugin::impl_extension_aliases!(Float64DataType, "float64",
    v3: "float64", [],
    v2: "<f8", ["<f8", ">f8"]
);

// DataTypeExtension implementations
impl_data_type_extension_numeric!(BFloat16DataType, 2, bf16);
impl_data_type_extension_numeric!(Float16DataType, 2, f16);
impl_data_type_extension_numeric!(Float32DataType, 4, f32);
impl_data_type_extension_numeric!(Float64DataType, 8, f64);

// Plugin registrations
register_data_type_plugin!(BFloat16DataType);
register_data_type_plugin!(Float16DataType);
register_data_type_plugin!(Float32DataType);
register_data_type_plugin!(Float64DataType);
