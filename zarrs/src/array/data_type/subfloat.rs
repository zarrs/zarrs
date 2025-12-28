//! Subfloat data type markers and implementations.

use super::macros::{impl_subfloat_data_type, register_data_type_plugin};

// Subfloats - No V2 equivalents

/// The `float4_e2m1fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float4E2M1FNDataType;
zarrs_plugin::impl_extension_aliases!(Float4E2M1FNDataType, "float4_e2m1fn");

/// The `float6_e2m3fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float6E2M3FNDataType;
zarrs_plugin::impl_extension_aliases!(Float6E2M3FNDataType, "float6_e2m3fn");

/// The `float6_e3m2fn` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float6E3M2FNDataType;
zarrs_plugin::impl_extension_aliases!(Float6E3M2FNDataType, "float6_e3m2fn");

/// The `float8_e3m4` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E3M4DataType;
zarrs_plugin::impl_extension_aliases!(Float8E3M4DataType, "float8_e3m4");

/// The `float8_e4m3` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E4M3DataType;
zarrs_plugin::impl_extension_aliases!(Float8E4M3DataType, "float8_e4m3");

/// The `float8_e4m3b11fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E4M3B11FNUZDataType;
zarrs_plugin::impl_extension_aliases!(Float8E4M3B11FNUZDataType, "float8_e4m3b11fnuz");

/// The `float8_e4m3fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E4M3FNUZDataType;
zarrs_plugin::impl_extension_aliases!(Float8E4M3FNUZDataType, "float8_e4m3fnuz");

/// The `float8_e5m2` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E5M2DataType;
zarrs_plugin::impl_extension_aliases!(Float8E5M2DataType, "float8_e5m2");

/// The `float8_e5m2fnuz` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E5M2FNUZDataType;
zarrs_plugin::impl_extension_aliases!(Float8E5M2FNUZDataType, "float8_e5m2fnuz");

/// The `float8_e8m0fnu` data type.
#[derive(Debug, Clone, Copy)]
pub struct Float8E8M0FNUDataType;
zarrs_plugin::impl_extension_aliases!(Float8E8M0FNUDataType, "float8_e8m0fnu");

// DataTypeExtension implementations
impl_subfloat_data_type!(Float4E2M1FNDataType);
impl_subfloat_data_type!(Float6E2M3FNDataType);
impl_subfloat_data_type!(Float6E3M2FNDataType);
impl_subfloat_data_type!(Float8E3M4DataType);
impl_subfloat_data_type!(Float8E4M3DataType);
impl_subfloat_data_type!(Float8E4M3B11FNUZDataType);
impl_subfloat_data_type!(Float8E4M3FNUZDataType);
impl_subfloat_data_type!(Float8E5M2DataType);
impl_subfloat_data_type!(Float8E5M2FNUZDataType);
impl_subfloat_data_type!(Float8E8M0FNUDataType);

// Plugin registrations
register_data_type_plugin!(Float4E2M1FNDataType);
register_data_type_plugin!(Float6E2M3FNDataType);
register_data_type_plugin!(Float6E3M2FNDataType);
register_data_type_plugin!(Float8E3M4DataType);
register_data_type_plugin!(Float8E4M3DataType);
register_data_type_plugin!(Float8E4M3B11FNUZDataType);
register_data_type_plugin!(Float8E4M3FNUZDataType);
register_data_type_plugin!(Float8E5M2DataType);
register_data_type_plugin!(Float8E5M2FNUZDataType);
register_data_type_plugin!(Float8E8M0FNUDataType);
