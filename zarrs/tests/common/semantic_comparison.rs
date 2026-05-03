//! Semantic float comparison utilities for round-trip verification.
//!
//! This module provides functions to compare floating-point values with NaN-aware equality
//! (where NaN == NaN) while preserving the sign of zero (-0.0 != +0.0).

use half::{bf16, f16};
use zarrs::array::{ArrayBytes, DataType};

/// Compare two ArrayBytes with semantic float equality (NaN == NaN, but preserves sign of zero)
pub(crate) fn arrays_equal(a: &ArrayBytes, b: &ArrayBytes, data_type: &DataType) -> bool {
    // Get the raw bytes for comparison
    let a_bytes = match a {
        ArrayBytes::Fixed(bytes) => bytes.as_ref(),
        ArrayBytes::Variable(var_bytes) => var_bytes.bytes().as_ref(),
        ArrayBytes::Optional(opt_bytes) => match opt_bytes.data() {
            ArrayBytes::Fixed(bytes) => bytes.as_ref(),
            ArrayBytes::Variable(var_bytes) => var_bytes.bytes().as_ref(),
            ArrayBytes::Optional(_) => unreachable!("nested optional not expected"),
        },
    };

    let b_bytes = match b {
        ArrayBytes::Fixed(bytes) => bytes.as_ref(),
        ArrayBytes::Variable(var_bytes) => var_bytes.bytes().as_ref(),
        ArrayBytes::Optional(opt_bytes) => match opt_bytes.data() {
            ArrayBytes::Fixed(bytes) => bytes.as_ref(),
            ArrayBytes::Variable(var_bytes) => var_bytes.bytes().as_ref(),
            ArrayBytes::Optional(_) => unreachable!("nested optional not expected"),
        },
    };

    // Length must match
    if a_bytes.len() != b_bytes.len() {
        return false;
    }

    // For variable-length types, also check offsets match
    match (a, b) {
        (ArrayBytes::Variable(a_var), ArrayBytes::Variable(b_var))
            if a_var.offsets() != b_var.offsets() =>
        {
            return false;
        }
        (ArrayBytes::Optional(a_opt), ArrayBytes::Optional(b_opt))
            if a_opt.mask() != b_opt.mask() =>
        {
            return false;
        }
        _ => {}
    }

    // Compare float values semantically
    compare_float_bytes_semantically(a_bytes, b_bytes, data_type)
}

/// Compare float bytes semantically (NaN == NaN, preserves sign of zero)
fn compare_float_bytes_semantically(a: &[u8], b: &[u8], data_type: &DataType) -> bool {
    use std::any::TypeId;
    use zarrs::array::data_type;

    if a.len() != b.len() {
        return false;
    }

    let type_id = data_type.as_any().type_id();

    if type_id == TypeId::of::<data_type::Float32DataType>() {
        compare_f32_slices(a, b)
    } else if type_id == TypeId::of::<data_type::Float64DataType>() {
        compare_f64_slices(a, b)
    } else if type_id == TypeId::of::<data_type::Float16DataType>() {
        compare_f16_slices(a, b)
    } else if type_id == TypeId::of::<data_type::BFloat16DataType>() {
        compare_bf16_slices(a, b)
    } else if type_id == TypeId::of::<data_type::ComplexFloat32DataType>()
        || type_id == TypeId::of::<data_type::Complex64DataType>()
    {
        compare_complex_f32_slices(a, b)
    } else if type_id == TypeId::of::<data_type::ComplexFloat64DataType>()
        || type_id == TypeId::of::<data_type::Complex128DataType>()
    {
        compare_complex_f64_slices(a, b)
    } else if type_id == TypeId::of::<data_type::ComplexFloat16DataType>() {
        compare_complex_f16_slices(a, b)
    } else if type_id == TypeId::of::<data_type::ComplexBFloat16DataType>() {
        compare_complex_bf16_slices(a, b)
    } else if type_id == TypeId::of::<data_type::Float8E4M3DataType>() {
        compare_float8_e4m3_slices(a, b)
    } else if type_id == TypeId::of::<data_type::Float8E5M2DataType>() {
        compare_float8_e5m2_slices(a, b)
    } else if type_id == TypeId::of::<data_type::Float8E3M4DataType>() {
        compare_float8_e3m4_slices(a, b)
    } else if type_id == TypeId::of::<data_type::ComplexFloat8E4M3DataType>() {
        compare_complex_float8_e4m3_slices(a, b)
    } else if type_id == TypeId::of::<data_type::ComplexFloat8E5M2DataType>() {
        compare_complex_float8_e5m2_slices(a, b)
    } else if type_id == TypeId::of::<data_type::ComplexFloat8E3M4DataType>() {
        compare_complex_float8_e3m4_slices(a, b)
    } else {
        // For other exotic floats or non-float types, use byte comparison
        a == b
    }
}

fn compare_f32_slices(a: &[u8], b: &[u8]) -> bool {
    let a_floats: &[f32] = unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), a.len() / 4) };
    let b_floats: &[f32] = unsafe { std::slice::from_raw_parts(b.as_ptr().cast(), b.len() / 4) };
    a_floats.len() == b_floats.len()
        && a_floats
            .iter()
            .zip(b_floats)
            .all(|(a, b)| f32_equal(*a, *b))
}

fn compare_f64_slices(a: &[u8], b: &[u8]) -> bool {
    let a_floats: &[f64] = unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), a.len() / 8) };
    let b_floats: &[f64] = unsafe { std::slice::from_raw_parts(b.as_ptr().cast(), b.len() / 8) };
    a_floats.len() == b_floats.len()
        && a_floats
            .iter()
            .zip(b_floats)
            .all(|(a, b)| f64_equal(*a, *b))
}

fn compare_f16_slices(a: &[u8], b: &[u8]) -> bool {
    let a_floats: &[f16] = unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), a.len() / 2) };
    let b_floats: &[f16] = unsafe { std::slice::from_raw_parts(b.as_ptr().cast(), b.len() / 2) };
    a_floats.len() == b_floats.len()
        && a_floats.iter().zip(b_floats).all(|(a, b)| {
            // Both NaN -> equal
            if a.is_nan() && b.is_nan() {
                return true;
            }
            // Compare bits to preserve sign of zero
            a.to_bits() == b.to_bits()
        })
}

fn compare_bf16_slices(a: &[u8], b: &[u8]) -> bool {
    let a_floats: &[bf16] = unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), a.len() / 2) };
    let b_floats: &[bf16] = unsafe { std::slice::from_raw_parts(b.as_ptr().cast(), b.len() / 2) };
    a_floats.len() == b_floats.len()
        && a_floats.iter().zip(b_floats).all(|(a, b)| {
            // Both NaN -> equal
            if a.is_nan() && b.is_nan() {
                return true;
            }
            // Compare bits to preserve sign of zero
            a.to_bits() == b.to_bits()
        })
}

fn compare_complex_f32_slices(a: &[u8], b: &[u8]) -> bool {
    let a_floats: &[f32] = unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), a.len() / 4) };
    let b_floats: &[f32] = unsafe { std::slice::from_raw_parts(b.as_ptr().cast(), b.len() / 4) };
    a_floats.len() == b_floats.len()
        && a_floats
            .iter()
            .zip(b_floats)
            .all(|(a, b)| f32_equal(*a, *b))
}

fn compare_complex_f64_slices(a: &[u8], b: &[u8]) -> bool {
    let a_floats: &[f64] = unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), a.len() / 8) };
    let b_floats: &[f64] = unsafe { std::slice::from_raw_parts(b.as_ptr().cast(), b.len() / 8) };
    a_floats.len() == b_floats.len()
        && a_floats
            .iter()
            .zip(b_floats)
            .all(|(a, b)| f64_equal(*a, *b))
}

fn compare_complex_f16_slices(a: &[u8], b: &[u8]) -> bool {
    let a_floats: &[f16] = unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), a.len() / 2) };
    let b_floats: &[f16] = unsafe { std::slice::from_raw_parts(b.as_ptr().cast(), b.len() / 2) };
    a_floats.len() == b_floats.len()
        && a_floats.iter().zip(b_floats).all(|(a, b)| {
            if a.is_nan() && b.is_nan() {
                return true;
            }
            a.to_bits() == b.to_bits()
        })
}

fn compare_complex_bf16_slices(a: &[u8], b: &[u8]) -> bool {
    let a_floats: &[bf16] = unsafe { std::slice::from_raw_parts(a.as_ptr().cast(), a.len() / 2) };
    let b_floats: &[bf16] = unsafe { std::slice::from_raw_parts(b.as_ptr().cast(), b.len() / 2) };
    a_floats.len() == b_floats.len()
        && a_floats.iter().zip(b_floats).all(|(a, b)| {
            if a.is_nan() && b.is_nan() {
                return true;
            }
            a.to_bits() == b.to_bits()
        })
}

/// Compare two f32 floats with NaN-aware equality (NaN == NaN, but -0.0 != +0.0)
fn f32_equal(a: f32, b: f32) -> bool {
    // Both NaN -> equal
    if a.is_nan() && b.is_nan() {
        return true;
    }
    // For zeros, preserve sign by comparing bits
    if a == 0.0 && b == 0.0 {
        return a.to_bits() == b.to_bits();
    }
    // Normal comparison
    a == b
}

/// Compare two f64 floats with NaN-aware equality (NaN == NaN, but -0.0 != +0.0)
fn f64_equal(a: f64, b: f64) -> bool {
    // Both NaN -> equal
    if a.is_nan() && b.is_nan() {
        return true;
    }
    // For zeros, preserve sign by comparing bits
    if a == 0.0 && b == 0.0 {
        return a.to_bits() == b.to_bits();
    }
    // Normal comparison
    a == b
}

/// Check if a float8_e4m3 byte is NaN (exponent all 1s and mantissa non-zero)
fn is_float8_e4m3_nan(bits: u8) -> bool {
    let exponent = (bits >> 3) & 0b1111;
    let mantissa = bits & 0b111;
    exponent == 0b1111 && mantissa != 0
}

/// Check if a float8_e5m2 byte is NaN (exponent all 1s and mantissa non-zero)
fn is_float8_e5m2_nan(bits: u8) -> bool {
    let exponent = (bits >> 2) & 0b11111;
    let mantissa = bits & 0b11;
    exponent == 0b11111 && mantissa != 0
}

/// Check if a float8_e3m4 byte is NaN (exponent all 1s and mantissa non-zero)
fn is_float8_e3m4_nan(bits: u8) -> bool {
    let exponent = (bits >> 4) & 0b111;
    let mantissa = bits & 0b1111;
    exponent == 0b111 && mantissa != 0
}

fn compare_float8_e4m3_slices(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            // Both NaN -> equal
            if is_float8_e4m3_nan(*a) && is_float8_e4m3_nan(*b) {
                return true;
            }
            // Otherwise compare bits
            a == b
        })
}

fn compare_float8_e5m2_slices(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            // Both NaN -> equal
            if is_float8_e5m2_nan(*a) && is_float8_e5m2_nan(*b) {
                return true;
            }
            // Otherwise compare bits
            a == b
        })
}

fn compare_float8_e3m4_slices(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(a, b)| {
            // Both NaN -> equal
            if is_float8_e3m4_nan(*a) && is_float8_e3m4_nan(*b) {
                return true;
            }
            // Otherwise compare bits
            a == b
        })
}

fn compare_complex_float8_e4m3_slices(a: &[u8], b: &[u8]) -> bool {
    compare_float8_e4m3_slices(a, b)
}

fn compare_complex_float8_e5m2_slices(a: &[u8], b: &[u8]) -> bool {
    compare_float8_e5m2_slices(a, b)
}

fn compare_complex_float8_e3m4_slices(a: &[u8], b: &[u8]) -> bool {
    compare_float8_e3m4_slices(a, b)
}
