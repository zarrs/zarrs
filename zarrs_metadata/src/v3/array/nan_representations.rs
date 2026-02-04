//! Zarr V3 NaN representations.
//!
//! Zarr uses the not-a-number (NaN) value where the sign bit is 0 (positive), the most significant bit (MSB) of the mantissa is 1, and all other bits of the mantissa are zero.

use std::mem::transmute;

use half::{bf16, f16};

// https://github.com/rust-lang/rust/issues/72447

/// The Zarr V3 "NaN" fill value for a 64-bit IEEE 754 floating point number.
#[allow(clippy::unusual_byte_groupings)]
#[allow(unknown_lints)]
#[allow(unnecessary_transmutes)] // lint is wrong until MSVC is 1.83
pub const ZARR_NAN_F64: f64 = unsafe {
    transmute::<u64, f64>(0b0_11111111111_1000000000000000000000000000000000000000000000000000)
};
// const ZARR_NAN_F64: f64 = f64::from_bits(0b0_11111111111_1000000000000000000000000000000000000000000000000000);

/// The Zarr V3 "NaN" fill value for a 32-bit IEEE 754 floating point number.
#[allow(clippy::unusual_byte_groupings)]
#[allow(unknown_lints)]
#[allow(unnecessary_transmutes)] // lint is wrong until MSVC is 1.83
pub const ZARR_NAN_F32: f32 =
    unsafe { transmute::<u32, f32>(0b0_11111111_10000000000000000000000) };
// const ZARR_NAN_F32: f32 = f32::from_bits(0b0_11111111_10000000000000000000000);

/// The Zarr V3 "NaN" fill value for a 16-bit IEEE 754 floating point number.
pub const ZARR_NAN_F16: f16 = f16::NAN;

/// The Zarr V3 "NaN" fill value for a 16-bit brain floating point number.
pub const ZARR_NAN_BF16: bf16 = bf16::NAN;


#[allow(clippy::unusual_byte_groupings)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_representations() {
        assert_eq!(
            bf16::NAN.to_ne_bytes(),
            bf16::from_bits(0b0_11111111_1000000).to_ne_bytes()
        );
        assert_eq!(
            f16::NAN.to_ne_bytes(),
            f16::from_bits(0b0_11111_1000000000).to_ne_bytes()
        );
    }
}
