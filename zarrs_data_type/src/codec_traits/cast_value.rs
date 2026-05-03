//! The `cast_value` codec data type traits.

use half::{bf16, f16};
use thiserror::Error;

/// Rounding mode used by the `cast_value` codec.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CastValueRoundingMode {
    /// Round to nearest with ties to even.
    NearestEven,
    /// Round towards zero.
    TowardsZero,
    /// Round towards positive infinity.
    TowardsPositive,
    /// Round towards negative infinity.
    TowardsNegative,
    /// Round to nearest with ties away from zero.
    NearestAway,
}

/// Out-of-range handling used by the `cast_value` codec.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CastValueOutOfRangeMode {
    /// Clamp to the target range.
    Clamp,
    /// Wrap modulo the integer target range.
    Wrap,
}

/// A scalar value decoded by a `CastValueDataTypeTraits` implementation.
#[derive(Clone, Copy, Debug)]
pub enum CastValueScalar {
    /// A signed integer value.
    Signed(i128),
    /// An unsigned integer value.
    Unsigned(u128),
    /// A floating point value.
    Float(CastValueFloatScalar),
}

/// A floating point scalar value.
#[derive(Clone, Copy, Debug)]
pub enum CastValueFloatScalar {
    /// `float16`.
    F16(f16),
    /// `bfloat16`.
    BF16(bf16),
    /// `float32` or a smaller float represented through `f32`.
    F32(f32),
    /// `float64`.
    F64(f64),
}

impl CastValueFloatScalar {
    /// Return the value as `f64`.
    #[must_use]
    pub fn to_f64(self) -> f64 {
        match self {
            Self::F16(value) => value.to_f64(),
            Self::BF16(value) => value.to_f64(),
            Self::F32(value) => f64::from(value),
            Self::F64(value) => value,
        }
    }

    /// Return the value as `f32`.
    #[must_use]
    pub fn to_f32(self) -> f32 {
        match self {
            Self::F16(value) => value.to_f32(),
            Self::BF16(value) => value.to_f32(),
            Self::F32(value) => value,
            #[expect(clippy::cast_possible_truncation)]
            Self::F64(value) => value as f32,
        }
    }

    /// Returns `true` if the value is NaN.
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.to_f64().is_nan()
    }

    /// Returns `true` if the value is infinite.
    #[must_use]
    pub fn is_infinite(self) -> bool {
        self.to_f64().is_infinite()
    }

    /// Returns `true` if the value has a negative sign.
    #[must_use]
    pub fn is_sign_negative(self) -> bool {
        self.to_f64().is_sign_negative()
    }
}

/// Errors raised while casting scalar values.
#[derive(Clone, Debug, Error)]
pub enum CastValueError {
    /// Invalid source element bytes.
    #[error("invalid source element bytes for cast_value")]
    InvalidElementBytes,
    /// The source scalar is not representable in the target data type.
    #[error("scalar is not representable in target data type")]
    NotRepresentable,
    /// The requested rounding mode is unsupported.
    #[error("unsupported cast_value rounding mode {0:?}")]
    UnsupportedRoundingMode(CastValueRoundingMode),
    /// Wrapping was requested for a non-integral target data type.
    #[error("cast_value wrap mode is only valid for integral target data types")]
    InvalidWrapTarget,
}

/// Traits for a data type supporting the `cast_value` codec.
pub trait CastValueDataTypeTraits {
    /// The effective numeric bit width of the data type.
    fn cast_value_bit_width(&self) -> u32;

    /// Returns `true` if the data type is integral.
    fn cast_value_is_integral(&self) -> bool;

    /// Decode one element from native-endian codec bytes.
    ///
    /// # Errors
    /// Returns an error if `bytes` are incompatible with the data type.
    fn cast_value_read(&self, bytes: &[u8]) -> Result<CastValueScalar, CastValueError>;

    /// Encode one scalar to native-endian codec bytes.
    ///
    /// # Errors
    /// Returns an error if `value` cannot be represented with the requested modes.
    fn cast_value_write(
        &self,
        value: CastValueScalar,
        rounding: CastValueRoundingMode,
        out_of_range: Option<CastValueOutOfRangeMode>,
    ) -> Result<Vec<u8>, CastValueError>;

    /// Cast one element to another `cast_value`-compatible data type.
    ///
    /// Implementors may override this for specialised direct conversions.
    ///
    /// # Errors
    /// Returns an error if the source cannot be converted to the target.
    fn cast_value_cast(
        &self,
        source: &[u8],
        target: &dyn CastValueDataTypeTraits,
        rounding: CastValueRoundingMode,
        out_of_range: Option<CastValueOutOfRangeMode>,
    ) -> Result<Vec<u8>, CastValueError> {
        target.cast_value_write(self.cast_value_read(source)?, rounding, out_of_range)
    }
}

// Generate the codec support infrastructure using the generic macro.
crate::define_data_type_support!(CastValue);

const fn signed_min(bits: u32) -> i128 {
    -(1_i128 << (bits - 1))
}

const fn signed_max(bits: u32) -> i128 {
    (1_i128 << (bits - 1)) - 1
}

const fn unsigned_max(bits: u32) -> u128 {
    (1_u128 << bits) - 1
}

fn wrap_signed(value: i128, bits: u32) -> i128 {
    let modulus = 1_i128 << bits;
    let wrapped = value.rem_euclid(modulus);
    let sign_bit = 1_i128 << (bits - 1);
    if wrapped >= sign_bit {
        wrapped - modulus
    } else {
        wrapped
    }
}

fn wrap_unsigned(value: i128, bits: u32) -> u128 {
    let modulus = 1_i128 << bits;
    value.rem_euclid(modulus) as u128
}

fn wrap_signed_unsigned(value: u128, bits: u32) -> i128 {
    let modulus = 1_u128 << bits;
    let wrapped = value % modulus;
    let sign_bit = 1_u128 << (bits - 1);
    if wrapped >= sign_bit {
        (wrapped as i128) - (modulus as i128)
    } else {
        wrapped as i128
    }
}

fn wrap_unsigned_unsigned(value: u128, bits: u32) -> u128 {
    value % (1_u128 << bits)
}

fn round_float_nearest_even(value: f64) -> f64 {
    let truncated = value.trunc();
    let fraction = (value - truncated).abs();
    if fraction < 0.5 {
        truncated
    } else if fraction > 0.5 {
        truncated + value.signum()
    } else if (truncated / 2.0).fract() == 0.0 {
        truncated
    } else {
        truncated + value.signum()
    }
}

fn round_float_to_integer(value: f64, rounding: CastValueRoundingMode) -> f64 {
    match rounding {
        CastValueRoundingMode::NearestEven => round_float_nearest_even(value),
        CastValueRoundingMode::TowardsZero => value.trunc(),
        CastValueRoundingMode::TowardsPositive => value.ceil(),
        CastValueRoundingMode::TowardsNegative => value.floor(),
        CastValueRoundingMode::NearestAway => value.round(),
    }
}

fn scalar_to_i128(
    value: CastValueScalar,
    rounding: CastValueRoundingMode,
) -> Result<i128, CastValueError> {
    match value {
        CastValueScalar::Signed(value) => Ok(value),
        CastValueScalar::Unsigned(value) => {
            i128::try_from(value).map_err(|_| CastValueError::NotRepresentable)
        }
        CastValueScalar::Float(value) => {
            if value.is_nan() || value.is_infinite() {
                return Err(CastValueError::NotRepresentable);
            }
            let value = round_float_to_integer(value.to_f64(), rounding);
            if value < i128::MIN as f64 || value > i128::MAX as f64 {
                return Err(CastValueError::NotRepresentable);
            }
            Ok(value as i128)
        }
    }
}

fn scalar_to_f64(value: CastValueScalar) -> f64 {
    match value {
        CastValueScalar::Signed(value) => value as f64,
        CastValueScalar::Unsigned(value) => value as f64,
        CastValueScalar::Float(value) => value.to_f64(),
    }
}

fn next_up_f16(value: f16) -> f16 {
    if value.is_nan() || value == f16::INFINITY {
        return value;
    }
    if value == f16::NEG_INFINITY {
        return f16::MIN;
    }
    let bits = value.to_bits();
    if bits == 0x8000 {
        f16::from_bits(0x0001)
    } else if value.is_sign_negative() {
        f16::from_bits(bits - 1)
    } else {
        f16::from_bits(bits + 1)
    }
}

fn next_down_f16(value: f16) -> f16 {
    if value.is_nan() || value == f16::NEG_INFINITY {
        return value;
    }
    if value == f16::INFINITY {
        return f16::MAX;
    }
    let bits = value.to_bits();
    if bits == 0x0000 {
        f16::from_bits(0x8001)
    } else if value.is_sign_negative() {
        f16::from_bits(bits + 1)
    } else {
        f16::from_bits(bits - 1)
    }
}

fn next_up_bf16(value: bf16) -> bf16 {
    if value.is_nan() || value == bf16::INFINITY {
        return value;
    }
    if value == bf16::NEG_INFINITY {
        return bf16::MIN;
    }
    let bits = value.to_bits();
    if bits == 0x8000 {
        bf16::from_bits(0x0001)
    } else if value.is_sign_negative() {
        bf16::from_bits(bits - 1)
    } else {
        bf16::from_bits(bits + 1)
    }
}

fn next_down_bf16(value: bf16) -> bf16 {
    if value.is_nan() || value == bf16::NEG_INFINITY {
        return value;
    }
    if value == bf16::INFINITY {
        return bf16::MAX;
    }
    let bits = value.to_bits();
    if bits == 0x0000 {
        bf16::from_bits(0x8001)
    } else if value.is_sign_negative() {
        bf16::from_bits(bits + 1)
    } else {
        bf16::from_bits(bits - 1)
    }
}

fn adjust_float_rounding<T>(
    value: f64,
    rounded: T,
    rounding: CastValueRoundingMode,
    to_f64: impl Fn(T) -> f64,
    next_down: impl Fn(T) -> T,
    next_up: impl Fn(T) -> T,
) -> T
where
    T: Copy,
{
    if !value.is_finite() {
        return rounded;
    }
    let rounded_value = to_f64(rounded);
    if rounded_value == value {
        return rounded;
    }
    match rounding {
        CastValueRoundingMode::NearestEven => rounded,
        CastValueRoundingMode::TowardsPositive => {
            if rounded_value < value {
                next_up(rounded)
            } else {
                rounded
            }
        }
        CastValueRoundingMode::TowardsNegative => {
            if rounded_value > value {
                next_down(rounded)
            } else {
                rounded
            }
        }
        CastValueRoundingMode::TowardsZero => {
            if value.is_sign_negative() {
                if rounded_value < value {
                    next_up(rounded)
                } else {
                    rounded
                }
            } else if rounded_value > value {
                next_down(rounded)
            } else {
                rounded
            }
        }
        CastValueRoundingMode::NearestAway => {
            let lower = if rounded_value < value {
                rounded
            } else {
                next_down(rounded)
            };
            let upper = if rounded_value > value {
                rounded
            } else {
                next_up(rounded)
            };
            let lower_value = to_f64(lower);
            let upper_value = to_f64(upper);
            if lower_value.is_finite()
                && upper_value.is_finite()
                && (value - lower_value).abs() == (upper_value - value).abs()
            {
                if value.is_sign_negative() {
                    lower
                } else {
                    upper
                }
            } else {
                rounded
            }
        }
    }
}

/// Round a scalar value to a custom floating point target.
#[doc(hidden)]
#[must_use]
pub fn round_float_with_next_after<T>(
    value: f64,
    rounded: T,
    rounding: CastValueRoundingMode,
    to_f64: impl Fn(T) -> f64,
    next_down: impl Fn(T) -> T,
    next_up: impl Fn(T) -> T,
) -> T
where
    T: Copy,
{
    adjust_float_rounding(value, rounded, rounding, to_f64, next_down, next_up)
}

/// Round a scalar value to `f16`.
#[doc(hidden)]
#[must_use]
pub fn round_f64_to_f16(value: f64, rounding: CastValueRoundingMode) -> f16 {
    adjust_float_rounding(
        value,
        f16::from_f64(value),
        rounding,
        f16::to_f64,
        next_down_f16,
        next_up_f16,
    )
}

/// Round a scalar value to `bf16`.
#[doc(hidden)]
#[must_use]
pub fn round_f64_to_bf16(value: f64, rounding: CastValueRoundingMode) -> bf16 {
    adjust_float_rounding(
        value,
        bf16::from_f64(value),
        rounding,
        bf16::to_f64,
        next_down_bf16,
        next_up_bf16,
    )
}

/// Round a scalar value to `f32`.
#[doc(hidden)]
#[expect(clippy::cast_possible_truncation)]
#[must_use]
pub fn round_f64_to_f32(value: f64, rounding: CastValueRoundingMode) -> f32 {
    adjust_float_rounding(
        value,
        value as f32,
        rounding,
        f64::from,
        f32::next_down,
        f32::next_up,
    )
}

/// Round a scalar value to `f64`.
#[doc(hidden)]
#[must_use]
pub fn round_f64_to_f64(value: f64, rounding: CastValueRoundingMode) -> f64 {
    adjust_float_rounding(
        value,
        value,
        rounding,
        |value| value,
        f64::next_down,
        f64::next_up,
    )
}

/// Cast a scalar to a signed integer.
///
/// # Errors
/// Returns an error if the scalar is not representable.
pub fn cast_scalar_to_signed_integer(
    value: CastValueScalar,
    bits: u32,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<i128, CastValueError> {
    let min = signed_min(bits);
    let max = signed_max(bits);
    if let CastValueScalar::Unsigned(value) = value {
        if value <= max as u128 {
            return Ok(value as i128);
        }
        return match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => Ok(max),
            Some(CastValueOutOfRangeMode::Wrap) => Ok(wrap_signed_unsigned(value, bits)),
            None => Err(CastValueError::NotRepresentable),
        };
    }
    let value = scalar_to_i128(value, rounding)?;
    if value < min || value > max {
        match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => Ok(value.clamp(min, max)),
            Some(CastValueOutOfRangeMode::Wrap) => Ok(wrap_signed(value, bits)),
            None => Err(CastValueError::NotRepresentable),
        }
    } else {
        Ok(value)
    }
}

/// Cast a scalar to an unsigned integer.
///
/// # Errors
/// Returns an error if the scalar is not representable.
pub fn cast_scalar_to_unsigned_integer(
    value: CastValueScalar,
    bits: u32,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<u128, CastValueError> {
    let max = unsigned_max(bits);
    if let CastValueScalar::Unsigned(value) = value {
        if value <= max {
            return Ok(value);
        }
        return match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => Ok(max),
            Some(CastValueOutOfRangeMode::Wrap) => Ok(wrap_unsigned_unsigned(value, bits)),
            None => Err(CastValueError::NotRepresentable),
        };
    }
    let value = scalar_to_i128(value, rounding)?;
    if value < 0 || u128::try_from(value).map_or(true, |value| value > max) {
        match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => {
                Ok(u128::try_from(value).map_or(0, |value| value.min(max)))
            }
            Some(CastValueOutOfRangeMode::Wrap) => Ok(wrap_unsigned(value, bits)),
            None => Err(CastValueError::NotRepresentable),
        }
    } else {
        Ok(value as u128)
    }
}

/// Cast a scalar to an `f64` quantity with target-range handling.
///
/// # Errors
/// Returns an error if the scalar is not representable.
pub fn cast_scalar_to_float_quantity(
    value: CastValueScalar,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
    min: f64,
    max: f64,
    has_nan: bool,
    has_infinity: bool,
) -> Result<f64, CastValueError> {
    let value = scalar_to_f64(value);
    if value.is_nan() {
        return has_nan
            .then_some(f64::NAN)
            .ok_or(CastValueError::NotRepresentable);
    }
    if value.is_infinite() {
        if has_infinity {
            return Ok(value);
        }
        return match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) if value.is_sign_negative() => Ok(min),
            Some(CastValueOutOfRangeMode::Clamp) => Ok(max),
            Some(CastValueOutOfRangeMode::Wrap) => Err(CastValueError::InvalidWrapTarget),
            None => Err(CastValueError::NotRepresentable),
        };
    }
    if value < min {
        match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => {
                Ok(if has_infinity { f64::NEG_INFINITY } else { min })
            }
            // The cast_value spec is currently underspecified for inverse decoding
            // when the configured encoded data type is integral with
            // out_of_range=wrap, but the decoded array data type is floating
            // point. Match zarr-python 3.2.1 by applying overflow rounding
            // to either the finite bound or infinity when the floating target
            // supports infinities.
            Some(CastValueOutOfRangeMode::Wrap) if has_infinity => match rounding {
                CastValueRoundingMode::TowardsZero | CastValueRoundingMode::TowardsPositive => {
                    Ok(min)
                }
                CastValueRoundingMode::NearestEven
                | CastValueRoundingMode::TowardsNegative
                | CastValueRoundingMode::NearestAway => Ok(f64::NEG_INFINITY),
            },
            Some(CastValueOutOfRangeMode::Wrap) => Err(CastValueError::InvalidWrapTarget),
            None => Err(CastValueError::NotRepresentable),
        }
    } else if value > max {
        match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => {
                Ok(if has_infinity { f64::INFINITY } else { max })
            }
            Some(CastValueOutOfRangeMode::Wrap) if has_infinity => match rounding {
                CastValueRoundingMode::TowardsZero | CastValueRoundingMode::TowardsNegative => {
                    Ok(max)
                }
                CastValueRoundingMode::NearestEven
                | CastValueRoundingMode::TowardsPositive
                | CastValueRoundingMode::NearestAway => Ok(f64::INFINITY),
            },
            Some(CastValueOutOfRangeMode::Wrap) => Err(CastValueError::InvalidWrapTarget),
            None => Err(CastValueError::NotRepresentable),
        }
    } else {
        Ok(value)
    }
}

/// Macro to implement `cast_value` for signed integer data types.
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_cast_value_data_type_traits_signed_integer {
    ($marker:ty, $stored:ty, $bits:expr) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_bit_width(&self) -> u32 {
                $bits
            }

            fn cast_value_is_integral(&self) -> bool {
                true
            }

            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; core::mem::size_of::<$stored>()] =
                    bytes.try_into().map_err(|_| {
                        $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                    })?;
                let value = <$stored>::from_ne_bytes(bytes) as i128;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Signed(
                    value,
                ))
            }

            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
            ) -> Result<Vec<u8>, $crate::codec_traits::cast_value::CastValueError> {
                let value = $crate::codec_traits::cast_value::cast_scalar_to_signed_integer(
                    value,
                    $bits,
                    rounding,
                    out_of_range,
                )?;
                let value = <$stored>::try_from(value).map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::NotRepresentable
                })?;
                Ok(value.to_ne_bytes().to_vec())
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
}

/// Macro to implement `cast_value` for unsigned integer data types.
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_cast_value_data_type_traits_unsigned_integer {
    ($marker:ty, $stored:ty, $bits:expr) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_bit_width(&self) -> u32 {
                $bits
            }

            fn cast_value_is_integral(&self) -> bool {
                true
            }

            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; core::mem::size_of::<$stored>()] =
                    bytes.try_into().map_err(|_| {
                        $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                    })?;
                let value = <$stored>::from_ne_bytes(bytes) as u128;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Unsigned(
                    value,
                ))
            }

            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
            ) -> Result<Vec<u8>, $crate::codec_traits::cast_value::CastValueError> {
                let value = $crate::codec_traits::cast_value::cast_scalar_to_unsigned_integer(
                    value,
                    $bits,
                    rounding,
                    out_of_range,
                )?;
                let value = <$stored>::try_from(value).map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::NotRepresentable
                })?;
                Ok(value.to_ne_bytes().to_vec())
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
}

/// Macro to implement `cast_value` for floating point data types.
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_cast_value_data_type_traits_float {
    ($marker:ty, f16) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_bit_width(&self) -> u32 {
                16
            }
            fn cast_value_is_integral(&self) -> bool {
                false
            }
            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; 2] = bytes.try_into().map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                })?;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Float(
                    $crate::codec_traits::cast_value::CastValueFloatScalar::F16(
                        half::f16::from_ne_bytes(bytes),
                    ),
                ))
            }
            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
            ) -> Result<Vec<u8>, $crate::codec_traits::cast_value::CastValueError> {
                let value = $crate::codec_traits::cast_value::cast_scalar_to_float_quantity(
                    value,
                    rounding,
                    out_of_range,
                    half::f16::MIN.to_f64(),
                    half::f16::MAX.to_f64(),
                    true,
                    true,
                )?;
                Ok(
                    $crate::codec_traits::cast_value::round_f64_to_f16(value, rounding)
                        .to_ne_bytes()
                        .to_vec(),
                )
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
    ($marker:ty, bf16) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_bit_width(&self) -> u32 {
                16
            }
            fn cast_value_is_integral(&self) -> bool {
                false
            }
            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; 2] = bytes.try_into().map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                })?;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Float(
                    $crate::codec_traits::cast_value::CastValueFloatScalar::BF16(
                        half::bf16::from_ne_bytes(bytes),
                    ),
                ))
            }
            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
            ) -> Result<Vec<u8>, $crate::codec_traits::cast_value::CastValueError> {
                let value = $crate::codec_traits::cast_value::cast_scalar_to_float_quantity(
                    value,
                    rounding,
                    out_of_range,
                    half::bf16::MIN.to_f64(),
                    half::bf16::MAX.to_f64(),
                    true,
                    true,
                )?;
                Ok(
                    $crate::codec_traits::cast_value::round_f64_to_bf16(value, rounding)
                        .to_ne_bytes()
                        .to_vec(),
                )
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
    ($marker:ty, f32) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_bit_width(&self) -> u32 {
                32
            }
            fn cast_value_is_integral(&self) -> bool {
                false
            }
            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; 4] = bytes.try_into().map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                })?;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Float(
                    $crate::codec_traits::cast_value::CastValueFloatScalar::F32(
                        f32::from_ne_bytes(bytes),
                    ),
                ))
            }
            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
            ) -> Result<Vec<u8>, $crate::codec_traits::cast_value::CastValueError> {
                let value = $crate::codec_traits::cast_value::cast_scalar_to_float_quantity(
                    value,
                    rounding,
                    out_of_range,
                    f64::from(f32::MIN),
                    f64::from(f32::MAX),
                    true,
                    true,
                )?;
                Ok(
                    $crate::codec_traits::cast_value::round_f64_to_f32(value, rounding)
                        .to_ne_bytes()
                        .to_vec(),
                )
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
    ($marker:ty, f64) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_bit_width(&self) -> u32 {
                64
            }
            fn cast_value_is_integral(&self) -> bool {
                false
            }
            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; 8] = bytes.try_into().map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                })?;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Float(
                    $crate::codec_traits::cast_value::CastValueFloatScalar::F64(
                        f64::from_ne_bytes(bytes),
                    ),
                ))
            }
            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
            ) -> Result<Vec<u8>, $crate::codec_traits::cast_value::CastValueError> {
                let value = $crate::codec_traits::cast_value::cast_scalar_to_float_quantity(
                    value,
                    rounding,
                    out_of_range,
                    f64::MIN,
                    f64::MAX,
                    true,
                    true,
                )?;
                Ok(
                    $crate::codec_traits::cast_value::round_f64_to_f64(value, rounding)
                        .to_ne_bytes()
                        .to_vec(),
                )
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
    ($marker:ty, microfloat, $float_type:ty, $bits:expr) => {
        impl $crate::codec_traits::cast_value::CastValueDataTypeTraits for $marker {
            fn cast_value_bit_width(&self) -> u32 {
                $bits
            }
            fn cast_value_is_integral(&self) -> bool {
                false
            }
            fn cast_value_read(
                &self,
                bytes: &[u8],
            ) -> Result<
                $crate::codec_traits::cast_value::CastValueScalar,
                $crate::codec_traits::cast_value::CastValueError,
            > {
                let bytes: [u8; 1] = bytes.try_into().map_err(|_| {
                    $crate::codec_traits::cast_value::CastValueError::InvalidElementBytes
                })?;
                Ok($crate::codec_traits::cast_value::CastValueScalar::Float(
                    $crate::codec_traits::cast_value::CastValueFloatScalar::F32(
                        <$float_type>::from_bits(bytes[0]).to_f32(),
                    ),
                ))
            }
            fn cast_value_write(
                &self,
                value: $crate::codec_traits::cast_value::CastValueScalar,
                rounding: $crate::codec_traits::cast_value::CastValueRoundingMode,
                out_of_range: Option<$crate::codec_traits::cast_value::CastValueOutOfRangeMode>,
            ) -> Result<Vec<u8>, $crate::codec_traits::cast_value::CastValueError> {
                let has_nan = <$float_type>::NAN.is_nan();
                let has_infinity = <$float_type>::INFINITY.to_f32().is_infinite();
                let value = $crate::codec_traits::cast_value::cast_scalar_to_float_quantity(
                    value,
                    rounding,
                    out_of_range,
                    f64::from(<$float_type>::MIN.to_f32()),
                    f64::from(<$float_type>::MAX.to_f32()),
                    has_nan,
                    has_infinity,
                )?;
                let quantity = value;
                let value = <$float_type>::from_f64(quantity);
                let value = $crate::codec_traits::cast_value::round_float_with_next_after(
                    quantity,
                    value,
                    rounding,
                    |value| f64::from(value.to_f32()),
                    <$float_type>::next_down,
                    <$float_type>::next_up,
                );
                Ok(vec![value.to_bits()])
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::cast_value::CastValueDataTypePlugin,
            $crate::codec_traits::cast_value::CastValueDataTypeTraits
        );
    };
}

pub use crate::{
    _impl_cast_value_data_type_traits_float as impl_cast_value_data_type_traits_float,
    _impl_cast_value_data_type_traits_signed_integer as impl_cast_value_data_type_traits_signed_integer,
    _impl_cast_value_data_type_traits_unsigned_integer as impl_cast_value_data_type_traits_unsigned_integer,
};
