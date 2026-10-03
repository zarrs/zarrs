//! Bind-time selected bulk cast kernels for the `cast_value` codec.
//!
//! Kernels are monomorphised per (source, target) primitive pair so the
//! per-element arithmetic runs at the native width of the pair (at most
//! 64-bit). 128-bit arithmetic is confined to rare big-magnitude branches;
//! the generic [`CastValueScalar`](super::CastValueScalar) path remains as the
//! fallback for data types without a [`CastValueRepr`].

use half::{bf16, f16};

use super::{
    CastValueError, CastValueOutOfRangeMode, CastValueRoundingMode, float_quantity_from_f64,
    round_f64_to_bf16, round_f64_to_f16, round_f64_to_f32, round_f64_to_f64,
    round_float_to_integer, wrap_float,
};

/// The stored integer primitive of a data type supporting cast kernels.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastValueIntStored {
    /// Stored as `i8`.
    I8,
    /// Stored as `i16`.
    I16,
    /// Stored as `i32`.
    I32,
    /// Stored as `i64`.
    I64,
    /// Stored as `u8`.
    U8,
    /// Stored as `u16`.
    U16,
    /// Stored as `u32`.
    U32,
    /// Stored as `u64`.
    U64,
}

impl CastValueIntStored {
    const fn bits(self) -> u32 {
        match self {
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 => 16,
            Self::I32 | Self::U32 => 32,
            Self::I64 | Self::U64 => 64,
        }
    }
}

/// Maps an integer primitive to its [`CastValueIntStored`].
#[doc(hidden)]
pub trait CastValueIntStoredPrimitive {
    /// The corresponding [`CastValueIntStored`].
    const STORED: CastValueIntStored;
}

macro_rules! impl_int_stored_primitive {
    ($($t:ty => $variant:ident),* $(,)?) => {
        $(impl CastValueIntStoredPrimitive for $t {
            const STORED: CastValueIntStored = CastValueIntStored::$variant;
        })*
    };
}
impl_int_stored_primitive!(
    i8 => I8, i16 => I16, i32 => I32, i64 => I64,
    u8 => U8, u16 => U16, u32 => U32, u64 => U64,
);

/// The numeric representation of a data type for `cast_value` kernel selection.
///
/// Data types exposing a representation via
/// [`CastValueDataTypeTraits::cast_value_repr`](super::CastValueDataTypeTraits::cast_value_repr)
/// opt in to monomorphised bulk cast kernels selected by [`select_cast_kernel`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastValueRepr {
    /// A two's complement integer.
    Int {
        /// The number of logical bits (may be less than the stored size, e.g. 2 for `int2`).
        bits: u32,
        /// The stored primitive.
        stored: CastValueIntStored,
    },
    /// IEEE 754 binary16.
    F16,
    /// bfloat16.
    BF16,
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary64.
    F64,
}

#[derive(Clone, Copy, Debug)]
struct CastKernelParams {
    target_bits: u32,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
}

type CastKernelFn = fn(&[u8], &mut Vec<u8>, CastKernelParams) -> Result<(), CastValueError>;

/// A monomorphised bulk cast selected for a (source, target, configuration) triple.
///
/// Observationally identical to the generic
/// [`CastValueDataTypeTraits::cast_value_cast_slice`](super::CastValueDataTypeTraits::cast_value_cast_slice)
/// path, but with per-element arithmetic at the native width of the pair.
#[derive(Clone, Copy, Debug)]
pub struct CastValueKernel {
    kernel: CastKernelFn,
    params: CastKernelParams,
}

impl CastValueKernel {
    /// Cast contiguous source elements, appending the encoded bytes to `output`.
    ///
    /// # Errors
    /// Returns an error if `source` is not element aligned or an element cannot be cast.
    pub fn cast(&self, source: &[u8], output: &mut Vec<u8>) -> Result<(), CastValueError> {
        (self.kernel)(source, output, self.params)
    }
}

/// A source value widened to 64-bit arithmetic.
///
/// `U` is only produced by `u64` sources; everything else fits `i64`. The
/// variant is compile-time constant per source type, so the `match` in the
/// write path is eliminated after monomorphisation.
#[derive(Clone, Copy)]
enum IntWide {
    I(i64),
    U(u64),
}

impl IntWide {
    fn magnitude_and_sign(self) -> (u64, bool) {
        match self {
            Self::I(value) => (value.unsigned_abs(), value < 0),
            Self::U(value) => (value, false),
        }
    }
}

/// Integer primitives usable as kernel sources and targets.
trait KernelInt: Copy {
    const SIGNED: bool;
    const SIZE: usize;
    fn read(bytes: &[u8]) -> Self;
    fn widen(self) -> IntWide;
    fn write_i64(value: i64, output: &mut Vec<u8>);
    fn write_u64(value: u64, output: &mut Vec<u8>);
}

macro_rules! impl_kernel_int {
    ($t:ty, $signed:literal, |$v:ident| $widen:expr) => {
        impl KernelInt for $t {
            const SIGNED: bool = $signed;
            const SIZE: usize = size_of::<$t>();
            #[inline]
            fn read(bytes: &[u8]) -> Self {
                <$t>::from_ne_bytes(bytes.try_into().expect("element aligned"))
            }
            #[inline]
            fn widen(self) -> IntWide {
                let $v = self;
                $widen
            }
            // `value` is within the logical range or an exact wrapped bit
            // pattern; `as` truncation is exact. `allow` rather than `expect`:
            // which cast lints fire varies per instantiation.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            #[inline]
            fn write_i64(value: i64, output: &mut Vec<u8>) {
                output.extend_from_slice(&(value as $t).to_ne_bytes());
            }
            #[allow(
                clippy::cast_possible_truncation,
                clippy::cast_possible_wrap,
                clippy::cast_sign_loss
            )]
            #[inline]
            fn write_u64(value: u64, output: &mut Vec<u8>) {
                output.extend_from_slice(&(value as $t).to_ne_bytes());
            }
        }
    };
}
impl_kernel_int!(i8, true, |v| IntWide::I(i64::from(v)));
impl_kernel_int!(i16, true, |v| IntWide::I(i64::from(v)));
impl_kernel_int!(i32, true, |v| IntWide::I(i64::from(v)));
impl_kernel_int!(i64, true, |v| IntWide::I(v));
impl_kernel_int!(u8, false, |v| IntWide::I(i64::from(v)));
impl_kernel_int!(u16, false, |v| IntWide::I(i64::from(v)));
impl_kernel_int!(u32, false, |v| IntWide::I(i64::from(v)));
impl_kernel_int!(u64, false, |v| IntWide::U(v));

/// Float sources readable as an exact `f64`.
trait KernelToF64: Copy {
    const SIZE: usize;
    fn read_to_f64(bytes: &[u8], rounding: CastValueRoundingMode) -> f64;
}

/// Float sources readable as an exact `f64`.
trait KernelFloatSource: Copy {
    const SIZE: usize;
    fn read_f64(bytes: &[u8]) -> f64;
}

macro_rules! impl_kernel_float_source {
    ($t:ty, $bytes:literal, |$v:ident| $to_f64:expr) => {
        impl KernelFloatSource for $t {
            const SIZE: usize = $bytes;
            #[inline]
            fn read_f64(bytes: &[u8]) -> f64 {
                let $v = <$t>::from_ne_bytes(bytes.try_into().expect("element aligned"));
                $to_f64
            }
        }
        impl KernelToF64 for $t {
            const SIZE: usize = $bytes;
            #[inline]
            fn read_to_f64(bytes: &[u8], _rounding: CastValueRoundingMode) -> f64 {
                <$t as KernelFloatSource>::read_f64(bytes)
            }
        }
    };
}
impl_kernel_float_source!(f16, 2, |v| v.to_f64());
impl_kernel_float_source!(bf16, 2, |v| v.to_f64());
impl_kernel_float_source!(f32, 4, |v| f64::from(v));
impl_kernel_float_source!(f64, 8, |v| v);

/// Float targets writable from an `f64` quantity.
trait KernelFloatTarget {
    const SIZE: usize;
    const MANTISSA_DIGITS: u32;
    fn min_f64() -> f64;
    fn max_f64() -> f64;
    fn round_write(value: f64, rounding: CastValueRoundingMode, output: &mut Vec<u8>);
}

macro_rules! impl_kernel_float_target {
    ($t:ty, $bytes:literal, $mantissa_digits:expr, $min:expr, $max:expr, $round:ident) => {
        impl KernelFloatTarget for $t {
            const SIZE: usize = $bytes;
            const MANTISSA_DIGITS: u32 = $mantissa_digits;
            #[inline]
            fn min_f64() -> f64 {
                $min
            }
            #[inline]
            fn max_f64() -> f64 {
                $max
            }
            #[inline]
            fn round_write(value: f64, rounding: CastValueRoundingMode, output: &mut Vec<u8>) {
                output.extend_from_slice(&$round(value, rounding).to_ne_bytes());
            }
        }
    };
}
impl_kernel_float_target!(
    f16,
    2,
    f16::MANTISSA_DIGITS,
    f16::MIN.to_f64(),
    f16::MAX.to_f64(),
    round_f64_to_f16
);
impl_kernel_float_target!(
    bf16,
    2,
    bf16::MANTISSA_DIGITS,
    bf16::MIN.to_f64(),
    bf16::MAX.to_f64(),
    round_f64_to_bf16
);
impl_kernel_float_target!(
    f32,
    4,
    f32::MANTISSA_DIGITS,
    f64::from(f32::MIN),
    f64::from(f32::MAX),
    round_f64_to_f32
);
impl_kernel_float_target!(
    f64,
    8,
    f64::MANTISSA_DIGITS,
    f64::MIN,
    f64::MAX,
    round_f64_to_f64
);

/// The logical bounds of an integer target, hoisted out of the element loop.
///
/// `min_i`/`max_i` are the bounds clamped into the `i64` domain (only the
/// `uint64` maximum is clipped, in which case the exceeded branch is
/// unreachable from `i64` values); `max_u` is the exact maximum in `u64`.
#[derive(Clone, Copy)]
struct IntBounds {
    min_i: i64,
    max_i: i64,
    max_u: u64,
}

fn int_bounds(bits: u32, signed: bool) -> IntBounds {
    debug_assert!((1..=64).contains(&bits));
    if signed {
        #[expect(clippy::cast_possible_wrap)]
        let max_i = ((1_u64 << (bits - 1)) - 1) as i64;
        #[expect(clippy::cast_sign_loss)]
        let max_u = max_i as u64; // non-negative
        IntBounds {
            min_i: i64::MIN >> (64 - bits),
            max_i,
            max_u,
        }
    } else {
        let max_u = if bits == 64 {
            u64::MAX
        } else {
            (1_u64 << bits) - 1
        };
        #[expect(clippy::cast_possible_wrap)]
        let max_i = max_u.min(i64::MAX as u64) as i64;
        IntBounds {
            min_i: 0,
            max_i,
            max_u,
        }
    }
}

/// Write a wrapped (modulo `2^bits`) value from its two's complement bit pattern.
#[inline]
fn write_wrapped<T: KernelInt>(pattern: u64, bits: u32, output: &mut Vec<u8>) {
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1_u64 << bits) - 1
    };
    let wrapped = pattern & mask;
    if T::SIGNED {
        #[expect(clippy::cast_possible_wrap)]
        let folded = if bits < 64 && wrapped >= (1_u64 << (bits - 1)) {
            // `wrapped - 2^bits` computed with wrapping `u64` arithmetic; the
            // bit pattern reinterprets exactly as the negative `i64` value
            wrapped.wrapping_sub(1_u64 << bits) as i64
        } else {
            wrapped as i64
        };
        T::write_i64(folded, output);
    } else {
        T::write_u64(wrapped, output);
    }
}

/// Write one 64-bit-domain value to an integer target with bounds handling.
#[inline]
fn write_int<T: KernelInt>(
    value: IntWide,
    bits: u32,
    bounds: IntBounds,
    out_of_range: Option<CastValueOutOfRangeMode>,
    output: &mut Vec<u8>,
) -> Result<(), CastValueError> {
    match value {
        IntWide::I(value) => {
            if value >= bounds.min_i && value <= bounds.max_i {
                if T::SIGNED {
                    T::write_i64(value, output);
                } else {
                    #[expect(clippy::cast_sign_loss)]
                    T::write_u64(value as u64, output); // non-negative (`min_i` is 0)
                }
                return Ok(());
            }
            match out_of_range {
                Some(CastValueOutOfRangeMode::Clamp) => {
                    let value = value.clamp(bounds.min_i, bounds.max_i);
                    if T::SIGNED {
                        T::write_i64(value, output);
                    } else {
                        #[expect(clippy::cast_sign_loss)]
                        T::write_u64(value as u64, output);
                    }
                    Ok(())
                }
                Some(CastValueOutOfRangeMode::Wrap) => {
                    #[expect(clippy::cast_sign_loss)]
                    write_wrapped::<T>(value as u64, bits, output);
                    Ok(())
                }
                None => Err(CastValueError::NotRepresentable),
            }
        }
        IntWide::U(value) => {
            if value <= bounds.max_u {
                if T::SIGNED {
                    #[expect(clippy::cast_possible_wrap)]
                    T::write_i64(value as i64, output); // `value <= max_i` here
                } else {
                    T::write_u64(value, output);
                }
                return Ok(());
            }
            match out_of_range {
                Some(CastValueOutOfRangeMode::Clamp) => {
                    if T::SIGNED {
                        T::write_i64(bounds.max_i, output);
                    } else {
                        T::write_u64(bounds.max_u, output);
                    }
                    Ok(())
                }
                Some(CastValueOutOfRangeMode::Wrap) => {
                    write_wrapped::<T>(value, bits, output);
                    Ok(())
                }
                None => Err(CastValueError::NotRepresentable),
            }
        }
    }
}

fn check_alignment(source: &[u8], element_size: usize) -> Result<usize, CastValueError> {
    if source.len().is_multiple_of(element_size) {
        Ok(source.len() / element_size)
    } else {
        Err(CastValueError::InvalidElementBytes)
    }
}

fn kernel_int_to_int<S: KernelInt, T: KernelInt>(
    source: &[u8],
    output: &mut Vec<u8>,
    params: CastKernelParams,
) -> Result<(), CastValueError> {
    let num_elements = check_alignment(source, S::SIZE)?;
    output.reserve(num_elements * T::SIZE);
    let bounds = int_bounds(params.target_bits, T::SIGNED);
    for element in source.chunks_exact(S::SIZE) {
        write_int::<T>(
            S::read(element).widen(),
            params.target_bits,
            bounds,
            params.out_of_range,
            output,
        )?;
    }
    Ok(())
}

/// Round an integer directly to a binary float precision, returning the exact
/// target value represented as `f64`.
///
/// The rounded magnitude has at most `mantissa_digits` significant bits, so
/// the final conversion to `f64` is exact. Keeping the discarded integer bits
/// here avoids double rounding through an intermediate `f64`.
fn round_int_to_f64(value: IntWide, mantissa_digits: u32, rounding: CastValueRoundingMode) -> f64 {
    debug_assert!((1..=53).contains(&mantissa_digits));
    let (magnitude, negative) = value.magnitude_and_sign();
    let signed = |value: f64| if negative { -value } else { value };
    let significant_bits = u64::BITS - magnitude.leading_zeros();
    if significant_bits <= mantissa_digits {
        #[expect(clippy::cast_precision_loss)]
        return signed(magnitude as f64);
    }

    let shift = significant_bits - mantissa_digits;
    let unit = 1_u64 << shift;
    let remainder = magnitude & (unit - 1);
    let truncated = magnitude - remainder;
    let increment = match rounding {
        CastValueRoundingMode::NearestEven => {
            let halfway = unit >> 1;
            remainder > halfway || (remainder == halfway && ((magnitude >> shift) & 1) != 0)
        }
        CastValueRoundingMode::TowardsZero => false,
        CastValueRoundingMode::TowardsPositive => !negative && remainder != 0,
        CastValueRoundingMode::TowardsNegative => negative && remainder != 0,
        CastValueRoundingMode::NearestAway => remainder >= (unit >> 1),
    };
    let rounded = if increment {
        let Some(rounded) = truncated.checked_add(unit) else {
            return signed(TWO_POW_64);
        };
        rounded
    } else {
        truncated
    };
    #[expect(clippy::cast_precision_loss)]
    signed(rounded as f64)
}

fn kernel_int_to_float<S: KernelInt, F: KernelFloatTarget>(
    source: &[u8],
    output: &mut Vec<u8>,
    params: CastKernelParams,
) -> Result<(), CastValueError> {
    let num_elements = check_alignment(source, S::SIZE)?;
    output.reserve(num_elements * F::SIZE);
    let min = F::min_f64();
    let max = F::max_f64();
    for element in source.chunks_exact(S::SIZE) {
        let value = S::read(element).widen();
        // This conversion is used only for target-range handling. It is exact
        // whenever an integer can be near the bounds of a supported target.
        let range_value = round_int_to_f64(value, f64::MANTISSA_DIGITS, params.rounding);
        if (min..=max).contains(&range_value) {
            let rounded = round_int_to_f64(value, F::MANTISSA_DIGITS, params.rounding);
            F::round_write(rounded, params.rounding, output);
        } else {
            let quantity = float_quantity_from_f64(
                range_value,
                params.rounding,
                params.out_of_range,
                min,
                max,
                true,
                true,
            )?;
            F::round_write(quantity, params.rounding, output);
        }
    }
    Ok(())
}

fn kernel_to_float<S: KernelToF64, F: KernelFloatTarget>(
    source: &[u8],
    output: &mut Vec<u8>,
    params: CastKernelParams,
) -> Result<(), CastValueError> {
    let num_elements = check_alignment(source, S::SIZE)?;
    output.reserve(num_elements * F::SIZE);
    let min = F::min_f64();
    let max = F::max_f64();
    for element in source.chunks_exact(S::SIZE) {
        let value = S::read_to_f64(element, params.rounding);
        let quantity = float_quantity_from_f64(
            value,
            params.rounding,
            params.out_of_range,
            min,
            max,
            true,
            true,
        )?;
        F::round_write(quantity, params.rounding, output);
    }
    Ok(())
}

/// 2^63 and 2^64: the `i64`/`u64` domain limits, exact in `f64`.
const TWO_POW_63: f64 = 9_223_372_036_854_775_808.0;
const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;

fn kernel_float_to_int<S: KernelFloatSource, T: KernelInt>(
    source: &[u8],
    output: &mut Vec<u8>,
    params: CastKernelParams,
) -> Result<(), CastValueError> {
    let num_elements = check_alignment(source, S::SIZE)?;
    output.reserve(num_elements * T::SIZE);
    let bits = params.target_bits;
    let bounds = int_bounds(bits, T::SIGNED);
    for element in source.chunks_exact(S::SIZE) {
        let value = S::read_f64(element);
        if value.is_nan() || value.is_infinite() {
            return Err(CastValueError::NotRepresentable);
        }
        let rounded = round_float_to_integer(value, params.rounding);
        #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let wide = if (-TWO_POW_63..TWO_POW_63).contains(&rounded) {
            IntWide::I(rounded as i64) // integer-valued and in range: exact
        } else if !T::SIGNED && (0.0..TWO_POW_64).contains(&rounded) {
            IntWide::U(rounded as u64)
        } else {
            // beyond the 64-bit domain: out of range for every supported target
            match params.out_of_range {
                Some(CastValueOutOfRangeMode::Clamp) => {
                    if T::SIGNED {
                        T::write_i64(
                            if rounded < 0.0 {
                                bounds.min_i
                            } else {
                                bounds.max_i
                            },
                            output,
                        );
                    } else {
                        T::write_u64(if rounded < 0.0 { 0 } else { bounds.max_u }, output);
                    }
                    continue;
                }
                Some(CastValueOutOfRangeMode::Wrap) => {
                    // rare branch; `wrap_float` is exact for these magnitudes
                    let wrapped = wrap_float(rounded, bits);
                    write_wrapped::<T>(wrapped as u64, bits, output); // < 2^bits: fits
                    continue;
                }
                None => return Err(CastValueError::NotRepresentable),
            }
        };
        write_int::<T>(wide, bits, bounds, params.out_of_range, output)?;
    }
    Ok(())
}

fn valid_int_bits(bits: u32, stored: CastValueIntStored) -> bool {
    (1..=stored.bits()).contains(&bits)
}

macro_rules! match_int_stored {
    ($stored:expr, $kernel:ident, $s:ty) => {
        match $stored {
            CastValueIntStored::I8 => $kernel::<$s, i8> as CastKernelFn,
            CastValueIntStored::I16 => $kernel::<$s, i16> as CastKernelFn,
            CastValueIntStored::I32 => $kernel::<$s, i32> as CastKernelFn,
            CastValueIntStored::I64 => $kernel::<$s, i64> as CastKernelFn,
            CastValueIntStored::U8 => $kernel::<$s, u8> as CastKernelFn,
            CastValueIntStored::U16 => $kernel::<$s, u16> as CastKernelFn,
            CastValueIntStored::U32 => $kernel::<$s, u32> as CastKernelFn,
            CastValueIntStored::U64 => $kernel::<$s, u64> as CastKernelFn,
        }
    };
}

fn select_from_int<S: KernelInt>(target: CastValueRepr) -> Option<CastKernelFn> {
    Some(match target {
        CastValueRepr::Int { bits, stored } => {
            if !valid_int_bits(bits, stored) {
                return None;
            }
            match_int_stored!(stored, kernel_int_to_int, S)
        }
        CastValueRepr::F16 => kernel_int_to_float::<S, f16> as CastKernelFn,
        CastValueRepr::BF16 => kernel_int_to_float::<S, bf16> as CastKernelFn,
        CastValueRepr::F32 => kernel_int_to_float::<S, f32> as CastKernelFn,
        CastValueRepr::F64 => kernel_int_to_float::<S, f64> as CastKernelFn,
    })
}

fn select_from_float<S: KernelFloatSource + KernelToF64>(
    target: CastValueRepr,
) -> Option<CastKernelFn> {
    Some(match target {
        CastValueRepr::Int { bits, stored } => {
            if !valid_int_bits(bits, stored) {
                return None;
            }
            match_int_stored!(stored, kernel_float_to_int, S)
        }
        CastValueRepr::F16 => kernel_to_float::<S, f16> as CastKernelFn,
        CastValueRepr::BF16 => kernel_to_float::<S, bf16> as CastKernelFn,
        CastValueRepr::F32 => kernel_to_float::<S, f32> as CastKernelFn,
        CastValueRepr::F64 => kernel_to_float::<S, f64> as CastKernelFn,
    })
}

/// Select a monomorphised bulk cast kernel for a (source, target, configuration) triple.
///
/// Returns `None` if no kernel is available (the caller should fall back to
/// the generic scalar path) or if an integer representation is invalid.
#[must_use]
pub fn select_cast_kernel(
    source: CastValueRepr,
    target: CastValueRepr,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Option<CastValueKernel> {
    let kernel = match source {
        CastValueRepr::Int { bits, stored } => {
            if !valid_int_bits(bits, stored) {
                return None;
            }
            match stored {
                CastValueIntStored::I8 => select_from_int::<i8>(target),
                CastValueIntStored::I16 => select_from_int::<i16>(target),
                CastValueIntStored::I32 => select_from_int::<i32>(target),
                CastValueIntStored::I64 => select_from_int::<i64>(target),
                CastValueIntStored::U8 => select_from_int::<u8>(target),
                CastValueIntStored::U16 => select_from_int::<u16>(target),
                CastValueIntStored::U32 => select_from_int::<u32>(target),
                CastValueIntStored::U64 => select_from_int::<u64>(target),
            }
        }
        CastValueRepr::F16 => select_from_float::<f16>(target),
        CastValueRepr::BF16 => select_from_float::<bf16>(target),
        CastValueRepr::F32 => select_from_float::<f32>(target),
        CastValueRepr::F64 => select_from_float::<f64>(target),
    }?;
    let target_bits = if let CastValueRepr::Int { bits, .. } = target {
        bits
    } else {
        0
    };
    Some(CastValueKernel {
        kernel,
        params: CastKernelParams {
            target_bits,
            rounding,
            out_of_range,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const INT64: CastValueRepr = CastValueRepr::Int {
        bits: 64,
        stored: CastValueIntStored::I64,
    };
    const UINT64: CastValueRepr = CastValueRepr::Int {
        bits: 64,
        stored: CastValueIntStored::U64,
    };
    const UINT8: CastValueRepr = CastValueRepr::Int {
        bits: 8,
        stored: CastValueIntStored::U8,
    };
    const INT2: CastValueRepr = CastValueRepr::Int {
        bits: 2,
        stored: CastValueIntStored::I8,
    };

    #[test]
    fn kernel_selection() {
        for target in [
            UINT8,
            INT2,
            CastValueRepr::F16,
            CastValueRepr::BF16,
            CastValueRepr::F32,
            CastValueRepr::F64,
        ] {
            for source in [UINT8, INT2, CastValueRepr::F32, CastValueRepr::F64] {
                assert!(
                    select_cast_kernel(source, target, CastValueRoundingMode::NearestEven, None)
                        .is_some(),
                    "{source:?} -> {target:?}"
                );
            }
        }
        // invalid logical bits
        let invalid = CastValueRepr::Int {
            bits: 9,
            stored: CastValueIntStored::U8,
        };
        assert!(
            select_cast_kernel(invalid, UINT8, CastValueRoundingMode::NearestEven, None).is_none()
        );
        assert!(
            select_cast_kernel(UINT8, invalid, CastValueRoundingMode::NearestEven, None).is_none()
        );
    }

    #[test]
    fn kernel_int_to_int_wrap() {
        // int16 -> int8 wrap, matching the spec examples
        let kernel = select_cast_kernel(
            CastValueRepr::Int {
                bits: 16,
                stored: CastValueIntStored::I16,
            },
            CastValueRepr::Int {
                bits: 8,
                stored: CastValueIntStored::I8,
            },
            CastValueRoundingMode::NearestEven,
            Some(CastValueOutOfRangeMode::Wrap),
        )
        .unwrap();
        let source: Vec<u8> = [127_i16, 128, 129, -129]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let mut output = Vec::new();
        kernel.cast(&source, &mut output).unwrap();
        let expected: Vec<u8> = [127_i8, -128, -127, 127]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        assert_eq!(output, expected);
    }

    fn cast_u64_to_f32(value: u64, rounding: CastValueRoundingMode) -> u32 {
        let kernel = select_cast_kernel(UINT64, CastValueRepr::F32, rounding, None).unwrap();
        let mut output = Vec::new();
        kernel.cast(&value.to_ne_bytes(), &mut output).unwrap();
        f32::from_ne_bytes(output.try_into().unwrap()).to_bits()
    }

    fn scalar_u64_to_f32(value: u64, rounding: CastValueRoundingMode) -> u32 {
        let quantity = super::super::cast_scalar_to_float_quantity_with_precision(
            super::super::CastValueScalar::Unsigned(u128::from(value)),
            f32::MANTISSA_DIGITS,
            rounding,
            None,
            f64::from(f32::MIN),
            f64::from(f32::MAX),
            true,
            true,
        )
        .unwrap();
        round_f64_to_f32(quantity, rounding).to_bits()
    }

    fn cast_i64_to_f32(value: i64, rounding: CastValueRoundingMode) -> u32 {
        let kernel = select_cast_kernel(INT64, CastValueRepr::F32, rounding, None).unwrap();
        let mut output = Vec::new();
        kernel.cast(&value.to_ne_bytes(), &mut output).unwrap();
        f32::from_ne_bytes(output.try_into().unwrap()).to_bits()
    }

    #[test]
    fn kernel_int_to_f32_rounds_once() {
        const LOWER: u64 = 1_u64 << 63;
        const HALFWAY: u64 = LOWER + (1_u64 << 39);
        const LOWER_BITS: u32 = 0x5f00_0000;
        const UPPER_BITS: u32 = 0x5f00_0001;

        assert_eq!(
            cast_u64_to_f32(HALFWAY + 1, CastValueRoundingMode::NearestEven),
            UPPER_BITS
        );
        assert_eq!(
            scalar_u64_to_f32(HALFWAY + 1, CastValueRoundingMode::NearestEven),
            UPPER_BITS
        );
        assert_eq!(
            cast_u64_to_f32(HALFWAY - 1, CastValueRoundingMode::NearestAway),
            LOWER_BITS
        );
        assert_eq!(
            scalar_u64_to_f32(HALFWAY - 1, CastValueRoundingMode::NearestAway),
            LOWER_BITS
        );
        assert_eq!(
            cast_u64_to_f32(HALFWAY, CastValueRoundingMode::NearestEven),
            LOWER_BITS
        );
        assert_eq!(
            cast_u64_to_f32(HALFWAY, CastValueRoundingMode::NearestAway),
            UPPER_BITS
        );

        assert_eq!(
            cast_u64_to_f32(HALFWAY + 1, CastValueRoundingMode::TowardsZero),
            LOWER_BITS
        );
        assert_eq!(
            cast_u64_to_f32(HALFWAY + 1, CastValueRoundingMode::TowardsPositive),
            UPPER_BITS
        );
        assert_eq!(
            cast_u64_to_f32(HALFWAY + 1, CastValueRoundingMode::TowardsNegative),
            LOWER_BITS
        );

        const NEGATIVE_HALFWAY: i64 = -((1_i64 << 62) + (1_i64 << 38));
        const NEGATIVE_LOWER_BITS: u32 = 0xde80_0000;
        const NEGATIVE_UPPER_BITS: u32 = 0xde80_0001;
        assert_eq!(
            cast_i64_to_f32(NEGATIVE_HALFWAY - 1, CastValueRoundingMode::NearestEven),
            NEGATIVE_UPPER_BITS
        );
        assert_eq!(
            cast_i64_to_f32(NEGATIVE_HALFWAY + 1, CastValueRoundingMode::NearestAway),
            NEGATIVE_LOWER_BITS
        );
        assert_eq!(
            cast_i64_to_f32(NEGATIVE_HALFWAY - 1, CastValueRoundingMode::TowardsPositive),
            NEGATIVE_LOWER_BITS
        );
        assert_eq!(
            cast_i64_to_f32(NEGATIVE_HALFWAY - 1, CastValueRoundingMode::TowardsNegative),
            NEGATIVE_UPPER_BITS
        );
    }

    #[test]
    fn kernel_int_to_bf16_rounds_once() {
        const LOWER: u64 = 1_u64 << 63;
        const HALFWAY: u64 = LOWER + (1_u64 << 55);

        let cast = |value, rounding| {
            let kernel = select_cast_kernel(UINT64, CastValueRepr::BF16, rounding, None).unwrap();
            let mut output = Vec::new();
            kernel.cast(&u64::to_ne_bytes(value), &mut output).unwrap();
            bf16::from_ne_bytes(output.try_into().unwrap()).to_bits()
        };

        assert_eq!(
            cast(HALFWAY + 1, CastValueRoundingMode::NearestEven),
            0x5f01
        );
        assert_eq!(
            cast(HALFWAY - 1, CastValueRoundingMode::NearestAway),
            0x5f00
        );
    }
}
