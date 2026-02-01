//! The `bitround` codec data type traits.

fn round_bits8(mut input: u8, keepbits: u32, maxbits: u32) -> u8 {
    if keepbits < maxbits {
        let maskbits = maxbits - keepbits;
        let all_set = u8::MAX;
        let mask = (all_set >> maskbits) << maskbits;
        let half_quantum1 = (1 << (maskbits - 1)) - 1;
        input = input.saturating_add(((input >> maskbits) & 1) + half_quantum1) & mask;
    }
    input
}

const fn round_bits16(mut input: u16, keepbits: u32, maxbits: u32) -> u16 {
    if keepbits < maxbits {
        let maskbits = maxbits - keepbits;
        let all_set = u16::MAX;
        let mask = (all_set >> maskbits) << maskbits;
        let half_quantum1 = (1 << (maskbits - 1)) - 1;
        input = input.saturating_add(((input >> maskbits) & 1) + half_quantum1) & mask;
    }
    input
}

const fn round_bits32(mut input: u32, keepbits: u32, maxbits: u32) -> u32 {
    if keepbits < maxbits {
        let maskbits = maxbits - keepbits;
        let all_set = u32::MAX;
        let mask = (all_set >> maskbits) << maskbits;
        let half_quantum1 = (1 << (maskbits - 1)) - 1;
        input = input.saturating_add(((input >> maskbits) & 1) + half_quantum1) & mask;
    }
    input
}

const fn round_bits64(mut input: u64, keepbits: u32, maxbits: u32) -> u64 {
    if keepbits < maxbits {
        let maskbits = maxbits - keepbits;
        let all_set = u64::MAX;
        let mask = (all_set >> maskbits) << maskbits;
        let half_quantum1 = (1 << (maskbits - 1)) - 1;
        input = input.saturating_add(((input >> maskbits) & 1) + half_quantum1) & mask;
    }
    input
}

/// Helper to round 8-bit integer values (from MSB).
pub fn round_bytes_int8(bytes: &mut [u8], keepbits: u32) {
    for element in bytes.iter_mut() {
        *element = round_bits8(*element, keepbits, 8 - element.leading_zeros());
    }
}

/// Helper to round 16-bit integer values (from MSB).
///
/// # Panics
/// Panics if `bytes.len()` is not a multiple of 2.
pub fn round_bytes_int16(bytes: &mut [u8], keepbits: u32) {
    for chunk in bytes.as_chunks_mut::<2>().0 {
        let element = u16::from_ne_bytes(*chunk);
        let rounded = round_bits16(element, keepbits, 16 - element.leading_zeros());
        chunk.copy_from_slice(&u16::to_ne_bytes(rounded));
    }
}

/// Helper to round 32-bit integer values (from MSB).
///
/// # Panics
/// Panics if `bytes.len()` is not a multiple of 4.
pub fn round_bytes_int32(bytes: &mut [u8], keepbits: u32) {
    for chunk in bytes.as_chunks_mut::<4>().0 {
        let element = u32::from_ne_bytes(*chunk);
        let rounded = round_bits32(element, keepbits, 32 - element.leading_zeros());
        chunk.copy_from_slice(&u32::to_ne_bytes(rounded));
    }
}

/// Helper to round 64-bit integer values (from MSB).
///
/// # Panics
/// Panics if `bytes.len()` is not a multiple of 8.
pub fn round_bytes_int64(bytes: &mut [u8], keepbits: u32) {
    for chunk in bytes.as_chunks_mut::<8>().0 {
        let element = u64::from_ne_bytes(*chunk);
        let rounded = round_bits64(element, keepbits, 64 - element.leading_zeros());
        chunk.copy_from_slice(&u64::to_ne_bytes(rounded));
    }
}

/// Helper to round 16-bit float values (fixed mantissa bits).
///
/// # Panics
/// Panics if `bytes.len()` is not a multiple of 2.
pub fn round_bytes_float16(bytes: &mut [u8], keepbits: u32, mantissa_bits: u32) {
    for chunk in bytes.as_chunks_mut::<2>().0 {
        let element = u16::from_ne_bytes(*chunk);
        let rounded = round_bits16(element, keepbits, mantissa_bits);
        chunk.copy_from_slice(&u16::to_ne_bytes(rounded));
    }
}

/// Helper to round 32-bit float values (fixed mantissa bits).
///
/// # Panics
/// Panics if `bytes.len()` is not a multiple of 4.
pub fn round_bytes_float32(bytes: &mut [u8], keepbits: u32, mantissa_bits: u32) {
    for chunk in bytes.as_chunks_mut::<4>().0 {
        let element = u32::from_ne_bytes(*chunk);
        let rounded = round_bits32(element, keepbits, mantissa_bits);
        chunk.copy_from_slice(&u32::to_ne_bytes(rounded));
    }
}

/// Helper to round 64-bit float values (fixed mantissa bits).
///
/// # Panics
/// Panics if `bytes.len()` is not a multiple of 8.
pub fn round_bytes_float64(bytes: &mut [u8], keepbits: u32, mantissa_bits: u32) {
    for chunk in bytes.as_chunks_mut::<8>().0 {
        let element = u64::from_ne_bytes(*chunk);
        let rounded = round_bits64(element, keepbits, mantissa_bits);
        chunk.copy_from_slice(&u64::to_ne_bytes(rounded));
    }
}

/// Traits for a data type supporting the `bitround` codec.
///
/// The bitround codec rounds the mantissa of floating point data types or
/// rounds integers from the most significant set bit to the specified number of bits.
pub trait BitroundDataTypeTraits {
    /// The number of bits to round to for floating point types.
    ///
    /// Returns `None` for integer types where rounding is from the MSB.
    fn mantissa_bits(&self) -> Option<u32>;

    /// Apply bit rounding to the bytes in-place.
    ///
    /// # Arguments
    /// * `bytes` - The bytes to round in-place
    /// * `keepbits` - The number of bits to keep
    fn round(&self, bytes: &mut [u8], keepbits: u32);
}

// Generate the codec support infrastructure using the generic macro
crate::define_data_type_support!(Bitround);

/// Macro to implement `BitroundDataTypeTraits` for data types and register support.
///
/// # Usage
/// ```ignore
/// // Float types (have mantissa bits):
/// impl_bitround_codec!(Float32DataType, 4, float32, 23);
/// impl_bitround_codec!(Float64DataType, 8, float64, 52);
/// impl_bitround_codec!(Float16DataType, 2, float16, 10);
/// impl_bitround_codec!(BFloat16DataType, 2, float16, 7);
///
/// // Integer types (no mantissa bits):
/// impl_bitround_codec!(Int32DataType, 4, int32);
/// impl_bitround_codec!(Int64DataType, 8, int64);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_bitround_codec {
    // Float16/BFloat16 types (use round_bytes_float16 with specified mantissa bits)
    ($marker:ty, 2, float16, $mantissa_bits:expr) => {
        impl $crate::codec_traits::bitround::BitroundDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                Some($mantissa_bits)
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::codec_traits::bitround::round_bytes_float16(
                    bytes,
                    keepbits,
                    $mantissa_bits,
                );
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bitround::BitroundDataTypePlugin,
            $crate::codec_traits::bitround::BitroundDataTypeTraits
        );
    };
    // Float32 types
    ($marker:ty, 4, float32, $mantissa_bits:expr) => {
        impl $crate::codec_traits::bitround::BitroundDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                Some($mantissa_bits)
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::codec_traits::bitround::round_bytes_float32(
                    bytes,
                    keepbits,
                    $mantissa_bits,
                );
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bitround::BitroundDataTypePlugin,
            $crate::codec_traits::bitround::BitroundDataTypeTraits
        );
    };
    // Float64 types
    ($marker:ty, 8, float64, $mantissa_bits:expr) => {
        impl $crate::codec_traits::bitround::BitroundDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                Some($mantissa_bits)
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::codec_traits::bitround::round_bytes_float64(
                    bytes,
                    keepbits,
                    $mantissa_bits,
                );
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bitround::BitroundDataTypePlugin,
            $crate::codec_traits::bitround::BitroundDataTypeTraits
        );
    };
    // Int8 types (no mantissa, round from MSB)
    ($marker:ty, 1, int8) => {
        impl $crate::codec_traits::bitround::BitroundDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::codec_traits::bitround::round_bytes_int8(bytes, keepbits);
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bitround::BitroundDataTypePlugin,
            $crate::codec_traits::bitround::BitroundDataTypeTraits
        );
    };
    // Int16 types
    ($marker:ty, 2, int16) => {
        impl $crate::codec_traits::bitround::BitroundDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::codec_traits::bitround::round_bytes_int16(bytes, keepbits);
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bitround::BitroundDataTypePlugin,
            $crate::codec_traits::bitround::BitroundDataTypeTraits
        );
    };
    // Int32 types
    ($marker:ty, 4, int32) => {
        impl $crate::codec_traits::bitround::BitroundDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::codec_traits::bitround::round_bytes_int32(bytes, keepbits);
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bitround::BitroundDataTypePlugin,
            $crate::codec_traits::bitround::BitroundDataTypeTraits
        );
    };
    // Int64 types
    ($marker:ty, 8, int64) => {
        impl $crate::codec_traits::bitround::BitroundDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::codec_traits::bitround::round_bytes_int64(bytes, keepbits);
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bitround::BitroundDataTypePlugin,
            $crate::codec_traits::bitround::BitroundDataTypeTraits
        );
    };
    // UInt8 types (use int8 rounding function)
    ($marker:ty, 1, uint8) => {
        impl $crate::codec_traits::bitround::BitroundDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::codec_traits::bitround::round_bytes_int8(bytes, keepbits);
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bitround::BitroundDataTypePlugin,
            $crate::codec_traits::bitround::BitroundDataTypeTraits
        );
    };
    // UInt16 types (use int16 rounding function)
    ($marker:ty, 2, uint16) => {
        impl $crate::codec_traits::bitround::BitroundDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::codec_traits::bitround::round_bytes_int16(bytes, keepbits);
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bitround::BitroundDataTypePlugin,
            $crate::codec_traits::bitround::BitroundDataTypeTraits
        );
    };
    // UInt32 types (use int32 rounding function)
    ($marker:ty, 4, uint32) => {
        impl $crate::codec_traits::bitround::BitroundDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::codec_traits::bitround::round_bytes_int32(bytes, keepbits);
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bitround::BitroundDataTypePlugin,
            $crate::codec_traits::bitround::BitroundDataTypeTraits
        );
    };
    // UInt64 types (use int64 rounding function)
    ($marker:ty, 8, uint64) => {
        impl $crate::codec_traits::bitround::BitroundDataTypeTraits for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                $crate::codec_traits::bitround::round_bytes_int64(bytes, keepbits);
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::bitround::BitroundDataTypePlugin,
            $crate::codec_traits::bitround::BitroundDataTypeTraits
        );
    };
}

#[doc(inline)]
pub use _impl_bitround_codec as impl_bitround_codec;
