/// Traits for a data type extension supporting the `bitround` codec.
///
/// The bitround codec rounds the mantissa of floating point data types or
/// rounds integers from the most significant set bit to the specified number of bits.
pub trait DataTypeExtensionBitroundCodec {
    /// The number of bits to round to for floating point types.
    ///
    /// Returns `None` for integer types where rounding is from the MSB.
    fn mantissa_bits(&self) -> Option<u32>;

    /// The size in bytes of each component to round.
    ///
    /// For complex types, this should be the size of each component (e.g., 4 for Complex64).
    fn component_size(&self) -> usize;

    /// Apply bit rounding to the bytes in-place.
    ///
    /// # Arguments
    /// * `bytes` - The bytes to round in-place
    /// * `keepbits` - The number of bits to keep
    fn round(&self, bytes: &mut [u8], keepbits: u32);
}

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
