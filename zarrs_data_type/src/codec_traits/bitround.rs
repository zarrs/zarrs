//! The `bitround` codec data type traits.

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
