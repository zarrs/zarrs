use std::ops::Deref;

use derive_more::derive::Display;
use thiserror::Error;

use crate::CowBytes;

/// The validity mask of [`ArrayBytesOptional`](crate::ArrayBytesOptional).
///
/// Each element has one byte, guaranteed to be 0 (null/missing) or 1 (present):
///
/// - A mask of [`bool`]s is converted without copying or checking.
/// - A mask of raw bytes is checked once by [`new`](ArrayBytesMask::new).
///
/// # Examples
///
/// ```
/// use zarrs_codec::ArrayBytesMask;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mask = ArrayBytesMask::from(vec![true, false, true]);
/// assert_eq!(&*mask, &[1, 0, 1]);
/// assert_eq!(mask.num_present(), 2);
///
/// let mask = ArrayBytesMask::new(vec![1u8, 0, 1])?;
/// assert_eq!(mask.num_present(), 2);
/// assert!(ArrayBytesMask::new(vec![1u8, 2]).is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ArrayBytesMask<'a> {
    /// Invariant: every byte is 0 or 1.
    bytes: CowBytes<'a>,
}

/// An error creating an [`ArrayBytesMask`].
#[derive(Clone, Copy, Debug, Display, Error)]
#[display("optional mask values must be 0 (null) or 1 (present)")]
pub struct ArrayBytesMaskCreateError;

/// Returns true if every byte of `mask` is 0 or 1.
fn is_binary(mask: &[u8]) -> bool {
    // Branchless, so it vectorises
    mask.iter().fold(0u8, |acc, &b| acc | b) <= 1
}

impl<'a> ArrayBytesMask<'a> {
    /// Creates a new `ArrayBytesMask`, checking its bytes.
    ///
    /// The bytes are retained without copying.
    ///
    /// # Errors
    /// Returns an [`ArrayBytesMaskCreateError`] if a byte is not 0 or 1.
    pub fn new(bytes: impl Into<CowBytes<'a>>) -> Result<Self, ArrayBytesMaskCreateError> {
        let bytes = bytes.into();
        if is_binary(&bytes) {
            Ok(Self { bytes })
        } else {
            Err(ArrayBytesMaskCreateError)
        }
    }

    /// Creates a new `ArrayBytesMask` without checking the bytes.
    ///
    /// # Safety
    /// Every byte must be 0 or 1.
    #[must_use]
    pub unsafe fn new_unchecked(bytes: impl Into<CowBytes<'a>>) -> Self {
        let bytes = bytes.into();
        debug_assert!(is_binary(&bytes));
        Self { bytes }
    }

    /// Returns the number of present (non-null) elements.
    #[must_use]
    pub fn num_present(&self) -> usize {
        // A block of up to 255 present elements sums without overflowing a `u8`.
        // Summing each block in `u8` lanes vectorises, unlike widening every byte, and is about 7 times faster.
        self.bytes
            .chunks(usize::from(u8::MAX))
            .map(|block| usize::from(block.iter().fold(0u8, |n, &b| n.wrapping_add(b))))
            .sum()
    }

    /// Converts into the mask bytes.
    #[must_use]
    pub fn into_bytes(self) -> CowBytes<'a> {
        self.bytes
    }

    /// Converts into an [`ArrayBytesMask<'static>`], copying only if the bytes are borrowed.
    #[must_use]
    pub fn into_static(self) -> ArrayBytesMask<'static> {
        ArrayBytesMask {
            bytes: self.bytes.into_static(),
        }
    }
}

impl Deref for ArrayBytesMask<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.bytes
    }
}

impl AsRef<[u8]> for ArrayBytesMask<'_> {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

impl From<Vec<bool>> for ArrayBytesMask<'_> {
    fn from(mask: Vec<bool>) -> Self {
        // Reinterpret the bools without copying
        Self {
            bytes: bytemuck::allocation::cast_vec::<bool, u8>(mask).into(),
        }
    }
}

impl<'a> From<&'a [bool]> for ArrayBytesMask<'a> {
    fn from(mask: &'a [bool]) -> Self {
        // Borrow the bools as bytes without copying
        Self {
            bytes: bytemuck::cast_slice::<bool, u8>(mask).into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_new() {
        assert_eq!(&*ArrayBytesMask::new(vec![0u8, 1, 1]).unwrap(), &[0, 1, 1]);
        assert!(ArrayBytesMask::new(vec![]).is_ok());
        for invalid in [2u8, 3, 0x80, 0xff] {
            assert!(ArrayBytesMask::new(vec![1, 0, invalid, 1]).is_err());
        }
    }

    #[test]
    fn mask_from_bool() {
        let bools = [true, false, true, true];
        let borrowed = ArrayBytesMask::from(&bools[..]);
        assert!(matches!(borrowed.bytes, CowBytes::Borrowed(_)));
        assert_eq!(borrowed.as_ptr(), bools.as_ptr().cast::<u8>());
        assert_eq!(&*borrowed, &[1, 0, 1, 1]);
        assert_eq!(borrowed.num_present(), 3);

        let bools = bools.to_vec();
        let ptr = bools.as_ptr().cast::<u8>();
        let owned = ArrayBytesMask::from(bools);
        assert_eq!(owned.as_ptr(), ptr);
        assert_eq!(owned, borrowed);
    }

    #[test]
    fn mask_num_present() {
        // Lengths around the 255-byte blocks of the count
        for len in [0, 1, 254, 255, 256, 510, 511, 1000] {
            let all = ArrayBytesMask::from(vec![true; len]);
            assert_eq!(all.num_present(), len);
            let none = ArrayBytesMask::from(vec![false; len]);
            assert_eq!(none.num_present(), 0);
            let pattern: Vec<bool> = (0..len).map(|i| i % 3 != 0).collect();
            let expected = pattern.iter().filter(|&&present| present).count();
            assert_eq!(ArrayBytesMask::from(pattern).num_present(), expected);
        }
    }
}
