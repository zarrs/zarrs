use std::ops::Deref;
use std::sync::Arc;

use derive_more::derive::Display;
use thiserror::Error;

/// Array element byte offsets.
///
/// Cloning retains the offset allocation without copying its elements.
/// These must be monotonically increasing. See [`ArrayBytes`](crate::ArrayBytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayBytesOffsets(Arc<Vec<usize>>);

impl Deref for ArrayBytesOffsets {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// An error creating [`ArrayBytesOffsets`].
#[derive(Clone, Debug, Display, Error)]
pub enum ArrayBytesOffsetsCreateError {
    /// The offsets length must be greater than zero.
    #[display("offsets length must be greater than zero")]
    ZeroLength,
    /// The offsets are not monotonically increasing.
    #[display("offsets are not monotonically increasing")]
    NotMonotonicallyIncreasing,
}

impl ArrayBytesOffsets {
    /// Creates a new `ArrayBytesOffsets`.
    ///
    /// # Errors
    /// Returns an error if the offsets are not monotonically increasing.
    pub fn new(offsets: impl Into<Vec<usize>>) -> Result<Self, ArrayBytesOffsetsCreateError> {
        let offsets = offsets.into();
        if offsets.is_empty() {
            Err(ArrayBytesOffsetsCreateError::ZeroLength)
        } else if offsets.windows(2).all(|w| w[1] >= w[0]) {
            Ok(Self(Arc::new(offsets)))
        } else {
            Err(ArrayBytesOffsetsCreateError::NotMonotonicallyIncreasing)
        }
    }

    /// Creates a new `ArrayBytesOffsets` without checking the offsets.
    ///
    /// # Safety
    /// The offsets must be monotonically increasing.
    #[must_use]
    pub unsafe fn new_unchecked(offsets: impl Into<Vec<usize>>) -> Self {
        let offsets = offsets.into();
        debug_assert!(!offsets.is_empty());
        debug_assert!(offsets.windows(2).all(|w| w[1] >= w[0]));
        Self(Arc::new(offsets))
    }

    /// Returns the last offset.
    #[must_use]
    pub fn last(&self) -> usize {
        unsafe {
            // SAFETY: The offsets cannot be empty.
            *self.0.last().unwrap_unchecked()
        }
    }
}

impl TryFrom<&[usize]> for ArrayBytesOffsets {
    type Error = ArrayBytesOffsetsCreateError;

    fn try_from(value: &[usize]) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<const N: usize> TryFrom<&[usize; N]> for ArrayBytesOffsets {
    type Error = ArrayBytesOffsetsCreateError;

    fn try_from(value: &[usize; N]) -> Result<Self, Self::Error> {
        Self::new(value.as_slice())
    }
}

impl TryFrom<Vec<usize>> for ArrayBytesOffsets {
    type Error = ArrayBytesOffsetsCreateError;

    fn try_from(value: Vec<usize>) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_bytes_offsets() {
        let owned = vec![0, 2, 4];
        let pointer = owned.as_ptr();
        let shared = ArrayBytesOffsets::new(owned).unwrap();
        assert_eq!(shared.as_ptr(), pointer);
        assert_eq!(shared.clone().as_ptr(), pointer);
        let offsets = ArrayBytesOffsets::new(vec![0, 1, 2, 3]).unwrap();
        assert_eq!(&*offsets, &[0, 1, 2, 3]);
        assert!(ArrayBytesOffsets::new(vec![]).is_err());
        assert!(ArrayBytesOffsets::new(vec![0]).is_ok());
        assert!(ArrayBytesOffsets::new(vec![10]).is_ok()); // nonsense, but not invalid
        assert!(ArrayBytesOffsets::new(vec![0, 1, 1]).is_ok());
        assert!(ArrayBytesOffsets::new(vec![0, 1, 0]).is_err());
        assert!(ArrayBytesOffsets::try_from(vec![0, 1, 2]).is_ok());
        assert!(ArrayBytesOffsets::try_from(vec![0, 1, 0]).is_err());
        assert!(ArrayBytesOffsets::try_from([0, 1, 2].as_slice()).is_ok());
        assert!(ArrayBytesOffsets::try_from([0, 1, 0].as_slice()).is_err());
        assert!(ArrayBytesOffsets::try_from(&[0, 1, 2]).is_ok());
        assert!(ArrayBytesOffsets::try_from(&[0, 1, 0]).is_err());
    }
}
