use std::borrow::Cow;
use std::ops::Deref;

use derive_more::derive::Display;
use thiserror::Error;

/// Array element byte offsets.
///
/// These must be monotonically increasing. See [`ArrayBytes`](crate::ArrayBytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayBytesOffsets<'a>(Cow<'a, [usize]>);

impl Deref for ArrayBytesOffsets<'_> {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// An error creating [`ArrayBytesOffsets`].
#[derive(Clone, Debug, Display, Error)]
pub enum ArrayBytesRawOffsetsCreateError {
    /// The offsets length must be greater than zero.
    #[display("offsets length must be greater than zero")]
    ZeroLength,
    /// The offsets are not monotonically increasing.
    #[display("offsets are not monotonically increasing")]
    NotMonotonicallyIncreasing,
}

impl<'a> ArrayBytesOffsets<'a> {
    /// Creates a new `ArrayBytesOffsets`.
    ///
    /// # Errors
    /// Returns an error if the offsets are not monotonically increasing.
    pub fn new(
        offsets: impl Into<Cow<'a, [usize]>>,
    ) -> Result<Self, ArrayBytesRawOffsetsCreateError> {
        let offsets = offsets.into();
        if offsets.is_empty() {
            Err(ArrayBytesRawOffsetsCreateError::ZeroLength)
        } else if offsets.windows(2).all(|w| w[1] >= w[0]) {
            Ok(Self(offsets))
        } else {
            Err(ArrayBytesRawOffsetsCreateError::NotMonotonicallyIncreasing)
        }
    }

    /// Creates a new `ArrayBytesOffsets` without checking the offsets.
    ///
    /// # Safety
    /// The offsets must be monotonically increasing.
    #[must_use]
    pub unsafe fn new_unchecked(offsets: impl Into<Cow<'a, [usize]>>) -> Self {
        let offsets = offsets.into();
        debug_assert!(!offsets.is_empty());
        debug_assert!(offsets.windows(2).all(|w| w[1] >= w[0]));
        Self(offsets)
    }

    /// Clones the offsets if not already owned.
    #[must_use]
    pub fn into_owned(self) -> ArrayBytesOffsets<'static> {
        ArrayBytesOffsets(self.0.into_owned().into())
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

impl<'a> TryFrom<Cow<'a, [usize]>> for ArrayBytesOffsets<'a> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: Cow<'a, [usize]>) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<'a> TryFrom<&'a [usize]> for ArrayBytesOffsets<'a> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: &'a [usize]) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<'a, const N: usize> TryFrom<&'a [usize; N]> for ArrayBytesOffsets<'a> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: &'a [usize; N]) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl TryFrom<Vec<usize>> for ArrayBytesOffsets<'_> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: Vec<usize>) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_bytes_offsets() {
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
        assert!(ArrayBytesOffsets::try_from(Cow::Owned(vec![0, 1, 0])).is_err());
    }
}
