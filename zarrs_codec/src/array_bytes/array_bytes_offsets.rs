use std::borrow::Cow;
use std::slice;

use derive_more::derive::Display;
use thiserror::Error;

/// Array element byte offsets.
///
/// These must be monotonically increasing. See [`ArrayBytes`](crate::ArrayBytes).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArrayBytesOffsets<'a> {
    /// 32-bit byte offsets.
    U32(Cow<'a, [u32]>),
    /// 64-bit byte offsets.
    U64(Cow<'a, [u64]>),
}

/// Convert into [`ArrayBytesOffsets`] without validation.
pub trait IntoArrayBytesOffsetsUnchecked<'a> {
    /// Convert into [`ArrayBytesOffsets`] without validation.
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'a>;
}

/// Iterator over [`ArrayBytesOffsets`] as `usize` values.
#[derive(Clone, Debug)]
pub enum ArrayBytesOffsetsIter<'a> {
    U32(slice::Iter<'a, u32>),
    U64(slice::Iter<'a, u64>),
}

impl Iterator for ArrayBytesOffsetsIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::U32(iter) => iter.next().map(|offset| usize::try_from(*offset).unwrap()),
            Self::U64(iter) => iter.next().map(|offset| usize::try_from(*offset).unwrap()),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::U32(iter) => iter.size_hint(),
            Self::U64(iter) => iter.size_hint(),
        }
    }
}

impl ExactSizeIterator for ArrayBytesOffsetsIter<'_> {}

fn offsets_u32_are_monotonically_increasing(offsets: &[u32]) -> bool {
    offsets.windows(2).all(|w| w[1] >= w[0])
}

fn offsets_u64_are_monotonically_increasing(offsets: &[u64]) -> bool {
    offsets.windows(2).all(|w| w[1] >= w[0])
}

#[cfg(target_pointer_width = "32")]
fn usize_offsets_to_array_bytes_offsets(offsets: &[usize]) -> ArrayBytesOffsets<'static> {
    ArrayBytesOffsets::U32(Cow::Owned(
        offsets
            .iter()
            .map(|&offset| u32::try_from(offset).unwrap())
            .collect(),
    ))
}

#[cfg(target_pointer_width = "64")]
fn usize_offsets_to_array_bytes_offsets(offsets: &[usize]) -> ArrayBytesOffsets<'static> {
    ArrayBytesOffsets::U64(Cow::Owned(
        offsets
            .iter()
            .map(|&offset| u64::try_from(offset).unwrap())
            .collect(),
    ))
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
        offsets: impl TryInto<ArrayBytesOffsets<'a>, Error = ArrayBytesRawOffsetsCreateError>,
    ) -> Result<Self, ArrayBytesRawOffsetsCreateError> {
        offsets.try_into()
    }

    fn validate(self) -> Result<Self, ArrayBytesRawOffsetsCreateError> {
        if self.is_empty() {
            Err(ArrayBytesRawOffsetsCreateError::ZeroLength)
        } else if self.is_monotonically_increasing() {
            Ok(self)
        } else {
            Err(ArrayBytesRawOffsetsCreateError::NotMonotonicallyIncreasing)
        }
    }

    /// Creates a new `ArrayBytesOffsets` without checking the offsets.
    ///
    /// # Safety
    /// The offsets must be monotonically increasing.
    #[must_use]
    pub unsafe fn new_unchecked(offsets: impl IntoArrayBytesOffsetsUnchecked<'a>) -> Self {
        let offsets = offsets.into_array_bytes_offsets_unchecked();
        debug_assert!(!offsets.is_empty());
        debug_assert!(offsets.is_monotonically_increasing());
        offsets
    }

    /// Creates a new `ArrayBytesOffsets` from native `usize` offsets using the same offset width as `self`.
    ///
    /// # Safety
    /// The offsets must be monotonically increasing. If `self` stores `u32` offsets, all offsets must fit in `u32`.
    #[must_use]
    pub unsafe fn new_unchecked_like(&self, offsets: &[usize]) -> ArrayBytesOffsets<'static> {
        debug_assert!(!offsets.is_empty());
        debug_assert!(offsets.windows(2).all(|w| w[1] >= w[0]));
        match self {
            Self::U32(_) => ArrayBytesOffsets::U32(Cow::Owned(
                offsets
                    .iter()
                    .map(|&offset| u32::try_from(offset).unwrap())
                    .collect(),
            )),
            Self::U64(_) => ArrayBytesOffsets::U64(Cow::Owned(
                offsets
                    .iter()
                    .map(|&offset| u64::try_from(offset).unwrap())
                    .collect(),
            )),
        }
    }

    /// Clones the offsets if not already owned.
    #[must_use]
    pub fn into_owned(self) -> ArrayBytesOffsets<'static> {
        match self {
            Self::U32(offsets) => ArrayBytesOffsets::U32(Cow::Owned(offsets.into_owned())),
            Self::U64(offsets) => ArrayBytesOffsets::U64(Cow::Owned(offsets.into_owned())),
        }
    }

    /// Returns the number of offsets.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U32(offsets) => offsets.len(),
            Self::U64(offsets) => offsets.len(),
        }
    }

    /// Returns `true` if there are no offsets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the offset at `index`.
    #[must_use]
    pub fn get(&self, index: usize) -> usize {
        match self {
            Self::U32(offsets) => usize::try_from(offsets[index]).unwrap(),
            Self::U64(offsets) => usize::try_from(offsets[index]).unwrap(),
        }
    }

    /// Returns an iterator over the offsets as `usize` values.
    #[must_use]
    pub fn iter(&self) -> ArrayBytesOffsetsIter<'_> {
        match self {
            Self::U32(offsets) => ArrayBytesOffsetsIter::U32(offsets.iter()),
            Self::U64(offsets) => ArrayBytesOffsetsIter::U64(offsets.iter()),
        }
    }

    /// Returns the last offset.
    #[must_use]
    pub fn last(&self) -> usize {
        match self {
            Self::U32(offsets) => unsafe {
                // SAFETY: The offsets cannot be empty.
                usize::try_from(*offsets.last().unwrap_unchecked()).unwrap()
            },
            Self::U64(offsets) => unsafe {
                // SAFETY: The offsets cannot be empty.
                usize::try_from(*offsets.last().unwrap_unchecked()).unwrap()
            },
        }
    }

    fn is_monotonically_increasing(&self) -> bool {
        match self {
            Self::U32(offsets) => offsets_u32_are_monotonically_increasing(offsets),
            Self::U64(offsets) => offsets_u64_are_monotonically_increasing(offsets),
        }
    }
}

impl<'a> TryFrom<Cow<'a, [u32]>> for ArrayBytesOffsets<'a> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: Cow<'a, [u32]>) -> Result<Self, Self::Error> {
        Self::U32(value).validate()
    }
}

impl<'a> IntoArrayBytesOffsetsUnchecked<'a> for ArrayBytesOffsets<'a> {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'a> {
        self
    }
}

impl<'a> IntoArrayBytesOffsetsUnchecked<'a> for Cow<'a, [u32]> {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'a> {
        ArrayBytesOffsets::U32(self)
    }
}

impl<'a> IntoArrayBytesOffsetsUnchecked<'a> for &'a [u32] {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'a> {
        ArrayBytesOffsets::U32(Cow::Borrowed(self))
    }
}

impl<'a, const N: usize> IntoArrayBytesOffsetsUnchecked<'a> for &'a [u32; N] {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'a> {
        ArrayBytesOffsets::U32(Cow::Borrowed(self))
    }
}

impl IntoArrayBytesOffsetsUnchecked<'static> for Vec<u32> {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'static> {
        ArrayBytesOffsets::U32(Cow::Owned(self))
    }
}

impl<'a> IntoArrayBytesOffsetsUnchecked<'a> for Cow<'a, [u64]> {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'a> {
        ArrayBytesOffsets::U64(self)
    }
}

impl<'a> IntoArrayBytesOffsetsUnchecked<'a> for &'a [u64] {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'a> {
        ArrayBytesOffsets::U64(Cow::Borrowed(self))
    }
}

impl<'a, const N: usize> IntoArrayBytesOffsetsUnchecked<'a> for &'a [u64; N] {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'a> {
        ArrayBytesOffsets::U64(Cow::Borrowed(self))
    }
}

impl IntoArrayBytesOffsetsUnchecked<'static> for Vec<u64> {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'static> {
        ArrayBytesOffsets::U64(Cow::Owned(self))
    }
}

impl IntoArrayBytesOffsetsUnchecked<'static> for Vec<usize> {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'static> {
        usize_offsets_to_array_bytes_offsets(&self)
    }
}

impl<'a> IntoArrayBytesOffsetsUnchecked<'static> for Cow<'a, [usize]> {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'static> {
        usize_offsets_to_array_bytes_offsets(&self)
    }
}

impl<'a> IntoArrayBytesOffsetsUnchecked<'static> for &'a [usize] {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'static> {
        usize_offsets_to_array_bytes_offsets(self)
    }
}

impl<'a, const N: usize> IntoArrayBytesOffsetsUnchecked<'static> for &'a [usize; N] {
    fn into_array_bytes_offsets_unchecked(self) -> ArrayBytesOffsets<'static> {
        usize_offsets_to_array_bytes_offsets(self)
    }
}

impl<'a> TryFrom<&'a [u32]> for ArrayBytesOffsets<'a> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: &'a [u32]) -> Result<Self, Self::Error> {
        Self::U32(Cow::Borrowed(value)).validate()
    }
}

impl<'a, const N: usize> TryFrom<&'a [u32; N]> for ArrayBytesOffsets<'a> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: &'a [u32; N]) -> Result<Self, Self::Error> {
        Self::U32(Cow::Borrowed(value)).validate()
    }
}

impl TryFrom<Vec<u32>> for ArrayBytesOffsets<'_> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: Vec<u32>) -> Result<Self, Self::Error> {
        Self::U32(Cow::Owned(value)).validate()
    }
}

impl<'a> TryFrom<Cow<'a, [u64]>> for ArrayBytesOffsets<'a> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: Cow<'a, [u64]>) -> Result<Self, Self::Error> {
        Self::U64(value).validate()
    }
}

impl<'a> TryFrom<&'a [u64]> for ArrayBytesOffsets<'a> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: &'a [u64]) -> Result<Self, Self::Error> {
        Self::U64(Cow::Borrowed(value)).validate()
    }
}

impl<'a, const N: usize> TryFrom<&'a [u64; N]> for ArrayBytesOffsets<'a> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: &'a [u64; N]) -> Result<Self, Self::Error> {
        Self::U64(Cow::Borrowed(value)).validate()
    }
}

impl TryFrom<Vec<u64>> for ArrayBytesOffsets<'_> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: Vec<u64>) -> Result<Self, Self::Error> {
        Self::U64(Cow::Owned(value)).validate()
    }
}

impl<'a> TryFrom<Cow<'a, [usize]>> for ArrayBytesOffsets<'static> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: Cow<'a, [usize]>) -> Result<Self, Self::Error> {
        usize_offsets_to_array_bytes_offsets(&value).validate()
    }
}

impl<'a> TryFrom<&'a [usize]> for ArrayBytesOffsets<'static> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: &'a [usize]) -> Result<Self, Self::Error> {
        usize_offsets_to_array_bytes_offsets(value).validate()
    }
}

impl<'a, const N: usize> TryFrom<&'a [usize; N]> for ArrayBytesOffsets<'static> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: &'a [usize; N]) -> Result<Self, Self::Error> {
        usize_offsets_to_array_bytes_offsets(value).validate()
    }
}

impl TryFrom<Vec<usize>> for ArrayBytesOffsets<'_> {
    type Error = ArrayBytesRawOffsetsCreateError;

    fn try_from(value: Vec<usize>) -> Result<Self, Self::Error> {
        usize_offsets_to_array_bytes_offsets(&value).validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_bytes_offsets() {
        let offsets = ArrayBytesOffsets::new(vec![0u32, 1, 2, 3]).unwrap();
        assert!(matches!(offsets, ArrayBytesOffsets::U32(_)));
        assert_eq!(offsets.iter().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert!(ArrayBytesOffsets::new(Vec::<u32>::new()).is_err());
        assert!(ArrayBytesOffsets::new(vec![0u32]).is_ok());
        assert!(ArrayBytesOffsets::new(vec![10u32]).is_ok()); // nonsense, but not invalid
        assert!(ArrayBytesOffsets::new(vec![0u32, 1, 1]).is_ok());
        assert!(ArrayBytesOffsets::new(vec![0u32, 1, 0]).is_err());
        assert!(ArrayBytesOffsets::new([0u32, 1, 2].as_slice()).is_ok());
        assert!(ArrayBytesOffsets::new([0u32, 1, 0].as_slice()).is_err());
        assert!(ArrayBytesOffsets::new(&[0u64, 1, 2]).is_ok());
        assert!(ArrayBytesOffsets::new(&[0u64, 1, 0]).is_err());
        assert!(ArrayBytesOffsets::new(Cow::<[u64]>::Owned(vec![0u64, 1, 0])).is_err());
        let offsets = ArrayBytesOffsets::new(&[0usize, 1, 2]).unwrap();
        #[cfg(target_pointer_width = "32")]
        assert!(matches!(offsets, ArrayBytesOffsets::U32(_)));
        #[cfg(target_pointer_width = "64")]
        assert!(matches!(offsets, ArrayBytesOffsets::U64(_)));
        assert!(ArrayBytesOffsets::new(&[0usize, 1, 0]).is_err());
    }
}
