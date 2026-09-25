use std::iter::FusedIterator;
use std::ops::Range;

use bytes::Bytes;
use derive_more::derive::Display;
use thiserror::Error;
use zarrs_storage::CowBytes;

mod private {
    pub trait Sealed {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
}

/// An [`ArrayBytesOffsets`] element type: [`u32`] or [`u64`].
///
/// This trait is sealed and cannot be implemented outside of this crate.
pub trait ArrayBytesOffsetsElement:
    private::Sealed + bytemuck::Pod + Ord + Into<u64> + Send + Sync + 'static
{
    #[doc(hidden)]
    const WIDTH_U32: bool;
}

impl ArrayBytesOffsetsElement for u32 {
    const WIDTH_U32: bool = true;
}

impl ArrayBytesOffsetsElement for u64 {
    const WIDTH_U32: bool = false;
}

/// Array element byte offsets.
///
/// Offsets are stored as either [`u32`] or [`u64`] native-endian elements in a [`CowBytes<'static>`] buffer.
/// Use [`as_slice`](ArrayBytesOffsets::as_slice) to access the offsets in their native width without conversion.
///
/// Cloning retains the offset allocation without copying its elements.
/// These must be monotonically increasing. See [`ArrayBytes`](crate::ArrayBytes).
///
/// # Examples
///
/// ```
/// use zarrs_codec::{ArrayBytesOffsets, ArrayBytesOffsetsSlice};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let offsets = ArrayBytesOffsets::new(vec![0u32, 2, 5])?;
/// assert_eq!(offsets.element_range(1), 2..5);
///
/// match offsets.as_slice() {
///     ArrayBytesOffsetsSlice::U32(values) => assert_eq!(values, &[0, 2, 5]),
///     ArrayBytesOffsetsSlice::U64(_) => unreachable!("constructed from u32 offsets"),
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct ArrayBytesOffsets {
    width: Width,
    /// Invariant: aligned for `width`, a non-zero multiple of the `width` size, monotonically increasing, and the last offset does not exceed [`usize::MAX`].
    bytes: CowBytes<'static>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Width {
    U32,
    U64,
}

impl Width {
    const fn of<T: ArrayBytesOffsetsElement>() -> Self {
        if T::WIDTH_U32 { Self::U32 } else { Self::U64 }
    }
}

/// Owns a vector of offsets so it can back a [`Bytes`] without copying.
struct VecOwner<T>(Vec<T>);

impl<T: bytemuck::NoUninit> AsRef<[u8]> for VecOwner<T> {
    fn as_ref(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

fn vec_into_bytes<T: ArrayBytesOffsetsElement>(offsets: Vec<T>) -> CowBytes<'static> {
    CowBytes::Shared(Bytes::from_owner(VecOwner(offsets)))
}

/// A borrowed view of [`ArrayBytesOffsets`] in their native element width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrayBytesOffsetsSlice<'a> {
    /// [`u32`] offsets.
    U32(&'a [u32]),
    /// [`u64`] offsets.
    U64(&'a [u64]),
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
    /// An offset exceeds [`usize::MAX`].
    #[display("an offset exceeds usize::MAX")]
    ExceedsUsizeMax,
    /// The length of the offsets bytes is not a multiple of the offset element size.
    #[display("offsets bytes length {_0} is not a multiple of the offset element size {_1}")]
    InvalidBytesLength(usize, usize),
}

fn validate<T: ArrayBytesOffsetsElement>(
    offsets: &[T],
) -> Result<(), ArrayBytesOffsetsCreateError> {
    let Some(&last) = offsets.last() else {
        return Err(ArrayBytesOffsetsCreateError::ZeroLength);
    };
    if !offsets.windows(2).all(|w| w[1] >= w[0]) {
        Err(ArrayBytesOffsetsCreateError::NotMonotonicallyIncreasing)
    } else if usize::try_from(last.into()).is_err() {
        Err(ArrayBytesOffsetsCreateError::ExceedsUsizeMax)
    } else {
        Ok(())
    }
}

impl std::fmt::Debug for ArrayBytesOffsets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("ArrayBytesOffsets")
            .field(&self.as_slice())
            .finish()
    }
}

impl ArrayBytesOffsets {
    /// Creates a new `ArrayBytesOffsets` from [`u32`] or [`u64`] offsets.
    ///
    /// The vector is retained without copying.
    ///
    /// # Errors
    /// Returns an error if the offsets are empty, not monotonically increasing, or exceed [`usize::MAX`].
    pub fn new<T: ArrayBytesOffsetsElement>(
        offsets: impl Into<Vec<T>>,
    ) -> Result<Self, ArrayBytesOffsetsCreateError> {
        let offsets = offsets.into();
        validate(&offsets)?;
        Ok(Self {
            width: Width::of::<T>(),
            bytes: vec_into_bytes(offsets),
        })
    }

    /// Creates a new `ArrayBytesOffsets` without checking the offsets.
    ///
    /// # Safety
    /// The offsets must be non-empty, monotonically increasing, and must not exceed [`usize::MAX`].
    #[must_use]
    pub unsafe fn new_unchecked<T: ArrayBytesOffsetsElement>(offsets: impl Into<Vec<T>>) -> Self {
        let offsets = offsets.into();
        debug_assert!(validate(&offsets).is_ok());
        Self {
            width: Width::of::<T>(),
            bytes: vec_into_bytes(offsets),
        }
    }

    /// Creates a new `ArrayBytesOffsets` from native-endian [`u32`] or [`u64`] offset bytes.
    ///
    /// The bytes are retained without copying if they are suitably aligned for `T`, otherwise they are copied.
    ///
    /// # Errors
    /// Returns an error if the length of `bytes` is not a multiple of the size of `T`, or if the offsets are empty, not monotonically increasing, or exceed [`usize::MAX`].
    pub fn from_ne_bytes<T: ArrayBytesOffsetsElement>(
        bytes: impl Into<CowBytes<'static>>,
    ) -> Result<Self, ArrayBytesOffsetsCreateError> {
        let bytes = bytes.into();
        if !bytes.len().is_multiple_of(size_of::<T>()) {
            return Err(ArrayBytesOffsetsCreateError::InvalidBytesLength(
                bytes.len(),
                size_of::<T>(),
            ));
        }
        let bytes = if bytemuck::try_cast_slice::<u8, T>(&bytes).is_ok() {
            bytes
        } else {
            vec_into_bytes(bytemuck::allocation::pod_collect_to_vec::<u8, T>(&bytes))
        };
        validate::<T>(bytemuck::cast_slice(&bytes))?;
        Ok(Self {
            width: Width::of::<T>(),
            bytes,
        })
    }

    /// Returns a view of the offsets in their native element width.
    #[must_use]
    pub fn as_slice(&self) -> ArrayBytesOffsetsSlice<'_> {
        match self.width {
            Width::U32 => ArrayBytesOffsetsSlice::U32(bytemuck::cast_slice(&self.bytes)),
            Width::U64 => ArrayBytesOffsetsSlice::U64(bytemuck::cast_slice(&self.bytes)),
        }
    }

    /// Returns the offsets if they are stored as [`u32`].
    #[must_use]
    pub fn as_u32(&self) -> Option<&[u32]> {
        match self.as_slice() {
            ArrayBytesOffsetsSlice::U32(offsets) => Some(offsets),
            ArrayBytesOffsetsSlice::U64(_) => None,
        }
    }

    /// Returns the offsets if they are stored as [`u64`].
    #[must_use]
    pub fn as_u64(&self) -> Option<&[u64]> {
        match self.as_slice() {
            ArrayBytesOffsetsSlice::U32(_) => None,
            ArrayBytesOffsetsSlice::U64(offsets) => Some(offsets),
        }
    }

    /// Returns the native-endian bytes of the offsets in their native element width.
    #[must_use]
    pub fn as_ne_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Returns true if the offsets are stored as [`u32`].
    #[must_use]
    pub fn is_u32(&self) -> bool {
        self.width == Width::U32
    }

    /// Returns true if the offsets are stored as [`u64`].
    #[must_use]
    pub fn is_u64(&self) -> bool {
        self.width == Width::U64
    }

    /// Convert the offsets into native-endian [`u32`] bytes.
    ///
    /// This does not copy if the offsets are stored as [`u32`].
    ///
    /// # Errors
    /// Returns `self` if an offset exceeds [`u32::MAX`].
    pub fn into_u32_ne_bytes(self) -> Result<CowBytes<'static>, Self> {
        match self.width {
            Width::U32 => Ok(self.bytes),
            Width::U64 => self.into_u32_vec().map(vec_into_bytes),
        }
    }

    /// Convert the offsets into native-endian [`u64`] bytes.
    ///
    /// This does not copy if the offsets are stored as [`u64`].
    #[must_use]
    pub fn into_u64_ne_bytes(self) -> CowBytes<'static> {
        match self.width {
            Width::U32 => vec_into_bytes(self.into_u64_vec()),
            Width::U64 => self.bytes,
        }
    }

    /// Convert the offsets into a [`u32`] vector. This always copies.
    ///
    /// # Errors
    /// Returns `self` if an offset exceeds [`u32::MAX`].
    pub fn into_u32_vec(self) -> Result<Vec<u32>, Self> {
        if u32::try_from(self.last()).is_err() {
            return Err(self);
        }
        match self.as_slice() {
            ArrayBytesOffsetsSlice::U32(offsets) => Ok(offsets.to_vec()),
            #[allow(clippy::cast_possible_truncation)]
            ArrayBytesOffsetsSlice::U64(offsets) => {
                Ok(offsets.iter().map(|&offset| offset as u32).collect())
            }
        }
    }

    /// Convert the offsets into a [`u64`] vector. This always copies.
    #[must_use]
    pub fn into_u64_vec(self) -> Vec<u64> {
        match self.as_slice() {
            ArrayBytesOffsetsSlice::U32(offsets) => {
                offsets.iter().copied().map(u64::from).collect()
            }
            ArrayBytesOffsetsSlice::U64(offsets) => offsets.to_vec(),
        }
    }

    /// Returns the number of offsets.
    ///
    /// This is one more than the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }

    /// Returns true if there are no offsets. This is always false.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Returns the number of elements (the number of offsets minus one).
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.len() - 1
    }

    /// Returns the offset at `index`, or [`None`] if out of bounds.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<usize> {
        self.as_slice().get(index)
    }

    /// Returns the last offset.
    #[must_use]
    pub fn last(&self) -> usize {
        self.as_slice()
            .get(self.len() - 1)
            .expect("offsets are non-empty")
    }

    /// Returns the byte range of the element at `index`.
    ///
    /// # Panics
    /// Panics if `index + 1` is out of bounds of the offsets.
    #[must_use]
    pub fn element_range(&self, index: usize) -> Range<usize> {
        self.as_slice().element_range(index)
    }

    /// Returns an iterator over the offsets.
    #[must_use]
    pub fn iter(&self) -> ArrayBytesOffsetsIter<'_> {
        self.as_slice().iter()
    }

    /// Returns an iterator over the byte ranges of each element.
    #[must_use]
    pub fn element_ranges(&self) -> ArrayBytesOffsetsRangesIter<'_> {
        self.as_slice().element_ranges()
    }
}

impl PartialEq for ArrayBytesOffsets {
    fn eq(&self, other: &Self) -> bool {
        match (self.as_slice(), other.as_slice()) {
            (ArrayBytesOffsetsSlice::U32(a), ArrayBytesOffsetsSlice::U32(b)) => a == b,
            (ArrayBytesOffsetsSlice::U64(a), ArrayBytesOffsetsSlice::U64(b)) => a == b,
            (ArrayBytesOffsetsSlice::U32(a), ArrayBytesOffsetsSlice::U64(b))
            | (ArrayBytesOffsetsSlice::U64(b), ArrayBytesOffsetsSlice::U32(a)) => {
                a.len() == b.len() && a.iter().zip(b).all(|(&a, &b)| u64::from(a) == b)
            }
        }
    }
}

impl Eq for ArrayBytesOffsets {}

impl<'a> ArrayBytesOffsetsSlice<'a> {
    /// Returns the number of offsets.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U32(offsets) => offsets.len(),
            Self::U64(offsets) => offsets.len(),
        }
    }

    /// Returns true if there are no offsets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the offset at `index`, or [`None`] if out of bounds.
    ///
    /// # Panics
    /// Panics if the offset exceeds [`usize::MAX`].
    #[must_use]
    pub fn get(&self, index: usize) -> Option<usize> {
        match self {
            Self::U32(offsets) => offsets
                .get(index)
                .map(|&offset| usize::try_from(offset).unwrap()),
            Self::U64(offsets) => offsets
                .get(index)
                .map(|&offset| usize::try_from(offset).unwrap()),
        }
    }

    /// Returns the byte range of the element at `index`.
    ///
    /// # Panics
    /// Panics if `index + 1` is out of bounds of the offsets or an offset exceeds [`usize::MAX`].
    #[must_use]
    pub fn element_range(&self, index: usize) -> Range<usize> {
        match self {
            Self::U32(offsets) => {
                usize::try_from(offsets[index]).unwrap()
                    ..usize::try_from(offsets[index + 1]).unwrap()
            }
            Self::U64(offsets) => {
                usize::try_from(offsets[index]).unwrap()
                    ..usize::try_from(offsets[index + 1]).unwrap()
            }
        }
    }

    /// Returns an iterator over the offsets.
    #[must_use]
    pub fn iter(&self) -> ArrayBytesOffsetsIter<'a> {
        ArrayBytesOffsetsIter {
            offsets: *self,
            front: 0,
            back: self.len(),
        }
    }

    /// Returns an iterator over the byte ranges of each element.
    #[must_use]
    pub fn element_ranges(&self) -> ArrayBytesOffsetsRangesIter<'a> {
        ArrayBytesOffsetsRangesIter {
            offsets: *self,
            front: 0,
            back: self.len().saturating_sub(1),
        }
    }
}

/// An iterator over [`ArrayBytesOffsets`].
#[derive(Clone, Debug)]
pub struct ArrayBytesOffsetsIter<'a> {
    offsets: ArrayBytesOffsetsSlice<'a>,
    front: usize,
    back: usize,
}

impl Iterator for ArrayBytesOffsetsIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let offset = self.offsets.get(self.front);
            self.front += 1;
            offset
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}

impl DoubleEndedIterator for ArrayBytesOffsetsIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;
            self.offsets.get(self.back)
        } else {
            None
        }
    }
}

impl ExactSizeIterator for ArrayBytesOffsetsIter<'_> {}

impl FusedIterator for ArrayBytesOffsetsIter<'_> {}

/// An iterator over the element byte ranges of [`ArrayBytesOffsets`].
#[derive(Clone, Debug)]
pub struct ArrayBytesOffsetsRangesIter<'a> {
    offsets: ArrayBytesOffsetsSlice<'a>,
    front: usize,
    back: usize,
}

impl Iterator for ArrayBytesOffsetsRangesIter<'_> {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            let range = self.offsets.element_range(self.front);
            self.front += 1;
            Some(range)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.back - self.front;
        (len, Some(len))
    }
}

impl DoubleEndedIterator for ArrayBytesOffsetsRangesIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front < self.back {
            self.back -= 1;
            Some(self.offsets.element_range(self.back))
        } else {
            None
        }
    }
}

impl ExactSizeIterator for ArrayBytesOffsetsRangesIter<'_> {}

impl FusedIterator for ArrayBytesOffsetsRangesIter<'_> {}

impl<T: ArrayBytesOffsetsElement> TryFrom<&[T]> for ArrayBytesOffsets {
    type Error = ArrayBytesOffsetsCreateError;

    fn try_from(value: &[T]) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl<T: ArrayBytesOffsetsElement, const N: usize> TryFrom<&[T; N]> for ArrayBytesOffsets {
    type Error = ArrayBytesOffsetsCreateError;

    fn try_from(value: &[T; N]) -> Result<Self, Self::Error> {
        Self::new(value.as_slice())
    }
}

impl<T: ArrayBytesOffsetsElement> TryFrom<Vec<T>> for ArrayBytesOffsets {
    type Error = ArrayBytesOffsetsCreateError;

    fn try_from(value: Vec<T>) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_bytes_offsets() {
        let owned = vec![0u32, 2, 4];
        let pointer = owned.as_ptr();
        let shared = ArrayBytesOffsets::new(owned).unwrap();
        assert_eq!(shared.as_u32().unwrap().as_ptr(), pointer);
        assert_eq!(shared.clone().as_u32().unwrap().as_ptr(), pointer);
        assert_eq!(
            shared.into_u32_ne_bytes().unwrap().as_ptr(),
            pointer.cast::<u8>()
        );

        let offsets = ArrayBytesOffsets::new(vec![0u64, 1, 2, 3]).unwrap();
        assert_eq!(
            offsets.as_slice(),
            ArrayBytesOffsetsSlice::U64(&[0, 1, 2, 3])
        );
        assert_eq!(offsets.iter().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        assert_eq!(offsets.len(), 4);
        assert_eq!(offsets.num_elements(), 3);
        assert_eq!(offsets.last(), 3);
        assert_eq!(offsets.get(1), Some(1));
        assert_eq!(offsets.get(4), None);
        assert_eq!(offsets.element_range(1), 1..2);
        assert!(ArrayBytesOffsets::new(Vec::<u32>::new()).is_err());
        assert!(ArrayBytesOffsets::new(vec![0u32]).is_ok());
        assert!(ArrayBytesOffsets::new(vec![10u32]).is_ok()); // nonsense, but not invalid
        assert!(ArrayBytesOffsets::new(vec![0u32, 1, 1]).is_ok());
        assert!(ArrayBytesOffsets::new(vec![0u32, 1, 0]).is_err());
        assert!(ArrayBytesOffsets::new(vec![0u64, 1, 0]).is_err());
        assert!(ArrayBytesOffsets::try_from(vec![0u32, 1, 2]).is_ok());
        assert!(ArrayBytesOffsets::try_from(vec![0u32, 1, 0]).is_err());
        assert!(ArrayBytesOffsets::try_from([0u32, 1, 2].as_slice()).is_ok());
        assert!(ArrayBytesOffsets::try_from([0u32, 1, 0].as_slice()).is_err());
        assert!(ArrayBytesOffsets::try_from(&[0u64, 1, 2]).is_ok());
        assert!(ArrayBytesOffsets::try_from(&[0u64, 1, 0]).is_err());
    }

    #[test]
    fn array_bytes_offsets_cross_width() {
        let offsets_u32 = ArrayBytesOffsets::new(vec![0u32, 2, 5]).unwrap();
        let offsets_u64 = ArrayBytesOffsets::new(vec![0u64, 2, 5]).unwrap();
        assert!(offsets_u32.is_u32());
        assert!(offsets_u64.is_u64());
        assert_eq!(offsets_u32, offsets_u64);
        assert_eq!(offsets_u64, offsets_u32);
        assert_ne!(
            offsets_u32,
            ArrayBytesOffsets::new(vec![0u64, 2, 6]).unwrap()
        );
        assert_ne!(offsets_u32, ArrayBytesOffsets::new(vec![0u64, 2]).unwrap());

        assert_eq!(offsets_u32.clone().into_u64_vec(), vec![0, 2, 5]);
        assert_eq!(offsets_u64.clone().into_u32_vec().unwrap(), vec![0, 2, 5]);
        #[cfg(target_pointer_width = "64")]
        {
            let offsets_large =
                ArrayBytesOffsets::new(vec![0u64, u64::from(u32::MAX) + 1]).unwrap();
            assert!(offsets_large.into_u32_vec().is_err());
        }

        assert_eq!(
            offsets_u32.element_ranges().collect::<Vec<_>>(),
            vec![0..2, 2..5]
        );
        assert_eq!(
            offsets_u64.element_ranges().rev().collect::<Vec<_>>(),
            vec![2..5, 0..2]
        );
        assert_eq!(offsets_u32.element_ranges().len(), 2);
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn array_bytes_offsets_exceeds_usize_max() {
        assert!(matches!(
            ArrayBytesOffsets::new(vec![0u64, u64::from(u32::MAX) + 1]),
            Err(ArrayBytesOffsetsCreateError::ExceedsUsizeMax)
        ));
    }

    #[test]
    fn array_bytes_offsets_from_ne_bytes() {
        // Aligned shared bytes are retained without copying
        let bytes = CowBytes::Shared(Bytes::from_owner(VecOwner(vec![0u64, 3, 7])));
        let pointer = bytes.as_ptr();
        let offsets = ArrayBytesOffsets::from_ne_bytes::<u64>(bytes).unwrap();
        assert_eq!(offsets.as_ne_bytes().as_ptr(), pointer);
        assert_eq!(offsets.as_u64().unwrap(), &[0, 3, 7]);
        assert_eq!(offsets.clone().into_u64_ne_bytes().as_ptr(), pointer);

        // Misaligned bytes are copied
        let mut raw = vec![0u8];
        for offset in [0u32, 1, 2] {
            raw.extend_from_slice(&offset.to_ne_bytes());
        }
        let misaligned = Bytes::from(raw).slice(1..);
        let pointer = misaligned.as_ptr();
        let offsets = ArrayBytesOffsets::from_ne_bytes::<u32>(misaligned).unwrap();
        assert_eq!(offsets.as_u32().unwrap(), &[0, 1, 2]);
        if !pointer.cast::<u32>().is_aligned() {
            assert_ne!(offsets.as_ne_bytes().as_ptr(), pointer);
        }

        // Invalid lengths and offsets
        assert!(matches!(
            ArrayBytesOffsets::from_ne_bytes::<u32>(Bytes::from(vec![0u8; 6])),
            Err(ArrayBytesOffsetsCreateError::InvalidBytesLength(6, 4))
        ));
        assert!(matches!(
            ArrayBytesOffsets::from_ne_bytes::<u32>(Bytes::new()),
            Err(ArrayBytesOffsetsCreateError::ZeroLength)
        ));
        let decreasing: Vec<u8> = [2u32, 1].iter().flat_map(|o| o.to_ne_bytes()).collect();
        assert!(matches!(
            ArrayBytesOffsets::from_ne_bytes::<u32>(Bytes::from(decreasing)),
            Err(ArrayBytesOffsetsCreateError::NotMonotonicallyIncreasing)
        ));
    }
}
