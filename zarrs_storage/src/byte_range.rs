//! Byte ranges.
//!
//! A [`ByteRange`] represents a byte range relative to the start or end of a byte sequence.
//! A byte range has an offset and optional length, which if omitted means to read all remaining bytes.
//!
//! [`extract_byte_ranges`] is a convenience function for extracting byte ranges from a slice of bytes.

use std::io::{Read, Seek, SeekFrom};
use std::ops::{
    Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

use thiserror::Error;
use unsafe_cell_slice::UnsafeCellSlice;

use crate::MaybeSend;

/// A byte offset.
pub type ByteOffset = u64;

/// A byte length.
pub type ByteLength = u64;

/// A byte range.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum ByteRange {
    /// A byte range from the start.
    ///
    /// If the byte length is [`None`], reads to the end of the value.
    FromStart(ByteOffset, Option<ByteLength>),
    /// A suffix byte range.
    Suffix(ByteLength),
}

impl Ord for ByteRange {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Self::FromStart(offset1, length1), Self::FromStart(offset2, length2)) => {
                offset1.cmp(offset2).then_with(|| length1.cmp(length2))
            }
            (Self::FromStart(_, _), Self::Suffix(_)) => std::cmp::Ordering::Less,
            (Self::Suffix(_), Self::FromStart(_, _)) => std::cmp::Ordering::Greater,
            (Self::Suffix(length1), Self::Suffix(length2)) => length1.cmp(length2),
        }
    }
}

impl PartialOrd for ByteRange {
    fn partial_cmp(&self, other: &ByteRange) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

macro_rules! impl_from_rangebounds {
    ($($t:ty),*) => {
        $(
            impl From<$t> for ByteRange {
                fn from(range: $t) -> Self {
                    Self::new(range)
                }
            }
        )*
    };
}

impl_from_rangebounds!(
    Range<u64>,
    RangeFrom<u64>,
    RangeFull,
    RangeTo<u64>,
    RangeInclusive<u64>,
    RangeToInclusive<u64>
);

impl ByteRange {
    /// Create a new byte range from a [`RangeBounds<u64>`].
    pub fn new(bounds: impl RangeBounds<u64>) -> Self {
        match (bounds.start_bound(), bounds.end_bound()) {
            (Bound::Included(start), Bound::Included(end)) => {
                Self::FromStart(*start, Some(end - start + 1))
            }
            (Bound::Included(start), Bound::Excluded(end)) => {
                Self::FromStart(*start, Some(end - start))
            }
            (Bound::Included(start), Bound::Unbounded) => Self::FromStart(*start, None),
            (Bound::Excluded(start), Bound::Included(end)) => {
                Self::FromStart(start + 1, Some(end - start))
            }
            (Bound::Excluded(start), Bound::Excluded(end)) => {
                Self::FromStart(start + 1, Some(end - start - 1))
            }
            (Bound::Excluded(start), Bound::Unbounded) => Self::FromStart(start + 1, None),
            (Bound::Unbounded, Bound::Included(length)) => Self::FromStart(0, Some(length + 1)),
            (Bound::Unbounded, Bound::Excluded(length)) => Self::FromStart(0, Some(*length)),
            // (Bound::Unbounded, Bound::Included(length)) => Self::Suffix(length + 1), // opendal style
            // (Bound::Unbounded, Bound::Excluded(length)) => Self::Suffix(*length), // opendal style
            (Bound::Unbounded, Bound::Unbounded) => Self::FromStart(0, None),
        }
    }

    /// Return the start of a byte range. `size` is the size of the entire bytes.
    #[must_use]
    pub fn start(&self, size: u64) -> u64 {
        match self {
            Self::FromStart(offset, _) => *offset,
            Self::Suffix(length) => size - *length,
        }
    }

    /// Return the exclusive end of a byte range. `size` is the size of the entire bytes.
    #[must_use]
    pub fn end(&self, size: u64) -> u64 {
        match self {
            Self::FromStart(offset, length) => {
                length.as_ref().map_or(size, |length| offset + length)
            }
            Self::Suffix(_) => size,
        }
    }

    /// Return the length of a byte range. `size` is the size of the entire bytes.
    #[must_use]
    pub fn length(&self, size: u64) -> u64 {
        match self {
            Self::FromStart(offset, None) => size - offset,
            Self::FromStart(_, Some(length)) | Self::Suffix(length) => *length,
        }
    }

    /// Convert the byte range to a [`Range<u64>`].
    #[must_use]
    pub fn to_range(&self, size: u64) -> Range<u64> {
        self.start(size)..self.end(size)
    }

    /// Convert the byte range to a [`Range<usize>`].
    ///
    /// # Panics
    ///
    /// Panics if the byte range exceeds [`usize::MAX`].
    #[must_use]
    pub fn to_range_usize(&self, size: u64) -> core::ops::Range<usize> {
        self.start(size).try_into().unwrap()..self.end(size).try_into().unwrap()
    }
}

impl std::fmt::Display for ByteRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        match self {
            Self::FromStart(offset, length) => write!(
                f,
                "{}..{}",
                if offset == &0 {
                    String::new()
                } else {
                    offset.to_string()
                },
                length.map_or(String::new(), |length| (offset + length).to_string())
            ),
            Self::Suffix(length) => write!(f, "-{length}.."),
        }
    }
}

/// An invalid byte range error.
#[derive(Copy, Clone, Debug, Error)]
#[error("invalid byte range {0} for bytes of length {1}")]
pub struct InvalidByteRangeError(ByteRange, u64);

impl InvalidByteRangeError {
    /// Create a new [`InvalidByteRangeError`].
    #[must_use]
    pub fn new(byte_range: ByteRange, bytes_len: u64) -> Self {
        Self(byte_range, bytes_len)
    }
}

fn is_valid(byte_range: ByteRange, bytes_len: u64) -> bool {
    match byte_range {
        ByteRange::FromStart(offset, length) => offset + length.unwrap_or(0) <= bytes_len,
        ByteRange::Suffix(length) => length <= bytes_len,
    }
}

/// Extract byte ranges from bytes.
///
/// # Errors
/// Returns [`InvalidByteRangeError`] if any bytes are requested beyond the end of `bytes`.
///
/// # Panics
/// Panics if requesting bytes beyond [`usize::MAX`].
pub fn extract_byte_ranges<R: Into<ByteRange>>(
    bytes: &[u8],
    byte_ranges: impl Iterator<Item = R>,
) -> Result<Vec<Vec<u8>>, InvalidByteRangeError> {
    let bytes_len = bytes.len() as u64;
    byte_ranges
        .map(|byte_range| {
            let byte_range: ByteRange = byte_range.into();
            let valid = is_valid(byte_range, bytes_len);
            if !valid {
                return Err(InvalidByteRangeError(byte_range, bytes_len));
            }
            let start = usize::try_from(byte_range.start(bytes.len() as u64)).unwrap();
            let end = usize::try_from(byte_range.end(bytes.len() as u64)).unwrap();
            Ok(bytes[start..end].to_vec())
        })
        .collect::<Result<Vec<Vec<u8>>, InvalidByteRangeError>>()
}

/// Extract byte ranges from bytes and concatenate.
///
/// # Errors
/// Returns [`InvalidByteRangeError`] if any bytes are requested beyond the end of `bytes`.
///
/// # Panics
/// Panics if requesting bytes beyond [`usize::MAX`].
pub fn extract_byte_ranges_concat<R: Into<ByteRange>>(
    bytes: &[u8],
    byte_ranges: impl Iterator<Item = R>,
) -> Result<Vec<u8>, InvalidByteRangeError> {
    let bytes_len = bytes.len() as u64;
    let lengths_and_starts = byte_ranges
        .map(|byte_range| {
            let byte_range: ByteRange = byte_range.into();
            let valid = is_valid(byte_range, bytes_len);
            if !valid {
                return Err(InvalidByteRangeError(byte_range, bytes_len));
            }
            Ok((byte_range.length(bytes_len), byte_range.start(bytes_len)))
        })
        .collect::<Result<Vec<(u64, u64)>, InvalidByteRangeError>>()?;
    let out_size = usize::try_from(
        lengths_and_starts
            .iter()
            .map(|(length, _)| length)
            .sum::<u64>(),
    )
    .unwrap();
    if out_size == 0 {
        return Ok(vec![]);
    }
    let mut out = Vec::with_capacity(out_size);
    let out_slice = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut out);
    let mut offset: usize = 0;
    for (length, start) in &lengths_and_starts {
        let start = usize::try_from(*start).unwrap();
        let byte_range_len = usize::try_from(*length).unwrap();
        // SAFETY: the slices are non-overlapping
        unsafe {
            out_slice
                .index_mut(offset..offset + byte_range_len)
                .copy_from_slice(&bytes[start..start + byte_range_len]);
        }
        offset += byte_range_len;
    }
    // SAFETY: each element is initialised
    unsafe {
        out.set_len(out_size);
    }
    Ok(out)
}

/// Extract byte ranges from bytes implementing [`Read`] and [`Seek`].
///
/// # Errors
///
/// Returns a [`std::io::Error`] if there is an error reading or seeking from `bytes`.
/// This can occur if the byte range is out-of-bounds of the `bytes`.
///
/// # Panics
///
/// Panics if a byte has length exceeding [`usize::MAX`].
pub fn extract_byte_ranges_read_seek<T: Read + Seek>(
    mut bytes: T,
    byte_ranges: impl Iterator<Item = ByteRange>,
) -> std::io::Result<Vec<Vec<u8>>> {
    let len: u64 = bytes.seek(SeekFrom::End(0))?;
    byte_ranges
        .map(|byte_range| {
            let data: Vec<u8> = match byte_range {
                ByteRange::FromStart(offset, None) => {
                    bytes.seek(SeekFrom::Start(offset))?;
                    let length = usize::try_from(len).unwrap();
                    let mut data = vec![0; length];
                    bytes.read_exact(&mut data)?;
                    data
                }
                ByteRange::FromStart(offset, Some(length)) => {
                    bytes.seek(SeekFrom::Start(offset))?;
                    let length = usize::try_from(length).unwrap();
                    let mut data = vec![0; length];
                    bytes.read_exact(&mut data)?;
                    data
                }
                ByteRange::Suffix(length) => {
                    bytes.seek(SeekFrom::End(-i64::try_from(length).unwrap()))?;
                    let length = usize::try_from(length).unwrap();
                    let mut data = vec![0; length];
                    bytes.read_exact(&mut data)?;
                    data
                }
            };
            Ok(data)
        })
        .collect::<std::io::Result<Vec<Vec<u8>>>>()
}

/// This trait combines [`Iterator<Item = ByteRange>`] and [`MaybeSend`],
/// as they cannot be combined together directly in function signatures.
pub trait MaybeSendByteRangeIterator: Iterator<Item = ByteRange> + MaybeSend {}

impl<T> MaybeSendByteRangeIterator for T where T: Iterator<Item = ByteRange> + MaybeSend {}

/// A [`ByteRange`] iterator.
pub type ByteRangeIterator<'a> = Box<dyn MaybeSendByteRangeIterator + 'a>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_ranges() {
        let byte_range = ByteRange::FromStart(1, None);
        assert_eq!(byte_range.to_range(10), 1..10);
        assert_eq!(byte_range.length(10), 9);

        let byte_range = ByteRange::Suffix(1);
        assert_eq!(byte_range.to_range(10), 9..10);
        assert_eq!(byte_range.length(10), 1);

        let byte_range = ByteRange::FromStart(1, Some(5));
        assert_eq!(byte_range.to_range(10), 1..6);
        assert_eq!(byte_range.to_range_usize(10), 1..6);
        assert_eq!(byte_range.length(10), 5);

        assert!(is_valid(ByteRange::FromStart(1, Some(5)), 6));
        assert!(!is_valid(ByteRange::FromStart(1, Some(5)), 2));

        assert!(is_valid(ByteRange::Suffix(5), 6));
        assert!(!is_valid(ByteRange::Suffix(5), 2));

        assert!(extract_byte_ranges(
            &[1, 2, 3],
            Box::new(vec![ByteRange::FromStart(1, Some(2))].into_iter())
        )
        .is_ok());
        let bytes = extract_byte_ranges(
            &[1, 2, 3],
            Box::new(vec![ByteRange::FromStart(1, Some(4))].into_iter()),
        );
        assert!(bytes.is_err());
        assert_eq!(
            bytes.unwrap_err().to_string(),
            "invalid byte range 1..5 for bytes of length 3"
        );
    }

    #[test]
    fn byte_range_rangebounds() {
        assert_eq!(ByteRange::FromStart(0, None), ByteRange::from(..));
        assert_eq!(ByteRange::FromStart(1, None), ByteRange::from(1..));
        assert_eq!(ByteRange::FromStart(0, Some(2)), ByteRange::from(0..2));
        assert_eq!(ByteRange::FromStart(1, Some(2)), ByteRange::from(1..3));
        assert_eq!(ByteRange::FromStart(0, Some(3)), ByteRange::from(..3));
        // assert_eq!(ByteRange::Suffix(3), ByteRange::from(..3)); // opendal style
    }

    #[test]
    fn byte_range_display() {
        assert_eq!(format!("{}", ByteRange::FromStart(0, None)), "..");
        assert_eq!(format!("{}", ByteRange::FromStart(5, None)), "5..");
        assert_eq!(format!("{}", ByteRange::FromStart(5, Some(2))), "5..7");
        assert_eq!(format!("{}", ByteRange::Suffix(2)), "-2..");
    }

    #[test]
    fn test_extract_byte_ranges_read_seek() {
        let data: Vec<u8> = (0..10).collect();
        let mut read = std::io::Cursor::new(data);
        let byte_ranges = vec![
            ByteRange::FromStart(3, Some(3)),
            ByteRange::FromStart(4, Some(1)),
            ByteRange::FromStart(1, Some(1)),
            ByteRange::Suffix(5),
        ];
        let out = extract_byte_ranges_read_seek(&mut read, &mut byte_ranges.into_iter()).unwrap();
        assert_eq!(
            out,
            vec![vec![3, 4, 5], vec![4], vec![1], vec![5, 6, 7, 8, 9]]
        );
    }
}
