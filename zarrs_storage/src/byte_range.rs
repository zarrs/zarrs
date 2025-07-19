//! Byte ranges.
//!
//! A [`ByteRange`] represents a byte range relative to the start or end of a byte sequence.
//! A byte range has an offset and optional length, which if omitted means to read all remaining bytes.
//!
//! [`extract_byte_ranges`] is a convenience function for extracting byte ranges from a slice of bytes.

use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Seek, SeekFrom},
    ops::{
        Bound, Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
    },
};

use itertools::Itertools;
use thiserror::Error;
use unsafe_cell_slice::UnsafeCellSlice;

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

fn validate_byte_ranges(
    byte_ranges: &[ByteRange],
    bytes_len: u64,
) -> Result<(), InvalidByteRangeError> {
    for byte_range in byte_ranges {
        let valid = match byte_range {
            ByteRange::FromStart(offset, length) => offset + length.unwrap_or(0) <= bytes_len,
            ByteRange::Suffix(length) => *length <= bytes_len,
        };
        if !valid {
            return Err(InvalidByteRangeError(*byte_range, bytes_len));
        }
    }
    Ok(())
}

/// Extract byte ranges from bytes.
///
/// # Errors
/// Returns [`InvalidByteRangeError`] if any bytes are requested beyond the end of `bytes`.
pub fn extract_byte_ranges(
    bytes: &[u8],
    byte_ranges: &[ByteRange],
) -> Result<Vec<Vec<u8>>, InvalidByteRangeError> {
    validate_byte_ranges(byte_ranges, bytes.len() as u64)?;
    Ok(unsafe { extract_byte_ranges_unchecked(bytes, byte_ranges) })
}

/// Extract byte ranges from bytes.
///
/// # Safety
/// All byte ranges in `byte_ranges` must specify a range within `bytes`.
///
/// # Panics
/// Panics if attempting to reference a byte beyond `usize::MAX`.
#[must_use]
pub unsafe fn extract_byte_ranges_unchecked(
    bytes: &[u8],
    byte_ranges: &[ByteRange],
) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(byte_ranges.len());
    for byte_range in byte_ranges {
        out.push({
            let start = usize::try_from(byte_range.start(bytes.len() as u64)).unwrap();
            let end = usize::try_from(byte_range.end(bytes.len() as u64)).unwrap();
            bytes[start..end].to_vec()
        });
    }
    out
}

/// Extract byte ranges from bytes and concatenate.
///
/// # Errors
/// Returns [`InvalidByteRangeError`] if any bytes are requested beyond the end of `bytes`.
pub fn extract_byte_ranges_concat(
    bytes: &[u8],
    byte_ranges: &[ByteRange],
) -> Result<Vec<u8>, InvalidByteRangeError> {
    validate_byte_ranges(byte_ranges, bytes.len() as u64)?;
    Ok(unsafe { extract_byte_ranges_concat_unchecked(bytes, byte_ranges) })
}

/// Extract byte ranges from bytes and concatenate.
///
/// # Safety
/// All byte ranges in `byte_ranges` must specify a range within `bytes`.
///
/// # Panics
/// Panics if attempting to reference a byte beyond `usize::MAX`.
#[must_use]
pub unsafe fn extract_byte_ranges_concat_unchecked(
    bytes: &[u8],
    byte_ranges: &[ByteRange],
) -> Vec<u8> {
    let out_size = usize::try_from(
        byte_ranges
            .iter()
            .map(|byte_range| byte_range.length(bytes.len() as u64))
            .sum::<u64>(),
    )
    .unwrap();
    if out_size == 0 {
        return vec![];
    }
    let mut out = Vec::with_capacity(out_size);
    let out_slice = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut out);
    let mut offset: usize = 0;
    for byte_range in byte_ranges {
        let start = usize::try_from(byte_range.start(bytes.len() as u64)).unwrap();
        let byte_range_len = usize::try_from(byte_range.length(bytes.len() as u64)).unwrap();
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
    out
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
    byte_ranges: &[ByteRange],
) -> std::io::Result<Vec<Vec<u8>>> {
    let len: u64 = bytes.seek(SeekFrom::End(0))?;
    let mut out = Vec::with_capacity(byte_ranges.len());
    for byte_range in byte_ranges {
        let data: Vec<u8> = match byte_range {
            ByteRange::FromStart(offset, None) => {
                bytes.seek(SeekFrom::Start(*offset))?;
                let length = usize::try_from(len).unwrap();
                let mut data = vec![0; length];
                bytes.read_exact(&mut data)?;
                data
            }
            ByteRange::FromStart(offset, Some(length)) => {
                bytes.seek(SeekFrom::Start(*offset))?;
                let length = usize::try_from(*length).unwrap();
                let mut data = vec![0; length];
                bytes.read_exact(&mut data)?;
                data
            }
            ByteRange::Suffix(length) => {
                bytes.seek(SeekFrom::End(-i64::try_from(*length).unwrap()))?;
                let length = usize::try_from(*length).unwrap();
                let mut data = vec![0; length];
                bytes.read_exact(&mut data)?;
                data
            }
        };
        out.push(data);
    }
    Ok(out)
}

/// Extract byte ranges from bytes implementing [`Read`].
///
/// # Errors
///
/// Returns a [`std::io::Error`] if there is an error reading from `bytes`.
/// This can occur if the byte range is out-of-bounds of the `bytes`.
///
/// # Panics
///
/// Panics if a byte has length exceeding [`usize::MAX`].
pub fn extract_byte_ranges_read<T: Read>(
    bytes: &mut T,
    size: u64,
    byte_ranges: &[ByteRange],
) -> std::io::Result<Vec<Vec<u8>>> {
    // Could this be cleaner/more efficient?

    // Allocate output and find the endpoints of the "segments" of bytes which must be read
    let mut out = Vec::with_capacity(byte_ranges.len());
    let mut segments_endpoints = BTreeSet::<u64>::new();
    for byte_range in byte_ranges {
        out.push(vec![0; usize::try_from(byte_range.length(size)).unwrap()]);
        segments_endpoints.insert(byte_range.start(size));
        segments_endpoints.insert(byte_range.end(size));
    }

    // Find the overlapping part of each byte range with each segment
    //                 SEGMENT start     , end        OUTPUT index, offset
    let mut overlap: BTreeMap<(ByteOffset, ByteOffset), Vec<(usize, ByteOffset)>> = BTreeMap::new();
    for (byte_range_index, byte_range) in byte_ranges.iter().enumerate() {
        let byte_range_start = byte_range.start(size);
        let range = segments_endpoints.range((
            std::ops::Bound::Included(byte_range_start),
            std::ops::Bound::Included(byte_range.end(size)),
        ));
        for (segment_start, segment_end) in range.tuple_windows() {
            let byte_range_offset = *segment_start - byte_range_start;
            overlap
                .entry((*segment_start, *segment_end))
                .or_default()
                .push((byte_range_index, byte_range_offset));
        }
    }

    let mut bytes_offset = 0u64;
    for ((segment_start, segment_end), outputs) in overlap {
        // Go to the start of the segment
        if segment_start > bytes_offset {
            std::io::copy(
                &mut bytes.take(segment_start - bytes_offset),
                &mut std::io::sink(),
            )
            .unwrap();
        }

        let segment_length = segment_end - segment_start;
        if outputs.is_empty() {
            // No byte ranges are associated with this segment, so just read it to sink
            std::io::copy(&mut bytes.take(segment_length), &mut std::io::sink()).unwrap();
        } else {
            // Populate all byte ranges in this segment with data
            let segment_length_usize = usize::try_from(segment_length).unwrap();
            let mut segment_bytes = vec![0; segment_length_usize];
            bytes.take(segment_length).read_exact(&mut segment_bytes)?;
            for (byte_range_index, byte_range_offset) in outputs {
                let byte_range_offset = usize::try_from(byte_range_offset).unwrap();
                out[byte_range_index][byte_range_offset..byte_range_offset + segment_length_usize]
                    .copy_from_slice(&segment_bytes);
            }
        }

        // Offset is now the end of the segment
        bytes_offset = segment_end;
    }

    Ok(out)
}

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

        assert!(validate_byte_ranges(&[ByteRange::FromStart(1, Some(5))], 6).is_ok());
        assert!(validate_byte_ranges(&[ByteRange::FromStart(1, Some(5))], 2).is_err());

        assert!(validate_byte_ranges(&[ByteRange::Suffix(5)], 6).is_ok());
        assert!(validate_byte_ranges(&[ByteRange::Suffix(5)], 2).is_err());

        assert!(extract_byte_ranges(&[1, 2, 3], &[ByteRange::FromStart(1, Some(2))]).is_ok());
        let bytes = extract_byte_ranges(&[1, 2, 3], &[ByteRange::FromStart(1, Some(4))]);
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
    fn test_extract_byte_ranges_read() {
        let data: Vec<u8> = (0..10).collect();
        let size = data.len() as u64;
        let mut read = std::io::Cursor::new(data);
        let byte_ranges = vec![
            ByteRange::FromStart(3, Some(3)),
            ByteRange::FromStart(4, Some(1)),
            ByteRange::FromStart(1, Some(1)),
            ByteRange::Suffix(5),
        ];
        let out = extract_byte_ranges_read(&mut read, size, &byte_ranges).unwrap();
        assert_eq!(
            out,
            vec![vec![3, 4, 5], vec![4], vec![1], vec![5, 6, 7, 8, 9]]
        );
    }
}
