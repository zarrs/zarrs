//! Byte sources of planned tasks and their window algebra.

use std::sync::Arc;

use zarrs_storage::byte_range::{ByteRange, InvalidByteRangeError};

use crate::{BytesRepresentation, CodecError, CowBytes};

/// Decodes byte ranges of the decoded bytes of an encoded value without decoding all of it.
///
/// For example, `blosc` can decode whole items of a compressed buffer.
pub trait RangeDecoder:
    core::fmt::Debug + zarrs_plugin::MaybeSend + zarrs_plugin::MaybeSync
{
    /// The length of the decoded bytes of `encoded`.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `encoded` is invalid.
    fn decoded_len(&self, encoded: &[u8]) -> Result<u64, CodecError>;

    /// Decode `ranges` of the decoded bytes of `encoded`.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `encoded` is invalid or a range is out of bounds.
    fn decode_ranges(
        &self,
        encoded: &[u8],
        ranges: &[ByteRange],
    ) -> Result<Vec<CowBytes<'static>>, CodecError>;
}

/// How planned partial decoding reads through a bytes-to-bytes codec.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SourceStep {
    /// Read and decode the entire input.
    Materialize,
    /// The decoded bytes are the encoded bytes without their first `prefix` and last `suffix` bytes.
    ///
    /// Partial reads of the decoded bytes are reads of this window of the encoded bytes.
    Window {
        /// The number of bytes before the decoded bytes.
        prefix: u64,
        /// The number of bytes after the decoded bytes.
        suffix: u64,
    },
    /// Read the entire input, then decode only the requested ranges of the decoded bytes.
    RangeDecode(Arc<dyn RangeDecoder>),
}

/// Identifies an external byte source (e.g. a store key) in the I/O backend of a drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SourceKey(pub usize);

/// The root of a [`ByteSource`].
#[derive(Debug, Clone)]
pub enum SourceRoot<'a> {
    /// Bytes retrieved by the I/O backend of a drive.
    External(SourceKey),
    /// Bytes in memory.
    ///
    /// Reads are zero-copy slices and need no I/O.
    Memory(CowBytes<'a>),
    /// The decoded bytes of encoded bytes in memory, of which reads decode only the requested ranges.
    ///
    /// Reads need no I/O.
    RangeDecoded {
        /// The encoded bytes.
        encoded: CowBytes<'a>,
        /// The decoder of ranges of the decoded bytes.
        decoder: Arc<dyn RangeDecoder>,
    },
}

/// A window into a root byte source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    /// The `len` bytes starting at `offset`.
    Known {
        /// The offset of the window in the root.
        offset: u64,
        /// The length of the window.
        len: u64,
    },
    /// The root without its first `prefix` and last `suffix` bytes, where the length of the root is unknown.
    Open {
        /// The number of bytes excluded from the start of the root.
        prefix: u64,
        /// The number of bytes excluded from the end of the root.
        suffix: u64,
    },
}

/// A byte range of a [`ByteSource`] resolved to its root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedRange {
    /// The byte range to read from the root.
    pub range: ByteRange,
    /// The number of bytes to drop from the end of the bytes read.
    pub drop_tail: u64,
}

impl ResolvedRange {
    /// Apply [`drop_tail`](Self::drop_tail) to `bytes` read from the root with [`range`](Self::range).
    ///
    /// # Errors
    /// Returns [`InvalidByteRangeError`] if `bytes` is shorter than `drop_tail`.
    pub fn finish<'a>(
        &self,
        mut bytes: CowBytes<'a>,
        requested: ByteRange,
    ) -> Result<CowBytes<'a>, InvalidByteRangeError> {
        let len = bytes.len() as u64;
        if len < self.drop_tail {
            return Err(InvalidByteRangeError::new(requested, len));
        }
        if self.drop_tail > 0 {
            bytes.truncate(usize::try_from(len - self.drop_tail).unwrap());
        }
        Ok(bytes)
    }
}

/// A window of bytes in a root source that a task reads from.
///
/// Windows compose in closed form, so a read through any number of nested windows
/// (e.g. a subchunk of a subchunk of a shard, after stripping a checksum) is a single
/// read of the root.
#[derive(Debug, Clone)]
pub struct ByteSource<'a> {
    root: SourceRoot<'a>,
    window: Window,
}

impl<'a> ByteSource<'a> {
    /// All bytes of an external source.
    #[must_use]
    pub fn external(key: SourceKey) -> Self {
        Self {
            root: SourceRoot::External(key),
            window: Window::Open {
                prefix: 0,
                suffix: 0,
            },
        }
    }

    /// All bytes of `bytes`.
    #[must_use]
    pub fn memory(bytes: CowBytes<'a>) -> Self {
        let len = bytes.len() as u64;
        Self {
            root: SourceRoot::Memory(bytes),
            window: Window::Known { offset: 0, len },
        }
    }

    /// The decoded bytes of `encoded`, of which reads decode only the requested ranges with `decoder`.
    ///
    /// The decoded length reported by the decoder (e.g. from a header of the encoded bytes) must be compatible with `decoded_representation`.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `encoded` is invalid or its decoded length is incompatible with `decoded_representation`.
    pub fn range_decoded(
        encoded: CowBytes<'a>,
        decoder: Arc<dyn RangeDecoder>,
        decoded_representation: BytesRepresentation,
    ) -> Result<Self, CodecError> {
        let len = decoder.decoded_len(&encoded)?;
        let valid = match decoded_representation {
            BytesRepresentation::FixedSize(size) => len == size,
            BytesRepresentation::BoundedSize(size) => len <= size,
            BytesRepresentation::UnboundedSize => true,
        };
        if !valid {
            return Err(CodecError::Other(format!(
                "the decoded length {len} is incompatible with the decoded representation {decoded_representation}"
            )));
        }
        Ok(Self {
            root: SourceRoot::RangeDecoded { encoded, decoder },
            window: Window::Known { offset: 0, len },
        })
    }

    /// The root of the source.
    #[must_use]
    pub fn root(&self) -> &SourceRoot<'a> {
        &self.root
    }

    /// The window of the source in its root.
    #[must_use]
    pub fn window(&self) -> Window {
        self.window
    }

    /// The length of the source, if known.
    #[must_use]
    pub fn len(&self) -> Option<u64> {
        match self.window {
            Window::Known { len, .. } => Some(len),
            Window::Open { .. } => None,
        }
    }

    /// The bytes of a source with a [`SourceRoot::Memory`] root, borrowed from the root.
    ///
    /// Returns [`None`] if the root is not in memory.
    ///
    /// # Errors
    /// Returns an [`InvalidByteRangeError`] if the window of the source is not within the root.
    pub fn memory_slice(&self) -> Option<Result<&[u8], InvalidByteRangeError>> {
        let SourceRoot::Memory(bytes) = &self.root else {
            return None;
        };
        let root_len = bytes.len() as u64;
        let range = ByteRange::FromStart(0, None);
        let invalid = || InvalidByteRangeError::new(range, root_len);
        let (start, end) = match self.window {
            Window::Known { offset, len } => (offset, offset.checked_add(len)),
            Window::Open { prefix, suffix } => (prefix, root_len.checked_sub(suffix)),
        };
        Some(match end {
            Some(end) if start <= end && end <= root_len => {
                // Cannot truncate: both are at most the length of the root
                Ok(&bytes[start as usize..end as usize])
            }
            _ => Err(invalid()),
        })
    }

    /// The size of the bytes in memory of the root of the source.
    #[must_use]
    pub fn size_held(&self) -> usize {
        match &self.root {
            SourceRoot::External(_) => 0,
            SourceRoot::Memory(bytes) | SourceRoot::RangeDecoded { encoded: bytes, .. } => {
                bytes.len()
            }
        }
    }

    /// Returns true if the source is known to be empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }

    /// Returns true if reads of the source need I/O.
    #[must_use]
    pub fn is_external(&self) -> bool {
        matches!(self.root, SourceRoot::External(_))
    }

    /// The `len` bytes of this source starting at `offset`.
    ///
    /// # Errors
    /// Returns [`InvalidByteRangeError`] if the interval exceeds the known length of the source,
    /// or its position in the root overflows.
    pub fn interval(&self, offset: u64, len: u64) -> Result<Self, InvalidByteRangeError> {
        let invalid =
            |size| InvalidByteRangeError::new(ByteRange::FromStart(offset, Some(len)), size);
        let window = match self.window {
            Window::Known {
                offset: window_offset,
                len: window_len,
            } => {
                if offset.checked_add(len).is_none_or(|end| end > window_len) {
                    return Err(invalid(window_len));
                }
                Window::Known {
                    // Cannot overflow: the window is within the root
                    offset: window_offset + offset,
                    len,
                }
            }
            Window::Open { prefix, .. } => {
                let offset = prefix
                    .checked_add(offset)
                    .filter(|offset| offset.checked_add(len).is_some())
                    .ok_or_else(|| invalid(u64::MAX))?;
                Window::Known { offset, len }
            }
        };
        Ok(Self {
            root: self.root.clone(),
            window,
        })
    }

    /// This source without its first `prefix` and last `suffix` bytes.
    ///
    /// # Errors
    /// Returns [`InvalidByteRangeError`] if the known length of the source is less than `prefix + suffix`,
    /// or the window overflows.
    pub fn strip(&self, prefix: u64, suffix: u64) -> Result<Self, InvalidByteRangeError> {
        let invalid = |size| InvalidByteRangeError::new(ByteRange::FromStart(prefix, None), size);
        let window = match self.window {
            Window::Known { offset, len } => {
                let stripped_len = prefix
                    .checked_add(suffix)
                    .and_then(|strip| len.checked_sub(strip))
                    .ok_or_else(|| invalid(len))?;
                Window::Known {
                    // Cannot overflow: the window is within the root
                    offset: offset + prefix,
                    len: stripped_len,
                }
            }
            Window::Open {
                prefix: window_prefix,
                suffix: window_suffix,
            } => Window::Open {
                prefix: window_prefix
                    .checked_add(prefix)
                    .ok_or_else(|| invalid(u64::MAX))?,
                suffix: window_suffix
                    .checked_add(suffix)
                    .ok_or_else(|| invalid(u64::MAX))?,
            },
        };
        Ok(Self {
            root: self.root.clone(),
            window,
        })
    }

    /// Resolve `range` of this source to a read of its root.
    ///
    /// # Errors
    /// Returns [`InvalidByteRangeError`] if `range` exceeds the known length of the source,
    /// or its position in the root overflows.
    pub fn resolve(&self, range: ByteRange) -> Result<ResolvedRange, InvalidByteRangeError> {
        match self.window {
            Window::Known { offset, len } => {
                let invalid = || InvalidByteRangeError::new(range, len);
                let (start, length) = match range {
                    ByteRange::FromStart(start, None) => {
                        (start, len.checked_sub(start).ok_or_else(invalid)?)
                    }
                    ByteRange::FromStart(start, Some(length)) => {
                        if start.checked_add(length).is_none_or(|end| end > len) {
                            return Err(invalid());
                        }
                        (start, length)
                    }
                    ByteRange::Suffix(length) => {
                        (len.checked_sub(length).ok_or_else(invalid)?, length)
                    }
                };
                Ok(ResolvedRange {
                    // Cannot overflow: the range is within the window, which is within the root
                    range: ByteRange::FromStart(offset + start, Some(length)),
                    drop_tail: 0,
                })
            }
            Window::Open { prefix, suffix } => {
                let invalid = || InvalidByteRangeError::new(range, u64::MAX);
                Ok(match range {
                    ByteRange::FromStart(start, None) => ResolvedRange {
                        range: ByteRange::FromStart(
                            prefix.checked_add(start).ok_or_else(invalid)?,
                            None,
                        ),
                        drop_tail: suffix,
                    },
                    ByteRange::FromStart(start, Some(length)) => {
                        let start = prefix
                            .checked_add(start)
                            .filter(|start| start.checked_add(length).is_some())
                            .ok_or_else(invalid)?;
                        ResolvedRange {
                            range: ByteRange::FromStart(start, Some(length)),
                            drop_tail: 0,
                        }
                    }
                    ByteRange::Suffix(length) => ResolvedRange {
                        range: ByteRange::Suffix(length.checked_add(suffix).ok_or_else(invalid)?),
                        drop_tail: suffix,
                    },
                })
            }
        }
    }

    /// Resolve `ranges` of this source to reads of its root, in place.
    ///
    /// The bytes read for a resolved range are finished with [`finish_read`](Self::finish_read).
    ///
    /// # Errors
    /// Returns [`InvalidByteRangeError`] if a range is invalid for the source (see [`resolve`](Self::resolve)).
    pub(crate) fn resolve_in_place(
        &self,
        ranges: &mut [ByteRange],
    ) -> Result<(), InvalidByteRangeError> {
        if let Window::Open {
            prefix: 0,
            suffix: 0,
        } = self.window
        {
            // The whole root
            return Ok(());
        }
        for range in ranges {
            *range = self.resolve(*range)?.range;
        }
        Ok(())
    }

    /// Finish `bytes` read from the root with a `resolved` range (see [`resolve_in_place`](Self::resolve_in_place)).
    ///
    /// # Errors
    /// Returns [`InvalidByteRangeError`] if `bytes` is shorter than the bytes of the root excluded by the window.
    pub(crate) fn finish_read<'b>(
        &self,
        bytes: CowBytes<'b>,
        resolved: ByteRange,
    ) -> Result<CowBytes<'b>, InvalidByteRangeError> {
        // Only unbounded reads of an open window include bytes after the window
        let drop_tail = match (self.window, resolved) {
            (Window::Open { suffix, .. }, ByteRange::FromStart(_, None) | ByteRange::Suffix(_)) => {
                suffix
            }
            _ => return Ok(bytes),
        };
        ResolvedRange {
            range: resolved,
            drop_tail,
        }
        .finish(bytes, resolved)
    }

    /// Read `ranges` of this source if its root is in memory.
    ///
    /// Returns [`None`] if the root is external.
    /// The bytes of a [`SourceRoot::Memory`] root are zero-copy slices of the root.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if a range exceeds the length of the source or cannot be decoded.
    pub fn read_memory_many(
        &self,
        ranges: &[ByteRange],
    ) -> Option<Result<Vec<CowBytes<'a>>, CodecError>> {
        match &self.root {
            SourceRoot::External(_) => None,
            SourceRoot::Memory(_) => Some(
                ranges
                    .iter()
                    .map(|range| {
                        self.read_memory(*range)
                            .expect("the source is in memory")
                            .map_err(CodecError::from)
                    })
                    .collect(),
            ),
            SourceRoot::RangeDecoded { encoded, decoder } => Some((|| {
                let resolved = ranges
                    .iter()
                    .map(|range| self.resolve(*range))
                    .collect::<Result<Vec<_>, _>>()?;
                let root_ranges: Vec<ByteRange> =
                    resolved.iter().map(|resolved| resolved.range).collect();
                let decoded = decoder.decode_ranges(encoded, &root_ranges)?;
                if decoded.len() != ranges.len() {
                    return Err(CodecError::Other(
                        "a range decoder returned an unexpected number of ranges".to_string(),
                    ));
                }
                // Ranges of a known window are resolved to bounded ranges
                for (bytes, range) in std::iter::zip(&decoded, &root_ranges) {
                    if let ByteRange::FromStart(_, Some(length)) = range
                        && bytes.len() as u64 != *length
                    {
                        return Err(CodecError::Other(format!(
                            "a range decoder returned {} bytes for a range of {length} bytes",
                            bytes.len()
                        )));
                    }
                }
                std::iter::zip(decoded, std::iter::zip(resolved, ranges))
                    .map(|(bytes, (resolved, range))| Ok(resolved.finish(bytes, *range)?))
                    .collect()
            })()),
        }
    }

    /// Read `range` of this source if its root is in memory.
    ///
    /// Returns [`None`] if the root is external or range decoded (see [`read_memory_many`](Self::read_memory_many)).
    /// The bytes are a zero-copy slice of the root.
    ///
    /// # Errors
    /// Returns [`InvalidByteRangeError`] if `range` exceeds the length of the source.
    pub fn read_memory(
        &self,
        range: ByteRange,
    ) -> Option<Result<CowBytes<'a>, InvalidByteRangeError>> {
        let SourceRoot::Memory(bytes) = &self.root else {
            return None;
        };
        Some(self.resolve(range).and_then(|resolved| {
            let root_len = bytes.len() as u64;
            let invalid = || InvalidByteRangeError::new(range, root_len);
            let (start, end) = match resolved.range {
                ByteRange::FromStart(start, None) => (start, root_len),
                ByteRange::FromStart(start, Some(length)) => {
                    (start, start.checked_add(length).ok_or_else(invalid)?)
                }
                ByteRange::Suffix(length) => {
                    (root_len.checked_sub(length).ok_or_else(invalid)?, root_len)
                }
            };
            if start > end || end > root_len {
                return Err(invalid());
            }
            // Cannot truncate: both are at most the length of the root
            let (start, end) = (start as usize, end as usize);
            resolved.finish(bytes.slice(start..end), range)
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use zarrs_storage::byte_range::extract_byte_ranges;

    use super::*;

    #[derive(Debug, Clone, Copy)]
    enum Op {
        Interval(u64, u64),
        Strip(u64, u64),
    }

    /// Apply `ops` to a slice: the reference for the window algebra.
    fn reference(bytes: &[u8], ops: &[Op]) -> Option<Vec<u8>> {
        let mut bytes = bytes;
        for op in ops {
            let len = bytes.len() as u64;
            bytes = match *op {
                Op::Interval(offset, n) => {
                    if offset + n > len {
                        return None;
                    }
                    &bytes[offset as usize..(offset + n) as usize]
                }
                Op::Strip(prefix, suffix) => {
                    if prefix + suffix > len {
                        return None;
                    }
                    &bytes[prefix as usize..(len - suffix) as usize]
                }
            };
        }
        Some(bytes.to_vec())
    }

    fn apply<'a>(mut source: ByteSource<'a>, ops: &[Op]) -> Option<ByteSource<'a>> {
        for op in ops {
            source = match *op {
                Op::Interval(offset, len) => source.interval(offset, len).ok()?,
                Op::Strip(prefix, suffix) => source.strip(prefix, suffix).ok()?,
            };
        }
        Some(source)
    }

    /// Read `range` from an external source of `bytes` as a store would.
    fn read_external(
        bytes: &[u8],
        source: &ByteSource<'_>,
        range: ByteRange,
    ) -> Result<Vec<u8>, InvalidByteRangeError> {
        let resolved = source.resolve(range)?;
        let read = extract_byte_ranges(bytes, [resolved.range].into_iter())
            .map_err(|_| InvalidByteRangeError::new(range, bytes.len() as u64))?
            .remove(0);
        Ok(resolved
            .finish(CowBytes::from(read), range)?
            .as_slice()
            .to_vec())
    }

    fn ranges(len: u64) -> Vec<ByteRange> {
        let mut ranges = vec![];
        for start in 0..=len {
            ranges.push(ByteRange::FromStart(start, None));
            for length in 0..=(len - start) {
                ranges.push(ByteRange::FromStart(start, Some(length)));
            }
        }
        for length in 0..=len {
            ranges.push(ByteRange::Suffix(length));
        }
        ranges
    }

    fn op_sequences() -> Vec<Vec<Op>> {
        let mut single = vec![];
        for a in 0..3 {
            for b in 0..4 {
                single.push(Op::Interval(a, b));
                single.push(Op::Strip(a, b));
            }
        }
        let mut sequences = vec![vec![]];
        for op in &single {
            sequences.push(vec![*op]);
            for op2 in &single {
                sequences.push(vec![*op, *op2]);
            }
        }
        sequences
    }

    /// Every valid read through every composition of windows matches the reference.
    #[test]
    fn window_algebra_matches_reference() {
        let bytes: Vec<u8> = (0..9).collect();
        for ops in op_sequences() {
            let Some(expected) = reference(&bytes, &ops) else {
                // Compositions that are invalid for a known length are rejected for a memory source
                assert!(apply(ByteSource::memory(CowBytes::from(&bytes[..])), &ops).is_none());
                continue;
            };
            let memory = apply(ByteSource::memory(CowBytes::from(&bytes[..])), &ops).unwrap();
            let external = apply(ByteSource::external(SourceKey(0)), &ops).unwrap();
            assert_eq!(memory.len().unwrap(), expected.len() as u64);
            for range in ranges(expected.len() as u64) {
                let expected_range = extract_byte_ranges(&expected, [range].into_iter())
                    .unwrap()
                    .remove(0);
                let from_memory = memory.read_memory(range).unwrap().unwrap();
                assert_eq!(from_memory.as_slice(), expected_range, "{ops:?} {range:?}");
                assert_eq!(
                    read_external(&bytes, &external, range).unwrap(),
                    expected_range,
                    "{ops:?} {range:?}"
                );
            }
            // Reads past the end of a known window are rejected
            let len = expected.len() as u64;
            for range in [
                ByteRange::FromStart(len + 1, None),
                ByteRange::FromStart(0, Some(len + 1)),
                ByteRange::Suffix(len + 1),
            ] {
                assert!(memory.resolve(range).is_err(), "{ops:?} {range:?}");
            }
        }
    }

    /// Window and range positions that overflow are rejected rather than wrapped.
    #[test]
    fn overflow_is_rejected() {
        let bytes: Vec<u8> = (0..16).collect();
        let memory = ByteSource::memory(CowBytes::from(&bytes[..]));
        assert!(memory.interval(u64::MAX, 2).is_err());
        assert!(memory.strip(u64::MAX, 2).is_err());
        assert!(
            memory
                .read_memory(ByteRange::FromStart(u64::MAX, Some(2)))
                .unwrap()
                .is_err()
        );
        let external = ByteSource::external(SourceKey(0)).strip(8, 0).unwrap();
        assert!(external.interval(u64::MAX - 4, 2).is_err());
        assert!(external.interval(4, u64::MAX).is_err());
        assert!(external.strip(u64::MAX, 0).is_err());
        assert!(
            external
                .resolve(ByteRange::FromStart(u64::MAX, None))
                .is_err()
        );
        assert!(
            external
                .resolve(ByteRange::FromStart(0, Some(u64::MAX)))
                .is_err()
        );
        let external = external.strip(0, 8).unwrap();
        assert!(external.resolve(ByteRange::Suffix(u64::MAX)).is_err());
        let interval = external.interval(u64::MAX - 16, 8).unwrap();
        assert!(interval.resolve(ByteRange::Suffix(u64::MAX)).is_err());
        assert!(interval.resolve(ByteRange::FromStart(9, None)).is_err());
    }

    /// A range decoder whose decoded length and outputs are controlled by the test.
    #[derive(Debug)]
    struct FakeRangeDecoder {
        decoded_len: u64,
        output_len: Option<usize>,
    }

    impl RangeDecoder for FakeRangeDecoder {
        fn decoded_len(&self, _encoded: &[u8]) -> Result<u64, CodecError> {
            Ok(self.decoded_len)
        }

        fn decode_ranges(
            &self,
            _encoded: &[u8],
            ranges: &[ByteRange],
        ) -> Result<Vec<CowBytes<'static>>, CodecError> {
            Ok(ranges
                .iter()
                .map(|range| {
                    let len = self
                        .output_len
                        .unwrap_or_else(|| range.length(self.decoded_len) as usize);
                    CowBytes::from(vec![0u8; len])
                })
                .collect())
        }
    }

    #[test]
    fn range_decoded_sources_are_validated() {
        let decoder = |decoded_len, output_len| -> Arc<dyn RangeDecoder> {
            Arc::new(FakeRangeDecoder {
                decoded_len,
                output_len,
            })
        };
        let encoded = CowBytes::from(vec![1u8; 4]);
        // The decoded length must match the decoded representation
        assert!(
            ByteSource::range_decoded(
                encoded.clone(),
                decoder(u64::MAX, None),
                BytesRepresentation::FixedSize(16)
            )
            .is_err()
        );
        assert!(
            ByteSource::range_decoded(
                encoded.clone(),
                decoder(17, None),
                BytesRepresentation::BoundedSize(16)
            )
            .is_err()
        );
        let source = ByteSource::range_decoded(
            encoded.clone(),
            decoder(16, None),
            BytesRepresentation::FixedSize(16),
        )
        .unwrap();
        let reads = source
            .read_memory_many(&[ByteRange::FromStart(2, Some(3)), ByteRange::Suffix(4)])
            .unwrap()
            .unwrap();
        assert_eq!(reads[0].len(), 3);
        assert_eq!(reads[1].len(), 4);
        // Decoded ranges must have the requested length
        let source = ByteSource::range_decoded(
            encoded,
            decoder(16, Some(1)),
            BytesRepresentation::FixedSize(16),
        )
        .unwrap();
        assert!(
            source
                .read_memory_many(&[ByteRange::FromStart(2, Some(3))])
                .unwrap()
                .is_err()
        );
    }

    /// Memory reads are zero-copy slices of the root.
    #[test]
    fn memory_reads_are_slices() {
        let bytes: Vec<u8> = (0..16).collect();
        let source = ByteSource::memory(CowBytes::from(&bytes[..]))
            .strip(1, 2)
            .unwrap()
            .interval(3, 5)
            .unwrap();
        let read = source
            .read_memory(ByteRange::FromStart(1, Some(2)))
            .unwrap()
            .unwrap();
        assert!(matches!(read, CowBytes::Borrowed(_)));
        assert_eq!(read.as_ptr(), bytes[5..].as_ptr());

        let shared = bytes::Bytes::from(bytes.clone());
        let source = ByteSource::memory(CowBytes::Shared(shared.clone()))
            .interval(4, 4)
            .unwrap();
        let read = source.read_memory(ByteRange::Suffix(2)).unwrap().unwrap();
        assert_eq!(read.as_ptr(), shared[6..].as_ptr());
    }
}
