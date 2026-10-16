//! Coalescing of queued reads of the same key into one storage request.
// Only the sync and async drivers coalesce reads
#![cfg_attr(all(target_family = "wasm", not(feature = "async")), allow(dead_code))]

use smallvec::SmallVec;
use zarrs_storage::byte_range::ByteRange;
use zarrs_storage::{ReadCoalescing, StorageError};

use super::super::ReadBytes;
use super::core::{IoRequest, IoWork, Work};
use crate::CodecError;

/// The maximum number of queued reads searched for reads of the same key as a dispatched read.
pub(crate) const SCAN_LIMIT: usize = 256;

/// For each range of a read, the index of the coalesced range that holds it and its offset in that range.
type ReadParts = SmallVec<[(usize, u64); 1]>;

/// One or more reads performed with one storage request.
pub(crate) struct ReadGroup<'a> {
    /// The reads and their requests.
    reads: SmallVec<[(IoWork<'a>, IoRequest); 1]>,
    /// The request performed for the reads, if coalesced (otherwise the request of the read).
    coalesced: Option<IoRequest>,
    /// For each read with more than one read, the coalesced range and offset in it of each of its ranges.
    parts: Option<Vec<ReadParts>>,
}

impl<'a> ReadGroup<'a> {
    /// A group of one read, performed as requested.
    pub(crate) fn single(read: IoWork<'a>, request: IoRequest) -> Self {
        Self {
            reads: smallvec::smallvec![(read, request)],
            coalesced: None,
            parts: None,
        }
    }

    /// Coalesce the ranges of `reads` of the same key.
    ///
    /// Every range must have a known length (see [`coalescable`]).
    pub(crate) fn coalesced(
        reads: SmallVec<[(IoWork<'a>, IoRequest); 1]>,
        coalescing: ReadCoalescing,
    ) -> Self {
        debug_assert!(reads.len() > 1);
        let key = reads[0].1.key;
        // The ranges of every read, ordered by offset
        let mut ranges: Vec<(u64, u64, usize, usize)> = reads
            .iter()
            .enumerate()
            .flat_map(|(read, (_, request))| {
                request
                    .ranges
                    .iter()
                    .enumerate()
                    .map(move |(index, range)| {
                        let (offset, length) = known_range(*range).expect("coalescable ranges");
                        (offset, offset + length, read, index)
                    })
            })
            .collect();
        ranges.sort_unstable_by_key(|&(start, end, ..)| (start, end));

        let mut coalesced: SmallVec<[ByteRange; 1]> = SmallVec::new();
        let mut parts: Vec<ReadParts> = reads
            .iter()
            .map(|(_, request)| smallvec::smallvec![(0, 0); request.ranges.len()])
            .collect();
        let mut current: Option<(u64, u64)> = None;
        for (start, end, read, index) in ranges {
            match current {
                Some((current_start, current_end))
                    if start <= current_end.saturating_add(coalescing.max_gap())
                        && end.max(current_end) - current_start
                            <= coalescing.max_coalesced_bytes() =>
                {
                    current = Some((current_start, end.max(current_end)));
                }
                _ => {
                    if let Some((current_start, current_end)) = current {
                        coalesced.push(ByteRange::FromStart(
                            current_start,
                            Some(current_end - current_start),
                        ));
                    }
                    current = Some((start, end));
                }
            }
            let (current_start, _) = current.expect("set above");
            parts[read][index] = (coalesced.len(), start - current_start);
        }
        if let Some((current_start, current_end)) = current {
            coalesced.push(ByteRange::FromStart(
                current_start,
                Some(current_end - current_start),
            ));
        }
        Self {
            reads,
            coalesced: Some(IoRequest {
                key,
                ranges: coalesced,
            }),
            parts: Some(parts),
        }
    }

    /// The storage request of the group.
    pub(crate) fn request(&self) -> &IoRequest {
        self.coalesced.as_ref().unwrap_or(&self.reads[0].1)
    }

    /// The compute steps of the reads of the group, given the `result` of its request.
    ///
    /// The bytes of each read are slices of the bytes of the request.
    pub(crate) fn complete(
        self,
        result: Result<Option<ReadBytes>, StorageError>,
    ) -> SmallVec<[Result<Work<'a>, CodecError>; 1]> {
        let Some(parts) = self.parts else {
            let (read, request) = self.reads.into_iter().next().expect("one read");
            return smallvec::smallvec![read.into_work(&request, result)];
        };
        let bytes = match result {
            Ok(Some(bytes)) => Some(bytes),
            Ok(None) => None,
            Err(err) => {
                return self
                    .reads
                    .into_iter()
                    .map(|(read, request)| read.into_work(&request, Err(err.clone())))
                    .collect();
            }
        };
        let coalesced = self.coalesced.as_ref().expect("coalesced");
        if let Some(bytes) = &bytes
            && (bytes.len() != coalesced.ranges.len()
                || std::iter::zip(bytes, &coalesced.ranges).any(|(bytes, range)| {
                    known_range(*range).is_none_or(|(_, length)| bytes.len() as u64 != length)
                }))
        {
            let err = StorageError::Other(
                "a coalesced read returned an unexpected number of bytes".to_string(),
            );
            return self
                .reads
                .into_iter()
                .map(|(read, request)| read.into_work(&request, Err(err.clone())))
                .collect();
        }
        std::iter::zip(self.reads, parts)
            .map(|((read, request), parts)| {
                let result = bytes.as_ref().map(|bytes| {
                    std::iter::zip(&request.ranges, parts)
                        .map(|(range, (coalesced, offset))| {
                            let (_, length) = known_range(*range).expect("coalescable ranges");
                            let start = usize::try_from(offset).expect("within a read");
                            let end = start + usize::try_from(length).expect("within a read");
                            bytes[coalesced].slice(start..end)
                        })
                        .collect()
                });
                read.into_work(&request, Ok(result))
            })
            .collect()
    }
}

/// The offset and length of a `range` whose length is known without the size of the value, and whose end is
/// representable.
pub(crate) fn known_range(range: ByteRange) -> Option<(u64, u64)> {
    match range {
        ByteRange::FromStart(offset, Some(length)) => {
            offset.checked_add(length).map(|_| (offset, length))
        }
        ByteRange::FromStart(_, None) | ByteRange::Suffix(_) => None,
    }
}

/// The total length of the ranges of `request` with a known length.
pub(crate) fn known_len(request: &IoRequest) -> u64 {
    request
        .ranges
        .iter()
        .filter_map(|range| known_range(*range))
        .fold(0, |len, (_, length)| len.saturating_add(length))
}

/// Returns true if every range of `request` can be coalesced.
pub(crate) fn coalescable(request: &IoRequest) -> bool {
    !request.ranges.is_empty()
        && request
            .ranges
            .iter()
            .all(|range| known_range(*range).is_some())
}

/// Returns true if any range of `candidate` coalesces with a range of `request` under `coalescing`.
pub(crate) fn coalesces_with(
    request: &IoRequest,
    candidate: &[ByteRange],
    coalescing: ReadCoalescing,
) -> bool {
    candidate.iter().any(|candidate| {
        let Some((start, length)) = known_range(*candidate) else {
            return false;
        };
        let end = start + length;
        request.ranges.iter().any(|range| {
            let Some((range_start, range_length)) = known_range(*range) else {
                return false;
            };
            let range_end = range_start + range_length;
            let gap = start
                .saturating_sub(range_end)
                .max(range_start.saturating_sub(end));
            gap <= coalescing.max_gap()
                && end.max(range_end) - start.min(range_start) <= coalescing.max_coalesced_bytes()
        })
    })
}
