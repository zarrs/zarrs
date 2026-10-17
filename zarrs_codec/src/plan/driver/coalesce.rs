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

/// Returns true if the ranges `(start, end)` `a` and `b` coalesce under `coalescing`: the gap between them is within
/// the maximum gap, and together they span at most the maximum coalesced bytes.
fn ranges_coalesce(a: (u64, u64), b: (u64, u64), coalescing: ReadCoalescing) -> bool {
    let gap = b.0.saturating_sub(a.1).max(a.0.saturating_sub(b.1));
    gap <= coalescing.max_gap() && a.1.max(b.1) - a.0.min(b.0) <= coalescing.max_coalesced_bytes()
}

/// The indices of the `candidates` (byte ranges of reads of the same key as `request`) that coalesce with `request`,
/// directly or through other candidates, in order.
///
/// The ranges are sorted by offset, and each range is checked against the previous range and the range reaching
/// furthest so far, which link chains of nearby ranges and overlapping ranges. This takes O(n log n) time for n ranges,
/// rather than comparing every pair of ranges.
pub(crate) fn coalescing_candidates(
    request: &IoRequest,
    candidates: &[&[ByteRange]],
    coalescing: ReadCoalescing,
) -> Vec<usize> {
    // The ranges of the request (owner 0) and of each candidate (owner i + 1), ordered by offset
    let mut ranges: Vec<(u64, u64, usize)> = std::iter::once(request.ranges.as_slice())
        .chain(candidates.iter().copied())
        .enumerate()
        .flat_map(|(owner, ranges)| {
            ranges.iter().filter_map(move |range| {
                known_range(*range).map(|(offset, length)| (offset, offset + length, owner))
            })
        })
        .collect();
    ranges.sort_unstable();

    // Union-find of the owners whose ranges coalesce
    let mut parent: Vec<usize> = (0..=candidates.len()).collect();
    let find = |parent: &mut Vec<usize>, mut owner: usize| {
        while parent[owner] != owner {
            parent[owner] = parent[parent[owner]];
            owner = parent[owner];
        }
        owner
    };
    let mut previous: Option<(u64, u64, usize)> = None;
    let mut reach: Option<(u64, u64, usize)> = None;
    for &(start, end, owner) in &ranges {
        for (other_start, other_end, other) in [previous, reach].into_iter().flatten() {
            if ranges_coalesce((other_start, other_end), (start, end), coalescing) {
                let (a, b) = (find(&mut parent, owner), find(&mut parent, other));
                parent[a] = b;
            }
        }
        if reach.is_none_or(|(_, reach_end, _)| end > reach_end) {
            reach = Some((start, end, owner));
        }
        previous = Some((start, end, owner));
    }
    let root = find(&mut parent, 0);
    (0..candidates.len())
        .filter(|&candidate| find(&mut parent, candidate + 1) == root)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::SourceKey;

    fn range(offset: u64, length: u64) -> ByteRange {
        ByteRange::FromStart(offset, Some(length))
    }

    fn request(ranges: &[ByteRange]) -> IoRequest {
        IoRequest {
            key: SourceKey(0),
            ranges: ranges.into(),
        }
    }

    /// The candidates coalescing with `request` by comparing every pair of ranges.
    fn naive(
        request: &IoRequest,
        candidates: &[&[ByteRange]],
        coalescing: ReadCoalescing,
    ) -> Vec<usize> {
        let owners: Vec<&[ByteRange]> = std::iter::once(request.ranges.as_slice())
            .chain(candidates.iter().copied())
            .collect();
        let coalesce = |a: &[ByteRange], b: &[ByteRange]| {
            a.iter().any(|a| {
                b.iter().any(|b| {
                    let (a, b) = (known_range(*a).unwrap(), known_range(*b).unwrap());
                    ranges_coalesce((a.0, a.0 + a.1), (b.0, b.0 + b.1), coalescing)
                })
            })
        };
        let mut joined = vec![false; owners.len()];
        joined[0] = true;
        let mut grew = true;
        while grew {
            grew = false;
            for i in 1..owners.len() {
                if !joined[i]
                    && (0..owners.len()).any(|j| joined[j] && coalesce(owners[i], owners[j]))
                {
                    joined[i] = true;
                    grew = true;
                }
            }
        }
        (0..candidates.len()).filter(|&i| joined[i + 1]).collect()
    }

    #[test]
    fn coalescing_candidates_match_pairwise_closure_of_equal_ranges() {
        // A deterministic pseudo-random sequence
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        for _ in 0..500 {
            let length = 1 + next(16);
            let coalescing = ReadCoalescing::new(next(8), length + next(64));
            let owners: Vec<Vec<ByteRange>> = (0..=next(12))
                .map(|_| (0..=next(3)).map(|_| range(next(256), length)).collect())
                .collect();
            let request = request(&owners[0]);
            let candidates: Vec<&[ByteRange]> = owners[1..].iter().map(Vec::as_slice).collect();
            assert_eq!(
                coalescing_candidates(&request, &candidates, coalescing),
                naive(&request, &candidates, coalescing),
                "{owners:?} {coalescing:?}"
            );
        }
    }

    #[test]
    fn coalescing_candidates_cases() {
        let coalescing = ReadCoalescing::new(2, 100);
        // A chain of nearby ranges, in any order
        let request = request(&[range(0, 10)]);
        let (a, b, c) = ([range(24, 10)], [range(12, 10)], [range(60, 10)]);
        assert_eq!(
            coalescing_candidates(&request, &[&a, &b, &c], coalescing),
            [0, 1]
        );
        // A range within a long range coalesces through the long range
        let (long, inner) = ([range(5, 80)], [range(40, 5)]);
        assert_eq!(
            coalescing_candidates(&request, &[&inner, &long], coalescing),
            [0, 1]
        );
        // A candidate with ranges far apart links the reads near each of them
        let (bridge, far) = ([range(11, 2), range(200, 5)], [range(206, 5)]);
        assert_eq!(
            coalescing_candidates(&request, &[&far, &bridge], coalescing),
            [0, 1]
        );
        // Ranges whose span exceeds the maximum do not coalesce
        let wide = [range(11, 95)];
        assert_eq!(
            coalescing_candidates(&request, &[&wide], coalescing),
            [0usize; 0]
        );
        // Ranges without a known length do not coalesce
        let unknown = [ByteRange::Suffix(10)];
        assert_eq!(
            coalescing_candidates(&request, &[&unknown], coalescing),
            [0usize; 0]
        );
    }
}
