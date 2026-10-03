use crate::array::chunk_grid::RegularBoundedChunkGrid;
use crate::array::{ArraySubsetTraits, ravel_indices};
use zarrs_chunk_grid::ChunkGridTraits;
use zarrs_codec::{CodecError, CodecOptions};

/// A set of byte-adjacent inner chunks that can be read in a single I/O call.
#[derive(Clone)]
pub(super) struct CoalescedGroup {
    /// Byte offset of the first byte in the shard for this group.
    pub(super) start: u64,
    /// Total byte length of the coalesced read.
    pub(super) total_len: u64,
    /// Positions into `chunk_indices_1d` in ascending byte-offset order.
    pub(super) chunks: Vec<usize>,
}

/// Collect the 1-D ravelled indices of all inner chunks overlapping `array_subset`.
pub(super) fn collect_chunk_indices(
    shard_chunk_grid: &RegularBoundedChunkGrid,
    array_subset: &dyn ArraySubsetTraits,
    chunks_per_shard: &[u64],
) -> Result<Vec<u64>, CodecError> {
    let chunks = shard_chunk_grid
        .chunks_in_array_subset(array_subset)?
        .expect("subchunks always within shard");
    let mut chunk_indices = Vec::with_capacity(chunks.num_elements_usize());
    for chunk_indices_nd in chunks.indices() {
        let idx = ravel_indices(&chunk_indices_nd, chunks_per_shard).expect("inbounds chunk");
        chunk_indices.push(idx);
    }
    Ok(chunk_indices)
}

/// Sort inner chunks by byte offset and merge exactly-adjacent ranges.
///
/// Returns coalesced groups and fill-value positions, both as positions into `chunk_indices_1d`.
///
/// # Errors
/// Returns an error if a shard index entry has only one of `offset`/`size` equal to `u64::MAX`,
/// which indicates a corrupted shard index.
pub(super) fn coalesce_chunks(
    chunk_indices_1d: &[u64],
    shard_index: &[u64],
) -> Result<(Vec<CoalescedGroup>, Vec<usize>), CodecError> {
    let mut fill_positions: Vec<usize> = Vec::new();
    let mut io_positions: Vec<usize> = Vec::new();
    for (pos, &idx) in chunk_indices_1d.iter().enumerate() {
        let i = usize::try_from(idx).unwrap();
        let offset = shard_index[i * 2];
        let size = shard_index[i * 2 + 1];
        match (offset == u64::MAX, size == u64::MAX) {
            (true, true) => fill_positions.push(pos),
            (false, false) => io_positions.push(pos),
            _ => {
                return Err(CodecError::Other(
                    "Shard index entry has mismatched sentinel values; the shard may be corrupted."
                        .to_string(),
                ));
            }
        }
    }

    io_positions.sort_by_key(|&pos| {
        let i = usize::try_from(chunk_indices_1d[pos]).unwrap();
        shard_index[i * 2]
    });

    let mut groups: Vec<CoalescedGroup> = Vec::new();
    for pos in io_positions {
        let i = usize::try_from(chunk_indices_1d[pos]).unwrap();
        let offset = shard_index[i * 2];
        let size = shard_index[i * 2 + 1];
        if let Some(last) = groups.last_mut()
            && last.start + last.total_len == offset
        {
            last.total_len += size;
            last.chunks.push(pos);
        } else {
            groups.push(CoalescedGroup {
                start: offset,
                total_len: size,
                chunks: vec![pos],
            });
        }
    }

    Ok((groups, fill_positions))
}

pub(super) fn group_read_concurrent_limit(options: &CodecOptions, num_groups: usize) -> usize {
    std::cmp::min(options.concurrent_target(), num_groups).max(1)
}

pub(super) fn ready_chunks(groups: &[CoalescedGroup]) -> Vec<(usize, usize)> {
    groups
        .iter()
        .enumerate()
        .flat_map(|(group_idx, group)| {
            (0..group.chunks.len()).map(move |chunk_idx| (group_idx, chunk_idx))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_read_concurrency_is_bounded() {
        let options = CodecOptions::default().with_concurrent_target(4);

        assert_eq!(group_read_concurrent_limit(&options, 0), 1);
        assert_eq!(group_read_concurrent_limit(&options, 2), 2);
        assert_eq!(group_read_concurrent_limit(&options, 8), 4);
    }

    #[test]
    fn ready_chunks_flattens_skewed_groups() {
        let groups = [
            CoalescedGroup {
                start: 0,
                total_len: 4,
                chunks: vec![0, 1, 2, 3],
            },
            CoalescedGroup {
                start: 8,
                total_len: 1,
                chunks: vec![4],
            },
            CoalescedGroup {
                start: 12,
                total_len: 1,
                chunks: vec![5],
            },
        ];

        assert_eq!(
            ready_chunks(&groups),
            vec![(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0)]
        );
    }
}
