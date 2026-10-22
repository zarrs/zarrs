pub(super) use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

pub(super) use libc::O_DIRECT;

/// A range of intersected pages, with `Ord` tailored for coalescing.
#[derive(Eq, PartialEq, Debug)]
pub(super) struct IntersectedPages(pub std::ops::Range<u64>);

impl Ord for IntersectedPages {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Increasing order for start, decreasing order for end
        self.0
            .start
            .cmp(&other.0.start)
            .then_with(|| other.0.end.cmp(&self.0.end))
    }
}

impl PartialOrd for IntersectedPages {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub(super) fn coalesce_byte_ranges_with_page_size(
    byte_ranges: impl Iterator<Item = std::ops::Range<u64>>,
    page_size: u64,
) -> impl Iterator<Item = IntersectedPages> {
    use itertools::Itertools;

    // Find intersected pages
    let intersected_pages: std::collections::BTreeSet<IntersectedPages> = byte_ranges
        .map(|range| {
            let start_page = range.start / page_size;
            let end_page = range.end.div_ceil(page_size);
            IntersectedPages(start_page..end_page)
        })
        .collect();

    // Determine the pages to read (joining neighbouring pages)
    intersected_pages.into_iter().coalesce(|a, b| {
        if a.0.end >= b.0.start {
            Ok(IntersectedPages(a.0.start..b.0.end.max(a.0.end)))
        } else {
            Err((a, b))
        }
    })
}

#[cfg(test)]
mod tests {
    use zarrs_storage::byte_range::ByteRange;

    use super::*;

    #[test]
    fn test_coalesce_byte_ranges_with_page_size() {
        let ps = 4;
        let file_size = 64;
        let byte_ranges = [
            ByteRange::FromStart(5, Some(2)),  // 1
            ByteRange::FromStart(0, Some(1)),  // 0
            ByteRange::FromStart(30, Some(4)), // 7-8
            ByteRange::Suffix(4),              // 15
            ByteRange::FromStart(8, Some(4)),  // 2
            ByteRange::FromStart(8, Some(8)),  // 2-3
            ByteRange::Suffix(7),              // 14-15
        ];
        let pages: Vec<_> = coalesce_byte_ranges_with_page_size(
            byte_ranges.iter().map(|range| range.to_range(file_size)),
            ps,
        )
        .collect();
        let expected = vec![
            IntersectedPages(0..4),
            IntersectedPages(7..9),
            IntersectedPages(14..16),
        ];
        assert_eq!(pages, expected);
    }
}
