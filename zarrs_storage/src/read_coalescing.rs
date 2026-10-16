//! The read coalescing hint of a store.

/// How reads of nearby byte ranges of the same value should be coalesced into one request.
///
/// Byte ranges whose gap is at most [`max_gap`](Self::max_gap) are read as one range, up to
/// [`max_coalesced_bytes`](Self::max_coalesced_bytes), and the bytes of the original ranges are
/// slices of the coalesced read.
/// Coalescing trades the bytes read in gaps for fewer requests, which suits stores where each request has a high
/// latency or cost (e.g. object stores).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReadCoalescing {
    max_gap: u64,
    max_coalesced_bytes: u64,
}

impl ReadCoalescing {
    /// Create a read coalescing hint.
    ///
    /// Byte ranges are coalesced if their gap is at most `max_gap` bytes and the coalesced range is at most
    /// `max_coalesced_bytes`.
    #[must_use]
    pub const fn new(max_gap: u64, max_coalesced_bytes: u64) -> Self {
        Self {
            max_gap,
            max_coalesced_bytes,
        }
    }

    /// Return the maximum gap in bytes between coalesced byte ranges.
    #[must_use]
    pub const fn max_gap(&self) -> u64 {
        self.max_gap
    }

    /// Return the maximum size in bytes of a coalesced byte range.
    #[must_use]
    pub const fn max_coalesced_bytes(&self) -> u64 {
        self.max_coalesced_bytes
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::byte_range::ByteRangeIterator;
    use crate::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
    use crate::store::MemoryStore;
    use crate::{MaybeBytesIterator, ReadableStorageTraits, StorageError, StorageHandle, StoreKey};

    /// A store with a read coalescing hint.
    struct CoalescingStore(MemoryStore);

    impl ReadableStorageTraits for CoalescingStore {
        fn get_partial_many<'a>(
            &'a self,
            key: &StoreKey,
            byte_ranges: ByteRangeIterator<'a>,
        ) -> Result<MaybeBytesIterator<'a>, StorageError> {
            self.0.get_partial_many(key, byte_ranges)
        }

        fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
            self.0.size_key(key)
        }

        fn supports_get_partial(&self) -> bool {
            true
        }

        fn read_coalescing(&self) -> Option<ReadCoalescing> {
            Some(ReadCoalescing::new(16, 1024))
        }
    }

    #[test]
    fn read_coalescing_is_forwarded_by_adapters() {
        assert_eq!(MemoryStore::new().read_coalescing(), None);
        let store = Arc::new(CoalescingStore(MemoryStore::new()));
        let hint = Some(ReadCoalescing::new(16, 1024));
        assert_eq!(StorageHandle::new(store.clone()).read_coalescing(), hint);
        assert_eq!(
            PerformanceMetricsStorageAdapter::new(store).read_coalescing(),
            hint
        );
    }
}
