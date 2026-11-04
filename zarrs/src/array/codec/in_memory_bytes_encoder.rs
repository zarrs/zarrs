use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use zarrs_codec::{BytesPartialDecoderTraits, BytesPartialEncoderTraits, CodecError, CodecOptions};
use zarrs_storage::byte_range::ByteRangeIterator;
use zarrs_storage::{OffsetBytesIterator, StorageError};

use crate::array::CowBytes;

/// Encoded bytes held in memory and stored whole once updated, tracking whether they have been written.
///
/// This does not report support for partial encoding, as every update rewrites the stored bytes entirely.
/// Partial encoders can therefore write compacted bytes rather than appending updates.
pub(crate) struct InMemoryBytesEncoder {
    bytes: Mutex<Option<Vec<u8>>>,
    written: AtomicBool,
}

impl InMemoryBytesEncoder {
    /// Create an in-memory encoder of the `existing` encoded bytes.
    pub(crate) fn new(existing: Option<Vec<u8>>) -> Self {
        Self {
            bytes: Mutex::new(existing),
            written: AtomicBool::new(false),
        }
    }

    /// Take the update of the bytes.
    pub(crate) fn take_update(&self) -> InMemoryBytesUpdate {
        if !self.written.load(Ordering::Relaxed) {
            InMemoryBytesUpdate::Unchanged
        } else if let Some(bytes) = self.bytes.lock().unwrap().take() {
            InMemoryBytesUpdate::Stored(bytes)
        } else {
            InMemoryBytesUpdate::Erased
        }
    }
}

/// An update of the bytes of an [`InMemoryBytesEncoder`].
pub(crate) enum InMemoryBytesUpdate {
    /// The bytes were not written.
    Unchanged,
    /// The bytes were written.
    Stored(Vec<u8>),
    /// The bytes were erased.
    Erased,
}

impl BytesPartialDecoderTraits for InMemoryBytesEncoder {
    fn exists(&self) -> Result<bool, StorageError> {
        self.bytes.exists()
    }

    fn size_held(&self) -> usize {
        self.bytes.size_held()
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<CowBytes<'_>>>, CodecError> {
        self.bytes.partial_decode_many(decoded_regions, options)
    }

    fn supports_partial_decode(&self) -> bool {
        self.bytes.supports_partial_decode()
    }
}

impl BytesPartialEncoderTraits for InMemoryBytesEncoder {
    fn erase(&self) -> Result<(), CodecError> {
        self.written.store(true, Ordering::Relaxed);
        self.bytes.erase()
    }

    fn encode(&self, bytes: CowBytes<'_>, options: &CodecOptions) -> Result<(), CodecError> {
        self.written.store(true, Ordering::Relaxed);
        self.bytes.encode(bytes, options)
    }

    fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<CowBytes<'_>>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.written.store(true, Ordering::Relaxed);
        self.bytes.partial_encode_many(offset_values, options)
    }

    fn supports_partial_encode(&self) -> bool {
        false
    }
}
