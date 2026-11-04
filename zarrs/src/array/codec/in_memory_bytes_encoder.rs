use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use zarrs_codec::{BytesPartialDecoderTraits, BytesPartialEncoderTraits, CodecError, CodecOptions};
use zarrs_storage::byte_range::{ByteRangeIterator, InvalidByteRangeError};
use zarrs_storage::{OffsetBytesIterator, StorageError};

use crate::array::CowBytes;

/// Encoded bytes held in memory and stored whole once updated, tracking whether they have been written.
///
/// This does not report support for partial encoding, as every update rewrites the stored bytes entirely.
/// Partial encoders can therefore write compacted bytes rather than appending updates.
///
/// The bytes are shared rather than owned, so neither the existing bytes nor partial reads of them are copied.
pub(crate) struct InMemoryBytesEncoder {
    bytes: Mutex<Option<Bytes>>,
    written: AtomicBool,
}

impl InMemoryBytesEncoder {
    /// Create an in-memory encoder of the `existing` encoded bytes.
    pub(crate) fn new(existing: Option<Bytes>) -> Self {
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
    Stored(Bytes),
    /// The bytes were erased.
    Erased,
}

impl BytesPartialDecoderTraits for InMemoryBytesEncoder {
    fn exists(&self) -> Result<bool, StorageError> {
        Ok(self.bytes.lock().unwrap().is_some())
    }

    fn size_held(&self) -> usize {
        self.bytes.lock().unwrap().as_ref().map_or(0, Bytes::len)
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<CowBytes<'_>>>, CodecError> {
        // Clone the handle rather than holding the lock while slicing
        let Some(bytes) = self.bytes.lock().unwrap().clone() else {
            return Ok(None);
        };
        let mut output = Vec::with_capacity(decoded_regions.size_hint().0);
        for byte_range in decoded_regions {
            let range = byte_range
                .checked_range_usize(bytes.len())
                .ok_or_else(|| InvalidByteRangeError::new(byte_range, bytes.len() as u64))?;
            output.push(CowBytes::from(bytes.slice(range)));
        }
        Ok(Some(output))
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

impl BytesPartialEncoderTraits for InMemoryBytesEncoder {
    fn erase(&self) -> Result<(), CodecError> {
        self.written.store(true, Ordering::Relaxed);
        *self.bytes.lock().unwrap() = None;
        Ok(())
    }

    fn encode(&self, bytes: CowBytes<'_>, _options: &CodecOptions) -> Result<(), CodecError> {
        self.written.store(true, Ordering::Relaxed);
        *self.bytes.lock().unwrap() = Some(bytes.into_bytes());
        Ok(())
    }

    /// Write `offset_values` into the bytes.
    ///
    /// Partial encoders do not call this, because this does not report support for partial encoding, so copying the
    /// bytes if they are shared is acceptable.
    fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<CowBytes<'_>>,
        _options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.written.store(true, Ordering::Relaxed);
        let mut bytes = self.bytes.lock().unwrap();
        let mut output = bytes
            .take()
            .map(|bytes| CowBytes::from(bytes).into_vec())
            .unwrap_or_default();
        for (offset, value) in offset_values {
            let offset = usize::try_from(offset).unwrap();
            if output.len() < offset + value.len() {
                output.resize(offset + value.len(), 0);
            }
            output[offset..offset + value.len()].copy_from_slice(&value);
        }
        *bytes = Some(Bytes::from(output));
        Ok(())
    }

    fn supports_partial_encode(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use zarrs_storage::byte_range::ByteRange;

    use super::*;

    #[test]
    fn in_memory_bytes_encoder_shares_bytes() {
        let options = CodecOptions::default();
        let existing = Bytes::from(vec![1u8, 2, 3, 4, 5, 6]);
        let encoder = InMemoryBytesEncoder::new(Some(existing.clone()));

        // Partial reads slice the existing bytes
        let decoded = encoder
            .partial_decode(ByteRange::new(2..5), &options)
            .unwrap()
            .unwrap();
        assert_eq!(&*decoded, &[3, 4, 5]);
        assert_eq!(decoded.as_ptr(), existing[2..].as_ptr());
        assert!(
            encoder
                .partial_decode(ByteRange::new(4..7), &options)
                .is_err()
        );
        assert!(matches!(
            encoder.take_update(),
            InMemoryBytesUpdate::Unchanged
        ));

        // Encoded bytes are stored without a copy
        let updated = Bytes::from(vec![7u8; 3]);
        encoder
            .encode(CowBytes::from(updated.clone()), &options)
            .unwrap();
        let InMemoryBytesUpdate::Stored(stored) = encoder.take_update() else {
            panic!("the bytes were stored");
        };
        assert_eq!(stored.as_ptr(), updated.as_ptr());
    }

    #[test]
    fn in_memory_bytes_encoder_partial_encode() {
        let options = CodecOptions::default();
        let encoder = InMemoryBytesEncoder::new(Some(Bytes::from(vec![1u8, 2, 3])));
        encoder
            .partial_encode_many(
                Box::new(
                    [
                        (1, CowBytes::from(vec![8u8])),
                        (4, CowBytes::from(vec![9u8])),
                    ]
                    .into_iter(),
                ),
                &options,
            )
            .unwrap();
        let InMemoryBytesUpdate::Stored(stored) = encoder.take_update() else {
            panic!("the bytes were stored");
        };
        assert_eq!(&*stored, &[1, 8, 3, 0, 9]);

        encoder.erase().unwrap();
        assert!(matches!(encoder.take_update(), InMemoryBytesUpdate::Erased));
    }
}
