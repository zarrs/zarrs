use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use super::byte_range::ByteRange;
use super::{ReadableStorageTraits, StoreKey};

#[cfg(feature = "async")]
mod r#async;

#[cfg(feature = "async")]
pub use r#async::AsyncStorageValueIO;

/// The storage key and seek position shared by the sync and async readers.
struct StorageValueCursor<TStorage: ?Sized> {
    storage: Arc<TStorage>,
    key: StoreKey,
    pos: u64,
    size: u64,
}

// Manual impl to avoid a `TStorage: Clone` bound, which `derive` would add.
impl<TStorage: ?Sized> Clone for StorageValueCursor<TStorage> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            key: self.key.clone(),
            pos: self.pos,
            size: self.size,
        }
    }
}

impl<TStorage: ?Sized> StorageValueCursor<TStorage> {
    fn new(storage: Arc<TStorage>, key: StoreKey, size: u64) -> Self {
        debug_assert!(size > 0);
        Self {
            storage,
            key,
            pos: 0,
            size,
        }
    }

    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        use std::io::{Error, ErrorKind};
        self.pos = match pos {
            SeekFrom::Start(offset) => offset,
            SeekFrom::Current(offset) => {
                let pos = i64::try_from(self.pos)
                    .map_err(|_| Error::from(ErrorKind::InvalidInput))?
                    + offset;
                u64::try_from(pos).map_err(|_| Error::from(ErrorKind::InvalidInput))?
            }
            SeekFrom::End(offset) => {
                let pos = i64::try_from(self.size)
                    .map_err(|_| Error::from(ErrorKind::InvalidInput))?
                    + offset;
                u64::try_from(pos).map_err(|_| Error::from(ErrorKind::InvalidInput))?
            }
        };
        Ok(self.pos)
    }

    /// The number of bytes to request, clamped to the end of the value.
    fn next_read_len(&self, requested: usize) -> usize {
        let remaining = self.size.saturating_sub(self.pos);
        let requested = u64::try_from(requested).unwrap_or(u64::MAX);
        usize::try_from(remaining.min(requested)).expect("bounded above by requested: usize")
    }

    /// Copy `data` into `buf` and advance the position, clamped in case a store returns
    /// more bytes than were requested.
    fn consume(&mut self, data: &[u8], buf: &mut [u8]) -> usize {
        let len = data.len().min(buf.len());
        buf[..len].copy_from_slice(&data[..len]);
        self.pos += len as u64;
        len
    }
}

/// Provides a [`Read`] interface to a storage value.
#[cfg_attr(
    feature = "async",
    doc = "\nSee [`AsyncStorageValueIO`] for the asynchronous equivalent."
)]
pub struct StorageValueIO<TStorage: ?Sized> {
    cursor: StorageValueCursor<TStorage>,
}

// Manual impl to avoid a `TStorage: Clone` bound, which `derive` would add.
impl<TStorage: ?Sized> Clone for StorageValueIO<TStorage> {
    fn clone(&self) -> Self {
        Self {
            cursor: self.cursor.clone(),
        }
    }
}

impl<TStorage: ?Sized> StorageValueIO<TStorage> {
    /// Create a new `StorageValueIO` for the `key` in `storage`.
    #[must_use]
    pub fn new(storage: Arc<TStorage>, key: StoreKey, size: u64) -> Self {
        Self {
            cursor: StorageValueCursor::new(storage, key, size),
        }
    }
}

impl<TStorage: ?Sized> Seek for StorageValueIO<TStorage> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.cursor.seek(pos)
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits> Read for StorageValueIO<TStorage> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let len = self.cursor.next_read_len(buf.len());
        if len == 0 {
            return Ok(0);
        }
        let data = self
            .cursor
            .storage
            .get_partial(
                &self.cursor.key,
                ByteRange::FromStart(self.cursor.pos, Some(len as u64)),
            )
            .map_err(|err| std::io::Error::other(err.to_string()))?;
        if let Some(data) = data {
            Ok(self.cursor.consume(&data, buf))
        } else {
            // This shouldn't happen, the data is only None if the key is not found. Which won't be the case if the size is known.
            Err(std::io::Error::other(
                "Failed to get partial values in StorageValueIO",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WritableStorageTraits;

    /// A store that returns the whole value regardless of the requested byte range.
    struct OverreadingStore;

    impl ReadableStorageTraits for OverreadingStore {
        fn get_partial_many<'a>(
            &'a self,
            _key: &StoreKey,
            _byte_ranges: crate::byte_range::ByteRangeIterator<'a>,
        ) -> Result<crate::MaybeBytesIterator<'a>, crate::StorageError> {
            unimplemented!()
        }

        fn get_partial(
            &self,
            _key: &StoreKey,
            _byte_range: ByteRange,
        ) -> Result<crate::MaybeBytes, crate::StorageError> {
            Ok(Some(crate::Bytes::from_static(b"0123456789")))
        }

        fn size_key(&self, _key: &StoreKey) -> Result<Option<u64>, crate::StorageError> {
            Ok(Some(10))
        }

        fn supports_get_partial(&self) -> bool {
            true
        }
    }

    #[test]
    fn read_clamps_oversized_store_response() {
        let key = StoreKey::new("value").unwrap();
        let mut reader = StorageValueIO::new(Arc::new(OverreadingStore), key, 10);

        let mut bytes = [0; 4];
        assert_eq!(reader.read(&mut bytes).unwrap(), 4);
        assert_eq!(&bytes, b"0123");
    }

    #[test]
    fn read_stops_at_end_of_value() {
        let store = Arc::new(crate::store::MemoryStore::new());
        let key = StoreKey::new("value").unwrap();
        store.set(&key, b"0123456789".as_slice().into()).unwrap();
        let mut reader = StorageValueIO::new(store, key, 10);
        std::io::Seek::seek(&mut reader, SeekFrom::End(-2)).unwrap();

        let mut bytes = [0; 4];
        assert_eq!(reader.read(&mut bytes).unwrap(), 2);
        assert_eq!(&bytes[..2], b"89");
        assert_eq!(reader.read(&mut bytes).unwrap(), 0);
    }
}
