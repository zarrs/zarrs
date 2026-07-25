use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;

use super::byte_range::ByteRange;
use super::{ReadableStorageTraits, StoreKey};

/// Provides a [`Read`] interface to a storage value.
#[derive(Clone)]
pub struct StorageValueIO<TStorage: ?Sized> {
    storage: Arc<TStorage>,
    key: StoreKey,
    pos: u64,
    size: u64,
}

impl<TStorage: ?Sized + ReadableStorageTraits> StorageValueIO<TStorage> {
    /// Create a new `StorageValueIO` for the `key` in `storage`.
    pub fn new(storage: Arc<TStorage>, key: StoreKey, size: u64) -> Self {
        debug_assert!(size > 0);
        Self {
            storage,
            key,
            pos: 0,
            size,
        }
    }
}

// #[cfg(feature = "async")]
// impl<TStorage: ?Sized + super::AsyncReadableStorageTraits> StorageValueIO<TStorage> {
//     /// Create a new `StorageValueIO` for the `key` in `storage`.
//     pub fn async_new(storage: Arc<TStorage>, key: StoreKey, size: u64) -> Self {
//         debug_assert!(size > 0);
//         Self {
//             storage,
//             key,
//             pos: 0,
//             size,
//         }
//     }
// }

impl<TStorage: ?Sized> Seek for StorageValueIO<TStorage> {
    fn seek(&mut self, pos: std::io::SeekFrom) -> std::io::Result<u64> {
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
}

impl<TStorage: ?Sized + ReadableStorageTraits> Read for StorageValueIO<TStorage> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let len =
            usize::try_from((self.size.saturating_sub(self.pos)).min(buf.len() as u64)).unwrap();
        if len == 0 {
            return Ok(0);
        }
        let data = self
            .storage
            .get_partial(&self.key, ByteRange::FromStart(self.pos, Some(len as u64)))
            .map_err(|err| std::io::Error::other(err.to_string()))?;
        if let Some(data) = data {
            buf[..data.len()].copy_from_slice(&data);
            self.pos += data.len() as u64;
            Ok(data.len())
        } else {
            // This shouldn't happen, the data is only None if the key is not found. Which won't be the case if the size is known.
            Err(std::io::Error::other(
                "Failed to get partial values in StorageValueIO",
            ))
        }
    }
}

// TODO: AsyncRead

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WritableStorageTraits;

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
