use std::future::Future;
use std::io::SeekFrom;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use super::StorageValueCursor;
use crate::byte_range::ByteRange;
use crate::{AsyncReadableStorageTraits, Bytes, MaybeBytes, StorageError, StoreKey};

#[cfg(not(target_arch = "wasm32"))]
type StorageReadFuture =
    Pin<Box<dyn Future<Output = Result<MaybeBytes, StorageError>> + Send + 'static>>;
#[cfg(target_arch = "wasm32")]
type StorageReadFuture = Pin<Box<dyn Future<Output = Result<MaybeBytes, StorageError>> + 'static>>;

enum AsyncReadState {
    /// No request is in flight and no data is buffered.
    Idle,
    /// A `get_partial` request is in flight.
    Pending(StorageReadFuture),
    /// Data has been fetched but not yet fully copied to the caller.
    Ready(Bytes),
}

/// Provides a [`futures::io::AsyncRead`] and [`futures::io::AsyncSeek`] interface to a storage value.
///
/// See [`StorageValueIO`](super::StorageValueIO) for the blocking equivalent.
pub struct AsyncStorageValueIO<TStorage: ?Sized> {
    cursor: StorageValueCursor<TStorage>,
    state: AsyncReadState,
}

impl<TStorage: ?Sized> Clone for AsyncStorageValueIO<TStorage> {
    /// Clone the reader, preserving the seek position.
    ///
    /// Any in-flight request or buffered data is not cloned, so the next read on the
    /// clone re-fetches from `pos`.
    fn clone(&self) -> Self {
        Self {
            cursor: self.cursor.clone(),
            state: AsyncReadState::Idle,
        }
    }
}

impl<TStorage: ?Sized> AsyncStorageValueIO<TStorage> {
    /// Create a new `AsyncStorageValueIO` for the `key` in `storage`.
    #[must_use]
    pub fn new(storage: Arc<TStorage>, key: StoreKey, size: u64) -> Self {
        Self {
            cursor: StorageValueCursor::new(storage, key, size),
            state: AsyncReadState::Idle,
        }
    }
}

impl<TStorage: ?Sized> futures::io::AsyncSeek for AsyncStorageValueIO<TStorage> {
    fn poll_seek(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        pos: SeekFrom,
    ) -> Poll<std::io::Result<u64>> {
        // Discard any in-flight request or buffered data, which is for the old position.
        self.state = AsyncReadState::Idle;
        Poll::Ready(self.cursor.seek(pos))
    }
}

impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> futures::io::AsyncRead
    for AsyncStorageValueIO<TStorage>
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        loop {
            match &mut self.state {
                AsyncReadState::Idle => {
                    let len = self.cursor.next_read_len(buf.len());
                    if len == 0 {
                        return Poll::Ready(Ok(0));
                    }
                    let storage = self.cursor.storage.clone();
                    let key = self.cursor.key.clone();
                    let pos = self.cursor.pos;
                    self.state = AsyncReadState::Pending(Box::pin(async move {
                        storage
                            .get_partial(&key, ByteRange::FromStart(pos, Some(len as u64)))
                            .await
                    }));
                }
                AsyncReadState::Pending(future) => match future.as_mut().poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Ok(Some(data))) => {
                        self.state = AsyncReadState::Ready(data);
                    }
                    Poll::Ready(Ok(None)) => {
                        self.state = AsyncReadState::Idle;
                        return Poll::Ready(Err(std::io::Error::other(
                            "Failed to get partial values in AsyncStorageValueIO",
                        )));
                    }
                    Poll::Ready(Err(error)) => {
                        self.state = AsyncReadState::Idle;
                        return Poll::Ready(Err(std::io::Error::other(error.to_string())));
                    }
                },
                AsyncReadState::Ready(data) => {
                    let data = std::mem::take(data);
                    let len = self.cursor.consume(&data, buf);
                    let remaining = data.slice(len..);
                    self.state = if remaining.is_empty() {
                        AsyncReadState::Idle
                    } else {
                        AsyncReadState::Ready(remaining)
                    };
                    return Poll::Ready(Ok(len));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AsyncWritableStorageTraits;
    use futures::io::{AsyncReadExt, AsyncSeekExt};

    #[test]
    fn async_read_and_seek() {
        futures::executor::block_on(async {
            let store = Arc::new(crate::store::AsyncMemoryStore::new());
            let key = StoreKey::new("value").unwrap();
            store
                .set(&key, b"0123456789".as_slice().into())
                .await
                .unwrap();
            let mut reader = AsyncStorageValueIO::new(store, key, 10);
            let mut bytes = [0; 4];
            assert_eq!(reader.read(&mut bytes).await.unwrap(), 4);
            assert_eq!(&bytes, b"0123");
            reader.seek(SeekFrom::End(-2)).await.unwrap();
            assert_eq!(reader.read(&mut bytes).await.unwrap(), 2);
            assert_eq!(&bytes[..2], b"89");
            assert_eq!(reader.read(&mut bytes).await.unwrap(), 0);
        });
    }

    /// A store that returns everything from the requested offset to the end of the value,
    /// ignoring the requested length. A small read therefore leaves data buffered.
    struct OverreadingStore;

    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    impl AsyncReadableStorageTraits for OverreadingStore {
        async fn get_partial_many<'a>(
            &'a self,
            _key: &StoreKey,
            _byte_ranges: crate::byte_range::ByteRangeIterator<'a>,
        ) -> Result<crate::AsyncMaybeBytesIterator<'a>, StorageError> {
            unimplemented!()
        }

        async fn get_partial<'a>(
            &'a self,
            _key: &StoreKey,
            byte_range: ByteRange,
        ) -> Result<MaybeBytes, StorageError> {
            let value = b"0123456789";
            let offset = usize::try_from(byte_range.start(value.len() as u64)).unwrap();
            Ok(Some(Bytes::from_static(value).slice(offset..)))
        }

        async fn size_key(&self, _key: &StoreKey) -> Result<Option<u64>, StorageError> {
            Ok(Some(10))
        }

        fn supports_get_partial(&self) -> bool {
            true
        }
    }

    /// A seek must discard data buffered for the previous position.
    #[test]
    fn async_seek_invalidates_buffered_data() {
        futures::executor::block_on(async {
            let key = StoreKey::new("value").unwrap();
            let mut reader = AsyncStorageValueIO::new(Arc::new(OverreadingStore), key, 10);

            // The store returns "0123456789", so this leaves "23456789" buffered.
            let mut small = [0; 2];
            assert_eq!(reader.read(&mut small).await.unwrap(), 2);
            assert_eq!(&small, b"01");

            // Seeking must drop that buffer. Serving the stale buffer would yield "23".
            reader.seek(SeekFrom::Start(5)).await.unwrap();
            assert_eq!(reader.read(&mut small).await.unwrap(), 2);
            assert_eq!(&small, b"56");
        });
    }

    /// The reader remains usable after a storage error.
    #[test]
    fn async_read_after_missing_key_error() {
        futures::executor::block_on(async {
            let store = Arc::new(crate::store::AsyncMemoryStore::new());
            let key = StoreKey::new("missing").unwrap();
            let mut reader = AsyncStorageValueIO::new(store.clone(), key.clone(), 10);

            let mut bytes = [0; 4];
            assert!(reader.read(&mut bytes).await.is_err());

            // The state was reset to idle, so a retry after the key appears succeeds.
            store
                .set(&key, b"0123456789".as_slice().into())
                .await
                .unwrap();
            assert_eq!(reader.read(&mut bytes).await.unwrap(), 4);
            assert_eq!(&bytes, b"0123");
        });
    }
}
