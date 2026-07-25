use std::io::{Read, Seek, SeekFrom};
use std::sync::Arc;
#[cfg(feature = "async")]
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use super::byte_range::ByteRange;
#[cfg(feature = "async")]
use super::{AsyncReadableStorageTraits, Bytes, MaybeBytes, StorageError};
use super::{ReadableStorageTraits, StoreKey};

#[cfg(all(feature = "async", not(target_arch = "wasm32")))]
type StorageReadFuture =
    Pin<Box<dyn Future<Output = Result<MaybeBytes, StorageError>> + Send + 'static>>;
#[cfg(all(feature = "async", target_arch = "wasm32"))]
type StorageReadFuture = Pin<Box<dyn Future<Output = Result<MaybeBytes, StorageError>> + 'static>>;

#[cfg(feature = "async")]
enum AsyncReadState {
    Idle,
    Pending(StorageReadFuture),
    Ready(Bytes),
}

/// Provides a [`Read`] or [`futures::io::AsyncRead`] interface to a storage value.
pub struct StorageValueIO<TStorage: ?Sized> {
    storage: Arc<TStorage>,
    key: StoreKey,
    pos: u64,
    size: u64,
    #[cfg(feature = "async")]
    async_read_state: AsyncReadState,
}

impl<TStorage: ?Sized> Clone for StorageValueIO<TStorage> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            key: self.key.clone(),
            pos: self.pos,
            size: self.size,
            #[cfg(feature = "async")]
            async_read_state: AsyncReadState::Idle,
        }
    }
}

impl<TStorage: ?Sized> StorageValueIO<TStorage> {
    /// Create a new `StorageValueIO` for the `key` in `storage`.
    #[must_use]
    pub fn new(storage: Arc<TStorage>, key: StoreKey, size: u64) -> Self {
        debug_assert!(size > 0);
        Self {
            storage,
            key,
            pos: 0,
            size,
            #[cfg(feature = "async")]
            async_read_state: AsyncReadState::Idle,
        }
    }

    fn seek_impl(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
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
        #[cfg(feature = "async")]
        {
            self.async_read_state = AsyncReadState::Idle;
        }
        Ok(self.pos)
    }

    fn next_read_len(&self, requested: usize) -> usize {
        usize::try_from((self.size.saturating_sub(self.pos)).min(requested as u64)).unwrap()
    }
}

impl<TStorage: ?Sized> Seek for StorageValueIO<TStorage> {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        self.seek_impl(pos)
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits> Read for StorageValueIO<TStorage> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        #[cfg(feature = "async")]
        {
            self.async_read_state = AsyncReadState::Idle;
        }
        let len = self.next_read_len(buf.len());
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
            Err(std::io::Error::other(
                "Failed to get partial values in StorageValueIO",
            ))
        }
    }
}

#[cfg(feature = "async")]
impl<TStorage: ?Sized> futures::io::AsyncSeek for StorageValueIO<TStorage> {
    fn poll_seek(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        pos: SeekFrom,
    ) -> Poll<std::io::Result<u64>> {
        Poll::Ready(self.seek_impl(pos))
    }
}

#[cfg(feature = "async")]
impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> futures::io::AsyncRead
    for StorageValueIO<TStorage>
{
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        loop {
            match &mut self.async_read_state {
                AsyncReadState::Idle => {
                    let len = self.next_read_len(buf.len());
                    if len == 0 {
                        return Poll::Ready(Ok(0));
                    }
                    let storage = self.storage.clone();
                    let key = self.key.clone();
                    let pos = self.pos;
                    self.async_read_state = AsyncReadState::Pending(Box::pin(async move {
                        storage
                            .get_partial(&key, ByteRange::FromStart(pos, Some(len as u64)))
                            .await
                    }));
                }
                AsyncReadState::Pending(future) => match future.as_mut().poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Ok(Some(data))) => {
                        self.async_read_state = AsyncReadState::Ready(data);
                    }
                    Poll::Ready(Ok(None)) => {
                        self.async_read_state = AsyncReadState::Idle;
                        return Poll::Ready(Err(std::io::Error::other(
                            "Failed to get partial values in StorageValueIO",
                        )));
                    }
                    Poll::Ready(Err(error)) => {
                        self.async_read_state = AsyncReadState::Idle;
                        return Poll::Ready(Err(std::io::Error::other(error.to_string())));
                    }
                },
                AsyncReadState::Ready(data) => {
                    let len = data.len().min(buf.len());
                    buf[..len].copy_from_slice(&data[..len]);
                    let remaining = data.slice(len..);
                    self.pos += len as u64;
                    self.async_read_state = if remaining.is_empty() {
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
    use crate::storage_adapter::sync_to_async::{
        SyncToAsyncSpawnBlocking, SyncToAsyncStorageAdapter,
    };
    use crate::WritableStorageTraits;
    #[cfg(feature = "async")]
    use futures::io::{AsyncReadExt, AsyncSeekExt};

    #[cfg(feature = "async")]
    struct InlineSpawnBlocking;

    #[cfg(feature = "async")]
    impl SyncToAsyncSpawnBlocking for InlineSpawnBlocking {
        async fn spawn_blocking<F, R>(&self, f: F) -> R
        where
            F: FnOnce() -> R + Send + 'static,
            R: Send + 'static,
        {
            f()
        }
    }

    fn store() -> (Arc<crate::store::MemoryStore>, StoreKey) {
        let store = Arc::new(crate::store::MemoryStore::new());
        let key = StoreKey::new("value").unwrap();
        store.set(&key, b"0123456789".as_slice().into()).unwrap();
        (store, key)
    }

    #[test]
    fn read_and_seek() {
        let (store, key) = store();
        let mut reader = StorageValueIO::new(store, key, 10);
        let mut bytes = [0; 4];
        assert_eq!(reader.read(&mut bytes).unwrap(), 4);
        assert_eq!(&bytes, b"0123");
        std::io::Seek::seek(&mut reader, SeekFrom::End(-2)).unwrap();
        assert_eq!(reader.read(&mut bytes).unwrap(), 2);
        assert_eq!(&bytes[..2], b"89");
        assert_eq!(reader.read(&mut bytes).unwrap(), 0);
    }

    #[cfg(feature = "async")]
    #[test]
    fn async_read_and_seek() {
        futures::executor::block_on(async {
            let (store, key) = store();
            let store = Arc::new(SyncToAsyncStorageAdapter::new(store, InlineSpawnBlocking));
            let mut reader = StorageValueIO::new(store, key, 10);
            let mut bytes = [0; 4];
            assert_eq!(reader.read(&mut bytes).await.unwrap(), 4);
            assert_eq!(&bytes, b"0123");
            AsyncSeekExt::seek(&mut reader, SeekFrom::End(-2))
                .await
                .unwrap();
            assert_eq!(reader.read(&mut bytes).await.unwrap(), 2);
            assert_eq!(&bytes[..2], b"89");
            assert_eq!(reader.read(&mut bytes).await.unwrap(), 0);
        });
    }
}
