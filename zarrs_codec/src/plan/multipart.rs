//! The multipart writes in progress of an I/O backend.

use std::sync::Mutex;

use zarrs_storage::{StorageError, StoreWriter};

use super::{MultipartId, MultipartOp, SourceKey};

/// The multipart writes in progress of an I/O backend, each with its writer.
///
/// An I/O backend of a store performs a [`WriteOp::Multipart`](super::WriteOp::Multipart) with
/// [`write`](MultipartWriters::write), which begins a write with the writer of the store
/// (e.g. [`WritableStorageTraits::set_multipart`](zarrs_storage::WritableStorageTraits::set_multipart)) and performs
/// later steps with it.
///
/// The steps of a write are performed in order, so a writer is taken from the table while a step is performed rather
/// than locked, and the steps of different writes are performed concurrently.
/// Writes that are not finished when the table is dropped (e.g. a drive failed) are dropped, which aborts them on a
/// best effort basis; [`abort_all`](MultipartWriters::abort_all) aborts them and reports errors.
pub struct MultipartWriters<W: ?Sized> {
    writers: Mutex<Vec<(MultipartId, Option<Box<W>>)>>,
}

impl<W: ?Sized> Default for MultipartWriters<W> {
    fn default() -> Self {
        Self {
            writers: Mutex::default(),
        }
    }
}

impl<W: ?Sized> std::fmt::Debug for MultipartWriters<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ids: Vec<_> = self.lock().iter().map(|(id, _)| *id).collect();
        f.debug_struct("MultipartWriters")
            .field("writes", &ids)
            .finish()
    }
}

fn unknown(id: MultipartId) -> StorageError {
    StorageError::Other(format!("the multipart write {id:?} is not in progress"))
}

impl<W: ?Sized> MultipartWriters<W> {
    /// Create an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<(MultipartId, Option<Box<W>>)>> {
        self.writers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// The number of writes in progress.
    #[must_use]
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Returns true if no writes are in progress.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    fn insert(&self, id: MultipartId, writer: Box<W>) -> Result<(), StorageError> {
        let mut writers = self.lock();
        if writers.iter().any(|(other, _)| *other == id) {
            return Err(StorageError::Other(format!(
                "the multipart write {id:?} has already begun"
            )));
        }
        writers.push((id, Some(writer)));
        Ok(())
    }

    /// Take the writer of `id` while a step is performed with it.
    fn take(&self, id: MultipartId) -> Result<Box<W>, StorageError> {
        self.lock()
            .iter_mut()
            .find(|(other, _)| *other == id)
            .and_then(|(_, writer)| writer.take())
            .ok_or_else(|| unknown(id))
    }

    /// Return the writer of `id` taken with [`take`](Self::take).
    fn restore(&self, id: MultipartId, writer: Box<W>) {
        if let Some((_, slot)) = self.lock().iter_mut().find(|(other, _)| *other == id) {
            *slot = Some(writer);
        }
    }

    /// Remove the writer of `id` to finish or abort it.
    fn remove(&self, id: MultipartId) -> Result<Box<W>, StorageError> {
        let mut writers = self.lock();
        let index = writers
            .iter()
            .position(|(other, _)| *other == id)
            .ok_or_else(|| unknown(id))?;
        writers.swap_remove(index).1.ok_or_else(|| unknown(id))
    }

    /// Remove all writers.
    fn drain(&self) -> Vec<Box<W>> {
        self.lock()
            .drain(..)
            .filter_map(|(_, writer)| writer)
            .collect()
    }
}

impl<'s> MultipartWriters<dyn StoreWriter + 's> {
    /// Perform the step `op` of the multipart write `id`, beginning it with `begin`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if the step fails, or the write is not in progress (or has already begun).
    pub fn write(
        &self,
        id: MultipartId,
        op: MultipartOp<'_>,
        begin: impl FnOnce(SourceKey, Option<u64>) -> Result<Box<dyn StoreWriter + 's>, StorageError>,
    ) -> Result<(), StorageError> {
        match op {
            MultipartOp::Begin(key, size_hint) => self.insert(id, begin(key, size_hint)?),
            MultipartOp::Parts(parts) => {
                let mut writer = self.take(id)?;
                let result = writer.write_many(parts);
                self.restore(id, writer);
                result
            }
            MultipartOp::Finish => self.remove(id)?.finish(),
            MultipartOp::Abort => self.remove(id)?.abort(),
        }
    }

    /// Abort all writes in progress, returning the first error.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if aborting a write fails. The other writes are still aborted.
    pub fn abort_all(&self) -> Result<(), StorageError> {
        self.drain()
            .into_iter()
            .map(StoreWriter::abort)
            .fold(Ok(()), Result::and)
    }
}

#[cfg(feature = "async")]
impl<'s> MultipartWriters<dyn zarrs_storage::AsyncStoreWriter + 's> {
    /// Perform the step `op` of the multipart write `id`, beginning it with `begin`.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if the step fails, or the write is not in progress (or has already begun).
    pub async fn write_async<F>(
        &self,
        id: MultipartId,
        op: MultipartOp<'_>,
        begin: impl FnOnce(SourceKey, Option<u64>) -> F,
    ) -> Result<(), StorageError>
    where
        F: Future<Output = Result<Box<dyn zarrs_storage::AsyncStoreWriter + 's>, StorageError>>,
    {
        match op {
            MultipartOp::Begin(key, size_hint) => self.insert(id, begin(key, size_hint).await?),
            MultipartOp::Parts(parts) => {
                let mut writer = self.take(id)?;
                let result = writer.write_many(parts).await;
                self.restore(id, writer);
                result
            }
            MultipartOp::Finish => self.remove(id)?.finish().await,
            MultipartOp::Abort => self.remove(id)?.abort().await,
        }
    }

    /// Abort all writes in progress, returning the first error.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if aborting a write fails. The other writes are still aborted.
    pub async fn abort_all_async(&self) -> Result<(), StorageError> {
        let mut result = Ok(());
        for writer in self.drain() {
            result = result.and(writer.abort().await);
        }
        result
    }
}

/// Multipart writes in progress whose parts are collected in memory, for an I/O backend that sets a value whole.
#[derive(Debug, Default)]
pub struct MultipartBuffers {
    buffers: Mutex<Vec<(MultipartId, SourceKey, Vec<u8>)>>,
}

impl MultipartBuffers {
    /// Create an empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Perform the step `op` of the multipart write `id`.
    ///
    /// Returns the key and value of a finished write, which the backend sets.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if the write is not in progress (or has already begun).
    pub fn write(
        &self,
        id: MultipartId,
        op: MultipartOp<'_>,
    ) -> Result<Option<(SourceKey, Vec<u8>)>, StorageError> {
        let mut buffers = self
            .buffers
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let position = buffers.iter().position(|(other, ..)| *other == id);
        match (op, position) {
            (MultipartOp::Begin(key, size_hint), None) => {
                let capacity = size_hint.map_or(0, |size| usize::try_from(size).unwrap_or(0));
                buffers.push((id, key, Vec::with_capacity(capacity)));
                Ok(None)
            }
            (MultipartOp::Begin(..), Some(_)) => Err(StorageError::Other(format!(
                "the multipart write {id:?} has already begun"
            ))),
            (MultipartOp::Parts(parts), Some(index)) => {
                let buffer = &mut buffers[index].2;
                for part in parts {
                    buffer.extend_from_slice(&part);
                }
                Ok(None)
            }
            (MultipartOp::Finish, Some(index)) => {
                let (_, key, value) = buffers.swap_remove(index);
                Ok(Some((key, value)))
            }
            (MultipartOp::Abort, Some(index)) => {
                buffers.swap_remove(index);
                Ok(None)
            }
            (_, None) => Err(unknown(id)),
        }
    }
}

#[cfg(test)]
mod tests {
    use zarrs_storage::store::MemoryStore;
    use zarrs_storage::{ReadableStorageTraits, StoreKey, WritableStorageTraits};

    use super::*;
    use crate::CowBytes;

    #[test]
    fn multipart_writers() {
        let store = MemoryStore::new();
        let writers = MultipartWriters::<dyn StoreWriter>::new();
        let begin = |key: SourceKey, size_hint| {
            store.set_multipart(&StoreKey::new(format!("{}", key.0)).unwrap(), size_hint)
        };
        let (a, b) = (MultipartId::unique(), MultipartId::unique());
        assert_ne!(a, b);
        writers
            .write(a, MultipartOp::Begin(SourceKey(0), Some(4)), begin)
            .unwrap();
        writers
            .write(b, MultipartOp::Begin(SourceKey(1), None), begin)
            .unwrap();
        assert!(
            writers
                .write(a, MultipartOp::Begin(SourceKey(0), None), begin)
                .is_err()
        );
        writers
            .write(
                a,
                MultipartOp::Parts(vec![CowBytes::from(vec![1, 2])]),
                begin,
            )
            .unwrap();
        writers
            .write(b, MultipartOp::Parts(vec![vec![9].into()]), begin)
            .unwrap();
        writers
            .write(
                a,
                MultipartOp::Parts(vec![vec![3].into(), vec![4].into()]),
                begin,
            )
            .unwrap();
        writers.write(a, MultipartOp::Finish, begin).unwrap();
        writers.write(b, MultipartOp::Abort, begin).unwrap();
        assert!(writers.is_empty());
        assert!(writers.write(a, MultipartOp::Finish, begin).is_err());
        assert!(writers.write(a, MultipartOp::Parts(vec![]), begin).is_err());
        assert_eq!(
            store.get(&StoreKey::new("0").unwrap()).unwrap().unwrap(),
            vec![1, 2, 3, 4]
        );
        assert!(store.get(&StoreKey::new("1").unwrap()).unwrap().is_none());

        // Abort all
        writers
            .write(a, MultipartOp::Begin(SourceKey(2), None), begin)
            .unwrap();
        writers
            .write(a, MultipartOp::Parts(vec![vec![1].into()]), begin)
            .unwrap();
        assert_eq!(writers.len(), 1);
        writers.abort_all().unwrap();
        assert!(writers.is_empty());
        assert!(store.get(&StoreKey::new("2").unwrap()).unwrap().is_none());
    }

    #[test]
    fn multipart_buffers() {
        let buffers = MultipartBuffers::new();
        let (a, b) = (MultipartId::unique(), MultipartId::unique());
        assert!(
            buffers
                .write(a, MultipartOp::Begin(SourceKey(3), Some(3)))
                .unwrap()
                .is_none()
        );
        buffers
            .write(b, MultipartOp::Begin(SourceKey(4), None))
            .unwrap();
        buffers
            .write(
                a,
                MultipartOp::Parts(vec![vec![1].into(), vec![2, 3].into()]),
            )
            .unwrap();
        assert!(buffers.write(b, MultipartOp::Abort).unwrap().is_none());
        assert_eq!(
            buffers.write(a, MultipartOp::Finish).unwrap(),
            Some((SourceKey(3), vec![1, 2, 3]))
        );
        assert!(buffers.write(a, MultipartOp::Finish).is_err());
    }
}
