//! I/O backends of drives.

use smallvec::SmallVec;
use zarrs_storage::byte_range::ByteRange;
use zarrs_storage::{Bytes, ReadCoalescing, StorageError};

use super::{SourceKey, WriteOp};

/// The bytes of a read of byte ranges, one per range.
pub type ReadBytes = SmallVec<[Bytes; 1]>;

/// A synchronous I/O backend of a drive.
///
/// It resolves the [`SourceKey`]s of tasks to storage (e.g. store keys of chunks).
pub trait SyncIo: zarrs_plugin::MaybeSend + zarrs_plugin::MaybeSync {
    /// Read `ranges` of the value of `key`.
    ///
    /// Returns [`None`] if the value does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if the read fails.
    fn read(&self, key: SourceKey, ranges: &[ByteRange])
    -> Result<Option<ReadBytes>, StorageError>;

    /// Returns how reads of nearby byte ranges of the same key should be coalesced into one request, if at all.
    ///
    /// The default is [`None`]. Backends of stores return the hint of the store
    /// ([`ReadableStorageTraits::read_coalescing`](zarrs_storage::ReadableStorageTraits::read_coalescing)).
    fn read_coalescing(&self) -> Option<ReadCoalescing> {
        None
    }

    /// Perform a write operation.
    ///
    /// The default implementation returns an error, for backends that are read-only.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if the write fails.
    fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
        let _ = op;
        Err(StorageError::ReadOnly)
    }
}

/// An asynchronous I/O backend of a drive.
///
/// It resolves the [`SourceKey`]s of tasks to storage (e.g. store keys of chunks).
#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncIo: zarrs_plugin::MaybeSend + zarrs_plugin::MaybeSync {
    /// Read `ranges` of the value of `key`.
    ///
    /// Returns [`None`] if the value does not exist.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if the read fails.
    async fn read(
        &self,
        key: SourceKey,
        ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError>;

    /// Returns how reads of nearby byte ranges of the same key should be coalesced into one request, if at all.
    ///
    /// The default is [`None`]. Backends of stores return the hint of the store
    /// ([`ReadableStorageTraits::read_coalescing`](zarrs_storage::ReadableStorageTraits::read_coalescing)).
    fn read_coalescing(&self) -> Option<ReadCoalescing> {
        None
    }

    /// Perform a write operation.
    ///
    /// The default implementation returns an error, for backends that are read-only.
    ///
    /// # Errors
    /// Returns a [`StorageError`] if the write fails.
    async fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
        let _ = op;
        Err(StorageError::ReadOnly)
    }
}
