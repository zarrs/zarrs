use std::sync::Arc;

use zarrs_storage::{
    ListableStorage, ReadableListableStorage, ReadableStorage, ReadableWritableListableStorage,
    ReadableWritableStorage, StorageError, WritableStorage,
};

/// A single resolved stage of a synchronous URL pipeline (a root store or the output of an
/// adapter).
///
/// A pipeline stage may not support every storage capability: for example, an adapter over a
/// read-only archive format (e.g. zip) can only ever produce a readable/listable stage, so its
/// `as_writable`/`as_readable_writable`/etc. implementations return
/// [`StorageError::Unsupported`].
///
/// This trait is entirely separate from [`AsyncPipelineStageTraits`]: a sync pipeline and an
/// async pipeline are resolved independently.
pub trait PipelineStageTraits: core::fmt::Debug + Send + Sync {
    /// Returns this stage as readable storage.
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] if this stage is not readable.
    fn as_readable(self: Arc<Self>) -> Result<ReadableStorage, StorageError>;

    /// Returns this stage as writable storage.
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] if this stage is not writable.
    fn as_writable(self: Arc<Self>) -> Result<WritableStorage, StorageError>;

    /// Returns this stage as listable storage.
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] if this stage is not listable.
    fn as_listable(self: Arc<Self>) -> Result<ListableStorage, StorageError>;

    /// Returns this stage as readable and writable storage.
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] if this stage is not both readable and writable.
    fn as_readable_writable(self: Arc<Self>) -> Result<ReadableWritableStorage, StorageError>;

    /// Returns this stage as readable and listable storage.
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] if this stage is not both readable and listable.
    fn as_readable_listable(self: Arc<Self>) -> Result<ReadableListableStorage, StorageError>;

    /// Returns this stage as readable, writable, and listable storage.
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] if this stage does not support all three capabilities.
    fn as_readable_writable_listable(
        self: Arc<Self>,
    ) -> Result<ReadableWritableListableStorage, StorageError>;
}

/// An [`Arc`] wrapped synchronous pipeline stage.
pub type PipelineStage = Arc<dyn PipelineStageTraits>;

/// A single resolved stage of an asynchronous URL pipeline (a root store or the output of an
/// adapter).
///
/// The asynchronous counterpart of [`PipelineStageTraits`]; see its documentation for the
/// capability model.
/// Only compiled when this crate's `async` feature is enabled.
#[cfg(feature = "async")]
pub trait AsyncPipelineStageTraits: core::fmt::Debug + Send + Sync {
    /// Returns this stage as asynchronous readable storage.
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] if this stage is not asynchronously readable.
    fn as_async_readable(
        self: Arc<Self>,
    ) -> Result<zarrs_storage::AsyncReadableStorage, StorageError>;

    /// Returns this stage as asynchronous writable storage.
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] if this stage is not asynchronously writable.
    fn as_async_writable(
        self: Arc<Self>,
    ) -> Result<zarrs_storage::AsyncWritableStorage, StorageError>;

    /// Returns this stage as asynchronous listable storage.
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] if this stage is not asynchronously listable.
    fn as_async_listable(
        self: Arc<Self>,
    ) -> Result<zarrs_storage::AsyncListableStorage, StorageError>;

    /// Returns this stage as asynchronous readable and writable storage.
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] if this stage is not both asynchronously readable and writable.
    fn as_async_readable_writable(
        self: Arc<Self>,
    ) -> Result<zarrs_storage::AsyncReadableWritableStorage, StorageError>;

    /// Returns this stage as asynchronous readable and listable storage.
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] if this stage is not both asynchronously readable and listable.
    fn as_async_readable_listable(
        self: Arc<Self>,
    ) -> Result<zarrs_storage::AsyncReadableListableStorage, StorageError>;

    /// Returns this stage as asynchronous readable, writable, and listable storage.
    ///
    /// # Errors
    /// Returns [`StorageError::Unsupported`] if this stage does not support all three asynchronous capabilities.
    fn as_async_readable_writable_listable(
        self: Arc<Self>,
    ) -> Result<zarrs_storage::AsyncReadableWritableListableStorage, StorageError>;
}

/// An [`Arc`] wrapped asynchronous pipeline stage.
#[cfg(feature = "async")]
pub type AsyncPipelineStage = Arc<dyn AsyncPipelineStageTraits>;

/// A convenience constructor for the [`StorageError::Unsupported`] a [`PipelineStageTraits`] or
/// [`AsyncPipelineStageTraits`] implementation should return for a capability it does not
/// implement.
#[must_use]
pub fn unsupported(capability: &str) -> StorageError {
    StorageError::Unsupported(format!("this pipeline stage does not support {capability}"))
}
