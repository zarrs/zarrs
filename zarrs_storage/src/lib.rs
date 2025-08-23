//! The storage API for the [`zarrs`](https://docs.rs/zarrs/latest/zarrs/index.html) crate.
//!
//! A Zarr store is a system that can be used to store and retrieve data from a Zarr hierarchy.
//! For example: a filesystem, HTTP server, FTP server, Amazon S3 bucket, ZIP file, etc.
//! The Zarr V3 storage API is detailed here: <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#storage>.
//!
//! This crate includes an in-memory store implementation. See [`zarrs` storage support](https://docs.rs/zarrs/latest/zarrs/index.html#storage-support) for a list of stores that implement the `zarrs_storage` API.
//!
//! ## Licence
//! `zarrs_storage` is licensed under either of
//! - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_storage/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//! - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_storage/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
#![cfg_attr(docsrs, feature(doc_auto_cfg))]

mod maybe;
pub mod storage_adapter;
mod storage_handle;
mod storage_sync;
mod storage_value_io;
pub mod store;
mod store_key;
mod store_prefix;

pub mod byte_range;
use byte_range::{ByteOffset, ByteRange, InvalidByteRangeError};

pub use maybe::{MaybeSend, MaybeSync};

#[cfg(feature = "async")]
mod storage_async;

#[cfg(feature = "tests")]
/// Store test utilities (for external store development).
pub mod store_test;

use std::sync::Arc;

use thiserror::Error;

pub use store_key::{StoreKey, StoreKeyError, StoreKeys};
pub use store_prefix::{StorePrefix, StorePrefixError, StorePrefixes};

#[cfg(feature = "async")]
pub use self::storage_async::{
    async_discover_children, async_store_set_partial_values, AsyncListableStorageTraits,
    AsyncReadableListableStorageTraits, AsyncReadableStorageTraits,
    AsyncReadableWritableListableStorageTraits, AsyncReadableWritableStorageTraits,
    AsyncWritableStorageTraits,
};

pub use self::storage_sync::{
    discover_children, store_set_partial_values, ListableStorageTraits,
    ReadableListableStorageTraits, ReadableStorageTraits, ReadableWritableListableStorageTraits,
    ReadableWritableStorageTraits, WritableStorageTraits,
};

pub use self::storage_handle::StorageHandle;

pub use storage_value_io::StorageValueIO;

/// [`Arc`] wrapped readable storage.
pub type ReadableStorage = Arc<dyn ReadableStorageTraits>;

/// [`Arc`] wrapped writable storage.
pub type WritableStorage = Arc<dyn WritableStorageTraits>;

/// [`Arc`] wrapped readable and writable storage.
pub type ReadableWritableStorage = Arc<dyn ReadableWritableStorageTraits>;

/// [`Arc`] wrapped readable, writable, and listable storage.
pub type ReadableWritableListableStorage = Arc<dyn ReadableWritableListableStorageTraits>;

/// [`Arc`] wrapped listable storage.
pub type ListableStorage = Arc<dyn ListableStorageTraits>;

/// [`Arc`] wrapped readable and listable storage.
pub type ReadableListableStorage = Arc<dyn ReadableListableStorageTraits>;

#[cfg(feature = "async")]
/// [`Arc`] wrapped asynchronous readable storage.
pub type AsyncReadableStorage = Arc<dyn AsyncReadableStorageTraits>;

#[cfg(feature = "async")]
/// [`Arc`] wrapped asynchronous writable storage.
pub type AsyncWritableStorage = Arc<dyn AsyncWritableStorageTraits>;

#[cfg(feature = "async")]
/// [`Arc`] wrapped asynchronous listable storage.
pub type AsyncListableStorage = Arc<dyn AsyncListableStorageTraits>;

#[cfg(feature = "async")]
/// [`Arc`] wrapped asynchronous readable and listable storage.
pub type AsyncReadableListableStorage = Arc<dyn AsyncReadableListableStorageTraits>;

#[cfg(feature = "async")]
/// [`Arc`] wrapped asynchronous readable, writable and listable storage.
pub type AsyncReadableWritableListableStorage = Arc<dyn AsyncReadableWritableListableStorageTraits>;

/// The type for bytes used in synchronous store set and get methods.
///
/// An alias for [`bytes::Bytes`].
pub type Bytes = bytes::Bytes;

/// An alias for bytes which may or may not be available.
///
/// When a value is read from a store, it returns `MaybeBytes` which is [`None`] if the key is not available.
///
/// A bytes to bytes codec only decodes `MaybeBytes` holding actual bytes, otherwise the bytes are propagated to the next decoder.
/// An array to bytes partial decoder must take care of converting missing chunks to the fill value.
pub type MaybeBytes = Option<Bytes>;

#[cfg(feature = "async")]
/// The type for bytes used in asynchronous store set and get methods.
///
/// An alias for [`bytes::Bytes`].
pub type AsyncBytes = bytes::Bytes;

#[cfg(feature = "async")]
/// An alias for bytes which may or may not be available.
///
/// When a value is read from a store, it returns `MaybeAsyncBytes` which is [`None`] if the key is not available.
pub type MaybeAsyncBytes = Option<AsyncBytes>;

/// A [`StoreKey`] and [`ByteRange`].
#[derive(Debug, Clone)]
pub struct StoreKeyRange {
    /// The key for the range.
    key: StoreKey,
    /// The byte range.
    byte_range: ByteRange,
}

impl StoreKeyRange {
    /// Create a new [`StoreKeyRange`].
    #[must_use]
    pub const fn new(key: StoreKey, byte_range: ByteRange) -> Self {
        Self { key, byte_range }
    }
}

impl std::fmt::Display for StoreKeyRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{}:{}", self.key, self.byte_range)
    }
}

/// A [`StoreKey`], [`ByteOffset`], and value (bytes).
#[derive(Debug, Clone)]
#[must_use]
pub struct StoreKeyOffsetValue<'a> {
    /// The key.
    key: StoreKey,
    /// The starting byte offset.
    offset: ByteOffset,
    /// The store value.
    value: &'a [u8],
}

impl StoreKeyOffsetValue<'_> {
    /// Create a new [`StoreKeyOffsetValue`].
    pub const fn new(key: StoreKey, offset: ByteOffset, value: &[u8]) -> StoreKeyOffsetValue<'_> {
        StoreKeyOffsetValue { key, offset, value }
    }

    /// Get the key.
    #[must_use]
    pub const fn key(&self) -> &StoreKey {
        &self.key
    }

    /// Get the offset.
    #[must_use]
    pub const fn offset(&self) -> ByteOffset {
        self.offset
    }

    /// Get the value.
    #[must_use]
    pub const fn value(&self) -> &[u8] {
        self.value
    }
}

/// [`StoreKeys`] and [`StorePrefixes`].
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
#[allow(dead_code)]
pub struct StoreKeysPrefixes {
    keys: StoreKeys,
    prefixes: StorePrefixes,
}

impl StoreKeysPrefixes {
    /// Create a new [`StoreKeysPrefixes`].
    #[must_use]
    pub fn new(keys: StoreKeys, prefixes: StorePrefixes) -> Self {
        Self { keys, prefixes }
    }

    /// Returns the keys.
    #[must_use]
    pub const fn keys(&self) -> &StoreKeys {
        &self.keys
    }

    /// Returns the prefixes.
    #[must_use]
    pub const fn prefixes(&self) -> &StorePrefixes {
        &self.prefixes
    }
}

/// A storage error.
#[derive(Debug, Clone, Error)]
pub enum StorageError {
    /// A write operation was attempted on a read only store.
    #[error("a write operation was attempted on a read only store")]
    ReadOnly,
    /// An IO error.
    #[error(transparent)]
    IOError(#[from] Arc<std::io::Error>),
    // /// An error serialising or deserialising JSON.
    // #[error(transparent)]
    // InvalidJSON(#[from] serde_json::Error),
    /// An error parsing the metadata for a key.
    #[error("error parsing metadata for {0}: {1}")]
    InvalidMetadata(StoreKey, String),
    #[error("missing metadata for store prefix {0}")]
    /// Missing metadata.
    MissingMetadata(StorePrefix),
    /// An invalid store prefix.
    #[error("invalid store prefix {0}")]
    StorePrefixError(#[from] StorePrefixError),
    /// An invalid store key.
    #[error("invalid store key {0}")]
    InvalidStoreKey(#[from] StoreKeyError),
    // /// An invalid node path.
    // #[error("invalid node path {0}")]
    // NodePathError(#[from] NodePathError),
    // /// An invalid node name.
    // #[error("invalid node name {0}")]
    // NodeNameError(#[from] NodeNameError),
    /// An invalid byte range.
    #[error("invalid byte range {0}")]
    InvalidByteRangeError(#[from] InvalidByteRangeError),
    /// The requested method is not supported.
    #[error("{0}")]
    Unsupported(String),
    /// Unknown key size where the key size must be known.
    #[error("{0}")]
    UnknownKeySize(StoreKey),
    /// Any other error.
    #[error("{0}")]
    Other(String),
}

impl From<std::io::Error> for StorageError {
    fn from(err: std::io::Error) -> Self {
        Self::IOError(Arc::new(err))
    }
}

impl From<&str> for StorageError {
    fn from(err: &str) -> Self {
        Self::Other(err.to_string())
    }
}

impl From<String> for StorageError {
    fn from(err: String) -> Self {
        Self::Other(err)
    }
}
