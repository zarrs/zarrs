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
pub mod url_pipeline;

pub mod byte_range;
use byte_range::{ByteOffset, InvalidByteRangeError};

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
    async_discover_children, async_store_set_partial_many, AsyncListableStorageTraits,
    AsyncReadableListableStorageTraits, AsyncReadableStorageTraits,
    AsyncReadableWritableListableStorageTraits, AsyncReadableWritableStorageTraits,
    AsyncWritableStorageTraits,
};

pub use self::storage_sync::{
    discover_children, store_set_partial_many, ListableStorageTraits,
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

/// [`Arc`] wrapped listable storage.
pub type ListableStorage = Arc<dyn ListableStorageTraits>;

/// [`Arc`] wrapped readable and listable storage.
pub type ReadableListableStorage = Arc<dyn ReadableListableStorageTraits>;

/// [`Arc`] wrapped readable, writable, and listable storage.
pub type ReadableWritableListableStorage = Arc<dyn ReadableWritableListableStorageTraits>;

#[cfg(feature = "async")]
/// [`Arc`] wrapped asynchronous readable storage.
pub type AsyncReadableStorage = Arc<dyn AsyncReadableStorageTraits>;

#[cfg(feature = "async")]
/// [`Arc`] wrapped asynchronous writable storage.
pub type AsyncWritableStorage = Arc<dyn AsyncWritableStorageTraits>;

#[cfg(feature = "async")]
/// [`Arc`] wrapped asynchronous readable and writable storage.
pub type AsyncReadableWritableStorage = Arc<dyn AsyncReadableWritableStorageTraits>;

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

/// An iterator of [`Bytes`].
type BytesIterator<'a> = Box<dyn Iterator<Item = Result<Bytes, StorageError>> + 'a>;

/// An iterator of [`Bytes`] which may be [`None`] indicating the bytes are not present.
pub type MaybeBytesIterator<'a> = Option<BytesIterator<'a>>;

#[cfg(feature = "async")]
/// An asynchronous iterator of [`Bytes`].
type AsyncBytesIterator<'a> = futures::stream::BoxStream<'a, Result<Bytes, StorageError>>;

#[cfg(feature = "async")]
/// An asynchronous iterator of [`Bytes`] which may be [`None`] indicating the bytes are not present.
pub type AsyncMaybeBytesIterator<'a> = Option<AsyncBytesIterator<'a>>;

/// This trait combines [`Iterator<Item = (Bytes, ByteOffset)>`] and [`MaybeSend`],
/// as they cannot be combined together directly in function signatures.
pub trait MaybeSendOffsetBytesIterator<T>: Iterator<Item = (ByteOffset, T)> + MaybeSend {}

impl<I, T> MaybeSendOffsetBytesIterator<T> for I where
    I: Iterator<Item = (ByteOffset, T)> + MaybeSend
{
}

/// A [`Bytes`] and [`ByteOffset`] iterator.
pub type OffsetBytesIterator<'a, T = Bytes> = Box<dyn MaybeSendOffsetBytesIterator<T> + 'a>;

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
