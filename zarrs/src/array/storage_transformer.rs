//! Zarr storage transformers.
//!
//! A Zarr storage transformer modifies a request to read or write data before passing that request to a following storage transformer or store.
//! A [`StorageTransformerChain`] represents a sequence of storage transformers.
//! A storage transformer chain and individual storage transformers all have the same interface as a [store](crate::storage::store).
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#storage-transformers>.
//!
#![doc = include_str!("../../doc/status/storage_transformers.md")]

mod storage_transformer_chain;
pub use storage_transformer_chain::StorageTransformerChain;
use zarrs_plugin::{Plugin, PluginUnsupportedError};
use zarrs_storage::{MaybeSend, MaybeSync};

use std::sync::Arc;

use crate::{
    metadata::v3::MetadataV3,
    node::NodePath,
    plugin::PluginCreateError,
    storage::{ListableStorage, ReadableStorage, StorageError, WritableStorage},
};

#[cfg(feature = "async")]
use crate::storage::{AsyncListableStorage, AsyncReadableStorage, AsyncWritableStorage};

/// An [`Arc`] wrapped storage transformer.
pub type StorageTransformer = Arc<dyn StorageTransformerExtension>;

/// A storage transformer plugin.
#[derive(derive_more::Deref)]
pub struct StorageTransformerPlugin(Plugin<StorageTransformer, (MetadataV3, NodePath)>);
inventory::collect!(StorageTransformerPlugin);

impl StorageTransformerPlugin {
    /// Create a new [`StorageTransformerPlugin`].
    pub const fn new(
        identifier: &'static str,
        match_name_fn: fn(name: &str) -> bool,
        create_fn: fn(
            inputs: &(MetadataV3, NodePath),
        ) -> Result<StorageTransformer, PluginCreateError>,
    ) -> Self {
        Self(Plugin::new(identifier, match_name_fn, create_fn))
    }
}

/// Create a storage transformer from metadata.
///
/// # Errors
///
/// Returns [`PluginCreateError`] if the metadata is invalid or not associated with a registered storage transformer plugin.
pub fn try_create_storage_transformer(
    metadata: &MetadataV3,
    path: &NodePath,
) -> Result<StorageTransformer, PluginCreateError> {
    for plugin in inventory::iter::<StorageTransformerPlugin> {
        if plugin.match_name(metadata.name()) {
            return plugin.create(&(metadata.clone(), path.clone()));
        }
    }
    Err(PluginUnsupportedError::new(
        metadata.name().to_string(),
        "storage transformer".to_string(),
    )
    .into())
}

/// A storage transformer extension.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
pub trait StorageTransformerExtension: core::fmt::Debug + MaybeSend + MaybeSync {
    /// Create metadata.
    fn create_metadata(&self) -> MetadataV3;

    /// Create a readable transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    fn create_readable_transformer(
        self: Arc<Self>,
        storage: ReadableStorage,
    ) -> Result<ReadableStorage, StorageError>;

    /// Create a writable transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    fn create_writable_transformer(
        self: Arc<Self>,
        storage: WritableStorage,
    ) -> Result<WritableStorage, StorageError>;

    /// Create a listable transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    fn create_listable_transformer(
        self: Arc<Self>,
        storage: ListableStorage,
    ) -> Result<ListableStorage, StorageError>;

    #[cfg(feature = "async")]
    /// Create an asynchronous readable transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    async fn create_async_readable_transformer(
        self: Arc<Self>,
        storage: AsyncReadableStorage,
    ) -> Result<AsyncReadableStorage, StorageError>;

    #[cfg(feature = "async")]
    /// Create an asynchronous writable transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    async fn create_async_writable_transformer(
        self: Arc<Self>,
        storage: AsyncWritableStorage,
    ) -> Result<AsyncWritableStorage, StorageError>;

    #[cfg(feature = "async")]
    /// Create an asynchronous listable transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    async fn create_async_listable_transformer(
        self: Arc<Self>,
        storage: AsyncListableStorage,
    ) -> Result<AsyncListableStorage, StorageError>;
}
