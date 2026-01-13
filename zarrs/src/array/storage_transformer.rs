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
use std::sync::{Arc, LazyLock};

pub use storage_transformer_chain::StorageTransformerChain;
use zarrs_plugin::{
    ExtensionAliases, Plugin2, PluginUnsupportedError, RuntimePlugin2, RuntimeRegistry,
    ZarrVersion3,
};

use crate::node::NodePath;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::{ExtensionName, PluginCreateError};
#[cfg(feature = "async")]
use zarrs_storage::{
    AsyncListableStorage, AsyncReadableStorage, AsyncReadableWritableStorage, AsyncWritableStorage,
};
use zarrs_storage::{
    ListableStorage, MaybeSend, MaybeSync, ReadableStorage, ReadableWritableStorage, StorageError,
    WritableStorage,
};

/// An [`Arc`] wrapped storage transformer.
pub type StorageTransformer = Arc<dyn StorageTransformerExtension>;

/// A storage transformer plugin.
#[derive(derive_more::Deref)]
pub struct StorageTransformerPlugin(Plugin2<StorageTransformer, MetadataV3, NodePath>);
inventory::collect!(StorageTransformerPlugin);

impl StorageTransformerPlugin {
    /// Create a new [`StorageTransformerPlugin`] for a type implementing [`ExtensionAliases<ZarrVersion3>`].
    ///
    /// The `match_name_fn` is automatically derived from `T::matches_name`.
    pub const fn new<T: ExtensionAliases<ZarrVersion3>>(
        create_fn: fn(
            metadata: &MetadataV3,
            path: &NodePath,
        ) -> Result<StorageTransformer, PluginCreateError>,
    ) -> Self {
        Self(Plugin2::new(|name| T::matches_name(name), create_fn))
    }
}

/// A runtime storage transformer plugin for dynamic registration.
pub type StorageTransformerRuntimePlugin = RuntimePlugin2<StorageTransformer, MetadataV3, NodePath>;

/// Global runtime registry for storage transformer plugins.
pub static STORAGE_TRANSFORMER_RUNTIME_REGISTRY: LazyLock<
    RuntimeRegistry<StorageTransformerRuntimePlugin>,
> = LazyLock::new(RuntimeRegistry::new);

/// A handle to a registered storage transformer plugin.
pub type StorageTransformerRuntimeRegistryHandle = Arc<StorageTransformerRuntimePlugin>;

/// Register a storage transformer plugin at runtime.
///
/// Runtime-registered plugins take precedence over compile-time registered plugins.
///
/// # Returns
/// A handle that can be used to unregister the plugin later.
pub fn register_storage_transformer(
    plugin: StorageTransformerRuntimePlugin,
) -> StorageTransformerRuntimeRegistryHandle {
    STORAGE_TRANSFORMER_RUNTIME_REGISTRY.register(plugin)
}

/// Unregister a runtime storage transformer plugin.
///
/// # Returns
/// `true` if the plugin was found and removed, `false` otherwise.
pub fn unregister_storage_transformer(handle: &StorageTransformerRuntimeRegistryHandle) -> bool {
    STORAGE_TRANSFORMER_RUNTIME_REGISTRY.unregister(handle)
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
    let name = metadata.name();

    // Check runtime registry first (higher priority)
    {
        let result = STORAGE_TRANSFORMER_RUNTIME_REGISTRY.with_plugins(|plugins| {
            for plugin in plugins {
                if plugin.match_name(name) {
                    return Some(plugin.create(metadata, path));
                }
            }
            None
        });
        if let Some(result) = result {
            return result;
        }
    }

    // Fall back to compile-time registered plugins
    for plugin in inventory::iter::<StorageTransformerPlugin> {
        if plugin.match_name(name) {
            return plugin.create(metadata, path);
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
pub trait StorageTransformerExtension:
    ExtensionName + core::fmt::Debug + MaybeSend + MaybeSync
{
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

    /// Create a readable and writable transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    fn create_readable_writable_transformer(
        self: Arc<Self>,
        storage: ReadableWritableStorage,
    ) -> Result<ReadableWritableStorage, StorageError>;

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
    /// Create an asynchronous readable and writable transformer.
    ///
    /// # Errors
    /// Returns an error if creation fails.
    async fn create_async_readable_writable_transformer(
        self: Arc<Self>,
        storage: AsyncReadableWritableStorage,
    ) -> Result<AsyncReadableWritableStorage, StorageError>;

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
