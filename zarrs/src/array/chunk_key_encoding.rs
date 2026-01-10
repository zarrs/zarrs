//! Zarr chunk key encodings. Includes a [default](default::DefaultChunkKeyEncoding) and [v2](v2::V2ChunkKeyEncoding) implementation.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/chunk-key-encodings/index.html>.
//!
#![doc = include_str!("../../doc/status/chunk_key_encodings.md")]

pub mod default;
pub mod default_suffix;
pub mod v2;

use std::sync::{Arc, LazyLock};

pub use default::{DefaultChunkKeyEncoding, DefaultChunkKeyEncodingConfiguration};
pub use default_suffix::{
    DefaultSuffixChunkKeyEncoding, DefaultSuffixChunkKeyEncodingConfiguration,
};
use derive_more::{Deref, From};
pub use v2::{V2ChunkKeyEncoding, V2ChunkKeyEncodingConfiguration};
use zarrs_plugin::{
    ExtensionAliases, PluginUnsupportedError, RuntimePlugin, RuntimeRegistry, ZarrVersion3,
};

pub use crate::metadata::ChunkKeySeparator;
use crate::metadata::Configuration;
use crate::metadata::v3::MetadataV3;
use crate::plugin::{ExtensionName, Plugin, PluginCreateError, ZarrVersions};
use crate::storage::{MaybeSend, MaybeSync, StoreKey};

/// A chunk key encoding.
#[derive(Debug, Clone, From, Deref)]
pub struct ChunkKeyEncoding(Arc<dyn ChunkKeyEncodingTraits>);

impl<T: ChunkKeyEncodingTraits + 'static> From<Arc<T>> for ChunkKeyEncoding {
    fn from(chunk_key_encoding: Arc<T>) -> Self {
        Self(chunk_key_encoding)
    }
}

/// A chunk key encoding plugin.
#[derive(derive_more::Deref)]
pub struct ChunkKeyEncodingPlugin(Plugin<ChunkKeyEncoding, MetadataV3>);
inventory::collect!(ChunkKeyEncodingPlugin);

impl ChunkKeyEncodingPlugin {
    /// Create a new [`ChunkKeyEncodingPlugin`] for a type implementing [`ExtensionAliases<ZarrVersion3>`].
    ///
    /// The `match_name_fn` is automatically derived from `T::matches_name`.
    pub const fn new<T: ExtensionAliases<ZarrVersion3>>(
        create_fn: fn(metadata: &MetadataV3) -> Result<ChunkKeyEncoding, PluginCreateError>,
    ) -> Self {
        Self(Plugin::new(|name| T::matches_name(name), create_fn))
    }
}

/// A runtime chunk key encoding plugin for dynamic registration.
pub type ChunkKeyEncodingRuntimePlugin = RuntimePlugin<ChunkKeyEncoding, MetadataV3>;

/// Global runtime registry for chunk key encoding plugins.
pub static CHUNK_KEY_ENCODING_RUNTIME_REGISTRY: LazyLock<
    RuntimeRegistry<ChunkKeyEncodingRuntimePlugin>,
> = LazyLock::new(RuntimeRegistry::new);

/// A handle to a registered chunk key encoding plugin.
pub type ChunkKeyEncodingRuntimeRegistryHandle = Arc<ChunkKeyEncodingRuntimePlugin>;

/// Register a chunk key encoding plugin at runtime.
///
/// Runtime-registered plugins take precedence over compile-time registered plugins.
///
/// # Returns
/// A handle that can be used to unregister the plugin later.
pub fn register_chunk_key_encoding(
    plugin: ChunkKeyEncodingRuntimePlugin,
) -> ChunkKeyEncodingRuntimeRegistryHandle {
    CHUNK_KEY_ENCODING_RUNTIME_REGISTRY.register(plugin)
}

/// Unregister a runtime chunk key encoding plugin.
///
/// # Returns
/// `true` if the plugin was found and removed, `false` otherwise.
pub fn unregister_chunk_key_encoding(handle: &ChunkKeyEncodingRuntimeRegistryHandle) -> bool {
    CHUNK_KEY_ENCODING_RUNTIME_REGISTRY.unregister(handle)
}

impl ExtensionName for ChunkKeyEncoding {
    fn name(&self, version: ZarrVersions) -> Option<std::borrow::Cow<'static, str>> {
        self.0.name(version)
    }
}

impl ChunkKeyEncoding {
    /// Create a chunk key encoding.
    pub fn new<T: ChunkKeyEncodingTraits + 'static>(chunk_key_encoding: T) -> Self {
        let chunk_key_encoding: Arc<dyn ChunkKeyEncodingTraits> = Arc::new(chunk_key_encoding);
        chunk_key_encoding.into()
    }

    /// Create the metadata for the chunk key encoding.
    ///
    /// # Panics
    ///
    /// Panics if the chunk key encoding has no name for V3.
    #[must_use]
    pub fn metadata(&self) -> MetadataV3 {
        let name = self
            .name_v3()
            .expect("chunk key encoding must have a V3 name");
        let configuration = self.0.configuration();
        if configuration.is_empty() {
            MetadataV3::new(name.into_owned())
        } else {
            MetadataV3::new_with_configuration(name.into_owned(), configuration)
        }
    }

    /// Create a chunk key encoding from metadata.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if the metadata is invalid or not associated with a registered chunk key encoding plugin.
    pub fn from_metadata(metadata: &MetadataV3) -> Result<Self, PluginCreateError> {
        let name = metadata.name();

        // Check runtime registry first (higher priority)
        {
            let result = CHUNK_KEY_ENCODING_RUNTIME_REGISTRY.with_plugins(|plugins| {
                for plugin in plugins {
                    if plugin.match_name(name) {
                        return Some(plugin.create(metadata));
                    }
                }
                None
            });
            if let Some(result) = result {
                return result;
            }
        }

        // Fall back to compile-time registered plugins
        for plugin in inventory::iter::<ChunkKeyEncodingPlugin> {
            if plugin.match_name(name) {
                return plugin.create(metadata);
            }
        }
        Err(PluginUnsupportedError::new(
            metadata.name().to_string(),
            "chunk key encoding".to_string(),
        )
        .into())
    }
}

impl<T> From<T> for ChunkKeyEncoding
where
    T: ChunkKeyEncodingTraits + 'static,
{
    fn from(chunk_key_encoding: T) -> Self {
        Self::new(chunk_key_encoding)
    }
}

/// Chunk key encoding traits.
pub trait ChunkKeyEncodingTraits: ExtensionName + core::fmt::Debug + MaybeSend + MaybeSync {
    /// The configuration of the chunk key encoding.
    fn configuration(&self) -> Configuration;

    /// Encode chunk grid indices (grid cell coordinates) into a store key.
    fn encode(&self, chunk_grid_indices: &[u64]) -> StoreKey;
}
