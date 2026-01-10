//! The plugin API for the [`zarrs`](https://crates.io/crates/zarrs) Rust crate.
//!
//! A [`Plugin`] creates concrete implementations of [Zarr V3 extension points](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#extension-points) from inputs.
//! Extension points include chunk grids, chunk key encodings, codecs, data types, and storage transformers.
//!
//! In `zarrs`, plugins are registered at compile time using the [`inventory`](https://docs.rs/inventory/latest/inventory/) crate.
//! At runtime, a name matching function is applied to identify which registered plugin is associated with the metadata.
//! If a match is found, the plugin is created from the metadata and other relevant inputs.
//!
//! ## Licence
//! `zarrs_plugin` is licensed under either of
//!  - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_plugin/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//!  - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_plugin/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

use thiserror::Error;

mod macros;

mod maybe;
pub use maybe::{MaybeSend, MaybeSync};

mod plugin;
pub use plugin::{Plugin, Plugin2};

mod runtime_plugin;
pub use runtime_plugin::{RuntimePlugin, RuntimePlugin2};

mod runtime_registry;
pub use runtime_registry::{RuntimeRegistry, RuntimeRegistryHandle};

mod extension_name;
pub use extension_name::{ExtensionName, ExtensionNameStatic};

mod extension_aliases;
pub use extension_aliases::{
    ExtensionAliases, ExtensionAliasesConfig, ExtensionAliasesV2, ExtensionAliasesV3,
};

mod extension_type;
pub use extension_type::{
    ExtensionType, ExtensionTypeChunkGrid, ExtensionTypeChunkKeyEncoding, ExtensionTypeCodec,
    ExtensionTypeDataType, ExtensionTypeStorageTransformer,
};

/// Re-export of [`regex::Regex`] for use in extension alias configurations.
pub use regex::Regex;

/// Re-export of [`paste`] for use in the [`impl_extension_aliases`] macro.
#[doc(hidden)]
pub use paste;

mod zarr_version;
pub use zarr_version::{ZarrVersion, ZarrVersion2, ZarrVersion3, ZarrVersions};

/// An unsupported plugin error
#[derive(Clone, Debug, Error)]
#[error("{plugin_type} {name} is not supported")]
pub struct PluginUnsupportedError {
    name: String,
    plugin_type: String,
}

impl PluginUnsupportedError {
    /// Create a new [`PluginUnsupportedError`].
    #[must_use]
    pub fn new(name: String, plugin_type: String) -> Self {
        Self { name, plugin_type }
    }
}

/// An invalid plugin metadata error.
#[derive(Clone, Debug, Error)]
#[error("configuration is unsupported: {reason}")]
pub struct PluginConfigurationInvalidError {
    reason: String,
}

impl PluginConfigurationInvalidError {
    /// Create a new [`PluginConfigurationInvalidError`].
    #[must_use]
    pub fn new(reason: String) -> Self {
        Self { reason }
    }
}

/// A plugin creation error.
#[derive(Clone, Debug, Error)]
#[allow(missing_docs)]
pub enum PluginCreateError {
    /// An unsupported plugin.
    #[error(transparent)]
    Unsupported(#[from] PluginUnsupportedError),
    /// Invalid name.
    #[error("invalid name: {name}")]
    NameInvalid { name: String },
    /// Invalid metadata.
    #[error(transparent)]
    ConfigurationInvalid(#[from] PluginConfigurationInvalidError),
    /// Other
    #[error("{_0}")]
    Other(String),
}

impl From<&str> for PluginCreateError {
    fn from(err_string: &str) -> Self {
        Self::Other(err_string.to_string())
    }
}

impl From<String> for PluginCreateError {
    fn from(err_string: String) -> Self {
        Self::Other(err_string)
    }
}
