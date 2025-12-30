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

use std::borrow::Cow;

use thiserror::Error;

mod maybe;
pub use maybe::{MaybeSend, MaybeSync};

mod extension_type;
pub use extension_type::{
    ExtensionAliases, ExtensionAliasesConfig, ExtensionIdentifier, ExtensionType,
    ExtensionTypeChunkGrid, ExtensionTypeChunkKeyEncoding, ExtensionTypeCodec,
    ExtensionTypeDataType, ExtensionTypeStorageTransformer,
};

/// Re-export of [`regex::Regex`] for use in extension alias configurations.
pub use regex::Regex;

mod zarr_version;
pub use zarr_version::{ZarrVersion, ZarrVersion2, ZarrVersion3, ZarrVersions};

/// A plugin.
pub struct Plugin<TPlugin, TInput> {
    /// the identifier of the plugin.
    identifier: &'static str,
    /// Tests if the name is a match for this plugin for a given Zarr version.
    match_name_fn: fn(name: &str, version: ZarrVersions) -> bool,
    /// Returns the default name for this codec for the given Zarr version.
    default_name_fn: fn(ZarrVersions) -> Cow<'static, str>,
    /// Create an implementation of this plugin from metadata.
    create_fn: fn(input: &TInput) -> Result<TPlugin, PluginCreateError>,
}

/// A plugin (two parameters).
pub struct Plugin2<TPlugin, TInput1, TInput2> {
    /// the identifier of the plugin.
    identifier: &'static str,
    /// Tests if the name is a match for this plugin for a given Zarr version.
    match_name_fn: fn(name: &str, version: ZarrVersions) -> bool,
    /// Returns the default name for this codec for the given Zarr version.
    default_name_fn: fn(ZarrVersions) -> Cow<'static, str>,
    /// Create an implementation of this plugin from metadata.
    create_fn: fn(input1: &TInput1, input2: &TInput2) -> Result<TPlugin, PluginCreateError>,
}

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
#[error("{plugin_type} {identifier} is unsupported with metadata: {metadata}")]
pub struct PluginMetadataInvalidError {
    identifier: &'static str,
    plugin_type: &'static str,
    metadata: String,
}

impl PluginMetadataInvalidError {
    /// Create a new [`PluginMetadataInvalidError`].
    #[must_use]
    pub fn new(identifier: &'static str, plugin_type: &'static str, metadata: String) -> Self {
        Self {
            identifier,
            plugin_type,
            metadata,
        }
    }
}

/// A plugin creation error.
#[derive(Clone, Debug, Error)]
#[allow(missing_docs)]
pub enum PluginCreateError {
    /// An unsupported plugin.
    #[error(transparent)]
    Unsupported(#[from] PluginUnsupportedError),
    /// Invalid metadata.
    #[error(transparent)]
    MetadataInvalid(#[from] PluginMetadataInvalidError),
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

impl<TPlugin, TInput> Plugin<TPlugin, TInput> {
    /// Create a new plugin for registration.
    pub const fn new(
        identifier: &'static str,
        match_name_fn: fn(name: &str, version: ZarrVersions) -> bool,
        default_name_fn: fn(ZarrVersions) -> Cow<'static, str>,
        create_fn: fn(inputs: &TInput) -> Result<TPlugin, PluginCreateError>,
    ) -> Self {
        Self {
            identifier,
            match_name_fn,
            default_name_fn,
            create_fn,
        }
    }

    /// Create a `TPlugin` plugin from `inputs`.
    ///
    /// # Errors
    ///
    /// Returns a [`PluginCreateError`] if plugin creation fails due to either:
    ///  - metadata name being unregistered,
    ///  - or the configuration is invalid, or
    ///  - some other reason specific to the plugin.
    pub fn create(&self, input: &TInput) -> Result<TPlugin, PluginCreateError> {
        (self.create_fn)(input)
    }

    /// Returns true if this plugin is associated with `name` for the given Zarr version.
    #[must_use]
    pub fn match_name(&self, name: &str, version: impl Into<ZarrVersions>) -> bool {
        (self.match_name_fn)(name, version.into())
    }

    /// Return the default name for this plugin for the given Zarr version.
    pub fn default_name(&self, version: impl Into<ZarrVersions>) -> Cow<'static, str> {
        (self.default_name_fn)(version.into())
    }

    /// Returns the identifier of the plugin.
    #[must_use]
    pub const fn identifier(&self) -> &'static str {
        self.identifier
    }
}

impl<TPlugin, TInput1, TInput2> Plugin2<TPlugin, TInput1, TInput2> {
    /// Create a new plugin for registration.
    pub const fn new(
        identifier: &'static str,
        match_name_fn: fn(name: &str, version: ZarrVersions) -> bool,
        default_name_fn: fn(ZarrVersions) -> Cow<'static, str>,
        create_fn: fn(input1: &TInput1, input2: &TInput2) -> Result<TPlugin, PluginCreateError>,
    ) -> Self {
        Self {
            identifier,
            match_name_fn,
            default_name_fn,
            create_fn,
        }
    }

    /// Create a `TPlugin` plugin from `inputs`.
    ///
    /// # Errors
    ///
    /// Returns a [`PluginCreateError`] if plugin creation fails due to either:
    ///  - metadata name being unregistered,
    ///  - or the configuration is invalid, or
    ///  - some other reason specific to the plugin.
    pub fn create(&self, input1: &TInput1, input2: &TInput2) -> Result<TPlugin, PluginCreateError> {
        (self.create_fn)(input1, input2)
    }

    /// Returns true if this plugin is associated with `name` for the given Zarr version.
    #[must_use]
    pub fn match_name(&self, name: &str, version: impl Into<ZarrVersions>) -> bool {
        (self.match_name_fn)(name, version.into())
    }

    /// Return the default name for this plugin for the given Zarr version.
    pub fn default_name(&self, version: impl Into<ZarrVersions>) -> Cow<'static, str> {
        (self.default_name_fn)(version.into())
    }

    /// Returns the identifier of the plugin.
    #[must_use]
    pub const fn identifier(&self) -> &'static str {
        self.identifier
    }
}
