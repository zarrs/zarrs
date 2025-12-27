use std::borrow::Cow;
use std::fmt::Debug;
use std::sync::{RwLockReadGuard, RwLockWriteGuard};

use crate::zarr_version::ZarrVersion;
use crate::{ZarrVersion2, ZarrVersion3, ZarrVersions};

/// Marker trait for extension types.
pub trait ExtensionType: Debug + Default {}

/// Per-extension runtime-configurable aliases.
///
/// Each extension type has its own thread-safe storage that can be atomically updated at runtime.
#[derive(Debug, Clone)]
pub struct ExtensionAliasesConfig {
    /// The default name used when serializing this extension.
    pub default_name: Cow<'static, str>,
    /// String aliases that map to this extension's identifier.
    pub aliases_str: Vec<Cow<'static, str>>,
    /// Regex patterns that match this extension.
    pub aliases_regex: Vec<regex::Regex>,
}

impl ExtensionAliasesConfig {
    /// Create a new [`ExtensionAliasesConfig`].
    #[must_use]
    pub fn new(
        default_name: impl Into<Cow<'static, str>>,
        aliases_str: Vec<Cow<'static, str>>,
        aliases_regex: Vec<regex::Regex>,
    ) -> Self {
        Self {
            default_name: default_name.into(),
            aliases_str,
            aliases_regex,
        }
    }
}

/// Per-version alias configuration for an extension.
///
/// This trait is implemented for each Zarr version (`ZarrVersion2`, `ZarrVersion3`)
/// to provide version-specific alias configuration.
pub trait ExtensionAliases<V: ZarrVersion>: Sized {
    /// Get a read lock on the aliases configuration for this version.
    fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig>;

    /// Get a write lock on the aliases configuration for this version.
    fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig>;

    /// Get the default name for this version.
    #[must_use]
    fn default_name() -> Cow<'static, str> {
        Self::aliases().default_name.clone()
    }

    /// Set the default name for this version.
    fn set_default_name(name: impl Into<Cow<'static, str>>) {
        Self::aliases_mut().default_name = name.into();
    }

    /// Check if the given name matches this extension's identifier or any of its aliases for the given Zarr version.
    ///
    /// This method checks in order:
    /// 1. The extension's `default_name`
    /// 2. String aliases from `aliases_str`
    /// 3. Regex patterns from `aliases_regex`
    #[must_use]
    fn matches_name(name: &str) -> bool {
        if name == Self::default_name() {
            return true;
        }
        let aliases = Self::aliases();
        aliases.aliases_str.iter().any(|a| a.as_ref() == name)
            || aliases.aliases_regex.iter().any(|r| r.is_match(name))
    }
}

/// Trait for extension types that have a unique identifier.
///
/// Extension types must implement [`ExtensionAliases`] for both V2 and V3.
pub trait ExtensionIdentifier:
    ExtensionAliases<ZarrVersion2> + ExtensionAliases<ZarrVersion3>
{
    /// The unique identifier for this extension.
    const IDENTIFIER: &'static str;

    /// Get the default name for the given Zarr version.
    #[must_use]
    fn default_name(version: crate::ZarrVersions) -> Cow<'static, str> {
        match version {
            ZarrVersions::V2 => <Self as ExtensionAliases<ZarrVersion2>>::default_name(),
            ZarrVersions::V3 => <Self as ExtensionAliases<ZarrVersion3>>::default_name(),
        }
    }

    /// Set the default name for the given Zarr version.
    fn set_default_name(
        name: impl Into<Cow<'static, str>>,
        version: impl Into<crate::ZarrVersions>,
    ) {
        match version.into() {
            ZarrVersions::V2 => <Self as ExtensionAliases<ZarrVersion2>>::set_default_name(name),
            ZarrVersions::V3 => <Self as ExtensionAliases<ZarrVersion3>>::set_default_name(name),
        }
    }

    /// Returns true if `name` matches this chunk grid for the given version.
    fn matches_name(name: &str, version: impl Into<crate::ZarrVersions>) -> bool {
        match version.into() {
            ZarrVersions::V2 => <Self as ExtensionAliases<ZarrVersion2>>::matches_name(name),
            ZarrVersions::V3 => <Self as ExtensionAliases<ZarrVersion3>>::matches_name(name),
        }
    }

    /// Get a read lock on the aliases configuration for the given version.
    fn aliases(
        version: impl Into<crate::ZarrVersions>,
    ) -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
        match version.into() {
            ZarrVersions::V2 => <Self as ExtensionAliases<ZarrVersion2>>::aliases(),
            ZarrVersions::V3 => <Self as ExtensionAliases<ZarrVersion3>>::aliases(),
        }
    }

    /// Get a write lock on the aliases configuration for the given version.
    fn aliases_mut(
        version: impl Into<crate::ZarrVersions>,
    ) -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
        match version.into() {
            ZarrVersions::V2 => <Self as ExtensionAliases<ZarrVersion2>>::aliases_mut(),
            ZarrVersions::V3 => <Self as ExtensionAliases<ZarrVersion3>>::aliases_mut(),
        }
    }
}

/// The *data type* extension type.
#[derive(Debug, Copy, Clone, Default)]
pub struct ExtensionTypeDataType;

/// The *chunk grid* extension type.
#[derive(Debug, Copy, Clone, Default)]
pub struct ExtensionTypeChunkGrid;

/// The *chunk key encoding* extension type.
#[derive(Debug, Copy, Clone, Default)]
pub struct ExtensionTypeChunkKeyEncoding;

/// The *codec* extension type.
#[derive(Debug, Copy, Clone, Default)]
pub struct ExtensionTypeCodec;

/// The *storage transformer* extension type.
#[derive(Debug, Copy, Clone, Default)]
pub struct ExtensionTypeStorageTransformer;

impl ExtensionType for ExtensionTypeDataType {}
impl ExtensionType for ExtensionTypeChunkGrid {}
impl ExtensionType for ExtensionTypeChunkKeyEncoding {}
impl ExtensionType for ExtensionTypeCodec {}
impl ExtensionType for ExtensionTypeStorageTransformer {}
