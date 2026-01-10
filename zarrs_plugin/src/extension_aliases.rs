use std::borrow::Cow;
use std::sync::{RwLockReadGuard, RwLockWriteGuard};

use crate::{ZarrVersion, ZarrVersion2, ZarrVersion3};

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

    /// Check if the given name matches this extension's identifier or any of its aliases for the given Zarr version.
    ///
    /// This method checks in order:
    /// 1. The extension's `default_name`
    /// 2. String aliases from `aliases_str`
    /// 3. Regex patterns from `aliases_regex`
    #[must_use]
    fn matches_name(name: &str) -> bool {
        let aliases = Self::aliases();
        if name == aliases.default_name && !aliases.default_name.is_empty() {
            return true;
        }
        aliases.aliases_str.iter().any(|a| a.as_ref() == name)
            || aliases.aliases_regex.iter().any(|r| r.is_match(name))
    }
}

/// Convenience trait for Zarr V3 extension aliases.
///
/// This trait is blanket-implemented for all types that implement
/// `ExtensionAliases<ZarrVersion3>`, providing convenient `_v3` suffixed methods.
pub trait ExtensionAliasesV3: ExtensionAliases<ZarrVersion3> {
    /// Get a read lock on the V3 aliases configuration.
    fn aliases_v3() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
        <Self as ExtensionAliases<ZarrVersion3>>::aliases()
    }

    /// Get a write lock on the V3 aliases configuration.
    fn aliases_v3_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
        <Self as ExtensionAliases<ZarrVersion3>>::aliases_mut()
    }

    /// Check if the given name matches this extension for Zarr V3.
    #[must_use]
    fn matches_name_v3(name: &str) -> bool {
        <Self as ExtensionAliases<ZarrVersion3>>::matches_name(name)
    }
}

impl<T: ExtensionAliases<ZarrVersion3>> ExtensionAliasesV3 for T {}

/// Convenience trait for Zarr V2 extension aliases.
///
/// This trait is blanket-implemented for all types that implement
/// `ExtensionAliases<ZarrVersion2>`, providing convenient `_v2` suffixed methods.
pub trait ExtensionAliasesV2: ExtensionAliases<ZarrVersion2> {
    /// Get a read lock on the V2 aliases configuration.
    fn aliases_v2() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
        <Self as ExtensionAliases<ZarrVersion2>>::aliases()
    }

    /// Get a write lock on the V2 aliases configuration.
    fn aliases_v2_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
        <Self as ExtensionAliases<ZarrVersion2>>::aliases_mut()
    }

    /// Check if the given name matches this extension for Zarr V2.
    #[must_use]
    fn matches_name_v2(name: &str) -> bool {
        <Self as ExtensionAliases<ZarrVersion2>>::matches_name(name)
    }
}

impl<T: ExtensionAliases<ZarrVersion2>> ExtensionAliasesV2 for T {}
