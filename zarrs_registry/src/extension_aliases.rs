use std::{borrow::Cow, collections::HashMap, marker::PhantomData};

use crate::{ExtensionType, ZarrVersion};

/// A mapping of extension identifiers to default extension `name`s.
pub type ExtensionNameMap = HashMap<&'static str, Cow<'static, str>>;

/// A mapping of extension aliases to extension identifiers (string match).
///
/// Extension identifiers are an implementation detail and may not match the extension `name`.
pub type ExtensionAliasMapString = HashMap<Cow<'static, str>, &'static str>;

/// A mapping of extension aliases to extension identifiers (regex match).
///
/// For example:
///  - "^r\d+$" to "r*"
///  - "^\|m\d+$" to "|mX"
pub type ExtensionAliasMapRegex = Vec<(regex::Regex, &'static str)>;

/// Aliases for Zarr extensions.
#[derive(Debug, Clone)]
pub struct ExtensionAliases<Z: ZarrVersion, T: ExtensionType> {
    zarr_version: PhantomData<Z>,
    extension_type: PhantomData<T>,
    /// The default serialised `name`s.
    pub default_names: ExtensionNameMap,
    /// `name` aliases (string match).
    pub aliases_str: ExtensionAliasMapString,
    /// `name` aliases (regex match).
    pub aliases_regex: ExtensionAliasMapRegex,
}

impl<Z: ZarrVersion, T: ExtensionType> ExtensionAliases<Z, T> {
    /// Create a new [`ExtensionAliases`].
    #[must_use]
    pub fn new(
        default_names: ExtensionNameMap,
        aliases_str: ExtensionAliasMapString,
        aliases_regex: ExtensionAliasMapRegex,
    ) -> Self {
        Self {
            zarr_version: PhantomData,
            extension_type: PhantomData,
            default_names,
            aliases_str,
            aliases_regex,
        }
    }

    /// Map an identifier to the default `name` / `id`.
    pub fn default_name<'a>(&'a self, identifier: &'a str) -> &'a str {
        self.default_names
            .get(identifier)
            .map_or(identifier, AsRef::as_ref)
    }

    /// Map a `name`/`id` that may be aliased to an identifier.
    ///
    /// The input `name` is returned if no string/regex matches are found.
    #[must_use]
    pub fn identifier<'a>(&'a self, name: &'a str) -> &'a str {
        if let Some(alias) = self.aliases_str.get(name) {
            alias
        } else {
            for (regex, identifier) in &self.aliases_regex {
                if regex.is_match(name) {
                    return self.aliases_str.get(*identifier).unwrap_or(identifier);
                }
            }
            name
        }
    }
}
