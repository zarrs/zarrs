use std::path::Path;

use derive_more::{Display, From};
use thiserror::Error;

/// A Zarr abstract store prefix.
///
/// See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#abstract-store-interface>.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Display)]
pub struct StorePrefix(String);

/// An invalid store prefix.
#[derive(Debug, Clone, Error, From)]
#[error("invalid store prefix {0}")]
pub struct StorePrefixError(String);

/// A list of [`StorePrefix`].
pub type StorePrefixes = Vec<StorePrefix>;

impl StorePrefix {
    /// Create a new Zarr Prefix from `prefix`.
    ///
    /// # Errors
    /// Returns [`StorePrefixError`] if `prefix` is not valid according to [`StorePrefix::validate`()].
    pub fn new(prefix: impl Into<String>) -> Result<Self, StorePrefixError> {
        let prefix = prefix.into();
        if Self::validate(&prefix) {
            Ok(Self(prefix))
        } else {
            Err(StorePrefixError(prefix))
        }
    }

    /// Create a new Zarr Prefix from `prefix`.
    ///
    /// # Safety
    /// `prefix` is not validated, so this can result in an invalid store prefix.
    #[must_use]
    pub unsafe fn new_unchecked(prefix: impl Into<String>) -> Self {
        let prefix = prefix.into();
        debug_assert!(Self::validate(&prefix));
        Self(prefix)
    }

    /// The root prefix.
    #[must_use]
    pub const fn root() -> Self {
        Self(String::new())
    }

    /// Extracts a string slice containing the Prefix `String`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Validates a prefix according to the following rules from the specification:
    /// - a prefix is a string containing only characters that are valid for use in keys,
    /// - and ending with a trailing / character.
    #[must_use]
    pub fn validate(prefix: &str) -> bool {
        prefix.is_empty() || (prefix.ends_with('/') && !prefix.starts_with('/'))
    }

    /// Returns the prefix of the parent, it if has one.
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        Path::new(&self.0).parent().map(|parent| {
            let parent = parent.to_str().unwrap_or_default();
            if parent.is_empty() {
                unsafe { Self::new_unchecked("") }
            } else {
                unsafe { Self::new_unchecked(parent.to_string() + "/") }
            }
        })
    }
}

impl TryFrom<&str> for StorePrefix {
    type Error = StorePrefixError;

    fn try_from(prefix: &str) -> Result<Self, StorePrefixError> {
        Self::new(prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid() {
        assert!(StorePrefix::new("").is_ok());
        assert!(StorePrefix::new("a/").is_ok());
        assert!(StorePrefix::new("a/b/").is_ok());
        assert!(StorePrefix::try_from("a/").is_ok());
        assert_eq!(StorePrefix::try_from("a/").unwrap().to_string(), "a/");
    }

    #[test]
    fn invalid() {
        assert!(StorePrefix::new("a").is_err());
        assert!(StorePrefix::new("a/b").is_err());
        assert_eq!(
            StorePrefix::new("a/b").unwrap_err().to_string(),
            "invalid store prefix a/b"
        );
    }

    #[test]
    fn parent() {
        let store_prefix = StorePrefix::new("a/b/").unwrap();
        assert_eq!(store_prefix.parent(), Some(StorePrefix::new("a/").unwrap()));
        let store_prefix = StorePrefix::new("a/").unwrap();
        assert_eq!(store_prefix.parent(), Some(StorePrefix::new("").unwrap()));
        let store_prefix = StorePrefix::new("").unwrap();
        assert_eq!(store_prefix.parent(), None);
    }
}
