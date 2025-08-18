use derive_more::{Display, From};
use thiserror::Error;

use super::StorePrefix;

/// A Zarr abstract store key.
///
/// See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#abstract-store-interface>.
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Display)]
pub struct StoreKey(String);

/// An invalid store key.
#[derive(Debug, Clone, From, Error)]
#[error("invalid store key {0}")]
pub struct StoreKeyError(String);

/// A list of [`StoreKey`].
pub type StoreKeys = Vec<StoreKey>;

impl StoreKey {
    /// Create a new Zarr abstract store key from `key`.
    ///
    /// # Errors
    ///
    /// Returns [`StoreKeyError`] if `key` is not valid according to [`StoreKey::validate()`].
    pub fn new(key: impl Into<String>) -> Result<Self, StoreKeyError> {
        let key = key.into();
        if Self::validate(&key) {
            Ok(Self(key))
        } else {
            Err(StoreKeyError(key))
        }
    }

    /// Create a new Zarr abstract store key from `key` without validation.
    ///
    /// # Safety
    ///
    /// `key` is not validated, so this can result in an invalid store key.
    #[must_use]
    pub unsafe fn new_unchecked(key: impl Into<String>) -> Self {
        let key = key.into();
        debug_assert!(Self::validate(&key));
        Self(key)
    }

    /// Extracts a string slice of the underlying Key [String].
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Validates a key according to the following rule from the specification:
    /// - a key is a Unicode string, where the final character is not a `/` character.
    ///
    /// Additional checks (not in the specification):
    /// - a key which starts with '/' is invalid, and
    /// - a key that contains '//' is invalid, and
    /// - a key cannot be an empty string.
    #[must_use]
    pub fn validate(key: &str) -> bool {
        !key.starts_with('/') && !key.ends_with('/') && !key.is_empty() && !key.contains("//")
    }

    /// Returns true if the key has prefix `prefix`.
    #[must_use]
    pub fn has_prefix(&self, prefix: &StorePrefix) -> bool {
        self.0.starts_with(prefix.as_str())
    }

    /// Convert to a [`StoreKey`].
    #[must_use]
    pub fn to_prefix(&self) -> StorePrefix {
        unsafe { StorePrefix::new_unchecked(self.0.clone() + "/") }
    }

    /// Returns the parent of this key.
    #[must_use]
    pub fn parent(&self) -> StorePrefix {
        let key_split: Vec<_> = self.as_str().split('/').collect();
        let mut parent = key_split[..key_split.len() - 1].join("/");
        if !parent.is_empty() {
            parent.push('/');
        }
        unsafe { StorePrefix::new_unchecked(&parent) }
    }
}

impl TryFrom<&str> for StoreKey {
    type Error = StoreKeyError;

    fn try_from(key: &str) -> Result<Self, Self::Error> {
        Self::new(key)
    }
}

impl From<&StorePrefix> for StoreKey {
    fn from(prefix: &StorePrefix) -> Self {
        let prefix = prefix.as_str();
        let key = prefix.strip_suffix('/').unwrap_or(prefix);
        unsafe { Self::new_unchecked(key.to_string()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_prefix() {
        assert!(StoreKey::new("a").is_ok());
        assert_eq!(StoreKey::new("a").unwrap().to_string(), "a");
        assert!(StoreKey::new("a/").is_err());
        assert_eq!(
            StoreKey::new("a/").unwrap_err().to_string(),
            "invalid store key a/"
        );
        assert!(StoreKey::new("/a").is_err());
        assert_eq!(
            StoreKey::new("a").unwrap().to_prefix(),
            StorePrefix::new("a/").unwrap()
        );
        assert_eq!(
            StoreKey::new("a/b").unwrap().parent(),
            StorePrefix::new("a/").unwrap()
        );
        assert_eq!(
            StoreKey::new("a").unwrap().parent(),
            StorePrefix::new("").unwrap()
        );
    }
}
