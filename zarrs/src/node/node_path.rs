use std::path::PathBuf;

use derive_more::Display;
use thiserror::Error;

use zarrs_storage::{StorePrefix, StorePrefixError};

/// A Zarr hierarchy node path.
///
/// See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#path>
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Display)]
#[display("{}", _0.to_string_lossy())]
pub struct NodePath(PathBuf);

/// An invalid node path.
#[derive(Clone, Debug, Error)]
#[error("invalid node path {0}")]
pub struct NodePathError(String);

impl NodePath {
    /// Create a new Zarr node path from `path`.
    ///
    /// # Errors
    ///
    /// Returns [`NodePathError`] if `path` is not valid according to [`NodePath::validate`()].
    pub fn new(path: &str) -> Result<Self, NodePathError> {
        if Self::validate(path) {
            Ok(Self(PathBuf::from(path)))
        } else {
            Err(NodePathError(path.to_string()))
        }
    }

    /// The root node.
    #[must_use]
    pub fn root() -> Self {
        Self(PathBuf::from("/"))
    }

    /// Extracts a string slice containing the node path `String`.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn as_str(&self) -> &str {
        self.0.to_str().unwrap()
    }

    /// Extracts the path as a [`std::path::Path`].
    #[must_use]
    pub fn as_path(&self) -> &std::path::Path {
        &self.0
    }

    /// Validates a path according to the following rules from the specification:
    /// - A path always starts with `/`, and
    /// - a non-root path cannot end with `/`, because node names must be non-empty and cannot contain `/`.
    ///
    /// Additionally, it checks that there are no empty nodes (i.e. a `//` substring).
    #[must_use]
    pub fn validate(path: &str) -> bool {
        path.eq("/") || (path.starts_with('/') && !path.ends_with('/') && !path.contains("//"))
    }
}

impl TryFrom<&str> for NodePath {
    type Error = NodePathError;

    fn try_from(path: &str) -> Result<Self, Self::Error> {
        Self::new(path)
    }
}

impl TryFrom<&StorePrefix> for NodePath {
    type Error = NodePathError;

    fn try_from(prefix: &StorePrefix) -> Result<Self, Self::Error> {
        let path = "/".to_string() + prefix.as_str().strip_suffix('/').unwrap();
        Self::new(&path)
    }
}

impl TryInto<StorePrefix> for &NodePath {
    type Error = StorePrefixError;

    fn try_into(self) -> Result<StorePrefix, StorePrefixError> {
        let path = self.as_str();
        if path.eq("/") {
            StorePrefix::new("")
        } else {
            StorePrefix::new(path.strip_prefix('/').unwrap_or(path).to_string() + "/")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn node_path() {
        assert!(NodePath::new("/").is_ok());
        assert!(NodePath::new("/a/b").is_ok());
        assert_eq!(NodePath::new("/a/b").unwrap().to_string(), "/a/b");
        assert!(NodePath::new("/a/b/").is_err());
        assert_eq!(
            NodePath::new("/a/b/").unwrap_err().to_string(),
            "invalid node path /a/b/"
        );
        assert!(NodePath::new("/a//b").is_err());
        assert_eq!(NodePath::new("/a/b").unwrap().as_path(), Path::new("/a/b/"));
    }
}
