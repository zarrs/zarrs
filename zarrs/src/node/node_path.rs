use std::path::PathBuf;

use derive_more::Display;
use thiserror::Error;

use super::NodeName;
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
    /// Additionally, every path component must be a valid [`NodeName`]:
    /// - must not start with the reserved prefix `__`, and
    /// - must not be composed only of period characters (`.` or `..`).
    #[must_use]
    pub fn validate(path: &str) -> bool {
        if path == "/" {
            return true;
        }
        if !path.starts_with('/') || path.ends_with('/') {
            return false;
        }
        for component in path[1..].split('/') {
            // An empty component means a `//` substring — reject it before checking
            // NodeName rules (where empty is valid for the root node name).
            if component.is_empty() || !NodeName::validate(component) {
                return false;
            }
        }
        true
    }

    /// Returns `true` if this is the root path.
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.0.as_os_str() == "/"
    }

    /// Returns the parent path of this node, or `None` if this is the root path.
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        self.0.parent().and_then(|p| {
            if p.as_os_str().is_empty() {
                // `PathBuf::parent("/")` returns `Some("")` — root has no parent.
                None
            } else {
                let s = p.to_string_lossy();
                Self::new(&s).ok()
            }
        })
    }

    /// Creates a new path by joining `relative` to this path.
    ///
    /// The `relative` argument should be a relative path segment (e.g., `"sub/foo"` or `"bar"`).
    /// A leading `/` in `relative` is stripped.
    ///
    /// Returns an error if the resulting path is invalid according to [`NodePath::validate`].
    pub fn join(&self, relative: &str) -> Result<Self, NodePathError> {
        // Strip leading slash if present
        let relative = relative.strip_prefix('/').unwrap_or(relative);
        if relative.is_empty() {
            return Ok(self.clone());
        }

        let mut path_buf = self.0.clone();
        path_buf.push(relative);
        let joined = path_buf.to_string_lossy().to_string();
        if Self::validate(&joined) {
            Ok(Self(path_buf))
        } else {
            Err(NodePathError(joined))
        }
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

    #[test]
    fn node_path_is_root() {
        assert!(NodePath::root().is_root());
        assert!(NodePath::new("/").unwrap().is_root());
        assert!(!NodePath::new("/foo").unwrap().is_root());
        assert!(!NodePath::new("/a/b/c").unwrap().is_root());
    }

    #[test]
    fn node_path_validate_rejects_invalid_components() {
        // Reserved prefix `__`
        assert!(!NodePath::validate("/__foo"));
        assert!(!NodePath::validate("/foo/__bar"));
        assert!(!NodePath::validate("/foo/__bar/baz"));

        // Period-only names
        assert!(!NodePath::validate("/."));
        assert!(!NodePath::validate("/.."));
        assert!(!NodePath::validate("/foo/../bar"));
        assert!(!NodePath::validate("/..."));
        assert!(!NodePath::validate("////"));

        // Valid paths still pass
        assert!(NodePath::validate("/"));
        assert!(NodePath::validate("/a/b"));
        assert!(NodePath::validate("/foo__bar")); // `__` not at start is OK
        assert!(NodePath::validate("/foo./bar")); // `.` not the only char is OK
        assert!(NodePath::validate("/foo/bar..")); // `..` not the only char is OK
    }

    #[test]
    fn node_path_join() {
        // Base root
        let root = NodePath::root();
        assert_eq!(root.join("foo").unwrap().as_str(), "/foo");
        assert_eq!(root.join("sub/leaf").unwrap().as_str(), "/sub/leaf");
        // Leading slash stripped
        assert_eq!(root.join("/foo").unwrap().as_str(), "/foo");
        // Empty relative returns clone
        assert_eq!(root.join("").unwrap(), root);

        // Non-root base
        let base = NodePath::new("/group").unwrap();
        assert_eq!(base.join("sub").unwrap().as_str(), "/group/sub");
        assert_eq!(base.join("sub/leaf").unwrap().as_str(), "/group/sub/leaf");
        assert_eq!(base.join("/sub").unwrap().as_str(), "/group/sub");

        // Reject `.` components (period-only)
        assert!(NodePath::new("/foo").unwrap().join("./bar").is_err());
        assert!(root.join("./bar").is_err());

        // Reject `..` components (period-only)
        assert!(root.join("../bar").is_err());
        assert!(base.join("sub/../other").is_err());
        assert!(base.join("../other").is_err());

        // Reject `...` (period-only)
        assert!(root.join(".../bar").is_err());

        // Reject empty components (from `//`)
        assert!(root.join("foo//bar").is_err());

        // Reject trailing slash (becomes empty component after split)
        assert!(root.join("foo/").is_err());

        // Reject `__` reserved prefix
        assert!(root.join("__bar").is_err());
        assert!(root.join("foo/__bar").is_err());
    }

    #[test]
    fn node_path_join_valid_prefixes() {
        // `__` not at start is valid
        assert_eq!(
            NodePath::root().join("foo__bar").unwrap().as_str(),
            "/foo__bar"
        );
        assert_eq!(
            NodePath::root().join("foo__bar/baz").unwrap().as_str(),
            "/foo__bar/baz"
        );

        // `.` or `..` not the only chars is valid
        assert_eq!(NodePath::root().join("foo.").unwrap().as_str(), "/foo.");
        assert_eq!(NodePath::root().join("..foo").unwrap().as_str(), "/..foo");
        assert_eq!(NodePath::root().join("foo..").unwrap().as_str(), "/foo..");
    }

    #[test]
    fn node_path_parent() {
        // Root has no parent
        assert!(NodePath::root().parent().is_none());

        // Direct child of root
        assert_eq!(
            NodePath::new("/foo").unwrap().parent().unwrap().as_str(),
            "/"
        );

        // Nested
        assert_eq!(
            NodePath::new("/foo/bar")
                .unwrap()
                .parent()
                .unwrap()
                .as_str(),
            "/foo"
        );
        assert_eq!(
            NodePath::new("/foo/bar/baz")
                .unwrap()
                .parent()
                .unwrap()
                .as_str(),
            "/foo/bar"
        );
    }
}
