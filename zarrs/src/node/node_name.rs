use thiserror::Error;

/// A Zarr hierarchy node name.
///
/// See
/// - <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#name>, and
/// - <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#node-names>.
#[derive(Debug, PartialEq, Eq, derive_more::Display)]
pub struct NodeName(String);

/// An invalid node name.
#[derive(Clone, Debug, Error)]
#[error("invalid node name {0}")]
pub struct NodeNameError(String);

impl NodeName {
    /// Create a new Zarr node name from `name`.
    ///
    /// # Errors
    ///
    /// Returns [`NodeNameError`] if `name` is not valid according to [`NodeName::validate`()].
    pub fn new(name: impl Into<String>) -> Result<Self, NodeNameError> {
        let name = name.into();
        if Self::validate(&name) {
            Ok(Self(name))
        } else {
            Err(NodeNameError(name))
        }
    }

    /// Create a new Zarr node name from `name`.
    ///
    /// # Safety
    ///
    /// `name` is not validated, so this can result in an invalid node name.
    #[must_use]
    pub unsafe fn new_unchecked(name: impl Into<String>) -> Self {
        let name: String = name.into();
        debug_assert!(Self::validate(&name));
        Self(name)
    }

    /// The root node.
    #[must_use]
    pub const fn root() -> Self {
        Self(String::new())
    }

    /// Extracts a string slice containing the node name `String`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Validates a node name according to the following rules from the specification:
    /// - The root node does not have a name and is the empty string "".
    ///
    /// Otherwise,
    /// - must not be the empty string (""),
    /// - must not include the character "/",
    /// - must not be a string composed only of period characters, e.g. "." or "..", and
    /// - must not start with the reserved prefix "__".
    #[must_use]
    pub fn validate(node_name: &str) -> bool {
        node_name.is_empty()
            || (!node_name.contains('/')
                && !node_name.starts_with("__")
                && !node_name.replace('.', "").is_empty())
    }

    /// Indicates if a node has the root node name ("").
    #[must_use]
    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_name() {
        assert_eq!(NodeName::root(), NodeName::new("").unwrap());
        assert!(NodeName::new("").unwrap().is_root());
        assert!(NodeName::new("a").is_ok());
        assert_eq!(NodeName::new("a").unwrap().to_string(), "a");
        assert!(NodeName::new("a/b").is_err());
        assert_eq!(
            NodeName::new("a/b").unwrap_err().to_string(),
            "invalid node name a/b"
        );
        assert!(NodeName::new("__").is_err());
        assert!(NodeName::new(".").is_err());
        assert!(NodeName::new("..").is_err());
    }
}
