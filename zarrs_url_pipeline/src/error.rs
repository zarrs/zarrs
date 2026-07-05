use thiserror::Error;
use zarrs_plugin::PluginUnsupportedError;
use zarrs_storage::StorageError;

/// A pipeline stage creation error.
#[derive(Debug, Clone, Error)]
#[allow(missing_docs)]
pub enum PipelineCreateError {
    /// No plugin is registered for a root or adapter scheme.
    #[error(transparent)]
    Unsupported(#[from] PluginUnsupportedError),
    /// A pipeline segment's scheme-specific part could not be interpreted as this scheme
    /// expects (e.g. not a valid URL, missing required authority).
    #[error("invalid {scheme} pipeline segment {rest:?}: {reason}")]
    InvalidSegment {
        /// The scheme of the segment that failed to parse.
        scheme: String,
        /// The raw scheme-specific part.
        rest: String,
        /// Why `rest` could not be interpreted.
        reason: String,
    },
    /// Other.
    #[error("{_0}")]
    Other(String),
}

impl PipelineCreateError {
    /// Create a new [`PipelineCreateError::Other`] from a displayable error or message.
    #[must_use]
    pub fn other(error: impl ToString) -> Self {
        Self::Other(error.to_string())
    }
}

impl From<&str> for PipelineCreateError {
    fn from(error: &str) -> Self {
        Self::Other(error.to_string())
    }
}

impl From<String> for PipelineCreateError {
    fn from(error: String) -> Self {
        Self::Other(error)
    }
}

/// An error resolving a URL pipeline.
#[derive(Debug, Clone, Error)]
#[allow(missing_docs)]
pub enum PipelineError {
    /// The pipeline string is empty.
    #[error("the url pipeline is empty")]
    EmptyPipeline,
    /// A pipeline segment could not be parsed.
    #[error("invalid pipeline segment {0:?}: {1}")]
    InvalidSegment(String, String),
    /// No stage plugin could be resolved or constructed for a segment.
    #[error(transparent)]
    PipelineCreateError(#[from] PipelineCreateError),
    /// The resolved storage does not support the requested capability.
    #[error(transparent)]
    StorageError(#[from] StorageError),
}
