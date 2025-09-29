//! URL pipeline syntax for Zarr stores.
//!
//! This module implements the URL pipeline syntax enhancement proposal (ZEP0008) for Zarr.
//! It allows specifying complex, nested data resource locations across different storage systems
//! and formats using a pipeline syntax with `|` as a delimiter.
//!
//! # Examples
//!
//! ```ignore
//! // Local filesystem
//! let store = url_pipeline::parse_and_create("file:///path/to/data")?;
//!
//! // HTTP store
//! let store = url_pipeline::parse_and_create("http://example.com/data")?;
//!
//! // ZIP file on filesystem
//! let store = url_pipeline::parse_and_create("file:///path/to/data.zip|zip:")?;
//!
//! // Nested: ZIP inside ZIP
//! let store = url_pipeline::parse_and_create("file:///path/to/outer.zip|zip:inner.zip|zip:path")?;
//!
//! // Cloud storage with ZIP
//! let store = url_pipeline::parse_and_create("s3://bucket/data.zip|zip:array")?;
//! ```

mod builders;
mod parser;
mod registry;

pub use builders::{create_store, init_builtin_stores};
pub use parser::{parse_url_pipeline, UrlComponent, UrlPipeline};
pub use registry::{
    get_global_registry, register_adapter_store, register_root_store, StoreRegistry,
};

#[cfg(feature = "async")]
pub use builders::create_async_store;

#[cfg(feature = "async")]
pub use registry::{register_async_adapter_store, register_async_root_store};

use crate::StorageError;

/// Error type for URL pipeline operations.
#[derive(Debug, thiserror::Error)]
pub enum UrlPipelineError {
    /// Invalid URL syntax
    #[error("invalid URL syntax: {0}")]
    InvalidUrl(String),

    /// Unsupported URL scheme
    #[error("unsupported URL scheme: {0}")]
    UnsupportedScheme(String),

    /// Invalid pipeline structure
    #[error("invalid pipeline structure: {0}")]
    InvalidPipeline(String),

    /// Store creation failed
    #[error("store creation failed: {0}")]
    StoreCreationFailed(String),

    /// Storage error
    #[error(transparent)]
    StorageError(#[from] StorageError),

    /// Other error
    #[error("{0}")]
    Other(String),
}

impl From<url::ParseError> for UrlPipelineError {
    fn from(err: url::ParseError) -> Self {
        Self::InvalidUrl(err.to_string())
    }
}

impl From<crate::StoreKeyError> for UrlPipelineError {
    fn from(err: crate::StoreKeyError) -> Self {
        Self::StoreCreationFailed(err.to_string())
    }
}

impl From<String> for UrlPipelineError {
    fn from(err: String) -> Self {
        Self::Other(err)
    }
}

impl From<&str> for UrlPipelineError {
    fn from(err: &str) -> Self {
        Self::Other(err.to_string())
    }
}

/// Parse a URL pipeline string and create a store.
///
/// # Errors
///
/// Returns an error if the URL is invalid or store creation fails.
pub fn parse_and_create(url: &str) -> Result<crate::ReadableStorage, UrlPipelineError> {
    let pipeline = parse_url_pipeline(url)?;
    create_store(&pipeline)
}

/// Parse a URL pipeline string and create an async store.
///
/// # Errors
///
/// Returns an error if the URL is invalid or store creation fails.
#[cfg(feature = "async")]
pub fn parse_and_create_async(url: &str) -> Result<crate::AsyncReadableStorage, UrlPipelineError> {
    let pipeline = parse_url_pipeline(url)?;
    create_async_store(&pipeline)
}
