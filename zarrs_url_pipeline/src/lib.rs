//! URL pipeline support for the [`zarrs`](https://docs.rs/zarrs/latest/zarrs/index.html) crate.
//!
//! Implements the [`url-pipeline`](https://github.com/jbms/url-pipeline) specification: a
//! way to describe a storage resource as a single string, e.g.:
//!
//! ```text
//! s3://bucket/path/to/archive.zip|zip:path/within/zip.zarr/|zarr3:
//! ```
//!
//! A pipeline is a root URL (naming an absolute resource, e.g. `s3:`, `file:`, `memory:`)
//! optionally followed by one or more `|`-separated adapter URLs (each transforming the
//! resource produced by the previous stage, e.g. `zip:`). Resolution proceeds strictly
//! left-to-right: the root segment is resolved first, then each adapter is applied in turn to
//! the previously resolved stage. This order is not specified by the spec itself but is the
//! only sane reading of its examples.
//!
//! Schemes are matched against a plugin registry that supports both compile-time registration
//! (via [`inventory`], mirroring the pattern used for `zarrs` codecs and storage transformers)
//! and runtime registration (via [`register_root_store_scheme`]/[`register_adapter_store_scheme`]).
//! Runtime registrations always take precedence over compile-time ones. See [`root`] and
//! [`adapter`] for details, including how scheme collisions between compile-time plugins (e.g.
//! two crates both registering `s3:`) are handled.
//!
//! The `memory:` root scheme (backed by [`zarrs_storage::store::MemoryStore`]) is always
//! available; other schemes are provided by backend crates that opt in to a `url-pipeline`
//! feature (e.g. `zarrs_filesystem`'s `file:`, `zarrs_zip`'s `zip:`).
//!
//! ## Licence
//! `zarrs_url_pipeline` is licensed under either of
//! - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_url_pipeline/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//! - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_url_pipeline/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
#![cfg_attr(docsrs, feature(doc_cfg))]

pub mod adapter;
mod error;
mod memory;
pub mod parse;
pub mod root;
mod stage;

pub use adapter::{
    AdapterStorePlugin, AdapterStoreRuntimePlugin, AdapterStoreRuntimeRegistryHandle,
    register_adapter_store_scheme, unregister_adapter_store_scheme,
};
#[cfg(feature = "async")]
pub use adapter::{
    AsyncAdapterStorePlugin, AsyncAdapterStoreRuntimePlugin,
    AsyncAdapterStoreRuntimeRegistryHandle, register_async_adapter_store_scheme,
    unregister_async_adapter_store_scheme,
};
pub use error::{PipelineCreateError, PipelineError};
#[cfg(feature = "async")]
pub use root::{
    AsyncRootStorePlugin, AsyncRootStoreRuntimePlugin, AsyncRootStoreRuntimeRegistryHandle,
    register_async_root_store_scheme, unregister_async_root_store_scheme,
};
pub use root::{
    RootStorePlugin, RootStoreRuntimePlugin, RootStoreRuntimeRegistryHandle,
    register_root_store_scheme, unregister_root_store_scheme,
};
#[cfg(feature = "async")]
pub use stage::{AsyncPipelineStage, AsyncPipelineStageTraits};
pub use stage::{PipelineStage, PipelineStageTraits, unsupported};

use adapter::{AdapterStoreInput, try_create_adapter_stage};
#[cfg(feature = "async")]
use adapter::{AsyncAdapterStoreInput, try_create_async_adapter_stage};
#[cfg(feature = "async")]
use root::try_create_async_root_stage;
use root::{RootStoreInput, try_create_root_stage};

#[cfg(feature = "async")]
use zarrs_storage::{
    AsyncListableStorage, AsyncReadableListableStorage, AsyncReadableStorage,
    AsyncReadableWritableListableStorage, AsyncReadableWritableStorage, AsyncWritableStorage,
};
use zarrs_storage::{
    ListableStorage, ReadableListableStorage, ReadableStorage, ReadableWritableListableStorage,
    ReadableWritableStorage, WritableStorage,
};

/// Resolve a pipeline into its ordered synchronous stage, folding each adapter segment against
/// the previously resolved stage.
fn resolve_stage(pipeline: &str) -> Result<PipelineStage, PipelineError> {
    let segments = parse::parse_pipeline(pipeline)?;
    let (root, adapters) = segments
        .split_first()
        .expect("parse_pipeline returns at least one segment or an error");

    let mut stage = try_create_root_stage(&RootStoreInput {
        scheme: root.scheme.clone(),
        rest: root.rest.clone(),
    })?;

    for segment in adapters {
        stage = try_create_adapter_stage(
            &AdapterStoreInput {
                scheme: segment.scheme.clone(),
                rest: segment.rest.clone(),
            },
            &stage,
        )?;
    }

    Ok(stage)
}

/// Resolve a pipeline into its ordered asynchronous stage, folding each adapter segment against
/// the previously resolved stage.
#[cfg(feature = "async")]
fn resolve_async_stage(pipeline: &str) -> Result<AsyncPipelineStage, PipelineError> {
    let segments = parse::parse_pipeline(pipeline)?;
    let (root, adapters) = segments
        .split_first()
        .expect("parse_pipeline returns at least one segment or an error");

    let mut stage = try_create_async_root_stage(&RootStoreInput {
        scheme: root.scheme.clone(),
        rest: root.rest.clone(),
    })?;

    for segment in adapters {
        stage = try_create_async_adapter_stage(
            &AsyncAdapterStoreInput {
                scheme: segment.scheme.clone(),
                rest: segment.rest.clone(),
            },
            &stage,
        )?;
    }

    Ok(stage)
}

/// Resolve a URL pipeline into readable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
pub fn try_resolve_readable(pipeline: &str) -> Result<ReadableStorage, PipelineError> {
    Ok(resolve_stage(pipeline)?.as_readable()?)
}

/// Resolve a URL pipeline into writable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
pub fn try_resolve_writable(pipeline: &str) -> Result<WritableStorage, PipelineError> {
    Ok(resolve_stage(pipeline)?.as_writable()?)
}

/// Resolve a URL pipeline into listable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
pub fn try_resolve_listable(pipeline: &str) -> Result<ListableStorage, PipelineError> {
    Ok(resolve_stage(pipeline)?.as_listable()?)
}

/// Resolve a URL pipeline into readable and writable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
pub fn try_resolve_readable_writable(
    pipeline: &str,
) -> Result<ReadableWritableStorage, PipelineError> {
    Ok(resolve_stage(pipeline)?.as_readable_writable()?)
}

/// Resolve a URL pipeline into readable and listable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
pub fn try_resolve_readable_listable(
    pipeline: &str,
) -> Result<ReadableListableStorage, PipelineError> {
    Ok(resolve_stage(pipeline)?.as_readable_listable()?)
}

/// Resolve a URL pipeline into readable, writable, and listable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
pub fn try_resolve_readable_writable_listable(
    pipeline: &str,
) -> Result<ReadableWritableListableStorage, PipelineError> {
    Ok(resolve_stage(pipeline)?.as_readable_writable_listable()?)
}

/// Resolve a URL pipeline into asynchronous readable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
#[cfg(feature = "async")]
pub fn try_resolve_async_readable(pipeline: &str) -> Result<AsyncReadableStorage, PipelineError> {
    Ok(resolve_async_stage(pipeline)?.as_async_readable()?)
}

/// Resolve a URL pipeline into asynchronous writable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
#[cfg(feature = "async")]
pub fn try_resolve_async_writable(pipeline: &str) -> Result<AsyncWritableStorage, PipelineError> {
    Ok(resolve_async_stage(pipeline)?.as_async_writable()?)
}

/// Resolve a URL pipeline into asynchronous listable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
#[cfg(feature = "async")]
pub fn try_resolve_async_listable(pipeline: &str) -> Result<AsyncListableStorage, PipelineError> {
    Ok(resolve_async_stage(pipeline)?.as_async_listable()?)
}

/// Resolve a URL pipeline into asynchronous readable and writable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
#[cfg(feature = "async")]
pub fn try_resolve_async_readable_writable(
    pipeline: &str,
) -> Result<AsyncReadableWritableStorage, PipelineError> {
    Ok(resolve_async_stage(pipeline)?.as_async_readable_writable()?)
}

/// Resolve a URL pipeline into asynchronous readable and listable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
#[cfg(feature = "async")]
pub fn try_resolve_async_readable_listable(
    pipeline: &str,
) -> Result<AsyncReadableListableStorage, PipelineError> {
    Ok(resolve_async_stage(pipeline)?.as_async_readable_listable()?)
}

/// Resolve a URL pipeline into asynchronous readable, writable, and listable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
#[cfg(feature = "async")]
pub fn try_resolve_async_readable_writable_listable(
    pipeline: &str,
) -> Result<AsyncReadableWritableListableStorage, PipelineError> {
    Ok(resolve_async_stage(pipeline)?.as_async_readable_writable_listable()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use zarrs_storage::{StoreKey, WritableStorageTraits};

    #[test]
    fn resolves_memory_root_readable_writable() {
        let storage = try_resolve_readable_writable("memory://").unwrap();
        let key = StoreKey::new("a/b").unwrap();
        storage
            .set(&key, zarrs_storage::Bytes::from_static(b"hello"))
            .unwrap();
        assert_eq!(
            storage.get(&key).unwrap(),
            Some(zarrs_storage::Bytes::from_static(b"hello"))
        );
    }

    #[test]
    fn each_resolution_of_memory_is_a_fresh_store() {
        let a = try_resolve_writable("memory://").unwrap();
        let b = try_resolve_readable("memory://").unwrap();
        let key = StoreKey::new("k").unwrap();
        a.set(&key, zarrs_storage::Bytes::from_static(b"x"))
            .unwrap();
        // `b` is a different store instance, so it must not see `a`'s write.
        assert_eq!(b.get(&key).unwrap(), None);
    }

    #[test]
    fn empty_pipeline_errors() {
        assert!(matches!(
            try_resolve_readable(""),
            Err(PipelineError::EmptyPipeline)
        ));
    }

    #[test]
    fn unregistered_scheme_errors() {
        assert!(try_resolve_readable("does-not-exist://foo").is_err());
    }
}
