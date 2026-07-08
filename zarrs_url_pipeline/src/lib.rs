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
    AsyncAdapterStoreInput, AsyncAdapterStorePlugin, AsyncAdapterStoreRuntimePlugin,
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

#[cfg(feature = "async")]
use adapter::try_create_async_adapter_stage;
use adapter::{AdapterStoreInput, try_create_adapter_stage};
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
async fn resolve_async_stage(pipeline: &str) -> Result<AsyncPipelineStage, PipelineError> {
    let segments = parse::parse_pipeline(pipeline)?;
    let (root, adapters) = segments
        .split_first()
        .expect("parse_pipeline returns at least one segment or an error");

    let mut stage = match try_create_async_root_stage(&RootStoreInput {
        scheme: root.scheme.clone(),
        rest: root.rest.clone(),
    })
    .await
    {
        Ok(stage) => Some(stage),
        Err(PipelineCreateError::Unsupported(_)) if !adapters.is_empty() => None,
        Err(error) => return Err(error.into()),
    };

    for (adapter_index, segment) in adapters.iter().enumerate() {
        let previous_segments = segments[..adapter_index + 1].to_vec();
        stage = Some(
            try_create_async_adapter_stage(&adapter::AsyncAdapterStoreInput {
                scheme: segment.scheme.clone(),
                rest: segment.rest.clone(),
                previous_segments,
                previous_stage: stage.clone(),
            })
            .await?,
        );
    }

    stage.ok_or_else(|| {
        PipelineCreateError::from(zarrs_plugin::PluginUnsupportedError::new(
            root.scheme.clone(),
            "async root store scheme".to_string(),
        ))
        .into()
    })
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
pub async fn try_resolve_async_readable(
    pipeline: &str,
) -> Result<AsyncReadableStorage, PipelineError> {
    Ok(resolve_async_stage(pipeline).await?.as_async_readable()?)
}

/// Resolve a URL pipeline into asynchronous writable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
#[cfg(feature = "async")]
pub async fn try_resolve_async_writable(
    pipeline: &str,
) -> Result<AsyncWritableStorage, PipelineError> {
    Ok(resolve_async_stage(pipeline).await?.as_async_writable()?)
}

/// Resolve a URL pipeline into asynchronous listable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
#[cfg(feature = "async")]
pub async fn try_resolve_async_listable(
    pipeline: &str,
) -> Result<AsyncListableStorage, PipelineError> {
    Ok(resolve_async_stage(pipeline).await?.as_async_listable()?)
}

/// Resolve a URL pipeline into asynchronous readable and writable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
#[cfg(feature = "async")]
pub async fn try_resolve_async_readable_writable(
    pipeline: &str,
) -> Result<AsyncReadableWritableStorage, PipelineError> {
    Ok(resolve_async_stage(pipeline)
        .await?
        .as_async_readable_writable()?)
}

/// Resolve a URL pipeline into asynchronous readable and listable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
#[cfg(feature = "async")]
pub async fn try_resolve_async_readable_listable(
    pipeline: &str,
) -> Result<AsyncReadableListableStorage, PipelineError> {
    Ok(resolve_async_stage(pipeline)
        .await?
        .as_async_readable_listable()?)
}

/// Resolve a URL pipeline into asynchronous readable, writable, and listable storage.
///
/// # Errors
/// Returns [`PipelineError`] if the pipeline cannot be parsed, a scheme is not registered, a
/// stage fails to construct, or the resolved storage does not support this capability.
#[cfg(feature = "async")]
pub async fn try_resolve_async_readable_writable_listable(
    pipeline: &str,
) -> Result<AsyncReadableWritableListableStorage, PipelineError> {
    Ok(resolve_async_stage(pipeline)
        .await?
        .as_async_readable_writable_listable()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::{Arc, Mutex};
    use zarrs_storage::{StoreKey, WritableStorageTraits};

    #[cfg(feature = "async")]
    #[derive(Debug, Default)]
    struct TestAsyncStore(Mutex<BTreeMap<StoreKey, zarrs_storage::Bytes>>);

    #[cfg(feature = "async")]
    #[async_trait::async_trait]
    impl zarrs_storage::AsyncReadableStorageTraits for TestAsyncStore {
        async fn get(
            &self,
            key: &StoreKey,
        ) -> Result<zarrs_storage::MaybeBytes, zarrs_storage::StorageError> {
            Ok(self.0.lock().unwrap().get(key).cloned())
        }

        async fn get_partial_many<'a>(
            &'a self,
            key: &StoreKey,
            byte_ranges: zarrs_storage::byte_range::ByteRangeIterator<'a>,
        ) -> Result<zarrs_storage::AsyncMaybeBytesIterator<'a>, zarrs_storage::StorageError>
        {
            let _ = (key, byte_ranges);
            Err(zarrs_storage::StorageError::Unsupported(
                "partial reads are not supported".to_string(),
            ))
        }

        async fn size_key(
            &self,
            key: &StoreKey,
        ) -> Result<Option<u64>, zarrs_storage::StorageError> {
            Ok(self
                .0
                .lock()
                .unwrap()
                .get(key)
                .map(|value| value.len() as u64))
        }

        fn supports_get_partial(&self) -> bool {
            false
        }
    }

    #[cfg(feature = "async")]
    #[async_trait::async_trait]
    impl zarrs_storage::AsyncWritableStorageTraits for TestAsyncStore {
        async fn set(
            &self,
            key: &StoreKey,
            value: zarrs_storage::Bytes,
        ) -> Result<(), zarrs_storage::StorageError> {
            self.0.lock().unwrap().insert(key.clone(), value);
            Ok(())
        }

        async fn set_partial_many<'a>(
            &'a self,
            _key: &StoreKey,
            _offset_values: zarrs_storage::OffsetBytesIterator<'a>,
        ) -> Result<(), zarrs_storage::StorageError> {
            Err(zarrs_storage::StorageError::Unsupported(
                "partial writes are not supported".to_string(),
            ))
        }

        async fn erase(&self, key: &StoreKey) -> Result<(), zarrs_storage::StorageError> {
            self.0.lock().unwrap().remove(key);
            Ok(())
        }

        async fn erase_prefix(
            &self,
            prefix: &zarrs_storage::StorePrefix,
        ) -> Result<(), zarrs_storage::StorageError> {
            self.0
                .lock()
                .unwrap()
                .retain(|key, _| !key.has_prefix(prefix));
            Ok(())
        }

        fn supports_set_partial(&self) -> bool {
            false
        }
    }

    #[cfg(feature = "async")]
    impl AsyncPipelineStageTraits for TestAsyncStore {
        fn as_async_readable(
            self: Arc<Self>,
        ) -> Result<zarrs_storage::AsyncReadableStorage, zarrs_storage::StorageError> {
            Ok(self)
        }

        fn as_async_writable(
            self: Arc<Self>,
        ) -> Result<zarrs_storage::AsyncWritableStorage, zarrs_storage::StorageError> {
            Ok(self)
        }

        fn as_async_listable(
            self: Arc<Self>,
        ) -> Result<zarrs_storage::AsyncListableStorage, zarrs_storage::StorageError> {
            Err(unsupported("async listable"))
        }

        fn as_async_readable_writable(
            self: Arc<Self>,
        ) -> Result<zarrs_storage::AsyncReadableWritableStorage, zarrs_storage::StorageError>
        {
            Ok(self)
        }

        fn as_async_readable_listable(
            self: Arc<Self>,
        ) -> Result<zarrs_storage::AsyncReadableListableStorage, zarrs_storage::StorageError>
        {
            Err(unsupported("async readable listable"))
        }

        fn as_async_readable_writable_listable(
            self: Arc<Self>,
        ) -> Result<zarrs_storage::AsyncReadableWritableListableStorage, zarrs_storage::StorageError>
        {
            Err(unsupported("async readable writable listable"))
        }
    }

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

    #[cfg(feature = "async")]
    #[test]
    fn async_context_adapter_can_consume_unresolved_root() {
        use crate::adapter::{
            AsyncAdapterStoreInput, AsyncAdapterStoreRuntimePlugin,
            register_async_adapter_store_scheme, unregister_async_adapter_store_scheme,
        };
        use crate::stage::AsyncPipelineStage;

        let handle = register_async_adapter_store_scheme(AsyncAdapterStoreRuntimePlugin::new(
            |s| s == "test-context",
            |input: &AsyncAdapterStoreInput| {
                assert!(input.previous_stage.is_none());
                assert_eq!(input.previous_segments.len(), 1);
                assert_eq!(input.previous_segments[0].scheme, "file");
                Box::pin(async { Ok(Arc::new(TestAsyncStore::default()) as AsyncPipelineStage) })
            },
        ));

        let storage = futures::executor::block_on(try_resolve_async_readable_writable(
            "file:///tmp/example.icechunk|test-context:",
        ))
        .unwrap();
        let key = StoreKey::new("a/b").unwrap();
        futures::executor::block_on(storage.set(&key, zarrs_storage::Bytes::from_static(b"hello")))
            .unwrap();
        assert_eq!(
            futures::executor::block_on(storage.get(&key)).unwrap(),
            Some(zarrs_storage::Bytes::from_static(b"hello"))
        );

        assert!(unregister_async_adapter_store_scheme(&handle));
    }

    #[cfg(feature = "async")]
    #[test]
    fn async_zarr3_adapter_is_passthrough() {
        let handle = crate::root::register_async_root_store_scheme(
            crate::root::AsyncRootStoreRuntimePlugin::new(
                |input| input.scheme == "test-async-memory",
                |_input| {
                    Box::pin(async {
                        Ok(Arc::new(TestAsyncStore::default()) as AsyncPipelineStage)
                    })
                },
            ),
        );

        let storage = futures::executor::block_on(try_resolve_async_readable_writable(
            "test-async-memory://|zarr3:",
        ))
        .unwrap();
        let key = StoreKey::new("a/b").unwrap();
        futures::executor::block_on(storage.set(&key, zarrs_storage::Bytes::from_static(b"hello")))
            .unwrap();
        assert_eq!(
            futures::executor::block_on(storage.get(&key)).unwrap(),
            Some(zarrs_storage::Bytes::from_static(b"hello"))
        );

        assert!(crate::root::unregister_async_root_store_scheme(&handle));
    }
}
