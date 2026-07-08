//! Registers `object_store`-backed schemes (`s3:`, `gs:`, `az:`, `http:`, ...) as async root
//! schemes for `zarrs_url_pipeline`, delegating scheme dispatch to `object_store::parse_url_opts`.
//!
//! `memory:` and `file:` are deliberately not registered here: `memory:` is a built-in scheme in
//! `zarrs_url_pipeline` itself, and `file:` is owned by `zarrs_filesystem`.

use std::sync::Arc;

use object_store::ObjectStoreScheme;
use object_store::prefix::PrefixStore;
use zarrs_storage::{
    AsyncListableStorage, AsyncReadableListableStorage, AsyncReadableStorage,
    AsyncReadableWritableListableStorage, AsyncReadableWritableStorage, AsyncWritableStorage,
    StorageError,
};
use zarrs_url_pipeline::root::RootStoreInput;
use zarrs_url_pipeline::{
    AsyncPipelineStage, AsyncPipelineStageTraits, AsyncRootStorePlugin, PipelineCreateError,
};

use crate::AsyncObjectStore;

/// `AsyncObjectStore` supports every asynchronous capability unconditionally, regardless of the
/// underlying `object_store::ObjectStore` implementation.
impl AsyncPipelineStageTraits
    for AsyncObjectStore<PrefixStore<Box<dyn object_store::ObjectStore>>>
{
    fn as_async_readable(self: Arc<Self>) -> Result<AsyncReadableStorage, StorageError> {
        Ok(self)
    }

    fn as_async_writable(self: Arc<Self>) -> Result<AsyncWritableStorage, StorageError> {
        Ok(self)
    }

    fn as_async_listable(self: Arc<Self>) -> Result<AsyncListableStorage, StorageError> {
        Ok(self)
    }

    fn as_async_readable_writable(
        self: Arc<Self>,
    ) -> Result<AsyncReadableWritableStorage, StorageError> {
        Ok(self)
    }

    fn as_async_readable_listable(
        self: Arc<Self>,
    ) -> Result<AsyncReadableListableStorage, StorageError> {
        Ok(self)
    }

    fn as_async_readable_writable_listable(
        self: Arc<Self>,
    ) -> Result<AsyncReadableWritableListableStorage, StorageError> {
        Ok(self)
    }
}

fn is_object_store(input: &RootStoreInput) -> bool {
    let full = format!("{}:{}", input.scheme, input.rest);
    let Ok(url) = url::Url::parse(&full) else {
        return false;
    };
    let Ok((scheme, _path)) = ObjectStoreScheme::parse(&url) else {
        return false;
    };
    #[allow(clippy::match_like_matches_macro)]
    match scheme {
        ObjectStoreScheme::Local => cfg!(feature = "fs"),
        ObjectStoreScheme::Memory => true,
        ObjectStoreScheme::AmazonS3 => cfg!(feature = "aws"),
        ObjectStoreScheme::GoogleCloudStorage => cfg!(feature = "gcp"),
        ObjectStoreScheme::MicrosoftAzure => cfg!(feature = "azure"),
        ObjectStoreScheme::Http => cfg!(feature = "http"),
        _ => false,
    }
}

fn create_object_store(
    input: &RootStoreInput,
) -> futures::future::BoxFuture<'static, Result<AsyncPipelineStage, PipelineCreateError>> {
    let input = input.clone();
    Box::pin(async move {
        let full = format!("{}:{}", input.scheme, input.rest);
        let url = url::Url::parse(&full).map_err(|e| PipelineCreateError::InvalidSegment {
            scheme: input.scheme.clone(),
            rest: input.rest.clone(),
            reason: format!("invalid {}: url: {e}", input.scheme),
        })?;
        let (store, path) = object_store::parse_url_opts(&url, std::iter::empty::<(&str, &str)>())
            .map_err(PipelineCreateError::other)?;
        let store = PrefixStore::new(store, path);
        Ok(Arc::new(AsyncObjectStore::new(store)) as AsyncPipelineStage)
    })
}

inventory::submit! {
    AsyncRootStorePlugin::new("zarrs_object_store", is_object_store, create_object_store)
}
