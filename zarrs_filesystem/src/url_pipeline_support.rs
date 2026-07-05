//! Registers `FilesystemStore` as the `file:` root scheme for `zarrs_url_pipeline`.

use std::sync::Arc;

use zarrs_storage::{
    ListableStorage, ReadableListableStorage, ReadableStorage, ReadableWritableListableStorage,
    ReadableWritableStorage, StorageError, WritableStorage,
};
use zarrs_url_pipeline::root::RootStoreInput;
use zarrs_url_pipeline::{
    PipelineCreateError, PipelineStage, PipelineStageTraits, RootStorePlugin,
};

use crate::FilesystemStore;

// `FilesystemStore` is sync-only, so it implements `PipelineStageTraits` only — there is no
// `AsyncPipelineStageTraits` impl (and none is needed, since sync and async pipelines are
// resolved entirely independently).
impl PipelineStageTraits for FilesystemStore {
    fn as_readable(self: Arc<Self>) -> Result<ReadableStorage, StorageError> {
        Ok(self)
    }

    fn as_writable(self: Arc<Self>) -> Result<WritableStorage, StorageError> {
        Ok(self)
    }

    fn as_listable(self: Arc<Self>) -> Result<ListableStorage, StorageError> {
        Ok(self)
    }

    fn as_readable_writable(self: Arc<Self>) -> Result<ReadableWritableStorage, StorageError> {
        Ok(self)
    }

    fn as_readable_listable(self: Arc<Self>) -> Result<ReadableListableStorage, StorageError> {
        Ok(self)
    }

    fn as_readable_writable_listable(
        self: Arc<Self>,
    ) -> Result<ReadableWritableListableStorage, StorageError> {
        Ok(self)
    }
}

fn create_filesystem_store(input: &RootStoreInput) -> Result<PipelineStage, PipelineCreateError> {
    // `rest` is everything after `file:`, e.g. `///tmp/dataset` for `file:///tmp/dataset`.
    // Parse via `url::Url` (reconstructing a parseable absolute URL) to correctly handle the
    // `file://[host]/path` authority form and percent-decoding.
    let full = format!("file:{}", input.rest);
    let url = url::Url::parse(&full).map_err(|e| PipelineCreateError::InvalidSegment {
        scheme: "file".to_string(),
        rest: input.rest.clone(),
        reason: format!("invalid file: url: {e}"),
    })?;
    let path = url
        .to_file_path()
        .map_err(|()| PipelineCreateError::InvalidSegment {
            scheme: "file".to_string(),
            rest: input.rest.clone(),
            reason: format!("invalid file: path: {full}"),
        })?;
    let store = FilesystemStore::new(path).map_err(PipelineCreateError::other)?;
    Ok(Arc::new(store))
}

inventory::submit! {
    RootStorePlugin::new("zarrs_filesystem", |s| s.scheme == "file", create_filesystem_store)
}
