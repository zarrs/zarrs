//! Built-in `memory:` root scheme, backed by [`zarrs_storage::store::MemoryStore`].
//!
//! Unlike other backends (`zarrs_filesystem`, `zarrs_object_store`, ...), `MemoryStore` lives in
//! `zarrs_storage` itself, which is deliberately kept free of plugin/URL dependencies. Since this
//! crate already depends on `zarrs_storage` unconditionally, `memory:` is registered here instead
//! as an always-available built-in, rather than adding an optional feature to `zarrs_storage`.

use std::sync::Arc;

use zarrs_storage::store::MemoryStore;
use zarrs_storage::{
    ListableStorage, ReadableListableStorage, ReadableStorage, ReadableWritableListableStorage,
    ReadableWritableStorage, StorageError, WritableStorage,
};

use crate::error::PipelineCreateError;
use crate::root::{RootStoreInput, RootStorePlugin};
use crate::stage::{PipelineStage, PipelineStageTraits};

// `MemoryStore` is sync-only, so it implements `PipelineStageTraits` only — there is no
// `AsyncPipelineStageTraits` impl (and none is needed, since sync and async pipelines are
// resolved entirely independently; see `crate::root`).
impl PipelineStageTraits for MemoryStore {
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

// The `Result` return type is required by `RootStorePlugin::new`'s `create_fn` signature, even
// though constructing a `MemoryStore` cannot fail.
#[allow(clippy::unnecessary_wraps)]
fn create_memory_store(_input: &RootStoreInput) -> Result<PipelineStage, PipelineCreateError> {
    Ok(Arc::new(MemoryStore::new()))
}

inventory::submit! {
    RootStorePlugin::new("zarrs_url_pipeline", |s| s.scheme == "memory", create_memory_store)
}
