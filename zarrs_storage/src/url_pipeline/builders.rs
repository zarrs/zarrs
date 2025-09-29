//! Store builders for URL pipeline system.
//!
//! This module provides the infrastructure for creating stores from URL pipelines.
//! Individual store crates (`zarrs_filesystem`, `zarrs_http`, etc.) should register
//! themselves using the provided registration functions.

use super::{UrlPipeline, UrlPipelineError};
use crate::{store::MemoryStore, ReadableStorage, StoreKey};
use std::sync::Arc;

#[cfg(feature = "async")]
use crate::AsyncReadableStorage;

/// Create a store from a URL pipeline.
///
/// # Errors
///
/// Returns an error if the root store or any adapter cannot be created.
pub fn create_store(pipeline: &UrlPipeline) -> Result<ReadableStorage, UrlPipelineError> {
    let registry = super::registry::get_global_registry();

    // Create root store
    let root_builder = registry
        .get_root_builder(&pipeline.root.scheme)
        .ok_or_else(|| {
            UrlPipelineError::UnsupportedScheme(format!(
                "no root store registered for scheme '{}'",
                pipeline.root.scheme
            ))
        })?;

    let mut store = root_builder(&pipeline.root)?;

    // Apply adapters in order
    for adapter in &pipeline.adapters {
        let adapter_builder = registry
            .get_adapter_builder(&adapter.scheme)
            .ok_or_else(|| {
                UrlPipelineError::UnsupportedScheme(format!(
                    "no adapter registered for scheme '{}'",
                    adapter.scheme
                ))
            })?;

        store = adapter_builder(store, adapter)?;
    }

    Ok(store)
}

#[cfg(feature = "async")]
/// Create an async store from a URL pipeline.
///
/// # Errors
///
/// Returns an error if the root store or any adapter cannot be created.
pub fn create_async_store(
    pipeline: &UrlPipeline,
) -> Result<AsyncReadableStorage, UrlPipelineError> {
    let registry = super::registry::get_global_registry();

    // Create root store
    let root_builder = registry
        .get_async_root_builder(&pipeline.root.scheme)
        .ok_or_else(|| {
            UrlPipelineError::UnsupportedScheme(format!(
                "no async root store registered for scheme '{}'",
                pipeline.root.scheme
            ))
        })?;

    let mut store = root_builder(&pipeline.root)?;

    // Apply adapters in order
    for adapter in &pipeline.adapters {
        let adapter_builder = registry
            .get_async_adapter_builder(&adapter.scheme)
            .ok_or_else(|| {
                UrlPipelineError::UnsupportedScheme(format!(
                    "no async adapter registered for scheme '{}'",
                    adapter.scheme
                ))
            })?;

        store = adapter_builder(store, adapter)?;
    }

    Ok(store)
}

/// Initialize built-in store builders.
///
/// This function is called automatically when the global registry is first accessed.
/// It registers the built-in memory store as an example.
///
/// # Panics
///
/// May panic if the registry lock is poisoned.
#[doc(hidden)]
pub fn init_builtin_stores(registry: &super::registry::StoreRegistry) {
    // Register memory store (example) - use registry directly to avoid deadlock
    registry.register_root_store("memory", |_component| Ok(Arc::new(MemoryStore::new())));

    // Note: MemoryStore doesn't implement AsyncReadableStorageTraits directly
    // Users would need to wrap it with SyncToAsyncStorageAdapter if needed
}

/// Helper function to parse a store key from a path component.
///
/// # Errors
///
/// Returns an error if the path is not a valid store key.
#[allow(dead_code)]
fn parse_store_key(path: &str) -> Result<StoreKey, UrlPipelineError> {
    StoreKey::try_from(path)
        .map_err(|e| UrlPipelineError::StoreCreationFailed(format!("invalid store key: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::url_pipeline::parse_url_pipeline;

    #[test]
    fn test_create_memory_store() {
        let pipeline = parse_url_pipeline("memory://").unwrap();
        let store = create_store(&pipeline).unwrap();
        // Just verify that the store was created successfully
        assert!(store.get(&"test".try_into().unwrap()).is_ok());
    }
}
