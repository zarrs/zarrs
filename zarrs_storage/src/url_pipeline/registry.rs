//! Store registry for URL pipeline system.

use super::{UrlComponent, UrlPipelineError};
use crate::ReadableStorage;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

#[cfg(feature = "async")]
use crate::AsyncReadableStorage;

/// A function that creates a root store from a URL component.
pub(super) type RootStoreBuilder =
    Arc<dyn Fn(&UrlComponent) -> Result<ReadableStorage, UrlPipelineError> + Send + Sync>;

/// A function that creates an adapter store from a parent store and URL component.
pub(super) type AdapterStoreBuilder = Arc<
    dyn Fn(ReadableStorage, &UrlComponent) -> Result<ReadableStorage, UrlPipelineError>
        + Send
        + Sync,
>;

#[cfg(feature = "async")]
/// A function that creates an async root store from a URL component.
pub(super) type AsyncRootStoreBuilder =
    Arc<dyn Fn(&UrlComponent) -> Result<AsyncReadableStorage, UrlPipelineError> + Send + Sync>;

#[cfg(feature = "async")]
/// A function that creates an async adapter store from a parent store and URL component.
pub(super) type AsyncAdapterStoreBuilder = Arc<
    dyn Fn(AsyncReadableStorage, &UrlComponent) -> Result<AsyncReadableStorage, UrlPipelineError>
        + Send
        + Sync,
>;

/// Registry for store builders.
#[allow(clippy::struct_field_names)]
pub struct StoreRegistry {
    root_builders: RwLock<HashMap<String, RootStoreBuilder>>,
    adapter_builders: RwLock<HashMap<String, AdapterStoreBuilder>>,
    #[cfg(feature = "async")]
    async_root_builders: RwLock<HashMap<String, AsyncRootStoreBuilder>>,
    #[cfg(feature = "async")]
    async_adapter_builders: RwLock<HashMap<String, AsyncAdapterStoreBuilder>>,
}

impl StoreRegistry {
    /// Create a new registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            root_builders: RwLock::new(HashMap::new()),
            adapter_builders: RwLock::new(HashMap::new()),
            #[cfg(feature = "async")]
            async_root_builders: RwLock::new(HashMap::new()),
            #[cfg(feature = "async")]
            async_adapter_builders: RwLock::new(HashMap::new()),
        }
    }

    /// Register a root store builder.
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned.
    pub fn register_root_store<F>(&self, scheme: &str, builder: F)
    where
        F: Fn(&UrlComponent) -> Result<ReadableStorage, UrlPipelineError> + Send + Sync + 'static,
    {
        self.root_builders
            .write()
            .unwrap()
            .insert(scheme.to_string(), Arc::new(builder));
    }

    /// Register an adapter store builder.
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned.
    pub fn register_adapter_store<F>(&self, scheme: &str, builder: F)
    where
        F: Fn(ReadableStorage, &UrlComponent) -> Result<ReadableStorage, UrlPipelineError>
            + Send
            + Sync
            + 'static,
    {
        self.adapter_builders
            .write()
            .unwrap()
            .insert(scheme.to_string(), Arc::new(builder));
    }

    #[cfg(feature = "async")]
    /// Register an async root store builder.
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned.
    pub fn register_async_root_store<F>(&self, scheme: &str, builder: F)
    where
        F: Fn(&UrlComponent) -> Result<AsyncReadableStorage, UrlPipelineError>
            + Send
            + Sync
            + 'static,
    {
        self.async_root_builders
            .write()
            .unwrap()
            .insert(scheme.to_string(), Arc::new(builder));
    }

    #[cfg(feature = "async")]
    /// Register an async adapter store builder.
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned.
    pub fn register_async_adapter_store<F>(&self, scheme: &str, builder: F)
    where
        F: Fn(
                AsyncReadableStorage,
                &UrlComponent,
            ) -> Result<AsyncReadableStorage, UrlPipelineError>
            + Send
            + Sync
            + 'static,
    {
        self.async_adapter_builders
            .write()
            .unwrap()
            .insert(scheme.to_string(), Arc::new(builder));
    }

    /// Get a root store builder.
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned.
    #[must_use]
    pub fn get_root_builder(&self, scheme: &str) -> Option<RootStoreBuilder> {
        self.root_builders.read().unwrap().get(scheme).cloned()
    }

    /// Get an adapter store builder.
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned.
    #[must_use]
    pub fn get_adapter_builder(&self, scheme: &str) -> Option<AdapterStoreBuilder> {
        self.adapter_builders.read().unwrap().get(scheme).cloned()
    }

    #[cfg(feature = "async")]
    /// Get an async root store builder.
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned.
    #[must_use]
    pub fn get_async_root_builder(&self, scheme: &str) -> Option<AsyncRootStoreBuilder> {
        self.async_root_builders
            .read()
            .unwrap()
            .get(scheme)
            .cloned()
    }

    #[cfg(feature = "async")]
    /// Get an async adapter store builder.
    ///
    /// # Panics
    ///
    /// Panics if the lock is poisoned.
    #[must_use]
    pub fn get_async_adapter_builder(&self, scheme: &str) -> Option<AsyncAdapterStoreBuilder> {
        self.async_adapter_builders
            .read()
            .unwrap()
            .get(scheme)
            .cloned()
    }
}

impl Default for StoreRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the global store registry.
#[must_use]
pub fn get_global_registry() -> &'static StoreRegistry {
    static REGISTRY: std::sync::OnceLock<StoreRegistry> = std::sync::OnceLock::new();

    REGISTRY.get_or_init(|| {
        let registry = StoreRegistry::new();
        // Initialize built-in stores
        super::builders::init_builtin_stores(&registry);
        registry
    })
}

/// Register a root store builder in the global registry.
///
/// This function is used by store implementations to register themselves.
pub fn register_root_store<F>(scheme: &str, builder: F)
where
    F: Fn(&UrlComponent) -> Result<ReadableStorage, UrlPipelineError> + Send + Sync + 'static,
{
    get_global_registry().register_root_store(scheme, builder);
}

/// Register an adapter store builder in the global registry.
///
/// This function is used by adapter implementations to register themselves.
pub fn register_adapter_store<F>(scheme: &str, builder: F)
where
    F: Fn(ReadableStorage, &UrlComponent) -> Result<ReadableStorage, UrlPipelineError>
        + Send
        + Sync
        + 'static,
{
    get_global_registry().register_adapter_store(scheme, builder);
}

#[cfg(feature = "async")]
/// Register an async root store builder in the global registry.
///
/// # Panics
///
/// May panic if the registry lock is poisoned.
pub fn register_async_root_store<F>(scheme: &str, builder: F)
where
    F: Fn(&UrlComponent) -> Result<AsyncReadableStorage, UrlPipelineError> + Send + Sync + 'static,
{
    get_global_registry().register_async_root_store(scheme, builder);
}

#[cfg(feature = "async")]
/// Register an async adapter store builder in the global registry.
///
/// # Panics
///
/// May panic if the registry lock is poisoned.
pub fn register_async_adapter_store<F>(scheme: &str, builder: F)
where
    F: Fn(AsyncReadableStorage, &UrlComponent) -> Result<AsyncReadableStorage, UrlPipelineError>
        + Send
        + Sync
        + 'static,
{
    get_global_registry().register_async_adapter_store(scheme, builder);
}
