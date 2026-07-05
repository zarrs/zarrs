//! Adapter store scheme registration (compile-time and runtime).
//!
//! Synchronous and asynchronous adapter schemes are entirely independent.

use std::sync::LazyLock;

use zarrs_plugin::{Plugin2, RuntimePlugin2, RuntimeRegistry, RuntimeRegistryHandle};

use crate::error::PipelineCreateError;
#[cfg(feature = "async")]
use crate::stage::AsyncPipelineStage;
use crate::stage::PipelineStage;

/// The input to an adapter store plugin: a parsed `scheme:rest` pipeline segment.
#[derive(Debug, Clone)]
pub struct AdapterStoreInput {
    /// The lowercased scheme, e.g. `"zip"`, `"gzip"`.
    pub scheme: String,
    /// The raw scheme-specific part, unparsed.
    pub rest: String,
}

/// A compile-time registered adapter store plugin.
///
/// Registered via [`inventory::submit!`] in a backend crate, e.g.:
/// ```ignore
/// inventory::submit! {
///     AdapterStorePlugin::new("my_crate", |s| s == "myadapter", my_crate::create_adapter)
/// }
/// ```
pub struct AdapterStorePlugin {
    /// The name of the crate that registered this plugin, used only for diagnostics when
    /// multiple compile-time plugins claim the same scheme.
    source_crate: &'static str,
    plugin: Plugin2<PipelineStage, AdapterStoreInput, PipelineStage, PipelineCreateError>,
}
inventory::collect!(AdapterStorePlugin);

impl AdapterStorePlugin {
    /// Create a new [`AdapterStorePlugin`] for registration.
    ///
    /// `source_crate` should be the name of the crate registering the plugin (e.g.
    /// `env!("CARGO_PKG_NAME")`), used only for diagnostics on scheme collisions.
    #[must_use]
    pub const fn new(
        source_crate: &'static str,
        match_fn: fn(&str) -> bool,
        create_fn: fn(
            &AdapterStoreInput,
            &PipelineStage,
        ) -> Result<PipelineStage, PipelineCreateError>,
    ) -> Self {
        Self {
            source_crate,
            plugin: Plugin2::new(match_fn, create_fn),
        }
    }
}

/// A runtime adapter store plugin for dynamic registration.
pub type AdapterStoreRuntimePlugin =
    RuntimePlugin2<PipelineStage, AdapterStoreInput, PipelineStage, PipelineCreateError>;

/// Global runtime registry for adapter store plugins.
///
/// Runtime-registered plugins always take precedence over compile-time (inventory) registered
/// plugins, and are the documented way to disambiguate when multiple compile-time plugins claim
/// the same scheme.
pub static ADAPTER_STORE_RUNTIME_REGISTRY: LazyLock<RuntimeRegistry<AdapterStoreRuntimePlugin>> =
    LazyLock::new(RuntimeRegistry::new);

/// A handle to a registered adapter store plugin.
pub type AdapterStoreRuntimeRegistryHandle = RuntimeRegistryHandle<AdapterStoreRuntimePlugin>;

/// Register an adapter store scheme plugin at runtime.
///
/// Runtime-registered plugins take precedence over compile-time registered plugins, and are the
/// way to pin a specific backend when multiple compile-time plugins claim the same scheme.
///
/// # Returns
/// A handle that can be used to unregister the plugin later.
pub fn register_adapter_store_scheme(
    plugin: AdapterStoreRuntimePlugin,
) -> AdapterStoreRuntimeRegistryHandle {
    ADAPTER_STORE_RUNTIME_REGISTRY.register(plugin)
}

/// Unregister a runtime adapter store scheme plugin.
///
/// # Returns
/// `true` if the plugin was found and removed, `false` otherwise.
pub fn unregister_adapter_store_scheme(handle: &AdapterStoreRuntimeRegistryHandle) -> bool {
    ADAPTER_STORE_RUNTIME_REGISTRY.unregister(handle)
}

/// Create the next pipeline stage by applying an adapter to the previous stage.
///
/// Lookup order and collision handling mirror [`try_create_root_stage`](crate::root::try_create_root_stage).
///
/// # Errors
/// Returns [`PipelineCreateError`] if no registered plugin matches the scheme, or if the matching
/// plugin fails to construct the adapter.
pub fn try_create_adapter_stage(
    input: &AdapterStoreInput,
    prev: &PipelineStage,
) -> Result<PipelineStage, PipelineCreateError> {
    // Check the runtime registry first (higher priority).
    let result = ADAPTER_STORE_RUNTIME_REGISTRY.with_plugins(|plugins| {
        for plugin in plugins {
            if plugin.match_name(&input.scheme) {
                return Some(plugin.create(input, prev));
            }
        }
        None
    });
    if let Some(result) = result {
        return result;
    }

    // Fall back to compile-time registered plugins.
    let mut matches =
        inventory::iter::<AdapterStorePlugin>().filter(|p| p.plugin.match_name(&input.scheme));
    let Some(first) = matches.next() else {
        return Err(zarrs_plugin::PluginUnsupportedError::new(
            input.scheme.clone(),
            "adapter store scheme".to_string(),
        )
        .into());
    };
    let others: Vec<&'static str> = matches.map(|p| p.source_crate).collect();
    if !others.is_empty() {
        log::debug!(
            "ambiguous adapter store scheme {:?}: matched by {} (used) and {:?}; register a runtime plugin via register_adapter_store_scheme(...) to disambiguate",
            input.scheme,
            first.source_crate,
            others
        );
    }
    first.plugin.create(input, prev)
}

/// The input to an asynchronous adapter store plugin: a parsed `scheme:rest` pipeline segment.
#[cfg(feature = "async")]
#[derive(Debug, Clone)]
pub struct AsyncAdapterStoreInput {
    /// The lowercased scheme, e.g. `"zip"`, `"gzip"`.
    pub scheme: String,
    /// The raw scheme-specific part, unparsed.
    pub rest: String,
}

/// A compile-time registered asynchronous adapter store plugin.
///
/// Registered via [`inventory::submit!`] in a backend crate, e.g.:
/// ```ignore
/// inventory::submit! {
///     AsyncAdapterStorePlugin::new("my_crate", |s| s == "myadapter", my_crate::create_adapter_async)
/// }
/// ```
#[cfg(feature = "async")]
pub struct AsyncAdapterStorePlugin {
    /// The name of the crate that registered this plugin, used only for diagnostics when
    /// multiple compile-time plugins claim the same scheme.
    source_crate: &'static str,
    plugin: Plugin2<
        AsyncPipelineStage,
        AsyncAdapterStoreInput,
        AsyncPipelineStage,
        PipelineCreateError,
    >,
}
#[cfg(feature = "async")]
inventory::collect!(AsyncAdapterStorePlugin);

#[cfg(feature = "async")]
impl AsyncAdapterStorePlugin {
    /// Create a new [`AsyncAdapterStorePlugin`] for registration.
    ///
    /// `source_crate` should be the name of the crate registering the plugin (e.g.
    /// `env!("CARGO_PKG_NAME")`), used only for diagnostics on scheme collisions.
    #[must_use]
    pub const fn new(
        source_crate: &'static str,
        match_fn: fn(&str) -> bool,
        create_fn: fn(
            &AsyncAdapterStoreInput,
            &AsyncPipelineStage,
        ) -> Result<AsyncPipelineStage, PipelineCreateError>,
    ) -> Self {
        Self {
            source_crate,
            plugin: Plugin2::new(match_fn, create_fn),
        }
    }
}

/// A runtime asynchronous adapter store plugin for dynamic registration.
#[cfg(feature = "async")]
pub type AsyncAdapterStoreRuntimePlugin = RuntimePlugin2<
    AsyncPipelineStage,
    AsyncAdapterStoreInput,
    AsyncPipelineStage,
    PipelineCreateError,
>;

/// Global runtime registry for asynchronous adapter store plugins.
///
/// Runtime-registered plugins always take precedence over compile-time (inventory) registered
/// plugins, and are the documented way to disambiguate when multiple compile-time plugins claim
/// the same scheme.
#[cfg(feature = "async")]
pub static ASYNC_ADAPTER_STORE_RUNTIME_REGISTRY: LazyLock<
    RuntimeRegistry<AsyncAdapterStoreRuntimePlugin>,
> = LazyLock::new(RuntimeRegistry::new);

/// A handle to a registered asynchronous adapter store plugin.
#[cfg(feature = "async")]
pub type AsyncAdapterStoreRuntimeRegistryHandle =
    RuntimeRegistryHandle<AsyncAdapterStoreRuntimePlugin>;

/// Register an asynchronous adapter store scheme plugin at runtime.
///
/// Runtime-registered plugins take precedence over compile-time registered plugins, and are the
/// way to pin a specific backend when multiple compile-time plugins claim the same scheme.
///
/// # Returns
/// A handle that can be used to unregister the plugin later.
#[cfg(feature = "async")]
pub fn register_async_adapter_store_scheme(
    plugin: AsyncAdapterStoreRuntimePlugin,
) -> AsyncAdapterStoreRuntimeRegistryHandle {
    ASYNC_ADAPTER_STORE_RUNTIME_REGISTRY.register(plugin)
}

/// Unregister a runtime asynchronous adapter store scheme plugin.
///
/// # Returns
/// `true` if the plugin was found and removed, `false` otherwise.
#[cfg(feature = "async")]
pub fn unregister_async_adapter_store_scheme(
    handle: &AsyncAdapterStoreRuntimeRegistryHandle,
) -> bool {
    ASYNC_ADAPTER_STORE_RUNTIME_REGISTRY.unregister(handle)
}

/// Create the next asynchronous pipeline stage by applying an adapter to the previous stage.
///
/// Lookup order and collision handling mirror
/// [`try_create_root_stage`](crate::root::try_create_root_stage).
///
/// # Errors
/// Returns [`PipelineCreateError`] if no registered plugin matches the scheme, or if the matching
/// plugin fails to construct the adapter.
#[cfg(feature = "async")]
pub fn try_create_async_adapter_stage(
    input: &AsyncAdapterStoreInput,
    prev: &AsyncPipelineStage,
) -> Result<AsyncPipelineStage, PipelineCreateError> {
    let result = ASYNC_ADAPTER_STORE_RUNTIME_REGISTRY.with_plugins(|plugins| {
        for plugin in plugins {
            if plugin.match_name(&input.scheme) {
                return Some(plugin.create(input, prev));
            }
        }
        None
    });
    if let Some(result) = result {
        return result;
    }

    let mut matches =
        inventory::iter::<AsyncAdapterStorePlugin>().filter(|p| p.plugin.match_name(&input.scheme));
    let Some(first) = matches.next() else {
        return Err(zarrs_plugin::PluginUnsupportedError::new(
            input.scheme.clone(),
            "async adapter store scheme".to_string(),
        )
        .into());
    };
    let others: Vec<&'static str> = matches.map(|p| p.source_crate).collect();
    if !others.is_empty() {
        log::debug!(
            "ambiguous async adapter store scheme {:?}: matched by {} (used) and {:?}; register a runtime plugin via register_async_adapter_store_scheme(...) to disambiguate",
            input.scheme,
            first.source_crate,
            others
        );
    }
    first.plugin.create(input, prev)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use zarrs_storage::store::MemoryStore;

    #[test]
    fn unknown_adapter_scheme_errors() {
        let prev: PipelineStage = Arc::new(MemoryStore::new());
        let input = AdapterStoreInput {
            scheme: "does-not-exist".to_string(),
            rest: String::new(),
        };
        assert!(try_create_adapter_stage(&input, &prev).is_err());
    }

    #[test]
    fn runtime_registration_is_used_for_matching_scheme() {
        let handle = register_adapter_store_scheme(AdapterStoreRuntimePlugin::new(
            |s| s == "test-passthrough",
            |_input: &AdapterStoreInput, prev: &PipelineStage| Ok(Arc::clone(prev)),
        ));

        let prev: PipelineStage = Arc::new(MemoryStore::new());
        let input = AdapterStoreInput {
            scheme: "test-passthrough".to_string(),
            rest: String::new(),
        };
        let stage = try_create_adapter_stage(&input, &prev).unwrap();
        assert!(Arc::ptr_eq(&stage, &prev));

        assert!(unregister_adapter_store_scheme(&handle));
        assert!(try_create_adapter_stage(&input, &prev).is_err());
    }
}
