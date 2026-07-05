//! Root store scheme registration (compile-time and runtime).
//!
//! Synchronous and asynchronous root schemes are entirely independent: separate input types,
//! separate compile-time (`inventory`) collections, and separate runtime registries.

use std::sync::LazyLock;

use zarrs_plugin::{Plugin, RuntimePlugin, RuntimeRegistry, RuntimeRegistryHandle};

use crate::error::PipelineCreateError;
#[cfg(feature = "async")]
use crate::stage::AsyncPipelineStage;
use crate::stage::PipelineStage;

/// The input to a root store plugin: a parsed `scheme:rest` pipeline segment.
#[derive(Debug, Clone)]
pub struct RootStoreInput {
    /// The lowercased scheme, e.g. `"s3"`, `"file"`, `"memory"`.
    pub scheme: String,
    /// The raw scheme-specific part, unparsed.
    pub rest: String,
}

/// A compile-time registered root store plugin.
///
/// Registered via [`inventory::submit!`] in a backend crate, e.g.:
/// ```ignore
/// inventory::submit! {
///     RootStorePlugin::new("my_crate", |s| s == "myscheme", my_crate::create_from_url)
/// }
/// ```
pub struct RootStorePlugin {
    /// The name of the crate that registered this plugin, used only for diagnostics when
    /// multiple compile-time plugins claim the same scheme.
    source_crate: &'static str,
    plugin: Plugin<PipelineStage, RootStoreInput, PipelineCreateError, RootStoreInput>,
}
inventory::collect!(RootStorePlugin);

impl RootStorePlugin {
    /// Create a new [`RootStorePlugin`] for registration.
    ///
    /// `source_crate` should be the name of the crate registering the plugin (e.g.
    /// `env!("CARGO_PKG_NAME")`), used only for diagnostics on scheme collisions.
    #[must_use]
    pub const fn new(
        source_crate: &'static str,
        match_fn: fn(&RootStoreInput) -> bool,
        create_fn: fn(&RootStoreInput) -> Result<PipelineStage, PipelineCreateError>,
    ) -> Self {
        Self {
            source_crate,
            plugin: Plugin::new(match_fn, create_fn),
        }
    }
}

/// A runtime root store plugin for dynamic registration.
pub type RootStoreRuntimePlugin =
    RuntimePlugin<PipelineStage, RootStoreInput, PipelineCreateError, RootStoreInput>;

/// Global runtime registry for root store plugins.
///
/// Runtime-registered plugins always take precedence over compile-time (inventory) registered
/// plugins, and are the documented way to disambiguate when multiple compile-time plugins claim
/// the same scheme (e.g. both `zarrs_object_store` and `zarrs_opendal` registering `s3:`).
pub static ROOT_STORE_RUNTIME_REGISTRY: LazyLock<RuntimeRegistry<RootStoreRuntimePlugin>> =
    LazyLock::new(RuntimeRegistry::new);

/// A handle to a registered root store plugin.
pub type RootStoreRuntimeRegistryHandle = RuntimeRegistryHandle<RootStoreRuntimePlugin>;

/// Register a root store scheme plugin at runtime.
///
/// Runtime-registered plugins take precedence over compile-time registered plugins, and are the
/// way to pin a specific backend when multiple compile-time plugins claim the same scheme.
///
/// # Returns
/// A handle that can be used to unregister the plugin later.
pub fn register_root_store_scheme(
    plugin: RootStoreRuntimePlugin,
) -> RootStoreRuntimeRegistryHandle {
    ROOT_STORE_RUNTIME_REGISTRY.register(plugin)
}

/// Unregister a runtime root store scheme plugin.
///
/// # Returns
/// `true` if the plugin was found and removed, `false` otherwise.
pub fn unregister_root_store_scheme(handle: &RootStoreRuntimeRegistryHandle) -> bool {
    ROOT_STORE_RUNTIME_REGISTRY.unregister(handle)
}

/// Create a root pipeline stage from a parsed root segment.
///
/// Lookup order:
/// 1. The runtime registry (first match wins, and always takes priority over compile-time plugins).
/// 2. Compile-time (inventory) registered plugins (first match wins).
///
/// If two or more compile-time plugins match the same scheme and no runtime override was
/// registered, the first-registered plugin wins (`inventory` iteration order is otherwise
/// unspecified) and a debug-level warning is logged naming the colliding crates.
///
/// # Errors
/// Returns [`PipelineCreateError`] if no registered plugin matches the scheme, or if the matching
/// plugin fails to construct the store.
pub fn try_create_root_stage(input: &RootStoreInput) -> Result<PipelineStage, PipelineCreateError> {
    // Check the runtime registry first (higher priority).
    let result = ROOT_STORE_RUNTIME_REGISTRY.with_plugins(|plugins| {
        for plugin in plugins {
            if plugin.match_name(input) {
                return Some(plugin.create(input));
            }
        }
        None
    });
    if let Some(result) = result {
        return result;
    }

    // Fall back to compile-time registered plugins.
    let mut matches = inventory::iter::<RootStorePlugin>().filter(|p| p.plugin.match_name(input));
    let Some(first) = matches.next() else {
        return Err(zarrs_plugin::PluginUnsupportedError::new(
            input.scheme.clone(),
            "root store scheme".to_string(),
        )
        .into());
    };
    let others: Vec<&'static str> = matches.map(|p| p.source_crate).collect();
    if !others.is_empty() {
        log::debug!(
            "ambiguous root store scheme {:?}: matched by {} (used) and {:?}; register a runtime plugin via register_root_store_scheme(...) to disambiguate",
            input.scheme,
            first.source_crate,
            others
        );
    }
    first.plugin.create(input)
}

/// A compile-time registered asynchronous root store plugin.
///
/// Registered via [`inventory::submit!`] in a backend crate, e.g.:
/// ```ignore
/// inventory::submit! {
///     AsyncRootStorePlugin::new("my_crate", |s| s == "myscheme", my_crate::create_from_url_async)
/// }
/// ```
#[cfg(feature = "async")]
pub struct AsyncRootStorePlugin {
    /// The name of the crate that registered this plugin, used only for diagnostics when
    /// multiple compile-time plugins claim the same scheme.
    source_crate: &'static str,
    plugin: Plugin<AsyncPipelineStage, RootStoreInput, PipelineCreateError, RootStoreInput>,
}
#[cfg(feature = "async")]
inventory::collect!(AsyncRootStorePlugin);

#[cfg(feature = "async")]
impl AsyncRootStorePlugin {
    /// Create a new [`AsyncRootStorePlugin`] for registration.
    ///
    /// `source_crate` should be the name of the crate registering the plugin (e.g.
    /// `env!("CARGO_PKG_NAME")`), used only for diagnostics on scheme collisions.
    #[must_use]
    pub const fn new(
        source_crate: &'static str,
        match_fn: fn(&RootStoreInput) -> bool,
        create_fn: fn(&RootStoreInput) -> Result<AsyncPipelineStage, PipelineCreateError>,
    ) -> Self {
        Self {
            source_crate,
            plugin: Plugin::new(match_fn, create_fn),
        }
    }
}

/// A runtime asynchronous root store plugin for dynamic registration.
#[cfg(feature = "async")]
pub type AsyncRootStoreRuntimePlugin =
    RuntimePlugin<AsyncPipelineStage, RootStoreInput, PipelineCreateError, RootStoreInput>;

/// Global runtime registry for asynchronous root store plugins.
///
/// Runtime-registered plugins always take precedence over compile-time (inventory) registered
/// plugins, and are the documented way to disambiguate when multiple compile-time plugins claim
/// the same scheme (e.g. both `zarrs_object_store` and `zarrs_opendal` registering `s3:`).
#[cfg(feature = "async")]
pub static ASYNC_ROOT_STORE_RUNTIME_REGISTRY: LazyLock<
    RuntimeRegistry<AsyncRootStoreRuntimePlugin>,
> = LazyLock::new(RuntimeRegistry::new);

/// A handle to a registered asynchronous root store plugin.
#[cfg(feature = "async")]
pub type AsyncRootStoreRuntimeRegistryHandle = RuntimeRegistryHandle<AsyncRootStoreRuntimePlugin>;

/// Register an asynchronous root store scheme plugin at runtime.
///
/// Runtime-registered plugins take precedence over compile-time registered plugins, and are the
/// way to pin a specific backend when multiple compile-time plugins claim the same scheme.
///
/// # Returns
/// A handle that can be used to unregister the plugin later.
#[cfg(feature = "async")]
pub fn register_async_root_store_scheme(
    plugin: AsyncRootStoreRuntimePlugin,
) -> AsyncRootStoreRuntimeRegistryHandle {
    ASYNC_ROOT_STORE_RUNTIME_REGISTRY.register(plugin)
}

/// Unregister a runtime asynchronous root store scheme plugin.
///
/// # Returns
/// `true` if the plugin was found and removed, `false` otherwise.
#[cfg(feature = "async")]
pub fn unregister_async_root_store_scheme(handle: &AsyncRootStoreRuntimeRegistryHandle) -> bool {
    ASYNC_ROOT_STORE_RUNTIME_REGISTRY.unregister(handle)
}

/// Create an asynchronous root pipeline stage from a parsed root segment.
///
/// Lookup order and collision handling mirror [`try_create_root_stage`].
///
/// # Errors
/// Returns [`PipelineCreateError`] if no registered plugin matches the scheme, or if the matching
/// plugin fails to construct the store.
#[cfg(feature = "async")]
pub fn try_create_async_root_stage(
    input: &RootStoreInput,
) -> Result<AsyncPipelineStage, PipelineCreateError> {
    let result = ASYNC_ROOT_STORE_RUNTIME_REGISTRY.with_plugins(|plugins| {
        for plugin in plugins {
            if plugin.match_name(input) {
                return Some(plugin.create(input));
            }
        }
        None
    });
    if let Some(result) = result {
        return result;
    }

    let mut matches =
        inventory::iter::<AsyncRootStorePlugin>().filter(|p| p.plugin.match_name(input));
    let Some(first) = matches.next() else {
        return Err(zarrs_plugin::PluginUnsupportedError::new(
            input.scheme.clone(),
            "async root store scheme".to_string(),
        )
        .into());
    };
    let others: Vec<&'static str> = matches.map(|p| p.source_crate).collect();
    if !others.is_empty() {
        log::debug!(
            "ambiguous async root store scheme {:?}: matched by {} (used) and {:?}; register a runtime plugin via register_async_root_store_scheme(...) to disambiguate",
            input.scheme,
            first.source_crate,
            others
        );
    }
    first.plugin.create(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_scheme_is_always_registered() {
        let input = RootStoreInput {
            scheme: "memory".to_string(),
            rest: "//".to_string(),
        };
        let stage = try_create_root_stage(&input).unwrap();
        // MemoryStore supports every synchronous capability.
        stage.as_readable().unwrap();
    }

    #[test]
    fn unknown_scheme_errors() {
        let input = RootStoreInput {
            scheme: "does-not-exist".to_string(),
            rest: String::new(),
        };
        assert!(try_create_root_stage(&input).is_err());
    }

    #[test]
    fn runtime_registration_takes_precedence_over_compile_time() {
        let handle = register_root_store_scheme(RootStoreRuntimePlugin::new(
            |s| s.scheme == "memory",
            |_input: &RootStoreInput| {
                // A distinguishable marker: fails unconditionally so we can prove this runtime
                // plugin (not the compile-time `memory:` plugin) was the one invoked.
                Err(PipelineCreateError::Other(
                    "runtime override invoked".to_string(),
                ))
            },
        ));

        let input = RootStoreInput {
            scheme: "memory".to_string(),
            rest: "//".to_string(),
        };
        let err = try_create_root_stage(&input).unwrap_err();
        assert!(
            matches!(err, PipelineCreateError::Other(msg) if msg == "runtime override invoked")
        );

        // Clean up so this test doesn't leak state into other tests in the same process.
        assert!(unregister_root_store_scheme(&handle));

        // With the override removed, the compile-time `memory:` plugin resolves again.
        try_create_root_stage(&input).unwrap();
    }
}
