//! Runtime registry for dynamic extension registration.
//!
//! This module provides the infrastructure for registering extension plugins
//! at runtime, complementing the compile-time registration via [`inventory`].

use std::sync::{Arc, RwLock};

/// A handle to a registered plugin. See [`RuntimeRegistry::register`].
pub type RuntimeRegistryHandle<P> = Arc<P>;

/// A runtime registry for extension plugins.
///
/// This registry stores plugins that are registered at runtime. It is thread-safe
/// and can be accessed from multiple threads concurrently.
///
/// Plugins are stored as [`RuntimeRegistryHandle`] and can be unregistered.
///
/// # Type Parameters
///
/// * `P` - The plugin type to store (e.g., `RuntimePlugin<Codec, MetadataV3>`)
#[derive(Debug)]
pub struct RuntimeRegistry<P> {
    plugins: RwLock<Vec<RuntimeRegistryHandle<P>>>,
}

impl<P> RuntimeRegistry<P> {
    /// Create a new empty registry.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            plugins: RwLock::new(Vec::new()),
        }
    }

    /// Register a plugin and return a handle for later unregistration.
    ///
    /// The returned `RuntimeRegistryHandle<P>` can be used to unregister the plugin later.
    ///
    /// # Panics
    ///
    /// Panics if the internal lock is poisoned.
    pub fn register(&self, plugin: P) -> RuntimeRegistryHandle<P> {
        let plugin = Arc::new(plugin);
        let handle = Arc::clone(&plugin);

        let mut plugins = self.plugins.write().unwrap();
        plugins.push(plugin);

        handle
    }

    /// Unregister a plugin by its handle.
    ///
    /// Uses `Arc::ptr_eq` to find and remove the plugin.
    ///
    /// Returns `true` if the plugin was found and removed, `false` otherwise.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    pub fn unregister(&self, handle: &RuntimeRegistryHandle<P>) -> bool {
        let mut plugins = self.plugins.write().unwrap();
        if let Some(pos) = plugins.iter().position(|p| Arc::ptr_eq(p, handle)) {
            plugins.remove(pos);
            true
        } else {
            false
        }
    }

    /// Execute a closure with read access to all registered plugins.
    ///
    /// This method holds a read lock for the duration of the closure, allowing concurrent reads but blocking writes.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    pub fn with_plugins<R>(&self, f: impl FnOnce(&[RuntimeRegistryHandle<P>]) -> R) -> R {
        let plugins = self.plugins.read().unwrap();
        f(&plugins)
    }

    /// Returns the number of registered plugins.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn len(&self) -> usize {
        self.plugins.read().unwrap().len()
    }

    /// Returns true if no plugins are registered.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plugins.read().unwrap().is_empty()
    }

    /// Clear all registered plugins.
    ///
    /// # Panics
    /// Panics if the internal lock is poisoned.
    pub fn clear(&self) {
        self.plugins.write().unwrap().clear();
    }
}

impl<P> Default for RuntimeRegistry<P> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_unregister() {
        let registry: RuntimeRegistry<String> = RuntimeRegistry::new();

        let handle1 = registry.register("plugin1".to_string());
        let handle2 = registry.register("plugin2".to_string());

        assert_eq!(registry.len(), 2);

        let removed = registry.unregister(&handle1);
        assert!(removed);
        assert_eq!(registry.len(), 1);

        // Unregistering again should return false
        let removed = registry.unregister(&handle1);
        assert!(!removed);

        let removed = registry.unregister(&handle2);
        assert!(removed);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_cloned_handle() {
        let registry: RuntimeRegistry<String> = RuntimeRegistry::new();

        let handle = registry.register("plugin".to_string());
        let handle_clone = Arc::clone(&handle);

        assert_eq!(registry.len(), 1);

        // Can unregister with the clone
        let removed = registry.unregister(&handle_clone);
        assert!(removed);
        assert!(registry.is_empty());

        // Original handle no longer valid
        let removed = registry.unregister(&handle);
        assert!(!removed);
    }

    #[test]
    fn test_with_plugins() {
        let registry: RuntimeRegistry<i32> = RuntimeRegistry::new();

        registry.register(1);
        registry.register(2);
        registry.register(3);

        let sum = registry.with_plugins(|plugins| plugins.iter().map(|p| **p).sum::<i32>());
        assert_eq!(sum, 6);
    }

    #[test]
    fn test_clear() {
        let registry: RuntimeRegistry<String> = RuntimeRegistry::new();

        registry.register("a".to_string());
        registry.register("b".to_string());
        assert_eq!(registry.len(), 2);

        registry.clear();
        assert!(registry.is_empty());
    }
}
