//! Runtime plugin types for dynamic extension registration.
//!
//! These types enable runtime registration of extension plugins.

use std::borrow::Cow;

use crate::{PluginCreateError, ZarrVersions};

macro_rules! impl_runtime_plugin {
    (plugin: [$($plugin_bound:tt)+], fn: [$($fn_bound:tt)*]) => {
        /// A runtime plugin (single input parameter).
        ///
        /// Unlike the compile-time [`Plugin`](crate::Plugin), these support dynamic registration at runtime.
        ///
        /// # Example
        /// ```ignore
        /// use zarrs_plugin::{RuntimePlugin, ZarrVersions, PluginCreateError};
        ///
        /// let plugin = RuntimePlugin::new(
        ///     "my.custom.codec",
        ///     |name, zarr_version| name == "my.custom.codec",
        ///     |zarr_version| "my.custom.codec".into(),
        ///     |metadata| {
        ///         // Create the extension from metadata
        ///         Ok(MyCustomCodec::from_metadata(metadata)?)
        ///     },
        /// );
        /// ```
        #[allow(clippy::type_complexity)]
        pub struct RuntimePlugin<TPlugin, TInput>
        where
            TPlugin: $($plugin_bound)+,
            TInput: ?Sized + 'static,
        {
            /// The identifier of the plugin.
            identifier: String,
            /// Tests if the name is a match for this plugin for a given Zarr version.
            match_name_fn: Box<dyn Fn(&str, ZarrVersions) -> bool $($fn_bound)*>,
            /// Returns the default name for this plugin for the given Zarr version.
            default_name_fn: Box<dyn Fn(ZarrVersions) -> Cow<'static, str> $($fn_bound)*>,
            /// Create an implementation of this plugin from input.
            create_fn: Box<dyn Fn(&TInput) -> Result<TPlugin, PluginCreateError> $($fn_bound)*>,
        }

        impl<TPlugin, TInput> RuntimePlugin<TPlugin, TInput>
        where
            TPlugin: $($plugin_bound)+,
            TInput: ?Sized + 'static,
        {
            /// Create a new runtime plugin for registration.
            pub fn new<M, D, C>(
                identifier: impl Into<String>,
                match_name_fn: M,
                default_name_fn: D,
                create_fn: C,
            ) -> Self
            where
                M: Fn(&str, ZarrVersions) -> bool $($fn_bound)*,
                D: Fn(ZarrVersions) -> Cow<'static, str> $($fn_bound)*,
                C: Fn(&TInput) -> Result<TPlugin, PluginCreateError> $($fn_bound)*,
            {
                Self {
                    identifier: identifier.into(),
                    match_name_fn: Box::new(match_name_fn),
                    default_name_fn: Box::new(default_name_fn),
                    create_fn: Box::new(create_fn),
                }
            }

            /// Create a `TPlugin` plugin from `input`.
            ///
            /// # Errors
            /// Returns a [`PluginCreateError`] if plugin creation fails.
            pub fn create(&self, input: &TInput) -> Result<TPlugin, PluginCreateError> {
                (self.create_fn)(input)
            }

            /// Returns true if this plugin is associated with `name` for the given Zarr version.
            #[must_use]
            pub fn match_name(&self, name: &str, version: impl Into<ZarrVersions>) -> bool {
                (self.match_name_fn)(name, version.into())
            }

            /// Return the default name for this plugin for the given Zarr version.
            pub fn default_name(&self, version: impl Into<ZarrVersions>) -> Cow<'static, str> {
                (self.default_name_fn)(version.into())
            }

            /// Returns the identifier of the plugin.
            #[must_use]
            pub fn identifier(&self) -> &str {
                &self.identifier
            }
        }

        impl<TPlugin, TInput> std::fmt::Debug for RuntimePlugin<TPlugin, TInput>
        where
            TPlugin: $($plugin_bound)+,
            TInput: ?Sized + 'static,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("RuntimePlugin")
                    .field("identifier", &self.identifier)
                    .finish_non_exhaustive()
            }
        }

        /// A runtime plugin (two input parameters).
        ///
        /// This is the runtime equivalent of [`Plugin2`](crate::Plugin2).
        #[allow(clippy::type_complexity)]
        pub struct RuntimePlugin2<TPlugin, TInput1, TInput2>
        where
            TPlugin: $($plugin_bound)+,
            TInput1: ?Sized + 'static,
            TInput2: ?Sized + 'static,
        {
            /// The identifier of the plugin.
            identifier: String,
            /// Tests if the name is a match for this plugin for a given Zarr version.
            match_name_fn: Box<dyn Fn(&str, ZarrVersions) -> bool $($fn_bound)*>,
            /// Returns the default name for this plugin for the given Zarr version.
            default_name_fn: Box<dyn Fn(ZarrVersions) -> Cow<'static, str> $($fn_bound)*>,
            /// Create an implementation of this plugin from input.
            create_fn: Box<dyn Fn(&TInput1, &TInput2) -> Result<TPlugin, PluginCreateError> $($fn_bound)*>,
        }

        impl<TPlugin, TInput1, TInput2> RuntimePlugin2<TPlugin, TInput1, TInput2>
        where
            TPlugin: $($plugin_bound)+,
            TInput1: ?Sized + 'static,
            TInput2: ?Sized + 'static,
        {
            /// Create a new runtime plugin for registration.
            pub fn new<M, D, C>(
                identifier: impl Into<String>,
                match_name_fn: M,
                default_name_fn: D,
                create_fn: C,
            ) -> Self
            where
                M: Fn(&str, ZarrVersions) -> bool $($fn_bound)*,
                D: Fn(ZarrVersions) -> Cow<'static, str> $($fn_bound)*,
                C: Fn(&TInput1, &TInput2) -> Result<TPlugin, PluginCreateError> $($fn_bound)*,
            {
                Self {
                    identifier: identifier.into(),
                    match_name_fn: Box::new(match_name_fn),
                    default_name_fn: Box::new(default_name_fn),
                    create_fn: Box::new(create_fn),
                }
            }

            /// Create a `TPlugin` plugin from `input1` and `input2`.
            ///
            /// # Errors
            /// Returns a [`PluginCreateError`] if plugin creation fails.
            pub fn create(&self, input1: &TInput1, input2: &TInput2) -> Result<TPlugin, PluginCreateError> {
                (self.create_fn)(input1, input2)
            }

            /// Returns true if this plugin is associated with `name` for the given Zarr version.
            #[must_use]
            pub fn match_name(&self, name: &str, version: impl Into<ZarrVersions>) -> bool {
                (self.match_name_fn)(name, version.into())
            }

            /// Return the default name for this plugin for the given Zarr version.
            pub fn default_name(&self, version: impl Into<ZarrVersions>) -> Cow<'static, str> {
                (self.default_name_fn)(version.into())
            }

            /// Returns the identifier of the plugin.
            #[must_use]
            pub fn identifier(&self) -> &str {
                &self.identifier
            }
        }

        impl<TPlugin, TInput1, TInput2> std::fmt::Debug for RuntimePlugin2<TPlugin, TInput1, TInput2>
        where
            TPlugin: $($plugin_bound)+,
            TInput1: ?Sized + 'static,
            TInput2: ?Sized + 'static,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct("RuntimePlugin2")
                    .field("identifier", &self.identifier)
                    .finish_non_exhaustive()
            }
        }
    };
}

#[cfg(not(target_arch = "wasm32"))]
impl_runtime_plugin!(plugin: [Send + Sync + 'static], fn: [+ Send + Sync + 'static]);

#[cfg(target_arch = "wasm32")]
impl_runtime_plugin!(plugin: ['static], fn: [+ 'static]);

// On WASM, manually implement Send/Sync since execution is single-threaded.
#[cfg(target_arch = "wasm32")]
mod wasm_impls {
    use super::*;

    // SAFETY: On WASM, execution is single-threaded.
    unsafe impl<TPlugin: 'static, TInput: ?Sized + 'static> Send for RuntimePlugin<TPlugin, TInput> {}
    unsafe impl<TPlugin: 'static, TInput: ?Sized + 'static> Sync for RuntimePlugin<TPlugin, TInput> {}

    unsafe impl<TPlugin: 'static, TInput1: ?Sized + 'static, TInput2: ?Sized + 'static> Send
        for RuntimePlugin2<TPlugin, TInput1, TInput2>
    {
    }
    unsafe impl<TPlugin: 'static, TInput1: ?Sized + 'static, TInput2: ?Sized + 'static> Sync
        for RuntimePlugin2<TPlugin, TInput1, TInput2>
    {
    }
}
