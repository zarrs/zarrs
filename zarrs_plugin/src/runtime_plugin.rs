//! Runtime plugin types for dynamic extension registration.
//!
//! These types enable runtime registration of extension plugins.

use crate::{MaybeSend, MaybeSync, PluginCreateError};

/// A runtime plugin (single input parameter).
///
/// Unlike the compile-time [`Plugin`](crate::Plugin), these support dynamic registration at runtime.
///
/// # Example
/// ```ignore
/// use zarrs_plugin::{RuntimePlugin, PluginCreateError};
///
/// let plugin = RuntimePlugin::new(
///     "my.custom.codec",
///     |name| name == "my.custom.codec",
///     |metadata| {
///         // Create the extension from metadata
///         Ok(MyCustomCodec::from_metadata(metadata)?)
///     },
/// );
/// ```
#[allow(clippy::type_complexity)]
pub struct RuntimePlugin<TPlugin, TInput>
where
    TPlugin: MaybeSend + MaybeSync + 'static,
    TInput: ?Sized + 'static,
{
    /// Tests if the name is a match for this plugin.
    match_name_fn: Box<dyn Fn(&str) -> bool + Send + Sync + 'static>,
    /// Create an implementation of this plugin from input.
    create_fn: Box<dyn Fn(&TInput) -> Result<TPlugin, PluginCreateError> + Send + Sync + 'static>,
}

impl<TPlugin, TInput> RuntimePlugin<TPlugin, TInput>
where
    TPlugin: MaybeSend + MaybeSync + 'static,
    TInput: ?Sized + 'static,
{
    /// Create a new runtime plugin for registration.
    pub fn new<M, C>(match_name_fn: M, create_fn: C) -> Self
    where
        M: Fn(&str) -> bool + Send + Sync + 'static,
        C: Fn(&TInput) -> Result<TPlugin, PluginCreateError> + Send + Sync + 'static,
    {
        Self {
            match_name_fn: Box::new(match_name_fn),
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

    /// Returns true if this plugin is associated with `name`.
    #[must_use]
    pub fn match_name(&self, name: &str) -> bool {
        (self.match_name_fn)(name)
    }
}

/// A runtime plugin (two input parameters).
///
/// This is the runtime equivalent of [`Plugin2`](crate::Plugin2).
#[allow(clippy::type_complexity)]
pub struct RuntimePlugin2<TPlugin, TInput1, TInput2>
where
    TPlugin: MaybeSend + MaybeSync + 'static,
    TInput1: ?Sized + 'static,
    TInput2: ?Sized + 'static,
{
    /// Tests if the name is a match for this plugin.
    match_name_fn: Box<dyn Fn(&str) -> bool + Send + Sync + 'static>,
    /// Create an implementation of this plugin from input.
    create_fn: Box<
        dyn Fn(&TInput1, &TInput2) -> Result<TPlugin, PluginCreateError> + Send + Sync + 'static,
    >,
}

impl<TPlugin, TInput1, TInput2> RuntimePlugin2<TPlugin, TInput1, TInput2>
where
    TPlugin: MaybeSend + MaybeSync + 'static,
    TInput1: ?Sized + 'static,
    TInput2: ?Sized + 'static,
{
    /// Create a new runtime plugin for registration.
    pub fn new<M, C>(match_name_fn: M, create_fn: C) -> Self
    where
        M: Fn(&str) -> bool + Send + Sync + 'static,
        C: Fn(&TInput1, &TInput2) -> Result<TPlugin, PluginCreateError> + Send + Sync + 'static,
    {
        Self {
            match_name_fn: Box::new(match_name_fn),
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

    /// Returns true if this plugin is associated with `name`.
    #[must_use]
    pub fn match_name(&self, name: &str) -> bool {
        (self.match_name_fn)(name)
    }
}

// SAFETY: On WASM, execution is single-threaded, so Send/Sync are safe.
// The boxed closures require Send + Sync bounds which are auto-traits on WASM
// (everything is Send + Sync in single-threaded execution).
#[cfg(target_arch = "wasm32")]
mod wasm_impls {
    use super::{RuntimePlugin, RuntimePlugin2};

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
