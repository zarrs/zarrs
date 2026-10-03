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
pub struct RuntimePlugin<TPlugin, TInput, TError = PluginCreateError, TMatch = str>
where
    TPlugin: MaybeSend + MaybeSync + 'static,
    TInput: ?Sized + 'static,
    TMatch: ?Sized,
{
    /// Tests if the name is a match for this plugin.
    match_fn: Box<dyn Fn(&TMatch) -> bool + Send + Sync + 'static>,
    /// Create an implementation of this plugin from input.
    create_fn: Box<dyn Fn(&TInput) -> Result<TPlugin, TError> + Send + Sync + 'static>,
}

impl<TPlugin, TInput, TError, TMatch> RuntimePlugin<TPlugin, TInput, TError, TMatch>
where
    TPlugin: MaybeSend + MaybeSync + 'static,
    TInput: ?Sized + 'static,
    TMatch: ?Sized,
{
    /// Create a new runtime plugin for registration.
    pub fn new<M, C>(match_fn: M, create_fn: C) -> Self
    where
        M: Fn(&TMatch) -> bool + Send + Sync + 'static,
        C: Fn(&TInput) -> Result<TPlugin, TError> + Send + Sync + 'static,
    {
        Self {
            match_fn: Box::new(match_fn),
            create_fn: Box::new(create_fn),
        }
    }

    /// Create a `TPlugin` plugin from `input`.
    ///
    /// # Errors
    /// Returns a `TError` if plugin creation fails.
    pub fn create(&self, input: &TInput) -> Result<TPlugin, TError> {
        (self.create_fn)(input)
    }

    /// Returns true if this plugin is associated with `match`.
    // TODO: Rename to `match` on breaking release
    #[must_use]
    pub fn match_name(&self, r#match: &TMatch) -> bool {
        (self.match_fn)(r#match)
    }
}

/// A runtime plugin (two input parameters).
///
/// This is the runtime equivalent of [`Plugin2`](crate::Plugin2).
#[allow(clippy::type_complexity)]
pub struct RuntimePlugin2<TPlugin, TInput1, TInput2, TError = PluginCreateError, TMatch = str>
where
    TPlugin: MaybeSend + MaybeSync + 'static,
    TInput1: ?Sized + 'static,
    TInput2: ?Sized + 'static,
    TMatch: ?Sized,
{
    /// Tests if the name is a match for this plugin.
    match_fn: Box<dyn Fn(&TMatch) -> bool + Send + Sync + 'static>,
    /// Create an implementation of this plugin from input.
    create_fn: Box<dyn Fn(&TInput1, &TInput2) -> Result<TPlugin, TError> + Send + Sync + 'static>,
}

impl<TPlugin, TInput1, TInput2, TError, TMatch>
    RuntimePlugin2<TPlugin, TInput1, TInput2, TError, TMatch>
where
    TPlugin: MaybeSend + MaybeSync + 'static,
    TInput1: ?Sized + 'static,
    TInput2: ?Sized + 'static,
    TMatch: ?Sized,
{
    /// Create a new runtime plugin for registration.
    pub fn new<M, C>(match_fn: M, create_fn: C) -> Self
    where
        M: Fn(&TMatch) -> bool + Send + Sync + 'static,
        C: Fn(&TInput1, &TInput2) -> Result<TPlugin, TError> + Send + Sync + 'static,
    {
        Self {
            match_fn: Box::new(match_fn),
            create_fn: Box::new(create_fn),
        }
    }

    /// Create a `TPlugin` plugin from `input1` and `input2`.
    ///
    /// # Errors
    /// Returns a `TError` if plugin creation fails.
    pub fn create(&self, input1: &TInput1, input2: &TInput2) -> Result<TPlugin, TError> {
        (self.create_fn)(input1, input2)
    }

    /// Returns true if this plugin is associated with `match`.with `match`.
    // TODO: Rename to `match` on breaking release
    #[must_use]
    pub fn match_name(&self, r#match: &TMatch) -> bool {
        (self.match_fn)(r#match)
    }
}

// SAFETY: On WASM, execution is single-threaded, so Send/Sync are safe.
// The boxed closures require Send + Sync bounds which are auto-traits on WASM
// (everything is Send + Sync in single-threaded execution).
#[cfg(target_arch = "wasm32")]
mod wasm_impls {
    use super::{RuntimePlugin, RuntimePlugin2};

    unsafe impl<TPlugin: 'static, TInput: ?Sized + 'static, TError: 'static, TMatch: ?Sized> Send
        for RuntimePlugin<TPlugin, TInput, TError, TMatch>
    {
    }
    unsafe impl<TPlugin: 'static, TInput: ?Sized + 'static, TError: 'static, TMatch: ?Sized> Sync
        for RuntimePlugin<TPlugin, TInput, TError, TMatch>
    {
    }

    unsafe impl<
        TPlugin: 'static,
        TInput1: ?Sized + 'static,
        TInput2: ?Sized + 'static,
        TError: 'static,
        TMatch: ?Sized,
    > Send for RuntimePlugin2<TPlugin, TInput1, TInput2, TError, TMatch>
    {
    }
    unsafe impl<
        TPlugin: 'static,
        TInput1: ?Sized + 'static,
        TInput2: ?Sized + 'static,
        TError: 'static,
        TMatch: ?Sized,
    > Sync for RuntimePlugin2<TPlugin, TInput1, TInput2, TError, TMatch>
    {
    }
}
