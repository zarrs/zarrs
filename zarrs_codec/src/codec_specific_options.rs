//! Codec-specific options for codec initialisation.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Codec-specific options, keyed by type.
///
/// This is a type map for codec-specific runtime configuration that is applied once at
/// array creation or opening time and baked into the codec's state.
/// It is separate from [`CodecOptions`](super::CodecOptions), which carries per-operation settings.
///
/// Each codec defines its own options type and retrieves it via [`CodecSpecificOptions::get_option`].
/// `zarrs_codec` itself has no knowledge of any specific codec type.
///
/// # Example
/// ```rust
/// # use zarrs_codec::CodecSpecificOptions;
/// #[derive(Debug, Clone, Default)]
/// pub struct MyCodecOptions {
///     pub some_setting: usize,
/// }
///
/// let opts = CodecSpecificOptions::default()
///     .with_option(MyCodecOptions { some_setting: 16 });
///
/// let retrieved = opts.get_option::<MyCodecOptions>().unwrap();
/// assert_eq!(retrieved.some_setting, 16);
/// ```
#[derive(Clone, Default)]
pub struct CodecSpecificOptions {
    map: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl fmt::Debug for CodecSpecificOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CodecSpecificOptions {{ {} entries }}", self.map.len())
    }
}

impl CodecSpecificOptions {
    /// Attach an option for a specific codec, keyed by its type.
    ///
    /// If an option of the same type already exists it is replaced.
    #[must_use]
    pub fn with_option<T: Any + Send + Sync>(mut self, option: T) -> Self {
        self.map.insert(TypeId::of::<T>(), Arc::new(option));
        self
    }

    /// Retrieve an option by type, returning `None` if it was not set.
    #[must_use]
    pub fn get_option<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.map.get(&TypeId::of::<T>())?.downcast_ref::<T>()
    }
}
