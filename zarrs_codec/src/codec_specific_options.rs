//! Codec-specific options for codec initialisation.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Codec-specific options.
///
/// This is a type map for codec-specific runtime configuration that is set once and baked into
/// a codec's state (e.g. when opening or creating an array).
/// It is distinct from [`CodecOptions`](super::CodecOptions), which carries per-operation settings passed at each encode/decode call.
///
/// Codecs may define their own options type (e.g. `ShardingCodecOptions`) and retrieve it via [`get_option`](CodecSpecificOptions::get_option).
/// `zarrs_codec` itself has no knowledge of any specific codec options type.
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

#[cfg(test)]
mod tests {
    use super::CodecSpecificOptions;

    #[derive(Debug, Clone, Default, PartialEq)]
    struct FooOptions {
        value: u32,
    }

    #[derive(Debug, Clone, Default, PartialEq)]
    struct BarOptions {
        enabled: bool,
    }

    #[test]
    fn get_returns_none_when_not_set() {
        let opts = CodecSpecificOptions::default();
        assert!(opts.get_option::<FooOptions>().is_none());
    }

    #[test]
    fn get_returns_value_after_set() {
        let opts = CodecSpecificOptions::default().with_option(FooOptions { value: 42 });
        assert_eq!(opts.get_option::<FooOptions>().unwrap().value, 42);
    }

    #[test]
    fn second_set_replaces_first() {
        let opts = CodecSpecificOptions::default()
            .with_option(FooOptions { value: 1 })
            .with_option(FooOptions { value: 2 });
        assert_eq!(opts.get_option::<FooOptions>().unwrap().value, 2);
    }

    #[test]
    fn multiple_types_stored_independently() {
        let opts = CodecSpecificOptions::default()
            .with_option(FooOptions { value: 7 })
            .with_option(BarOptions { enabled: true });
        assert_eq!(opts.get_option::<FooOptions>().unwrap().value, 7);
        assert!(opts.get_option::<BarOptions>().unwrap().enabled);
    }

    #[test]
    fn clone_is_independent() {
        let opts = CodecSpecificOptions::default().with_option(FooOptions { value: 3 });
        let opts2 = opts.clone().with_option(FooOptions { value: 99 });
        assert_eq!(opts.get_option::<FooOptions>().unwrap().value, 3);
        assert_eq!(opts2.get_option::<FooOptions>().unwrap().value, 99);
    }

    #[test]
    fn debug_shows_entry_count() {
        let opts = CodecSpecificOptions::default()
            .with_option(FooOptions::default())
            .with_option(BarOptions::default());
        let s = format!("{opts:?}");
        assert!(s.contains("2 entries"), "got: {s}");
    }
}
