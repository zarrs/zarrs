use std::borrow::Cow;

use crate::ZarrVersion;

/// Trait for types that provide a static default name function.
///
/// This trait is implemented by the `impl_extension_aliases!` macro.
/// It provides a const function pointer that can be used to get the default name
/// for a given Zarr version, enabling a blanket implementation of [`ExtensionName`].
pub trait ExtensionNameStatic {
    /// Function pointer to get the default name for a given Zarr version.
    ///
    /// Returns `None` if this extension does not support the given version
    /// (indicated by an empty `default_name` in the aliases configuration).
    const DEFAULT_NAME_FN: fn(ZarrVersion) -> Option<Cow<'static, str>>;
}

/// Object-safe trait for getting the name of an extension instance.
///
/// This trait provides a way to get the serialization name for an extension
/// instance for a given Zarr version. The default implementation (via
/// [`ExtensionNameStatic`]) returns the first alias from [`ExtensionAliases`](crate::ExtensionAliases)
/// for that version.
///
/// Types can implement this trait manually if they need custom naming behavior.
pub trait ExtensionName {
    /// Get the name for this extension instance for the given Zarr version.
    ///
    /// Returns `None` if this extension does not support the given version.
    fn name(&self, version: ZarrVersion) -> Option<Cow<'static, str>>;

    /// Get the name for this extension instance for Zarr V3.
    ///
    /// Returns `None` if this extension does not support V3.
    fn name_v3(&self) -> Option<Cow<'static, str>> {
        self.name(ZarrVersion::V3)
    }

    /// Get the name for this extension instance for Zarr V2.
    ///
    /// Returns `None` if this extension does not support V2.
    fn name_v2(&self) -> Option<Cow<'static, str>> {
        self.name(ZarrVersion::V2)
    }
}

// Blanket implementation for types that implement ExtensionNameStatic.
// Types can override this by implementing ExtensionName directly.
impl<T: ExtensionNameStatic> ExtensionName for T {
    fn name(&self, version: ZarrVersion) -> Option<Cow<'static, str>> {
        (T::DEFAULT_NAME_FN)(version)
    }
}

// Blanket implementation for Arc<T> where T: ExtensionName.
// This allows Arc<dyn TraitWithExtensionName> to implement ExtensionName.
impl<T: ExtensionName + ?Sized> ExtensionName for std::sync::Arc<T> {
    fn name(&self, version: ZarrVersion) -> Option<Cow<'static, str>> {
        self.as_ref().name(version)
    }
}
