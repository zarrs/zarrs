//! Generic codec support infrastructure for data type extensions.
//!
//! This module provides macros for defining codec support registries in a generic way.
//!
//! # For Codec Authors
//!
//! Use [`define_data_type_support!`] to create all the infrastructure for a new codec:
//!
//! ```ignore
//! // Define the codec trait
//! pub trait MyCodecDataTypeTraits {
//!     fn my_method(&self) -> u32;
//! }
//!
//! // Generate the registry infrastructure
//! zarrs::array::codec::define_data_type_support!(MyCodec, MyCodecDataTypeTraits);
//!
//! // This creates:
//! // - `MyCodecPlugin` struct (for inventory registration)
//! // - `get_mycodec_support()` lookup function
//! ```
//!
//! # For Data Type Authors
//!
//! Use [`register_data_type_extension_codec!`] to register codec support:
//!
//! ```ignore
//! impl MyCodecDataTypeTraits for MyDataType {
//!     // ...
//! }
//!
//! zarrs::array::codec::register_data_type_extension_codec!(
//!     MyDataType,
//!     crate::path::to::MyCodecPlugin,
//!     crate::path::to::MyCodecDataTypeTraits
//! );
//! ```

/// Define codec support infrastructure for a data type extension codec.
///
/// This macro generates:
/// - A `{Name}Plugin` struct for inventory registration
/// - A `get_{name}_support()` function for looking up codec support
///
/// # Arguments
///
/// - `$name`: The codec name (e.g., `Bitround`, `Bytes`, `PackBits`)
/// - `$trait_name`: The codec trait (e.g., `BitroundCodecDataTypeTraits`)
///
/// # Example
///
/// ```ignore
/// zarrs::array::codec::define_data_type_support!(MyCodec, MyCodecDataTypeTraits);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _define_data_type_support {
    ($name:ident, $trait_name:ident) => {
        ::paste::paste! {
            /// Plugin for registering codec support for a data type.
            ///
            /// Use [`register_data_type_extension_codec!`](crate::array::codec::register_data_type_extension_codec) to register.
            pub struct [<$name Plugin>] {
                /// The data type identifier.
                pub data_type_id: &'static str,
                /// Function that casts from `&dyn Any` to the codec trait.
                pub caster: fn(&dyn ::std::any::Any) -> ::core::option::Option<&dyn $trait_name>,
            }

            ::inventory::collect!([<$name Plugin>]);

            type [<$name CasterFn>] = fn(&dyn ::std::any::Any) -> ::core::option::Option<&dyn $trait_name>;

            static [<$name:upper _CASTERS>]: ::std::sync::LazyLock<
                ::std::collections::HashMap<&'static str, [<$name CasterFn>]>
            > = ::std::sync::LazyLock::new(|| {
                ::inventory::iter::<[<$name Plugin>]>()
                    .map(|p| (p.data_type_id, p.caster))
                    .collect()
            });

            /// Get codec support for a data type.
            ///
            /// Returns `Some(&dyn Trait)` if the data type supports the codec.
            pub fn [<get_ $name:lower _support>](
                data_type: &zarrs_data_type::DataType,
            ) -> ::core::option::Option<&dyn $trait_name> {
                let caster = [<$name:upper _CASTERS>].get(data_type.identifier())?;
                caster(data_type.as_any())
            }
        }
    };
}

#[doc(inline)]
pub use _define_data_type_support as define_data_type_support;

/// Register a data type with a codec's `Plugin`.
///
/// This macro submits a data type to inventory for a specific codec's plugin,
/// allowing the codec to discover and use the data type at runtime.
///
/// # Arguments
///
/// - `$data_type`: The data type marker struct (must implement `ExtensionIdentifier`)
/// - `$plugin`: The path to the `Plugin` struct (e.g., `$crate::array::codec::BitroundPlugin`)
/// - `$trait`: The path to the codec trait (e.g., `$crate::array::codec::BitroundCodecDataTypeTraits`)
///
/// # Example
///
/// ```ignore
/// impl crate::array::codec::BitroundCodecDataTypeTraits for Float32DataType {
///     // ...
/// }
///
/// crate::array::codec::register_data_type_extension_codec!(
///     Float32DataType,
///     $crate::array::codec::BitroundPlugin,
///     $crate::array::codec::BitroundCodecDataTypeTraits
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _register_data_type_extension_codec {
    ($data_type:ty, $plugin:path, $trait:path) => {
        ::inventory::submit! {
            $plugin {
                data_type_id: <$data_type as ::zarrs_plugin::ExtensionIdentifier>::IDENTIFIER,
                caster: |any: &dyn ::std::any::Any| -> ::core::option::Option<&dyn $trait> {
                    any.downcast_ref::<$data_type>().map(|t| t as &dyn $trait)
                },
            }
        }
    };
}

#[doc(inline)]
pub use _register_data_type_extension_codec as register_data_type_extension_codec;
