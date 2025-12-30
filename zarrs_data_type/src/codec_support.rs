//! Generic codec support infrastructure for data type extensions.
//!
//! This module provides macros for defining codec support registries in a generic way.
//!
//! # For Codec Authors
//!
//! Use [`define_codec_support!`] to create all the infrastructure for a new codec:
//!
//! ```ignore
//! use zarrs_data_type::define_codec_support;
//!
//! // Define the codec trait
//! pub trait DataTypeExtensionMyCodec {
//!     fn my_method(&self) -> u32;
//! }
//!
//! // Generate the registry infrastructure
//! define_codec_support!(MyCodec, DataTypeExtensionMyCodec);
//!
//! // This creates:
//! // - `MyCodecCasterPlugin` struct
//! // - `get_mycodec_support()` lookup function
//! // - `register_mycodec_support!` macro
//! ```
//!
//! # For Data Type Authors
//!
//! Use the generated `register_*_support!` macro to register codec support:
//!
//! ```ignore
//! use zarrs_data_type::register_bytes_support;
//!
//! impl DataTypeExtensionBytesCodec for MyDataType {
//!     // ...
//! }
//!
//! register_bytes_support!(MyDataType);
//! ```

/// Define codec support infrastructure for a data type extension codec.
///
/// This macro generates:
/// - A `{Name}CasterPlugin` struct for inventory registration
/// - A `get_{name}_support()` function for looking up codec support
/// - A `register_{name}_support!` macro for registering data types
///
/// # Arguments
///
/// - `$name`: The codec name (e.g., `Bitround`, `Bytes`, `PackBits`)
/// - `$trait_name`: The codec trait (e.g., `DataTypeExtensionBitroundCodec`)
///
/// # Example
///
/// ```ignore
/// define_codec_support!(MyCodec, DataTypeExtensionMyCodec);
/// ```
#[macro_export]
macro_rules! define_codec_support {
    ($name:ident, $trait_name:ident) => {
        ::paste::paste! {
            /// Plugin for registering codec support for a data type.
            ///
            /// This is used internally by the registration macro.
            pub struct [<$name CasterPlugin>] {
                /// The data type identifier.
                pub data_type_id: &'static str,
                /// Function that casts from `&dyn Any` to the codec trait.
                pub caster: fn(&dyn ::std::any::Any) -> ::core::option::Option<&dyn $trait_name>,
            }

            ::inventory::collect!([<$name CasterPlugin>]);

            type [<$name CasterFn>] = fn(&dyn ::std::any::Any) -> ::core::option::Option<&dyn $trait_name>;

            static [<$name:upper _CASTERS>]: ::std::sync::LazyLock<
                ::std::collections::HashMap<&'static str, [<$name CasterFn>]>
            > = ::std::sync::LazyLock::new(|| {
                ::inventory::iter::<[<$name CasterPlugin>]>()
                    .map(|p| (p.data_type_id, p.caster))
                    .collect()
            });

            /// Get codec support for a data type.
            ///
            /// Returns `Some(&dyn Trait)` if the data type supports the codec.
            pub fn [<get_ $name:lower _support>](
                data_type: &dyn $crate::DataTypeExtension,
            ) -> ::core::option::Option<&dyn $trait_name> {
                let caster = [<$name:upper _CASTERS>].get(data_type.identifier())?;
                caster(data_type.as_any())
            }

            /// Register that a data type supports this codec.
            ///
            /// The data type must implement the codec trait.
            #[macro_export]
            macro_rules! [<register_ $name:lower _support>] {
                ($data_type:ty) => {
                    ::inventory::submit! {
                        $crate::[<$name CasterPlugin>] {
                            data_type_id: <$data_type as ::zarrs_plugin::ExtensionIdentifier>::IDENTIFIER,
                            caster: |any: &dyn ::std::any::Any| -> ::core::option::Option<&dyn $crate::$trait_name> {
                                any.downcast_ref::<$data_type>().map(|t| t as &dyn $crate::$trait_name)
                            },
                        }
                    }
                };
            }
        }
    };
}
