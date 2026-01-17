//! Generic codec support infrastructure for data type extensions.

/// Define codec support infrastructure for a data type extension codec.
///
/// This macro generates:
/// - A `{Name}DataTypePlugin` struct for inventory registration
/// - A `{Name}DataTypeExt` extension trait with a `codec_{name}()` method on [`DataType`](crate::DataType)
///
/// The data type traits for the codec must be in scope and named `{Name}DataTypeTraits`.
///
/// # Arguments
///
/// - `$name`: The codec name (e.g., `Bitround`, `Bytes`, `PackBits`)
///
/// # Example
///
/// ```ignore
/// zarrs_data_type::define_data_type_support!(MyCodec);
///
/// // This creates:
/// // - `MyCodecDataTypePlugin` struct for inventory registration
/// // - `MyCodecDataTypeExt` trait with `codec_mycodec()` method
/// // - Uses `MyCodecDataTypeTraits` as the trait name
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _define_data_type_support {
    ($name:ident) => {
        ::paste::paste! {
            #[doc = "Plugin for registering `" $name:lower "` codec support for a data type."]
            ///
            /// Use the [`register_data_type_extension_codec`](crate::register_data_type_extension_codec) macro to register.
            pub struct [<$name DataTypePlugin>] {
                /// The data type's [`TypeId`](std::any::TypeId).
                pub type_id: ::std::any::TypeId,
                /// Function that casts from `&dyn Any` to the codec trait.
                pub caster: fn(&dyn ::std::any::Any) -> ::core::option::Option<&dyn [<$name DataTypeTraits>]>,
            }

            ::inventory::collect!([<$name DataTypePlugin>]);

            type [<$name CasterFn>] = fn(&dyn ::std::any::Any) -> ::core::option::Option<&dyn [<$name DataTypeTraits>]>;

            static [<$name:upper _CASTERS>]: ::std::sync::LazyLock<
                ::std::collections::HashMap<::std::any::TypeId, [<$name CasterFn>]>
            > = ::std::sync::LazyLock::new(|| {
                ::inventory::iter::<[<$name DataTypePlugin>]>()
                    .map(|p| (p.type_id, p.caster))
                    .collect()
            });

            #[doc = "Extension trait for [`DataType`](crate::DataType) to access [`" $name "DataTypeTraits`]."]
            pub trait [<$name DataTypeExt>] {
                #[doc = "Get `" $name:lower "` codec support for this data type."]
                ///
                #[doc = "Returns `Ok(&dyn " $name "DataTypeTraits)` if the data type supports the codec,"]
                /// or an error if the data type does not support it.
                ///
                /// # Errors
                /// Returns [`DataTypeCodecError`](crate::DataTypeCodecError) if the data type does not support the codec.
                fn [<codec_ $name:lower>](&self) -> ::core::result::Result<&dyn [<$name DataTypeTraits>], $crate::DataTypeCodecError>;
            }

            impl [<$name DataTypeExt>] for $crate::DataType {
                fn [<codec_ $name:lower>](&self) -> ::core::result::Result<&dyn [<$name DataTypeTraits>], $crate::DataTypeCodecError> {
                    let caster = [<$name:upper _CASTERS>].get(&self.as_any().type_id())
                        .ok_or_else(|| $crate::DataTypeCodecError::UnsupportedDataType {
                            data_type: self.clone(),
                            codec_name: stringify!([<$name>]),
                        })?;
                    caster(self.as_any())
                        .ok_or_else(|| $crate::DataTypeCodecError::UnsupportedDataType {
                            data_type: self.clone(),
                            codec_name: stringify!([<$name>]),
                        })
                }
            }
        }
    };
}

/// Register a data type with a codec's `DataTypePlugin`.
///
/// This macro submits a data type to inventory for a specific codec's plugin,
/// allowing the codec to discover and use the data type at runtime.
///
/// # Arguments
///
/// - `$data_type`: The data type marker struct
/// - `$plugin`: The path to the `DataTypePlugin` struct (e.g., `zarrs_data_type::codec_traits::BitroundDataTypePlugin`)
/// - `$trait`: The path to the codec trait (e.g., `zarrs_data_type::codec_traits::BitroundDataTypeTraits`)
///
/// # Example
///
/// ```ignore
/// impl crate::array::codec::BitroundDataTypeTraits for Float32DataType {
///     // ...
/// }
///
/// zarrs_data_type::register_data_type_extension_codec!(
///     Float32DataType,
///     zarrs_data_type::codec_traits::BitroundDataTypePlugin,
///     zarrs_data_type::codec_traits::BitroundDataTypeTraits
/// );
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _register_data_type_extension_codec {
    ($data_type:ty, $plugin:path, $trait:path) => {
        ::inventory::submit! {
            $plugin {
                type_id: ::std::any::TypeId::of::<$data_type>(),
                caster: |any: &dyn ::std::any::Any| -> ::core::option::Option<&dyn $trait> {
                    any.downcast_ref::<$data_type>().map(|t| t as &dyn $trait)
                },
            }
        }
    };
}
