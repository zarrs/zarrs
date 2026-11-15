//! Generic codec support infrastructure for data type extensions.

/// Define codec support infrastructure for a data type extension codec.
///
/// This macro generates:
/// - A `{Name}DataTypePlugin` struct for inventory registration
/// - A `{Name}DataTypeExt` extension trait with `codec_{name}()` and `codec_{name}_arc()` methods on [`DataType`](crate::DataType)
/// - A `{Name}DataTypeTraitsArc` alias for the shared codec support returned by `codec_{name}_arc()`
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
/// // - `MyCodecDataTypeExt` trait with `codec_mycodec()` and `codec_mycodec_arc()` methods
/// // - `MyCodecDataTypeTraitsArc` type alias
/// // - Uses `MyCodecDataTypeTraits` as the trait name
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _define_data_type_support {
    ($name:ident) => {
        ::paste::paste! {
            #[doc = "A shared [`" $name "DataTypeTraits`] of a data type, returned by [`" $name "DataTypeExt::codec_" $name:lower "_arc`]."]
            #[cfg(not(target_arch = "wasm32"))]
            pub type [<$name DataTypeTraitsArc>] = ::std::sync::Arc<dyn [<$name DataTypeTraits>] + ::core::marker::Send + ::core::marker::Sync>;

            #[doc = "A shared [`" $name "DataTypeTraits`] of a data type, returned by [`" $name "DataTypeExt::codec_" $name:lower "_arc`]."]
            #[cfg(target_arch = "wasm32")]
            pub type [<$name DataTypeTraitsArc>] = ::std::sync::Arc<dyn [<$name DataTypeTraits>]>;

            #[doc = "Plugin for registering `" $name:lower "` codec support for a data type."]
            ///
            /// Use the [`register_data_type_extension_codec`](crate::register_data_type_extension_codec) macro to register.
            pub struct [<$name DataTypePlugin>] {
                /// The data type's [`TypeId`](std::any::TypeId).
                pub type_id: ::std::any::TypeId,
                /// Function that casts from `&dyn Any` to the codec trait.
                pub caster: fn(&dyn ::std::any::Any) -> ::core::option::Option<&dyn [<$name DataTypeTraits>]>,
                #[doc = "Function that casts a data type to a shared [`" $name "DataTypeTraitsArc`]."]
                pub arc_caster: fn(&$crate::DataType) -> ::core::option::Option<[<$name DataTypeTraitsArc>]>,
            }

            ::inventory::collect!([<$name DataTypePlugin>]);

            static [<$name:upper _PLUGINS>]: ::std::sync::LazyLock<
                ::std::collections::HashMap<::std::any::TypeId, &'static [<$name DataTypePlugin>]>
            > = ::std::sync::LazyLock::new(|| {
                ::inventory::iter::<[<$name DataTypePlugin>]>()
                    .map(|p| (p.type_id, p))
                    .collect()
            });

            impl [<$name DataTypePlugin>] {
                fn find(data_type: &$crate::DataType) -> ::core::result::Result<&'static Self, $crate::DataTypeCodecError> {
                    [<$name:upper _PLUGINS>]
                        .get(&data_type.as_any().type_id())
                        .copied()
                        .ok_or_else(|| Self::unsupported(data_type))
                }

                fn unsupported(data_type: &$crate::DataType) -> $crate::DataTypeCodecError {
                    $crate::DataTypeCodecError::UnsupportedDataType {
                        data_type: data_type.clone(),
                        codec_name: stringify!([<$name>]),
                    }
                }
            }

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

                #[doc = "Get shared `" $name:lower "` codec support for this data type, which can be retained without the data type."]
                ///
                #[doc = "This looks up the codec support of the data type once, whereas a retained [`" $name "DataTypeTraitsArc`] is used without a lookup."]
                ///
                /// # Errors
                /// Returns [`DataTypeCodecError`](crate::DataTypeCodecError) if the data type does not support the codec.
                fn [<codec_ $name:lower _arc>](&self) -> ::core::result::Result<[<$name DataTypeTraitsArc>], $crate::DataTypeCodecError>;
            }

            impl [<$name DataTypeExt>] for $crate::DataType {
                fn [<codec_ $name:lower>](&self) -> ::core::result::Result<&dyn [<$name DataTypeTraits>], $crate::DataTypeCodecError> {
                    let plugin = [<$name DataTypePlugin>]::find(self)?;
                    (plugin.caster)(self.as_any()).ok_or_else(|| [<$name DataTypePlugin>]::unsupported(self))
                }

                fn [<codec_ $name:lower _arc>](&self) -> ::core::result::Result<[<$name DataTypeTraitsArc>], $crate::DataTypeCodecError> {
                    let plugin = [<$name DataTypePlugin>]::find(self)?;
                    (plugin.arc_caster)(self).ok_or_else(|| [<$name DataTypePlugin>]::unsupported(self))
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
                arc_caster: |data_type| data_type.downcast_arc::<$data_type>().map(|t| t as _),
            }
        }
    };
}
