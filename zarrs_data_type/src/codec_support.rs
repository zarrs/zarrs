//! Generic codec support infrastructure for data type extensions.

use std::any::TypeId;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// A map keyed by [`TypeId`], which hashes a [`TypeId`] as itself as it is already a hash.
#[doc(hidden)]
pub type TypeIdMap<V> = HashMap<TypeId, V, BuildHasherDefault<TypeIdHasher>>;

/// A [`Hasher`] for a [`TypeId`], which [hashes](std::hash::Hash) as a [`u64`] that is already a hash.
#[doc(hidden)]
#[derive(Default)]
pub struct TypeIdHasher(u64);

impl Hasher for TypeIdHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // Not used by `TypeId`, but correct for any input
        for &byte in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(byte);
        }
    }

    fn write_u64(&mut self, i: u64) {
        self.0 ^= i;
    }
}

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
                $crate::TypeIdMap<[<$name CasterFn>]>
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
                        .ok_or_else(|| [<$name:lower _unsupported>](self))?;
                    caster(self.as_any()).ok_or_else(|| [<$name:lower _unsupported>](self))
                }
            }

            fn [<$name:lower _unsupported>](data_type: &$crate::DataType) -> $crate::DataTypeCodecError {
                $crate::DataTypeCodecError::UnsupportedDataType {
                    data_type: data_type.clone(),
                    codec_name: stringify!([<$name>]),
                }
            }

            #[doc = "The `" $name:lower "` codec support of a [`DataType`](crate::DataType), looked up once."]
            ///
            #[doc = "[`" $name DataTypeExt "`] looks up the support of a data type in a registry on each call, whereas [`get`](Self::get) only casts the data type."]
            #[derive(Clone, Debug)]
            pub struct [<$name DataTypeSupport>] {
                data_type: $crate::DataType,
                caster: [<$name CasterFn>],
            }

            impl [<$name DataTypeSupport>] {
                #[doc = "Look up the `" $name:lower "` codec support of `data_type`."]
                ///
                /// # Errors
                /// Returns [`DataTypeCodecError`](crate::DataTypeCodecError) if the data type does not support the codec.
                pub fn new(data_type: &$crate::DataType) -> ::core::result::Result<Self, $crate::DataTypeCodecError> {
                    let caster = *[<$name:upper _CASTERS>].get(&data_type.as_any().type_id())
                        .ok_or_else(|| [<$name:lower _unsupported>](data_type))?;
                    caster(data_type.as_any()).ok_or_else(|| [<$name:lower _unsupported>](data_type))?;
                    Ok(Self {
                        data_type: data_type.clone(),
                        caster,
                    })
                }

                /// The data type.
                #[must_use]
                #[inline]
                pub fn data_type(&self) -> &$crate::DataType {
                    &self.data_type
                }

                #[doc = "The `" $name:lower "` codec support of the data type."]
                #[must_use]
                #[inline]
                pub fn get(&self) -> &dyn [<$name DataTypeTraits>] {
                    (self.caster)(self.data_type.as_any())
                        .unwrap_or_else(|| unreachable!("the caster of a data type casts it"))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_id_map() {
        let map: TypeIdMap<&str> = [
            (TypeId::of::<u8>(), "u8"),
            (TypeId::of::<u16>(), "u16"),
            (TypeId::of::<String>(), "String"),
        ]
        .into_iter()
        .collect();
        assert_eq!(map.get(&TypeId::of::<u16>()), Some(&"u16"));
        assert_eq!(map.get(&TypeId::of::<String>()), Some(&"String"));
        assert_eq!(map.get(&TypeId::of::<u32>()), None);
    }
}
