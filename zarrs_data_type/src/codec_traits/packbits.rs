//! The `packbits` codec data type traits.

/// Traits for a data type supporting the `packbits` codec.
pub trait PackBitsDataTypeTraits {
    /// The component size in bits.
    fn component_size_bits(&self) -> u64;

    /// The number of components.
    fn num_components(&self) -> u64;

    /// True if the components need sign extension.
    ///
    /// This should be set to `true` for signed integer types.
    fn sign_extension(&self) -> bool;
}

// Generate the codec support infrastructure using the generic macro
crate::define_data_type_support!(PackBits);

/// Macro to implement `PackBitsDataTypeTraits` for data types and register support.
///
/// # Usage
/// ```ignore
/// // For single-component types:
/// zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits!(Int32DataType, 32, signed, 1);
/// zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits!(UInt32DataType, 32, unsigned, 1);
/// zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits!(Float32DataType, 32, float, 1);
///
/// // For complex types (2 components):
/// zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits!(Complex64DataType, 32, float, 2);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_pack_bits_data_type_traits {
    // Multi-component, signed integer
    ($marker:ty, $bits:expr, signed, $components:expr) => {
        impl $crate::codec_traits::PackBitsDataTypeTraits for $marker {
            fn component_size_bits(&self) -> u64 {
                $bits
            }
            fn num_components(&self) -> u64 {
                $components
            }
            fn sign_extension(&self) -> bool {
                true
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::PackBitsDataTypePlugin,
            $crate::codec_traits::PackBitsDataTypeTraits
        );
    };
    // Multi-component, unsigned integer
    ($marker:ty, $bits:expr, unsigned, $components:expr) => {
        impl $crate::codec_traits::PackBitsDataTypeTraits for $marker {
            fn component_size_bits(&self) -> u64 {
                $bits
            }
            fn num_components(&self) -> u64 {
                $components
            }
            fn sign_extension(&self) -> bool {
                false
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::PackBitsDataTypePlugin,
            $crate::codec_traits::PackBitsDataTypeTraits
        );
    };
    // Multi-component, float (no sign extension)
    ($marker:ty, $bits:expr, float, $components:expr) => {
        impl $crate::codec_traits::PackBitsDataTypeTraits for $marker {
            fn component_size_bits(&self) -> u64 {
                $bits
            }
            fn num_components(&self) -> u64 {
                $components
            }
            fn sign_extension(&self) -> bool {
                false
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::PackBitsDataTypePlugin,
            $crate::codec_traits::PackBitsDataTypeTraits
        );
    };
}

#[doc(inline)]
pub use _impl_pack_bits_data_type_traits as impl_pack_bits_data_type_traits;
