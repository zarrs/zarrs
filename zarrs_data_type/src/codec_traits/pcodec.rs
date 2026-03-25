//! The `pcodec` codec data type traits.

/// The pcodec element type for dispatching to the pcodec library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcodecElementType {
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 16-bit floating point
    F16,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
}

impl PcodecElementType {
    /// Returns the element size in bytes.
    #[must_use]
    pub const fn size(&self) -> usize {
        match self {
            Self::U16 | Self::I16 | Self::F16 => 2,
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::U64 | Self::I64 | Self::F64 => 8,
        }
    }
}

/// Traits for a data type supporting the `pcodec` codec.
///
/// The pcodec codec losslessly compresses numerical data with high compression ratio.
pub trait PcodecDataTypeTraits {
    /// Returns the pcodec element type for this data type.
    fn pcodec_element_type(&self) -> PcodecElementType;

    /// Returns the number of elements per data type element.
    fn pcodec_elements_per_element(&self) -> usize;
}

// Generate the codec support infrastructure using the generic macro
crate::define_data_type_support!(Pcodec);

/// Macro to implement `PcodecDataTypeTraits` for data types and register support.
///
/// # Usage
/// ```ignore
/// zarrs_data_type::impl_pcodec_data_type_traits!(Int32DataType, I32, 1);
/// zarrs_data_type::impl_pcodec_data_type_traits!(Float32DataType, F32, 1);
/// zarrs_data_type::impl_pcodec_data_type_traits!(Complex64DataType, F32, 2);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_pcodec_data_type_traits {
    ($marker:ty, $element_type:ident, $elements_per_element:expr) => {
        impl $crate::codec_traits::pcodec::PcodecDataTypeTraits for $marker {
            fn pcodec_element_type(&self) -> $crate::codec_traits::pcodec::PcodecElementType {
                $crate::codec_traits::pcodec::PcodecElementType::$element_type
            }
            fn pcodec_elements_per_element(&self) -> usize {
                $elements_per_element
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::pcodec::PcodecDataTypePlugin,
            $crate::codec_traits::pcodec::PcodecDataTypeTraits
        );
    };
}

#[doc(inline)]
pub use _impl_pcodec_data_type_traits as impl_pcodec_data_type_traits;
