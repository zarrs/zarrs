//! The `fixedscaleoffset` codec data type traits.

/// The numeric element type for fixedscaleoffset operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedScaleOffsetElementType {
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
}

impl FixedScaleOffsetElementType {
    /// Returns the element size in bytes.
    #[must_use]
    pub const fn size(&self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }

    /// Returns the float type to use for intermediate calculations.
    #[must_use]
    pub const fn intermediate_float(&self) -> FixedScaleOffsetFloatType {
        match self {
            Self::I8 | Self::U8 | Self::I16 | Self::U16 | Self::F32 => {
                FixedScaleOffsetFloatType::F32
            }
            Self::I32 | Self::U32 | Self::I64 | Self::U64 | Self::F64 => {
                FixedScaleOffsetFloatType::F64
            }
        }
    }
}

/// The intermediate float type for fixedscaleoffset calculations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedScaleOffsetFloatType {
    /// 32-bit floating point
    F32,
    /// 64-bit floating point
    F64,
}

/// Traits for a data type supporting the `fixedscaleoffset` codec.
///
/// The fixedscaleoffset codec applies a linear transformation to numerical data.
pub trait FixedScaleOffsetDataTypeTraits {
    /// Returns the element type for this data type.
    fn fixedscaleoffset_element_type(&self) -> FixedScaleOffsetElementType;
}

// Generate the codec support infrastructure using the generic macro
crate::define_data_type_support!(FixedScaleOffset);

/// Macro to implement `FixedScaleOffsetDataTypeTraits` for data types and register support.
///
/// # Usage
/// ```ignore
/// zarrs_data_type::impl_fixed_scale_offset_data_type_traits!(Int32DataType, I32);
/// zarrs_data_type::impl_fixed_scale_offset_data_type_traits!(Float32DataType, F32);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_fixed_scale_offset_data_type_traits {
    ($marker:ty, $element_type:ident) => {
        impl $crate::codec_traits::FixedScaleOffsetDataTypeTraits for $marker {
            fn fixedscaleoffset_element_type(
                &self,
            ) -> $crate::codec_traits::FixedScaleOffsetElementType {
                $crate::codec_traits::FixedScaleOffsetElementType::$element_type
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::FixedScaleOffsetDataTypePlugin,
            $crate::codec_traits::FixedScaleOffsetDataTypeTraits
        );
    };
}

#[doc(inline)]
pub use _impl_fixed_scale_offset_data_type_traits as impl_fixed_scale_offset_data_type_traits;
