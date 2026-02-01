//! The `zfp` codec data type traits.

/// The native zfp type for the zfp library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZfpNativeType {
    /// 32-bit integer (`zfp_type_int32`)
    Int32,
    /// 64-bit integer (`zfp_type_int64`)
    Int64,
    /// 32-bit float (`zfp_type_float`)
    Float,
    /// 64-bit float (`zfp_type_double`)
    Double,
}

/// The zfp encoding strategy for a data type.
///
/// Each variant represents a source data type. Types that don't map directly to
/// zfp's native types (int32, int64, float, double) are promoted/demoted internally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZfpEncoding {
    /// i8 data (promoted to i32 internally with left shift 23)
    Int8,
    /// i16 data (promoted to i32 internally with left shift 15)
    Int16,
    /// i32 data (native zfp type)
    Int32,
    /// i64 data (native zfp type)
    Int64,
    /// u8 data (promoted to i32 internally: (i32(u) - 0x80) << 23)
    UInt8,
    /// u16 data (promoted to i32 internally: (i32(u) - 0x8000) << 15)
    UInt16,
    /// u32 data (clamped to `i32::MAX` internally)
    UInt32,
    /// u64 data (clamped to `i64::MAX` internally)
    UInt64,
    /// f32 data (native zfp type)
    Float32,
    /// f64 data (native zfp type)
    Float64,
}

impl ZfpEncoding {
    /// Returns the native zfp type for this encoding.
    #[must_use]
    pub const fn native_type(self) -> ZfpNativeType {
        match self {
            Self::Int8 | Self::Int16 | Self::Int32 | Self::UInt8 | Self::UInt16 | Self::UInt32 => {
                ZfpNativeType::Int32
            }
            Self::Int64 | Self::UInt64 => ZfpNativeType::Int64,
            Self::Float32 => ZfpNativeType::Float,
            Self::Float64 => ZfpNativeType::Double,
        }
    }
}

/// Traits for a data type supporting the `zfp` codec.
///
/// The zfp codec provides lossy and lossless compression for floating point and integer data.
pub trait ZfpDataTypeTraits {
    /// Returns the zfp encoding strategy for this data type.
    fn zfp_encoding(&self) -> ZfpEncoding;
}

// Generate the codec support infrastructure using the generic macro
crate::define_data_type_support!(Zfp);

/// Macro to implement `ZfpDataTypeTraits` for data types and register support.
///
/// # Usage
/// ```ignore
/// zarrs_data_type::impl_zfp_data_type_traits!(Int8DataType, Int8);
/// zarrs_data_type::impl_zfp_data_type_traits!(Int32DataType, Int32);
/// zarrs_data_type::impl_zfp_data_type_traits!(Float32DataType, Float32);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_zfp_data_type_traits {
    ($marker:ty, $encoding:ident) => {
        impl $crate::codec_traits::zfp::ZfpDataTypeTraits for $marker {
            fn zfp_encoding(&self) -> $crate::codec_traits::zfp::ZfpEncoding {
                $crate::codec_traits::zfp::ZfpEncoding::$encoding
            }
        }
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::zfp::ZfpDataTypePlugin,
            $crate::codec_traits::zfp::ZfpDataTypeTraits
        );
    };
}

#[doc(inline)]
pub use _impl_zfp_data_type_traits as impl_zfp_data_type_traits;
