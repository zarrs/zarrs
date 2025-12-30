/// The zfp data type for dispatching to the zfp library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZfpType {
    /// 32-bit integer (`zfp_type_int32`)
    Int32,
    /// 64-bit integer (`zfp_type_int64`)
    Int64,
    /// 32-bit float (`zfp_type_float`)
    Float,
    /// 64-bit float (`zfp_type_double`)
    Double,
}

/// Promotion strategy for types that need to be promoted before zfp encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ZfpPromotion {
    /// No promotion needed - use the type directly
    None,
    /// Promote i8 to i32 with left shift of 23 bits
    I8ToI32,
    /// Promote u8 to i32: (i32(u) - 0x80) << 23
    U8ToI32,
    /// Promote i16 to i32 with left shift of 15 bits
    I16ToI32,
    /// Promote u16 to i32: (i32(u) - 0x8000) << 15
    U16ToI32,
    /// Promote u32 to i32: min(u, `i32::MAX`) as i32
    U32ToI32,
    /// Promote u64 to i64: min(u, `i64::MAX`) as i64
    U64ToI64,
}

/// Traits for a data type extension supporting the `zfp` codec.
///
/// The zfp codec provides lossy and lossless compression for floating point and integer data.
pub trait DataTypeExtensionZfpCodec {
    /// Returns the zfp type for this data type.
    ///
    /// Returns `None` if the data type is not supported by zfp.
    fn zfp_type(&self) -> Option<ZfpType>;

    /// Returns the promotion strategy for encoding.
    ///
    /// For types that don't need promotion, returns `ZfpPromotion::None`.
    fn zfp_promotion(&self) -> ZfpPromotion {
        ZfpPromotion::None
    }
}

// Generate the codec support infrastructure using the generic macro
crate::define_codec_support!(Zfp, DataTypeExtensionZfpCodec);
