/// The pcodec element type for dispatching to the pcodec library.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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

/// Traits for a data type extension supporting the `pcodec` codec.
///
/// The pcodec codec losslessly compresses numerical data with high compression ratio.
pub trait DataTypeExtensionPcodecCodec {
    /// Returns the pcodec element type for this data type.
    fn pcodec_element_type(&self) -> PcodecElementType;

    /// Returns the number of elements per data type element.
    fn pcodec_elements_per_element(&self) -> usize;
}

// Generate the codec support infrastructure using the generic macro
crate::define_codec_support!(Pcodec, DataTypeExtensionPcodecCodec);
