//! The `scale_offset` codec data type traits.

use num::traits::{
    CheckedAdd, CheckedDiv, CheckedMul, CheckedSub, Float, FromBytes, ToBytes, Zero,
};
use thiserror::Error;

/// Errors raised while applying the `scale_offset` transformation.
#[derive(Clone, Debug, Error)]
pub enum ScaleOffsetError {
    /// Invalid element bytes.
    #[error("invalid element bytes for scale_offset")]
    InvalidElementBytes,
    /// A value produced during the transformation is not representable in the data type.
    #[error("scale_offset produced a value that is not representable in the data type")]
    NotRepresentable,
    /// The `scale` is zero, so decoding (division) is undefined.
    #[error("scale_offset scale must be non-zero for decoding")]
    DivisionByZero,
}

/// Traits for a data type supporting the `scale_offset` codec.
///
/// The `scale_offset` codec applies the linear transformation `(value - offset) * scale`
/// on encode and the inverse `(value / scale) + offset` on decode, using the arithmetic
/// semantics of the data type.
///
/// `offset` and `scale` are provided as native-endian bytes of this data type (i.e. encoded
/// with the Zarr V3 fill value encoding of this data type). A value of `None` represents the
/// identity element (`0` for `offset`, `1` for `scale`).
pub trait ScaleOffsetDataTypeTraits {
    /// Apply the encoding transformation `(value - offset) * scale` to each element in `bytes`.
    ///
    /// # Errors
    /// Returns an error if `bytes`, `offset`, or `scale` are not valid element bytes, or if a
    /// transformed value is not representable in the data type.
    fn scale_offset_encode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError>;

    /// Apply the decoding transformation `(value / scale) + offset` to each element in `bytes`.
    ///
    /// # Errors
    /// Returns an error if `bytes`, `offset`, or `scale` are not valid element bytes, if `scale`
    /// is zero, or if a transformed value is not representable in the data type.
    fn scale_offset_decode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError>;
}

// Generate the codec support infrastructure using the generic macro.
crate::define_data_type_support!(ScaleOffset);

/// Compute `(value - offset) * scale` for an integer, using checked arithmetic in the native type.
///
/// # Errors
/// Returns [`ScaleOffsetError::NotRepresentable`] if the subtraction or multiplication overflows
/// the native type `T`.
pub fn scale_offset_encode_int<T>(value: &T, offset: &T, scale: &T) -> Result<T, ScaleOffsetError>
where
    T: CheckedSub + CheckedMul,
{
    value
        .checked_sub(offset)
        .and_then(|difference| difference.checked_mul(scale))
        .ok_or(ScaleOffsetError::NotRepresentable)
}

/// Compute `(value / scale) + offset` for an integer, using checked arithmetic in the native type.
///
/// # Errors
/// Returns [`ScaleOffsetError::DivisionByZero`] if `scale` is zero, or
/// [`ScaleOffsetError::NotRepresentable`] if the division or addition overflows the native type `T`.
pub fn scale_offset_decode_int<T>(value: &T, offset: &T, scale: &T) -> Result<T, ScaleOffsetError>
where
    T: CheckedDiv + CheckedAdd + Zero,
{
    if scale.is_zero() {
        return Err(ScaleOffsetError::DivisionByZero);
    }
    value
        .checked_div(scale)
        .and_then(|quotient| quotient.checked_add(offset))
        .ok_or(ScaleOffsetError::NotRepresentable)
}

/// Apply the `scale_offset` transformation to a slice of bytes representing a float type.
///
/// `op` is the per-element operation (encode: `(value - offset) * scale`, decode: `(value / scale) + offset`).
/// Float infinity and NaN are valid results — no overflow check is performed.
///
/// # Errors
/// Returns [`ScaleOffsetError::InvalidElementBytes`] if the offset or scale bytes have the wrong length.
pub fn scale_offset_float<T, B, F>(
    bytes: &mut [u8],
    offset: Option<&[u8]>,
    scale: Option<&[u8]>,
    op: F,
) -> Result<(), ScaleOffsetError>
where
    T: Float + FromBytes<Bytes = B> + ToBytes<Bytes = B>,
    B: AsRef<[u8]> + for<'a> TryFrom<&'a [u8], Error = std::array::TryFromSliceError>,
    F: Fn(T, T, T) -> T,
{
    let offset: T = match offset {
        Some(bytes) => {
            let arr: B = bytes
                .try_into()
                .map_err(|_| ScaleOffsetError::InvalidElementBytes)?;
            T::from_ne_bytes(&arr)
        }
        None => T::zero(),
    };
    let scale: T = match scale {
        Some(bytes) => {
            let arr: B = bytes
                .try_into()
                .map_err(|_| ScaleOffsetError::InvalidElementBytes)?;
            T::from_ne_bytes(&arr)
        }
        None => T::one(),
    };
    let elem_size = scale.to_ne_bytes().as_ref().len();
    for chunk in bytes.chunks_exact_mut(elem_size) {
        let arr: B = chunk.as_ref().try_into().unwrap();
        let value = T::from_ne_bytes(&arr);
        let result = op(value, offset, scale);
        chunk.copy_from_slice(result.to_ne_bytes().as_ref());
    }
    Ok(())
}

/// Macro to register a data type as supporting the `scale_offset` codec.
///
/// The actual [`ScaleOffsetDataTypeTraits`] implementation should be provided by the caller
/// (typically in the data type's source file). This macro only handles plugin registration.
///
/// # Usage
/// ```ignore
/// // In the data type file:
/// impl zarrs_data_type::codec_traits::scale_offset::ScaleOffsetDataTypeTraits for MyDataType {
///     fn scale_offset_encode(&self, bytes: &mut [u8], offset: Option<&[u8]>, scale: Option<&[u8]>)
///         -> Result<(), zarrs_data_type::codec_traits::scale_offset::ScaleOffsetError> { ... }
///     fn scale_offset_decode(&self, bytes: &mut [u8], offset: Option<&[u8]>, scale: Option<&[u8]>)
///         -> Result<(), zarrs_data_type::codec_traits::scale_offset::ScaleOffsetError> { ... }
/// }
///
/// zarrs_data_type::impl_scale_offset_data_type_traits!(MyDataType);
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! _impl_scale_offset_data_type_traits {
    ($marker:ty) => {
        $crate::register_data_type_extension_codec!(
            $marker,
            $crate::codec_traits::scale_offset::ScaleOffsetDataTypePlugin,
            $crate::codec_traits::scale_offset::ScaleOffsetDataTypeTraits
        );
    };
}

#[doc(inline)]
pub use _impl_scale_offset_data_type_traits as impl_scale_offset_data_type_traits;
