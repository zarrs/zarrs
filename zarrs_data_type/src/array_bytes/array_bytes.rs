use super::{
    ArrayBytesOffsets, ArrayBytesOptional, ArrayBytesRaw, ArrayBytesRawOffsetsOutOfBoundsError,
    ArrayBytesValidateError, ArrayBytesVariableLength, ExpectedFixedLengthBytesError,
    ExpectedOptionalBytesError, ExpectedVariableLengthBytesError, InvalidBytesLengthError,
};
use crate::{DataType, DataTypeFillValueError, FillValue};
use zarrs_metadata::DataTypeSize;

/// Fixed or variable length array bytes.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::From)]
pub enum ArrayBytes<'a> {
    /// Bytes for a fixed length array.
    ///
    /// These represent elements in C-contiguous order (i.e. row-major order) where the last dimension varies the fastest.
    Fixed(ArrayBytesRaw<'a>),
    /// Bytes and element byte offsets for a variable length array.
    Variable(ArrayBytesVariableLength<'a>),
    /// Bytes for an optional array (data with a validity mask).
    ///
    /// The data can be either Fixed or Variable length.
    /// The mask is validated at construction to have 1 byte per element.
    Optional(ArrayBytesOptional<'a>),
}

impl<'a> ArrayBytes<'a> {
    /// Create a new fixed length array bytes from `bytes`.
    ///
    /// `bytes` must be C-contiguous.
    #[must_use]
    pub fn new_flen(bytes: impl Into<ArrayBytesRaw<'a>>) -> Self {
        Self::Fixed(bytes.into())
    }

    /// Create a new variable length array bytes from `bytes` and `offsets`.
    ///
    /// # Errors
    /// Returns a [`ArrayBytesRawOffsetsOutOfBoundsError`] if the last offset is out of bounds of the bytes.
    pub fn new_vlen(
        bytes: impl Into<ArrayBytesRaw<'a>>,
        offsets: ArrayBytesOffsets<'a>,
    ) -> Result<Self, ArrayBytesRawOffsetsOutOfBoundsError> {
        ArrayBytesVariableLength::new(bytes, offsets).map(Self::Variable)
    }

    /// Create a new variable length array bytes from `bytes` and `offsets` without checking the offsets.
    ///
    /// # Safety
    /// The last offset must be less than or equal to the length of the bytes.
    #[must_use]
    pub unsafe fn new_vlen_unchecked(
        bytes: impl Into<ArrayBytesRaw<'a>>,
        offsets: ArrayBytesOffsets<'a>,
    ) -> Self {
        Self::Variable(unsafe { ArrayBytesVariableLength::new_unchecked(bytes, offsets) })
    }

    /// Wrap the array bytes with an optional validity mask.
    ///
    /// This creates an `Optional` variant that contains the current array bytes and the provided mask.
    #[must_use]
    pub fn with_optional_mask(self, mask: impl Into<ArrayBytesRaw<'a>>) -> Self {
        Self::Optional(ArrayBytesOptional::new(self, mask))
    }

    /// Create a new [`ArrayBytes`] with `num_elements` composed entirely of the `fill_value`.
    ///
    /// # Errors
    /// Returns [`DataTypeFillValueError`] if the fill value is incompatible with the data type.
    ///
    /// # Panics
    /// Panics if `num_elements` exceeds [`usize::MAX`].
    pub fn new_fill_value(
        data_type: &DataType,
        num_elements: u64,
        fill_value: &FillValue,
    ) -> Result<Self, DataTypeFillValueError> {
        if let Some(opt) = data_type.as_optional() {
            let num_elements_usize = usize::try_from(num_elements).unwrap();
            if opt.is_fill_value_null(fill_value) {
                // Null fill value for optional type: create mask of all zeros
                let inner_fill_value = if opt.is_fixed() {
                    FillValue::from(vec![0u8; opt.fixed_size().unwrap()])
                } else {
                    FillValue::from(&[])
                };
                let mask = vec![0u8; num_elements_usize];
                return Ok(ArrayBytes::new_fill_value(
                    opt.data_type(),
                    num_elements,
                    &inner_fill_value,
                )?
                .with_optional_mask(mask));
            }
            // Non-null fill value for optional type: strip suffix and use inner bytes
            let inner_bytes = opt.fill_value_inner_bytes(fill_value);
            let inner_fill_value = FillValue::new(inner_bytes.to_vec());
            let mask = vec![1u8; num_elements_usize]; // all non-null
            return Ok(ArrayBytes::new_fill_value(
                opt.data_type(),
                num_elements,
                &inner_fill_value,
            )?
            .with_optional_mask(mask));
        }

        match data_type.size() {
            DataTypeSize::Fixed(data_type_size) => {
                let num_elements = usize::try_from(num_elements).unwrap();
                if fill_value.size() == data_type_size {
                    Ok(Self::new_flen(
                        fill_value.as_ne_bytes().repeat(num_elements),
                    ))
                } else {
                    Err(DataTypeFillValueError)
                }
            }
            DataTypeSize::Variable => {
                let num_elements = usize::try_from(num_elements).unwrap();
                let offsets = unsafe {
                    // SAFETY: The offsets are monotonically increasing.
                    ArrayBytesOffsets::new_unchecked(
                        (0..=num_elements)
                            .map(|i| i * fill_value.size())
                            .collect::<Vec<_>>(),
                    )
                };
                Ok(unsafe {
                    // SAFETY: The last offset is equal to the length of the bytes
                    Self::new_vlen_unchecked(fill_value.as_ne_bytes().repeat(num_elements), offsets)
                })
            }
        }
    }

    /// Validate that the array bytes have a valid encoding for the given data type.
    ///
    /// # Errors
    /// Returns an [`ArrayBytesValidateError`] if the array bytes are not valid.
    ///
    /// # Panics
    /// Panics if `num_elements` exceeds [`usize::MAX`].
    pub fn validate(
        &self,
        num_elements: u64,
        data_type: &DataType,
    ) -> Result<(), ArrayBytesValidateError> {
        match self {
            ArrayBytes::Fixed(bytes) => {
                if data_type.is_optional() {
                    return Err(ArrayBytesValidateError::ExpectedOptionalBytes);
                }
                match data_type.size() {
                    DataTypeSize::Fixed(data_type_size) => {
                        let expected_len =
                            usize::try_from(num_elements * data_type_size as u64).unwrap();
                        if bytes.len() == expected_len {
                            Ok(())
                        } else {
                            Err(InvalidBytesLengthError::new(bytes.len(), expected_len).into())
                        }
                    }
                    DataTypeSize::Variable => {
                        Err(ArrayBytesValidateError::ExpectedVariableLengthBytes)
                    }
                }
            }
            ArrayBytes::Variable(vlen) => {
                if data_type.is_optional() {
                    return Err(ArrayBytesValidateError::ExpectedOptionalBytes);
                }
                match data_type.size() {
                    DataTypeSize::Variable => {
                        Self::validate_vlen(vlen.bytes(), vlen.offsets(), num_elements)
                    }
                    DataTypeSize::Fixed(_) => {
                        Err(ArrayBytesValidateError::ExpectedFixedLengthBytes)
                    }
                }
            }
            ArrayBytes::Optional(optional_bytes) => {
                let Some(opt) = data_type.as_optional() else {
                    return Err(ArrayBytesValidateError::UnexpectedOptionalBytes);
                };
                // Mask validation is already done at construction time
                // Just validate the underlying data with the inner type
                optional_bytes
                    .data()
                    .validate(num_elements, opt.data_type())
            }
        }
    }

    /// Validate variable length array bytes.
    fn validate_vlen(
        bytes: &ArrayBytesRaw<'_>,
        offsets: &ArrayBytesOffsets<'_>,
        num_elements: u64,
    ) -> Result<(), ArrayBytesValidateError> {
        if offsets.len() as u64 != num_elements + 1 {
            return Err(ArrayBytesValidateError::InvalidVariableSizedArrayOffsets);
        }
        let len = bytes.len();
        let mut offset_last = 0;
        for offset in offsets.iter() {
            if *offset < offset_last || *offset > len {
                return Err(ArrayBytesValidateError::InvalidVariableSizedArrayOffsets);
            }
            offset_last = *offset;
        }
        if offset_last == len {
            Ok(())
        } else {
            Err(ArrayBytesValidateError::InvalidVariableSizedArrayOffsets)
        }
    }

    /// Convert the array bytes into fixed length bytes.
    ///
    /// # Errors
    /// Returns an [`ExpectedFixedLengthBytesError`] if the bytes are not fixed.
    pub fn into_fixed(self) -> Result<ArrayBytesRaw<'a>, ExpectedFixedLengthBytesError> {
        match self {
            Self::Fixed(bytes) => Ok(bytes),
            Self::Variable(..) | Self::Optional(..) => Err(ExpectedFixedLengthBytesError),
        }
    }

    /// Convert the array bytes into variable length bytes and element byte offsets.
    ///
    /// # Errors
    /// Returns an [`ExpectedVariableLengthBytesError`] if the bytes are not variable.
    pub fn into_variable(
        self,
    ) -> Result<ArrayBytesVariableLength<'a>, ExpectedVariableLengthBytesError> {
        match self {
            Self::Fixed(..) | Self::Optional(..) => Err(ExpectedVariableLengthBytesError),
            Self::Variable(variable_length_bytes) => Ok(variable_length_bytes),
        }
    }

    /// Convert the array bytes into optional data and validity mask.
    ///
    /// # Errors
    /// Returns an [`ExpectedOptionalBytesError`] if the bytes are not optional.
    pub fn into_optional(self) -> Result<ArrayBytesOptional<'a>, ExpectedOptionalBytesError> {
        match self {
            Self::Optional(optional_bytes) => Ok(optional_bytes),
            Self::Fixed(..) | Self::Variable(..) => Err(ExpectedOptionalBytesError),
        }
    }

    /// Returns the size (in bytes) of the underlying element bytes.
    ///
    /// This only considers the size of the element bytes, and does not include the element offsets for a variable sized array or the mask for optional arrays.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::Fixed(bytes) => bytes.len(),
            Self::Variable(vlen) => vlen.bytes().len(),
            Self::Optional(optional_bytes) => optional_bytes.data().size(),
        }
    }

    /// Return the byte offsets for variable sized bytes. Returns [`None`] for fixed size bytes.
    #[must_use]
    pub fn offsets(&self) -> Option<&ArrayBytesOffsets<'a>> {
        match self {
            Self::Fixed(..) => None,
            Self::Variable(vlen) => Some(vlen.offsets()),
            Self::Optional(optional_bytes) => optional_bytes.data().offsets(),
        }
    }

    /// Convert into owned [`ArrayBytes<'static>`].
    #[must_use]
    pub fn into_owned(self) -> ArrayBytes<'static> {
        match self {
            Self::Fixed(bytes) => ArrayBytes::Fixed(bytes.into_owned().into()),
            Self::Variable(vlen) => ArrayBytes::Variable(vlen.into_owned()),
            Self::Optional(optional_bytes) => ArrayBytes::Optional(optional_bytes.into_owned()),
        }
    }

    /// Returns [`true`] if the array is empty for the given fill value.
    #[must_use]
    pub fn is_fill_value(&self, fill_value: &FillValue) -> bool {
        match self {
            Self::Fixed(bytes) => fill_value.equals_all(bytes),
            Self::Variable(vlen) => fill_value.equals_all(vlen.bytes()),
            Self::Optional(optional_bytes) => {
                let bytes = fill_value.as_ne_bytes();
                // For optional types, check the suffix byte (last byte)
                let is_null = bytes.last() == Some(&0);
                if is_null {
                    // For optional arrays with null fill value, check if mask is all zeros
                    optional_bytes.mask().iter().all(|&b| b == 0)
                } else {
                    // For optional arrays with non-null fill value, check if mask is all ones
                    // and data matches inner fill value (fill value without suffix)
                    let inner_fill_value = if bytes.is_empty() {
                        FillValue::new(vec![])
                    } else {
                        FillValue::new(bytes[..bytes.len() - 1].to_vec())
                    };
                    optional_bytes.mask().iter().all(|&b| b == 1)
                        && optional_bytes.data().is_fill_value(&inner_fill_value)
                }
            }
        }
    }
}

impl<'a> From<&'a [u8]> for ArrayBytes<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        ArrayBytes::new_flen(bytes)
    }
}

impl From<Vec<u8>> for ArrayBytes<'_> {
    fn from(bytes: Vec<u8>) -> Self {
        ArrayBytes::new_flen(bytes)
    }
}

impl<'a, const N: usize> From<&'a [u8; N]> for ArrayBytes<'a> {
    fn from(bytes: &'a [u8; N]) -> Self {
        ArrayBytes::new_flen(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array_bytes_vlen() {
        let data = [0u8, 1, 2, 3, 4];
        assert!(ArrayBytes::new_vlen(&data, vec![0].try_into().unwrap()).is_ok());
        assert!(ArrayBytes::new_vlen(&data, vec![0, 5].try_into().unwrap()).is_ok());
        assert!(ArrayBytes::new_vlen(&data, vec![0, 5, 5].try_into().unwrap()).is_ok());
        assert!(ArrayBytes::new_vlen(&data, vec![0, 5, 6].try_into().unwrap()).is_err());
        assert!(ArrayBytes::new_vlen(&data, vec![0, 1, 3, 5].try_into().unwrap()).is_ok());
        assert!(ArrayBytes::new_vlen(&data, vec![0, 1, 3, 6].try_into().unwrap()).is_err());
    }
}
