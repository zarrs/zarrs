use crate::array::{
    ArrayBytes, DataType, convert_from_bytes_slice, data_type, transmute_to_bytes,
    transmute_to_bytes_vec,
};

use super::{Element, ElementError, ElementOwned};

use ElementError::IncompatibleElementType as IET;

impl<const N: usize> Element for &[u8; N] {
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        // RawBits and fixed size equal to N
        if data_type.is::<data_type::RawBitsDataType>() && data_type.fixed_size() == Some(N) {
            Ok(())
        } else {
            Err(IET)
        }
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes: Vec<u8> = elements.iter().flat_map(|i| i.iter()).copied().collect();
        Ok(bytes.into())
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes: Vec<u8> = elements.into_iter().flatten().copied().collect();
        Ok(bytes.into())
    }
}

impl<const N: usize> Element for [u8; N] {
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        // RawBits and fixed size equal to N
        if data_type.is::<data_type::RawBitsDataType>() && data_type.fixed_size() == Some(N) {
            Ok(())
        } else {
            Err(IET)
        }
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError> {
        Self::validate_data_type(data_type)?;
        Ok(transmute_to_bytes(elements).into())
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Self::validate_data_type(data_type)?;
        Ok(transmute_to_bytes_vec(elements).into())
    }
}

impl<const N: usize> ElementOwned for [u8; N] {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes = bytes.into_fixed()?;
        Ok(convert_from_bytes_slice::<Self>(&bytes))
    }
}
