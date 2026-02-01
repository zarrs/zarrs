use std::mem::ManuallyDrop;

use crate::array::{
    ArrayBytes, DataType, convert_from_bytes_slice, data_type, transmute_to_bytes,
    transmute_to_bytes_vec,
};

use super::{Element, ElementError, ElementOwned};

impl Element for bool {
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        data_type
            .is::<data_type::BoolDataType>()
            .then_some(())
            .ok_or(ElementError::IncompatibleElementType)
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

impl ElementOwned for bool {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes = bytes.into_fixed()?;
        let elements_u8 = convert_from_bytes_slice::<u8>(&bytes);
        if elements_u8.iter().all(|&u| u <= 1) {
            let length: usize = elements_u8.len();
            let capacity: usize = elements_u8.capacity();
            let mut manual_drop_vec = ManuallyDrop::new(elements_u8);
            let vec_ptr: *mut u8 = manual_drop_vec.as_mut_ptr();
            let ptr: *mut Self = vec_ptr.cast::<Self>();
            Ok(unsafe { Vec::from_raw_parts(ptr, length, capacity) })
        } else {
            Err(ElementError::InvalidElementValue)
        }
    }
}
