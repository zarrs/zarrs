use std::any::TypeId;

use itertools::Itertools;

use crate::array::{ArrayBytes, ArrayBytesOffsets, DataType};

use super::{Element, ElementError, ElementOwned};

use ElementError::IncompatibleElementType as IET;

macro_rules! impl_element_string {
    ($raw_type:ty) => {
        impl Element for $raw_type {
            fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
                let my_type_id = TypeId::of::<$raw_type>();
                data_type
                    .compatible_element_types()
                    .contains(&my_type_id)
                    .then_some(())
                    .ok_or(IET)
            }

            fn to_array_bytes<'a>(
                data_type: &DataType,
                elements: &'a [Self],
            ) -> Result<ArrayBytes<'a>, ElementError> {
                Self::validate_data_type(data_type)?;

                // Calculate offsets
                let mut len: usize = 0;
                let mut offsets = Vec::with_capacity(elements.len());
                for element in elements {
                    offsets.push(len);
                    len = len.checked_add(element.len()).unwrap();
                }
                offsets.push(len);
                let offsets = unsafe {
                    // SAFETY: The offsets are monotonically increasing.
                    ArrayBytesOffsets::new_unchecked(offsets)
                };

                // Concatenate bytes
                let mut bytes = Vec::with_capacity(usize::try_from(len).unwrap());
                for element in elements {
                    bytes.extend_from_slice(element.as_bytes());
                }
                let array_bytes = unsafe {
                    // SAFETY: The last offset is the length of the bytes.
                    ArrayBytes::new_vlen_unchecked(bytes, offsets)
                };
                Ok(array_bytes)
            }

            fn into_array_bytes(
                data_type: &DataType,
                elements: Vec<Self>,
            ) -> Result<ArrayBytes<'static>, ElementError> {
                Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
            }
        }
    };
}

impl_element_string!(&str);
impl_element_string!(String);

impl ElementOwned for String {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;
        let (bytes, offsets) = bytes.into_variable()?.into_parts();
        let mut elements = Vec::with_capacity(offsets.len().saturating_sub(1));
        for (curr, next) in offsets.iter().tuple_windows() {
            elements.push(
                Self::from_utf8(bytes[*curr..*next].to_vec())
                    .map_err(|_| ElementError::InvalidElementValue)?,
            );
        }
        Ok(elements)
    }
}
