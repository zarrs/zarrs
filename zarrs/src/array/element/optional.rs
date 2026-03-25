use crate::array::{ArrayBytes, DataType};

use super::{Element, ElementError, ElementOwned};

use ElementError::IncompatibleElementType as IET;

impl<T> Element for Option<T>
where
    T: Element + Default,
{
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        let opt = data_type.as_optional().ok_or(IET)?;
        T::validate_data_type(opt.data_type())
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError> {
        Self::validate_data_type(data_type)?;

        let opt = data_type.as_optional().ok_or(IET)?;

        let num_elements = elements.len();

        // Create validity mask - one byte per element
        let mut mask = Vec::with_capacity(num_elements);

        // Create dense data - all elements, using default/zero for None values
        // We need to use a placeholder value for None elements
        let default_value = T::default();
        let mut dense_elements = Vec::with_capacity(num_elements);

        for element in elements {
            if let Some(value) = element {
                mask.push(1u8);
                dense_elements.push(value.clone());
            } else {
                mask.push(0u8);
                dense_elements.push(default_value.clone());
            }
        }

        // Convert all elements (dense) to ArrayBytes
        let data = T::into_array_bytes(opt.data_type(), dense_elements)?.into_owned();

        // Create optional ArrayBytes by adding mask to the data
        Ok(data.with_optional_mask(mask))
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
    }
}

impl<T> ElementOwned for Option<T>
where
    T: ElementOwned + Clone + Default,
{
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;

        let opt = data_type.as_optional().ok_or(IET)?;

        // Extract mask and dense data from optional ArrayBytes
        let optional_bytes = bytes.into_optional()?;
        let (data, mask) = optional_bytes.into_parts();

        // Convert the dense inner data to a Vec<T>
        let dense_values = T::from_array_bytes(opt.data_type(), *data)?;

        // Build the result vector using mask to determine Some vs None
        let mut elements = Vec::with_capacity(mask.len());
        for (i, &mask_byte) in mask.iter().enumerate() {
            if mask_byte == 0 {
                // None value
                elements.push(None);
            } else {
                // Some value - take from dense data
                if i >= dense_values.len() {
                    return Err(ElementError::Other(format!(
                        "Not enough dense values for mask at index {i}"
                    )));
                }
                elements.push(Some(dense_values[i].clone()));
            }
        }

        Ok(elements)
    }
}
