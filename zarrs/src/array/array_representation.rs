use std::num::NonZeroU64;

use super::{ArrayShape, DataType, DataTypeSize, FillValue};
use crate::array::data_type::DataTypeFillValueError;
use derive_more::Display;

/// The shape, data type, and fill value of an `array`.
#[derive(Clone, Debug, Display)]
#[display("{array_shape:?} {data_type} {fill_value}")]
pub struct ArrayRepresentationBase<TDim>
where
    TDim: Into<u64> + core::fmt::Debug + Copy,
{
    /// The shape of the array.
    array_shape: Vec<TDim>,
    /// The data type of the array.
    data_type: DataType,
    /// The fill value of the array.
    fill_value: FillValue,
}

/// The array representation of an array, which can have zero dimensions.
pub type ArrayRepresentation = ArrayRepresentationBase<u64>;

/// The array representation of a chunk, which must have nonzero dimensions.
pub type ChunkRepresentation = ArrayRepresentationBase<NonZeroU64>;

impl<TDim> ArrayRepresentationBase<TDim>
where
    TDim: Into<u64> + core::fmt::Debug + Copy,
{
    /// Create a new [`ArrayRepresentation`].
    ///
    /// # Errors
    ///
    /// Returns [`DataTypeFillValueError`] if the `data_type` and `fill_value` are incompatible.
    pub fn new(
        array_shape: Vec<TDim>,
        data_type: DataType,
        fill_value: impl Into<FillValue>,
    ) -> Result<Self, DataTypeFillValueError> {
        let fill_value = fill_value.into();
        match data_type.size() {
            DataTypeSize::Fixed(size) => {
                // Allow null fill values ONLY for optional data types
                if (fill_value.is_null() && data_type.is_optional()) || size == fill_value.size() {
                    Ok(Self {
                        array_shape,
                        data_type,
                        fill_value,
                    })
                } else {
                    Err(DataTypeFillValueError::new(data_type.name(), fill_value))
                }
            }
            DataTypeSize::Variable => Ok(Self {
                array_shape,
                data_type,
                fill_value,
            }),
        }
    }

    /// Create a new [`ArrayRepresentation`].
    ///
    /// # Safety
    /// `data_type` and `fill_value` must be compatible.
    #[must_use]
    pub unsafe fn new_unchecked(
        array_shape: Vec<TDim>,
        data_type: DataType,
        fill_value: impl Into<FillValue>,
    ) -> Self {
        let fill_value = fill_value.into();
        if let Some(data_type_size) = data_type.fixed_size() {
            // Allow null fill values ONLY for optional data types
            debug_assert!(
                (fill_value.is_null() && data_type.is_optional())
                    || data_type_size == fill_value.size(),
                "data type size {} does not match fill value size {}",
                data_type_size,
                fill_value.size()
            );
        }
        Self {
            array_shape,
            data_type,
            fill_value,
        }
    }

    /// Return the shape of the array.
    #[must_use]
    pub fn shape(&self) -> &[TDim] {
        &self.array_shape
    }

    /// Return the dimensionality of the array.
    #[must_use]
    pub fn dimensionality(&self) -> usize {
        self.array_shape.len()
    }

    /// Return the shape as an [`ArrayShape`] ([`Vec<u64>`]).
    #[must_use]
    pub fn shape_u64(&self) -> ArrayShape {
        self.array_shape
            .iter()
            .map(|&i| i.into())
            .collect::<Vec<u64>>()
    }

    /// Return the data type of the array.
    #[must_use]
    pub const fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Return the fill value of the array.
    #[must_use]
    pub const fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    /// Return the number of elements in the array.
    ///
    /// Equal to the product of its shape.
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.array_shape.iter().map(|&i| i.into()).product::<u64>()
    }

    /// Return the number of elements of the array as a `usize`.
    ///
    /// # Panics
    ///
    /// Panics if [`num_elements()`](Self::num_elements()) is greater than [`usize::MAX`].
    #[must_use]
    pub fn num_elements_usize(&self) -> usize {
        usize::try_from(self.num_elements()).unwrap()
    }

    /// Return the element size.
    #[must_use]
    pub fn element_size(&self) -> DataTypeSize {
        self.data_type().size()
    }

    /// Returns the element size in bytes with a fixed-size data type, otherwise returns [`None`].
    #[must_use]
    pub fn fixed_element_size(&self) -> Option<usize> {
        self.data_type().fixed_size()
    }

    /// Return the array size in bytes with a fixed-size data type, otherwise returns [`None`].
    ///
    /// # Panics
    /// Panics if the size does not fit in [`usize::MAX`].
    #[must_use]
    pub fn fixed_size(&self) -> Option<usize> {
        let num_elements = self.num_elements();
        match self.element_size() {
            DataTypeSize::Fixed(data_type_size) => {
                Some(usize::try_from(num_elements * data_type_size as u64).unwrap())
            }
            DataTypeSize::Variable => None,
        }
    }
}
