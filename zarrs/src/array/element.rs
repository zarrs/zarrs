use super::{ArrayBytes, DataType};

mod bool;
mod bytes;
mod error;
mod numpy;
mod optional;
mod pod;
mod raw_bits;
mod string;

pub use error::ElementError;

/// A trait representing an array element type.
pub trait Element: Sized + Clone {
    /// Validate the data type.
    ///
    /// # Errors
    /// Returns an [`ElementError`] if the data type is incompatible with [`Element`].
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError>;

    /// Convert a slice of elements into [`ArrayBytes`].
    ///
    /// # Errors
    /// Returns an [`ElementError`] if the data type is incompatible with [`Element`].
    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError>;

    /// Convert a vector of elements into [`ArrayBytes`].
    ///
    /// Avoids an extra copy compared to `to_array_bytes` when possible.
    ///
    /// # Errors
    /// Returns an [`ElementError`] if the data type is incompatible with [`Element`].
    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError>;
}

/// A trait representing an owned array element type.
pub trait ElementOwned: Element {
    /// Convert bytes into a [`Vec<ElementOwned>`].
    ///
    /// # Errors
    /// Returns an [`ElementError`] if the data type is incompatible with [`Element`].
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError>;
}
