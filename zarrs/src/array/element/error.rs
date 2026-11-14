//! Element error types.

use zarrs_codec::{
    ExpectedFixedLengthBytesError, ExpectedOptionalBytesError, ExpectedVariableLengthBytesError,
};

/// An element error.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ElementError {
    /// Incompatible element type for data type.
    #[error("Incompatible element type for data type")]
    IncompatibleElementType,
    /// Invalid element value.
    #[error("Invalid element value")]
    InvalidElementValue,
    /// The shape of elements with a layout (e.g. an `ndarray` array or a `Tensor`) differs from the shape of the target.
    #[error(
        "the shape {actual:?} of the elements does not match the shape {expected:?} of the target"
    )]
    IncompatibleShape {
        /// The shape of the target.
        expected: Vec<u64>,
        /// The shape of the elements.
        actual: Vec<u64>,
    },
    /// Expected fixed length bytes.
    #[error(transparent)]
    ExpectedFixedLengthBytes(#[from] ExpectedFixedLengthBytesError),
    /// Expected variable length bytes.
    #[error(transparent)]
    ExpectedVariableLengthBytes(#[from] ExpectedVariableLengthBytesError),
    /// Expected optional bytes.
    #[error(transparent)]
    ExpectedOptionalBytes(#[from] ExpectedOptionalBytesError),
    /// Other error.
    #[error("{0}")]
    Other(String),
}
