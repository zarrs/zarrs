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
