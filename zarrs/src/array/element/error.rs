//! Element error types.

use zarrs_codec::CodecError;

/// An element error.
#[derive(Clone, Debug, thiserror::Error)]
pub enum ElementError {
    /// Incompatible element type for data type.
    #[error("Incompatible element type for data type")]
    IncompatibleElementType,
    /// Invalid element value.
    #[error("Invalid element value")]
    InvalidElementValue,
    /// Codec error.
    #[error(transparent)]
    CodecError(#[from] CodecError),
    /// Other error.
    #[error("{0}")]
    Other(String),
}
