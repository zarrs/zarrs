use derive_more::derive::Display;
use thiserror::Error;

use super::ArrayBytesRawOffsetsCreateError;

/// An error indicating the length of bytes does not match the expected length.
#[derive(Clone, Debug, Display, Error)]
#[display("Invalid bytes len {len}, expected {expected_len}")]
pub struct InvalidBytesLengthError {
    /// The actual length.
    pub len: usize,
    /// The expected length.
    pub expected_len: usize,
}

impl InvalidBytesLengthError {
    /// Create a new [`InvalidBytesLengthError`].
    #[must_use]
    pub fn new(len: usize, expected_len: usize) -> Self {
        Self { len, expected_len }
    }
}

/// An error raised if variable length array bytes offsets are out of bounds.
#[derive(Clone, Debug, Display, Error)]
#[display("Offset {offset} is out of bounds for bytes of length {len}")]
pub struct ArrayBytesRawOffsetsOutOfBoundsError {
    offset: usize,
    len: usize,
}

impl ArrayBytesRawOffsetsOutOfBoundsError {
    /// Create a new [`ArrayBytesRawOffsetsOutOfBoundsError`].
    #[must_use]
    pub fn new(offset: usize, len: usize) -> Self {
        Self { offset, len }
    }
}

/// Expected fixed length array bytes but found variable or optional.
#[derive(Clone, Copy, Debug, Display, Error)]
#[display("Expected fixed length array bytes")]
pub struct ExpectedFixedLengthBytesError;

/// Expected variable length array bytes but found fixed or optional.
#[derive(Clone, Copy, Debug, Display, Error)]
#[display("Expected variable length array bytes")]
pub struct ExpectedVariableLengthBytesError;

/// Expected optional array bytes but found fixed or variable.
#[derive(Clone, Copy, Debug, Display, Error)]
#[display("Expected optional array bytes")]
pub struct ExpectedOptionalBytesError;

/// Errors related to [`ArrayBytes`](super::ArrayBytes).
#[derive(Clone, Debug, Error)]
pub enum ArrayBytesError {
    /// Expected fixed length bytes.
    #[error("Expected fixed length array bytes")]
    ExpectedFixedLengthBytes,
    /// Expected variable length bytes.
    #[error("Expected variable length array bytes")]
    ExpectedVariableLengthBytes,
    /// Expected optional bytes.
    #[error("Expected optional array bytes")]
    ExpectedOptionalBytesError,
    /// Offsets creation error.
    #[error(transparent)]
    RawBytesOffsetsCreate(#[from] ArrayBytesRawOffsetsCreateError),
    /// Offsets out of bounds.
    #[error(transparent)]
    RawBytesOffsetsOutOfBounds(#[from] ArrayBytesRawOffsetsOutOfBoundsError),
}

impl From<ExpectedFixedLengthBytesError> for ArrayBytesError {
    fn from(_: ExpectedFixedLengthBytesError) -> Self {
        Self::ExpectedFixedLengthBytes
    }
}

impl From<ExpectedVariableLengthBytesError> for ArrayBytesError {
    fn from(_: ExpectedVariableLengthBytesError) -> Self {
        Self::ExpectedVariableLengthBytes
    }
}

impl From<ExpectedOptionalBytesError> for ArrayBytesError {
    fn from(_: ExpectedOptionalBytesError) -> Self {
        Self::ExpectedOptionalBytesError
    }
}

/// An error that can occur when validating [`ArrayBytes`](super::ArrayBytes).
#[derive(Clone, Debug, Error)]
pub enum ArrayBytesValidateError {
    /// The bytes length does not match the expected length.
    #[error(transparent)]
    InvalidBytesLength(#[from] InvalidBytesLengthError),
    /// The variable sized array offsets are invalid.
    #[error("Invalid variable sized array offsets")]
    InvalidVariableSizedArrayOffsets,
    /// Used non-optional array bytes with an optional data type.
    #[error("Used non-optional array bytes with an optional data type")]
    ExpectedOptionalBytes,
    /// Used optional array bytes with a non-optional data type.
    #[error("Used optional array bytes with a non-optional data type")]
    UnexpectedOptionalBytes,
    /// Used fixed length array bytes with a variable sized data type.
    #[error("Used fixed length array bytes with a variable sized data type")]
    ExpectedVariableLengthBytes,
    /// Used variable length array bytes with a fixed length data type.
    #[error("Used variable length array bytes with a fixed length data type")]
    ExpectedFixedLengthBytes,
}
