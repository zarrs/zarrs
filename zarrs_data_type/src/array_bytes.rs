//! Array bytes types for representing array data in byte form.
//!
//! This module provides the core types for representing array element data as bytes:
//! - [`ArrayBytes`]: The main enum for fixed, variable, or optional array bytes
//! - [`ArrayBytesRaw`]: Raw byte data (type alias for `Cow<'a, [u8]>`)
//! - [`ArrayBytesOffsets`]: Monotonically increasing offsets for variable-length data
//! - [`ArrayBytesVariableLength`]: Variable-length bytes with offsets
//! - [`ArrayBytesOptional`]: Array bytes with a validity mask

#[allow(clippy::module_inception)]
mod array_bytes;
mod array_bytes_offsets;
mod array_bytes_optional;
mod array_bytes_raw;
mod array_bytes_variable_length;
mod error;

pub use array_bytes::ArrayBytes;
pub use array_bytes_offsets::{ArrayBytesOffsets, ArrayBytesRawOffsetsCreateError};
pub use array_bytes_optional::ArrayBytesOptional;
pub use array_bytes_raw::ArrayBytesRaw;
pub use array_bytes_variable_length::ArrayBytesVariableLength;
pub use error::{
    ArrayBytesError, ArrayBytesRawOffsetsOutOfBoundsError, ArrayBytesValidateError,
    ExpectedFixedLengthBytesError, ExpectedOptionalBytesError, ExpectedVariableLengthBytesError,
    InvalidBytesLengthError,
};
