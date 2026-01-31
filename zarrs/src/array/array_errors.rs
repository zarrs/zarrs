use serde_json::Value;
use thiserror::Error;
use zarrs_codec::{
    CodecError, ExpectedFixedLengthBytesError, ExpectedOptionalBytesError,
    ExpectedVariableLengthBytesError,
};
use zarrs_data_type::FillValue;
use zarrs_metadata::FillValueMetadata;

use super::{ArrayBytesFixedDisjointViewCreateError, ArrayIndices, ArrayShape};
use crate::array::{ArraySubset, ArraySubsetError, IncompatibleDimensionalityError};
use crate::node::NodePathError;
use zarrs_plugin::PluginCreateError;
use zarrs_storage::StorageError;

/// An array creation error.
#[derive(Clone, Debug, Error)]
pub enum ArrayCreateError {
    /// An invalid node path
    #[error(transparent)]
    NodePathError(#[from] NodePathError),
    /// Unsupported additional field.
    #[error(transparent)]
    AdditionalFieldUnsupportedError(#[from] AdditionalFieldUnsupportedError),
    /// Unsupported data type.
    #[error(transparent)]
    DataTypeCreateError(PluginCreateError),
    /// Invalid fill value.
    #[error("invalid fill value for data type `{data_type_name}`: {fill_value}")]
    InvalidFillValue {
        /// The data type name.
        data_type_name: String,
        /// The fill value.
        fill_value: FillValue,
    },
    // /// Unparseable metadata.
    // #[error("unparseable metadata: {_0:?}")]
    // UnparseableMetadata(String),
    // /// Invalid data type metadata.
    // #[error("unsupported data type metadata: {_0:?}")]
    // UnsupportedDataTypeMetadata(MetadataV3),
    // /// Invalid chunk grid metadata.
    // #[error("unsupported chunk grid metadata: {_0:?}")]
    // UnsupportedChunkGridMetadata(MetadataV3),
    /// Invalid fill value metadata.
    #[error("invalid fill value metadata for data type `{data_type_name}`: {fill_value_metadata}")]
    InvalidFillValueMetadata {
        /// The data type name.
        data_type_name: String,
        /// The fill value metadata.
        fill_value_metadata: FillValueMetadata,
    },
    /// Error creating codecs.
    #[error(transparent)]
    CodecsCreateError(PluginCreateError),
    /// Storage transformer creation error.
    #[error(transparent)]
    StorageTransformersCreateError(PluginCreateError),
    /// Chunk grid create error.
    #[error(transparent)]
    ChunkGridCreateError(PluginCreateError),
    /// Chunk key encoding create error.
    #[error(transparent)]
    ChunkKeyEncodingCreateError(PluginCreateError),
    /// The dimensionality of the chunk grid does not match the array shape.
    #[error("chunk grid dimensionality {0} does not match array dimensionality {1}")]
    InvalidChunkGridDimensionality(usize, usize),
    /// The number of dimension names does not match the array dimensionality.
    #[error("the number of dimension names {0} does not match array dimensionality {1}")]
    InvalidDimensionNames(usize, usize),
    /// Invalid subchunk shape (contains zero).
    #[error("invalid subchunk shape {0:?}: all elements must be non-zero")]
    InvalidSubchunkShape(super::ArrayShape),
    /// Storage error.
    #[error(transparent)]
    StorageError(#[from] StorageError),
    /// Missing metadata.
    #[error("array metadata is missing")]
    MissingMetadata,
    /// The Zarr V2 array is unsupported.
    #[error("unsupported Zarr V2 array: {_0}")]
    UnsupportedZarrV2Array(String),
}

/// Array errors.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum ArrayError {
    /// Error when a disjoint view creation cannot be done
    #[error(transparent)]
    ArrayBytesFixedDisjointViewCreateError(#[from] ArrayBytesFixedDisjointViewCreateError),
    /// A store error.
    #[error(transparent)]
    StorageError(#[from] StorageError),
    /// A codec error.
    #[error(transparent)]
    CodecError(#[from] CodecError),
    // /// Invalid array indices.
    // #[error(transparent)]
    // InvalidArrayIndicesError(#[from] InvalidArrayIndicesError),
    /// Invalid chunk grid indices.
    #[error("invalid chunk grid indices: {_0:?}")]
    InvalidChunkGridIndicesError(Vec<u64>),
    /// Incompatible dimensionality.
    #[error(transparent)]
    IncompatibleDimensionalityError(#[from] IncompatibleDimensionalityError),
    /// An [`ArraySubsetError`].
    #[error(transparent)]
    ArraySubsetError(#[from] ArraySubsetError),
    /// Incompatible array subset.
    #[error("array subset {_0} is not compatible with array shape {_1:?}")]
    InvalidArraySubset(ArraySubset, ArrayShape),
    /// Incompatible chunk subset.
    #[error("chunk subset {_0} is not compatible with chunk {_1:?} with shape {_2:?}")]
    InvalidChunkSubset(ArraySubset, ArrayIndices, ArrayShape),
    /// An unexpected chunk decoded size.
    #[error("got chunk decoded size {_0:?}, expected {_1:?}")]
    UnexpectedChunkDecodedSize(usize, usize),
    /// An unexpected bytes input size.
    #[error("got bytes with size {_0:?}, expected {_1:?}")]
    InvalidBytesInputSize(usize, u64),
    /// An unexpected chunk decoded shape.
    #[error("got chunk decoded shape {_0:?}, expected {_1:?}")]
    UnexpectedChunkDecodedShape(ArrayShape, ArrayShape),
    /// Incompatible element size.
    #[error("the element types does not match the data type")]
    IncompatibleElementType,
    /// Invalid data shape.
    #[error("data has shape {_0:?}, expected {_1:?}")]
    InvalidDataShape(Vec<usize>, Vec<usize>),
    /// Invalid element value.
    ///
    /// For example
    ///  - a bool array with a value not equal to 0 (false) or 1 (true).
    ///  - a string with invalid utf-8 encoding.
    #[error("Invalid element value")]
    InvalidElementValue, // TODO: Add reason
    /// Unsupported method.
    #[error("unsupported array method: {_0}")]
    UnsupportedMethod(String),
    #[cfg(feature = "dlpack")]
    /// A [`TensorError`](super::TensorError).
    #[error(transparent)]
    TensorError(#[from] super::TensorError),
    /// Any other error.
    #[error("{_0}")]
    Other(String),
}

impl From<ExpectedFixedLengthBytesError> for ArrayError {
    fn from(err: ExpectedFixedLengthBytesError) -> Self {
        Self::CodecError(err.into())
    }
}

impl From<ExpectedVariableLengthBytesError> for ArrayError {
    fn from(err: ExpectedVariableLengthBytesError) -> Self {
        Self::CodecError(err.into())
    }
}

impl From<ExpectedOptionalBytesError> for ArrayError {
    fn from(err: ExpectedOptionalBytesError) -> Self {
        Self::CodecError(err.into())
    }
}

/// An unsupported additional field error.
///
/// An unsupported field in array or group metadata is an unrecognised field without `"must_understand": false`.
#[derive(Clone, Debug, Error)]
#[error("unsupported additional field {name} with value {value}")]
pub struct AdditionalFieldUnsupportedError {
    name: String,
    value: Value,
}

impl AdditionalFieldUnsupportedError {
    /// Create a new [`AdditionalFieldUnsupportedError`].
    #[must_use]
    pub fn new(name: String, value: Value) -> AdditionalFieldUnsupportedError {
        Self { name, value }
    }

    /// Return the name of the unsupported additional field.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the value of the unsupported additional field.
    #[must_use]
    pub const fn value(&self) -> &Value {
        &self.value
    }
}
