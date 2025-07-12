use serde_json::Value;
use thiserror::Error;

use crate::{
    array::data_type::{DataTypeFillValueError, DataTypeFillValueMetadataError},
    array_subset::{
        ArraySubset, IncompatibleDimensionalityError, IncompatibleOffsetError,
        IncompatibleStartEndIndicesError,
    },
    node::NodePathError,
    plugin::PluginCreateError,
    storage::StorageError,
};

use super::{codec::CodecError, ArrayBytesFixedDisjointViewCreateError, ArrayIndices, ArrayShape};

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
    #[error(transparent)]
    InvalidFillValue(#[from] DataTypeFillValueError),
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
    #[error(transparent)]
    InvalidFillValueMetadata(#[from] DataTypeFillValueMetadataError),
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
    /// Incompatible start and end indices.
    #[error(transparent)]
    IncompatibleStartEndIndicesError(#[from] IncompatibleStartEndIndicesError),
    /// An incompatible offset.
    #[error(transparent)]
    IncompatibleOffset(#[from] IncompatibleOffsetError),
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
    /// A `DLPack` error
    #[error(transparent)]
    DlPackError(#[from] super::array_dlpack_ext::ArrayDlPackExtError),
    /// Any other error.
    #[error("{_0}")]
    Other(String),
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
