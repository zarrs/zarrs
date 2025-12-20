use std::borrow::Cow;

use derive_more::Display;
use thiserror::Error;

use crate::array::ArrayBytesRaw;

use super::DataType;

/// Errors related to [`Tensor`] operations.
#[derive(Clone, Debug, Display, Error)]
#[non_exhaustive]
pub enum TensorError {
    /// The data type is not supported.
    #[display("Data type {_0:?} is not supported for this operation.")]
    UnsupportedDataType(DataType),
}

/// A tensor holding raw bytes with data type and shape metadata.
///
/// This represents a multidimensional array of fixed-size elements in C-contiguous (row-major) order.
pub struct Tensor {
    bytes: ArrayBytesRaw<'static>,
    data_type: DataType,
    shape: Vec<u64>,
}

impl Tensor {
    /// Create a new [`Tensor`].
    #[must_use]
    pub fn new(
        bytes: impl Into<ArrayBytesRaw<'static>>,
        data_type: DataType,
        shape: Vec<u64>,
    ) -> Self {
        Self {
            bytes: bytes.into(),
            data_type,
            shape,
        }
    }

    /// Get the raw bytes.
    #[must_use]
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Get the data type.
    #[must_use]
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Get the shape.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Consume self and return the parts.
    #[must_use]
    pub fn into_parts(self) -> (Cow<'static, [u8]>, DataType, Vec<u64>) {
        (self.bytes, self.data_type, self.shape)
    }
}
