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
    /// The shape is not supported.
    #[display("Shape {_0:?} is not supported for this operation.")]
    UnsupportedShape(Vec<u64>),
    /// The tensor bytes are too short for its shape and data type.
    #[display("Tensor needs at least {expected} bytes, but has {actual}.")]
    InsufficientBytes {
        /// The number of bytes required.
        expected: usize,
        /// The number of bytes in the tensor.
        actual: usize,
    },
}

/// A tensor holding raw bytes with data type and shape metadata.
///
/// This represents a multidimensional array of fixed-size elements in C-contiguous (row-major) order.
///
/// # Element layout
/// The [`bytes`](Self::bytes) hold the elements in the same layout as decoded array bytes: the first
/// element starts at offset zero, each element occupies [`DataType::fixed_size`] bytes in native
/// endianness, and there is no padding between elements.
///
/// A sub-byte data type (`bool`, `int2`, `int4`, `uint2`, `uint4`, `float4_e2m1fn`,
/// `float6_e2m3fn`, or `float6_e3m2fn`) is therefore stored one element per byte, with the value
/// sign or zero extended into the padding bits. This is not how the `packbits` codec stores such
/// elements, which is bit-packed with no padding.
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

    /// Get the number of elements.
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Consume self and return the parts.
    #[must_use]
    pub fn into_parts(self) -> (Cow<'static, [u8]>, DataType, Vec<u64>) {
        (self.bytes, self.data_type, self.shape)
    }

    /// Get references to all parts of the tensor.
    ///
    /// Returns `(bytes, data_type, shape)` as references.
    #[must_use]
    pub fn as_parts(&self) -> (&[u8], &DataType, &[u64]) {
        (&self.bytes, &self.data_type, &self.shape)
    }
}
