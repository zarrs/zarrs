use derive_more::Display;
use thiserror::Error;

use crate::array::CowBytes;

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
#[derive(Clone, Debug)]
pub struct Tensor<'a> {
    bytes: CowBytes<'a>,
    data_type: DataType,
    shape: Vec<u64>,
}

impl<'a> Tensor<'a> {
    /// Create a new [`Tensor`].
    #[must_use]
    pub fn new(bytes: impl Into<CowBytes<'a>>, data_type: DataType, shape: Vec<u64>) -> Self {
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
        if self.shape.contains(&0) {
            return 0;
        }
        self.shape.iter().product()
    }

    /// Convert into a `Tensor<'static>`, copying only if the bytes are borrowed.
    #[must_use]
    pub fn into_static(self) -> Tensor<'static> {
        Tensor {
            bytes: self.bytes.into_static(),
            data_type: self.data_type,
            shape: self.shape,
        }
    }

    /// Consume self and return the parts.
    #[must_use]
    pub fn into_parts(self) -> (CowBytes<'a>, DataType, Vec<u64>) {
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

#[cfg(test)]
mod tests {
    use zarrs_storage::store::MemoryStore;

    use crate::array::{ArrayBuilder, Tensor, data_type};

    #[test]
    fn tensor_borrowed_bytes_are_not_copied() {
        let bytes = vec![0u8; 4 * size_of::<f32>()];
        let tensor = Tensor::new(&bytes[..], data_type::float32(), vec![2, 2]);
        assert_eq!(tensor.bytes().as_ptr(), bytes.as_ptr());
    }

    #[test]
    fn tensor_borrowed_store_chunk_roundtrip() {
        let elements = [0.0f32, 1.0, 2.0, 3.0];
        let bytes: Vec<u8> = elements.iter().flat_map(|f| f.to_ne_bytes()).collect();

        let store = MemoryStore::new();
        let array = ArrayBuilder::new(vec![2, 2], vec![2, 2], data_type::float32(), -1.0f32)
            .build(store.into(), "/")
            .unwrap();

        let tensor = Tensor::new(&bytes[..], data_type::float32(), vec![2, 2]);
        array.store_chunk(&[0, 0], tensor).unwrap();

        let retrieved: Vec<f32> = array.retrieve_chunk(&[0, 0]).unwrap();
        assert_eq!(retrieved, elements);
    }

    #[test]
    fn tensor_into_static_borrowed_copies() {
        let bytes: Vec<u8> = (1u8..=4).collect();
        let tensor = Tensor::new(&bytes[..], data_type::uint8(), vec![4]);
        let tensor = tensor.into_static();
        assert_ne!(tensor.bytes().as_ptr(), bytes.as_ptr());
        assert_eq!(tensor.bytes(), &[1, 2, 3, 4]);
    }
}
