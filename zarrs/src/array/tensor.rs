use std::borrow::Cow;

use derive_more::Display;
use thiserror::Error;
use zarrs_codec::{ElementLayout, ElementPacking};
use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;

use crate::array::ArrayBytesRaw;
use crate::array::codec::array_to_bytes::packbits::{pack_bits, pack_bits_components};

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
    /// The element layout is not supported.
    #[display("Element layout {_0:?} is not supported for this operation.")]
    UnsupportedLayout(ElementLayout),
    /// The tensor bytes are too short for its shape, data type, and element layout.
    #[display("Tensor needs at least {expected} bytes, but has {actual}.")]
    InsufficientBytes {
        /// The number of bytes required.
        expected: usize,
        /// The number of bytes in the tensor.
        actual: usize,
    },
}

/// A tensor holding raw bytes with data type, shape, and element layout metadata.
///
/// This represents a multidimensional array of fixed-size elements in C-contiguous (row-major) order.
///
/// # Element layout
/// The [`layout`](Self::layout) determines how the [`bytes`](Self::bytes) encode the elements.
///
/// A tensor from a `retrieve_*` method has the default layout, the layout of decoded array bytes:
/// the first element starts at offset zero, each element occupies [`DataType::fixed_size`] bytes in
/// native endianness, and there is no padding between elements. A sub-byte data type (`bool`,
/// `int2`, `int4`, `uint2`, `uint4`, `float4_e2m1fn`, `float6_e2m3fn`, or `float6_e3m2fn`) is
/// therefore stored one element per byte, with the value sign or zero extended into the padding
/// bits.
///
/// Use [`into_packed`](Self::into_packed) to bit-pack the elements instead, which is how the
/// `packbits` codec stores them, or
/// [`ArrayReadOps::retrieve_chunk_stored_layout`](crate::array::ArrayReadOps::retrieve_chunk_stored_layout)
/// to read a chunk in the layout it is stored in.
pub struct Tensor {
    bytes: ArrayBytesRaw<'static>,
    data_type: DataType,
    shape: Vec<u64>,
    layout: ElementLayout,
}

impl Tensor {
    /// Create a new [`Tensor`] with the default [`ElementLayout`].
    ///
    /// Each element occupies [`DataType::fixed_size`] bytes, in native endianness.
    #[must_use]
    pub fn new(
        bytes: impl Into<ArrayBytesRaw<'static>>,
        data_type: DataType,
        shape: Vec<u64>,
    ) -> Self {
        Self::new_with_layout(bytes, data_type, shape, ElementLayout::default())
    }

    /// Create a new [`Tensor`] with an explicit [`ElementLayout`].
    #[must_use]
    pub fn new_with_layout(
        bytes: impl Into<ArrayBytesRaw<'static>>,
        data_type: DataType,
        shape: Vec<u64>,
        layout: ElementLayout,
    ) -> Self {
        Self {
            bytes: bytes.into(),
            data_type,
            shape,
            layout,
        }
    }

    /// Get the raw bytes.
    ///
    /// These are interpreted according to the [`layout`](Self::layout).
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

    /// Get the element layout.
    #[must_use]
    pub fn layout(&self) -> ElementLayout {
        self.layout
    }

    /// Get the number of elements.
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Bit-pack the tensor elements, least significant bit first.
    ///
    /// The resulting tensor has [`ElementPacking::PackedLsb0`], where an element of `N` bits
    /// occupies exactly `N` bits rather than `ceil(N / 8)` bytes. For a data type whose elements
    /// are already a whole number of bytes this is a relabelling and does not copy.
    ///
    /// This is the packing performed by the `packbits` codec.
    ///
    /// # Errors
    /// Returns [`TensorError::UnsupportedDataType`] if the data type does not have a fixed size or
    /// does not support the `packbits` codec, or [`TensorError::UnsupportedLayout`] if the tensor
    /// is not in the default layout.
    pub fn into_packed(self) -> Result<Self, TensorError> {
        if self.layout.packing == ElementPacking::PackedLsb0 {
            return Ok(self);
        }
        if self.layout != ElementLayout::default() {
            return Err(TensorError::UnsupportedLayout(self.layout));
        }

        let components = pack_bits_components(&self.data_type)
            .map_err(|_| TensorError::UnsupportedDataType(self.data_type.clone()))?;
        let packed_layout = ElementLayout {
            packing: ElementPacking::PackedLsb0,
            ..ElementLayout::default()
        };

        // A whole-byte element is packed identically to how it is padded, so avoid the copy
        if components.component_size_bits.is_multiple_of(8) {
            return Ok(Self {
                layout: packed_layout,
                ..self
            });
        }

        let bytes = pack_bits(
            &self.bytes,
            self.num_elements(),
            components,
            0,
            components.component_size_bits - 1,
            PackBitsPaddingEncoding::None,
        );
        Ok(Self {
            bytes: bytes.into(),
            layout: packed_layout,
            ..self
        })
    }

    /// Consume self and return the parts.
    #[must_use]
    pub fn into_parts(self) -> (Cow<'static, [u8]>, DataType, Vec<u64>, ElementLayout) {
        (self.bytes, self.data_type, self.shape, self.layout)
    }

    /// Get references to all parts of the tensor.
    ///
    /// Returns `(bytes, data_type, shape, layout)`.
    #[must_use]
    pub fn as_parts(&self) -> (&[u8], &DataType, &[u64], ElementLayout) {
        (&self.bytes, &self.data_type, &self.shape, self.layout)
    }
}

#[cfg(test)]
mod tests {
    use super::{ElementLayout, ElementPacking, Tensor, TensorError};
    use crate::array::data_type;

    #[test]
    fn tensor_into_packed_sub_byte() {
        // int4 elements -1, 7, -8, 1 are stored sign extended, one per byte
        let tensor = Tensor::new(vec![0xFFu8, 0x07, 0xF8, 0x01], data_type::int4(), vec![4]);
        assert_eq!(tensor.layout(), ElementLayout::default());

        let packed = tensor.into_packed().unwrap();
        assert_eq!(packed.layout().packing, ElementPacking::PackedLsb0);
        // Two 4-bit elements per byte, least significant nibble first
        assert_eq!(packed.bytes(), &[0x7F, 0x18]);
        // The shape and data type are unchanged
        assert_eq!(packed.shape(), &[4]);
        assert_eq!(packed.data_type(), &data_type::int4());
    }

    #[test]
    fn tensor_into_packed_byte_aligned_is_a_relabel() {
        let bytes = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let tensor = Tensor::new(bytes.clone(), data_type::uint16(), vec![4]);
        let packed = tensor.into_packed().unwrap();

        // A whole-byte element packs to exactly the same bytes
        assert_eq!(packed.bytes(), bytes);
        assert_eq!(packed.layout().packing, ElementPacking::PackedLsb0);
    }

    #[test]
    fn tensor_into_packed_is_idempotent() {
        let tensor = Tensor::new(vec![0x0Fu8, 0x01], data_type::uint4(), vec![2]);
        let packed = tensor.into_packed().unwrap();
        let bytes = packed.bytes().to_vec();
        let packed_again = packed.into_packed().unwrap();
        assert_eq!(packed_again.bytes(), bytes);
    }

    #[test]
    fn tensor_into_packed_unsupported() {
        // A variable sized data type has no packbits support
        let tensor = Tensor::new(vec![0u8; 4], data_type::string(), vec![1]);
        assert!(matches!(
            tensor.into_packed(),
            Err(TensorError::UnsupportedDataType(_))
        ));

        // A non-default layout is not converted
        let layout = ElementLayout {
            byte_offset: 1,
            ..ElementLayout::default()
        };
        let tensor = Tensor::new_with_layout(vec![0u8; 5], data_type::int4(), vec![4], layout);
        assert!(matches!(
            tensor.into_packed(),
            Err(TensorError::UnsupportedLayout(_))
        ));
    }
}
