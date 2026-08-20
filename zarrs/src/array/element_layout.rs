//! Internal element layout scaffolding.
//!
//! Nothing in `zarrs` produces a non-default layout yet, so every [`Tensor`] has
//! [`ElementLayout::default()`]. Support for bit-packed tensors will promote these types (to
//! `zarrs_codec`, alongside a codec trait method that declares the element layout of a codec's
//! encoded output) and add the operations that produce them.
#![allow(
    dead_code,
    reason = "the packed variant and the layout accessor are only exercised by tests until zarrs produces packed tensors"
)]

use zarrs_metadata::Endianness;

use super::Tensor;

/// How the elements of a data type are packed within a buffer.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub(crate) enum ElementPacking {
    /// Each element occupies a whole number of bytes.
    ///
    /// An element of `N` bits occupies `ceil(N / 8)` bytes. This is how `zarrs` represents
    /// elements in memory, so sub-byte elements have padding bits.
    #[default]
    Padded,
    /// Elements are bit-contiguous, least significant bit first.
    ///
    /// An element of `N` bits occupies exactly `N` bits and may straddle byte boundaries. Element
    /// `i` is stored in bits `[i * N, (i + 1) * N)` of the buffer, where bit `j` is bit `j % 8` of
    /// byte `j / 8`. This is the packing performed by the `packbits` codec.
    PackedLsb0,
}

/// The layout of elements within a buffer.
///
/// [`ElementLayout::default()`] is the layout of decoded array bytes: [`ElementPacking::Padded`],
/// no byte offset, and native endianness.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) struct ElementLayout {
    /// How elements are packed.
    pub(crate) packing: ElementPacking,
    /// The byte offset of the first element from the start of the buffer.
    pub(crate) byte_offset: u64,
    /// The endianness of multi-byte components.
    pub(crate) endianness: Endianness,
}

impl Default for ElementLayout {
    fn default() -> Self {
        Self {
            packing: ElementPacking::default(),
            byte_offset: 0,
            endianness: Endianness::native(),
        }
    }
}

impl Tensor {
    /// The layout of the elements in [`bytes`](Tensor::bytes).
    ///
    /// Always [`ElementLayout::default()`]; `zarrs` does not produce tensors in another layout yet.
    pub(crate) fn layout(&self) -> ElementLayout {
        ElementLayout::default()
    }
}
