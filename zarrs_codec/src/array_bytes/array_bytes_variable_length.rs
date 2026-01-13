use crate::{ArrayBytesOffsets, ArrayBytesRaw, ArrayRawBytesOffsetsOutOfBoundsError};

/// Variable length array bytes composed of bytes and element bytes offsets.
///
/// The bytes and offsets are modeled on the [Apache Arrow Variable-size Binary Layout](https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout).
/// - The offsets buffer contains length + 1 ~~signed integers (either 32-bit or 64-bit, depending on the data type)~~ usize integers.
/// - Offsets must be monotonically increasing, that is `offsets[j+1] >= offsets[j]` for `0 <= j < length`, even for null slots. Thus, the bytes represent C-contiguous elements with padding permitted.
/// - The final offset must be less than or equal to the length of the bytes buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArrayBytesVariableLength<'a> {
    pub(crate) bytes: ArrayBytesRaw<'a>,
    pub(crate) offsets: ArrayBytesOffsets<'a>,
}

impl<'a> ArrayBytesVariableLength<'a> {
    /// Create a new variable length bytes from `bytes` and `offsets`.
    ///
    /// # Errors
    /// Returns a [`ArrayRawBytesOffsetsOutOfBoundsError`] if the last offset is out of bounds of the bytes or if the offsets are not monotonically increasing.
    pub fn new(
        bytes: impl Into<ArrayBytesRaw<'a>>,
        offsets: ArrayBytesOffsets<'a>,
    ) -> Result<Self, ArrayRawBytesOffsetsOutOfBoundsError> {
        let bytes = bytes.into();
        if offsets.last() <= bytes.len() {
            Ok(ArrayBytesVariableLength { bytes, offsets })
        } else {
            Err(ArrayRawBytesOffsetsOutOfBoundsError {
                offset: offsets.last(),
                len: bytes.len(),
            })
        }
    }

    /// Create a new variable length bytes from `bytes` and `offsets`.
    ///
    /// # Safety
    /// The last offset must be less than or equal to the length of the bytes.
    pub unsafe fn new_unchecked(
        bytes: impl Into<ArrayBytesRaw<'a>>,
        offsets: ArrayBytesOffsets<'a>,
    ) -> Self {
        let bytes = bytes.into();
        debug_assert!(offsets.last() <= bytes.len());
        Self { bytes, offsets }
    }

    /// Get the underlying bytes.
    #[must_use]
    pub fn bytes(&self) -> &ArrayBytesRaw<'_> {
        &self.bytes
    }

    /// Get the underlying offsets.
    #[must_use]
    pub fn offsets(&self) -> &ArrayBytesOffsets<'_> {
        &self.offsets
    }

    /// Consume self and return the bytes and offsets.
    #[must_use]
    pub fn into_parts(self) -> (ArrayBytesRaw<'a>, ArrayBytesOffsets<'a>) {
        (self.bytes, self.offsets)
    }
}
