//! Encoded subchunks.

use std::num::NonZeroU64;

use zarrs_metadata::ChunkShape;
use zarrs_storage::Bytes;

/// The encoded bytes of a subchunk and the shape to decode them with.
///
/// Decode the [`bytes`](Self::bytes) with the subchunk codec at the level the subchunk was retrieved from
/// ([`subchunk_codecs_at_level`](crate::ArrayToBytesCodecSubchunkingTraits::subchunk_codecs_at_level)), passing
/// [`shape`](Self::shape). That codec is a property of the array rather than of an individual subchunk, whereas the
/// shape is resolved per subchunk.
///
/// The bytes are in the *encoded domain* of the codec that created the subchunk, which is not necessarily the domain
/// of the array they were retrieved from. If the array has array-to-array codecs, then decoding yields array bytes
/// that those codecs have yet to decode:
/// - the values are in the data type of the subchunk codec, which may differ from the data type of the array (e.g.
///   with a `cast_value` codec), and
/// - the elements are ordered by [`shape`](Self::shape), which may differ in extent and even in dimensionality from
///   the shape of the subchunk in the subchunk grid of the array (e.g. with a `transpose`, `reshape`, or `squeeze`
///   codec).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedSubchunk {
    bytes: Bytes,
    shape: ChunkShape,
}

impl EncodedSubchunk {
    /// Create a new [`EncodedSubchunk`].
    ///
    /// The `shape` must be the shape of the subchunk in the encoded domain of the codec that created it.
    #[must_use]
    pub fn new(bytes: Bytes, shape: ChunkShape) -> Self {
        Self { bytes, shape }
    }

    /// Return the encoded bytes of the subchunk.
    ///
    /// These bytes are suitable for passthrough without decoding, such as copying a subchunk into another array with
    /// matching subchunk codecs.
    #[must_use]
    pub fn bytes(&self) -> &Bytes {
        &self.bytes
    }

    /// Return the shape to decode the [`bytes`](Self::bytes) with.
    ///
    /// This is the shape of the subchunk in the encoded domain of its subchunk codec, which may differ in extent and
    /// dimensionality from the shape of the subchunk in the subchunk grid it was addressed by.
    #[must_use]
    pub fn shape(&self) -> &[NonZeroU64] {
        &self.shape
    }

    /// Consume the [`EncodedSubchunk`] and return its bytes.
    #[must_use]
    pub fn into_bytes(self) -> Bytes {
        self.bytes
    }

    /// Consume the [`EncodedSubchunk`] and return its bytes and shape.
    #[must_use]
    pub fn into_parts(self) -> (Bytes, ChunkShape) {
        (self.bytes, self.shape)
    }
}
