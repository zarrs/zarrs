use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::{ArrayIndices, ArraySubset, ChunkGrid, ChunkGridCreateError};
use zarrs_data_type::{DataType, FillValue};
use zarrs_metadata::ChunkShape;

use crate::codec_partial_default::ArrayToArrayCodecPartialDefault;
use crate::{
    ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ChunkGridDecoded, ChunkGridDecodedRef, ChunkGridEncoded, ChunkGridEncodedRef, CodecCreateError,
    CodecError, CodecOptions, CodecSpecificOptions, CodecTraits,
};
#[cfg(feature = "async")]
use crate::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};

/// Subchunking traits for an array-to-array codec bound to a data type and fill value.
///
/// This trait maps subchunk grids, element indices, and subsets between the decoded and
/// encoded representations of an array-to-array codec without encoding or decoding array data:
/// - **Whole-array grid mappings** ([`encoded_chunk_grid`](Self::encoded_chunk_grid),
///   [`decoded_subchunk_grid`](Self::decoded_subchunk_grid)) propagate chunk and subchunk grids
///   across the full array when possible, or fall back to [`ChunkGridDecoded::ChunkLocal`] or
///   [`ChunkGridDecoded::None`].
/// - **Chunk-local mappings** operate on a single chunk of `decoded_shape`:
///   - [`decoded_local_subchunk_grid`](Self::decoded_local_subchunk_grid) and
///     [`decoded_local_subchunk_grids`](Self::decoded_local_subchunk_grids) map subchunk grids of
///     the encoded chunk back to the decoded chunk.
///   - [`encoded_element_indices`](Self::encoded_element_indices) and
///     [`encoded_subset`](Self::encoded_subset) map element indices and
///     exact rectangular subsets from the decoded chunk to the encoded chunk.
///
/// Implement [`ArrayToArrayCodecSubchunkingIdentityTraits`] instead of this trait when the codec
/// preserves chunk shape and element ordering.
pub trait ArrayToArrayCodecSubchunkingTraits: ArrayCodecTraits {
    /// Map a chunk grid from the decoded representation to the encoded representation.
    ///
    /// [`ChunkGridEncoded::ChunkLocal`] indicates that the codec can
    /// propagate a concrete grid for each chunk at runtime, but no global
    /// encoded grid exists. A codec must not map `ChunkLocal` back to `Array`.
    ///
    /// # Errors
    /// Returns a [`ChunkGridCreateError`] if the decoded chunk grid is not supported by this codec.
    fn encoded_chunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridEncoded, ChunkGridCreateError>;

    /// Map a subchunk grid from the encoded representation to the decoded representation.
    ///
    /// Returns [`ChunkGridDecoded::None`] if this codec cannot preserve or transform the subchunk grid.
    ///
    /// # Errors
    /// Returns a [`ChunkGridCreateError`] if the decoded chunk grid or encoded subchunk grid is not supported by this codec.
    fn decoded_subchunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
        encoded_subchunk_grid: ChunkGridEncodedRef<'_>,
    ) -> Result<ChunkGridDecoded, ChunkGridCreateError>;

    /// Map a chunk-local `encoded_subchunk_grid` of the encoded chunk back to a decoded chunk of `decoded_shape`.
    ///
    /// Returns [`None`] if `encoded_subchunk_grid` cannot be represented as a chunk grid in the decoded chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `decoded_shape` is invalid or `encoded_subchunk_grid` is incompatible with the encoded chunk.
    fn decoded_local_subchunk_grid(
        &self,
        decoded_shape: &[NonZeroU64],
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, CodecError>;

    /// Map each level of a chunk-local subchunk grid hierarchy from the encoded chunk back to a decoded chunk of `decoded_shape`.
    ///
    /// Applies [`decoded_local_subchunk_grid`](Self::decoded_local_subchunk_grid) to each resolved level, preserving [`None`] entries.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `decoded_shape` is invalid or any grid is incompatible with the encoded chunk.
    fn decoded_local_subchunk_grids(
        &self,
        decoded_shape: &[NonZeroU64],
        encoded_subchunk_grids: Vec<Option<ChunkGrid>>,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        encoded_subchunk_grids
            .into_iter()
            .map(|grid| {
                grid.map_or(Ok(None), |grid| {
                    self.decoded_local_subchunk_grid(decoded_shape, &grid)
                })
            })
            .collect()
    }

    /// Map element `decoded_element_indices` in a decoded chunk of `decoded_shape` to their indices in the encoded chunk.
    ///
    /// Returns [`None`] if the codec does not map individual elements (e.g. `numcodecs.packbits`).
    /// Implementations can use [`validate_chunk_element_indices`] to bounds-check `decoded_element_indices`.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `decoded_shape` is invalid or `decoded_element_indices` are out of bounds of `decoded_shape`.
    fn encoded_element_indices(
        &self,
        decoded_shape: &[NonZeroU64],
        decoded_element_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, CodecError>;
    /// Map `decoded_subset` of a decoded chunk of `decoded_shape` to the subset of the encoded chunk containing the exact same elements.
    ///
    /// Returns [`None`] if the elements of `decoded_subset` do not form an axis-aligned rectangular
    /// subset in the encoded chunk, or if the codec does not map individual elements.
    /// Implementations can use [`validate_chunk_subset`] to validate `decoded_subset`.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `decoded_shape` is invalid or `decoded_subset` is empty or out of bounds of `decoded_shape`.
    fn encoded_subset(
        &self,
        decoded_shape: &[NonZeroU64],
        decoded_subset: &ArraySubset,
    ) -> Result<Option<ArraySubset>, CodecError>;
}

/// Check that `element_indices` are in bounds of a chunk of `shape`.
///
/// # Errors
/// Returns a [`CodecError`] if `element_indices` have a different dimensionality to `shape` or are out of bounds.
pub fn validate_chunk_element_indices(
    shape: &[NonZeroU64],
    element_indices: &[u64],
) -> Result<(), CodecError> {
    if element_indices.len() == shape.len()
        && std::iter::zip(element_indices, shape).all(|(&index, length)| index < length.get())
    {
        Ok(())
    } else {
        Err(CodecError::Other(format!(
            "element indices {element_indices:?} are out of bounds of a chunk with shape {shape:?}"
        )))
    }
}

/// Check that `subset` is nonempty and in bounds of a chunk of `shape`.
///
/// # Errors
/// Returns a [`CodecError`] if `subset` is empty, has a different dimensionality to `shape`, or is out of bounds.
pub fn validate_chunk_subset(shape: &[NonZeroU64], subset: &ArraySubset) -> Result<(), CodecError> {
    if !subset.is_empty() && subset.inbounds_shape(bytemuck::must_cast_slice(shape)) {
        Ok(())
    } else {
        Err(CodecError::Other(format!(
            "subset {subset:?} is empty or out of bounds of a chunk with shape {shape:?}"
        )))
    }
}

/// Marker trait for array-to-array codecs with identity subchunking mappings.
///
/// This implements [`ArrayToArrayCodecSubchunkingTraits`] with mappings that leave chunk grids, element indices,
/// and subsets unchanged.
/// Implement this trait only when the codec preserves element ordering and chunk shape.
pub trait ArrayToArrayCodecSubchunkingIdentityTraits {}

impl<T> ArrayToArrayCodecSubchunkingTraits for T
where
    T: ArrayCodecTraits + ArrayToArrayCodecSubchunkingIdentityTraits + ?Sized,
{
    fn encoded_chunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridEncoded, ChunkGridCreateError> {
        Ok(decoded_chunk_grid.into())
    }

    fn decoded_subchunk_grid(
        &self,
        _decoded_chunk_grid: ChunkGridDecodedRef<'_>,
        encoded_subchunk_grid: ChunkGridEncodedRef<'_>,
    ) -> Result<ChunkGridDecoded, ChunkGridCreateError> {
        Ok(encoded_subchunk_grid.into())
    }

    fn encoded_element_indices(
        &self,
        decoded_shape: &[NonZeroU64],
        decoded_element_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, CodecError> {
        validate_chunk_element_indices(decoded_shape, decoded_element_indices)?;
        Ok(Some(decoded_element_indices.into()))
    }

    fn encoded_subset(
        &self,
        decoded_shape: &[NonZeroU64],
        decoded_subset: &ArraySubset,
    ) -> Result<Option<ArraySubset>, CodecError> {
        validate_chunk_subset(decoded_shape, decoded_subset)?;
        Ok(Some(decoded_subset.clone()))
    }

    fn decoded_local_subchunk_grid(
        &self,
        decoded_shape: &[NonZeroU64],
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        if encoded_subchunk_grid.array_shape()
            != bytemuck::must_cast_slice::<NonZeroU64, u64>(decoded_shape)
        {
            return Err(CodecError::Other(format!(
                "encoded subchunk grid array shape {:?} does not match encoded chunk shape {decoded_shape:?}",
                encoded_subchunk_grid.array_shape()
            )));
        }
        Ok(Some(encoded_subchunk_grid.clone()))
    }
}

/// Traits for array to array codecs.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
pub trait UnboundArrayToArrayCodecTraits: CodecTraits + core::fmt::Debug {
    /// Return a dynamic version of the codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits>;

    /// Bind this codec to a decoded data type, fill value, and codec-specific options.
    ///
    /// Binding eagerly validates and derives the encoded context.
    /// A codec may read its own options type from `codec_specific_options`, and must bind any codecs it contains with the same `codec_specific_options`.
    ///
    /// A `fill_value` of [`None`] means the fill value is undefined (e.g. within the `data_codecs` of an `optional` codec with a `null` fill value).
    /// A codec must then bind the codecs that follow it with an undefined fill value, rather than one derived from a placeholder.
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if the `data_type` or `fill_value` is not supported by this codec.
    /// Codecs whose encoded representation depends on the fill value must return an error if it is undefined.
    fn with_context(
        &self,
        data_type: DataType,
        fill_value: Option<FillValue>,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError>;
}

/// Runtime traits for an array-to-array codec bound to a data type and fill value.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
pub trait ArrayToArrayCodecTraits: ArrayToArrayCodecSubchunkingTraits + core::fmt::Debug {
    /// Return a dynamic version of the bound codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits>;

    /// Return the encoded data type.
    fn encoded_data_type(&self) -> &DataType;

    /// Return the encoded fill value, or [`None`] if it is undefined.
    fn encoded_fill_value(&self) -> Option<&FillValue>;

    /// Returns the shape of the encoded chunk for a given decoded chunk shape.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the `decoded_shape` is not supported by this codec.
    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        Ok(decoded_shape.to_vec())
    }

    /// Map a partial decode granularity from the encoded representation to the decoded representation.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the decoded shape or encoded granularity is not supported by this codec.
    fn partial_decode_granularity(
        &self,
        decoded_shape: &[NonZeroU64],
        encoded_granularity: &[NonZeroU64],
    ) -> Result<ChunkShape, CodecError> {
        let encoded_shape = self.encoded_shape(decoded_shape)?;
        if encoded_shape == decoded_shape {
            Ok(encoded_granularity.to_vec())
        } else {
            Ok(decoded_shape.to_vec())
        }
    }

    /// Encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or `bytes` is incompatible with the decoded representation.
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Decode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the decoded output is incompatible with the decoded representation.
    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Initialise a partial decoder.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        _ = options;
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().cloned(),
            self.into_dyn(),
        )))
    }

    /// Initialise a partial encoder.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        _ = options;
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_output_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().cloned(),
            self.into_dyn(),
        )))
    }

    #[cfg(feature = "async")]
    /// Initialise an asynchronous partial decoder.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        _ = options;
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().cloned(),
            self.into_dyn(),
        )))
    }

    #[cfg(feature = "async")]
    /// Initialise an asynchronous partial encoder.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        _ = options;
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_output_handle,
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().cloned(),
            self.into_dyn(),
        )))
    }
}
