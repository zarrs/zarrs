use super::codec::ShardingCodecConfiguration;
use super::{Array, ArrayShape, ChunkGrid, ChunkShape};
use crate::array::codec::array_to_bytes::sharding::ShardingCodec;
use crate::array::{ArrayError, ArraySubset};
use zarrs_metadata::ConfigurationSerialize;

/// An [`Array`] extension trait to simplify working with arrays using the `sharding_indexed` codec.
pub trait ArrayShardedExt: private::Sealed {
    /// Returns true if the array to bytes codec of the array is `sharding_indexed`.
    fn is_sharded(&self) -> bool;

    /// Returns true if the array-to-bytes codec of the array is `sharding_indexed` and the array has no array-to-array or bytes-to-bytes codecs.
    fn is_exclusively_sharded(&self) -> bool;

    /// Return the subchunk shape as defined in the `sharding_indexed` codec metadata.
    ///
    /// Returns [`None`] for an unsharded array.
    fn subchunk_shape(&self) -> Option<ChunkShape>;

    /// The effective subchunk shape.
    ///
    /// The effective subchunk shape is the "read granularity" of the sharded array that accounts for array-to-array codecs preceding the sharding codec.
    /// For example, the transpose codec changes the shape of an array subset that corresponds to a single subchunk.
    /// The effective subchunk shape is used when determining the subchunk grid of a sharded array.
    ///
    /// Returns [`None`] for an unsharded array of if the effective subchunk shape is indeterminate.
    fn effective_subchunk_shape(&self) -> Option<ChunkShape>;

    /// Retrieve the subchunk grid.
    ///
    /// This uses the effective subchunk shape so that reading a subchunk reads only one contiguous byte range.
    ///
    /// Returns the normal chunk grid for an unsharded array.
    fn subchunk_grid(&self) -> ChunkGrid;

    /// Return the shape of the subchunk grid (i.e., the number of subchunks).
    ///
    /// Returns the normal chunk grid shape for an unsharded array.
    fn subchunk_grid_shape(&self) -> ArrayShape;
}

impl<TStorage: ?Sized> ArrayShardedExt for Array<TStorage> {
    fn is_sharded(&self) -> bool {
        self.codecs
            .array_to_bytes_codec()
            .as_any()
            .is::<ShardingCodec>()
    }

    fn is_exclusively_sharded(&self) -> bool {
        self.is_sharded()
            && self.codecs.array_to_array_codecs().is_empty()
            && self.codecs.bytes_to_bytes_codecs().is_empty()
    }

    fn subchunk_shape(&self) -> Option<ChunkShape> {
        let configuration = self
            .codecs
            .array_to_bytes_codec()
            .configuration_v3(self.metadata_options.codec_metadata_options())
            .expect("the array to bytes codec should have metadata");
        if let Ok(ShardingCodecConfiguration::V1(sharding_configuration)) =
            ShardingCodecConfiguration::try_from_configuration(configuration)
        {
            Some(sharding_configuration.chunk_shape)
        } else {
            None
        }
    }

    fn effective_subchunk_shape(&self) -> Option<ChunkShape> {
        let mut subchunk_shape = self.subchunk_shape()?;
        for codec in self.codecs().array_to_array_codecs().iter().rev() {
            if let Ok(Some(subchunk_shape_)) = codec.decoded_shape(&subchunk_shape) {
                subchunk_shape = subchunk_shape_;
            } else {
                return None;
            }
        }
        Some(subchunk_shape)
    }

    fn subchunk_grid(&self) -> ChunkGrid {
        // FIXME: Create the subchunk grid in `Array` and return a ref
        if let Some(subchunk_shape) = self.effective_subchunk_shape() {
            ChunkGrid::new(
                crate::array::chunk_grid::RegularChunkGrid::new(
                    self.shape().to_vec(),
                    subchunk_shape,
                ).expect("the subchunk grid dimensionality is already confirmed to match the array dimensionality"),
            )
        } else {
            self.chunk_grid().clone()
        }
    }

    fn subchunk_grid_shape(&self) -> ArrayShape {
        self.subchunk_grid().grid_shape().to_vec()
    }
}

pub(super) fn subchunk_shard_index_and_subset<TStorage: ?Sized>(
    array: &Array<TStorage>,
    subchunk_grid: &ChunkGrid,
    subchunk_indices: &[u64],
) -> Result<(Vec<u64>, ArraySubset), ArrayError> {
    // TODO: Can this logic be simplified?
    let array_subset = subchunk_grid
        .subset(subchunk_indices)?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?;
    let shards = array
        .chunks_in_array_subset(&array_subset)?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?;
    if shards.num_elements() != 1 {
        // This should not happen, but it is checked just in case.
        return Err(ArrayError::InvalidChunkGridIndicesError(
            subchunk_indices.to_vec(),
        ));
    }
    let shard_indices = shards.start();
    let shard_origin = array.chunk_origin(shard_indices)?;
    let shard_subset = array_subset.relative_to(&shard_origin)?;
    Ok((shard_indices.to_vec(), shard_subset))
}

pub(super) fn subchunk_shard_index_and_chunk_index<TStorage: ?Sized>(
    array: &Array<TStorage>,
    subchunk_grid: &ChunkGrid,
    subchunk_indices: &[u64],
) -> Result<(Vec<u64>, Vec<u64>), ArrayError> {
    // TODO: Simplify this?
    let (shard_indices, shard_subset) =
        subchunk_shard_index_and_subset(array, subchunk_grid, subchunk_indices)?;
    let effective_subchunk_shape = array.effective_subchunk_shape().expect("array is sharded");
    let chunk_indices: Vec<u64> = shard_subset
        .start()
        .iter()
        .zip(effective_subchunk_shape.as_slice())
        .map(|(o, s)| o / s.get())
        .collect();
    Ok((shard_indices, chunk_indices))
}

mod private {
    use super::Array;

    pub trait Sealed {}

    impl<TStorage: ?Sized> Sealed for Array<TStorage> {}
}
