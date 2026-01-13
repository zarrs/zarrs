use super::codec::ShardingCodecConfiguration;
use super::{Array, ArrayShape, ChunkGrid, ChunkShape};
use crate::array::codec::array_to_bytes::sharding::ShardingCodec;
use zarrs_metadata::ConfigurationSerialize;

/// An [`Array`] extension trait to simplify working with arrays using the `sharding_indexed` codec.
pub trait ArrayShardedExt: private::Sealed {
    /// Returns true if the array to bytes codec of the array is `sharding_indexed`.
    fn is_sharded(&self) -> bool;

    /// Returns true if the array-to-bytes codec of the array is `sharding_indexed` and the array has no array-to-array or bytes-to-bytes codecs.
    fn is_exclusively_sharded(&self) -> bool;

    /// Return the inner chunk shape as defined in the `sharding_indexed` codec metadata.
    ///
    /// Returns [`None`] for an unsharded array.
    fn inner_chunk_shape(&self) -> Option<ChunkShape>;

    /// The effective inner chunk shape.
    ///
    /// The effective inner chunk shape is the "read granularity" of the sharded array that accounts for array-to-array codecs preceding the sharding codec.
    /// For example, the transpose codec changes the shape of an array subset that corresponds to a single inner chunk.
    /// The effective inner chunk shape is used when determining the inner chunk grid of a sharded array.
    ///
    /// Returns [`None`] for an unsharded array of if the effective inner chunk shape is indeterminate.
    fn effective_inner_chunk_shape(&self) -> Option<ChunkShape>;

    /// Retrieve the inner chunk grid.
    ///
    /// This uses the effective inner shape so that reading an inner chunk reads only one contiguous byte range.
    ///
    /// Returns the normal chunk grid for an unsharded array.
    fn inner_chunk_grid(&self) -> ChunkGrid;

    /// Return the shape of the inner chunk grid (i.e., the number of inner chunks).
    ///
    /// Returns the normal chunk grid shape for an unsharded array.
    fn inner_chunk_grid_shape(&self) -> ArrayShape;
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

    fn inner_chunk_shape(&self) -> Option<ChunkShape> {
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

    fn effective_inner_chunk_shape(&self) -> Option<ChunkShape> {
        let mut inner_chunk_shape = self.inner_chunk_shape()?;
        for codec in self.codecs().array_to_array_codecs().iter().rev() {
            if let Ok(Some(inner_chunk_shape_)) = codec.decoded_shape(&inner_chunk_shape) {
                inner_chunk_shape = inner_chunk_shape_;
            } else {
                return None;
            }
        }
        Some(inner_chunk_shape)
    }

    fn inner_chunk_grid(&self) -> ChunkGrid {
        // FIXME: Create the inner chunk grid in `Array` and return a ref
        if let Some(inner_chunk_shape) = self.effective_inner_chunk_shape() {
            ChunkGrid::new(
                crate::array::chunk_grid::RegularChunkGrid::new(
                    self.shape().to_vec(),
                    inner_chunk_shape,
                ).expect("the chunk grid dimensionality is already confirmed to match the array dimensionality"),
            )
        } else {
            self.chunk_grid().clone()
        }
    }

    fn inner_chunk_grid_shape(&self) -> ArrayShape {
        self.inner_chunk_grid().grid_shape().to_vec()
    }
}

mod private {
    use super::Array;

    pub trait Sealed {}

    impl<TStorage: ?Sized> Sealed for Array<TStorage> {}
}
