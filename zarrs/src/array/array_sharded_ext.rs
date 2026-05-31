use std::num::NonZeroU64;

use super::chunk_grid::{RectilinearChunkGrid, RegularChunkGrid};
use super::codec::ShardingCodecConfiguration;
use super::{Array, ArrayError, ArrayShape, ArraySubset, ChunkGrid, ChunkShape, CodecChain};
use crate::array::chunk_grid::ChunkEdgeLengths;
use crate::array::codec::array_to_bytes::sharding::ShardingCodec;
use zarrs_codec::ArrayToBytesCodecTraits;
use zarrs_metadata::ConfigurationSerialize;
use zarrs_metadata_ext::chunk_grid::rectilinear::RunLengthElement;

/// Iterate over subchunk sizes for a parent chunk along a single dimension.
fn subchunk_sizes(chunk_size: u64, subchunk_size: NonZeroU64) -> impl Iterator<Item = NonZeroU64> {
    let subchunk_size = subchunk_size.get();
    (0..chunk_size.div_ceil(subchunk_size)).map(move |i| {
        let start = i * subchunk_size;
        let remaining = chunk_size - start;
        NonZeroU64::new(remaining.min(subchunk_size)).expect("size is non-zero")
    })
}

/// Compute the subchunk grid shape and edge lengths for a single dimension.
///
/// Returns `None` if the dimension has zero grid shape and the decoded chunk
/// shape is not evenly divisible by the subchunk size (caller should return
/// the original chunk grid unchanged).
fn compute_dimension_subchunk_info(
    dim: usize,
    chunk_edge_lengths: &[NonZeroU64],
    decoded_chunk_shape: &ChunkShape,
    subchunk_shape: &ChunkShape,
) -> Option<(u64, ChunkEdgeLengths)> {
    let subchunk_size = subchunk_shape[dim];

    if chunk_edge_lengths.is_empty() {
        if decoded_chunk_shape[dim]
            .get()
            .is_multiple_of(subchunk_size.get())
        {
            return Some((0, ChunkEdgeLengths::Scalar(subchunk_shape[dim])));
        }
        return None;
    }

    let mut dimension_shape = 0;
    let mut sizes: Option<Vec<RunLengthElement>> = None;
    let mut regular_subchunk_count = 0;

    for chunk_size in chunk_edge_lengths.iter().map(|chunk_size| chunk_size.get()) {
        dimension_shape += chunk_size;

        if chunk_size % subchunk_size.get() == 0 {
            let count = chunk_size / subchunk_size.get();
            if let Some(sizes) = &mut sizes {
                sizes.push(RunLengthElement::Repeated([
                    subchunk_size,
                    NonZeroU64::new(count).expect("chunk size is non-zero"),
                ]));
            } else {
                regular_subchunk_count += count;
            }
        } else {
            let sizes = sizes.get_or_insert_with(|| {
                if let Some(count) = NonZeroU64::new(regular_subchunk_count) {
                    vec![RunLengthElement::Repeated([subchunk_size, count])]
                } else {
                    Vec::new()
                }
            });
            sizes.extend(subchunk_sizes(chunk_size, subchunk_size).map(RunLengthElement::Single));
        }
    }

    let edge_lengths = if let Some(sizes) = sizes {
        ChunkEdgeLengths::Varying(sizes)
    } else {
        ChunkEdgeLengths::Scalar(subchunk_size)
    };

    Some((dimension_shape, edge_lengths))
}

pub(crate) fn create_subchunk_grid(
    chunk_grid: &ChunkGrid,
    codecs: &CodecChain,
) -> Option<ChunkGrid> {
    if !codecs.array_to_bytes_codec().as_any().is::<ShardingCodec>() {
        return None;
    }

    let dimensionality = chunk_grid.dimensionality();
    let origin_chunk = vec![0; dimensionality];
    let decoded_chunk_shape = chunk_grid.chunk_shape(&origin_chunk).ok().flatten()?;
    let subchunk_shape = codecs.partial_decode_granularity(&decoded_chunk_shape);
    if subchunk_shape == decoded_chunk_shape {
        return Some(chunk_grid.clone());
    }

    let mut needs_rectilinear = false;
    let mut subchunk_grid_shape = Vec::with_capacity(dimensionality);
    let mut subchunk_edge_lengths = Vec::with_capacity(dimensionality);

    for dim in 0..dimensionality {
        let chunk_edge_lengths = chunk_grid.chunk_edge_lengths(dim).ok()?;
        let Some((dimension_shape, edge_lengths)) = compute_dimension_subchunk_info(
            dim,
            &chunk_edge_lengths,
            &decoded_chunk_shape,
            &subchunk_shape,
        ) else {
            return Some(chunk_grid.clone());
        };

        if let ChunkEdgeLengths::Varying(_) = &edge_lengths {
            needs_rectilinear = true;
        }
        subchunk_grid_shape.push(dimension_shape);
        subchunk_edge_lengths.push(edge_lengths);
    }

    if needs_rectilinear {
        Some(ChunkGrid::new(
            RectilinearChunkGrid::new(subchunk_grid_shape, &subchunk_edge_lengths).ok()?,
        ))
    } else {
        Some(ChunkGrid::new(
            RegularChunkGrid::new(subchunk_grid_shape, subchunk_shape).ok()?,
        ))
    }
}

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

    /// Retrieve the subchunk grid.
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

    fn subchunk_grid(&self) -> ChunkGrid {
        self.subchunk_grid
            .clone()
            .unwrap_or_else(|| self.chunk_grid().clone())
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
    let subchunk_shape = subchunk_grid
        .chunk_shape(subchunk_indices)?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?;
    let chunk_indices: Vec<u64> = shard_subset
        .start()
        .iter()
        .zip(subchunk_shape.as_slice())
        .map(|(o, s)| o / s.get())
        .collect();
    Ok((shard_indices, chunk_indices))
}
mod private {
    use super::Array;

    pub trait Sealed {}

    impl<TStorage: ?Sized> Sealed for Array<TStorage> {}
}
