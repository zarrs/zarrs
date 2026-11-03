//! The `sharding` array to bytes codec (Core).
//!
//! Sharding logically splits chunks (shards) into sub-chunks (inner chunks) that can be individually compressed and accessed.
//! This allows to colocate multiple chunks within one storage object, bundling them in shards.
//!
//! The [`ShardingCodecBuilder`] can help with creating a [`ShardingCodec`].
//!
//! ### Non-Divisible Subchunk Shapes
//! The subchunk shape need not evenly divide the shard shape.
//! Subchunks straddling the shard boundary are clipped to the shard shape.
//! For example, a subchunk shape of `[12]` with a shard shape of `[30]` has subchunks with shapes `[12]`, `[12]`, and `[6]`.
//!
//! <div class="warning">
//! Non-divisible subchunk shapes are proposed in <a href="https://github.com/zarr-developers/zarr-specs/pull/370">zarr-specs #370</a> (sharding 1.1-draft), but are not yet part of the specification.
//! Implementations of version 1.0 of the sharding specification must reject such arrays.
//! Existing arrays can be opened, but [`ArrayBuilder::build`](crate::array::ArrayBuilder::build) does not create them.
//! </div>
//!
//! ### Compatible Implementations
//! This is a core codec and should be compatible with all Zarr V3 implementations that support it.
//! Non-divisible subchunk shapes may be incompatible with other implementations.
//!
//! ### Specification
//!
//! - <https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html>
//! - <https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/sharding_indexed>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `sharding_indexed`
//!
//! ### Codec `configuration` Example - [`ShardingCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "chunk_shape": [32, 32, 32],
//!     "codecs": [
//!         {
//!             "name": "endian",
//!             "configuration": {
//!                 "endian": "little"
//!             }
//!         },
//!         {
//!             "name": "gzip",
//!             "configuration": {
//!                 "level": 1
//!             }
//!         }
//!     ],
//!     "index_codecs": [
//!         {
//!             "name": "endian",
//!             "configuration": {
//!                 "endian": "little"
//!             }
//!         },
//!         { "name": "crc32c" }
//!     ]
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::sharding::ShardingCodecConfigurationV1;
//! # serde_json::from_str::<ShardingCodecConfigurationV1>(JSON).unwrap();
//! ```
//!
//! ### Advanced Usage
//!
//! #### Retrieving Encoded Subchunks
//! Encoded subchunk bytes can be retrieved manually with [`ShardingPartialDecoder`] or [`AsyncShardingPartialDecoder`].
//! 1. Downcast the bound array-to-bytes codec to [`ShardingCodecBound`],
//! 2. create a storage partial decoder for the shard object,
//! 3. construct the sharding partial decoder with the bound codec accessors, and
//! 4. call [`ShardingPartialDecoder::retrieve_subchunk_encoded`] with subchunk indices local to the shard.
//!
//! See the `sharding_partial_decoder_retrieve_subchunk_encoded` test for an example.

#[cfg(test)]
mod indexer_tests;
mod sharding_codec;
mod sharding_codec_builder;
mod sharding_options;
#[cfg(feature = "async")]
mod sharding_partial_decoder_async;
mod sharding_partial_decoder_sync;
mod sharding_partial_encoder;
#[cfg(feature = "async")]
mod sharding_partial_encoder_async;

#[cfg(feature = "async")]
pub use sharding_partial_decoder_async::AsyncShardingPartialDecoder;
pub use sharding_partial_decoder_sync::ShardingPartialDecoder;

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::{BTreeMap, HashSet};
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;
use zarrs_chunk_grid::{ChunkGridTraits, IndexerIterator};
use zarrs_codec::{ComputeResources, CowBytes};

use crate::array::array_bytes_internal::split_optional;
use crate::array::chunk_grid::RegularBoundedChunkGrid;
use crate::array::concurrency::{split_compute_resources, split_resources};
use crate::array::{
    ArrayBytes, ArrayIndices, BytesRepresentation, ChunkGrid, ChunkShape, CodecChain,
    CodecChainBound, DataType, FillValue, Indexer, IndexerError, RecommendedConcurrency,
    ravel_indices, transmute_to_bytes,
};
pub use sharding_codec::{ShardingCodec, ShardingCodecBound};
pub use sharding_codec_builder::ShardingCodecBuilder;
pub(crate) use sharding_options::RequireDivisibleSubchunks;
pub use sharding_options::{ShardingCodecOptions, SubchunkWriteOrder};
use zarrs_codec::{
    ArrayCodecTraits, ArrayToBytesCodecTraits, BytesPartialDecoderTraits, ChunkGridDecoded, Codec,
    CodecError, CodecOptions, CodecPluginV3, CodecTraitsV3, Resources,
};
use zarrs_metadata::v3::MetadataV3;
pub use zarrs_metadata_ext::codec::sharding::{
    ShardingCodecConfiguration, ShardingCodecConfigurationV1, ShardingIndexLocation,
};
use zarrs_storage::byte_range::ByteRange;

zarrs_plugin::impl_extension_aliases!(ShardingCodec, v3: "sharding_indexed");

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<ShardingCodec>()
}

fn nested_local_subchunk_grids(
    subchunk_grid: ChunkGrid,
    inner_codecs: &CodecChainBound,
) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
    let mapped_subchunk_grid = ChunkGridDecoded::Array(subchunk_grid.clone());
    let mut subchunk_grids = vec![Some(subchunk_grid)];
    subchunk_grids.extend(
        zarrs_codec::ArrayToBytesCodecSubchunkingTraits::decoded_subchunk_grids(
            inner_codecs,
            (&mapped_subchunk_grid).into(),
        )
        .map_err(|err| CodecError::Other(err.to_string()))?
        .into_iter()
        .map(|grid| match grid {
            ChunkGridDecoded::Array(grid) => Some(grid),
            ChunkGridDecoded::None | ChunkGridDecoded::ChunkLocal => None,
        }),
    );
    Ok(subchunk_grids)
}

/// Return the subchunk codec hierarchy of a shard encoded with `inner_codecs`.
///
/// The subchunks of a shard are encoded with the inner codecs, and any subchunks nested inside
/// those subchunks are encoded with the subchunk codecs of the inner codecs.
fn nested_subchunk_codecs(
    inner_codecs: &Arc<CodecChainBound>,
) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
    let mut codecs: Vec<Arc<dyn ArrayToBytesCodecTraits>> =
        vec![ArrayToBytesCodecTraits::into_dyn(inner_codecs.clone())];
    codecs.extend(
        zarrs_codec::ArrayToBytesCodecSubchunkingTraits::subchunk_codecs(inner_codecs.as_ref()),
    );
    codecs
}

impl CodecTraitsV3 for ShardingCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: ShardingCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(ShardingCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToBytes(codec))
    }
}

fn calculate_chunks_per_shard(
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
) -> Result<ChunkShape, CodecError> {
    if shard_shape.len() != subchunk_shape.len() {
        return Err(CodecError::Other(format!(
            "subchunk shape {subchunk_shape:?} dimensionality does not match shard shape {shard_shape:?}"
        )));
    }
    // Subchunks straddling the shard boundary are clipped to the shard shape
    // See https://github.com/zarr-developers/zarr-specs/pull/370
    Ok(std::iter::zip(shard_shape, subchunk_shape)
        .map(|(s, c)| s.div_ceil(*c))
        .collect())
}

/// Return the subchunk grid of a shard, with subchunks straddling the shard boundary clipped to the shard shape.
fn subchunk_grid(
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
) -> Result<RegularBoundedChunkGrid, CodecError> {
    Ok(RegularBoundedChunkGrid::new(
        bytemuck::must_cast_slice(shard_shape).to_vec(),
        subchunk_shape.to_vec(),
    )?)
}

/// Return the number of subchunks in a shard.
fn num_subchunks(subchunk_grid: &RegularBoundedChunkGrid) -> usize {
    usize::try_from(subchunk_grid.grid_shape().iter().product::<u64>()).unwrap()
}

fn sharding_index_shape(chunks_per_shard: &[NonZeroU64]) -> ChunkShape {
    let mut index_shape = Vec::with_capacity(chunks_per_shard.len() + 1);
    index_shape.extend(chunks_per_shard);
    index_shape.push(unsafe { NonZeroU64::new_unchecked(2) });
    ChunkShape::from(index_shape)
}

fn compute_index_encoded_size(
    index_codecs: &CodecChainBound,
    sharding_index_shape: &[NonZeroU64],
) -> Result<u64, CodecError> {
    let bytes_representation = index_codecs.encoded_representation(sharding_index_shape)?;
    match bytes_representation {
        BytesRepresentation::FixedSize(size) => Ok(size),
        BytesRepresentation::BoundedSize(_) | BytesRepresentation::UnboundedSize => {
            Err(CodecError::Other(
                "the array index cannot include a variable size output codec".to_string(),
            ))
        }
    }
}

/// Decode and validate a shard index.
///
/// Returns an error if a subchunk span overflows, or overlaps the shard index at the start of the shard.
fn decode_shard_index(
    encoded_shard_index: &[u8],
    index_shape: &[NonZeroU64],
    index_codecs: &CodecChainBound,
    index_location: ShardingIndexLocation,
    options: &CodecOptions,
    resources: &ComputeResources,
) -> Result<Vec<u64>, CodecError> {
    // Decode the shard index
    let decoded_shard_index = index_codecs.decode(
        CowBytes::Borrowed(encoded_shard_index),
        index_shape,
        options,
        resources,
    )?;
    let decoded_shard_index = decoded_shard_index.into_fixed()?;
    let shard_index: Vec<u64> = decoded_shard_index
        .as_chunks::<8>()
        .0
        .iter()
        .map(|v| u64::from_ne_bytes(*v))
        .collect();

    // Validate the subchunk spans
    let data_start = match index_location {
        ShardingIndexLocation::Start => encoded_shard_index.len() as u64,
        ShardingIndexLocation::End => 0,
    };
    for &[offset, size] in shard_index.as_chunks::<2>().0 {
        let empty = offset == u64::MAX && size == u64::MAX;
        let overflows = offset.checked_add(size).is_none();
        let overlaps_index = offset < data_start && size > 0;
        if !empty && (overflows || overlaps_index) {
            return Err(shard_index_out_of_bounds());
        }
    }
    Ok(shard_index)
}

fn get_index_byte_range(
    index_shape: &[NonZeroU64],
    index_codecs: &CodecChainBound,
    index_location: ShardingIndexLocation,
) -> Result<ByteRange, CodecError> {
    let index_encoded_size = compute_index_encoded_size(index_codecs, index_shape)
        .map_err(|e| CodecError::Other(e.to_string()))?;
    Ok(match index_location {
        ShardingIndexLocation::Start => ByteRange::FromStart(0, Some(index_encoded_size)),
        ShardingIndexLocation::End => ByteRange::Suffix(index_encoded_size),
    })
}

fn subchunk_byte_range(
    shard_index: Option<&[u64]>,
    subchunk_grid: &RegularBoundedChunkGrid,
    chunk_indices: &[u64],
) -> Result<Option<ByteRange>, CodecError> {
    if let Some(shard_index) = shard_index {
        let chunks_per_shard = subchunk_grid.grid_shape();
        let shard_index_idx = ravel_indices(chunk_indices, chunks_per_shard).ok_or_else(|| {
            CodecError::Other(format!(
                "subchunk indices {chunk_indices:?} are out of bounds for chunks per shard {chunks_per_shard:?}"
            ))
        })?;
        let shard_index_idx = usize::try_from(shard_index_idx).unwrap();
        Ok(subchunk_offset_size(shard_index, shard_index_idx)
            .map(|(offset, size)| ByteRange::FromStart(offset, Some(size))))
    } else {
        Ok(None)
    }
}

/// Returns the byte offset and size of the subchunk at `subchunk_index` in a shard index.
///
/// Returns [`None`] if the subchunk is empty.
/// The span of a subchunk in a shard index from [`decode_shard_index`] does not overflow.
fn subchunk_offset_size(shard_index: &[u64], subchunk_index: usize) -> Option<(u64, u64)> {
    let offset = shard_index[subchunk_index * 2];
    let size = shard_index[subchunk_index * 2 + 1];
    if offset == u64::MAX && size == u64::MAX {
        None
    } else {
        Some((offset, size))
    }
}

/// Returns the byte range of the subchunk at `subchunk_index` in an encoded shard.
///
/// Returns [`None`] if the subchunk is empty.
///
/// # Panics
/// Panics if the subchunk span exceeds [`usize::MAX`], which cannot occur for a shard index from [`ShardingCodecBound::decode_index`].
fn subchunk_range_usize(shard_index: &[u64], subchunk_index: usize) -> Option<Range<usize>> {
    subchunk_offset_size(shard_index, subchunk_index).map(|(offset, size)| {
        let offset = usize::try_from(offset).unwrap();
        offset..offset + usize::try_from(size).unwrap()
    })
}

/// Check the encoded size of a subchunk in a shard index against the inner codecs.
///
/// Returns an error if the inner codecs have a fixed encoded size that differs from `size`.
fn validate_subchunk_encoded_size(
    inner_codecs: &CodecChainBound,
    subchunk_shape: &[NonZeroU64],
    size: u64,
) -> Result<(), CodecError> {
    if let BytesRepresentation::FixedSize(fixed_size) =
        inner_codecs.encoded_representation(subchunk_shape)?
        && fixed_size != size
    {
        return Err(CodecError::Other(format!(
            "The shard index has an encoded subchunk of {size} bytes, but its encoded size is {fixed_size} bytes. The chunk may be corrupted."
        )));
    }
    Ok(())
}

fn partial_decode_empty_shard<'a>(
    data_type: &DataType,
    fill_value: &FillValue,
    indexer: &dyn crate::array::Indexer,
) -> Result<ArrayBytes<'a>, CodecError> {
    ArrayBytes::new_fill_value(data_type, indexer.len(), fill_value).map_err(CodecError::from)
}

/// Selected elements in one subchunk, in their original indexer order.
struct SubchunkIndices {
    /// The shape of the subchunk, clipped to the shard shape.
    subchunk_shape: ChunkShape,
    /// The indices of each element within the subchunk, flattened.
    indices: Vec<u64>,
    /// The position of each element in the shard indexer.
    positions: Vec<usize>,
}

impl SubchunkIndices {
    fn new(subchunk_shape: ChunkShape) -> Self {
        Self {
            subchunk_shape,
            indices: Vec::new(),
            positions: Vec::new(),
        }
    }

    fn elements(&self) -> std::slice::ChunksExact<'_, u64> {
        self.indices.chunks_exact(self.dimensionality())
    }
}

impl Indexer for SubchunkIndices {
    fn dimensionality(&self) -> usize {
        self.subchunk_shape.len()
    }

    fn len(&self) -> u64 {
        self.positions.len() as u64
    }

    fn output_shape(&self) -> Vec<u64> {
        vec![self.len()]
    }

    fn validate(&self, array_shape: &[u64]) -> Result<(), IndexerError> {
        if self.dimensionality() != array_shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                self.dimensionality(),
                array_shape.len(),
            ));
        }
        self.elements().try_for_each(|indices| {
            if std::iter::zip(indices, array_shape).any(|(index, size)| index >= size) {
                Err(IndexerError::new_oob(
                    indices.to_vec(),
                    array_shape.to_vec(),
                ))
            } else {
                Ok(())
            }
        })
    }

    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndices>> {
        let dimensionality = self.dimensionality();
        let indices = self.indices.clone();
        Box::new((0..self.positions.len()).map(move |element| {
            indices[element * dimensionality..(element + 1) * dimensionality]
                .iter()
                .copied()
                .collect()
        }))
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
        if self.dimensionality() != array_shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                self.dimensionality(),
                array_shape.len(),
            ));
        }
        let elements = self.elements();
        let mut linearised_indices = Vec::with_capacity(elements.size_hint().0);
        for indices in elements {
            linearised_indices
                .push(ravel_indices(indices, array_shape).ok_or_else(|| {
                    IndexerError::new_oob(indices.to_vec(), array_shape.to_vec())
                })?);
        }
        Ok(Box::new(linearised_indices.into_iter()))
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
        Ok(Box::new(
            self.iter_linearised_indices(array_shape)?
                .map(|index| (index, 1)),
        ))
    }
}

/// Group a validated shard indexer by subchunk, retaining duplicate coordinates.
///
/// Groups are ordered by raveled subchunk index.
fn group_indices_by_subchunk(
    subchunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    indexer: &dyn Indexer,
) -> BTreeMap<u64, SubchunkIndices> {
    let mut groups: BTreeMap<u64, SubchunkIndices> = BTreeMap::new();
    for (position, indices) in indexer.iter_indices().enumerate() {
        let subchunk_indices: ArrayIndices = std::iter::zip(&indices, subchunk_shape)
            .map(|(&i, &cs)| i / cs)
            .collect();
        let subchunk_index = ravel_indices(&subchunk_indices, subchunk_grid.grid_shape())
            .expect("validated indexer");
        let group = groups.entry(subchunk_index).or_insert_with(|| {
            SubchunkIndices::new(
                subchunk_grid
                    .chunk_shape(&subchunk_indices)
                    .expect("matching dimensionality")
                    .expect("validated indexer"),
            )
        });
        group
            .indices
            .extend(std::iter::zip(&indices, subchunk_shape).map(|(&i, &cs)| i % cs));
        group.positions.push(position);
    }
    groups
}

/// Restore indexer order after independently decoding each subchunk's selected elements.
fn merge_indexer_subchunks(
    groups: Vec<(ArrayBytes<'static>, Vec<usize>)>,
    num_elements: usize,
    data_type: &DataType,
) -> Result<ArrayBytes<'static>, CodecError> {
    let mut num_decoded = 0;
    for (bytes, positions) in &groups {
        bytes.validate(positions.len() as u64, data_type)?;
        num_decoded += positions.len();
    }
    if num_decoded != num_elements {
        return Err(CodecError::Other(format!(
            "indexer yielded {num_decoded} elements, but has a length of {num_elements}"
        )));
    }
    merge_validated_indexer_subchunks(groups, num_elements, data_type)
}

/// Merge subchunk groups validated by [`merge_indexer_subchunks`].
fn merge_validated_indexer_subchunks(
    groups: Vec<(ArrayBytes<'static>, Vec<usize>)>,
    num_elements: usize,
    data_type: &DataType,
) -> Result<ArrayBytes<'static>, CodecError> {
    if groups.len() == 1 {
        // Positions within a group are ascending and cover every element, so already in order
        let (bytes, _) = groups.into_iter().next().expect("one group");
        return Ok(bytes);
    }
    if let Some(inner_data_type) = data_type.optional_inner() {
        // Merge the inner data and validity mask independently
        let (data_groups, mask_groups) = split_optional(groups)?;
        let data = merge_validated_indexer_subchunks(data_groups, num_elements, inner_data_type)?;
        let mask = merge_validated_indexer_subchunks(
            mask_groups,
            num_elements,
            &crate::array::data_type::uint8(),
        )?;
        return Ok(data.with_optional_mask(mask.into_fixed()?));
    }
    if let Some(element_size) = data_type.fixed_size() {
        let mut output = vec![0; num_elements * element_size];
        for (bytes, positions) in groups {
            let bytes = bytes.into_fixed()?;
            for (element, position) in positions.into_iter().enumerate() {
                output[position * element_size..(position + 1) * element_size]
                    .copy_from_slice(&bytes[element * element_size..(element + 1) * element_size]);
            }
        }
        Ok(output.into())
    } else {
        let groups = groups
            .into_iter()
            .map(|(bytes, positions)| Ok((positions, bytes.into_variable()?)))
            .collect::<Result<Vec<_>, CodecError>>()?;
        let mut elements: Vec<&[u8]> = vec![&[]; num_elements];
        let mut bytes_len = 0;
        for (positions, bytes) in &groups {
            for (element, &position) in positions.iter().enumerate() {
                let value = &bytes.bytes()[bytes.offsets().element_range(element)];
                elements[position] = value;
                bytes_len += value.len();
            }
        }
        let mut bytes = Vec::with_capacity(bytes_len);
        let mut offsets = Vec::with_capacity(num_elements + 1);
        offsets.push(0);
        for element in elements {
            bytes.extend_from_slice(element);
            offsets.push(bytes.len());
        }
        Ok(ArrayBytes::new_vlen(
            bytes,
            crate::array::array_bytes_internal::offsets_from_usize(offsets)?,
        )?)
    }
}

/// An update to a subchunk from a partial encode.
struct SubchunkUpdate<'a> {
    /// The raveled subchunk index within the shard.
    subchunk_index: u64,
    /// The shape of the subchunk, clipped to the shard shape.
    subchunk_shape: ChunkShape,
    /// The updated elements, relative to the subchunk origin.
    indexer: Box<dyn Indexer>,
    /// The updated element bytes, in `indexer` order.
    bytes: ArrayBytes<'a>,
    /// Whether the subchunk is entirely overwritten, so its existing data need not be read.
    fully_covered: bool,
}

/// Split a shard-level partial encode into updates of each intersected subchunk.
///
/// The `indexer` must be validated against the shard shape.
fn subchunk_updates<'a>(
    chunk_grid: &RegularBoundedChunkGrid,
    subchunk_shape: &[NonZeroU64],
    indexer: &dyn Indexer,
    bytes: &'a ArrayBytes<'_>,
    data_type: &DataType,
) -> Result<Vec<SubchunkUpdate<'a>>, CodecError> {
    if let Some(subset) = indexer.as_array_subset() {
        let subchunks = chunk_grid
            .chunks_in_array_subset(subset)?
            .expect("subchunks always within shard");
        let subset_start = subset.start();
        let subset_end_exc = subset.end_exc();
        let subset_shape = subset.shape();

        #[cfg(not(target_arch = "wasm32"))]
        let iterator = subchunks.indices().into_par_iter();
        #[cfg(target_arch = "wasm32")]
        let iterator = subchunks.indices().into_iter();

        iterator
            .map(|subchunk_indices: ArrayIndices| {
                let subchunk_index = ravel_indices(&subchunk_indices, chunk_grid.grid_shape())
                    .expect("inbounds chunk");
                let subchunk_subset = chunk_grid
                    .subset(&subchunk_indices)
                    .expect("matching dimensionality")
                    .expect("subchunk always within shard");
                let fully_covered = subchunk_subset
                    .start()
                    .iter()
                    .zip(subset_start.iter())
                    .all(|(a, b)| a >= b)
                    && subchunk_subset
                        .end_exc()
                        .iter()
                        .zip(subset_end_exc.iter())
                        .all(|(a, b)| a <= b);
                let overlap = subset.overlap(&subchunk_subset).unwrap();
                let bytes = bytes.extract_array_subset(
                    &overlap.relative_to(&subset_start).unwrap(),
                    &subset_shape,
                    data_type,
                )?;
                Ok(SubchunkUpdate {
                    subchunk_index,
                    subchunk_shape: subchunk_subset.chunk_shape().expect("nonempty subchunk"),
                    indexer: Box::new(overlap.relative_to(subchunk_subset.start()).unwrap()),
                    bytes,
                    fully_covered,
                })
            })
            .collect()
    } else {
        let groups = group_indices_by_subchunk(chunk_grid, subchunk_shape, indexer);
        let bytes_shape = [indexer.len()];
        groups
            .into_iter()
            .map(|(subchunk_index, group)| {
                let positions: Vec<ArrayIndices> = group
                    .positions
                    .iter()
                    .map(|&position| std::iter::once(position as u64).collect())
                    .collect();
                Ok(SubchunkUpdate {
                    subchunk_index,
                    subchunk_shape: group.subchunk_shape.clone(),
                    indexer: Box::new(group),
                    bytes: bytes.extract_array_subset(&positions, &bytes_shape, data_type)?,
                    // Duplicate indices are permitted, so coverage is not tracked
                    fully_covered: false,
                })
            })
            .collect()
    }
}

/// Return the indices and byte ranges of the stored subchunks of a shard that are not in `updated_subchunks`.
fn retained_subchunks<T>(
    shard_index: &[u64],
    updated_subchunks: &[(u64, T)],
) -> (Vec<u64>, Vec<ByteRange>) {
    let updated: HashSet<u64> = updated_subchunks.iter().map(|(index, _)| *index).collect();
    (0..shard_index.len() / 2)
        .filter_map(|subchunk_index| {
            let (offset, size) = subchunk_offset_size(shard_index, subchunk_index)?;
            let subchunk_index = subchunk_index as u64;
            (!updated.contains(&subchunk_index))
                .then_some((subchunk_index, ByteRange::FromStart(offset, Some(size))))
        })
        .unzip()
}

fn missing_retained_subchunks() -> CodecError {
    CodecError::from("the shard is missing but its index references stored subchunks")
}

fn invalid_shard_index() -> CodecError {
    CodecError::from("the shard index has an encoded subchunk beyond the maximum shard size")
}

/// The compacted shard is too large to allocate.
fn compacted_shard_too_large() -> CodecError {
    CodecError::from("the compacted shard is too large to allocate")
}

/// The shard index references bytes outside of the encoded subchunks of a shard.
fn shard_index_out_of_bounds() -> CodecError {
    CodecError::Other(
        "The shard index references out-of-bounds bytes. The chunk may be corrupted.".to_string(),
    )
}

/// Return the byte range of the encoded subchunks in a shard of `shard_len` bytes, excluding its index.
///
/// `shard_len` must be at least `index_encoded_size`.
fn shard_data_region(
    shard_len: usize,
    index_encoded_size: usize,
    index_location: ShardingIndexLocation,
) -> Range<usize> {
    match index_location {
        ShardingIndexLocation::Start => index_encoded_size..shard_len,
        ShardingIndexLocation::End => 0..shard_len - index_encoded_size,
    }
}

/// Returns the byte range of the subchunk at `subchunk_index` in a shard index.
///
/// Returns [`None`] if the subchunk is empty.
///
/// # Errors
/// Returns an error if the byte range overflows or is not within `data_region`.
fn checked_subchunk_range(
    shard_index: &[u64],
    subchunk_index: usize,
    data_region: &Range<usize>,
) -> Result<Option<Range<usize>>, CodecError> {
    subchunk_offset_size(shard_index, subchunk_index)
        .map(|(offset, size)| {
            ByteRange::FromStart(offset, Some(size))
                .checked_range_usize(data_region.end)
                .filter(|range| range.start >= data_region.start)
                .ok_or_else(shard_index_out_of_bounds)
        })
        .transpose()
}

/// A storage write that applies a partial shard update.
enum ShardWrite {
    /// Erase the shard, since every subchunk is empty.
    Erase,
    /// Replace the entire shard.
    Replace(Vec<u8>),
    /// Write each `(offset, bytes)` into the existing shard.
    Partial(Vec<(u64, CowBytes<'static>)>),
}

/// Update `shard_index` with the encoded `updated_subchunks` of a partial encode, and return the write that applies the update.
///
/// An updated subchunk is [`None`] if it is entirely the fill value.
/// New subchunks are appended after `max_data_offset`, the end of the encoded subchunks in `shard_index` before the update.
/// The shard is replaced rather than appended to if no other subchunks remain.
fn encode_shard_update(
    shard_index: &mut [u64],
    max_data_offset: u64,
    updated_subchunks: Vec<(u64, Option<CowBytes<'_>>)>,
    index_codecs: &CodecChainBound,
    index_shape: &[NonZeroU64],
    index_location: ShardingIndexLocation,
    options: &CodecOptions,
    resources: &ComputeResources,
) -> Result<ShardWrite, CodecError> {
    // Check if the shard can be entirely rewritten instead of appended
    //  This occurs if the shard index is empty if all of the intersected subchunks are removed
    for (subchunk_index, _) in &updated_subchunks {
        shard_index[usize::try_from(subchunk_index * 2).unwrap()] = u64::MAX;
        shard_index[usize::try_from(subchunk_index * 2 + 1).unwrap()] = u64::MAX;
    }
    let rewrite = shard_index.iter().all(|&x| x == u64::MAX);
    let max_data_offset = if rewrite { 0 } else { max_data_offset };

    // Get the offset for new data
    let index_encoded_size = compute_index_encoded_size(index_codecs, index_shape)?;
    let offset_new_chunks = match index_location {
        ShardingIndexLocation::Start => max_data_offset.max(index_encoded_size),
        ShardingIndexLocation::End => max_data_offset,
    };

    // Validate the entire append before changing the cached index or storage.
    let appended_size = updated_subchunks
        .iter()
        .try_fold(0u64, |size, (_, encoded)| {
            size.checked_add(encoded.as_ref().map_or(0, |bytes| bytes.len() as u64))
                .ok_or_else(invalid_shard_index)
        })?;
    let data_end = offset_new_chunks
        .checked_add(appended_size)
        .ok_or_else(invalid_shard_index)?;
    if index_location == ShardingIndexLocation::End {
        data_end
            .checked_add(index_encoded_size)
            .ok_or_else(invalid_shard_index)?;
    }

    // Update the shard index
    {
        let mut offset_append = offset_new_chunks;
        for (subchunk_index, subchunk_encoded) in &updated_subchunks {
            if let Some(subchunk_encoded) = subchunk_encoded {
                let len = subchunk_encoded.len() as u64;
                shard_index[usize::try_from(subchunk_index * 2).unwrap()] = offset_append;
                shard_index[usize::try_from(subchunk_index * 2 + 1).unwrap()] = len;
                offset_append += len;
            } else {
                shard_index[usize::try_from(subchunk_index * 2).unwrap()] = u64::MAX;
                shard_index[usize::try_from(subchunk_index * 2 + 1).unwrap()] = u64::MAX;
            }
        }
    }

    if shard_index.iter().all(|&x| x == u64::MAX) {
        // Erase the shard if all chunks are empty
        return Ok(ShardWrite::Erase);
    }

    // Encode the updated shard index
    let shard_index_bytes: CowBytes = transmute_to_bytes(&*shard_index).into();
    let encoded_array_index = index_codecs
        .encode(shard_index_bytes.into(), index_shape, options, resources)?
        .into_static();

    // Get the total size of the encoded subchunks
    let encoded_subchunks_size = updated_subchunks
        .iter()
        .filter_map(|(_, subchunk_encoded)| subchunk_encoded.as_ref().map(CowBytes::len))
        .sum::<usize>();

    // Get the suffix write size
    let suffix_write_size = match index_location {
        ShardingIndexLocation::Start => encoded_subchunks_size,
        ShardingIndexLocation::End => encoded_subchunks_size + encoded_array_index.len(),
    };

    // Concatenate the updated subchunks
    let mut encoded_output = Vec::with_capacity(suffix_write_size);
    for (_, subchunk_encoded) in updated_subchunks {
        if let Some(subchunk_encoded) = subchunk_encoded {
            encoded_output.extend_from_slice(&subchunk_encoded);
        }
    }

    if rewrite {
        // Replace the shard rather than erasing it first, so a failure leaves it intact
        let shard = match index_location {
            ShardingIndexLocation::Start => {
                debug_assert_eq!(offset_new_chunks, encoded_array_index.len() as u64);
                let mut shard =
                    Vec::with_capacity(encoded_array_index.len() + encoded_output.len());
                shard.extend_from_slice(&encoded_array_index);
                shard.extend_from_slice(&encoded_output);
                shard
            }
            ShardingIndexLocation::End => {
                debug_assert_eq!(offset_new_chunks, 0);
                encoded_output.extend_from_slice(&encoded_array_index);
                encoded_output
            }
        };
        Ok(ShardWrite::Replace(shard))
    } else {
        // Write the encoded index and updated subchunks
        Ok(ShardWrite::Partial(match index_location {
            ShardingIndexLocation::Start => vec![
                (0, encoded_array_index),
                (offset_new_chunks, CowBytes::from(encoded_output)),
            ],
            ShardingIndexLocation::End => {
                encoded_output.extend_from_slice(&encoded_array_index);
                vec![(offset_new_chunks, CowBytes::from(encoded_output))]
            }
        }))
    }
}

fn get_concurrent_target_and_resources(
    inner_codecs: &CodecChainBound,
    subchunk_shape: &[NonZeroU64],
    num_subchunks: usize,
    resources: &ComputeResources,
) -> Result<(usize, ComputeResources), CodecError> {
    Ok(split_compute_resources(
        resources,
        &RecommendedConcurrency::new_maximum(num_subchunks),
        &inner_codecs.recommended_concurrency(subchunk_shape)?,
    ))
}

fn get_concurrent_target_and_partial_resources(
    inner_codecs: &CodecChainBound,
    subchunk_shape: &[NonZeroU64],
    num_subchunks: usize,
    resources: &Resources,
) -> Result<(usize, Resources), CodecError> {
    Ok(split_resources(
        resources,
        &RecommendedConcurrency::new_maximum(num_subchunks),
        &inner_codecs.recommended_concurrency(subchunk_shape)?,
    ))
}

/// Returns `None` if there is no shard.
fn decode_shard_index_partial_decoder(
    input_handle: &dyn BytesPartialDecoderTraits,
    index_codecs: &CodecChainBound,
    index_location: ShardingIndexLocation,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Option<Vec<u64>>, CodecError> {
    let chunks_per_shard = calculate_chunks_per_shard(shard_shape, subchunk_shape)?;
    let index_shape = sharding_index_shape(&chunks_per_shard);
    let index_byte_range = get_index_byte_range(&index_shape, index_codecs, index_location)?;
    let encoded_shard_index = input_handle.partial_decode(index_byte_range, options, resources)?;
    Ok(match encoded_shard_index {
        Some(encoded_shard_index) => Some(decode_shard_index(
            &encoded_shard_index,
            &index_shape,
            index_codecs,
            index_location,
            options,
            resources.compute(),
        )?),
        None => None,
    })
}

#[cfg(feature = "async")]
/// Returns `None` if there is no shard.
async fn decode_shard_index_async_partial_decoder(
    input_handle: &dyn zarrs_codec::AsyncBytesPartialDecoderTraits,
    index_codecs: &CodecChainBound,
    index_location: ShardingIndexLocation,
    shard_shape: &[NonZeroU64],
    subchunk_shape: &[NonZeroU64],
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Option<Vec<u64>>, CodecError> {
    let chunks_per_shard = calculate_chunks_per_shard(shard_shape, subchunk_shape)?;
    let index_shape = sharding_index_shape(&chunks_per_shard);
    let index_byte_range = get_index_byte_range(&index_shape, index_codecs, index_location)?;
    let encoded_shard_index = input_handle
        .partial_decode(index_byte_range, options, resources)
        .await?;
    Ok(match encoded_shard_index {
        Some(encoded_shard_index) => Some(decode_shard_index(
            &encoded_shard_index,
            &index_shape,
            index_codecs,
            index_location,
            options,
            resources.compute(),
        )?),
        None => None,
    })
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;
    use crate::array::codec::bytes_to_bytes::test_unbounded::TestUnboundedCodec;
    use crate::array::{ArrayBytes, ArraySubset, ChunkShapeTraits, data_type};
    use zarrs_chunk_grid::Indexer;
    use zarrs_codec::{
        BytesToBytesCodecTraits, CodecSpecificOptions, UnboundArrayToBytesCodecTraits,
    };

    #[test]
    fn checked_subchunk_range_within_data_region() {
        let data_region = 8..16;
        let range = |offset, size| checked_subchunk_range(&[offset, size], 0, &data_region);
        assert_eq!(range(u64::MAX, u64::MAX).unwrap(), None);
        assert_eq!(range(8, 8).unwrap(), Some(8..16));
        assert_eq!(range(10, 0).unwrap(), Some(10..10));
        assert_eq!(range(16, 0).unwrap(), Some(16..16));
        // Into the index
        assert!(range(7, 1).is_err());
        assert!(range(0, 0).is_err());
        // Beyond the shard
        assert!(range(16, 1).is_err());
        assert!(range(17, 0).is_err());
        // Half-missing or overflowing
        assert!(range(u64::MAX, 0).is_err());
        assert!(range(8, u64::MAX).is_err());
        assert!(range(u64::MAX - 1, 2).is_err());
    }

    #[test]
    fn codec_sharding_decode_rejects_corrupt_subchunk_ranges() {
        const INDEX_SIZE: usize = size_of::<[u64; 8]>();
        let shape = [NonZeroU64::new(4).unwrap()];
        let options = CodecOptions::default();
        let resources = ComputeResources::default();
        for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
            let codec = Arc::new(
                ShardingCodecBuilder::new(vec![NonZeroU64::new(1).unwrap()], &data_type::uint16())
                    .index_bytes_to_bytes_codecs(vec![])
                    .index_location(location)
                    .build(),
            )
            .with_context(
                data_type::uint16(),
                Some(FillValue::from(0u16)),
                &CodecSpecificOptions::default(),
            )
            .unwrap();
            for (offset, size) in [
                (u64::MAX - 1, 2),
                (1, u64::MAX),
                (u64::MAX, 0),
                // Within the shard, but in its index
                (0, 1),
                (INDEX_SIZE as u64, 1),
            ] {
                let mut index = [u64::MAX; 8];
                index[6] = offset;
                index[7] = size;
                let encoded: Vec<u8> = index.into_iter().flat_map(u64::to_le_bytes).collect();
                let error = codec
                    .decode(encoded.clone().into(), &shape, &options, &resources)
                    .unwrap_err();
                assert!(error.to_string().contains("out-of-bounds bytes"), "{error}");

                let mut output = vec![0u8; 8];
                let error = codec
                    .decode_into(
                        CowBytes::from(encoded).into(),
                        &shape,
                        zarrs_codec::ArrayBytesDecodeIntoTarget::Fixed(
                            &mut super::indexer_tests::fixed_view(
                                &mut output,
                                &[4],
                                ArraySubset::new_with_shape(vec![4]),
                            ),
                        ),
                        &options,
                        &resources,
                    )
                    .unwrap_err();
                assert!(error.to_string().contains("out-of-bounds bytes"), "{error}");
            }
        }
    }

    fn get_concurrent_target(parallel: bool) -> usize {
        if parallel {
            rayon::current_num_threads()
        } else {
            1
        }
    }

    const JSON_VALID2: &str = r#"{
    "chunk_shape": [1, 2, 2],
    "codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        },
        {
            "name": "gzip",
            "configuration": {
                "level": 1
            }
        }
    ],
    "index_codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        },
        { "name": "crc32c" }
    ]
}"#;

    const JSON_VALID3: &str = r#"{
    "chunk_shape": [2, 2],
    "codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        }
    ],
    "index_codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        }
    ],
    "index_location": "start"
}"#;

    enum FillValueAmount {
        Partial,
        All,
        None,
    }

    /// Every write order.
    fn subchunk_write_orders() -> Vec<SubchunkWriteOrder> {
        vec![
            SubchunkWriteOrder::Lexicographic,
            SubchunkWriteOrder::Colexicographic,
            SubchunkWriteOrder::Morton,
            SubchunkWriteOrder::Unordered,
        ]
    }

    /// Bit-interleaved Morton code, first axis most significant.
    fn morton_key(indices: &[u64]) -> u64 {
        let mut key = 0;
        for bit in (0..u64::BITS as usize / indices.len()).rev() {
            for &c in indices {
                key = (key << 1) | ((c >> bit) & 1);
            }
        }
        key
    }

    /// Assert the written subchunks (sorted by offset) appear in the order implied by `order`.
    ///
    /// Unordered orders are not checked.
    fn assert_subchunk_write_order(index: &[u64], order: SubchunkWriteOrder, grid_shape: &[u64]) {
        let mut written: Vec<(u64, u64)> = index
            .chunks(2)
            .enumerate()
            .filter(|(_, e)| e[0] != u64::MAX)
            .map(|(i, e)| (e[0], i as u64))
            .collect();
        written.sort_unstable();
        let actual: Vec<u64> = written.iter().map(|(_, i)| *i).collect();
        let mut expected = actual.clone();
        let unravel = |i: u64| crate::array::unravel_index(i, grid_shape).unwrap();
        match order {
            SubchunkWriteOrder::Unordered => return,
            SubchunkWriteOrder::C | SubchunkWriteOrder::Lexicographic => expected.sort_unstable(),
            SubchunkWriteOrder::Colexicographic => expected.sort_by_key(|&i| {
                let mut idx = unravel(i);
                idx.reverse();
                idx
            }),
            SubchunkWriteOrder::Morton => expected.sort_by_key(|&i| morton_key(&unravel(i))),
        }
        assert_eq!(actual, expected);
    }

    fn codec_sharding_round_trip_impl(
        options: &CodecOptions,
        resources: &Resources,
        unbounded: bool,
        index_at_end: bool,
        fill_value_amount: &FillValueAmount,
        bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
        subchunk_write_order: SubchunkWriteOrder,
    ) {
        const NUM_AXES: usize = 3;
        let chunk_size = 16;
        let subchunk_size = 2;
        let chunk_shape = vec![NonZeroU64::new(chunk_size).unwrap(); NUM_AXES];
        let subchunk_shape = vec![NonZeroU64::new(subchunk_size).unwrap(); NUM_AXES];
        let data_type = data_type::uint16();
        let fill_value = FillValue::from(0u16);
        let elem_size = std::mem::size_of::<u16>();
        let elements: Vec<u16> = match fill_value_amount {
            FillValueAmount::All => vec![0u16; chunk_shape.num_elements_usize()],
            FillValueAmount::Partial => {
                // Two separate subsets, one entirely within a subchunk, other other crossing a boundary.
                // Three total chunks thus are written two..
                let subset1 = ArraySubset::new_with_ranges(&[(0..2), (0..2), (0..2)]);
                let subset2 = ArraySubset::new_with_ranges(&[(5..7), (0..2), (0..2)]);
                let mut data = vec![0u16; chunk_shape.num_elements_usize()];
                subset1
                    .iter_contiguous_byte_ranges(&[chunk_size; NUM_AXES], 2)
                    .unwrap()
                    .for_each(|r| {
                        let start = r.start as usize / elem_size;
                        let end = r.end as usize / elem_size;
                        data[start..end].fill(1);
                    });
                subset2
                    .iter_contiguous_byte_ranges(&[chunk_size; NUM_AXES], 2)
                    .unwrap()
                    .for_each(|r| {
                        let start = r.start as usize / elem_size;
                        let end = r.end as usize / elem_size;
                        data[start..end].fill(1);
                    });
                data
            }
            FillValueAmount::None => (1..=(chunk_shape.num_elements_usize() as u16)).collect(),
        };
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let codec = sharding_codec(
            subchunk_shape,
            data_type,
            fill_value,
            index_at_end,
            subchunk_write_order,
            bytes_to_bytes_codecs,
            unbounded,
        );
        let codec = codec.as_any().downcast_ref::<ShardingCodecBound>().unwrap();
        let encoded = codec
            .encode(bytes.clone(), &chunk_shape, options, resources.compute())
            .unwrap();
        let decoded = codec
            .decode(encoded.clone(), &chunk_shape, options, resources.compute())
            .unwrap();
        assert_eq!(bytes, decoded);
        assert_ne!(encoded, decoded.into_fixed().unwrap());
        let index = codec
            .decode_index(
                &encoded,
                &[NonZeroU64::new(chunk_size / subchunk_size).unwrap(); NUM_AXES],
                options,
                resources.compute(),
            )
            .unwrap();
        let grid_shape = [chunk_size / subchunk_size; NUM_AXES];
        let subchunk_write_order = codec.options.subchunk_write_order();
        match fill_value_amount {
            FillValueAmount::None => {
                assert_subchunk_write_order(&index, subchunk_write_order, &grid_shape);
            }
            FillValueAmount::Partial => {
                // 3 subchunks written, 2 entries (offset + len) each.
                assert_eq!(index.iter().filter(|e| **e != u64::MAX).count(), 6);
                assert_subchunk_write_order(&index, subchunk_write_order, &grid_shape);
            }
            FillValueAmount::All => assert_eq!(index, vec![u64::MAX; 8 * 8 * 8 * 2]),
        }
    }

    #[test]
    fn codec_sharding_round_trip1() {
        for subchunk_write_order in subchunk_write_orders() {
            for index_at_end in [true, false] {
                for fill_value_amount in [
                    FillValueAmount::All,
                    FillValueAmount::None,
                    FillValueAmount::Partial,
                ] {
                    for unbounded in [true, false] {
                        for parallel in [true, false] {
                            let concurrent_target = get_concurrent_target(parallel);
                            let options = CodecOptions::default();
                            let resources =
                                Resources::default().with_compute_target(concurrent_target);
                            codec_sharding_round_trip_impl(
                                &options,
                                &resources,
                                unbounded,
                                index_at_end,
                                &fill_value_amount,
                                vec![],
                                subchunk_write_order,
                            );
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "gzip")]
    #[cfg(feature = "crc32c")]
    #[test]
    fn codec_sharding_round_trip2() {
        use crate::array::codec::{Crc32cCodec, GzipCodec};
        for subchunk_write_order in [SubchunkWriteOrder::C, SubchunkWriteOrder::Unordered] {
            for index_at_end in [true, false] {
                for fill_value_amount in [
                    FillValueAmount::All,
                    FillValueAmount::None,
                    FillValueAmount::Partial,
                ] {
                    for unbounded in [true, false] {
                        for parallel in [true, false] {
                            let concurrent_target = get_concurrent_target(parallel);
                            let options = CodecOptions::default();
                            let resources =
                                Resources::default().with_compute_target(concurrent_target);
                            codec_sharding_round_trip_impl(
                                &options,
                                &resources,
                                unbounded,
                                index_at_end,
                                &fill_value_amount,
                                vec![
                                    Arc::new(GzipCodec::new(5).unwrap()),
                                    Arc::new(Crc32cCodec::new()),
                                ],
                                subchunk_write_order,
                            );
                        }
                    }
                }
            }
        }
    }

    /// Shard and subchunk shapes where the subchunk shape does not evenly divide the shard shape.
    fn nondivisible_shapes() -> Vec<(Vec<u64>, Vec<u64>)> {
        vec![
            // One axis divisible, the other not
            (vec![16, 15], vec![4, 4]),
            // Subchunk larger than the shard
            (vec![5], vec![12]),
            (vec![7, 5, 3], vec![2, 2, 2]),
        ]
    }

    fn to_nonzero(shape: &[u64]) -> ChunkShape {
        shape.iter().map(|&s| NonZeroU64::new(s).unwrap()).collect()
    }

    fn sharding_codec(
        subchunk_shape: ChunkShape,
        data_type: DataType,
        fill_value: FillValue,
        index_at_end: bool,
        subchunk_write_order: SubchunkWriteOrder,
        mut bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
        unbounded: bool,
    ) -> Arc<dyn ArrayToBytesCodecTraits> {
        if unbounded {
            bytes_to_bytes_codecs.push(Arc::new(TestUnboundedCodec::new()));
        }
        ShardingCodecBuilder::new(subchunk_shape, &data_type)
            .index_location(if index_at_end {
                ShardingIndexLocation::End
            } else {
                ShardingIndexLocation::Start
            })
            .bytes_to_bytes_codecs(bytes_to_bytes_codecs)
            .build()
            .with_context(
                data_type,
                Some(fill_value),
                &CodecSpecificOptions::default().with_option(
                    ShardingCodecOptions::default().with_subchunk_write_order(subchunk_write_order),
                ),
            )
            .unwrap()
    }

    fn fill_value(data_type: &DataType) -> FillValue {
        if data_type.is_fixed() {
            FillValue::from(0u16)
        } else {
            FillValue::from("")
        }
    }

    /// Shard data that is not the fill value, except for the first subchunk.
    fn nondivisible_data(
        shard_shape: &[u64],
        subchunk_shape: &[u64],
        data_type: &DataType,
    ) -> ArrayBytes<'static> {
        let first_subchunk = ArraySubset::from(
            std::iter::zip(shard_shape, subchunk_shape).map(|(&s, &c)| 0..s.min(c)),
        );
        let num_elements = shard_shape.iter().product::<u64>();
        let is_fill =
            |i: u64| first_subchunk.contains(&crate::array::unravel_index(i, shard_shape).unwrap());
        if data_type.is_fixed() {
            let elements: Vec<u16> = (0..num_elements)
                .map(|i| {
                    if is_fill(i) {
                        0
                    } else {
                        u16::try_from(i).unwrap() + 1
                    }
                })
                .collect();
            crate::array::transmute_to_bytes_vec(elements).into()
        } else {
            let elements: Vec<String> = (0..num_elements)
                .map(|i| {
                    if is_fill(i) {
                        String::new()
                    } else {
                        i.to_string()
                    }
                })
                .collect();
            crate::array::Element::into_array_bytes(data_type, elements).unwrap()
        }
    }

    #[test]
    fn codec_sharding_round_trip_nondivisible() {
        let data_type = data_type::uint16();
        for (shard_shape, subchunk_shape) in nondivisible_shapes() {
            let shard_shape_nz = to_nonzero(&shard_shape);
            let chunks_per_shard =
                calculate_chunks_per_shard(&shard_shape_nz, &to_nonzero(&subchunk_shape)).unwrap();
            let bytes = nondivisible_data(&shard_shape, &subchunk_shape, &data_type);
            for (subchunk_write_order, index_at_end, unbounded, parallel) in itertools::iproduct!(
                subchunk_write_orders(),
                [true, false],
                [true, false],
                [true, false]
            ) {
                let codec = sharding_codec(
                    to_nonzero(&subchunk_shape),
                    data_type.clone(),
                    fill_value(&data_type),
                    index_at_end,
                    subchunk_write_order,
                    vec![],
                    unbounded,
                );
                let options = CodecOptions::default();
                let resources =
                    Resources::default().with_compute_target(get_concurrent_target(parallel));
                let encoded = codec
                    .encode(
                        bytes.clone(),
                        &shard_shape_nz,
                        &options,
                        resources.compute(),
                    )
                    .unwrap();
                let decoded = codec
                    .decode(
                        encoded.clone(),
                        &shard_shape_nz,
                        &options,
                        resources.compute(),
                    )
                    .unwrap();
                assert_eq!(bytes, decoded);

                // The index has one entry per (clipped) subchunk, and the fill value subchunk is empty
                let index = codec
                    .as_any()
                    .downcast_ref::<ShardingCodecBound>()
                    .unwrap()
                    .decode_index(
                        &encoded,
                        chunks_per_shard.as_slice(),
                        &options,
                        resources.compute(),
                    )
                    .unwrap();
                assert_eq!(index.len() as u64, chunks_per_shard.num_elements_u64() * 2);
                assert_eq!(&index[..2], &[u64::MAX, u64::MAX]);
                assert_eq!(index.iter().filter(|&&v| v == u64::MAX).count(), 2);
                let grid_shape: Vec<u64> = chunks_per_shard.iter().map(|n| n.get()).collect();
                assert_subchunk_write_order(&index, subchunk_write_order, &grid_shape);

                // Decode into a larger output
                let output_shape: Vec<u64> = shard_shape.iter().map(|s| s + 2).collect();
                let output_subset = ArraySubset::new_with_start_shape(
                    vec![1; shard_shape.len()],
                    shard_shape.clone(),
                )
                .unwrap();
                let mut output =
                    vec![0u8; usize::try_from(output_shape.iter().product::<u64>()).unwrap() * 2];
                codec
                    .decode_into(
                        encoded.into(),
                        &shard_shape_nz,
                        zarrs_codec::ArrayBytesDecodeIntoTarget::Fixed(
                            &mut super::indexer_tests::fixed_view(
                                &mut output,
                                &output_shape,
                                output_subset.clone(),
                            ),
                        ),
                        &options,
                        resources.compute(),
                    )
                    .unwrap();
                let output: ArrayBytes = output.into();
                assert_eq!(
                    output
                        .extract_array_subset(&output_subset, &output_shape, &data_type)
                        .unwrap(),
                    bytes
                );
            }
        }
    }

    #[test]
    fn codec_sharding_round_trip_nondivisible_vlen() {
        for (shard_shape, subchunk_shape) in nondivisible_shapes() {
            let shard_shape_nz = to_nonzero(&shard_shape);
            let num_elements = shard_shape.iter().product::<u64>();
            // The final element is the fill value, so the final subchunk may be empty
            let elements: Vec<String> = (0..num_elements)
                .map(|i| {
                    if i + 1 == num_elements {
                        String::new()
                    } else {
                        "a".repeat(usize::try_from(i % 5).unwrap() + 1)
                    }
                })
                .collect();
            let bytes =
                crate::array::Element::into_array_bytes(&data_type::string(), elements).unwrap();
            for index_at_end in [true, false] {
                for unbounded in [true, false] {
                    for order in subchunk_write_orders() {
                        let options = CodecOptions::default();
                        let codec = sharding_codec(
                            to_nonzero(&subchunk_shape),
                            data_type::string(),
                            FillValue::from(""),
                            index_at_end,
                            order,
                            vec![],
                            unbounded,
                        );
                        let encoded = codec
                            .encode(
                                bytes.clone(),
                                &shard_shape_nz,
                                &options,
                                &ComputeResources::default(),
                            )
                            .unwrap();
                        let decoded = codec
                            .decode(
                                encoded,
                                &shard_shape_nz,
                                &options,
                                &ComputeResources::default(),
                            )
                            .unwrap();
                        assert_eq!(bytes, decoded);
                    }
                }
            }
        }
    }

    /// An encoded shard with non evenly divisible subchunks, and the expected partial decodes of some indexers.
    struct NondivisiblePartialDecodeCase {
        shard_shape: ChunkShape,
        codec: Arc<dyn ArrayToBytesCodecTraits>,
        encoded: CowBytes<'static>,
        expected: Vec<(Box<dyn Indexer>, ArrayBytes<'static>)>,
    }

    /// Partial decode cases, including regions at the final shard boundary.
    fn nondivisible_partial_decode_cases() -> Vec<NondivisiblePartialDecodeCase> {
        let mut cases = Vec::new();
        for data_type in [data_type::uint16(), data_type::string()] {
            for (shard_shape, subchunk_shape) in nondivisible_shapes() {
                let bytes = nondivisible_data(&shard_shape, &subchunk_shape, &data_type);
                for index_at_end in [true, false] {
                    for order in subchunk_write_orders() {
                        let codec = sharding_codec(
                            to_nonzero(&subchunk_shape),
                            data_type.clone(),
                            fill_value(&data_type),
                            index_at_end,
                            order,
                            vec![],
                            false,
                        );
                        let encoded = codec
                            .encode(
                                bytes.clone(),
                                &to_nonzero(&shard_shape),
                                &CodecOptions::default(),
                                &ComputeResources::default(),
                            )
                            .unwrap();
                        let indexers: Vec<Box<dyn Indexer>> = vec![
                            Box::new(ArraySubset::new_with_shape(shard_shape.clone())),
                            Box::new(ArraySubset::from(shard_shape.iter().map(|&s| (s - 1)..s))),
                            Box::new(ArraySubset::from(shard_shape.iter().map(|&s| (s / 2)..s))),
                            Box::new(vec![
                                shard_shape.iter().map(|&s| s - 1).collect::<ArrayIndices>(),
                                std::iter::repeat_n(0, shard_shape.len()).collect(),
                                shard_shape.iter().map(|&s| s / 2).collect(),
                                shard_shape.iter().map(|&s| s - 1).collect(),
                            ]),
                        ];
                        let expected = indexers
                            .into_iter()
                            .map(|indexer| {
                                let expected = bytes
                                    .extract_array_subset(
                                        indexer.as_ref(),
                                        &shard_shape,
                                        &data_type,
                                    )
                                    .unwrap()
                                    .into_owned();
                                (indexer, expected)
                            })
                            .collect();
                        cases.push(NondivisiblePartialDecodeCase {
                            shard_shape: to_nonzero(&shard_shape),
                            codec,
                            encoded,
                            expected,
                        });
                    }
                }
            }
        }
        cases
    }

    #[test]
    fn codec_sharding_partial_decode_nondivisible() {
        let options = CodecOptions::default();
        for case in nondivisible_partial_decode_cases() {
            let partial_decoder = case
                .codec
                .partial_decoder(
                    Arc::new(case.encoded),
                    &case.shard_shape,
                    &options,
                    &Resources::default(),
                )
                .unwrap();
            for (indexer, expected) in case.expected {
                assert_eq!(
                    partial_decoder
                        .partial_decode(indexer.as_ref(), &options, &Resources::default())
                        .unwrap(),
                    expected
                );
            }
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_sharding_async_partial_decode_nondivisible() {
        let options = CodecOptions::default();
        for case in nondivisible_partial_decode_cases() {
            let partial_decoder = case
                .codec
                .async_partial_decoder(
                    Arc::new(case.encoded),
                    &case.shard_shape,
                    &options,
                    &Resources::default(),
                )
                .await
                .unwrap();
            for (indexer, expected) in case.expected {
                assert_eq!(
                    partial_decoder
                        .partial_decode(indexer.as_ref(), &options, &Resources::default())
                        .await
                        .unwrap(),
                    expected
                );
            }
        }
    }

    /// Regression test for <https://github.com/zarrs/zarrs/issues/444>.
    ///
    /// A DEFLATE encoder may emit more blocks than `size / 32768`, so the gzip
    /// encoded size of poorly compressible data can exceed a bound that assumes
    /// a minimum block size, overflowing the shard buffer.
    /// With gzip level 0 this is deterministic: `miniz_oxide` splits a 32768
    /// byte input into two stored blocks.
    #[cfg(feature = "gzip")]
    #[cfg(feature = "crc32c")]
    #[test]
    fn codec_sharding_gzip_stored_blocks_exceed_naive_bound() {
        use crate::array::codec::GzipCodec;
        let subchunk_shape = ChunkShape::from(vec![NonZeroU64::new(32768).unwrap()]);
        let shard_shape = vec![NonZeroU64::new(65536).unwrap()];
        let bytes: ArrayBytes = vec![1u8; 65536].into();
        for index_at_end in [true, false] {
            for parallel in [true, false] {
                let options = CodecOptions::default();
                let resources =
                    Resources::default().with_compute_target(get_concurrent_target(parallel));
                let codec = Arc::new(
                    ShardingCodecBuilder::new(subchunk_shape.clone(), &data_type::uint8())
                        .index_location(if index_at_end {
                            ShardingIndexLocation::End
                        } else {
                            ShardingIndexLocation::Start
                        })
                        .bytes_to_bytes_codecs(vec![Arc::new(GzipCodec::new(0).unwrap())])
                        .build(),
                )
                .with_context(
                    data_type::uint8(),
                    Some(FillValue::from(0u8)),
                    &CodecSpecificOptions::default(),
                )
                .unwrap();
                let encoded = codec
                    .encode(bytes.clone(), &shard_shape, &options, resources.compute())
                    .unwrap();
                let decoded = codec
                    .decode(encoded, &shard_shape, &options, resources.compute())
                    .unwrap();
                assert_eq!(bytes, decoded);
            }
        }
    }

    #[cfg(feature = "async")]
    async fn codec_sharding_async_round_trip_impl(
        options: &CodecOptions,
        resources: &Resources,
        unbounded: bool,
        index_at_end: bool,
        all_fill_value: bool,
        mut bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    ) {
        let shape = vec![NonZeroU64::new(4).unwrap(); 2];
        let data_type = data_type::uint16();
        let fill_value = FillValue::from(0u16);
        let elements: Vec<u16> = if all_fill_value {
            vec![0; shape.num_elements_usize()]
        } else {
            (0..shape.num_elements_usize() as u16).collect()
        };
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        if unbounded {
            bytes_to_bytes_codecs.push(Arc::new(TestUnboundedCodec::new()));
        }
        let codec = Arc::new(
            ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type::uint16())
                .index_location(if index_at_end {
                    ShardingIndexLocation::End
                } else {
                    ShardingIndexLocation::Start
                })
                .bytes_to_bytes_codecs(bytes_to_bytes_codecs)
                .build(),
        )
        .with_context(
            data_type,
            Some(fill_value),
            &CodecSpecificOptions::default(),
        )
        .unwrap();

        let encoded = codec
            .encode(bytes.clone(), &shape, options, resources.compute())
            .unwrap();
        let partial_decoder = codec
            .async_partial_decoder(Arc::new(encoded.clone()), &shape, options, resources)
            .await
            .unwrap();
        let decoded = partial_decoder
            .partial_decode(
                &ArraySubset::new_with_shape(shape.to_array_shape()),
                options,
                resources,
            )
            .await
            .unwrap();
        assert_eq!(bytes, decoded);
        assert_ne!(encoded, decoded.into_fixed().unwrap());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_sharding_async_round_trip() {
        for index_at_end in [true, false] {
            for all_fill_value in [true, false] {
                for unbounded in [true, false] {
                    for parallel in [true, false] {
                        let concurrent_target = get_concurrent_target(parallel);
                        let options = CodecOptions::default();
                        let resources = Resources::default().with_compute_target(concurrent_target);
                        codec_sharding_async_round_trip_impl(
                            &options,
                            &resources,
                            unbounded,
                            index_at_end,
                            all_fill_value,
                            vec![],
                        )
                        .await;
                    }
                }
            }
        }
    }

    fn codec_sharding_partial_decode(
        options: &CodecOptions,
        resources: &Resources,
        unbounded: bool,
        index_at_end: bool,
        all_fill_value: bool,
    ) {
        let chunk_shape: ChunkShape = vec![NonZeroU64::new(4).unwrap(); 2];
        let data_type = data_type::uint8();
        let fill_value = FillValue::from(0u8);
        let elements: Vec<u8> = if all_fill_value {
            vec![0; chunk_shape.num_elements_usize()]
        } else {
            (0..chunk_shape.num_elements_usize() as u8).collect()
        };
        let answer: Vec<u8> = if all_fill_value {
            vec![0, 0]
        } else {
            vec![4, 8]
        };

        let bytes: ArrayBytes = elements.into();

        let bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>> = if unbounded {
            vec![Arc::new(TestUnboundedCodec::new())]
        } else {
            vec![]
        };
        let codec = Arc::new(
            ShardingCodecBuilder::new(chunk_shape.clone(), &data_type)
                .index_location(if index_at_end {
                    ShardingIndexLocation::End
                } else {
                    ShardingIndexLocation::Start
                })
                .bytes_to_bytes_codecs(bytes_to_bytes_codecs)
                .build(),
        )
        .with_context(
            data_type.clone(),
            Some(fill_value),
            &CodecSpecificOptions::default(),
        )
        .unwrap();

        let encoded = codec
            .encode(bytes.clone(), &chunk_shape, options, resources.compute())
            .unwrap();
        let decoded_region = ArraySubset::new_with_ranges(&[1..3, 0..1]);
        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .partial_decoder(input_handle, &chunk_shape, options, resources)
            .unwrap();
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_region, options, resources)
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_fixed()
            .unwrap()
            .as_chunks::<1>()
            .0
            .iter()
            .map(|b| u8::from_ne_bytes(*b))
            .collect();
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[test]
    fn codec_sharding_partial_decode_all() {
        for index_at_end in [true, false] {
            for all_fill_value in [true, false] {
                for unbounded in [true, false] {
                    for parallel in [true, false] {
                        let concurrent_target = get_concurrent_target(parallel);
                        let options = CodecOptions::default();
                        let resources = Resources::default().with_compute_target(concurrent_target);
                        codec_sharding_partial_decode(
                            &options,
                            &resources,
                            unbounded,
                            index_at_end,
                            all_fill_value,
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "async")]
    async fn codec_sharding_async_partial_decode(
        options: &CodecOptions,
        resources: &Resources,
        unbounded: bool,
        index_at_end: bool,
        all_fill_value: bool,
    ) {
        let chunk_shape: ChunkShape = vec![NonZeroU64::new(4).unwrap(); 2];
        let data_type = data_type::uint8();
        let fill_value = FillValue::from(0u8);
        let elements: Vec<u8> = if all_fill_value {
            vec![0; chunk_shape.num_elements_usize()]
        } else {
            (0..chunk_shape.num_elements_usize() as u8).collect()
        };
        let answer: Vec<u8> = if all_fill_value {
            vec![0, 0]
        } else {
            vec![4, 8]
        };
        let bytes: ArrayBytes = elements.into();

        let bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>> = if unbounded {
            vec![Arc::new(TestUnboundedCodec::new())]
        } else {
            vec![]
        };
        let codec = Arc::new(
            ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type::uint8())
                .index_location(if index_at_end {
                    ShardingIndexLocation::End
                } else {
                    ShardingIndexLocation::Start
                })
                .bytes_to_bytes_codecs(bytes_to_bytes_codecs)
                .build(),
        )
        .with_context(
            data_type,
            Some(fill_value),
            &CodecSpecificOptions::default(),
        )
        .unwrap();

        let encoded = codec
            .encode(bytes.clone(), &chunk_shape, options, resources.compute())
            .unwrap();
        let decoded_region = ArraySubset::new_with_ranges(&[1..3, 0..1]);
        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .async_partial_decoder(input_handle, &chunk_shape, options, resources)
            .await
            .unwrap();
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_region, options, resources)
            .await
            .unwrap()
            .into_fixed()
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .as_chunks::<1>()
            .0
            .iter()
            .map(|b| u8::from_ne_bytes(*b))
            .collect();
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_sharding_async_partial_decode_all() {
        for index_at_end in [true, false] {
            for all_fill_value in [true, false] {
                for unbounded in [true, false] {
                    for parallel in [true, false] {
                        let concurrent_target = get_concurrent_target(parallel);
                        let options = CodecOptions::default();
                        let resources = Resources::default().with_compute_target(concurrent_target);
                        codec_sharding_async_partial_decode(
                            &options,
                            &resources,
                            unbounded,
                            index_at_end,
                            all_fill_value,
                        )
                        .await;
                    }
                }
            }
        }
    }

    #[cfg(feature = "gzip")]
    #[cfg(feature = "crc32c")]
    #[test]
    fn codec_sharding_partial_decode2() {
        let chunk_shape: ChunkShape = vec![
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(4).unwrap(),
            NonZeroU64::new(4).unwrap(),
        ];
        let data_type = data_type::uint16();
        let fill_value = FillValue::from(0u16);
        let elements: Vec<u16> = (0..chunk_shape.num_elements_usize() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let codec_configuration: ShardingCodecConfiguration =
            serde_json::from_str(JSON_VALID2).unwrap();
        let codec = Arc::new(ShardingCodec::new_with_configuration(&codec_configuration).unwrap())
            .with_context(
                data_type,
                Some(fill_value),
                &CodecSpecificOptions::default(),
            )
            .unwrap();

        let encoded = codec
            .encode(
                bytes,
                &chunk_shape,
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();
        let decoded_region = ArraySubset::new_with_ranges(&[1..2, 0..2, 0..3]);
        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &chunk_shape,
                &CodecOptions::default(),
                &Resources::default(),
            )
            .unwrap();
        assert_eq!(
            partial_decoder.size_held(),
            input_handle.size_held() + size_of::<u64>() * 2 * 2 * 2 * 2
        ); // sharding partial decoder holds the shard index
        let decoded_partial_chunk = partial_decoder
            .partial_decode(
                &decoded_region,
                &CodecOptions::default(),
                &Resources::default(),
            )
            .unwrap();
        println!("decoded_partial_chunk {decoded_partial_chunk:?}");
        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .into_fixed()
            .unwrap()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();

        let answer: Vec<u16> = vec![16, 17, 18, 20, 21, 22];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[test]
    fn codec_sharding_partial_decode3() {
        let chunk_shape: ChunkShape = vec![NonZeroU64::new(4).unwrap(); 2];
        let data_type = data_type::uint8();
        let fill_value = FillValue::from(0u8);
        let elements: Vec<u8> = (0..chunk_shape.num_elements_usize() as u8).collect();
        let bytes: ArrayBytes = elements.into();

        let codec_configuration: ShardingCodecConfiguration =
            serde_json::from_str(JSON_VALID3).unwrap();
        let codec = Arc::new(ShardingCodec::new_with_configuration(&codec_configuration).unwrap())
            .with_context(
                data_type,
                Some(fill_value),
                &CodecSpecificOptions::default(),
            )
            .unwrap();

        let encoded = codec
            .encode(
                bytes,
                &chunk_shape,
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();
        let decoded_region = ArraySubset::new_with_ranges(&[1..3, 0..1]);
        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &chunk_shape,
                &CodecOptions::default(),
                &Resources::default(),
            )
            .unwrap();
        assert_eq!(
            partial_decoder.size_held(),
            input_handle.size_held() + size_of::<u64>() * 2 * 2 * 2
        ); // sharding partial decoder holds the shard index
        let decoded_partial_chunk = partial_decoder
            .partial_decode(
                &decoded_region,
                &CodecOptions::default(),
                &Resources::default(),
            )
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_fixed()
            .unwrap()
            .as_chunks::<1>()
            .0
            .iter()
            .map(|b| u8::from_ne_bytes(*b))
            .collect();
        let answer: Vec<u8> = vec![4, 8];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[test]
    fn codec_sharding_compact() {
        // Test that compact removes gaps in sharded data
        // 1. Fully encode a shard
        // 2. Update a subchunk and verify the shard size increased
        // 3. Compact and check that the size matches the original fully encoded shard

        let chunk_shape: ChunkShape = vec![NonZeroU64::new(4).unwrap(); 2];
        let data_type = data_type::uint16();
        let fill_value = Some(FillValue::from(0u16));
        let elements: Vec<u16> = (0..chunk_shape.num_elements_usize() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let original_bytes: ArrayBytes = bytes.into();

        // Create a sharding codec with 2x2 subchunks
        let codec_configuration: ShardingCodecConfiguration =
            serde_json::from_str(JSON_VALID3).unwrap();
        let codec = Arc::new(ShardingCodec::new_with_configuration(&codec_configuration).unwrap())
            .with_context(data_type, fill_value, &CodecSpecificOptions::default())
            .unwrap();

        // Step 1: Fully encode the shard
        let original_encoded = codec
            .encode(
                original_bytes.clone(),
                &chunk_shape,
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();
        let original_size = original_encoded.len();

        // Verify round trip works
        let decoded = codec
            .decode(
                original_encoded.clone(),
                &chunk_shape,
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();
        assert_eq!(original_bytes, decoded);

        // Step 2: Update a subchunk using partial encoder
        let input_output_handle = Arc::new(Mutex::new(Some(original_encoded.to_vec())));
        {
            // Update a single subchunk (e.g., chunk at position [0, 1])
            let subchunk_subset = ArraySubset::new_with_ranges(&[0..2, 2..4]);
            let updated_elements: Vec<u16> = vec![100, 101, 102, 103]; // New values for this subchunk
            let updated_bytes = crate::array::transmute_to_bytes_vec(updated_elements);
            let partial_encoder = codec
                .clone()
                .partial_encoder(
                    input_output_handle.clone(),
                    &chunk_shape,
                    &CodecOptions::default(),
                    &Resources::default(),
                )
                .unwrap();

            partial_encoder
                .partial_encode(
                    &subchunk_subset,
                    &ArrayBytes::from(updated_bytes),
                    &CodecOptions::default(),
                    &Resources::default(),
                )
                .unwrap();
        }
        let updated_encoded = Arc::try_unwrap(input_output_handle)
            .unwrap()
            .into_inner()
            .expect("single ref")
            .expect("non-empty shard");

        // The updated shard should be larger due to gaps from partial encoding
        let updated_size = updated_encoded.len();
        assert!(
            updated_size > original_size,
            "Updated shard size ({updated_size}) should be > original size ({original_size})"
        );

        // Step 3: Compact the updated shard
        let compacted = codec
            .compact(
                updated_encoded.into(),
                &chunk_shape,
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();
        let compacted = compacted.expect("compaction should have occurred");

        // The compacted shard should be the same size as the original fully encoded shard
        let compacted_size = compacted.len();
        assert_eq!(
            compacted_size, original_size,
            "Compacted size ({compacted_size}) should be equal to original size ({original_size})"
        );

        // Verify the compacted shard decodes correctly
        let decoded_after_compact = codec
            .decode(
                compacted.clone(),
                &chunk_shape,
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();

        // Build expected result: original data with the updated subchunk
        let mut expected_elements: Vec<u16> =
            (0..chunk_shape.num_elements_usize() as u16).collect();
        expected_elements[2] = 100; // Row 0, Col 2
        expected_elements[3] = 101; // Row 0, Col 3
        expected_elements[6] = 102; // Row 1, Col 2
        expected_elements[7] = 103; // Row 1, Col 3
        let expected_bytes = crate::array::transmute_to_bytes_vec(expected_elements);

        assert_eq!(ArrayBytes::from(expected_bytes), decoded_after_compact);

        // Verify that compacting an already compact shard is a no-op
        let compacted_again = codec
            .compact(
                CowBytes::Borrowed(&compacted),
                &chunk_shape,
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();
        assert!(compacted_again.is_none());

        // Verify that compacting the original encoded shard is a no-op (returns same reference)
        let original_compacted = codec
            .compact(
                CowBytes::Borrowed(&original_encoded),
                &chunk_shape,
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
            .unwrap();
        assert!(original_compacted.is_none());
    }

    /// An in-memory shard whose writes fail while `fail_writes` is set.
    struct FailingHandle {
        inner: Mutex<Option<Vec<u8>>>,
        fail_writes: std::sync::atomic::AtomicBool,
    }

    impl FailingHandle {
        fn new(bytes: Vec<u8>) -> Self {
            Self {
                inner: Mutex::new(Some(bytes)),
                fail_writes: std::sync::atomic::AtomicBool::new(false),
            }
        }

        fn set_fail_writes(&self, fail: bool) {
            self.fail_writes
                .store(fail, std::sync::atomic::Ordering::SeqCst);
        }

        fn check_write(&self) -> Result<(), CodecError> {
            if self.fail_writes.load(std::sync::atomic::Ordering::SeqCst) {
                Err(CodecError::Other("injected write failure".to_string()))
            } else {
                Ok(())
            }
        }

        fn bytes(&self) -> Option<Vec<u8>> {
            self.inner.lock().unwrap().clone()
        }
    }

    impl BytesPartialDecoderTraits for FailingHandle {
        fn exists(&self) -> Result<bool, zarrs_storage::StorageError> {
            self.inner.exists()
        }

        fn size_held(&self) -> usize {
            self.inner.size_held()
        }

        fn partial_decode_many(
            &self,
            decoded_regions: zarrs_storage::byte_range::ByteRangeIterator,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Option<Vec<CowBytes<'_>>>, CodecError> {
            self.inner
                .partial_decode_many(decoded_regions, options, resources)
        }

        fn supports_partial_decode(&self) -> bool {
            true
        }
    }

    impl zarrs_codec::BytesPartialEncoderTraits for FailingHandle {
        fn erase(&self) -> Result<(), CodecError> {
            zarrs_codec::BytesPartialEncoderTraits::erase(&self.inner)
        }

        fn encode(
            &self,
            bytes: CowBytes<'_>,
            options: &CodecOptions,
            resources: &zarrs_codec::Resources,
        ) -> Result<(), CodecError> {
            self.check_write()?;
            self.inner.encode(bytes, options, resources)
        }

        fn size(
            &self,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Option<u64>, CodecError> {
            zarrs_codec::BytesPartialEncoderTraits::size(&self.inner, options, resources)
        }

        fn partial_encode_many(
            &self,
            offset_values: zarrs_storage::OffsetBytesIterator<CowBytes<'_>>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), CodecError> {
            self.check_write()?;
            self.inner
                .partial_encode_many(offset_values, options, resources)
        }

        fn supports_partial_encode(&self) -> bool {
            true
        }
    }

    #[cfg(feature = "async")]
    #[async_trait::async_trait]
    impl zarrs_codec::AsyncBytesPartialDecoderTraits for FailingHandle {
        async fn exists(&self) -> Result<bool, zarrs_storage::StorageError> {
            BytesPartialDecoderTraits::exists(self)
        }

        fn size_held(&self) -> usize {
            BytesPartialDecoderTraits::size_held(self)
        }

        async fn partial_decode_many<'a>(
            &'a self,
            decoded_regions: zarrs_storage::byte_range::ByteRangeIterator<'a>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Option<Vec<CowBytes<'a>>>, CodecError> {
            BytesPartialDecoderTraits::partial_decode_many(
                self,
                decoded_regions,
                options,
                resources,
            )
        }

        fn supports_partial_decode(&self) -> bool {
            true
        }
    }

    #[cfg(feature = "async")]
    #[async_trait::async_trait]
    impl zarrs_codec::AsyncBytesPartialEncoderTraits for FailingHandle {
        async fn erase(&self) -> Result<(), CodecError> {
            zarrs_codec::BytesPartialEncoderTraits::erase(self)
        }

        async fn encode(
            &self,
            bytes: CowBytes<'_>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), CodecError> {
            zarrs_codec::BytesPartialEncoderTraits::encode(self, bytes, options, resources)
        }

        async fn size(
            &self,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<Option<u64>, CodecError> {
            zarrs_codec::BytesPartialEncoderTraits::size(self, options, resources)
        }

        async fn partial_encode_many<'a>(
            &'a self,
            offset_values: zarrs_storage::OffsetBytesIterator<'a, CowBytes<'_>>,
            options: &CodecOptions,
            resources: &Resources,
        ) -> Result<(), CodecError> {
            zarrs_codec::BytesPartialEncoderTraits::partial_encode_many(
                self,
                offset_values,
                options,
                resources,
            )
        }

        fn supports_partial_encode(&self) -> bool {
            true
        }
    }

    /// A `[2, 4]` shard of two `[2, 2]` subchunks, A (columns 0..2) and B (columns 2..4).
    ///
    /// Returns the codec, the shard shape, and the encoded shard with A and B fully written.
    fn failed_rewrite_shard(
        index_location: ShardingIndexLocation,
    ) -> (Arc<dyn ArrayToBytesCodecTraits>, ChunkShape, Vec<u8>) {
        let shape: ChunkShape = vec![NonZeroU64::new(2).unwrap(), NonZeroU64::new(4).unwrap()];
        let codec =
            ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type::uint8())
                .index_location(index_location)
                .build()
                .with_context(
                    data_type::uint8(),
                    Some(FillValue::from(FAILED_REWRITE_FILL)),
                    &CodecSpecificOptions::default(),
                )
                .unwrap();
        let elements: Vec<u8> = (1..=8).collect();
        let encoded = codec
            .encode(
                elements.into(),
                &shape,
                &CodecOptions::default(),
                &zarrs_codec::ComputeResources::default(),
            )
            .unwrap()
            .into_vec();
        (codec, shape, encoded)
    }

    /// Nonzero, so that bytes zero-filled by a write past the end of an erased shard are not
    /// mistaken for the fill value.
    const FAILED_REWRITE_FILL: u8 = 7;

    /// A = 30, B unchanged, since the failed rewrite left the existing shard intact.
    fn failed_rewrite_expected() -> ArrayBytes<'static> {
        vec![30u8, 30, 3, 4, 30, 30, 7, 8].into()
    }

    #[test]
    fn sharding_partial_encoder_failed_rewrite_invalidates_index() {
        let options = CodecOptions::default();
        let whole = ArraySubset::new_with_ranges(&[0..2, 0..4]);
        let subchunk_a = ArraySubset::new_with_ranges(&[0..2, 0..2]);
        for index_location in [ShardingIndexLocation::End, ShardingIndexLocation::Start] {
            let (codec, shape, shard_bytes) = failed_rewrite_shard(index_location);
            let handle = Arc::new(FailingHandle::new(shard_bytes));
            let encoder = codec
                .clone()
                .partial_encoder(
                    handle.clone(),
                    &shape,
                    &options,
                    &zarrs_codec::Resources::default(),
                )
                .unwrap();

            // Overwriting A and B rewrites the shard, and the write fails.
            handle.set_fail_writes(true);
            assert!(
                encoder
                    .partial_encode(
                        &whole,
                        &ArrayBytes::from(vec![20u8; 8]),
                        &options,
                        &zarrs_codec::Resources::default()
                    )
                    .is_err()
            );
            handle.set_fail_writes(false);

            encoder
                .partial_encode(
                    &subchunk_a,
                    &ArrayBytes::from(vec![30u8; 4]),
                    &options,
                    &zarrs_codec::Resources::default(),
                )
                .unwrap();

            let expected = failed_rewrite_expected();
            assert_eq!(
                encoder
                    .partial_decode(&whole, &options, &zarrs_codec::Resources::default())
                    .unwrap(),
                expected
            );
            let stored = handle.bytes().unwrap();
            assert_eq!(
                codec
                    .decode(
                        stored.into(),
                        &shape,
                        &options,
                        &zarrs_codec::ComputeResources::default()
                    )
                    .unwrap(),
                expected
            );
        }
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn sharding_async_partial_encoder_failed_rewrite_invalidates_index() {
        let options = CodecOptions::default();
        let whole = ArraySubset::new_with_ranges(&[0..2, 0..4]);
        let subchunk_a = ArraySubset::new_with_ranges(&[0..2, 0..2]);
        for index_location in [ShardingIndexLocation::End, ShardingIndexLocation::Start] {
            let (codec, shape, shard_bytes) = failed_rewrite_shard(index_location);
            let handle = Arc::new(FailingHandle::new(shard_bytes));
            let encoder = codec
                .clone()
                .async_partial_encoder(
                    handle.clone(),
                    &shape,
                    &options,
                    &zarrs_codec::Resources::default(),
                )
                .await
                .unwrap();

            // Overwriting A and B rewrites the shard, and the write fails.
            handle.set_fail_writes(true);
            assert!(
                encoder
                    .partial_encode(
                        &whole,
                        &ArrayBytes::from(vec![20u8; 8]),
                        &options,
                        &zarrs_codec::Resources::default()
                    )
                    .await
                    .is_err()
            );
            handle.set_fail_writes(false);

            encoder
                .partial_encode(
                    &subchunk_a,
                    &ArrayBytes::from(vec![30u8; 4]),
                    &options,
                    &zarrs_codec::Resources::default(),
                )
                .await
                .unwrap();

            let expected = failed_rewrite_expected();
            assert_eq!(
                encoder
                    .partial_decode(&whole, &options, &zarrs_codec::Resources::default())
                    .await
                    .unwrap(),
                expected
            );
            let stored = handle.bytes().unwrap();
            assert_eq!(
                codec
                    .decode(
                        stored.into(),
                        &shape,
                        &options,
                        &zarrs_codec::ComputeResources::default()
                    )
                    .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn sharding_partial_encoder_size_held() {
        let options = CodecOptions::default();
        let shape: ChunkShape = vec![NonZeroU64::new(2).unwrap(), NonZeroU64::new(4).unwrap()];
        let codec =
            ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type::uint8())
                .build()
                .with_context(
                    data_type::uint8(),
                    Some(FillValue::from(0u8)),
                    &CodecSpecificOptions::default(),
                )
                .unwrap();
        let elements: Vec<u8> = (1..=8).collect();
        let shard_bytes = codec
            .encode(
                elements.into(),
                &shape,
                &options,
                &ComputeResources::default(),
            )
            .unwrap()
            .into_vec();
        let handle = Arc::new(Mutex::new(Some(shard_bytes)));
        let encoder = codec
            .partial_encoder(handle.clone(), &shape, &options, &Resources::default())
            .unwrap();
        // The encoder holds the input handle and a shard index of 2 subchunks x (offset, size)
        assert_eq!(
            encoder.size_held(),
            BytesPartialDecoderTraits::size_held(handle.as_ref()) + 2 * 2 * size_of::<u64>()
        );
    }
}
