// TODO: reshape partial decoder

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_plugin::ZarrVersion;

use crate::array::chunk_grid::{ChunkEdgeLengths, RectilinearChunkGrid};
use crate::array::{ChunkGrid, ChunkShape, DataType, FillValue};
use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArrayToArrayCodecTraits, ChunkGridDecoded, ChunkGridDecodedRef, ChunkGridEncoded,
    ChunkGridEncodedRef, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecTraits, PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    UnboundArrayToArrayCodecTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::reshape::{
    ReshapeCodecConfiguration, ReshapeCodecConfigurationV1, ReshapeDim, ReshapeShape,
};
use zarrs_plugin::PluginCreateError;

/// A `reshape` codec implementation.
#[derive(Clone, Debug)]
pub struct ReshapeCodec {
    shape: ReshapeShape,
}

/// A `reshape` codec implementation bound to a data type and fill value.
#[derive(Clone, Debug)]
struct ReshapeCodecBound {
    shape: ReshapeShape,
    data_type: DataType,
    fill_value: FillValue,
}

fn chunk_edge_lengths_to_regular_granularity(
    chunk_grid: &ChunkGrid,
) -> Result<Option<ChunkShape>, ChunkGridCreateError> {
    get_regular_granularity_for_dimensions(chunk_grid)
}

/// Return the common chunk edge length for each dimension.
///
/// Returns `None` if any dimension is empty or has varying edge lengths.
fn get_regular_granularity_for_dimensions(
    chunk_grid: &ChunkGrid,
) -> Result<Option<ChunkShape>, ChunkGridCreateError> {
    let mut granularity = Vec::with_capacity(chunk_grid.dimensionality());
    for dim in 0..chunk_grid.dimensionality() {
        let edge_lengths = chunk_grid
            .chunk_edge_lengths(dim)
            .map_err(ChunkGridCreateError::from)?;
        let Some(first) = edge_lengths.first().copied() else {
            return Ok(None);
        };
        if edge_lengths.iter().any(|&edge_length| edge_length != first) {
            return Ok(None);
        }
        granularity.push(first);
    }
    Ok(Some(granularity))
}

fn input_dim_partitions(
    reshape_shape: &ReshapeShape,
    decoded_dimensionality: usize,
) -> Option<Vec<Vec<usize>>> {
    let mut seen = vec![false; decoded_dimensionality];
    let mut partitions = Vec::with_capacity(reshape_shape.0.len());
    for reshape_dim in &reshape_shape.0 {
        let ReshapeDim::InputDims(input_dims) = reshape_dim else {
            return None;
        };
        if input_dims.is_empty() {
            return None;
        }
        let input_dims = input_dims
            .iter()
            .copied()
            .map(usize::try_from)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        for &input_dim in &input_dims {
            let seen = seen.get_mut(input_dim)?;
            if *seen {
                return None;
            }
            *seen = true;
        }
        partitions.push(input_dims);
    }
    seen.into_iter().all(|seen| seen).then_some(partitions)
}

fn input_dim_partitions_are_identity(partitions: &[Vec<usize>]) -> bool {
    partitions
        .iter()
        .enumerate()
        .all(|(dim, partition)| partition.as_slice() == [dim])
}

fn cartesian_product_edge_lengths(edge_lengths: &[Vec<NonZeroU64>]) -> Option<Vec<NonZeroU64>> {
    let mut products = vec![NonZeroU64::new(1).unwrap()];
    for edges in edge_lengths {
        let capacity = products.len().checked_mul(edges.len())?;
        let mut next = Vec::with_capacity(capacity);
        for product in products {
            for edge in edges {
                next.push(NonZeroU64::new(product.get().checked_mul(edge.get())?)?);
            }
        }
        products = next;
    }
    Some(products)
}

fn encoded_chunk_grid_for_input_dim_partitions(
    partitions: &[Vec<usize>],
    decoded_chunk_grid: &ChunkGrid,
) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
    let mut array_shape = Vec::with_capacity(partitions.len());
    let mut chunk_shapes = Vec::with_capacity(partitions.len());
    for partition in partitions {
        let edge_lengths = partition
            .iter()
            .map(|&dim| decoded_chunk_grid.chunk_edge_lengths(dim))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(encoded_edge_lengths) = cartesian_product_edge_lengths(&edge_lengths) else {
            return Ok(None);
        };
        let Some(encoded_dim_shape) = encoded_edge_lengths
            .iter()
            .try_fold(0u64, |sum, edge| sum.checked_add(edge.get()))
        else {
            return Ok(None);
        };
        array_shape.push(encoded_dim_shape);
        chunk_shapes.push(ChunkEdgeLengths::encode(&encoded_edge_lengths));
    }

    Ok(Some(ChunkGrid::new(RectilinearChunkGrid::new(
        array_shape,
        &chunk_shapes,
    )?)))
}

fn repeat_shape(grid_shape: &[u64], dimensionality: usize) -> Option<Vec<u64>> {
    if dimensionality == 0 {
        return None;
    }
    let num_chunks = grid_shape
        .iter()
        .try_fold(1u64, |product, count| product.checked_mul(*count))?;
    let mut repeats = vec![1; dimensionality];
    *repeats.last_mut().unwrap() = num_chunks;
    Some(repeats)
}

fn repeated_chunk_grid(
    chunk_shape: &[NonZeroU64],
    repeats: &[u64],
) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
    let mut array_shape = Vec::with_capacity(chunk_shape.len());
    let mut chunk_shapes = Vec::with_capacity(chunk_shape.len());
    for (edge, repeat) in chunk_shape.iter().zip(repeats) {
        let Some(shape) = edge.get().checked_mul(*repeat) else {
            return Ok(None);
        };
        let Ok(repeat) = usize::try_from(*repeat) else {
            return Ok(None);
        };
        array_shape.push(shape);
        chunk_shapes.push(ChunkEdgeLengths::encode(&vec![*edge; repeat]));
    }
    Ok(Some(ChunkGrid::new(RectilinearChunkGrid::new(
        array_shape,
        &chunk_shapes,
    )?)))
}

fn split_edge_refinements(
    outer_edges: &[NonZeroU64],
    refined_edges: &[NonZeroU64],
) -> Option<Vec<Vec<NonZeroU64>>> {
    let mut refined = refined_edges.iter().copied();
    let mut groups = Vec::with_capacity(outer_edges.len());
    for outer_edge in outer_edges {
        let mut group = Vec::new();
        let mut sum = 0u64;
        while sum < outer_edge.get() {
            let edge = refined.next()?;
            sum = sum.checked_add(edge.get())?;
            if sum > outer_edge.get() {
                return None;
            }
            group.push(edge);
        }
        groups.push(group);
    }
    refined.next().is_none().then_some(groups)
}

fn unravel_flat_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut indices = vec![0; shape.len()];
    for (index_out, size) in indices.iter_mut().zip(shape).rev() {
        *index_out = index % size;
        index /= size;
    }
    indices
}

/// Map a sequence of contiguous row-major intervals to a rectilinear grid.
pub(super) fn decoded_edge_lengths_from_linear_intervals(
    decoded_shape: &[NonZeroU64],
    intervals: &[NonZeroU64],
) -> Option<Vec<Vec<NonZeroU64>>> {
    let num_elements = decoded_shape
        .iter()
        .try_fold(1u64, |product, dim| product.checked_mul(dim.get()))?;
    if intervals
        .iter()
        .try_fold(0u64, |sum, edge| sum.checked_add(edge.get()))?
        != num_elements
    {
        return None;
    }

    for pivot in 0..decoded_shape.len() {
        let trailing = decoded_shape[pivot + 1..]
            .iter()
            .try_fold(1u64, |product, dim| product.checked_mul(dim.get()))?;
        let outer_repeats = decoded_shape[..pivot]
            .iter()
            .try_fold(1u64, |product, dim| product.checked_mul(dim.get()))?;
        let outer_repeats = usize::try_from(outer_repeats).ok()?;

        let mut pivot_edges = Vec::new();
        let mut pivot_sum = 0u64;
        for interval in intervals {
            if !interval.get().is_multiple_of(trailing) {
                break;
            }
            let edge = interval.get() / trailing;
            pivot_sum = pivot_sum.checked_add(edge)?;
            if pivot_sum > decoded_shape[pivot].get() {
                break;
            }
            pivot_edges.push(NonZeroU64::new(edge)?);
            if pivot_sum == decoded_shape[pivot].get() {
                break;
            }
        }
        if pivot_sum != decoded_shape[pivot].get() {
            continue;
        }
        let expected_len = pivot_edges.len().checked_mul(outer_repeats)?;
        if intervals.len() != expected_len {
            continue;
        }
        let expected_intervals = (0..outer_repeats)
            .flat_map(|_| pivot_edges.iter())
            .map(|edge| NonZeroU64::new(edge.get().checked_mul(trailing)?))
            .collect::<Option<Vec<_>>>()?;
        if intervals != expected_intervals {
            continue;
        }

        let mut edge_lengths = Vec::with_capacity(decoded_shape.len());
        for dim in &decoded_shape[..pivot] {
            edge_lengths.push(vec![
                NonZeroU64::new(1).unwrap();
                usize::try_from(dim.get()).ok()?
            ]);
        }
        edge_lengths.push(pivot_edges);
        for dim in &decoded_shape[pivot + 1..] {
            edge_lengths.push(vec![*dim]);
        }
        return Some(edge_lengths);
    }
    None
}

pub(super) fn linear_intervals_from_rectilinear_grid(
    shape: &[NonZeroU64],
    grid: &ChunkGrid,
) -> Result<Option<Vec<NonZeroU64>>, ChunkGridCreateError> {
    if grid.dimensionality() != shape.len()
        || !grid
            .array_shape()
            .iter()
            .zip(shape)
            .all(|(grid, shape)| *grid == shape.get())
    {
        return Ok(None);
    }
    let edge_lengths = (0..grid.dimensionality())
        .map(|dim| grid.chunk_edge_lengths(dim))
        .collect::<Result<Vec<_>, _>>()?;
    for pivot in 0..shape.len() {
        let leading_is_unit = edge_lengths[..pivot]
            .iter()
            .all(|edges| edges.iter().all(|edge| edge.get() == 1));
        let trailing_is_full = edge_lengths[pivot + 1..]
            .iter()
            .zip(&shape[pivot + 1..])
            .all(|(edges, shape)| edges.as_slice() == [*shape]);
        if leading_is_unit && trailing_is_full {
            return Ok(cartesian_product_edge_lengths(&edge_lengths));
        }
    }
    Ok(None)
}

fn decoded_subchunk_grid_for_input_dim_partitions(
    partitions: &[Vec<usize>],
    decoded_chunk_grid: &ChunkGrid,
    encoded_subchunk_grid: &ChunkGrid,
) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
    if encoded_subchunk_grid.dimensionality() != partitions.len() {
        return Err(ChunkGridCreateError::new(format!(
            "encoded subchunk grid dimensionality {} is incompatible with encoded dimensionality {}",
            encoded_subchunk_grid.dimensionality(),
            partitions.len()
        )));
    }

    let decoded_edge_lengths = (0..decoded_chunk_grid.dimensionality())
        .map(|dim| decoded_chunk_grid.chunk_edge_lengths(dim))
        .collect::<Result<Vec<_>, _>>()?;
    let mut local_edges = decoded_edge_lengths
        .iter()
        .map(|edges| vec![None; edges.len()])
        .collect::<Vec<Vec<Option<Vec<NonZeroU64>>>>>();

    for (encoded_dim, partition) in partitions.iter().enumerate() {
        let partition_edges = partition
            .iter()
            .map(|&dim| decoded_edge_lengths[dim].clone())
            .collect::<Vec<_>>();
        let Some(encoded_outer_edges) = cartesian_product_edge_lengths(&partition_edges) else {
            return Ok(None);
        };
        let refined_edges = encoded_subchunk_grid.chunk_edge_lengths(encoded_dim)?;
        let Some(refined_by_outer_chunk) =
            split_edge_refinements(&encoded_outer_edges, &refined_edges)
        else {
            return Ok(None);
        };
        let partition_grid_shape = partition_edges.iter().map(Vec::len).collect::<Vec<_>>();

        for (flat_index, intervals) in refined_by_outer_chunk.iter().enumerate() {
            let indices = unravel_flat_index(flat_index, &partition_grid_shape);
            let decoded_local_shape = partition
                .iter()
                .zip(&indices)
                .map(|(&dim, &index)| decoded_edge_lengths[dim][index])
                .collect::<Vec<_>>();
            let Some(decoded_local_edges) =
                decoded_edge_lengths_from_linear_intervals(&decoded_local_shape, intervals)
            else {
                return Ok(None);
            };

            for ((&decoded_dim, &outer_index), edges) in
                partition.iter().zip(&indices).zip(decoded_local_edges)
            {
                let candidate = &mut local_edges[decoded_dim][outer_index];
                if candidate
                    .as_ref()
                    .is_some_and(|candidate| candidate != &edges)
                {
                    return Ok(None);
                }
                *candidate = Some(edges);
            }
        }
    }

    let chunk_shapes = local_edges
        .into_iter()
        .map(|per_outer_chunk| {
            per_outer_chunk
                .into_iter()
                .flatten()
                .flatten()
                .collect::<Vec<_>>()
        })
        .map(|edges| ChunkEdgeLengths::encode(&edges))
        .collect::<Vec<_>>();
    Ok(Some(ChunkGrid::new(RectilinearChunkGrid::new(
        decoded_chunk_grid.array_shape().to_vec(),
        &chunk_shapes,
    )?)))
}

impl ReshapeCodec {
    /// Create a new reshape codec from configuration.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if there is a configuration issue.
    pub fn new_with_configuration(
        configuration: &ReshapeCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            ReshapeCodecConfiguration::V1(configuration) => {
                Ok(Self::new(configuration.shape.clone()))
            }
            _ => Err(PluginCreateError::Other(
                "this reshape codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    /// Create a new reshape codec.
    #[must_use]
    pub const fn new(shape: ReshapeShape) -> Self {
        Self { shape }
    }
}

impl CodecTraits for ReshapeCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = ReshapeCodecConfiguration::V1(ReshapeCodecConfigurationV1 {
            shape: self.shape.clone(),
        });
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: true,
        }
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl UnboundArrayToArrayCodecTraits for ReshapeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self as Arc<dyn UnboundArrayToArrayCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        Ok(Arc::new(ReshapeCodecBound {
            shape: self.shape.clone(),
            data_type,
            fill_value,
        }))
    }
}

impl ArrayCodecTraits for ReshapeCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for ReshapeCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.data_type
    }

    fn encoded_fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        super::get_encoded_shape(&self.shape, decoded_shape)
    }

    fn encoded_chunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridEncoded, ChunkGridCreateError> {
        let ChunkGridDecodedRef::Array(decoded_chunk_grid) = decoded_chunk_grid else {
            return Ok(decoded_chunk_grid.into());
        };
        if decoded_chunk_grid.array_shape().contains(&0) {
            return Ok(ChunkGridEncoded::None);
        }

        if let Some(partitions) =
            input_dim_partitions(&self.shape, decoded_chunk_grid.dimensionality())
        {
            if input_dim_partitions_are_identity(&partitions) {
                return Ok(ChunkGridEncoded::Array(decoded_chunk_grid.clone()));
            }
            return encoded_chunk_grid_for_input_dim_partitions(&partitions, decoded_chunk_grid)
                .map(ChunkGridEncoded::from);
        }

        // Other reshape forms can be repeated when every decoded chunk has the
        // same shape. The reshape is evaluated on that chunk shape, rather
        // than on the whole array shape.
        let Some(decoded_chunk_shape) =
            chunk_edge_lengths_to_regular_granularity(decoded_chunk_grid)?
        else {
            return Ok(ChunkGridEncoded::ChunkLocal);
        };
        let Ok(encoded_chunk_shape) = super::get_encoded_shape(&self.shape, &decoded_chunk_shape)
        else {
            return Ok(ChunkGridEncoded::None);
        };
        if encoded_chunk_shape == decoded_chunk_shape {
            return Ok(ChunkGridEncoded::Array(decoded_chunk_grid.clone()));
        }
        let Some(repeats) =
            repeat_shape(decoded_chunk_grid.grid_shape(), encoded_chunk_shape.len())
        else {
            return Ok(ChunkGridEncoded::None);
        };
        repeated_chunk_grid(&encoded_chunk_shape, &repeats).map(ChunkGridEncoded::from)
    }

    fn decoded_subchunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
        encoded_subchunk_grid: ChunkGridEncodedRef<'_>,
    ) -> Result<ChunkGridDecoded, ChunkGridCreateError> {
        let ChunkGridEncodedRef::Array(encoded_subchunk_grid) = encoded_subchunk_grid else {
            return Ok(encoded_subchunk_grid.into());
        };
        let ChunkGridDecodedRef::Array(decoded_chunk_grid) = decoded_chunk_grid else {
            return Ok(ChunkGridDecoded::None);
        };
        if decoded_chunk_grid.array_shape().contains(&0) {
            return Ok(ChunkGridDecoded::None);
        }

        if let Some(partitions) =
            input_dim_partitions(&self.shape, decoded_chunk_grid.dimensionality())
        {
            if input_dim_partitions_are_identity(&partitions) {
                if encoded_subchunk_grid.dimensionality() != partitions.len() {
                    return Err(ChunkGridCreateError::new(format!(
                        "encoded subchunk grid dimensionality {} is incompatible with encoded dimensionality {}",
                        encoded_subchunk_grid.dimensionality(),
                        partitions.len()
                    )));
                }
                return Ok(ChunkGridDecoded::Array(encoded_subchunk_grid.clone()));
            }
            return Ok(decoded_subchunk_grid_for_input_dim_partitions(
                &partitions,
                decoded_chunk_grid,
                encoded_subchunk_grid,
            )?
            .map_or(ChunkGridDecoded::None, ChunkGridDecoded::Array));
        }

        let Some(decoded_chunk_shape) =
            chunk_edge_lengths_to_regular_granularity(decoded_chunk_grid)?
        else {
            return Ok(ChunkGridDecoded::None);
        };
        let encoded_chunk_shape = super::get_encoded_shape(&self.shape, &decoded_chunk_shape)
            .map_err(|err| ChunkGridCreateError::new(err.to_string()))?;
        if encoded_subchunk_grid.dimensionality() != encoded_chunk_shape.len() {
            return Err(ChunkGridCreateError::new(format!(
                "encoded subchunk grid dimensionality {} is incompatible with encoded dimensionality {}",
                encoded_subchunk_grid.dimensionality(),
                encoded_chunk_shape.len()
            )));
        }
        if encoded_chunk_shape == decoded_chunk_shape {
            return Ok(ChunkGridDecoded::Array(encoded_subchunk_grid.clone()));
        }
        let Some(repeats) =
            repeat_shape(decoded_chunk_grid.grid_shape(), encoded_chunk_shape.len())
        else {
            return Ok(ChunkGridDecoded::None);
        };

        let mut local_encoded_edges = Vec::with_capacity(encoded_chunk_shape.len());
        for (dim, (chunk_edge, repeat)) in encoded_chunk_shape.iter().zip(&repeats).enumerate() {
            let outer_edges = vec![
                *chunk_edge;
                usize::try_from(*repeat).map_err(|err| {
                    ChunkGridCreateError::new(err.to_string())
                })?
            ];
            let refined_edges = encoded_subchunk_grid.chunk_edge_lengths(dim)?;
            let Some(refined_tiles) = split_edge_refinements(&outer_edges, &refined_edges) else {
                return Ok(ChunkGridDecoded::None);
            };
            let Some(first) = refined_tiles.first() else {
                return Ok(ChunkGridDecoded::None);
            };
            if refined_tiles.iter().any(|tile| tile != first) {
                return Ok(ChunkGridDecoded::None);
            }
            local_encoded_edges.push(ChunkEdgeLengths::encode(first));
        }
        let local_encoded_grid = ChunkGrid::new(RectilinearChunkGrid::new(
            encoded_chunk_shape.iter().map(|dim| dim.get()).collect(),
            &local_encoded_edges,
        )?);
        let Some(intervals) =
            linear_intervals_from_rectilinear_grid(&encoded_chunk_shape, &local_encoded_grid)?
        else {
            return Ok(ChunkGridDecoded::None);
        };
        let Some(local_decoded_edges) =
            decoded_edge_lengths_from_linear_intervals(&decoded_chunk_shape, &intervals)
        else {
            return Ok(ChunkGridDecoded::None);
        };
        let chunk_shapes = local_decoded_edges
            .into_iter()
            .zip(decoded_chunk_grid.grid_shape())
            .map(|(edges, repeat)| {
                let repeat = usize::try_from(*repeat).ok()?;
                let mut repeated = Vec::with_capacity(edges.len().checked_mul(repeat)?);
                for _ in 0..repeat {
                    repeated.extend_from_slice(&edges);
                }
                Some(ChunkEdgeLengths::encode(&repeated))
            })
            .collect::<Option<Vec<_>>>();
        let Some(chunk_shapes) = chunk_shapes else {
            return Ok(ChunkGridDecoded::None);
        };
        Ok(ChunkGridDecoded::Array(ChunkGrid::new(
            RectilinearChunkGrid::new(decoded_chunk_grid.array_shape().to_vec(), &chunk_shapes)?,
        )))
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        let encoded_shape = super::get_encoded_shape(&self.shape, shape)?;
        Ok(Arc::new(
            super::reshape_codec_partial::ReshapeCodecPartial::new(
                input_handle,
                shape,
                &self.data_type,
                &self.fill_value,
                encoded_shape,
            ),
        ))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        let encoded_shape = super::get_encoded_shape(&self.shape, shape)?;
        Ok(Arc::new(
            super::reshape_codec_partial::ReshapeCodecPartial::new(
                input_output_handle,
                shape,
                &self.data_type,
                &self.fill_value,
                encoded_shape,
            ),
        ))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        let encoded_shape = super::get_encoded_shape(&self.shape, shape)?;
        Ok(Arc::new(
            super::reshape_codec_partial::ReshapeCodecPartial::new(
                input_handle,
                shape,
                &self.data_type,
                &self.fill_value,
                encoded_shape,
            ),
        ))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        let encoded_shape = super::get_encoded_shape(&self.shape, shape)?;
        Ok(Arc::new(
            super::reshape_codec_partial::ReshapeCodecPartial::new(
                input_output_handle,
                shape,
                &self.data_type,
                &self.fill_value,
                encoded_shape,
            ),
        ))
    }
}
