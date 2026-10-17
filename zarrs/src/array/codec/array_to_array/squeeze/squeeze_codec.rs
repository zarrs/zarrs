use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs_codec::plan::{ArrayToArrayDecodePlan, DecodeRequest, PlanIndexer};

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_plugin::ZarrVersion;

use crate::array::chunk_grid::{ChunkEdgeLengths, RectilinearChunkGrid};
use crate::array::{ArrayIndices, ChunkGrid, ChunkShape, DataType, FillValue};
use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecTraits, ChunkGridDecoded, ChunkGridDecodedRef,
    ChunkGridEncoded, ChunkGridEncodedRef, CodecCreateError, CodecError, CodecMetadataOptions,
    CodecOptions, CodecSpecificOptions, CodecTraits, Resources, UnboundArrayToArrayCodecTraits,
};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::squeeze::{SqueezeCodecConfiguration, SqueezeCodecConfigurationV0};
use zarrs_plugin::PluginCreateError;

/// A Squeeze codec implementation.
// #[deprecated(since = "0.24.0", note = "The squeeze codec is superseded by the reshape codec")]
#[derive(Clone, Debug)]
pub struct SqueezeCodec {}

/// A Squeeze codec implementation bound to a data type and fill value.
#[derive(Clone, Debug)]
struct SqueezeCodecBound {
    data_type: DataType,
    fill_value: FillValue,
}

impl SqueezeCodec {
    /// Create a new squeeze codec from configuration.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if there is a configuration issue.
    pub fn new_with_configuration(
        configuration: &SqueezeCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            SqueezeCodecConfiguration::V0(_configuration) => Ok(Self::new()),
            _ => Err(PluginCreateError::Other(
                "this squeeze codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    /// Create a new squeeze codec.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }
}

fn squeeze_chunk_edge_lengths(edge_lengths: &[NonZeroU64]) -> Option<bool> {
    let mut has_unit_edge = false;
    let mut has_non_unit_edge = false;
    for edge_length in edge_lengths {
        if edge_length.get() == 1 {
            has_unit_edge = true;
        } else {
            has_non_unit_edge = true;
        }
    }

    match (has_unit_edge, has_non_unit_edge) {
        (true, false) => Some(true),
        (false, true) => Some(false),
        // The encoded dimensionality would vary across chunks.
        (true, true) | (false, false) => None,
    }
}

impl Default for SqueezeCodec {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a chunk-local subchunk grid of the encoded chunk to the decoded chunk of `shape`.
pub(super) fn decoded_local_subchunk_grid(
    shape: &[NonZeroU64],
    encoded_subchunk_grid: &ChunkGrid,
) -> Result<ChunkGrid, CodecError> {
    let expected_dimensionality = shape.iter().filter(|dim| dim.get() > 1).count().max(1);
    if encoded_subchunk_grid.dimensionality() != expected_dimensionality {
        return Err(CodecError::Other(
                "local subchunk grid dimensionality is incompatible with squeeze encoded dimensionality"
                    .to_string(),
            ));
    }

    let mut encoded_dim = 0;
    let chunk_shapes = shape
        .iter()
        .map(|dim| {
            if dim.get() == 1 {
                Ok(ChunkEdgeLengths::Scalar(NonZeroU64::new(1).unwrap()))
            } else {
                let edge_lengths = encoded_subchunk_grid.chunk_edge_lengths(encoded_dim)?;
                encoded_dim += 1;
                Ok(ChunkEdgeLengths::encode(&edge_lengths))
            }
        })
        .collect::<Result<Vec<_>, zarrs_chunk_grid::ChunkGridCreateError>>()
        .map_err(|err| CodecError::Other(err.to_string()))?;
    let array_shape = bytemuck::must_cast_slice(shape).to_vec();
    Ok(ChunkGrid::new(
        RectilinearChunkGrid::new(array_shape, &chunk_shapes)
            .map_err(|err| CodecError::Other(err.to_string()))?,
    ))
}

impl CodecTraits for SqueezeCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = SqueezeCodecConfiguration::V0(SqueezeCodecConfigurationV0 {});
        Some(configuration.into())
    }
}

impl UnboundArrayToArrayCodecTraits for SqueezeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self as Arc<dyn UnboundArrayToArrayCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        Ok(Arc::new(SqueezeCodecBound {
            data_type,
            fill_value,
        }))
    }
}

impl ArrayCodecTraits for SqueezeCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }
}

impl zarrs_codec::ArrayToArrayCodecSubchunkingTraits for SqueezeCodecBound {
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

        let mut array_shape = Vec::new();
        let mut chunk_shapes = Vec::new();
        for decoded_dim in 0..decoded_chunk_grid.dimensionality() {
            let edge_lengths = decoded_chunk_grid
                .chunk_edge_lengths(decoded_dim)
                .map_err(ChunkGridCreateError::from)?;
            let Some(squeeze) = squeeze_chunk_edge_lengths(&edge_lengths) else {
                return Ok(ChunkGridEncoded::ChunkLocal);
            };
            if !squeeze {
                let edge_lengths = decoded_chunk_grid
                    .chunk_edge_lengths(decoded_dim)
                    .map_err(ChunkGridCreateError::from)?;
                array_shape.push(decoded_chunk_grid.array_shape()[decoded_dim]);
                chunk_shapes.push(ChunkEdgeLengths::encode(&edge_lengths));
            }
        }

        let chunk_shapes = if chunk_shapes.is_empty() {
            array_shape.push(1);
            vec![ChunkEdgeLengths::Scalar(NonZeroU64::new(1).unwrap())]
        } else {
            chunk_shapes
        };

        Ok(ChunkGridEncoded::Array(ChunkGrid::new(
            RectilinearChunkGrid::new(array_shape, &chunk_shapes)?,
        )))
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

        let mut squeeze = Vec::with_capacity(decoded_chunk_grid.dimensionality());
        for decoded_dim in 0..decoded_chunk_grid.dimensionality() {
            let edge_lengths = decoded_chunk_grid
                .chunk_edge_lengths(decoded_dim)
                .map_err(ChunkGridCreateError::from)?;
            let Some(squeeze_dim) = squeeze_chunk_edge_lengths(&edge_lengths) else {
                return Ok(ChunkGridDecoded::None);
            };
            squeeze.push(squeeze_dim);
        }

        let num_unsqueezed_dims = squeeze.iter().filter(|&&squeeze| !squeeze).count();
        let expected_encoded_dimensionality = num_unsqueezed_dims.max(1);
        if encoded_subchunk_grid.dimensionality() != expected_encoded_dimensionality {
            return Err(ChunkGridCreateError::new(format!(
                "encoded subchunk grid dimensionality {} is incompatible with squeezed dimensionality {}",
                encoded_subchunk_grid.dimensionality(),
                expected_encoded_dimensionality
            )));
        }

        let mut encoded_dim = 0usize;
        let chunk_shapes = squeeze
            .iter()
            .map(|&squeeze_dim| {
                if squeeze_dim {
                    Ok(ChunkEdgeLengths::Scalar(NonZeroU64::new(1).unwrap()))
                } else {
                    let edge_lengths = encoded_subchunk_grid
                        .chunk_edge_lengths(encoded_dim)
                        .map_err(ChunkGridCreateError::from)?;
                    encoded_dim += 1;
                    Ok(ChunkEdgeLengths::encode(&edge_lengths))
                }
            })
            .collect::<Result<Vec<_>, ChunkGridCreateError>>()?;

        Ok(ChunkGridDecoded::Array(ChunkGrid::new(
            RectilinearChunkGrid::new(decoded_chunk_grid.array_shape().to_vec(), &chunk_shapes)?,
        )))
    }

    fn encoded_coordinates(
        &self,
        decoded_shape: &[NonZeroU64],
        decoded_coordinates: &[u64],
    ) -> Result<ArrayIndices, CodecError> {
        // Squeezed dimensions have a length of one, so their coordinates are zero
        let encoded_coordinates: ArrayIndices = std::iter::zip(decoded_coordinates, decoded_shape)
            .filter(|(_, length)| length.get() > 1)
            .map(|(&coordinate, _)| coordinate)
            .collect();
        if encoded_coordinates.is_empty() {
            Ok(vec![0])
        } else {
            Ok(encoded_coordinates)
        }
    }
}

impl ArrayToArrayCodecTraits for SqueezeCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn decoded_local_subchunk_grid(
        &self,
        shape: &[NonZeroU64],
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        decoded_local_subchunk_grid(shape, encoded_subchunk_grid).map(Some)
    }

    fn partial_encode_request<'a>(
        &self,
        shape: &[NonZeroU64],
        indexer: PlanIndexer<'a>,
        bytes: ArrayBytes<'a>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<(PlanIndexer<'a>, ArrayBytes<'a>)>, CodecError> {
        let _ = (options, resources);
        // The squeezed indexer selects the same elements in the same order
        let indexer = indexer.as_indexer();
        let indexer = if let Some(subset) = indexer.as_array_subset() {
            PlanIndexer::Subset(super::get_squeezed_array_subset(subset, shape)?)
        } else {
            PlanIndexer::Shared(Arc::new(super::get_squeezed_indexer(indexer, shape)?))
        };
        Ok(Some((indexer, bytes)))
    }

    fn plans_partial_encode(&self) -> bool {
        true
    }

    fn plan_decode_request<'a>(
        &self,
        shape: &[NonZeroU64],
        request: DecodeRequest<'a>,
    ) -> Result<ArrayToArrayDecodePlan<'a>, CodecError> {
        // The squeezed indexer selects the same elements in the same order
        let indexer = request.indexer.as_indexer();
        let indexer = if let Some(subset) = indexer.as_array_subset() {
            PlanIndexer::Subset(super::get_squeezed_array_subset(subset, shape)?)
        } else {
            PlanIndexer::Shared(Arc::new(super::get_squeezed_indexer(indexer, shape)?))
        };
        Ok(ArrayToArrayDecodePlan::Forward(DecodeRequest {
            indexer,
            target: request.target,
        }))
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.data_type
    }

    fn encoded_fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        let encoded_shape: Vec<_> = decoded_shape
            .iter()
            .filter(|dim| dim.get() > 1)
            .copied()
            .collect();
        if encoded_shape.is_empty() {
            Ok(vec![NonZeroU64::new(1).unwrap()])
        } else {
            Ok(encoded_shape)
        }
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }
}
