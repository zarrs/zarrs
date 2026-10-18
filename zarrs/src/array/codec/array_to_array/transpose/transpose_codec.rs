use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs_codec::plan::{ArrayToArrayDecodePlan, DecodeRequest, PlanIndexer};

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_plugin::ZarrVersion;

use super::{
    TransposeCodecConfiguration, TransposeOrder, apply_permutation, inverse_permutation, permute,
};
use crate::array::chunk_grid::{ChunkEdgeLengths, RectilinearChunkGrid};
use crate::array::{ArrayBytes, ArrayIndices, ChunkGrid, ChunkShape, DataType, FillValue};
use zarrs_codec::{
    ArrayCodecTraits, ArrayToArrayCodecTraits, ChunkGridDecoded, ChunkGridDecodedRef,
    ChunkGridEncoded, ChunkGridEncodedRef, CodecCreateError, CodecError, CodecMetadataOptions,
    CodecOptions, CodecSpecificOptions, CodecTraits, ComputeResources,
    UnboundArrayToArrayCodecTraits,
};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::transpose::TransposeCodecConfigurationV1;
use zarrs_plugin::PluginCreateError;

/// A Transpose codec implementation.
#[derive(Clone, Debug)]
pub struct TransposeCodec {
    pub(crate) order: TransposeOrder,
}

/// A Transpose codec implementation bound to a data type and fill value.
#[derive(Clone, Debug)]
struct TransposeCodecBound {
    order: TransposeOrder,
    data_type: DataType,
    fill_value: FillValue,
}

impl TransposeCodec {
    /// Create a new transpose codec from configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if there is a configuration issue.
    pub fn new_with_configuration(
        configuration: &TransposeCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            TransposeCodecConfiguration::V1(configuration) => {
                Ok(Self::new(configuration.order.clone()))
            }
            _ => Err(PluginCreateError::Other(
                "this transpose codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    /// Create a new transpose codec.
    #[must_use]
    pub const fn new(order: TransposeOrder) -> Self {
        Self { order }
    }
}

impl TransposeCodecBound {
    /// Validate the shape and data type for this codec.
    fn validate(&self, shape: &[NonZeroU64]) -> Result<(), CodecError> {
        if self.order.0.len() != shape.len() {
            return Err(CodecError::Other(
                "Length of transpose codec `order` does not match array dimensionality".to_string(),
            ));
        }
        Ok(())
    }
}

/// Map a chunk-local subchunk grid of the encoded chunk to the decoded chunk of `shape`.
pub(super) fn decoded_local_subchunk_grid(
    shape: &[NonZeroU64],
    order_inverse: &[usize],
    encoded_subchunk_grid: &ChunkGrid,
) -> Result<ChunkGrid, CodecError> {
    if order_inverse.len() != encoded_subchunk_grid.dimensionality() {
        return Err(CodecError::Other(
            "Length of transpose codec `order` does not match local subchunk grid dimensionality"
                .to_string(),
        ));
    }

    let chunk_shapes = order_inverse
        .iter()
        .map(|&encoded_dim| {
            let edge_lengths = encoded_subchunk_grid.chunk_edge_lengths(encoded_dim)?;
            Ok(ChunkEdgeLengths::encode(&edge_lengths))
        })
        .collect::<Result<Vec<_>, zarrs_chunk_grid::ChunkGridCreateError>>()
        .map_err(|err| CodecError::Other(err.to_string()))?;
    let array_shape = bytemuck::must_cast_slice(shape).to_vec();
    Ok(ChunkGrid::new(
        RectilinearChunkGrid::new(array_shape, &chunk_shapes)
            .map_err(|err| CodecError::Other(err.to_string()))?,
    ))
}

impl CodecTraits for TransposeCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = TransposeCodecConfiguration::V1(TransposeCodecConfigurationV1 {
            order: self.order.clone(),
        });
        Some(configuration.into())
    }
}

impl UnboundArrayToArrayCodecTraits for TransposeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self as Arc<dyn UnboundArrayToArrayCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        Ok(Arc::new(TransposeCodecBound {
            order: self.order.clone(),
            data_type,
            fill_value,
        }))
    }
}

impl ArrayCodecTraits for TransposeCodecBound {
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

impl zarrs_codec::ArrayToArrayCodecSubchunkingTraits for TransposeCodecBound {
    fn encoded_chunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridEncoded, ChunkGridCreateError> {
        let ChunkGridDecodedRef::Array(decoded_chunk_grid) = decoded_chunk_grid else {
            return Ok(decoded_chunk_grid.into());
        };
        let Some(array_shape) = permute(decoded_chunk_grid.array_shape(), &self.order.0) else {
            return Err(ChunkGridCreateError::new(
                "Length of transpose codec `order` does not match chunk grid dimensionality",
            ));
        };
        let chunk_shapes = self
            .order
            .0
            .iter()
            .map(|&decoded_dim| {
                let edge_lengths = decoded_chunk_grid.chunk_edge_lengths(decoded_dim)?;
                Ok(ChunkEdgeLengths::encode(&edge_lengths))
            })
            .collect::<Result<Vec<_>, ChunkGridCreateError>>()?;

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
        if self.order.0.len() != decoded_chunk_grid.dimensionality() {
            return Err(ChunkGridCreateError::new(
                "Length of transpose codec `order` does not match `decoded_chunk_grid` dimensionality",
            ));
        } else if self.order.0.len() != encoded_subchunk_grid.dimensionality() {
            return Err(ChunkGridCreateError::new(
                "Length of transpose codec `order` does not match `encoded_subchunk_grid` dimensionality",
            ));
        }

        let inverse = inverse_permutation(&self.order.0);
        let chunk_shapes = inverse
            .iter()
            .map(|&encoded_dim| {
                let edge_lengths = encoded_subchunk_grid.chunk_edge_lengths(encoded_dim)?;
                Ok(ChunkEdgeLengths::encode(&edge_lengths))
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
        let _ = decoded_shape;
        permute(decoded_coordinates, &self.order.0).ok_or_else(|| {
            CodecError::Other(
                "Length of transpose codec `order` does not match array dimensionality".to_string(),
            )
        })
    }
}

impl ArrayToArrayCodecTraits for TransposeCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn decoded_local_subchunk_grid(
        &self,
        shape: &[NonZeroU64],
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        decoded_local_subchunk_grid(
            shape,
            &inverse_permutation(&self.order.0),
            encoded_subchunk_grid,
        )
        .map(Some)
    }

    fn partial_encode_request<'a>(
        &self,
        shape: &[NonZeroU64],
        indexer: PlanIndexer<'a>,
        bytes: ArrayBytes<'a>,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<Option<(PlanIndexer<'a>, ArrayBytes<'a>)>, CodecError> {
        let _ = (options, resources);
        let indexer = indexer.as_indexer();
        if let Some(subset) = indexer.as_array_subset() {
            // Transpose the region, which updates the transposed region
            let transposed = super::get_transposed_array_subset(&self.order.0, shape, subset)?;
            let bytes = apply_permutation(&bytes, &subset.shape(), &self.order.0, &self.data_type)?;
            Ok(Some((PlanIndexer::Subset(transposed), bytes)))
        } else {
            // The transposed indices select the same elements in the same order
            let transposed = super::get_transposed_indexer(&self.order.0, shape, indexer)?;
            Ok(Some((PlanIndexer::Shared(Arc::new(transposed)), bytes)))
        }
    }

    fn plans_partial_encode(&self) -> bool {
        true
    }

    fn plan_decode_request<'a>(
        &self,
        shape: &[NonZeroU64],
        request: DecodeRequest<'a>,
    ) -> Result<ArrayToArrayDecodePlan<'a>, CodecError> {
        let indexer = request.indexer.as_indexer();
        if let Some(subset) = indexer.as_array_subset() {
            // Decode the transposed region, then transpose it
            let transposed = super::get_transposed_array_subset(&self.order.0, shape, subset)?;
            let region_shape: Option<Vec<NonZeroU64>> = subset
                .shape()
                .iter()
                .map(|&size| NonZeroU64::new(size))
                .collect();
            Ok(match region_shape {
                Some(region_shape) => ArrayToArrayDecodePlan::Decode {
                    indexer: PlanIndexer::Subset(transposed),
                    shape: region_shape,
                    target: request.target,
                },
                None => ArrayToArrayDecodePlan::Forward(DecodeRequest {
                    indexer: PlanIndexer::Subset(transposed),
                    target: request.target,
                }),
            })
        } else {
            // The transposed indices select the same elements in the same order
            let transposed = super::get_transposed_indexer(&self.order.0, shape, indexer)?;
            Ok(ArrayToArrayDecodePlan::Forward(DecodeRequest {
                indexer: PlanIndexer::Shared(Arc::new(transposed)),
                target: request.target,
            }))
        }
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.data_type
    }

    fn encoded_fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        if self.order.0.len() != decoded_shape.len() {
            return Err(CodecError::Other(
                "Length of transpose codec `order` does not match array dimensionality".to_string(),
            ));
        }
        Ok(permute(decoded_shape, &self.order.0).expect("matching dimensionality"))
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.validate(shape)?;

        // Encode: apply the transpose order to the decoded shape
        let shape_u64 = bytemuck::must_cast_slice(shape);
        apply_permutation(&bytes, shape_u64, &self.order.0, &self.data_type)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.validate(shape)?;

        // Decode: apply the inverse permutation to the encoded (transposed) shape
        let shape_u64 = bytemuck::must_cast_slice(shape);
        let transposed_shape: ArrayIndices = permute(shape_u64, &self.order.0).expect("validated");
        apply_permutation(
            &bytes,
            &transposed_shape,
            &inverse_permutation(&self.order.0),
            &self.data_type,
        )
    }
}
