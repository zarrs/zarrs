use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_plugin::ZarrVersion;

use crate::array::chunk_grid::{
    ChunkEdgeLengths, RectilinearChunkGrid, edge_lengths_to_chunk_edge_lengths,
};
use crate::array::{ChunkGrid, ChunkShape, DataType, FillValue};
use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArrayToArrayCodecTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecTraits, PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    SubchunkGrid, UnboundArrayToArrayCodecTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::squeeze::{SqueezeCodecConfiguration, SqueezeCodecConfigurationV0};
use zarrs_plugin::PluginCreateError;

/// A Squeeze codec implementation.
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

impl CodecTraits for SqueezeCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = SqueezeCodecConfiguration::V0(SqueezeCodecConfigurationV0 {});
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
impl UnboundArrayToArrayCodecTraits for SqueezeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self as Arc<dyn UnboundArrayToArrayCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
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

    fn recommended_concurrency(
        &self,
        _shape: &[u64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for SqueezeCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.data_type
    }

    fn encoded_fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn encoded_shape(&self, decoded_shape: &[u64]) -> Result<ChunkShape, CodecError> {
        let encoded_shape: Vec<_> = decoded_shape
            .iter()
            .filter(|&&dim| dim > 1)
            .copied()
            .collect();
        if encoded_shape.is_empty() {
            Ok(vec![])
        } else {
            Ok(encoded_shape)
        }
    }

    fn encoded_chunk_grid(
        &self,
        decoded_chunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
        if decoded_chunk_grid.array_shape().contains(&0) {
            return Ok(None);
        }

        let mut array_shape = Vec::new();
        let mut chunk_shapes = Vec::new();
        for decoded_dim in 0..decoded_chunk_grid.dimensionality() {
            let edge_lengths = decoded_chunk_grid
                .chunk_edge_lengths(decoded_dim)
                .map_err(ChunkGridCreateError::from)?;
            let Some(edge_lengths) = edge_lengths else {
                return Ok(None);
            };
            let Some(squeeze) = squeeze_chunk_edge_lengths(&edge_lengths) else {
                return Ok(None);
            };
            if !squeeze {
                let edge_lengths = decoded_chunk_grid
                    .chunk_edge_lengths(decoded_dim)
                    .map_err(ChunkGridCreateError::from)?;
                let Some(edge_lengths) = edge_lengths else {
                    return Ok(None);
                };
                array_shape.push(decoded_chunk_grid.array_shape()[decoded_dim]);
                chunk_shapes.push(edge_lengths_to_chunk_edge_lengths(&edge_lengths));
            }
        }

        let chunk_shapes = if chunk_shapes.is_empty() {
            array_shape.push(1);
            vec![ChunkEdgeLengths::Scalar(NonZeroU64::new(1).unwrap())]
        } else {
            chunk_shapes
        };

        Ok(Some(ChunkGrid::new(RectilinearChunkGrid::new(
            array_shape,
            &chunk_shapes,
        )?)))
    }

    fn decoded_subchunk_grid(
        &self,
        decoded_chunk_grid: &ChunkGrid,
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<SubchunkGrid, ChunkGridCreateError> {
        // Empty chunk grids have no subchunk grid
        if decoded_chunk_grid.array_shape().contains(&0) {
            return Ok(SubchunkGrid::None);
        }

        let mut squeeze = Vec::with_capacity(decoded_chunk_grid.dimensionality());
        for decoded_dim in 0..decoded_chunk_grid.dimensionality() {
            let edge_lengths = decoded_chunk_grid
                .chunk_edge_lengths(decoded_dim)
                .map_err(ChunkGridCreateError::from)?;
            let Some(edge_lengths) = edge_lengths else {
                return Ok(SubchunkGrid::None);
            };
            let Some(squeeze_dim) = squeeze_chunk_edge_lengths(&edge_lengths) else {
                return Ok(SubchunkGrid::None);
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
                    let Some(edge_lengths) = edge_lengths else {
                        return Err(ChunkGridCreateError::new(format!(
                            "encoded subchunk grid edge lengths in dimension {encoded_dim} cannot be represented"
                        )));
                    };
                    encoded_dim += 1;
                    Ok(edge_lengths_to_chunk_edge_lengths(&edge_lengths))
                }
            })
            .collect::<Result<Vec<_>, ChunkGridCreateError>>()?;

        Ok(SubchunkGrid::Array(ChunkGrid::new(
            RectilinearChunkGrid::new(decoded_chunk_grid.array_shape().to_vec(), &chunk_shapes)?,
        )))
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            super::squeeze_codec_partial::SqueezeCodecPartial::new(
                input_handle,
                shape,
                &self.data_type,
                &self.fill_value,
            ),
        ))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(
            super::squeeze_codec_partial::SqueezeCodecPartial::new(
                input_output_handle,
                shape,
                &self.data_type,
                &self.fill_value,
            ),
        ))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            super::squeeze_codec_partial::SqueezeCodecPartial::new(
                input_handle,
                shape,
                &self.data_type,
                &self.fill_value,
            ),
        ))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(
            super::squeeze_codec_partial::SqueezeCodecPartial::new(
                input_output_handle,
                shape,
                &self.data_type,
                &self.fill_value,
            ),
        ))
    }
}
