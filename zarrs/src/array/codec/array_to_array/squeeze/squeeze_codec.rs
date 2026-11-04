use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_plugin::ZarrVersion;

use crate::array::chunk_grid::{ChunkEdgeLengths, RectilinearChunkGrid};
use crate::array::{ArrayIndices, ChunkGrid, ChunkShape, DataType, FillValue};
use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArrayToArrayCodecTraits, ChunkGridDecoded, ChunkGridDecodedRef, ChunkGridEncoded,
    ChunkGridEncodedRef, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecSpecificOptions, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
    RecommendedConcurrency, UnboundArrayToArrayCodecTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
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
    fill_value: Option<FillValue>,
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
        fill_value: Option<FillValue>,
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

    fn fill_value(&self) -> Option<&FillValue> {
        self.fill_value.as_ref()
    }

    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
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

        Ok(ChunkGridDecoded::Array(decoded_local_subchunk_grid(
            decoded_chunk_grid.array_shape().to_vec(),
            &squeeze,
            encoded_subchunk_grid,
        )?))
    }

    fn encoded_element_indices(
        &self,
        decoded_shape: &[NonZeroU64],
        decoded_element_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, CodecError> {
        zarrs_codec::validate_chunk_element_indices(decoded_shape, decoded_element_indices)?;
        let encoded_element_indices =
            super::squeeze_dims(decoded_element_indices.iter().copied(), decoded_shape, 0);
        Ok(Some(ArrayIndices::from(encoded_element_indices.as_slice())))
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

    fn encoded_fill_value(&self) -> Option<&FillValue> {
        self.fill_value.as_ref()
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        Ok(super::squeeze_dims(
            decoded_shape.iter().copied(),
            decoded_shape,
            NonZeroU64::MIN,
        ))
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
        Ok(Arc::new(
            super::squeeze_codec_partial::SqueezeCodecPartial::new(
                input_handle,
                shape,
                &self.data_type,
            ),
        ))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(
            super::squeeze_codec_partial::SqueezeCodecPartial::new(
                input_output_handle,
                shape,
                &self.data_type,
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
        Ok(Arc::new(
            super::squeeze_codec_partial::SqueezeCodecPartial::new(
                input_handle,
                shape,
                &self.data_type,
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
        Ok(Arc::new(
            super::squeeze_codec_partial::SqueezeCodecPartial::new(
                input_output_handle,
                shape,
                &self.data_type,
            ),
        ))
    }
}

/// Map the edge lengths of `encoded_subchunk_grid` to a grid over `array_shape`, with unit edges in `squeeze`d dimensions.
pub(super) fn decoded_local_subchunk_grid(
    array_shape: Vec<u64>,
    squeeze: &[bool],
    encoded_subchunk_grid: &ChunkGrid,
) -> Result<ChunkGrid, ChunkGridCreateError> {
    let expected_encoded_dimensionality =
        squeeze.iter().filter(|&&squeeze| !squeeze).count().max(1);
    if encoded_subchunk_grid.dimensionality() != expected_encoded_dimensionality {
        return Err(ChunkGridCreateError::new(format!(
            "encoded subchunk grid dimensionality {} is incompatible with squeezed dimensionality {}",
            encoded_subchunk_grid.dimensionality(),
            expected_encoded_dimensionality
        )));
    }

    let mut encoded_dim = 0;
    let chunk_shapes = squeeze
        .iter()
        .map(|&squeeze| {
            if squeeze {
                Ok(ChunkEdgeLengths::Scalar(NonZeroU64::MIN))
            } else {
                let edge_lengths = encoded_subchunk_grid.chunk_edge_lengths(encoded_dim)?;
                encoded_dim += 1;
                Ok(ChunkEdgeLengths::encode(&edge_lengths))
            }
        })
        .collect::<Result<Vec<_>, ChunkGridCreateError>>()?;
    Ok(ChunkGrid::new(RectilinearChunkGrid::new(
        array_shape,
        &chunk_shapes,
    )?))
}
