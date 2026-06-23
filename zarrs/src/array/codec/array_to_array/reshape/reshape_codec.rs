// TODO: reshape partial decoder

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
    UnboundArrayToArrayCodecTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::reshape::{
    ReshapeCodecConfiguration, ReshapeCodecConfigurationV1, ReshapeShape,
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

/// Return the row-major linear length of an encoded granule if every encoded
/// granule is contiguous in linear storage order.
///
/// Returns `None` if the granularity does not tile the encoded shape or if an
/// encoded granule spans a non-contiguous rectangle in row-major order.
fn encoded_granularity_linear_interval_len(
    encoded_shape: &[NonZeroU64],
    encoded_granularity: &[NonZeroU64],
) -> Option<u64> {
    for (shape, granularity) in encoded_shape.iter().zip(encoded_granularity) {
        if granularity.get() > shape.get() || !shape.get().is_multiple_of(granularity.get()) {
            return None;
        }
    }

    let Some(first_non_unit_axis) = encoded_granularity
        .iter()
        .position(|granularity| granularity.get() > 1)
    else {
        return Some(1);
    };

    let mut interval_len = 1u64;
    for (axis, (shape, granularity)) in encoded_shape
        .iter()
        .zip(encoded_granularity)
        .enumerate()
        .skip(first_non_unit_axis)
    {
        if axis > first_non_unit_axis && granularity != shape {
            return None;
        }
        interval_len = interval_len.checked_mul(granularity.get())?;
    }

    Some(interval_len)
}

/// Convert a row-major linear interval length into a decoded rectangular
/// granularity.
///
/// Returns `None` if the interval cannot tile `decoded_shape` as regular
/// row-major rectangles.
fn decoded_granularity_from_linear_interval(
    decoded_shape: &[NonZeroU64],
    interval_len: u64,
) -> Option<ChunkShape> {
    if interval_len == 0 {
        return None;
    }

    let num_elements = decoded_shape
        .iter()
        .try_fold(1u64, |product, dim| product.checked_mul(dim.get()))?;
    if interval_len > num_elements || !num_elements.is_multiple_of(interval_len) {
        return None;
    }
    if interval_len == num_elements {
        return Some(decoded_shape.to_vec());
    }

    let mut granularity = vec![NonZeroU64::new(1).unwrap(); decoded_shape.len()];
    let mut remaining = interval_len;
    for (dim, decoded_dim) in decoded_shape.iter().enumerate().rev() {
        if remaining == 1 {
            break;
        }

        let decoded_dim = decoded_dim.get();
        if remaining >= decoded_dim {
            if !remaining.is_multiple_of(decoded_dim) {
                return None;
            }
            granularity[dim] = NonZeroU64::new(decoded_dim).unwrap();
            remaining /= decoded_dim;
        } else {
            if !decoded_dim.is_multiple_of(remaining) {
                return None;
            }
            granularity[dim] = NonZeroU64::new(remaining).unwrap();
            remaining = 1;
        }
    }

    (remaining == 1).then_some(granularity)
}

fn chunk_edge_lengths_to_regular_granularity(
    chunk_grid: &ChunkGrid,
) -> Result<Option<ChunkShape>, ChunkGridCreateError> {
    let mut chunk_shape = Vec::with_capacity(chunk_grid.dimensionality());
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
        chunk_shape.push(first);
    }
    Ok(Some(chunk_shape))
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

    fn partial_decode_granularity(
        &self,
        decoded_shape: &[NonZeroU64],
        encoded_granularity: &[NonZeroU64],
    ) -> Result<ChunkShape, CodecError> {
        let encoded_shape = super::get_encoded_shape(&self.shape, decoded_shape)?;
        if encoded_granularity.len() != encoded_shape.len() {
            return Err(CodecError::Other(format!(
                "encoded granularity dimensionality {} is incompatible with encoded dimensionality {}",
                encoded_granularity.len(),
                encoded_shape.len()
            )));
        }

        if encoded_shape == decoded_shape {
            return Ok(encoded_granularity.to_vec());
        }

        // Reshape preserves row-major linear order, so a decoded rectangular
        // granularity can be inferred only when each encoded granule is a
        // contiguous linear interval that also tiles decoded row-major
        // coordinates as a rectangle. Otherwise, use the full decoded chunk.
        let Some(interval_len) =
            encoded_granularity_linear_interval_len(&encoded_shape, encoded_granularity)
        else {
            return Ok(decoded_shape.to_vec());
        };
        Ok(
            decoded_granularity_from_linear_interval(decoded_shape, interval_len)
                .unwrap_or_else(|| decoded_shape.to_vec()),
        )
    }

    fn encoded_chunk_grid(
        &self,
        _decoded_chunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
        Ok(None)
    }

    fn decoded_subchunk_grid(
        &self,
        decoded_chunk_grid: &ChunkGrid,
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
        let decoded_shape = decoded_chunk_grid
            .array_shape()
            .iter()
            .copied()
            .map(NonZeroU64::new)
            .collect::<Option<ChunkShape>>();
        let Some(decoded_shape) = decoded_shape else {
            return Ok(None);
        };

        let encoded_shape = super::get_encoded_shape(&self.shape, &decoded_shape)
            .map_err(|err| ChunkGridCreateError::new(err.to_string()))?;
        if encoded_subchunk_grid.dimensionality() != encoded_shape.len() {
            return Err(ChunkGridCreateError::new(format!(
                "encoded subchunk grid dimensionality {} is incompatible with encoded dimensionality {}",
                encoded_subchunk_grid.dimensionality(),
                encoded_shape.len()
            )));
        }

        let Some(encoded_granularity) =
            chunk_edge_lengths_to_regular_granularity(encoded_subchunk_grid)?
        else {
            return Ok(None);
        };

        let decoded_granularity = if encoded_shape == decoded_shape {
            encoded_granularity
        } else if let Some(interval_len) =
            encoded_granularity_linear_interval_len(&encoded_shape, &encoded_granularity)
        {
            decoded_granularity_from_linear_interval(&decoded_shape, interval_len)
                .unwrap_or_else(|| decoded_shape.clone())
        } else {
            decoded_shape.clone()
        };

        let chunk_shapes = decoded_shape
            .iter()
            .zip(&decoded_granularity)
            .map(|(shape, granularity)| {
                if !shape.get().is_multiple_of(granularity.get()) {
                    return None;
                }
                let edge_lengths =
                    vec![*granularity; (shape.get() / granularity.get()) as usize];
                Some(edge_lengths_to_chunk_edge_lengths(&edge_lengths))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                ChunkGridCreateError::new(format!(
                    "decoded granularity {decoded_granularity:?} is incompatible with decoded shape {decoded_shape:?}"
                ))
            })?;

        let chunk_shapes = if chunk_shapes.is_empty() {
            vec![ChunkEdgeLengths::Scalar(NonZeroU64::new(1).unwrap())]
        } else {
            chunk_shapes
        };

        Ok(Some(ChunkGrid::new(RectilinearChunkGrid::new(
            decoded_chunk_grid.array_shape().to_vec(),
            &chunk_shapes,
        )?)))
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
