use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_plugin::ZarrVersion;

use super::reshape_codec_grid_mapping::reshape_rectilinear_grid;
use crate::array::{
    ArrayIndices, ArraySubset, ChunkGrid, ChunkShape, DataType, FillValue, ravel_indices,
    unravel_index,
};
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
use zarrs_metadata_ext::codec::reshape::{
    ReshapeCodecConfiguration, ReshapeCodecConfigurationV1, ReshapeShape,
};
use zarrs_plugin::PluginCreateError;

use super::reshape_codec_grid_mapping::{decoded_subchunk_grid, encoded_chunk_grid};

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
    fill_value: Option<FillValue>,
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

impl ReshapeCodecBound {
    fn new_partial<T: ?Sized>(
        &self,
        input_handle: Arc<T>,
        decoded_shape: &[NonZeroU64],
    ) -> Result<super::reshape_codec_partial::ReshapeCodecPartial<T>, CodecError> {
        let encoded_shape = super::get_encoded_shape(&self.shape, decoded_shape)?;
        Ok(super::reshape_codec_partial::ReshapeCodecPartial::new(
            input_handle,
            decoded_shape,
            &self.data_type,
            encoded_shape,
        ))
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
        fill_value: Option<FillValue>,
        _codec_specific_options: &CodecSpecificOptions,
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

impl zarrs_codec::ArrayToArrayCodecSubchunkingTraits for ReshapeCodecBound {
    fn encoded_chunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridEncoded, ChunkGridCreateError> {
        encoded_chunk_grid(&self.shape, decoded_chunk_grid)
    }

    fn decoded_subchunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
        encoded_subchunk_grid: ChunkGridEncodedRef<'_>,
    ) -> Result<ChunkGridDecoded, ChunkGridCreateError> {
        decoded_subchunk_grid(&self.shape, decoded_chunk_grid, encoded_subchunk_grid)
    }

    fn encoded_coordinates(
        &self,
        decoded_shape: &[NonZeroU64],
        decoded_coordinates: &[u64],
    ) -> Result<Option<ArrayIndices>, CodecError> {
        zarrs_codec::validate_chunk_coordinates(decoded_shape, decoded_coordinates)?;
        let encoded_shape = self.encoded_shape(decoded_shape)?;
        Ok(Some(reshape_coordinates(
            decoded_shape,
            &encoded_shape,
            decoded_coordinates,
        )))
    }

    fn decoded_local_subchunk_grid(
        &self,
        decoded_shape: &[NonZeroU64],
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        reshape_rectilinear_grid(
            &self.encoded_shape(decoded_shape)?,
            decoded_shape,
            encoded_subchunk_grid,
        )
        .map_err(|err| CodecError::Other(err.to_string()))
    }

    fn encoded_subset(
        &self,
        decoded_shape: &[NonZeroU64],
        decoded_subset: &ArraySubset,
    ) -> Result<Option<ArraySubset>, CodecError> {
        zarrs_codec::validate_chunk_subset(decoded_shape, decoded_subset)?;
        let encoded_shape = self.encoded_shape(decoded_shape)?;
        let last = decoded_subset
            .end_inc()
            .expect("a validated subset is nonempty");
        let start = reshape_coordinates(decoded_shape, &encoded_shape, decoded_subset.start());
        let last = reshape_coordinates(decoded_shape, &encoded_shape, &last);
        if std::iter::zip(&start, &last).any(|(start, last)| last < start) {
            return Ok(None);
        }
        let candidate =
            ArraySubset::from(std::iter::zip(start, last).map(|(start, last)| start..last + 1));
        if candidate.num_elements() != decoded_subset.num_elements() {
            return Ok(None);
        }
        // The corners alone do not determine the elements, so compare the selected linear indices.
        // Runs are maximal, so equal selections yield equal runs.
        let decoded_runs = decoded_subset
            .contiguous_linearised_indices(bytemuck::must_cast_slice(decoded_shape))?;
        let encoded_runs =
            candidate.contiguous_linearised_indices(bytemuck::must_cast_slice(&encoded_shape))?;
        Ok(decoded_runs
            .into_iter()
            .eq(encoded_runs)
            .then_some(candidate))
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

    fn encoded_fill_value(&self) -> Option<&FillValue> {
        self.fill_value.as_ref()
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        super::get_encoded_shape(&self.shape, decoded_shape)
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
        Ok(Arc::new(self.new_partial(input_handle, shape)?))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(self.new_partial(input_output_handle, shape)?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(self.new_partial(input_handle, shape)?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(self.new_partial(input_output_handle, shape)?))
    }
}

/// Map in-bounds `coordinates` of a chunk of `decoded_shape` to a chunk of `encoded_shape`, preserving C order.
fn reshape_coordinates(
    decoded_shape: &[NonZeroU64],
    encoded_shape: &[NonZeroU64],
    coordinates: &[u64],
) -> ArrayIndices {
    let index = ravel_indices(coordinates, bytemuck::must_cast_slice(decoded_shape))
        .expect("coordinates are in bounds");
    unravel_index(index, bytemuck::must_cast_slice(encoded_shape))
        .expect("reshaping preserves the number of elements")
}
