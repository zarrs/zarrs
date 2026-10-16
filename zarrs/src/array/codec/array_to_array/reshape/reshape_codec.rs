use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs_codec::plan::{ArrayToArrayDecodePlan, DecodeRequest, PlanIndexer};

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_plugin::ZarrVersion;

use crate::array::{ChunkGrid, ChunkShape, DataType, FillValue};
use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecTraits, ChunkGridDecoded, ChunkGridDecodedRef,
    ChunkGridEncoded, ChunkGridEncodedRef, CodecCreateError, CodecError, CodecMetadataOptions,
    CodecOptions, CodecSpecificOptions, CodecTraits, PartialDecoderCapability,
    PartialEncoderCapability, Resources, UnboundArrayToArrayCodecTraits,
};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::reshape::{
    ReshapeCodecConfiguration, ReshapeCodecConfigurationV1, ReshapeShape,
};
use zarrs_plugin::PluginCreateError;

use super::reshape_codec_grid_mapping::{
    decoded_subchunk_grid, encoded_chunk_grid, reshape_rectilinear_grid,
};

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

impl UnboundArrayToArrayCodecTraits for ReshapeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self as Arc<dyn UnboundArrayToArrayCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
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

    fn fill_value(&self) -> &FillValue {
        &self.fill_value
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
}

impl ArrayToArrayCodecTraits for ReshapeCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn decoded_local_subchunk_grid(
        &self,
        shape: &[NonZeroU64],
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        reshape_rectilinear_grid(&self.encoded_shape(shape)?, shape, encoded_subchunk_grid)
            .map_err(|err| CodecError::Other(err.to_string()))
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
        // The reshaped indices select the same elements in the same order
        let encoded_shape = self.encoded_shape(shape)?;
        let indexer = super::get_reshaped_indexer(indexer.as_indexer(), shape, &encoded_shape)?;
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
        // The reshaped indices select the same elements in the same order
        let encoded_shape = self.encoded_shape(shape)?;
        let indexer =
            super::get_reshaped_indexer(request.indexer.as_indexer(), shape, &encoded_shape)?;
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
        super::get_encoded_shape(&self.shape, decoded_shape)
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
