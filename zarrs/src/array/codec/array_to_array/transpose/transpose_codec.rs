use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_plugin::{ExtensionAliasesV3, ZarrVersion};

use super::{
    TransposeCodecConfiguration, TransposeOrder, apply_permutation, inverse_permutation, permute,
};
use crate::array::chunk_grid::{RectilinearChunkGrid, edge_lengths_to_chunk_edge_lengths};
use crate::array::{ArrayBytes, ChunkGrid, ChunkShape, DataType, FillValue};
use zarrs_codec::{
    ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArrayToArrayCodecTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecTraits, PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    SubchunkGrid, UnboundArrayToArrayCodecTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
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
    fn validate(&self, shape: &[u64]) -> Result<(), CodecError> {
        if self.data_type.is_optional() {
            return Err(CodecError::UnsupportedDataType(
                self.data_type.clone(),
                TransposeCodec::aliases_v3().default_name.to_string(),
            ));
        }
        if self.order.0.len() != shape.len() {
            return Err(CodecError::Other(
                "Length of transpose codec `order` does not match array dimensionality".to_string(),
            ));
        }
        Ok(())
    }
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
impl UnboundArrayToArrayCodecTraits for TransposeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self as Arc<dyn UnboundArrayToArrayCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        if data_type.is_optional() {
            return Err(CodecCreateError::UnsupportedDataType(
                data_type,
                Self::aliases_v3().default_name.to_string(),
            ));
        }
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

    fn recommended_concurrency(
        &self,
        _shape: &[u64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        // TODO: This could be increased, need to implement `transpose_array` without ndarray
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for TransposeCodecBound {
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
        if self.order.0.len() != decoded_shape.len() {
            return Err(CodecError::Other(
                "Length of transpose codec `order` does not match array dimensionality".to_string(),
            ));
        }
        Ok(permute(decoded_shape, &self.order.0).expect("matching dimensionality"))
    }

    fn encoded_chunk_grid(
        &self,
        decoded_chunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, ChunkGridCreateError> {
        if self.order.0.len() != decoded_chunk_grid.dimensionality() {
            return Err(ChunkGridCreateError::new(
                "Length of transpose codec `order` does not match chunk grid dimensionality",
            ));
        }

        let array_shape = permute(decoded_chunk_grid.array_shape(), &self.order.0)
            .expect("matching dimensionality");
        let chunk_shapes = self
            .order
            .0
            .iter()
            .map(|&decoded_dim| {
                let edge_lengths = decoded_chunk_grid.chunk_edge_lengths(decoded_dim)?;
                Ok(edge_lengths_to_chunk_edge_lengths(&edge_lengths))
            })
            .collect::<Result<Vec<_>, ChunkGridCreateError>>()?;

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
        if self.order.0.len() != decoded_chunk_grid.dimensionality()
            || self.order.0.len() != encoded_subchunk_grid.dimensionality()
        {
            return Err(ChunkGridCreateError::new(
                "Length of transpose codec `order` does not match subchunk grid dimensionality",
            ));
        }

        let inverse = inverse_permutation(&self.order.0);
        let chunk_shapes = inverse
            .iter()
            .map(|&encoded_dim| {
                let edge_lengths = encoded_subchunk_grid.chunk_edge_lengths(encoded_dim)?;
                Ok(edge_lengths_to_chunk_edge_lengths(&edge_lengths))
            })
            .collect::<Result<Vec<_>, ChunkGridCreateError>>()?;

        Ok(SubchunkGrid::Array(ChunkGrid::new(
            RectilinearChunkGrid::new(decoded_chunk_grid.array_shape().to_vec(), &chunk_shapes)?,
        )))
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.validate(shape)?;

        // Encode: apply the transpose order to the decoded shape
        let shape_u64 = bytemuck::must_cast_slice(shape);
        apply_permutation(&bytes, shape_u64, &self.order.0, &self.data_type)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.validate(shape)?;

        // Decode: apply the inverse permutation to the encoded (transposed) shape
        let shape_u64 = bytemuck::must_cast_slice(shape);
        let transposed_shape = permute(shape_u64, &self.order.0).expect("validated");
        apply_permutation(
            &bytes,
            &transposed_shape,
            &inverse_permutation(&self.order.0),
            &self.data_type,
        )
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(
            super::transpose_codec_partial::TransposeCodecPartial::new(
                input_handle,
                shape,
                &self.data_type,
                &self.fill_value,
                self.order.0.clone(),
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
            super::transpose_codec_partial::TransposeCodecPartial::new(
                input_output_handle,
                shape,
                &self.data_type,
                &self.fill_value,
                self.order.0.clone(),
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
            super::transpose_codec_partial::TransposeCodecPartial::new(
                input_handle,
                shape,
                &self.data_type,
                &self.fill_value,
                self.order.0.clone(),
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
            super::transpose_codec_partial::TransposeCodecPartial::new(
                input_output_handle,
                shape,
                &self.data_type,
                &self.fill_value,
                self.order.0.clone(),
            ),
        ))
    }
}
