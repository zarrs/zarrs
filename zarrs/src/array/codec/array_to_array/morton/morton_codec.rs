use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_plugin::ZarrVersion;

use super::{
    MortonCodecConfiguration, MortonCodecConfigurationV0, encoded_position_by_decoded_linear_index,
    encoded_runs_for_indexer, one_dimensional_indices, partial_positions_for_decoded_order,
    reorder_decoded_to_morton, reorder_morton_to_decoded, shape_u64,
};
use crate::array::{ArrayBytes, ChunkShape, DataType, FillValue, Indexer};
use zarrs_codec::{
    ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArrayToArrayCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_metadata::Configuration;
use zarrs_plugin::PluginCreateError;
use zarrs_storage::StorageError;

/// A Morton-order codec implementation.
#[derive(Clone, Debug)]
pub struct MortonCodec {}

impl MortonCodec {
    /// Create a new Morton codec from configuration.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if there is a configuration issue.
    pub fn new_with_configuration(
        configuration: &MortonCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            MortonCodecConfiguration::V0(_configuration) => Ok(Self::new()),
            _ => Err(PluginCreateError::Other(
                "this morton codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    /// Create a new Morton codec.
    #[must_use]
    pub const fn new() -> Self {
        Self {}
    }
}

impl Default for MortonCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl CodecTraits for MortonCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = MortonCodecConfiguration::V0(MortonCodecConfigurationV0 {});
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
impl ArrayToArrayCodecTraits for MortonCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        Ok(decoded_data_type.clone())
    }

    fn encoded_fill_value(
        &self,
        _decoded_data_type: &DataType,
        decoded_fill_value: &FillValue,
    ) -> Result<FillValue, CodecError> {
        Ok(decoded_fill_value.clone())
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        let num_elements = decoded_shape
            .iter()
            .try_fold(1u64, |product, dim| product.checked_mul(dim.get()))
            .ok_or_else(|| CodecError::Other("morton codec shape product overflow".to_string()))?;
        Ok(vec![NonZeroU64::new(num_elements).unwrap()])
    }

    fn decoded_shape(
        &self,
        _encoded_shape: &[NonZeroU64],
    ) -> Result<Option<ChunkShape>, CodecError> {
        Ok(None)
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let decoded_shape = shape_u64(shape);
        let num_elements = decoded_shape.iter().product();
        bytes.validate(num_elements, data_type)?;
        reorder_decoded_to_morton(&bytes, &decoded_shape, data_type)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let decoded_shape = shape_u64(shape);
        let num_elements = decoded_shape.iter().product();
        bytes.validate(num_elements, data_type)?;
        reorder_morton_to_decoded(&bytes, &decoded_shape, data_type)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(MortonCodecPartial::new(
            input_handle,
            shape,
            data_type,
        )?))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(MortonCodecPartial::new(
            input_output_handle,
            shape,
            data_type,
        )?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(MortonCodecPartial::new(
            input_handle,
            shape,
            data_type,
        )?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(MortonCodecPartial::new(
            input_output_handle,
            shape,
            data_type,
        )?))
    }
}

impl ArrayCodecTraits for MortonCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn partial_decode_granularity(&self, shape: &[NonZeroU64]) -> ChunkShape {
        vec![NonZeroU64::new(1).unwrap(); shape.len()]
    }
}

/// Generic partial codec for the Morton codec.
pub(crate) struct MortonCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    decoded_shape: Vec<u64>,
    encoded_position_by_decoded_linear_index: Vec<u64>,
    data_type: DataType,
}

impl<T: ?Sized> MortonCodecPartial<T> {
    /// Create a new [`MortonCodecPartial`].
    #[must_use]
    pub(crate) fn new(
        input_output_handle: Arc<T>,
        shape: &[NonZeroU64],
        data_type: &DataType,
    ) -> Result<Self, CodecError> {
        let decoded_shape = shape_u64(shape);
        let encoded_position_by_decoded_linear_index =
            encoded_position_by_decoded_linear_index(&decoded_shape)?;
        Ok(Self {
            input_output_handle,
            decoded_shape,
            encoded_position_by_decoded_linear_index,
            data_type: data_type.clone(),
        })
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits for MortonCodecPartial<T>
where
    T: ArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let (runs, encoded_positions_sorted, decoded_output_indices_in_encoded_order) =
            encoded_runs_for_indexer(
                indexer,
                &self.decoded_shape,
                &self.encoded_position_by_decoded_linear_index,
            )?;
        let encoded_partial = self.input_output_handle.partial_decode(&runs, options)?;
        let positions = partial_positions_for_decoded_order(
            &encoded_positions_sorted,
            &decoded_output_indices_in_encoded_order,
        );
        let indices = one_dimensional_indices(positions);
        let encoded_shape = [u64::try_from(indices.len()).unwrap()];
        Ok(encoded_partial
            .extract_array_subset(&indices, &encoded_shape, &self.data_type)?
            .into_owned())
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits for MortonCodecPartial<T>
where
    T: ArrayPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn ArrayPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let (runs, _encoded_positions_sorted, decoded_output_indices_in_encoded_order) =
            encoded_runs_for_indexer(
                indexer,
                &self.decoded_shape,
                &self.encoded_position_by_decoded_linear_index,
            )?;
        let indices = one_dimensional_indices(decoded_output_indices_in_encoded_order);
        let bytes = bytes.extract_array_subset(&indices, &[indexer.len()], &self.data_type)?;
        self.input_output_handle
            .partial_encode(&runs, &bytes, options)
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits for MortonCodecPartial<T>
where
    T: AsyncArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let (runs, encoded_positions_sorted, decoded_output_indices_in_encoded_order) =
            encoded_runs_for_indexer(
                indexer,
                &self.decoded_shape,
                &self.encoded_position_by_decoded_linear_index,
            )?;
        let encoded_partial = self
            .input_output_handle
            .partial_decode(&runs, options)
            .await?;
        let positions = partial_positions_for_decoded_order(
            &encoded_positions_sorted,
            &decoded_output_indices_in_encoded_order,
        );
        let indices = one_dimensional_indices(positions);
        let encoded_shape = [u64::try_from(indices.len()).unwrap()];
        Ok(encoded_partial
            .extract_array_subset(&indices, &encoded_shape, &self.data_type)?
            .into_owned())
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits for MortonCodecPartial<T>
where
    T: AsyncArrayPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncArrayPartialDecoderTraits> {
        self.clone()
    }

    async fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        indexer: &dyn Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let (runs, _encoded_positions_sorted, decoded_output_indices_in_encoded_order) =
            encoded_runs_for_indexer(
                indexer,
                &self.decoded_shape,
                &self.encoded_position_by_decoded_linear_index,
            )?;
        let indices = one_dimensional_indices(decoded_output_indices_in_encoded_order);
        let bytes = bytes.extract_array_subset(&indices, &[indexer.len()], &self.data_type)?;
        self.input_output_handle
            .partial_encode(&runs, &bytes, options)
            .await
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}
