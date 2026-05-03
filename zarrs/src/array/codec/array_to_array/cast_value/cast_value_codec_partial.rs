use std::sync::Arc;

use zarrs_codec::{
    ArrayBytes, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, CodecError, CodecOptions,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_storage::StorageError;

use super::cast_value_codec::{CastBytesContext, ScalarMap, cast_bytes};
use crate::array::{DataType, Indexer};
use zarrs_data_type::codec_traits::cast_value::CastValueDataTypeExt;
use zarrs_metadata_ext::codec::cast_value::{CastValueOutOfRangeMode, CastValueRoundingMode};

/// Partial encoder and decoder for the `cast_value` codec.
pub(crate) struct CastValueCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    decoded_data_type: DataType,
    encoded_data_type: DataType,
    encode_scalar_map: Option<ScalarMap>,
    decode_scalar_map: Option<ScalarMap>,
    rounding: Option<CastValueRoundingMode>,
    out_of_range: Option<CastValueOutOfRangeMode>,
}

impl<T: ?Sized> CastValueCodecPartial<T> {
    pub(crate) fn new(
        input_output_handle: Arc<T>,
        decoded_data_type: &DataType,
        encoded_data_type: &DataType,
        encode_scalar_map: Option<ScalarMap>,
        decode_scalar_map: Option<ScalarMap>,
        rounding: Option<CastValueRoundingMode>,
        out_of_range: Option<CastValueOutOfRangeMode>,
    ) -> Result<Self, CodecError> {
        decoded_data_type.codec_castvalue()?;
        encoded_data_type.codec_castvalue()?;
        require_fixed(decoded_data_type)?;
        require_fixed(encoded_data_type)?;
        Ok(Self {
            input_output_handle,
            decoded_data_type: decoded_data_type.clone(),
            encoded_data_type: encoded_data_type.clone(),
            encode_scalar_map,
            decode_scalar_map,
            rounding,
            out_of_range,
        })
    }

    fn cast(
        &self,
        bytes: ArrayBytes<'_>,
        source_data_type: &DataType,
        target_data_type: &DataType,
        scalar_map: Option<&ScalarMap>,
    ) -> Result<ArrayBytes<'static>, CodecError> {
        let source = source_data_type.codec_castvalue()?;
        let target = target_data_type.codec_castvalue()?;
        Ok(ArrayBytes::from(cast_bytes(
            &bytes.into_fixed()?,
            &CastBytesContext {
                source_data_type,
                target_data_type,
                source,
                target,
                scalar_map,
                rounding: self.rounding,
                out_of_range: self.out_of_range,
            },
        )?))
    }

    fn encode(&self, bytes: ArrayBytes<'_>) -> Result<ArrayBytes<'static>, CodecError> {
        self.cast(
            bytes,
            &self.decoded_data_type,
            &self.encoded_data_type,
            self.encode_scalar_map.as_ref(),
        )
    }

    fn decode(&self, bytes: ArrayBytes<'_>) -> Result<ArrayBytes<'static>, CodecError> {
        self.cast(
            bytes,
            &self.encoded_data_type,
            &self.decoded_data_type,
            self.decode_scalar_map.as_ref(),
        )
    }
}

fn require_fixed(data_type: &DataType) -> Result<(), CodecError> {
    data_type.fixed_size().ok_or_else(|| {
        CodecError::Other(format!(
            "cast_value requires fixed-size data types, got {data_type}"
        ))
    })?;
    Ok(())
}

impl<T: ?Sized> ArrayPartialDecoderTraits for CastValueCodecPartial<T>
where
    T: ArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        &self.decoded_data_type
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
        let encoded = self.input_output_handle.partial_decode(indexer, options)?;
        encoded.validate(indexer.len(), &self.encoded_data_type)?;
        self.decode(encoded)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits for CastValueCodecPartial<T>
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
        bytes.validate(indexer.len(), &self.decoded_data_type)?;
        let encoded = self.encode(bytes.clone())?;
        self.input_output_handle
            .partial_encode(indexer, &encoded, options)
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits for CastValueCodecPartial<T>
where
    T: AsyncArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        &self.decoded_data_type
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
        let encoded = self
            .input_output_handle
            .partial_decode(indexer, options)
            .await?;
        encoded.validate(indexer.len(), &self.encoded_data_type)?;
        self.decode(encoded)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits for CastValueCodecPartial<T>
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
        bytes.validate(indexer.len(), &self.decoded_data_type)?;
        let encoded = self.encode(bytes.clone())?;
        self.input_output_handle
            .partial_encode(indexer, &encoded, options)
            .await
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}
