use std::{borrow::Cow, sync::Arc};

use zarrs_storage::{byte_range::{extract_byte_ranges, ByteRangeIterator}, OffsetBytesIterator};

use crate::array::{BytesRepresentation, RawBytes};

use super::{BytesPartialDecoderTraits, BytesPartialEncoderTraits, BytesToBytesCodecTraits, CodecError, CodecOptions};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits};

/// Generic partial codec for bytes-to-bytes operations with default behavior.
pub struct BytesToBytesCodecPartialDefault<T: ?Sized> {
    input_output_handle: Arc<T>,
    decoded_representation: BytesRepresentation,
    codec: Arc<dyn BytesToBytesCodecTraits>,
}

impl<T: ?Sized> BytesToBytesCodecPartialDefault<T> {
    /// Create a new [`BytesToBytesCodecPartialDefault`].
    #[must_use]
    pub fn new(
        input_output_handle: Arc<T>,
        decoded_representation: BytesRepresentation,
        codec: Arc<dyn BytesToBytesCodecTraits>,
    ) -> Self {
        Self {
            input_output_handle,
            decoded_representation,
            codec,
        }
    }
}

impl<T: ?Sized> BytesPartialDecoderTraits for BytesToBytesCodecPartialDefault<T>
where
    T: BytesPartialDecoderTraits,
{
    fn size(&self) -> usize {
        self.input_output_handle.size()
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let encoded_value = self.input_output_handle.decode(options)?;

        let Some(encoded_value) = encoded_value else {
            return Ok(None);
        };

        let decoded_value = self.codec
            .decode(encoded_value, &self.decoded_representation, options)?
            .into_owned();

        Ok(Some(
            extract_byte_ranges(&decoded_value, decoded_regions)
                .map_err(CodecError::InvalidByteRangeError)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
    }
}

impl<T: ?Sized> BytesPartialEncoderTraits for BytesToBytesCodecPartialDefault<T>
where
    T: BytesPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn BytesPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<crate::array::RawBytes<'_>>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        let encoded_value = self.input_output_handle.decode(options)?.map(Cow::into_owned);

        let mut decoded_value = if let Some(encoded_value) = encoded_value {
            self.codec
                .decode(Cow::Owned(encoded_value), &self.decoded_representation, options)?
                .into_owned()
        } else {
            vec![]
        };

        for (offset, value) in offset_values {
            let offset = usize::try_from(offset).unwrap();
            if decoded_value.len() < offset + value.len() {
                decoded_value.resize(offset + value.len(), 0);
            }
            decoded_value[offset..offset + value.len()].copy_from_slice(&value);
        }

        let bytes_encoded = self.codec
            .encode(Cow::Owned(decoded_value), options)?
            .into_owned();

        self.input_output_handle.partial_encode(0, Cow::Owned(bytes_encoded), options)
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncBytesPartialDecoderTraits for BytesToBytesCodecPartialDefault<T>
where
    T: AsyncBytesPartialDecoderTraits,
{
    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'a>>>, CodecError> {
        let encoded_value = self.input_output_handle.decode(options).await?;

        let Some(encoded_value) = encoded_value else {
            return Ok(None);
        };

        let decoded_value = self.codec
            .decode(encoded_value, &self.decoded_representation, options)?
            .into_owned();

        Ok(Some(
            extract_byte_ranges(&decoded_value, decoded_regions)
                .map_err(CodecError::InvalidByteRangeError)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncBytesPartialEncoderTraits for BytesToBytesCodecPartialDefault<T>
where
    T: AsyncBytesPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncBytesPartialDecoderTraits> {
        self.clone()
    }

    async fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode_many<'a>(
        &'a self,
        offset_values: OffsetBytesIterator<'a, crate::array::RawBytes<'_>>,
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        let encoded_value = self.input_output_handle.decode(options).await?.map(Cow::into_owned);

        let mut decoded_value = if let Some(encoded_value) = encoded_value {
            self.codec
                .decode(Cow::Owned(encoded_value), &self.decoded_representation, options)?
                .into_owned()
        } else {
            vec![]
        };

        for (offset, value) in offset_values {
            let offset = usize::try_from(offset).unwrap();
            if decoded_value.len() < offset + value.len() {
                decoded_value.resize(offset + value.len(), 0);
            }
            decoded_value[offset..offset + value.len()].copy_from_slice(&value);
        }

        let bytes_encoded = self.codec
            .encode(Cow::Owned(decoded_value), options)?
            .into_owned();

        self.input_output_handle.partial_encode(0, Cow::Owned(bytes_encoded), options).await
    }
}