use std::{borrow::Cow, sync::Arc};

use zarrs_storage::byte_range::{ByteOffset, ByteRangeIterator};

use crate::array::{
    codec::{CodecError, CodecOptions},
    BytesRepresentation, RawBytes,
};

use super::{BytesPartialDecoderTraits, BytesPartialEncoderTraits, BytesToBytesCodecTraits};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits};

#[cfg_attr(feature = "async", async_generic::async_generic(
    async_signature(
    input_output_handle: &Arc<dyn AsyncBytesPartialEncoderTraits>,
    decoded_representation: &BytesRepresentation,
    codec: &Arc<dyn BytesToBytesCodecTraits>,
    offsets_and_bytes: &[(ByteOffset, crate::array::RawBytes<'_>)],
    options: &super::CodecOptions,
)))]
fn partial_encode(
    input_output_handle: &Arc<dyn BytesPartialEncoderTraits>,
    decoded_representation: &BytesRepresentation,
    codec: &Arc<dyn BytesToBytesCodecTraits>,
    offsets_and_bytes: &[(ByteOffset, crate::array::RawBytes<'_>)],
    options: &super::CodecOptions,
) -> Result<(), super::CodecError> {
    #[cfg(feature = "async")]
    let encoded_value = if _async {
        input_output_handle.decode(options).await
    } else {
        input_output_handle.decode(options)
    }?
    .map(Cow::into_owned);
    #[cfg(not(feature = "async"))]
    let encoded_value = input_output_handle.decode(options)?.map(Cow::into_owned);

    let mut decoded_value = if let Some(encoded_value) = encoded_value {
        codec
            .decode(Cow::Owned(encoded_value), decoded_representation, options)?
            .into_owned()
    } else {
        vec![]
    };

    // The decoded value must be resized to the maximum byte range end
    let decoded_value_len = offsets_and_bytes
        .iter()
        .map(|(offset, bytes)| usize::try_from(offset + bytes.len() as u64).unwrap())
        .max()
        .unwrap();
    decoded_value.resize(decoded_value_len, 0);

    for (offset, bytes) in offsets_and_bytes {
        let start = usize::try_from(*offset).unwrap();
        decoded_value[start..start + bytes.len()].copy_from_slice(bytes);
    }

    let bytes_encoded = codec
        .encode(Cow::Owned(decoded_value), options)?
        .into_owned();

    #[cfg(feature = "async")]
    if _async {
        input_output_handle
            .partial_encode(&[(0, Cow::Owned(bytes_encoded))], options)
            .await
    } else {
        input_output_handle.partial_encode(&[(0, Cow::Owned(bytes_encoded))], options)
    }
    #[cfg(not(feature = "async"))]
    input_output_handle.partial_encode(&[(0, Cow::Owned(bytes_encoded))], options)
}

/// The default bytes-to-bytes partial encoder. Decodes the entire chunk, updates it, and writes the entire chunk.
pub struct BytesToBytesPartialEncoderDefault {
    input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
    decoded_representation: BytesRepresentation,
    codec: Arc<dyn BytesToBytesCodecTraits>,
}

impl BytesToBytesPartialEncoderDefault {
    /// Create a new [`BytesToBytesPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
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

impl BytesPartialDecoderTraits for BytesToBytesPartialEncoderDefault {
    fn size(&self) -> usize {
        self.input_output_handle.size()
    }

    fn partial_decode(
        &self,
        decoded_regions: &mut dyn ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        super::bytes_to_bytes_partial_decoder_default::partial_decode(
            &self.input_output_handle.clone().into_dyn_decoder(),
            &self.decoded_representation,
            &self.codec,
            decoded_regions,
            options,
        )
    }
}

impl BytesPartialEncoderTraits for BytesToBytesPartialEncoderDefault {
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn BytesPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode(
        &self,
        offsets_and_bytes: &[(ByteOffset, crate::array::RawBytes<'_>)],
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        partial_encode(
            &self.input_output_handle,
            &self.decoded_representation,
            &self.codec,
            offsets_and_bytes,
            options,
        )
    }
}

#[cfg(feature = "async")]
/// The default asynchronous bytes-to-bytes partial encoder. Decodes the entire chunk, updates it, and writes the entire chunk.
pub struct AsyncBytesToBytesPartialEncoderDefault {
    input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
    decoded_representation: BytesRepresentation,
    codec: Arc<dyn BytesToBytesCodecTraits>,
}

#[cfg(feature = "async")]
impl AsyncBytesToBytesPartialEncoderDefault {
    /// Create a new [`AsyncBytesToBytesPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
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

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for AsyncBytesToBytesPartialEncoderDefault {
    async fn partial_decode(
        &self,
        decoded_regions: &mut dyn ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        super::bytes_to_bytes_partial_decoder_default::partial_decode_async(
            &self.input_output_handle.clone().into_dyn_decoder(),
            &self.decoded_representation,
            &self.codec,
            decoded_regions,
            options,
        )
        .await
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialEncoderTraits for AsyncBytesToBytesPartialEncoderDefault {
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncBytesPartialDecoderTraits> {
        self.clone()
    }

    async fn erase(&self) -> Result<(), super::CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        offsets_and_bytes: &[(ByteOffset, crate::array::RawBytes<'_>)],
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        partial_encode_async(
            &self.input_output_handle,
            &self.decoded_representation,
            &self.codec,
            offsets_and_bytes,
            options,
        )
        .await
    }
}
