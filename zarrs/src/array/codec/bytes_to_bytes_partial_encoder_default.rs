use std::{borrow::Cow, sync::Arc};

use zarrs_storage::byte_range::ByteOffset;

use crate::array::BytesRepresentation;

use super::{BytesPartialDecoderTraits, BytesPartialEncoderTraits, BytesToBytesCodecTraits};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits};

#[cfg_attr(feature = "async", async_generic::async_generic(
    async_signature(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    output_handle: &Arc<dyn AsyncBytesPartialEncoderTraits>,
    decoded_representation: &BytesRepresentation,
    codec: &Arc<dyn BytesToBytesCodecTraits>,
    offsets_and_bytes: &[(ByteOffset, crate::array::RawBytes<'_>)],
    options: &super::CodecOptions,
)))]
fn partial_encode(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    output_handle: &Arc<dyn BytesPartialEncoderTraits>,
    decoded_representation: &BytesRepresentation,
    codec: &Arc<dyn BytesToBytesCodecTraits>,
    offsets_and_bytes: &[(ByteOffset, crate::array::RawBytes<'_>)],
    options: &super::CodecOptions,
) -> Result<(), super::CodecError> {
    #[cfg(feature = "async")]
    let encoded_value = if _async {
        input_handle.decode(options).await
    } else {
        input_handle.decode(options)
    }?
    .map(Cow::into_owned);
    #[cfg(not(feature = "async"))]
    let encoded_value = input_handle.decode(options)?.map(Cow::into_owned);

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
        output_handle
            .partial_encode(&[(0, Cow::Owned(bytes_encoded))], options)
            .await
    } else {
        output_handle.partial_encode(&[(0, Cow::Owned(bytes_encoded))], options)
    }
    #[cfg(not(feature = "async"))]
    output_handle.partial_encode(&[(0, Cow::Owned(bytes_encoded))], options)
}

/// The default bytes-to-bytes partial encoder. Decodes the entire chunk, updates it, and writes the entire chunk.
pub struct BytesToBytesPartialEncoderDefault {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    output_handle: Arc<dyn BytesPartialEncoderTraits>,
    decoded_representation: BytesRepresentation,
    codec: Arc<dyn BytesToBytesCodecTraits>,
}

impl BytesToBytesPartialEncoderDefault {
    /// Create a new [`BytesToBytesPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        output_handle: Arc<dyn BytesPartialEncoderTraits>,
        decoded_representation: BytesRepresentation,
        codec: Arc<dyn BytesToBytesCodecTraits>,
    ) -> Self {
        Self {
            input_handle,
            output_handle,
            decoded_representation,
            codec,
        }
    }
}

impl BytesPartialEncoderTraits for BytesToBytesPartialEncoderDefault {
    fn erase(&self) -> Result<(), super::CodecError> {
        self.output_handle.erase()
    }

    fn partial_encode(
        &self,
        offsets_and_bytes: &[(ByteOffset, crate::array::RawBytes<'_>)],
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        partial_encode(
            &self.input_handle,
            &self.output_handle,
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
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
    decoded_representation: BytesRepresentation,
    codec: Arc<dyn BytesToBytesCodecTraits>,
}

#[cfg(feature = "async")]
impl AsyncBytesToBytesPartialEncoderDefault {
    /// Create a new [`AsyncBytesToBytesPartialEncoderDefault`].
    #[must_use]
    pub fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        decoded_representation: BytesRepresentation,
        codec: Arc<dyn BytesToBytesCodecTraits>,
    ) -> Self {
        Self {
            input_handle,
            output_handle,
            decoded_representation,
            codec,
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialEncoderTraits for AsyncBytesToBytesPartialEncoderDefault {
    async fn erase(&self) -> Result<(), super::CodecError> {
        self.output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        offsets_and_bytes: &[(ByteOffset, crate::array::RawBytes<'_>)],
        options: &super::CodecOptions,
    ) -> Result<(), super::CodecError> {
        partial_encode_async(
            &self.input_handle,
            &self.output_handle,
            &self.decoded_representation,
            &self.codec,
            offsets_and_bytes,
            options,
        )
        .await
    }
}
