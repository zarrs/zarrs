use std::{borrow::Cow, sync::Arc};

use zarrs_storage::byte_range::{extract_byte_ranges, ByteRange};
use zarrs_shared::{MaybeSend, MaybeSync};

use crate::array::{BytesRepresentation, RawBytes};

use super::{BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecError, CodecOptions};

#[cfg(feature = "async")]
use crate::array::codec::AsyncBytesPartialDecoderTraits;

#[cfg_attr(feature = "async", async_generic::async_generic(
    async_signature(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    decoded_representation: &BytesRepresentation,
    codec: &Arc<dyn BytesToBytesCodecTraits>,
    decoded_regions: &mut (dyn Iterator<Item = ByteRange> + MaybeSend),
    options: &CodecOptions,
)))]
fn partial_decode<'a>(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    decoded_representation: &BytesRepresentation,
    codec: &Arc<dyn BytesToBytesCodecTraits>,
    decoded_regions: &mut (dyn Iterator<Item = ByteRange> + MaybeSend),
    options: &CodecOptions,
) -> Result<Option<Vec<RawBytes<'a>>>, CodecError> {
    #[cfg(feature = "async")]
    let encoded_value = if _async {
        input_handle.decode(options).await
    } else {
        input_handle.decode(options)
    }?;
    #[cfg(not(feature = "async"))]
    let encoded_value = input_handle.decode(options)?;

    let Some(encoded_value) = encoded_value else {
        return Ok(None);
    };

    let decoded_value = codec
        .decode(encoded_value, decoded_representation, options)?
        .into_owned();

    Ok(Some(
        extract_byte_ranges(&decoded_value, decoded_regions)
            .map_err(CodecError::InvalidByteRangeError)?
            .into_iter()
            .map(Cow::Owned)
            .collect(),
    ))
}

/// The default bytes-to-bytes partial decoder. Decodes the entire chunk and extracts the desired region.
pub struct BytesToBytesPartialDecoderDefault {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    decoded_representation: BytesRepresentation,
    codec: Arc<dyn BytesToBytesCodecTraits>,
}

impl BytesToBytesPartialDecoderDefault {
    /// Create a new [`BytesToBytesPartialDecoderDefault`].
    #[must_use]
    pub fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: BytesRepresentation,
        codec: Arc<dyn BytesToBytesCodecTraits>,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
            codec,
        }
    }
}

impl BytesPartialDecoderTraits for BytesToBytesPartialDecoderDefault {
    fn size(&self) -> usize {
        self.input_handle.size()
    }

    fn partial_decode(
        &self,
        decoded_regions: &mut (dyn Iterator<Item = ByteRange> + MaybeSend),
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        partial_decode(
            &self.input_handle,
            &self.decoded_representation,
            &self.codec,
            decoded_regions,
            options,
        )
    }
}

#[cfg(feature = "async")]
/// The default asynchronous bytes-to-bytes partial decoder. Decodes the entire chunk and extracts the desired region.
pub struct AsyncBytesToBytesPartialDecoderDefault {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    decoded_representation: BytesRepresentation,
    codec: Arc<dyn BytesToBytesCodecTraits>,
}

#[cfg(feature = "async")]
impl AsyncBytesToBytesPartialDecoderDefault {
    /// Create a new [`AsyncBytesToBytesPartialDecoderDefault`].
    #[must_use]
    pub fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: BytesRepresentation,
        codec: Arc<dyn BytesToBytesCodecTraits>,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
            codec,
        }
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl AsyncBytesPartialDecoderTraits for AsyncBytesToBytesPartialDecoderDefault {
    async fn partial_decode(
        &self,
        decoded_regions: &mut (dyn Iterator<Item = ByteRange> + MaybeSend),
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        partial_decode_async(
            &self.input_handle,
            &self.decoded_representation,
            &self.codec,
            decoded_regions,
            options,
        )
        .await
    }
}
