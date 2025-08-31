use std::{borrow::Cow, sync::Arc};

use crate::{
    array::{
        codec::{
            bytes_to_bytes::blosc::blosc_nbytes, BytesPartialDecoderTraits, CodecError,
            CodecOptions,
        },
        RawBytes,
    },
    storage::byte_range::ByteRangeIterator,
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncBytesPartialDecoderTraits;

use super::{blosc_decompress_bytes_partial, blosc_typesize, blosc_validate};

/// Partial decoder for the `blosc` codec.
pub(crate) struct BloscPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
}

impl BloscPartialDecoder {
    pub(crate) fn new(input_handle: Arc<dyn BytesPartialDecoderTraits>) -> Self {
        Self { input_handle }
    }
}

impl BytesPartialDecoderTraits for BloscPartialDecoder {
    fn size(&self) -> usize {
        self.input_handle.size()
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let encoded_value = self.input_handle.decode(options)?;
        let Some(encoded_value) = encoded_value else {
            return Ok(None);
        };

        if let Some(_destsize) = blosc_validate(&encoded_value) {
            let nbytes = blosc_nbytes(&encoded_value);
            let typesize = blosc_typesize(&encoded_value);
            if let (Some(nbytes), Some(typesize)) = (nbytes, typesize) {
                let decoded_byte_ranges = decoded_regions
                    .map(|byte_range| {
                        let start = usize::try_from(byte_range.start(nbytes as u64)).unwrap();
                        let end = usize::try_from(byte_range.end(nbytes as u64)).unwrap();
                        blosc_decompress_bytes_partial(&encoded_value, start, end - start, typesize)
                            .map(Cow::Owned)
                            .map_err(|err| CodecError::from(err.to_string()))
                    })
                    .collect::<Result<Vec<_>, CodecError>>()?;
                return Ok(Some(decoded_byte_ranges));
            }
        }
        Err(CodecError::from("blosc encoded value is invalid"))
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the `blosc` codec.
pub(crate) struct AsyncBloscPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
}

#[cfg(feature = "async")]
impl AsyncBloscPartialDecoder {
    pub(crate) fn new(input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>) -> Self {
        Self { input_handle }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for AsyncBloscPartialDecoder {
    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'a>>>, CodecError> {
        let encoded_value = self.input_handle.decode(options).await?;
        let Some(encoded_value) = encoded_value else {
            return Ok(None);
        };

        if let Some(_destsize) = blosc_validate(&encoded_value) {
            let nbytes = blosc_nbytes(&encoded_value);
            let typesize = blosc_typesize(&encoded_value);
            if let (Some(nbytes), Some(typesize)) = (nbytes, typesize) {
                let decoded_byte_ranges = decoded_regions
                    .map(|byte_range| {
                        let start = usize::try_from(byte_range.start(nbytes as u64)).unwrap();
                        let end = usize::try_from(byte_range.end(nbytes as u64)).unwrap();
                        blosc_decompress_bytes_partial(&encoded_value, start, end - start, typesize)
                            .map(Cow::Owned)
                            .map_err(|err| CodecError::from(err.to_string()))
                    })
                    .collect::<Result<Vec<_>, CodecError>>()?;
                return Ok(Some(decoded_byte_ranges));
            }
        }
        Err(CodecError::from("blosc encoded value is invalid"))
    }
}
