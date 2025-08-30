use std::{borrow::Cow, sync::Arc};

use crate::{
    array::{
        codec::{BytesPartialDecoderTraits, CodecError, CodecOptions},
        RawBytes,
    },
    storage::byte_range::{extract_byte_ranges, ByteRangeIterator},
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncBytesPartialDecoderTraits;

/// Partial decoder for the `test_unbounded` codec.
pub(crate) struct TestUnboundedPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
}

impl TestUnboundedPartialDecoder {
    /// Create a new partial decoder for the `test_unbounded` codec.
    pub(crate) fn new(input_handle: Arc<dyn BytesPartialDecoderTraits>) -> Self {
        Self { input_handle }
    }
}

impl BytesPartialDecoderTraits for TestUnboundedPartialDecoder {
    fn size(&self) -> usize {
        self.input_handle.size()
    }

    fn partial_decode(
        &self,
        decoded_regions: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let encoded_value = self.input_handle.decode(options)?;
        let Some(encoded_value) = encoded_value else {
            return Ok(None);
        };

        Ok(Some(
            extract_byte_ranges(&encoded_value, decoded_regions)
                .map_err(CodecError::InvalidByteRangeError)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the `test_unbounded` codec.
pub(crate) struct AsyncTestUnboundedPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
}

#[cfg(feature = "async")]
impl AsyncTestUnboundedPartialDecoder {
    /// Create a new partial decoder for the `test_unbounded` codec.
    pub(crate) fn new(input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>) -> Self {
        Self { input_handle }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for AsyncTestUnboundedPartialDecoder {
    async fn partial_decode<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let encoded_value = self.input_handle.decode(options).await?;
        let Some(encoded_value) = encoded_value else {
            return Ok(None);
        };

        Ok(Some(
            extract_byte_ranges(&encoded_value, decoded_regions)
                .map_err(CodecError::InvalidByteRangeError)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
    }
}
