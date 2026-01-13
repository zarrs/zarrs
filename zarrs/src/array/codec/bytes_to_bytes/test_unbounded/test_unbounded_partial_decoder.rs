use std::borrow::Cow;
use std::sync::Arc;

use crate::array::ArrayBytesRaw;
use crate::storage::StorageError;
use crate::storage::byte_range::{ByteRangeIterator, extract_byte_ranges};
#[cfg(feature = "async")]
use zarrs_codec::AsyncBytesPartialDecoderTraits;
use zarrs_codec::{BytesPartialDecoderTraits, CodecError, CodecOptions};

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
    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError> {
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

    fn supports_partial_decode(&self) -> bool {
        false
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
    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError> {
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

    fn supports_partial_decode(&self) -> bool {
        false
    }
}
