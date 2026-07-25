use std::borrow::Cow;
use std::sync::Arc;

use ambisync::ambisync;

use crate::array::ArrayBytesRaw;
#[cfg(feature = "async")]
use zarrs_codec::AsyncBytesPartialDecoderTraits;
use zarrs_codec::{BytesPartialDecoderTraits, CodecError, CodecOptions};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteRangeIterator, extract_byte_ranges};

/// Asynchronous partial decoder for the `test_unbounded` codec.
#[ambisync(
    sync(
        types(
            AsyncTestUnboundedPartialDecoder => TestUnboundedPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
pub(crate) struct AsyncTestUnboundedPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
}

#[ambisync(
    sync(
        fns("{}"),
        types(
            AsyncTestUnboundedPartialDecoder => TestUnboundedPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
impl AsyncTestUnboundedPartialDecoder {
    /// Create a new partial decoder for the `test_unbounded` codec.
    pub(crate) fn new(input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>) -> Self {
        Self { input_handle }
    }
}

#[ambisync(
    sync(
        fns("{}"),
        types(
            AsyncTestUnboundedPartialDecoder => TestUnboundedPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32")),
    ),
)]
impl AsyncBytesPartialDecoderTraits for AsyncTestUnboundedPartialDecoder {
    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    #[sync_signature(
        fn partial_decode_many(
            &self,
            decoded_regions: ByteRangeIterator,
            options: &CodecOptions,
        ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError>
    )]
    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'_>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
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
