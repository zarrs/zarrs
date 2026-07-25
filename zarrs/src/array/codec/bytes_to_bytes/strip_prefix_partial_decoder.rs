use std::sync::Arc;

use ambisync::ambisync;

use crate::array::ArrayBytesRaw;
#[cfg(feature = "async")]
use zarrs_codec::AsyncBytesPartialDecoderTraits;
use zarrs_codec::{BytesPartialDecoderTraits, CodecError, CodecOptions};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteRange, ByteRangeIterator};

/// Asynchronous partial decoder for stripping a prefix (e.g. checksum).
#[ambisync(
    sync(
        types(
            AsyncStripPrefixPartialDecoder => StripPrefixPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
pub(crate) struct AsyncStripPrefixPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    prefix_size: usize,
}

#[ambisync(
    sync(
        fns("{}"),
        types(
            AsyncStripPrefixPartialDecoder => StripPrefixPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
impl AsyncStripPrefixPartialDecoder {
    /// Create a new "strip prefix" partial decoder.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        prefix_size: usize,
    ) -> Self {
        Self {
            input_handle,
            prefix_size,
        }
    }
}

#[ambisync(
    sync(
        fns("{}"),
        types(
            AsyncStripPrefixPartialDecoder => StripPrefixPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32")),
    ),
)]
impl AsyncBytesPartialDecoderTraits for AsyncStripPrefixPartialDecoder {
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
        let decoded_regions = decoded_regions.map(|range| match range {
            ByteRange::FromStart(offset, length) => {
                ByteRange::FromStart(offset.checked_add(self.prefix_size as u64).unwrap(), length)
            }
            ByteRange::Suffix(length) => ByteRange::Suffix(length),
        });

        self.input_handle
            .partial_decode_many(Box::new(decoded_regions), options)
            .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}
