use std::borrow::Cow;
use std::sync::Arc;

use ambisync::ambisync;

use crate::array::ArrayBytesRaw;
#[cfg(feature = "async")]
use zarrs_codec::AsyncBytesPartialDecoderTraits;
use zarrs_codec::{BytesPartialDecoderTraits, CodecError, CodecOptions};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteRange, ByteRangeIterator};

/// Asynchronous partial decoder for stripping a suffix (e.g. checksum).
#[ambisync(
    sync(
        types(
            AsyncStripSuffixPartialDecoder => StripSuffixPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
pub(crate) struct AsyncStripSuffixPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    suffix_size: usize,
}

#[ambisync(
    sync(
        fns("{}"),
        types(
            AsyncStripSuffixPartialDecoder => StripSuffixPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
impl AsyncStripSuffixPartialDecoder {
    /// Create a new "strip suffix" partial decoder.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        suffix_size: usize,
    ) -> Self {
        Self {
            input_handle,
            suffix_size,
        }
    }
}

#[ambisync(
    sync(
        fns("{}"),
        types(
            AsyncStripSuffixPartialDecoder => StripSuffixPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32")),
    ),
)]
impl AsyncBytesPartialDecoderTraits for AsyncStripSuffixPartialDecoder {
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
        ambisync::alt!(
            sync => decoded_regions
                .map(|decoded_region| {
                    let bytes = self.input_handle.partial_decode(decoded_region, options)?;
                    Ok::<_, CodecError>(bytes.map(|bytes| match decoded_region {
                        ByteRange::FromStart(_, Some(_)) => bytes,
                        ByteRange::FromStart(_, None) => {
                            let length = bytes.len() - self.suffix_size;
                            let mut bytes = bytes.into_owned();
                            bytes.truncate(length);
                            Cow::Owned(bytes)
                        }
                        ByteRange::Suffix(_) => {
                            let length = bytes.len() as u64 - (self.suffix_size as u64);
                            let length = usize::try_from(length).unwrap();
                            let mut bytes = bytes.into_owned();
                            bytes.truncate(length);
                            Cow::Owned(bytes)
                        }
                    }))
                })
                .collect(),
            async => {
                use futures::{StreamExt, TryStreamExt};

                let futures = decoded_regions.map(|decoded_region| async move {
                    match decoded_region {
                        ByteRange::FromStart(_, Some(_)) => Ok::<_, CodecError>(
                            self.input_handle
                                .partial_decode(decoded_region, options)
                                .await?,
                        ),
                        ByteRange::FromStart(_, None) | ByteRange::Suffix(_) => {
                            let bytes = self
                                .input_handle
                                .partial_decode(decoded_region, options)
                                .await?;
                            if let Some(bytes) = bytes {
                                let length = bytes.len() - self.suffix_size;
                                let mut bytes = bytes.into_owned();
                                bytes.truncate(length);
                                Ok(Some(Cow::Owned(bytes)))
                            } else {
                                Ok(None)
                            }
                        }
                    }
                });
                let results: Vec<Option<_>> = futures::stream::iter(futures)
                    .buffered(options.concurrent_target())
                    .try_collect()
                    .await?;
                let results: Option<Vec<_>> = results.into_iter().collect();
                Ok(results)
            },
        )
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}
