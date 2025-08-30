use std::{borrow::Cow, sync::Arc};

use crate::{
    array::{
        codec::{BytesPartialDecoderTraits, CodecError, CodecOptions},
        RawBytes,
    },
    storage::byte_range::{ByteRange, ByteRangeIterator},
};

#[cfg(feature = "async")]
use crate::array::codec::AsyncBytesPartialDecoderTraits;

/// Partial decoder for stripping a suffix (e.g. checksum).
pub(crate) struct StripSuffixPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    suffix_size: usize,
}

impl StripSuffixPartialDecoder {
    /// Create a new "strip suffix" partial decoder.
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        suffix_size: usize,
    ) -> Self {
        Self {
            input_handle,
            suffix_size,
        }
    }
}

impl BytesPartialDecoderTraits for StripSuffixPartialDecoder {
    fn size(&self) -> usize {
        self.input_handle.size()
    }

    fn partial_decode(
        &self,
        decoded_regions: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        decoded_regions
            .map(|decoded_region| {
                let bytes = self
                    .input_handle
                    .partial_decode_concat(Box::new([decoded_region].into_iter()), options)?;
                Ok::<_, CodecError>(bytes.map(|bytes| match decoded_region {
                    ByteRange::FromStart(_, Some(_)) => bytes,
                    ByteRange::FromStart(_, None) => {
                        let length = bytes.len() - self.suffix_size;
                        Cow::Owned(bytes[..length].to_vec())
                    }
                    ByteRange::Suffix(_) => {
                        let length = bytes.len() as u64 - (self.suffix_size as u64);
                        let length = usize::try_from(length).unwrap();
                        Cow::Owned(bytes[..length].to_vec())
                    }
                }))
            })
            .collect()
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for stripping a suffix (e.g. checksum).
pub(crate) struct AsyncStripSuffixPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    suffix_size: usize,
}

#[cfg(feature = "async")]
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

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for AsyncStripSuffixPartialDecoder {
    async fn partial_decode<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        use futures::{StreamExt, TryStreamExt};

        let futures = decoded_regions.map(|decoded_region| async move {
            match decoded_region {
                ByteRange::FromStart(_, Some(_)) => Ok::<_, CodecError>(
                    self.input_handle
                        .partial_decode_concat(Box::new([decoded_region].into_iter()), options)
                        .await?,
                ),
                ByteRange::FromStart(_, None) | ByteRange::Suffix(_) => {
                    let bytes = self
                        .input_handle
                        .partial_decode_concat(Box::new([decoded_region].into_iter()), options)
                        .await?;
                    if let Some(bytes) = bytes {
                        let length = bytes.len() - self.suffix_size;
                        Ok(Some(Cow::Owned(bytes[..length].to_vec())))
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
    }
}
