use std::sync::Arc;

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
        decoded_regions: &mut dyn ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<RawBytes<'_>>, CodecError> {
        let mut bytes_out = Vec::new();
        for decoded_region in decoded_regions {
            let bytes = self
                .input_handle
                .partial_decode(&mut [decoded_region].into_iter(), options)?;
            if let Some(bytes) = bytes {
                let bytes_stripped = match decoded_region {
                    ByteRange::FromStart(_, Some(_)) => &bytes[..],
                    ByteRange::FromStart(_, None) => {
                        let length = bytes.len() - self.suffix_size;
                        &bytes[..length]
                    }
                    ByteRange::Suffix(_) => {
                        let length = bytes.len() as u64 - (self.suffix_size as u64);
                        let length = usize::try_from(length).unwrap();
                        &bytes[..length]
                    }
                };
                bytes_out.extend_from_slice(bytes_stripped);
            } else {
                return Ok(None);
            }
        }
        Ok(Some(bytes_out.into()))
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
#[async_trait::async_trait]
impl AsyncBytesPartialDecoderTraits for AsyncStripSuffixPartialDecoder {
    async fn partial_decode(
        &self,
        decoded_regions: &mut dyn ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<RawBytes<'_>>, CodecError> {
        use futures::StreamExt;

        let futures = decoded_regions.map(|decoded_region| async move {
            match decoded_region {
                ByteRange::FromStart(_, Some(_)) => Ok::<_, CodecError>(
                    self.input_handle
                        .partial_decode(&mut [decoded_region].into_iter(), options)
                        .await?,
                ),
                ByteRange::FromStart(_, None) | ByteRange::Suffix(_) => {
                    let bytes = self
                        .input_handle
                        .partial_decode(&mut [decoded_region].into_iter(), options)
                        .await?;
                    if let Some(bytes) = bytes {
                        use std::borrow::Cow;

                        let length = bytes.len() - self.suffix_size;
                        Ok(Some(Cow::Owned(bytes[..length].to_vec())))
                    } else {
                        Ok(None)
                    }
                }
            }
        });
        let mut futures = futures::stream::iter(futures).buffered(options.concurrent_target());
        let mut output = Vec::new();
        while let Some(region) = futures.next().await {
            let region = region?;
            if let Some(region) = region {
                output.extend_from_slice(&region);
            } else {
                return Ok(None);
            }
        }
        Ok(Some(output.into()))
    }
}
