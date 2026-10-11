use std::sync::Arc;

use crate::array::CowBytes;
#[cfg(feature = "async")]
use zarrs_codec::AsyncBytesPartialDecoderTraits;
use zarrs_codec::{BytesPartialDecoderTraits, CodecError, CodecOptions};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{ByteRange, ByteRangeIterator};

/// Map a byte range of the decoded bytes to a byte range of the encoded bytes.
///
/// The returned byte range includes the suffix if it extends to the end of the encoded bytes.
fn encoded_region(decoded_region: ByteRange, suffix_size: usize) -> ByteRange {
    match decoded_region {
        ByteRange::FromStart(_, _) => decoded_region,
        ByteRange::Suffix(length) => ByteRange::Suffix(length + suffix_size as u64),
    }
}

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
    ) -> Result<Option<Vec<CowBytes<'_>>>, CodecError> {
        decoded_regions
            .map(|decoded_region| {
                let bytes = self
                    .input_handle
                    .partial_decode(encoded_region(decoded_region, self.suffix_size), options)?;
                Ok::<_, CodecError>(bytes.map(|bytes| match decoded_region {
                    ByteRange::FromStart(_, Some(_)) => bytes,
                    ByteRange::FromStart(_, None) | ByteRange::Suffix(_) => {
                        let length = bytes.len() - self.suffix_size;
                        bytes.slice(0..length)
                    }
                }))
            })
            .collect()
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
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
    ) -> Result<Option<Vec<CowBytes<'a>>>, CodecError> {
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
                        .partial_decode(encoded_region(decoded_region, self.suffix_size), options)
                        .await?;
                    if let Some(bytes) = bytes {
                        let length = bytes.len() - self.suffix_size;
                        Ok(Some(bytes.slice(0..length)))
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

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENCODED: [u8; 8] = [0, 1, 2, 3, 4, 5, 98, 99];
    const SUFFIX_SIZE: usize = 2;

    fn decode(byte_range: ByteRange) -> Vec<u8> {
        let input: Arc<dyn BytesPartialDecoderTraits> = Arc::new(ENCODED.to_vec());
        let decoder = StripSuffixPartialDecoder::new(input, SUFFIX_SIZE);
        decoder
            .partial_decode(byte_range, &CodecOptions::default())
            .unwrap()
            .unwrap()
            .to_vec()
    }

    #[test]
    fn strip_suffix_partial_decoder() {
        assert_eq!(
            decode(ByteRange::FromStart(0, None)),
            vec![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(decode(ByteRange::FromStart(2, None)), vec![2, 3, 4, 5]);
        assert_eq!(decode(ByteRange::FromStart(1, Some(2))), vec![1, 2]);
        assert_eq!(decode(ByteRange::Suffix(2)), vec![4, 5]);
        assert_eq!(decode(ByteRange::Suffix(6)), vec![0, 1, 2, 3, 4, 5]);
    }

    #[cfg(feature = "async")]
    async fn async_decode(byte_range: ByteRange) -> Vec<u8> {
        let input: Arc<dyn AsyncBytesPartialDecoderTraits> = Arc::new(ENCODED.to_vec());
        let decoder = AsyncStripSuffixPartialDecoder::new(input, SUFFIX_SIZE);
        decoder
            .partial_decode(byte_range, &CodecOptions::default())
            .await
            .unwrap()
            .unwrap()
            .to_vec()
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_strip_suffix_partial_decoder() {
        assert_eq!(
            async_decode(ByteRange::FromStart(0, None)).await,
            vec![0, 1, 2, 3, 4, 5]
        );
        assert_eq!(
            async_decode(ByteRange::FromStart(2, None)).await,
            vec![2, 3, 4, 5]
        );
        assert_eq!(
            async_decode(ByteRange::FromStart(1, Some(2))).await,
            vec![1, 2]
        );
        assert_eq!(async_decode(ByteRange::Suffix(2)).await, vec![4, 5]);
        assert_eq!(
            async_decode(ByteRange::Suffix(6)).await,
            vec![0, 1, 2, 3, 4, 5]
        );
    }
}
