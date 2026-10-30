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
        let mut results = Vec::with_capacity(decoded_regions.size_hint().0);
        for decoded_region in decoded_regions {
            let Some(bytes) = self
                .input_handle
                .partial_decode(encoded_region(decoded_region, self.suffix_size), options)?
            else {
                return Ok(None);
            };
            results.push(match decoded_region {
                ByteRange::FromStart(_, Some(_)) => bytes,
                ByteRange::FromStart(_, None) | ByteRange::Suffix(_) => {
                    let length = bytes.len() - self.suffix_size;
                    bytes.slice(0..length)
                }
            });
        }
        Ok(Some(results))
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
        let mut values = Vec::with_capacity(futures.size_hint().0);
        let mut results = futures::stream::iter(futures)
            // A zero target is unconstrained, but buffered(0) never makes progress.
            .buffered(match options.concurrent_target() {
                0 => usize::MAX,
                target => target,
            });

        while let Some(value) = results.try_next().await? {
            values.push(value);
        }
        // Resolve missing values only after collecting, so later errors still propagate.
        let results: Option<Vec<_>> = values.into_iter().collect();
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

    #[test]
    fn strip_suffix_empty_and_missing() {
        let input = Arc::new(std::sync::Mutex::new(None));
        let decoder = StripSuffixPartialDecoder::new(input, SUFFIX_SIZE);
        let options = CodecOptions::default();
        assert_eq!(
            decoder
                .partial_decode_many(Box::new(std::iter::empty()), &options)
                .unwrap()
                .unwrap()
                .len(),
            0
        );
        let ranges =
            std::iter::once(ByteRange::FromStart(0, Some(1))).chain(std::iter::once_with(|| {
                panic!("must stop at a missing value")
            }));
        assert!(
            decoder
                .partial_decode_many(Box::new(ranges), &options)
                .unwrap()
                .is_none()
        );
    }

    #[cfg(feature = "async")]
    mod collection_tests {
        use super::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        #[derive(Clone, Copy)]
        enum Outcome {
            Value,
            Missing,
            Error,
            Pending,
        }

        struct Input {
            outcomes: [Outcome; 3],
            second_started: std::sync::Mutex<Option<futures::channel::oneshot::Sender<()>>>,
            first_resume: std::sync::Mutex<Option<futures::channel::oneshot::Receiver<()>>>,
            active: AtomicUsize,
            completed: std::sync::Mutex<Vec<u64>>,
        }

        struct Active<'a>(&'a AtomicUsize);
        impl Drop for Active<'_> {
            fn drop(&mut self) {
                self.0.fetch_sub(1, Ordering::SeqCst);
            }
        }

        #[async_trait::async_trait]
        impl AsyncBytesPartialDecoderTraits for Input {
            async fn exists(&self) -> Result<bool, StorageError> {
                Ok(true)
            }
            fn size_held(&self) -> usize {
                0
            }
            fn supports_partial_decode(&self) -> bool {
                true
            }

            async fn partial_decode_many<'a>(
                &'a self,
                mut ranges: ByteRangeIterator<'a>,
                _options: &CodecOptions,
            ) -> Result<Option<Vec<CowBytes<'a>>>, CodecError> {
                let index = ranges.next().unwrap().start(3);
                self.active.fetch_add(1, Ordering::SeqCst);
                let _active = Active(&self.active);
                if index == 0 {
                    let resume = self.first_resume.lock().unwrap().take().unwrap();
                    resume.await.unwrap();
                } else if index == 1 {
                    self.second_started
                        .lock()
                        .unwrap()
                        .take()
                        .unwrap()
                        .send(())
                        .unwrap();
                }
                let result = match self.outcomes[usize::try_from(index).unwrap()] {
                    Outcome::Value => Ok(Some(vec![vec![u8::try_from(index).unwrap()].into()])),
                    Outcome::Missing => Ok(None),
                    Outcome::Error => Err(CodecError::Other("collection error".into())),
                    Outcome::Pending => std::future::pending().await,
                };
                self.completed.lock().unwrap().push(index);
                result
            }
        }

        #[tokio::test]
        async fn ordering_missing_errors_and_cancellation() {
            use Outcome::{Error, Missing, Pending, Value};
            for outcomes in [
                [Value, Value, Value],
                [Missing, Error, Value],
                [Error, Pending, Pending],
                [Missing, Value, Value],
            ] {
                let (started, resume) = futures::channel::oneshot::channel();
                let input = Arc::new(Input {
                    outcomes,
                    second_started: std::sync::Mutex::new(Some(started)),
                    first_resume: std::sync::Mutex::new(Some(resume)),
                    active: AtomicUsize::new(0),
                    completed: std::sync::Mutex::new(Vec::new()),
                });
                let decoder = AsyncStripSuffixPartialDecoder::new(input.clone(), SUFFIX_SIZE);
                let options = CodecOptions::default().with_concurrent_target(3);
                let ranges = (0u64..3).map(|i| ByteRange::FromStart(i, Some(1)));
                let result = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    decoder.partial_decode_many(Box::new(ranges), &options),
                )
                .await
                .unwrap();
                assert_eq!(input.active.load(Ordering::SeqCst), 0);
                match outcomes {
                    [Value, Value, Value] => {
                        let values = result.unwrap().unwrap();
                        assert_eq!(
                            values.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
                            vec![&[0][..], &[1][..], &[2][..]]
                        );
                        assert_eq!(input.completed.lock().unwrap()[0], 1);
                    }
                    [Missing, Value, Value] => {
                        assert!(result.unwrap().is_none());
                        assert_eq!(input.completed.lock().unwrap().len(), 3);
                    }
                    _ => assert_eq!(result.unwrap_err().to_string(), "collection error"),
                }
                let empty = decoder
                    .partial_decode_many(Box::new(std::iter::empty()), &options)
                    .await
                    .unwrap()
                    .unwrap();
                assert_eq!(empty.len(), 0);
            }
        }
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
    #[cfg(feature = "async")]
    #[tokio::test]
    async fn zero_concurrency_strip_suffix_partial_decoder() {
        let input: Arc<dyn AsyncBytesPartialDecoderTraits> = Arc::new(ENCODED.to_vec());
        let decoder = AsyncStripSuffixPartialDecoder::new(input, SUFFIX_SIZE);
        let options = CodecOptions::default().with_concurrent_target(0);
        let ranges = [
            ByteRange::Suffix(2),
            ByteRange::FromStart(2, None),
            ByteRange::FromStart(1, Some(2)),
        ];
        let decoded = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            decoder.partial_decode_many(Box::new(ranges.into_iter()), &options),
        )
        .await
        .unwrap()
        .unwrap()
        .unwrap();
        assert_eq!(
            decoded
                .iter()
                .map(std::convert::AsRef::as_ref)
                .collect::<Vec<_>>(),
            vec![&[4, 5][..], &[2, 3, 4, 5][..], &[1, 2][..]]
        );
    }
}
