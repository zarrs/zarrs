use std::sync::Arc;

use zarrs_storage::StorageError;
use zarrs_storage::byte_range::{
    ByteLength, ByteOffset, ByteRange, ByteRangeIterator, InvalidByteRangeError,
};

use crate::{CodecError, CodecOptions, CowBytes, Resources};

#[cfg(feature = "async")]
use crate::AsyncBytesPartialDecoderTraits;
use crate::BytesPartialDecoderTraits;

/// Map byte ranges within the interval `byte_offset..byte_offset + byte_length` to byte ranges of the whole input.
///
/// Returns an [`InvalidByteRangeError`] if a byte range is not within the interval, or the interval end overflows.
fn interval_byte_ranges(
    byte_ranges: ByteRangeIterator,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
) -> Result<Vec<ByteRange>, InvalidByteRangeError> {
    let interval_valid = byte_offset.checked_add(byte_length).is_some();
    byte_ranges
        .map(|byte_range| {
            byte_range
                .checked_range(byte_length)
                .filter(|_| interval_valid)
                .map(|range| {
                    ByteRange::FromStart(byte_offset + range.start, Some(range.end - range.start))
                })
                .ok_or_else(|| InvalidByteRangeError::new(byte_range, byte_length))
        })
        .collect()
}

/// A partial decoder for a byte interval of a [`BytesPartialDecoderTraits`] partial decoder.
///
/// Modifies byte range requests to a specific byte interval in an inner bytes partial decoder.
/// Requests outside of the interval return a [`CodecError::InvalidByteRangeError`].
pub struct ByteIntervalPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
}

impl ByteIntervalPartialDecoder {
    /// Create a new byte interval partial decoder.
    pub fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        byte_offset: ByteOffset,
        byte_length: ByteLength,
    ) -> Self {
        Self {
            input_handle,
            byte_offset,
            byte_length,
        }
    }
}

impl BytesPartialDecoderTraits for ByteIntervalPartialDecoder {
    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    fn partial_decode_many(
        &self,
        byte_ranges: ByteRangeIterator,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Vec<CowBytes<'_>>>, CodecError> {
        let byte_ranges = interval_byte_ranges(byte_ranges, self.byte_offset, self.byte_length)?;
        self.input_handle
            .partial_decode_many(Box::new(byte_ranges.into_iter()), options, resources)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
/// A partial decoder for a byte interval of a [`AsyncBytesPartialDecoderTraits`] partial decoder.
///
/// Modifies byte range requests to a specific byte interval in an inner bytes partial decoder.
/// Requests outside of the interval return a [`CodecError::InvalidByteRangeError`].
pub struct AsyncByteIntervalPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    byte_offset: ByteOffset,
    byte_length: ByteLength,
}

#[cfg(feature = "async")]
impl AsyncByteIntervalPartialDecoder {
    /// Create a new byte interval partial decoder.
    pub fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        byte_offset: ByteOffset,
        byte_length: ByteLength,
    ) -> Self {
        Self {
            input_handle,
            byte_offset,
            byte_length,
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for AsyncByteIntervalPartialDecoder {
    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        byte_ranges: ByteRangeIterator<'a>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Vec<CowBytes<'a>>>, CodecError> {
        let byte_ranges = interval_byte_ranges(byte_ranges, self.byte_offset, self.byte_length)?;
        self.input_handle
            .partial_decode_many(Box::new(byte_ranges.into_iter()), options, resources)
            .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(byte_range: ByteRange) -> Vec<u8> {
        let input: Arc<dyn BytesPartialDecoderTraits> = Arc::new((0..10u8).collect::<Vec<u8>>());
        let decoder = ByteIntervalPartialDecoder::new(input, 2, 5);
        decoder
            .partial_decode(byte_range, &CodecOptions::default(), &Resources::default())
            .unwrap()
            .unwrap()
            .to_vec()
    }

    #[test]
    fn byte_interval_partial_decoder() {
        assert_eq!(decode(ByteRange::FromStart(0, None)), vec![2, 3, 4, 5, 6]);
        assert_eq!(decode(ByteRange::FromStart(1, None)), vec![3, 4, 5, 6]);
        assert_eq!(decode(ByteRange::FromStart(1, Some(2))), vec![3, 4]);
        assert_eq!(decode(ByteRange::Suffix(2)), vec![5, 6]);
        assert_eq!(decode(ByteRange::FromStart(5, None)), Vec::<u8>::new());
    }

    #[test]
    fn byte_interval_partial_decoder_out_of_bounds() {
        let input: Arc<dyn BytesPartialDecoderTraits> = Arc::new((0..10u8).collect::<Vec<u8>>());
        let options = CodecOptions::default();
        let decoder = ByteIntervalPartialDecoder::new(input.clone(), 2, 5);
        for byte_range in [
            ByteRange::FromStart(6, None),
            ByteRange::FromStart(4, Some(2)),
            ByteRange::FromStart(u64::MAX, Some(1)),
            ByteRange::Suffix(6),
        ] {
            assert!(matches!(
                decoder.partial_decode(byte_range, &options, &Resources::default()),
                Err(CodecError::InvalidByteRangeError(_))
            ));
        }

        // The interval end overflows
        let decoder = ByteIntervalPartialDecoder::new(input, u64::MAX - 1, 2);
        for byte_range in [
            ByteRange::FromStart(0, Some(1)),
            ByteRange::FromStart(1, None),
            ByteRange::Suffix(1),
        ] {
            assert!(matches!(
                decoder.partial_decode(byte_range, &options, &Resources::default()),
                Err(CodecError::InvalidByteRangeError(_))
            ));
        }
    }

    #[cfg(feature = "async")]
    fn async_decode(byte_range: ByteRange) -> Vec<u8> {
        let input: Arc<dyn AsyncBytesPartialDecoderTraits> =
            Arc::new((0..10u8).collect::<Vec<u8>>());
        let decoder = AsyncByteIntervalPartialDecoder::new(input, 2, 5);
        futures::executor::block_on(decoder.partial_decode(
            byte_range,
            &CodecOptions::default(),
            &Resources::default(),
        ))
        .unwrap()
        .unwrap()
        .to_vec()
    }

    #[cfg(feature = "async")]
    #[test]
    fn async_byte_interval_partial_decoder() {
        assert_eq!(
            async_decode(ByteRange::FromStart(0, None)),
            vec![2, 3, 4, 5, 6]
        );
        assert_eq!(
            async_decode(ByteRange::FromStart(1, None)),
            vec![3, 4, 5, 6]
        );
        assert_eq!(async_decode(ByteRange::FromStart(1, Some(2))), vec![3, 4]);
        assert_eq!(async_decode(ByteRange::Suffix(2)), vec![5, 6]);
    }
}
