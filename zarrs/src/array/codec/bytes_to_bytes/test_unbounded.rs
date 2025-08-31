//! The `test_unbounded` bytes to bytes codec.
//!
//! This is used in tests to validate behaviour for codecs with an unbounded encoded size.

mod test_unbounded_codec;
mod test_unbounded_partial_decoder;

pub use test_unbounded_codec::TestUnboundedCodec;

#[cfg(test)]
mod tests {
    use std::{borrow::Cow, sync::Arc};

    use crate::{
        array::{
            codec::{BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecOptions},
            BytesRepresentation,
        },
        storage::byte_range::ByteRange,
    };

    use super::*;

    #[test]
    fn codec_test_unbounded_round_trip1() {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec: TestUnboundedCodec = TestUnboundedCodec::new();

        let encoded = codec
            .encode(Cow::Borrowed(&bytes), &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(encoded, &bytes_representation, &CodecOptions::default())
            .unwrap();
        assert_eq!(bytes, decoded.to_vec());
    }

    #[test]
    fn codec_test_unbounded_partial_decode() {
        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec = Arc::new(TestUnboundedCodec::new());

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions = [
            ByteRange::FromStart(4, Some(4)),
            ByteRange::FromStart(10, Some(2)),
        ];

        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &bytes_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size(), input_handle.size()); // test unbounded partial decoder does not hold bytes
        let decoded_partial_chunk = partial_decoder
            .partial_decode_many(
                Box::new(decoded_regions.into_iter()),
                &CodecOptions::default(),
            )
            .unwrap()
            .unwrap()
            .concat();

        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .to_vec()
            .chunks_exact(size_of::<u16>())
            .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let answer: Vec<u16> = vec![2, 3, 5];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_test_unbounded_async_partial_decode() {
        use crate::array::codec::CodecOptions;

        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec = Arc::new(TestUnboundedCodec::new());

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions = [
            ByteRange::FromStart(4, Some(4)),
            ByteRange::FromStart(10, Some(2)),
        ];

        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .async_partial_decoder(
                input_handle,
                &bytes_representation,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
        let decoded_partial_chunk = partial_decoder
            .partial_decode_many(
                Box::new(decoded_regions.into_iter()),
                &CodecOptions::default(),
            )
            .await
            .unwrap()
            .unwrap()
            .concat();

        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .to_vec()
            .chunks_exact(size_of::<u16>())
            .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let answer: Vec<u16> = vec![2, 3, 5];
        assert_eq!(answer, decoded_partial_chunk);
    }
}
