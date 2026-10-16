//! The `test_unbounded` bytes to bytes codec.
//!
//! This is used in tests to validate behaviour for codecs with an unbounded encoded size.

mod test_unbounded_codec;

pub use test_unbounded_codec::TestUnboundedCodec;

#[cfg(test)]
mod tests {
    use crate::array::Resources;
    use crate::array::codec::test_util::partial_decode_bytes;
    use std::sync::Arc;
    use zarrs_codec::CowBytes;

    use super::*;
    use crate::array::BytesRepresentation;
    use zarrs_codec::{BytesToBytesCodecTraits, CodecOptions};
    use zarrs_storage::byte_range::ByteRange;

    #[test]
    fn codec_test_unbounded_round_trip1() {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec: TestUnboundedCodec = TestUnboundedCodec::new();

        let encoded = codec
            .encode(
                CowBytes::Borrowed(&bytes),
                &CodecOptions::default(),
                &Resources::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(
                encoded,
                &bytes_representation,
                &CodecOptions::default(),
                &Resources::default(),
            )
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
            .encode(
                CowBytes::from(bytes),
                &CodecOptions::default(),
                &Resources::default(),
            )
            .unwrap();
        let decoded_regions = [
            ByteRange::FromStart(4, Some(4)),
            ByteRange::FromStart(10, Some(2)),
        ];

        let decoded_partial_chunk = partial_decode_bytes(
            codec.clone(),
            &encoded,
            bytes_representation.size().unwrap(),
            &decoded_regions,
        )
        .concat();

        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        let answer: Vec<u16> = vec![2, 3, 5];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[test]
    fn default_decode_into() {
        let codec = TestUnboundedCodec::new();
        let options = CodecOptions::default();
        let encoded = CowBytes::from(&b"decoded"[..]);
        let decode_into = |output: &mut [u8]| {
            codec.decode_into(
                encoded.clone(),
                &BytesRepresentation::UnboundedSize,
                output,
                &options,
                &Resources::default(),
            )
        };

        let mut output = [0; 7];
        decode_into(&mut output).unwrap();
        assert_eq!(&output, b"decoded");
        assert!(decode_into(&mut [0; 8]).is_err());
        assert!(decode_into(&mut [0; 6]).is_err());
    }
}
