//! The `blosc` bytes to bytes codec (Core).
//!
//! It uses the [blosc](https://www.blosc.org/) container format.
//!
//! ### Compatible Implementations
//! This is a core codec and should be compatible with all Zarr V3 implementations that support it.
//!
//! ### Specification
//! - <https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html>
//! - <https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/blosc>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `blosc`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `blosc`
//!
//! `zarrs` automatically converts Zarr V2 `blosc` metadata (without a `typesize` field) to Zarr V3.
//!
//! ### Codec `configuration` Example - [`BloscCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "cname": "lz4",
//!     "clevel": 1,
//!     "shuffle": "shuffle",
//!     "typesize": 4,
//!     "blocksize": 0
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::blosc::BloscCodecConfiguration;
//! # serde_json::from_str::<BloscCodecConfiguration>(JSON).unwrap();
//! ```

// NOTE: Zarr implementations MAY provide users an option to choose a shuffle mode automatically based on the typesize or other information, but MUST record in the metadata the mode that is chosen.
// TODO: Need to validate blosc typesize matches element size and also that endianness is specified if typesize > 1
mod blosc_codec;
mod blosc_partial_decoder;

#[cfg(not(target_arch = "wasm32"))]
#[path = "blosc/blosc_via_blosc_src.rs"]
mod blosc_impl;

#[cfg(target_arch = "wasm32")]
#[path = "blosc/blosc_via_blusc.rs"]
mod blosc_impl;

pub use blosc_codec::BloscCodec;
pub use blosc_impl::{
    BloscCodecConfiguration, BloscCodecConfigurationNumcodecs, BloscCodecConfigurationV1,
    BloscCompressionLevel, BloscCompressor, BloscError, BloscShuffleMode,
    BloscShuffleModeNumcodecs, blosc_compress_bytes, blosc_decompress_bytes,
    blosc_decompress_bytes_partial, blosc_nbytes, blosc_typesize, blosc_validate,
};
use zarrs_codec::{CodecPluginV2, CodecPluginV3};

zarrs_plugin::impl_extension_aliases!(BloscCodec, v3: "blosc", v2: "blosc");

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<BloscCodec>()
}

// Register the V2 codec.
inventory::submit! {
    CodecPluginV2::new::<BloscCodec>()
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use zarrs_codec::CowBytes;

    use super::*;
    use crate::array::{ArraySubset, BytesRepresentation, ChunkShapeTraits, Indexer, data_type};
    use zarrs_codec::{
        BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecError, CodecOptions,
    };
    use zarrs_storage::byte_range::ByteRange;

    const JSON_VALID1: &str = r#"
{
    "cname": "lz4",
    "clevel": 5,
    "shuffle": "shuffle",
    "typesize": 2,
    "blocksize": 0
}"#;

    const JSON_VALID2: &str = r#"
{
    "cname": "lz4",
    "clevel": 4,
    "shuffle": "bitshuffle",
    "typesize": 2,
    "blocksize": 0
}"#;

    const JSON_VALID3: &str = r#"
{
    "cname": "lz4",
    "clevel": 4,
    "shuffle": "noshuffle",
    "blocksize": 0
}"#;

    const JSON_INVALID1: &str = r#"
{
    "cname": "lz4",
    "clevel": 4,
    "shuffle": "bitshuffle",
    "typesize": 0,
    "blocksize": 0
}"#;

    fn codec_blosc_round_trip(json: &str) {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let codec_configuration: BloscCodecConfiguration = serde_json::from_str(json).unwrap();
        let codec = BloscCodec::new_with_configuration(&codec_configuration).unwrap();

        let encoded = codec
            .encode(CowBytes::Borrowed(&bytes), &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(encoded, &bytes_representation, &CodecOptions::default())
            .unwrap();
        assert_eq!(bytes, decoded.to_vec());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_blosc_decode_into() {
        let options = CodecOptions::default();
        let codec_configuration: BloscCodecConfiguration =
            serde_json::from_str(JSON_VALID1).unwrap();
        let codec = BloscCodec::new_with_configuration(&codec_configuration).unwrap();
        let decoded = crate::array::transmute_to_bytes_vec((0..4096u16).collect::<Vec<_>>());
        let len = decoded.len();
        let encoded = codec
            .encode(CowBytes::Borrowed(&decoded), &options)
            .unwrap();
        let decode_into = |encoded: CowBytes<'_>, output: &mut [u8]| {
            codec.decode_into(
                encoded,
                &BytesRepresentation::FixedSize(len as u64),
                output,
                &options,
            )
        };

        let mut output = vec![0; len];
        decode_into(encoded.clone(), &mut output).unwrap();
        assert_eq!(output, decoded);

        // An output of the wrong length is reported as an invalid length
        assert!(matches!(
            decode_into(encoded.clone(), &mut vec![0; len + 1]),
            Err(CodecError::UnexpectedChunkDecodedSize(_))
        ));
        assert!(matches!(
            decode_into(encoded, &mut output[..len - 1]),
            Err(CodecError::UnexpectedChunkDecodedSize(_))
        ));

        // Bytes that are not a blosc buffer
        assert!(decode_into(CowBytes::from(vec![0u8; 64]), &mut output).is_err());
    }

    /// Retrieving whole chunks decodes blosc directly into the output where the output is
    /// contiguous (chunks spanning full rows) and falls back otherwise (square chunks).
    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_blosc_array_round_trip() {
        use crate::array::ArrayBuilder;

        let codec_configuration: BloscCodecConfiguration =
            serde_json::from_str(JSON_VALID1).unwrap();
        let data: Vec<u16> = (0..64).collect();
        let subsets = [
            ArraySubset::new_with_ranges(&[0..8, 0..8]),
            ArraySubset::new_with_ranges(&[2..6, 0..8]),
            ArraySubset::new_with_ranges(&[1..7, 2..6]),
        ];
        for chunk_shape in [vec![2, 8], vec![4, 4]] {
            let codec = Arc::new(BloscCodec::new_with_configuration(&codec_configuration).unwrap());
            let array = ArrayBuilder::new(vec![8, 8], chunk_shape, data_type::uint16(), 0u16)
                .bytes_to_bytes_codecs(vec![codec])
                .build_arc(Arc::new(zarrs_storage::store::MemoryStore::new()), "/")
                .unwrap();
            array
                .store_array_subset(&array.subset_all(), &data)
                .unwrap();
            for subset in &subsets {
                // `data` is the linear index of each element
                let expected: Vec<u16> = subset
                    .linearised_indices(&[8, 8])
                    .unwrap()
                    .into_iter()
                    .map(|i| u16::try_from(i).unwrap())
                    .collect();
                assert_eq!(
                    array.retrieve_array_subset::<Vec<u16>>(subset).unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_blosc_round_trip1() {
        codec_blosc_round_trip(JSON_VALID1);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_blosc_round_trip2() {
        codec_blosc_round_trip(JSON_VALID2);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_blosc_round_trip3() {
        codec_blosc_round_trip(JSON_VALID3);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_blosc_round_trip_snappy() {
        let json = r#"
{
    "cname": "snappy",
    "clevel": 4,
    "shuffle": "noshuffle",
    "blocksize": 0
}"#;
        codec_blosc_round_trip(json);
    }

    #[test]
    #[should_panic]
    #[cfg_attr(miri, ignore)]
    fn codec_blosc_invalid_typesize_with_shuffling() {
        codec_blosc_round_trip(JSON_INVALID1);
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_blosc_partial_decode() {
        let shape = vec![NonZeroU64::new(2).unwrap(); 3];
        let data_type = data_type::uint16();
        let data_type_size = data_type.fixed_size().unwrap();
        let array_size = shape.num_elements_usize() * data_type_size;
        let bytes_representation = BytesRepresentation::FixedSize(array_size as u64);

        let elements: Vec<u16> = (0..shape.num_elements_usize() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: BloscCodecConfiguration =
            serde_json::from_str(JSON_VALID2).unwrap();
        let codec = Arc::new(BloscCodec::new_with_configuration(&codec_configuration).unwrap());

        let encoded = codec
            .encode(CowBytes::from(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions = ArraySubset::new_with_ranges(&[0..2, 1..2, 0..1])
            .iter_contiguous_byte_ranges(bytemuck::must_cast_slice(&shape), data_type_size)
            .unwrap()
            .map(ByteRange::new);
        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &bytes_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // blosc partial decoder does not hold bytes
        let decoded = partial_decoder
            .partial_decode_many(Box::new(decoded_regions), &CodecOptions::default())
            .unwrap()
            .unwrap()
            .concat();

        let decoded: Vec<u16> = decoded
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();

        let answer: Vec<u16> = vec![2, 6];
        assert_eq!(answer, decoded);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn codec_blosc_async_partial_decode() {
        use crate::array::Indexer;

        let shape = vec![NonZeroU64::new(2).unwrap(); 3];
        let data_type = data_type::uint16();
        let data_type_size = data_type.fixed_size().unwrap();
        let array_size = shape.num_elements_usize() * data_type_size;
        let bytes_representation = BytesRepresentation::FixedSize(array_size as u64);

        let elements: Vec<u16> = (0..shape.num_elements_usize() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);

        let codec_configuration: BloscCodecConfiguration =
            serde_json::from_str(JSON_VALID2).unwrap();
        let codec = Arc::new(BloscCodec::new_with_configuration(&codec_configuration).unwrap());

        let encoded = codec
            .encode(CowBytes::from(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions = ArraySubset::new_with_ranges(&[0..2, 1..2, 0..1])
            .iter_contiguous_byte_ranges(bytemuck::must_cast_slice(&shape), data_type_size)
            .unwrap()
            .map(ByteRange::new);
        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .async_partial_decoder(
                input_handle,
                &bytes_representation,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
        let decoded = partial_decoder
            .partial_decode_many(Box::new(decoded_regions), &CodecOptions::default())
            .await
            .unwrap()
            .unwrap()
            .concat();

        let decoded: Vec<u16> = decoded
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();

        let answer: Vec<u16> = vec![2, 6];
        assert_eq!(answer, decoded);
    }
}
