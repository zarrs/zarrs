//! Test for runtime codec registration.

use std::borrow::Cow;
use std::sync::Arc;

use serial_test::serial;
use zarrs::array::codec::{
    BytesToBytesCodecTraits, Codec, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency, register_codec_v3,
    unregister_codec_v3,
};
use zarrs::array::{Array, ArrayBuilder, ArrayBytesRaw, BytesRepresentation, data_type};
use zarrs::metadata::Configuration;
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::store::MemoryStore;
use zarrs_plugin::{RuntimePlugin, ZarrVersion};

/// A simple passthrough codec for testing runtime registration.
#[derive(Clone, Debug)]
struct TestPassthroughCodec;

zarrs_plugin::impl_extension_aliases!(TestPassthroughCodec, v3: "test.passthrough", v2: "test.passthrough");

impl CodecTraits for TestPassthroughCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(Configuration::default())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(
    all(feature = "async", target_arch = "wasm32"),
    async_trait::async_trait(?Send)
)]
impl BytesToBytesCodecTraits for TestPassthroughCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn recommended_concurrency(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encode<'a>(
        &self,
        decoded_value: ArrayBytesRaw<'a>,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        Ok(decoded_value)
    }

    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        Ok(encoded_value)
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        *decoded_representation
    }
}

#[test]
#[serial]
fn codec_runtime_registration() {
    // Verify the codec is not registered initially
    let metadata = MetadataV3::new("test.passthrough");
    assert!(Codec::from_metadata(&metadata).is_err());

    // Register the codec at runtime
    let handle = register_codec_v3(RuntimePlugin::new(
        |name| name == "test.passthrough",
        |_metadata| Ok(Codec::BytesToBytes(Arc::new(TestPassthroughCodec))),
    ));

    // Verify the codec can now be created from metadata
    let codec = Codec::from_metadata(&metadata).unwrap();
    assert!(matches!(codec, Codec::BytesToBytes(_)));

    // Test encode/decode round-trip
    if let Codec::BytesToBytes(codec) = codec {
        let data = vec![1u8, 2, 3, 4, 5];
        let encoded = codec
            .encode(Cow::Borrowed(&data), &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(
                encoded,
                &BytesRepresentation::FixedSize(5),
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(data, decoded.as_ref());
    }

    // Unregister the codec
    assert!(unregister_codec_v3(&handle));

    // Verify the codec is no longer registered
    let metadata = MetadataV3::new("test.passthrough");
    assert!(Codec::from_metadata(&metadata).is_err());

    // Unregistering again should return false
    assert!(!unregister_codec_v3(&handle));
}

#[test]
#[serial]
fn codec_runtime_registration_array_roundtrip() {
    // Register the codec at runtime
    let handle = register_codec_v3(RuntimePlugin::new(
        |name| name == "test.passthrough",
        |_metadata| Ok(Codec::BytesToBytes(Arc::new(TestPassthroughCodec))),
    ));

    // Create a store and build an array with the runtime-registered codec
    let store = Arc::new(MemoryStore::default());
    let array_path = "/array";
    let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint8(), 0u8)
        .bytes_to_bytes_codecs(vec![Arc::new(TestPassthroughCodec)])
        .build(store.clone(), array_path)
        .expect("Failed to build array with runtime-registered codec");

    // Store the array metadata
    array
        .store_metadata()
        .expect("Failed to store array metadata");

    // Store some data
    let expected_full_data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    array
        .store_array_subset(&array.subset_all(), &expected_full_data)
        .expect("Failed to store full array subset");

    // Re-open the array from storage (simulates a fresh open)
    let reopened_array: Array<MemoryStore> =
        Array::open(store.clone(), array_path).expect("Failed to open array");

    // Retrieve the entire array and verify consistency (row-major order)
    let full_data: Vec<u8> = reopened_array
        .retrieve_array_subset(&zarrs::array::ArraySubset::new_with_ranges(&[0..4, 0..4]))
        .expect("Failed to retrieve full array");
    assert_eq!(full_data, expected_full_data);

    // Unregister the codec
    assert!(unregister_codec_v3(&handle));

    // Opening the array after unregistering should fail because the codec is no longer available
    let open_result: Result<Array<MemoryStore>, _> = Array::open(store.clone(), array_path);
    assert!(
        open_result.is_err(),
        "Opening array should fail after codec is unregistered"
    );
}
