#![allow(missing_docs)]

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::ReshapeDim;
use zarrs::array::{ArrayBuilder, ArraySubset, ChunkShapeTraits, data_type};
use zarrs::storage::ReadableStorageTraits;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs_codec::{ArrayToArrayCodecTraits, BytesToBytesCodecTraits, CodecOptions};

/// Test sync partial encoding for array-to-array codecs in isolation
fn test_array_to_array_codec_sync_partial_encoding<
    T: ArrayToArrayCodecTraits + Send + Sync + 'static,
>(
    codec: Arc<T>,
    codec_name: &str,
    supports_partial_encoding: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let opt = CodecOptions::default().with_experimental_partial_encoding(true);

    let store = Arc::new(MemoryStore::default());
    let store_perf = Arc::new(PerformanceMetricsStorageAdapter::new(store.clone()));

    let array_path = "/test_array";
    let mut builder = ArrayBuilder::new(
        vec![8, 8], // array shape
        vec![4, 4], // chunk shape
        data_type::float32(),
        -1.0,
    );

    // Set the codec being tested
    builder.array_to_array_codecs(vec![codec]);

    let array = builder.build(store_perf.clone(), array_path).unwrap();

    let chunk_key = array.chunk_key_encoding().encode(&[0, 0]);

    // Verify the chunk doesn't exist initially
    assert!(store.get(&chunk_key).unwrap().is_none());
    assert_eq!(store_perf.reads(), 0);
    assert_eq!(store_perf.bytes_read(), 0);

    // Store a subset of elements
    let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
    let elements = vec![10.0f32, 20.0, 30.0, 40.0];
    array
        .store_array_subset_opt(&subset, &elements, &opt)
        .unwrap();

    // Verify that data was written
    let writes_after_store = store_perf.writes();
    let bytes_written_after_store = store_perf.bytes_written();

    println!(
        "Codec {codec_name}: Writes after store: {writes_after_store}, Bytes written: {bytes_written_after_store}"
    );

    assert!(
        writes_after_store > 0,
        "Codec {codec_name} should have written data"
    );

    // Get the full chunk size for comparison
    let full_chunk_size = array.chunk_shape(&[0, 0]).unwrap().num_elements_usize()
        * array.data_type().fixed_size().unwrap();

    store_perf.reset();

    // Retrieve and verify the data
    let retrieved = array.retrieve_array_subset::<Vec<f32>>(&subset).unwrap();
    assert_eq!(retrieved, elements, "Codec {codec_name} round-trip failed");

    // Test partial encoding by storing overlapping data
    let subset2 = ArraySubset::new_with_ranges(&[0..2, 0..2]);
    let elements2 = vec![100f32, 200.0, 300.0, 400.0];

    array
        .store_array_subset_opt(&subset2, &elements2, &opt)
        .unwrap();

    let writes_after_partial = store_perf.writes();
    let bytes_written_after_partial = store_perf.bytes_written();
    let reads_after_partial = store_perf.reads();
    let bytes_read_after_partial = store_perf.bytes_read();

    println!(
        "Codec {codec_name}: Writes after partial update: {writes_after_partial}, Bytes written: {bytes_written_after_partial}, Reads: {reads_after_partial}, Bytes read: {bytes_read_after_partial}"
    );

    // For array-to-array codecs that support partial encoding, verify efficient behavior
    if supports_partial_encoding {
        if reads_after_partial > 0 && bytes_read_after_partial > 0 {
            // Should read less than full chunk if partial encoding is working efficiently
            if bytes_read_after_partial < full_chunk_size {
                println!(
                    "Codec {codec_name}: ✓ Confirmed partial encoding - read only {bytes_read_after_partial} of {full_chunk_size} bytes"
                );
            } else {
                panic!(
                    "Codec {codec_name}: ⚠ Expected partial encoding but read full {bytes_read_after_partial} bytes"
                );
            }
        }
    } else {
        println!("Codec {codec_name}: Info: Does not support partial encoding");
    }

    // Retrieve the full chunk to verify overlapping data was handled correctly
    let full_chunk = array.retrieve_chunk::<Vec<f32>>(&[0, 0]).unwrap();
    assert_eq!(
        full_chunk,
        vec![
            100.0, 200.0, -1.0, -1.0, //
            300.0, 400.0, 20.0, -1.0, //
            -1.0, 30.0, 40.0, -1.0, //
            -1.0, -1.0, -1.0, -1.0, //
        ]
    );

    // Test partial encoder methods
    let partial_encoder = array.partial_encoder(&[0, 0], &opt).unwrap();
    assert!(partial_encoder.exists().unwrap());
    let encoder_size_held = partial_encoder.size_held();
    println!("Codec {codec_name} partial encoder size_held(): {encoder_size_held}");
    partial_encoder.erase().unwrap();
    assert!(!partial_encoder.exists().unwrap());

    Ok(())
}

/// Test sync partial encoding for bytes-to-bytes codecs in isolation  
fn test_bytes_to_bytes_codec_sync_partial_encoding<
    T: BytesToBytesCodecTraits + Send + Sync + 'static,
>(
    codec: Arc<T>,
    codec_name: &str,
    supports_partial_encoding: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let opt = CodecOptions::default().with_experimental_partial_encoding(true);

    let store = Arc::new(MemoryStore::default());
    let store_perf = Arc::new(PerformanceMetricsStorageAdapter::new(store.clone()));

    let array_path = "/test_array";
    let mut builder = ArrayBuilder::new(
        vec![8, 8], // array shape
        vec![4, 4], // chunk shape
        data_type::float32(),
        -1.0,
    );

    // Set the codec being tested
    builder.bytes_to_bytes_codecs(vec![codec]);

    let array = builder.build(store_perf.clone(), array_path).unwrap();

    let chunk_key = array.chunk_key_encoding().encode(&[0, 0]);

    // Verify the chunk doesn't exist initially
    assert!(store.get(&chunk_key).unwrap().is_none());

    // Store a subset of elements
    let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
    let elements = vec![10f32, 20f32, 30f32, 40f32];

    let initial_reads = store_perf.reads();
    let initial_bytes_read = store_perf.bytes_read();

    array
        .store_array_subset_opt(&subset, &elements, &opt)
        .unwrap();

    let writes_after_store = store_perf.writes();
    let bytes_written_after_store = store_perf.bytes_written();
    let reads_after_store = store_perf.reads();
    let bytes_read_after_store = store_perf.bytes_read();

    println!(
        "Codec {}: Initial store - Writes: {}, Bytes written: {}, Reads: {}, Bytes read: {}",
        codec_name,
        writes_after_store,
        bytes_written_after_store,
        reads_after_store - initial_reads,
        bytes_read_after_store - initial_bytes_read
    );

    assert!(
        writes_after_store > 0,
        "Codec {codec_name} should have written data"
    );

    // Get the chunk size for comparison
    let full_chunk_size = array.chunk_shape(&[0, 0]).unwrap().num_elements_usize()
        * array.data_type().fixed_size().unwrap();

    store_perf.reset();

    // Test partial encoding with overlapping data
    let subset2 = ArraySubset::new_with_ranges(&[0..2, 0..2]);
    let elements2 = vec![100f32, 200f32, 300f32, 400f32];

    array
        .store_array_subset_opt(&subset2, &elements2, &opt)
        .unwrap();

    let writes_after_partial = store_perf.writes();
    let bytes_written_after_partial = store_perf.bytes_written();
    let reads_after_partial = store_perf.reads();
    let bytes_read_after_partial = store_perf.bytes_read();

    println!(
        "Codec {codec_name}: Partial update - Writes: {writes_after_partial}, Bytes written: {bytes_written_after_partial}, Reads: {reads_after_partial}, Bytes read: {bytes_read_after_partial}"
    );

    // For bytes-to-bytes codecs, verify expected partial encoding behavior
    if supports_partial_encoding {
        if reads_after_partial > 0 && bytes_read_after_partial > 0 {
            if bytes_read_after_partial < full_chunk_size {
                println!(
                    "Codec {codec_name}: ✓ Confirmed partial encoding - read only {bytes_read_after_partial} of {full_chunk_size} bytes"
                );
            } else {
                panic!(
                    "Codec {codec_name}: ⚠ Expected partial encoding but read full {bytes_read_after_partial} bytes"
                );
            }
        }
    } else {
        // Most bytes-to-bytes codecs (compression, checksums) don't support partial encoding
        println!(
            "Codec {}: Info: Does not support partial encoding (expected for {}-type codec)",
            codec_name,
            if codec_name.contains("crc")
                || codec_name.contains("adler")
                || codec_name.contains("fletcher")
            {
                "checksum"
            } else {
                "compression"
            }
        );

        if reads_after_partial > 0 && bytes_read_after_partial == full_chunk_size {
            println!(
                "Codec {codec_name}: ✓ Expected behavior - full chunk read for recompression/rechecksum"
            );
        } else if reads_after_partial == 0 {
            panic!(
                "Codec {codec_name}: ⚠ No reads during partial update - may indicate full rewrite strategy"
            );
        }
    }

    // Retrieve and verify the final data
    let full_chunk = array.retrieve_chunk::<Vec<f32>>(&[0, 0]).unwrap();
    assert_eq!(
        full_chunk,
        vec![
            100.0, 200.0, -1.0, -1.0, //
            300.0, 400.0, 20.0, -1.0, //
            -1.0, 30.0, 40.0, -1.0, //
            -1.0, -1.0, -1.0, -1.0, //
        ]
    );

    // Test partial encoder methods
    let partial_encoder = array.partial_encoder(&[0, 0], &opt).unwrap();
    assert!(partial_encoder.exists().unwrap());
    let encoder_size_held = partial_encoder.size_held();
    println!("Codec {codec_name} partial encoder size_held(): {encoder_size_held}");
    partial_encoder.erase().unwrap();
    assert!(!partial_encoder.exists().unwrap());

    Ok(())
}

// Array-to-Array Codec Tests

#[cfg(feature = "bitround")]
#[test]
fn test_bitround_sync_partial_encoding() {
    use zarrs::array::codec::BitroundCodec;
    use zarrs::metadata_ext::codec::bitround::BitroundCodecConfiguration;

    let config: BitroundCodecConfiguration = serde_json::from_str(r#"{"keepbits": 8}"#).unwrap();
    let codec = Arc::new(BitroundCodec::new_with_configuration(&config).unwrap());

    // Bitround supports partial encoding (confirmed from codec implementation)
    test_array_to_array_codec_sync_partial_encoding(codec, "bitround", true).unwrap();
}

#[cfg(feature = "transpose")]
#[test]
fn test_transpose_sync_partial_encoding() {
    use zarrs::array::codec::TransposeCodec;
    use zarrs::metadata_ext::codec::transpose::TransposeOrder;

    let order = TransposeOrder::new(&[1, 0]).unwrap();
    let codec = Arc::new(TransposeCodec::new(order));

    // Transpose supports partial encoding (confirmed from codec implementation)
    test_array_to_array_codec_sync_partial_encoding(codec, "transpose", true).unwrap();
}

#[test]
#[ignore = "partial encoding with reshape is not yet supported"] // FIXME
fn test_reshape_sync_partial_encoding() {
    use zarrs::array::codec::ReshapeCodec;
    use zarrs::metadata_ext::codec::reshape::ReshapeShape;

    let shape = vec![
        ReshapeDim::from(NonZeroU64::try_from(1).unwrap()),
        ReshapeDim::auto(),
    ];
    let reshape_shape = ReshapeShape::new(shape).unwrap();
    let codec = Arc::new(ReshapeCodec::new(reshape_shape));

    // Reshape supports partial encoding (confirmed from codec implementation)
    test_array_to_array_codec_sync_partial_encoding(codec, "reshape", true).unwrap();
}

#[test]
fn test_squeeze_sync_partial_encoding() {
    use zarrs::array::codec::SqueezeCodec;

    let codec = Arc::new(SqueezeCodec::default());

    // Squeeze supports partial encoding (confirmed from codec implementation)
    test_array_to_array_codec_sync_partial_encoding(codec, "squeeze", true).unwrap();
}

#[test]
#[ignore = "partial encoding with fixedscaleoffset is not yet supported"] // FIXME
fn test_fixedscaleoffset_sync_partial_encoding() {
    use zarrs::array::codec::FixedScaleOffsetCodec;
    use zarrs::metadata_ext::codec::fixedscaleoffset::FixedScaleOffsetCodecConfiguration;

    let config: FixedScaleOffsetCodecConfiguration =
        serde_json::from_str(r#"{"offset": 100.0, "scale": 2, "dtype": "f4", "astype": "f8"}"#)
            .unwrap();
    let codec = Arc::new(FixedScaleOffsetCodec::new_with_configuration(&config).unwrap());

    test_array_to_array_codec_sync_partial_encoding(codec, "fixedscaleoffset", true).unwrap();
}

// Bytes-to-Bytes Codec Tests

#[cfg(feature = "gzip")]
#[test]
fn test_gzip_sync_partial_encoding() {
    use zarrs::array::codec::GzipCodec;

    let codec = Arc::new(GzipCodec::new(5).unwrap());

    // Gzip does not support partial encoding due to compression
    test_bytes_to_bytes_codec_sync_partial_encoding(codec, "gzip", false).unwrap();
}

#[cfg(feature = "zstd")]
#[test]
fn test_zstd_sync_partial_encoding() {
    use zarrs::array::codec::ZstdCodec;

    let codec = Arc::new(ZstdCodec::new(5, true));

    // Zstd does not support partial encoding due to compression
    test_bytes_to_bytes_codec_sync_partial_encoding(codec, "zstd", false).unwrap();
}

#[cfg(feature = "blosc")]
#[test]
fn test_blosc_sync_partial_encoding() {
    use zarrs::array::codec::BloscCodec;
    use zarrs::metadata_ext::codec::blosc::{
        BloscCompressionLevel, BloscCompressor, BloscShuffleMode,
    };

    let codec = Arc::new(
        BloscCodec::new(
            BloscCompressor::BloscLZ,
            BloscCompressionLevel::try_from(5u8).unwrap(),
            None,
            BloscShuffleMode::NoShuffle,
            None,
        )
        .unwrap(),
    );

    // Blosc does not support partial encoding due to compression
    test_bytes_to_bytes_codec_sync_partial_encoding(codec, "blosc", false).unwrap();
}

#[cfg(feature = "bz2")]
#[test]
fn test_bz2_sync_partial_encoding() {
    use zarrs::array::codec::Bz2Codec;
    use zarrs::metadata_ext::codec::bz2::Bz2CompressionLevel;

    let codec = Arc::new(Bz2Codec::new(Bz2CompressionLevel::try_from(5u8).unwrap()));

    // Bz2 does not support partial encoding due to compression
    test_bytes_to_bytes_codec_sync_partial_encoding(codec, "bz2", false).unwrap();
}

#[cfg(feature = "crc32c")]
#[test]
fn test_crc32c_sync_partial_encoding() {
    use zarrs::array::codec::Crc32cCodec;

    let codec = Arc::new(Crc32cCodec::new());

    // CRC32C is a checksum codec - does not support partial encoding
    test_bytes_to_bytes_codec_sync_partial_encoding(codec, "crc32c", false).unwrap();
}

#[cfg(feature = "adler32")]
#[test]
fn test_adler32_sync_partial_encoding() {
    use zarrs::array::codec::Adler32Codec;

    let codec = Arc::new(Adler32Codec::default());

    // Adler32 is a checksum codec - does not support partial encoding
    test_bytes_to_bytes_codec_sync_partial_encoding(codec, "adler32", false).unwrap();
}

#[cfg(feature = "fletcher32")]
#[test]
fn test_fletcher32_sync_partial_encoding() {
    use zarrs::array::codec::Fletcher32Codec;

    let codec = Arc::new(Fletcher32Codec);

    // Fletcher32 is a checksum codec - does not support partial encoding
    test_bytes_to_bytes_codec_sync_partial_encoding(codec, "fletcher32", false).unwrap();
}

#[test]
fn test_shuffle_sync_partial_encoding() {
    use zarrs::array::codec::ShuffleCodec;
    use zarrs::metadata_ext::codec::shuffle::ShuffleCodecConfiguration;

    let config: ShuffleCodecConfiguration = serde_json::from_str(r#"{"elementsize": 2}"#).unwrap();
    let codec = Arc::new(ShuffleCodec::new_with_configuration(&config).unwrap());

    // TODO: Shuffle rearranges bytes - may not support efficient partial encoding
    test_bytes_to_bytes_codec_sync_partial_encoding(codec, "shuffle", false).unwrap();
}

#[cfg(feature = "zlib")]
#[test]
fn test_zlib_sync_partial_encoding() {
    use zarrs::array::codec::ZlibCodec;
    use zarrs::metadata_ext::codec::zlib::ZlibCompressionLevel;
    let codec = Arc::new(ZlibCodec::new(ZlibCompressionLevel::try_from(5u8).unwrap()));

    // Zlib does not support partial encoding due to compression
    test_bytes_to_bytes_codec_sync_partial_encoding(codec, "zlib", false).unwrap();
}

#[cfg(feature = "gdeflate")]
#[test]
fn test_gdeflate_sync_partial_encoding() {
    use zarrs::array::codec::GDeflateCodec;

    let codec = Arc::new(GDeflateCodec::new(5).unwrap());

    // Gdeflate does not support partial encoding due to compression
    test_bytes_to_bytes_codec_sync_partial_encoding(codec, "gdeflate", false).unwrap();
}

// Combined codec chain test
#[test]
fn test_codec_chain_sync_partial_encoding() {
    let opt = CodecOptions::default().with_experimental_partial_encoding(true);

    let store = Arc::new(MemoryStore::default());
    let store_perf = Arc::new(PerformanceMetricsStorageAdapter::new(store.clone()));

    let array_path = "/test_chain";
    let mut builder = ArrayBuilder::new(
        vec![8, 8], // array shape
        vec![4, 4], // chunk shape
        data_type::float32(),
        -1.0,
    );

    // Set up a codec chain with multiple codecs
    #[cfg(feature = "transpose")]
    {
        use zarrs::array::codec::TransposeCodec;
        use zarrs::metadata_ext::codec::transpose::TransposeOrder;

        let order = TransposeOrder::new(&[1, 0]).unwrap();
        let transpose_codec = Arc::new(TransposeCodec::new(order));
        builder.array_to_array_codecs(vec![transpose_codec]);
    }

    #[cfg(feature = "gzip")]
    {
        use zarrs::array::codec::GzipCodec;
        let gzip_codec = Arc::new(GzipCodec::new(1).unwrap());
        builder.bytes_to_bytes_codecs(vec![gzip_codec]);
    }

    let array = builder.build(store_perf.clone(), array_path).unwrap();

    // Test storing data with the codec chain
    let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
    let elements = vec![10f32, 20f32, 30f32, 40f32];

    array
        .store_array_subset_opt(&subset, &elements, &opt)
        .unwrap();

    let writes_after_store = store_perf.writes();
    let bytes_written_after_store = store_perf.bytes_written();

    println!(
        "Codec chain: Writes after store: {writes_after_store}, Bytes written: {bytes_written_after_store}"
    );

    assert!(
        writes_after_store > 0,
        "Codec chain should have written data"
    );

    // Verify round-trip
    let retrieved = array.retrieve_array_subset::<Vec<f32>>(&subset).unwrap();
    assert_eq!(retrieved, elements, "Codec chain round-trip failed");

    store_perf.reset();

    // Test partial update
    let subset2 = ArraySubset::new_with_ranges(&[0..2, 0..2]);
    let elements2 = vec![100f32, 200f32, 300f32, 400f32];

    array
        .store_array_subset_opt(&subset2, &elements2, &opt)
        .unwrap();

    let writes_after_partial = store_perf.writes();
    let bytes_written_after_partial = store_perf.bytes_written();
    let reads_after_partial = store_perf.reads();
    let bytes_read_after_partial = store_perf.bytes_read();

    println!(
        "Codec chain partial update: Writes: {writes_after_partial}, Bytes written: {bytes_written_after_partial}, Reads: {reads_after_partial}, Bytes read: {bytes_read_after_partial}"
    );

    // Verify data integrity after partial update
    let full_chunk = array.retrieve_chunk::<Vec<f32>>(&[0, 0]).unwrap();
    assert_eq!(
        full_chunk,
        vec![
            100.0, 200.0, -1.0, -1.0, //
            300.0, 400.0, 20.0, -1.0, //
            -1.0, 30.0, 40.0, -1.0, //
            -1.0, -1.0, -1.0, -1.0, //
        ]
    );

    // Test partial encoder methods
    let partial_encoder = array.partial_encoder(&[0, 0], &opt).unwrap();
    assert!(partial_encoder.exists().unwrap());
    let encoder_size_held = partial_encoder.size_held();
    println!(
        "Codec chain partial encoder size_held(): {encoder_size_held}"
    );
    partial_encoder.erase().unwrap();
    assert!(!partial_encoder.exists().unwrap());
}
