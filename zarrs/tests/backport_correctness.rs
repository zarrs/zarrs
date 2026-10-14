//! Regression checks for compatible fixes backported from main.
use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::codec::{BytesCodec, PackBitsCodec, VlenCodec, VlenCodecConfiguration};
use zarrs::array::{
    ArrayBuilder, ArrayBytes, ArrayMetadata, ArrayMetadataOptions, ArraySubset,
    ArrayToBytesCodecTraits, CodecOptions, FillValue, data_type,
};
use zarrs::config::MetadataConvertVersion;
use zarrs::metadata::IntoDimensionName;
use zarrs::storage::store::MemoryStore;

#[test]
fn overflowing_codec_sizes_are_errors() {
    let codecs: [Arc<dyn ArrayToBytesCodecTraits>; 2] = [
        Arc::new(BytesCodec::default()),
        Arc::new(PackBitsCodec::default()),
    ];
    for codec in codecs {
        for shape in [
            vec![NonZeroU64::new(u64::MAX).unwrap()],
            vec![NonZeroU64::new(1 << 32).unwrap(); 2],
        ] {
            let dtype = data_type::uint16();
            let fill = FillValue::from(0u16);
            let options = CodecOptions::default();
            assert!(codec.encoded_representation(&shape, &dtype, &fill).is_err());
            assert!(
                codec
                    .encode(
                        ArrayBytes::new_flen(vec![0; 2]),
                        &shape,
                        &dtype,
                        &fill,
                        &options
                    )
                    .is_err()
            );
            assert!(
                codec
                    .decode(vec![0; 2].into(), &shape, &dtype, &fill, &options)
                    .is_err()
            );
        }
    }
    let codec = PackBitsCodec::default();
    let shape = [NonZeroU64::new(u64::MAX / 2).unwrap()];
    assert!(
        codec
            .encoded_representation(&shape, &data_type::uint4(), &FillValue::from(0u8))
            .is_err()
    );
}

#[test]
fn vlen_rejects_malformed_index_lengths() {
    for location in ["start", "end"] {
        for dtype in ["uint32", "uint64"] {
            let configuration: VlenCodecConfiguration = serde_json::from_value(serde_json::json!({
                "data_codecs": [{"name": "bytes"}],
                "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "index_data_type": dtype, "index_location": location
            }))
            .unwrap();
            let codec = VlenCodec::new_with_configuration(&configuration).unwrap();
            for length in [1u64, 100, u64::MAX] {
                let mut bytes = vec![0u8; 3];
                if location == "start" {
                    bytes.splice(0..0, length.to_le_bytes());
                } else {
                    bytes.extend_from_slice(&length.to_le_bytes());
                }
                assert!(
                    codec
                        .decode(
                            bytes.into(),
                            &[NonZeroU64::new(1).unwrap()],
                            &data_type::string(),
                            &FillValue::from(""),
                            &CodecOptions::default()
                        )
                        .is_err()
                );
            }
        }
    }
}

#[test]
fn dimension_names_are_stored_and_cleared() {
    let store = Arc::new(MemoryStore::new());
    let mut array = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type::uint8(), 0u8)
        .build(store.clone(), "/array")
        .unwrap();
    let names = Some(vec!["y".into_dimension_name(), "x".into_dimension_name()]);
    array
        .set_dimension_names(names.clone())
        .store_metadata()
        .unwrap();
    let reopened = zarrs::array::Array::open(store.clone(), "/array").unwrap();
    assert_eq!(reopened.dimension_names(), &names);
    array.set_dimension_names(None).store_metadata().unwrap();
    assert_eq!(
        zarrs::array::Array::open(store, "/array")
            .unwrap()
            .dimension_names(),
        &None
    );
}

#[test]
fn v2_conversion_keeps_dimension_names() {
    let metadata: ArrayMetadata = serde_json::from_value(serde_json::json!({
        "zarr_format": 2, "shape": [8], "chunks": [4], "dtype": "|u1", "compressor": null,
        "fill_value": 0, "order": "C", "filters": null
    }))
    .unwrap();
    let mut array =
        zarrs::array::Array::new_with_metadata(Arc::new(MemoryStore::new()), "/", metadata)
            .unwrap();
    let names = Some(vec!["x".into_dimension_name()]);
    array.set_dimension_names(names.clone());
    let metadata = array.metadata_opt(
        &ArrayMetadataOptions::default().with_metadata_convert_version(MetadataConvertVersion::V3),
    );
    assert!(matches!(metadata, ArrayMetadata::V3(metadata) if metadata.dimension_names == names));
    let array = array.to_v3().unwrap();
    assert!(
        matches!(array.metadata(), ArrayMetadata::V3(metadata) if metadata.dimension_names == names)
    );
    assert_eq!(array.subset_all(), ArraySubset::new_with_shape(vec![8]));
}
