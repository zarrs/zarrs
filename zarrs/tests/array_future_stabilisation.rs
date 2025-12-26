#![allow(missing_docs)]
#![cfg(feature = "bz2")]

use std::sync::Arc;

use zarrs::registry::codec::BZ2;
use zarrs::{
    array::{Array, ArrayMetadataOptions},
    config::{global_config, global_config_mut},
};
use zarrs_filesystem::FilesystemStore;

/// bz2 could stabilise as is, so test supporting that via the codec map
#[test]
fn array_future_stabilisation_bz2() {
    let alias = global_config()
        .codec_aliases_v3()
        .default_names
        .get(BZ2.into())
        .expect("bz2 in aliases")
        .to_string();

    global_config_mut()
        .codec_aliases_v3_mut()
        .default_names
        .entry(BZ2.into())
        .and_modify(|entry| {
            *entry = "stable.bz2".into();
        });

    let path = "tests/data/v3/array_bz2.zarr";
    let store = Arc::new(FilesystemStore::new(path).unwrap());
    let array = Array::open(store, "/").unwrap();
    let elements = array
        .retrieve_array_subset::<Vec<f32>>(&array.subset_all())
        .unwrap();
    assert_eq!(
        &elements,
        &[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, //
            20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, //
            30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, //
            40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, //
            50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, //
            60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, //
            70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, //
            80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, //
            90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, //
        ],
    );

    let mut options = ArrayMetadataOptions::default();
    assert!(
        array
            .metadata_opt(&options)
            .to_string()
            .contains(r#""numcodecs.bz2"#)
    );
    assert!(
        !array
            .metadata_opt(&options)
            .to_string()
            .contains(r#""stable.bz2"#)
    );

    options.set_convert_aliased_extension_names(true);
    assert!(
        array
            .metadata_opt(&options)
            .to_string()
            .contains(r#""stable.bz2"#)
    );

    global_config_mut()
        .codec_aliases_v3_mut()
        .default_names
        .entry(BZ2.into())
        .and_modify(|entry| {
            *entry = alias.into();
        });
}
