#![allow(missing_docs)]

use zarrs_metadata::v3::ArrayMetadataV3;

#[test]
fn array_extensions() {
    let json = r#"{
    "zarr_format": 3,
    "node_type": "array",
    "data_type": "https://example.com/zarr/string",
    "fill_value": "",
    "chunk_key_encoding": {
        "name": "default",
        "configuration": { "separator": "." }
    },
    "codecs": [
        {
            "name": "https://numcodecs.dev/vlen-utf8"
        },
        {
            "name": "zstd",
            "configuration": {}
        }
    ],
    "chunk_grid": {
        "name": "regular",
        "configuration": { "chunk_shape": [ 32 ] }
    },
    "shape": [ 128 ],
    "dimension_names": [ "x" ],
    "attributes": {},
    "storage_transformers": [],
    "extensions": [
        {
            "name": "https://example.com/zarr/offset",
            "configuration": { "offset": [ 12 ] }
        },
        {
            "name": "https://example.com/zarr/array-statistics",
            "configuration": {
                "min": 5,
                "max": 12
            },
            "must_understand": false
        },
        {
            "name": "https://example.com/zarr/consolidated-metadata",
            "configuration": {},
            "must_understand": false
        }
    ]
}"#;

    let metadata: ArrayMetadataV3 = serde_json::from_str(json).unwrap();
    assert_eq!(metadata.data_type.name(), "https://example.com/zarr/string");
}
