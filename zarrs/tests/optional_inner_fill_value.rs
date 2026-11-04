#![allow(missing_docs)]

//! The fill value of the `optional` codec's inner data codec chain.
//!
//! `data_codecs` are bound with the unwrapped fill value of the `optional` data type if it is non-null
//! (e.g. `[42]` -> `42`). Sharding within `data_codecs` omits subchunks that consist entirely of that
//! fill value, so the encoded representation depends on it.
//! If the fill value is `null`, codecs whose encoded representation depends on the fill value
//! (e.g. `sharding_indexed`) are not permitted within `data_codecs`.
//! See the "Sharding" section of the `optional` codec specification.

use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs_codec::{ComputeResources, Resources};

use zarrs::array::codec::ShardingCodec;
use zarrs::array::{Array, ArrayCreateError, ArrayMetadata, ArraySubset, FillValue, data_type};
use zarrs::storage::store::MemoryStore;
use zarrs_codec::{CodecOptions, CodecSpecificOptions, UnboundArrayToBytesCodecTraits};
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata_ext::codec::sharding::ShardingCodecConfiguration;

const FILL: u8 = 42;

const BYTES: &str = r#"{"name": "bytes"}"#;

const BYTES_LE: &str = r#"{"name": "bytes", "configuration": {"endian": "little"}}"#;

const OPTIONAL_U8: &str = r#"{"name": "zarrs.optional", "configuration": {"name": "uint8"}}"#;

const OPTIONAL_OPTIONAL_U8: &str = r#"{
    "name": "zarrs.optional",
    "configuration": {"name": "zarrs.optional", "configuration": {"name": "uint8"}}
}"#;

fn try_array(
    data_type: &str,
    fill_value: &str,
    codecs: &str,
) -> Result<Array<MemoryStore>, ArrayCreateError> {
    let metadata: ArrayMetadata = serde_json::from_str(&format!(
        r#"{{
            "zarr_format": 3,
            "node_type": "array",
            "shape": [8],
            "data_type": {data_type},
            "chunk_grid": {{"name": "regular", "configuration": {{"chunk_shape": [8]}}}},
            "chunk_key_encoding": {{"name": "default"}},
            "fill_value": {fill_value},
            "codecs": {codecs}
        }}"#
    ))
    .unwrap();
    Array::new_with_metadata(Arc::new(MemoryStore::default()), "/", metadata)
}

/// Assert that array creation failed due to a fill value dependent codec in `data_codecs` with a null fill value.
fn assert_rejected(result: Result<Array<MemoryStore>, ArrayCreateError>) {
    let err = result.expect_err("array creation should fail").to_string();
    assert!(err.contains("requires a defined fill value"), "{err}");
}

/// An `optional` codec with `data_codecs` (a comma-separated list of codecs).
fn optional_codec(data_codecs: &str) -> String {
    format!(
        r#"{{
            "name": "zarrs.optional",
            "configuration": {{
                "mask_codecs": [{{"name": "packbits"}}],
                "data_codecs": [{data_codecs}]
            }}
        }}"#
    )
}

/// A `sharding_indexed` codec with a subchunk shape of `[2]`, the index at the end, and `codecs` (a comma-separated list of codecs).
fn sharding_codec(codecs: &str) -> String {
    format!(
        r#"{{
            "name": "sharding_indexed",
            "configuration": {{
                "chunk_shape": [2],
                "codecs": [{codecs}],
                "index_codecs": [{{"name": "bytes", "configuration": {{"endian": "little"}}}}],
                "index_location": "end"
            }}
        }}"#
    )
}

/// An `Option<u8>` array with `sharding_indexed` in `data_codecs`.
fn sharded_data_array(fill_value: &str) -> Result<Array<MemoryStore>, ArrayCreateError> {
    try_array(
        OPTIONAL_U8,
        fill_value,
        &format!("[{}]", optional_codec(&sharding_codec(BYTES))),
    )
}

/// Elements with 7 valid elements: `[0, 0 | 42, 42 | 1, 2 | 3]`, the last subchunk is clipped.
fn elements() -> Vec<Option<u8>> {
    vec![
        Some(0u8),
        Some(0),
        None,
        Some(FILL),
        Some(FILL),
        Some(1),
        Some(2),
        Some(3),
    ]
}

/// The fill value with `values` written from index `start`.
fn filled_with(start: usize, values: &[Option<u8>]) -> Vec<Option<u8>> {
    let mut elements = vec![Some(FILL); 8];
    elements[start..start + values.len()].copy_from_slice(values);
    elements
}

/// Split an encoded `optional` chunk into its encoded mask and encoded data.
fn split_optional(encoded: &[u8]) -> (&[u8], &[u8]) {
    let mask_len = usize::try_from(u64::from_le_bytes(encoded[0..8].try_into().unwrap())).unwrap();
    let data_len = usize::try_from(u64::from_le_bytes(encoded[8..16].try_into().unwrap())).unwrap();
    let (mask, data) = encoded[16..].split_at(mask_len);
    assert_eq!(data.len(), data_len);
    (mask, data)
}

/// Whether each subchunk of a shard (with the index at the end) is omitted.
fn omitted_subchunks(shard: &[u8], num_subchunks: usize) -> Vec<bool> {
    let index = &shard[shard.len() - num_subchunks * 16..];
    // An omitted subchunk has an offset and size of u64::MAX
    index
        .as_chunks::<16>()
        .0
        .iter()
        .map(|entry| *entry == [u8::MAX; 16])
        .collect()
}

/// Decode a shard of 7 `uint8` elements encoded with `sharding_codec(BYTES)`, assuming `fill_value`.
fn decode_shard_with_fill(
    shard: &[u8],
    fill_value: u8,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let configuration: ShardingCodecConfiguration =
        serde_json::from_str::<MetadataV3>(&sharding_codec(BYTES))?.to_typed_configuration()?;
    let codec = Arc::new(ShardingCodec::new_with_configuration(&configuration)?).with_context(
        data_type::uint8(),
        Some(FillValue::from(fill_value)),
        &CodecSpecificOptions::default(),
    )?;
    Ok(codec
        .decode(
            shard.to_vec().into(),
            &[NonZeroU64::new(7).unwrap()],
            &CodecOptions::default(),
            &ComputeResources::default(),
        )?
        .into_fixed()?
        .to_vec())
}

/// Sharding within `data_codecs` round trips, including the valid element count not evenly dividing
/// the subchunk shape, and partial writes that leave elements as the (non-null) fill value.
#[test]
#[allow(clippy::single_range_in_vec_init)]
fn optional_sharded_data_codecs_round_trip() -> Result<(), Box<dyn std::error::Error>> {
    for fill_value in [format!("[{FILL}]"), "[0]".to_string()] {
        let array = sharded_data_array(&fill_value)?;
        array.store_chunk(&[0], elements(), &Resources::default())?;
        assert_eq!(
            array.retrieve_chunk::<Vec<Option<u8>>>(&[0], &Resources::default())?,
            elements()
        );
    }

    // Partial write: the unwritten elements are the fill value Some(42), and are valid elements
    // passed to the data codec chain.
    let array = sharded_data_array(&format!("[{FILL}]"))?;
    let written = [Some(0u8), None, Some(0)];
    array.store_array_subset(
        &ArraySubset::new_with_ranges(&[2..5]),
        written.to_vec(),
        &Resources::default(),
    )?;
    assert_eq!(
        array.retrieve_chunk::<Vec<Option<u8>>>(&[0], &Resources::default())?,
        filled_with(2, &written)
    );
    // Valid elements (7): [42, 42 | 0, 0 | 42, 42 | 42]
    let encoded = array.retrieve_encoded_chunk(&[0])?.unwrap();
    let (_mask, shard) = split_optional(&encoded);
    assert_eq!(omitted_subchunks(shard, 4), vec![true, false, true, true]);
    Ok(())
}

/// With a non-null fill value, `data_codecs` are bound with the unwrapped fill value.
/// Subchunks of valid elements equal to the unwrapped fill value are omitted, and all-zero subchunks are stored.
#[test]
fn optional_sharded_data_codecs_omitted_subchunks() -> Result<(), Box<dyn std::error::Error>> {
    let array = sharded_data_array(&format!("[{FILL}]"))?;
    array.store_chunk(&[0], elements(), &Resources::default())?;

    let encoded = array.retrieve_encoded_chunk(&[0])?.unwrap();
    let (_mask, shard) = split_optional(&encoded);
    assert_eq!(omitted_subchunks(shard, 4), vec![false, true, false, false]);

    // The shard decodes correctly only with the unwrapped fill value
    assert_eq!(
        decode_shard_with_fill(shard, FILL)?,
        vec![0, 0, FILL, FILL, 1, 2, 3]
    );
    assert_eq!(decode_shard_with_fill(shard, 0)?, vec![0, 0, 0, 0, 1, 2, 3]);
    Ok(())
}

/// With a null fill value, the data codec chain fill value is undefined, so sharding is rejected within `data_codecs`.
#[test]
fn optional_sharded_data_codecs_null_fill_value_rejected() {
    assert_rejected(sharded_data_array("null"));
    // Including after other codecs in `data_codecs`
    #[cfg(feature = "transpose")]
    {
        let codecs = format!(
            "[{}]",
            optional_codec(&format!(
                r#"{{"name": "transpose", "configuration": {{"order": [0]}}}}, {}"#,
                sharding_codec(BYTES)
            ))
        );
        assert_rejected(try_array(OPTIONAL_U8, "null", &codecs));
        assert!(try_array(OPTIONAL_U8, &format!("[{FILL}]"), &codecs).is_ok());
    }
    // Codecs that do not depend on the fill value are permitted
    assert!(try_array(OPTIONAL_U8, "null", &format!("[{}]", optional_codec(BYTES))).is_ok());
}

/// The undefined fill value propagates through array-to-array codecs that transform the fill value.
#[test]
fn optional_cast_value_sharded_data_codecs_null_fill_value_rejected() {
    let codecs = format!(
        "[{}]",
        optional_codec(&format!(
            r#"{{"name": "cast_value", "configuration": {{"data_type": "uint16"}}}}, {}"#,
            sharding_codec(BYTES_LE)
        ))
    );
    assert_rejected(try_array(OPTIONAL_U8, "null", &codecs));
    assert!(try_array(OPTIONAL_U8, &format!("[{FILL}]"), &codecs).is_ok());
}

/// The fill value is unwrapped at each level of nested `optional` codecs.
#[test]
fn optional_nested_sharded_data_codecs_omitted_subchunks() -> Result<(), Box<dyn std::error::Error>>
{
    let codecs = format!(
        "[{}]",
        optional_codec(&optional_codec(&sharding_codec(BYTES)))
    );
    // Innermost valid elements (7): [0, 0 | 42, 42 | 1, 2 | 3]
    let elements: Vec<Option<Option<u8>>> = elements().into_iter().map(Some).collect();

    // `[[42]]` -> `[42]` -> `42`
    let array = try_array(OPTIONAL_OPTIONAL_U8, &format!("[[{FILL}]]"), &codecs)?;
    array.store_chunk(&[0], elements.clone(), &Resources::default())?;
    assert_eq!(
        array.retrieve_chunk::<Vec<Option<Option<u8>>>>(&[0], &Resources::default())?,
        elements
    );
    let encoded = array.retrieve_encoded_chunk(&[0])?.unwrap();
    let (_mask, inner) = split_optional(&encoded);
    let (_mask, shard) = split_optional(inner);
    assert_eq!(omitted_subchunks(shard, 4), vec![false, true, false, false]);

    // `[null]` -> `null`, and `null` -> `null`: sharding is rejected in the inner `data_codecs`
    assert_rejected(try_array(OPTIONAL_OPTIONAL_U8, "[null]", &codecs));
    assert_rejected(try_array(OPTIONAL_OPTIONAL_U8, "null", &codecs));
    Ok(())
}

/// With sharding as the outer codec, subchunks are bound with the array fill value.
/// Unwritten subchunks are omitted and decoded as the array fill value, not zero.
#[test]
#[allow(clippy::single_range_in_vec_init)]
fn optional_sharded_outer_unwritten_subchunks() -> Result<(), Box<dyn std::error::Error>> {
    let codecs = format!("[{}]", sharding_codec(&optional_codec(BYTES)));
    let array = try_array(OPTIONAL_U8, &format!("[{FILL}]"), &codecs)?;
    let written = [Some(0u8), None];
    array.store_array_subset(
        &ArraySubset::new_with_ranges(&[2..4]),
        written.to_vec(),
        &Resources::default(),
    )?;
    let mut expected = filled_with(2, &written);
    assert_eq!(
        array.retrieve_chunk::<Vec<Option<u8>>>(&[0], &Resources::default())?,
        expected
    );
    let shard = array.retrieve_encoded_chunk(&[0])?.unwrap();
    assert_eq!(omitted_subchunks(&shard, 4), vec![true, false, true, true]);

    // All-zero valid elements are not the fill value, so they are stored
    let written = [Some(0u8), Some(0)];
    array.store_array_subset(
        &ArraySubset::new_with_ranges(&[6..8]),
        written.to_vec(),
        &Resources::default(),
    )?;
    let shard = array.retrieve_encoded_chunk(&[0])?.unwrap();
    assert_eq!(omitted_subchunks(&shard, 4), vec![true, false, true, false]);
    expected[6..8].copy_from_slice(&written);
    assert_eq!(
        array.retrieve_chunk::<Vec<Option<u8>>>(&[0], &Resources::default())?,
        expected
    );
    Ok(())
}

/// Nested codecs bound with a fill value of their own are permitted to depend on it, even if the fill value is `null`.
#[test]
fn optional_null_fill_value_nested_defined_fill_values() -> Result<(), Box<dyn std::error::Error>> {
    // `vlen` index codecs are bound with a fill value of 0
    let vlen = format!(
        r#"{{
            "name": "zarrs.vlen",
            "configuration": {{
                "index_codecs": [{}],
                "data_codecs": [{BYTES}],
                "index_data_type": "uint32",
                "index_location": "end"
            }}
        }}"#,
        sharding_codec(BYTES_LE)
    );
    let optional_string = r#"{"name": "zarrs.optional", "configuration": {"name": "string"}}"#;
    let array = try_array(
        optional_string,
        "null",
        &format!("[{}]", optional_codec(&vlen)),
    )?;
    let elements: Vec<Option<String>> = ["a", "", "bc", "d", "", "", "efg", "h"]
        .iter()
        .enumerate()
        .map(|(i, s)| (i % 3 != 1).then(|| (*s).to_string()))
        .collect();
    array.store_chunk(&[0], elements.clone(), &Resources::default())?;
    assert_eq!(
        array.retrieve_chunk::<Vec<Option<String>>>(&[0], &Resources::default())?,
        elements
    );

    // `optional` mask codecs are bound with a fill value of `false`
    let inner_optional = format!(
        r#"{{
            "name": "zarrs.optional",
            "configuration": {{
                "mask_codecs": [{}],
                "data_codecs": [{BYTES}]
            }}
        }}"#,
        sharding_codec(r#"{"name": "packbits"}"#)
    );
    let array = try_array(
        OPTIONAL_OPTIONAL_U8,
        "null",
        &format!("[{}]", optional_codec(&inner_optional)),
    )?;
    let elements: Vec<Option<Option<u8>>> = vec![
        Some(None),
        Some(None),
        Some(Some(1)),
        None,
        Some(Some(2)),
        Some(None),
        None,
        Some(Some(3)),
    ];
    array.store_chunk(&[0], elements.clone(), &Resources::default())?;
    assert_eq!(
        array.retrieve_chunk::<Vec<Option<Option<u8>>>>(&[0], &Resources::default())?,
        elements
    );

    // The data codecs of an `optional(optional(T))` are bound with a `null` fill value, which is defined
    let array = try_array(
        OPTIONAL_OPTIONAL_U8,
        "null",
        &format!(
            "[{}]",
            optional_codec(&sharding_codec(&optional_codec(BYTES)))
        ),
    )?;
    array.store_chunk(&[0], elements.clone(), &Resources::default())?;
    assert_eq!(
        array.retrieve_chunk::<Vec<Option<Option<u8>>>>(&[0], &Resources::default())?,
        elements
    );
    // Valid outer elements (6): [null, null | 1, 2 | null, 3]
    let encoded = array.retrieve_encoded_chunk(&[0])?.unwrap();
    let (_mask, shard) = split_optional(&encoded);
    assert_eq!(omitted_subchunks(shard, 3), vec![true, false, false]);
    Ok(())
}
