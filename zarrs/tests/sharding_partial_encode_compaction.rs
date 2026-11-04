#![allow(missing_docs)]
#![cfg(feature = "crc32c")]

use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;

use zarrs::array::codec::Crc32cCodec;
use zarrs::array::codec::array_to_bytes::sharding::{
    ShardingCodecBuilder, ShardingCodecOptions, SubchunkWriteOrder,
};
use zarrs::array::{
    ArrayBuilder, ArrayBytes, ArraySubset, CodecOptions, CodecSpecificOptions, data_type,
};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::storage::store::MemoryStore;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// A single shard of 2x2 subchunks.
const SHAPE: [u64; 2] = [4, 4];

/// Partial encodes of `(subset, value)`, where a `value` of 0 writes the fill value.
const UPDATES: [([Range<u64>; 2], u8); 8] = [
    // Fully covers subchunk 1
    ([0..2, 2..4], 20),
    // Straddles subchunks 0 and 1, retaining 2 and 3
    ([0..1, 0..3], 30),
    ([0..1, 0..3], 40),
    // Removes subchunk 2
    ([2..4, 0..2], 0),
    // Fully covers subchunk 0, without resurrecting subchunk 2
    ([0..2, 0..2], 50),
    // Straddles subchunks 2 and 3, reading the removed subchunk 2 as fill
    ([3..4, 1..3], 60),
    // Erases the shard
    ([0..4, 0..4], 0),
    // Straddles every subchunk of a missing shard
    ([1..3, 1..3], 70),
];

const WRITE_ORDERS: [SubchunkWriteOrder; 3] = [
    SubchunkWriteOrder::Lexicographic,
    SubchunkWriteOrder::Colexicographic,
    SubchunkWriteOrder::Morton,
];

fn configurations() -> impl Iterator<Item = (ShardingIndexLocation, SubchunkWriteOrder)> {
    [ShardingIndexLocation::Start, ShardingIndexLocation::End]
        .into_iter()
        .flat_map(|location| WRITE_ORDERS.map(|write_order| (location, write_order)))
}

fn builder(location: ShardingIndexLocation, write_order: SubchunkWriteOrder) -> ArrayBuilder {
    let data_type = data_type::uint8();
    let mut sharding = ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type);
    sharding.index_location(location);
    let mut builder = ArrayBuilder::new(SHAPE.to_vec(), SHAPE.to_vec(), data_type, 0u8);
    builder.array_to_bytes_codec(sharding.build_arc());
    // The checksum codec rewrites the entire shard, so the shard cannot be partially encoded.
    builder.bytes_to_bytes_codecs(vec![Arc::new(Crc32cCodec::new())]);
    builder.codec_specific_options(
        CodecSpecificOptions::default()
            .with_option(ShardingCodecOptions::default().with_subchunk_write_order(write_order)),
    );
    builder
}

/// Apply an update to the `expected` array elements and return the update elements.
fn update(expected: &mut [u8], ranges: &[Range<u64>; 2], value: u8) -> (ArraySubset, Vec<u8>) {
    let subset = ArraySubset::new_with_ranges(ranges);
    let mut elements = Vec::new();
    for i in ranges[0].clone() {
        for j in ranges[1].clone() {
            let element = if value == 0 {
                0
            } else {
                value + u8::try_from(elements.len()).unwrap()
            };
            expected[usize::try_from(i * SHAPE[1] + j).unwrap()] = element;
            elements.push(element);
        }
    }
    (subset, elements)
}

#[test]
fn sharding_partial_encode_compacts_whole_rewrites() -> TestResult {
    for (location, write_order) in configurations() {
        let config = format!("{location:?} {write_order:?}");
        let array = builder(location, write_order).build(Arc::new(MemoryStore::default()), "/")?;
        let reference =
            builder(location, write_order).build(Arc::new(MemoryStore::default()), "/")?;
        let mut expected: Vec<u8> = (1..=16).collect();
        array.store_chunk(&[0, 0], expected.clone())?;
        let encoder = array.partial_encoder(&[0, 0])?;
        assert!(!encoder.supports_partial_encode());
        for (ranges, value) in UPDATES {
            let (subset, elements) = update(&mut expected, &ranges, value);
            encoder.partial_encode(
                &subset,
                &ArrayBytes::new_flen(elements),
                &CodecOptions::default(),
            )?;
            assert_eq!(
                array.retrieve_chunk::<Vec<u8>>(&[0, 0])?,
                expected,
                "{config} {ranges:?}"
            );
            // The shard is compact and laid out like a whole shard encode
            reference.store_chunk(&[0, 0], expected.clone())?;
            assert_eq!(
                array.retrieve_encoded_chunk(&[0, 0])?,
                reference.retrieve_encoded_chunk(&[0, 0])?,
                "{config} {ranges:?}"
            );
        }
    }
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_sharding_partial_encode_compacts_whole_rewrites() -> TestResult {
    use zarrs::storage::store::AsyncMemoryStore;
    for (location, write_order) in configurations() {
        let config = format!("{location:?} {write_order:?}");
        let array = builder(location, write_order).build(Arc::new(AsyncMemoryStore::new()), "/")?;
        let reference =
            builder(location, write_order).build(Arc::new(AsyncMemoryStore::new()), "/")?;
        let mut expected: Vec<u8> = (1..=16).collect();
        array.async_store_chunk(&[0, 0], expected.clone()).await?;
        let encoder = array.async_partial_encoder(&[0, 0]).await?;
        assert!(!encoder.supports_partial_encode());
        for (ranges, value) in UPDATES {
            let (subset, elements) = update(&mut expected, &ranges, value);
            encoder
                .partial_encode(
                    &subset,
                    &ArrayBytes::new_flen(elements),
                    &CodecOptions::default(),
                )
                .await?;
            assert_eq!(
                array.async_retrieve_chunk::<Vec<u8>>(&[0, 0]).await?,
                expected,
                "{config} {ranges:?}"
            );
            reference
                .async_store_chunk(&[0, 0], expected.clone())
                .await?;
            assert_eq!(
                array.async_retrieve_encoded_chunk(&[0, 0]).await?,
                reference.async_retrieve_encoded_chunk(&[0, 0]).await?,
                "{config} {ranges:?}"
            );
        }
    }
    Ok(())
}
