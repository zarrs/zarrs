#![allow(missing_docs)]
#![cfg(feature = "crc32c")]

use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs_codec::Resources;

use zarrs::array::codec::Crc32cCodec;
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::{ArrayBuilder, ArrayBytes, ArraySubset, CodecOptions, data_type};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::storage::store::MemoryStore;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn builder(location: ShardingIndexLocation) -> ArrayBuilder {
    let data_type = data_type::uint8();
    let mut sharding = ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap()], &data_type);
    sharding.index_location(location);
    let mut builder = ArrayBuilder::new(vec![8], vec![8], data_type, 0u8);
    builder.array_to_bytes_codec(sharding.build_arc());
    // The checksum codec rewrites the entire shard and adds a fixed number of bytes.
    builder.bytes_to_bytes_codecs(vec![Arc::new(Crc32cCodec::new())]);
    builder
}

#[test]
fn sharding_partial_encode_compacts_whole_rewrites() -> TestResult {
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        let array = builder(location).build(Arc::new(MemoryStore::default()), "/")?;
        array.store_chunk(&[0], (1..=8u8).collect::<Vec<_>>(), &Resources::default())?;
        let original_size = array.retrieve_encoded_chunk(&[0])?.unwrap().len();
        let encoder = array.partial_encoder(&[0], &Resources::default())?;

        for value in 42..46u8 {
            encoder.partial_encode(
                &ArraySubset::new_with_ranges(&[2..4]),
                &ArrayBytes::new_flen(vec![value, value + 1]),
                &CodecOptions::default(),
                &Resources::default(),
            )?;
            assert_eq!(
                array.retrieve_encoded_chunk(&[0])?.unwrap().len(),
                original_size
            );
            assert_eq!(
                array.retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())?,
                [1, 2, value, value + 1, 5, 6, 7, 8]
            );
        }
    }
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_sharding_partial_encode_compacts_whole_rewrites() -> TestResult {
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        let store = Arc::new(zarrs::storage::store::AsyncMemoryStore::new());
        let array = builder(location).build(store, "/")?;
        array
            .async_store_chunk(&[0], (1..=8u8).collect::<Vec<_>>(), &Resources::default())
            .await?;
        let original_size = array
            .async_retrieve_encoded_chunk(&[0])
            .await?
            .unwrap()
            .len();
        let encoder = array
            .async_partial_encoder(&[0], &Resources::default())
            .await?;

        for value in 42..46u8 {
            encoder
                .partial_encode(
                    &ArraySubset::new_with_ranges(&[2..4]),
                    &ArrayBytes::new_flen(vec![value, value + 1]),
                    &CodecOptions::default(),
                    &Resources::default(),
                )
                .await?;
            assert_eq!(
                array
                    .async_retrieve_encoded_chunk(&[0])
                    .await?
                    .unwrap()
                    .len(),
                original_size
            );
            assert_eq!(
                array
                    .async_retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                    .await?,
                [1, 2, value, value + 1, 5, 6, 7, 8]
            );
        }
    }
    Ok(())
}

#[test]
fn sharding_partial_encoder_reuse_after_erase() -> TestResult {
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        let array = builder(location)
            .bytes_to_bytes_codecs(vec![])
            .build(Arc::new(MemoryStore::default()), "/")?;
        let resources = Resources::default();
        let encoder = array.partial_encoder(&[0], &resources)?;
        encoder.partial_encode(
            &ArraySubset::new_with_ranges(&[0..2]),
            &ArrayBytes::new_flen(vec![1, 2]),
            &CodecOptions::default(),
            &resources,
        )?;
        assert_eq!(encoder.size_held(), 4 * 2 * size_of::<u64>());
        encoder.erase()?;
        assert_eq!(encoder.size_held(), 0);
        encoder.partial_encode(
            &ArraySubset::new_with_ranges(&[2..4]),
            &ArrayBytes::new_flen(vec![3, 4]),
            &CodecOptions::default(),
            &resources,
        )?;
        assert_eq!(
            array.retrieve_chunk::<Vec<u8>>(&[0], &resources)?,
            [0, 0, 3, 4, 0, 0, 0, 0]
        );
    }
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_sharding_partial_encoder_reuse_after_erase() -> TestResult {
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        let array = builder(location).bytes_to_bytes_codecs(vec![]).build(
            Arc::new(zarrs::storage::store::AsyncMemoryStore::new()),
            "/",
        )?;
        let resources = Resources::default();
        let encoder = array.async_partial_encoder(&[0], &resources).await?;
        encoder
            .partial_encode(
                &ArraySubset::new_with_ranges(&[0..2]),
                &ArrayBytes::new_flen(vec![1, 2]),
                &CodecOptions::default(),
                &resources,
            )
            .await?;
        assert_eq!(encoder.size_held(), 4 * 2 * size_of::<u64>());
        encoder.erase().await?;
        assert_eq!(encoder.size_held(), 0);
        encoder
            .partial_encode(
                &ArraySubset::new_with_ranges(&[2..4]),
                &ArrayBytes::new_flen(vec![3, 4]),
                &CodecOptions::default(),
                &resources,
            )
            .await?;
        assert_eq!(
            array
                .async_retrieve_chunk::<Vec<u8>>(&[0], &resources)
                .await?,
            [0, 0, 3, 4, 0, 0, 0, 0]
        );
    }
    Ok(())
}
