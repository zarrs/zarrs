//! A corrupt shard index with an overflowing subchunk span must return an error rather than panic.
#![allow(missing_docs)]

use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::codec::ShardingCodecBuilder;
use zarrs::array::codec::array_to_bytes::sharding::{ShardingCodecBound, ShardingPartialDecoder};
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArraySubset,
    data_type,
};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{StorageHandle, WritableStorageTraits};

/// Shard indexes with a subchunk span that overflows a `u64` or exceeds the shard.
fn corrupt_indexes() -> Vec<[u64; 8]> {
    let mut indexes = Vec::new();
    for (offset, size) in [
        (u64::MAX - 1, 2),
        (2, u64::MAX - 1),
        (u64::MAX - 1, 1),
        (0, u64::MAX - 1),
    ] {
        let mut index = [u64::MAX; 8];
        index[6] = offset;
        index[7] = size;
        indexes.push(index);
    }
    // The sum of the subchunk sizes overflows
    indexes.push([
        0,
        u64::MAX - 1,
        0,
        2,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
    ]);
    indexes
}

fn arrays() -> Vec<Array<MemoryStore>> {
    arrays_with(&corrupt_indexes())
}

fn arrays_with(indexes: &[[u64; 8]]) -> Vec<Array<MemoryStore>> {
    let mut arrays = Vec::new();
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        for &index in indexes {
            let codec = ShardingCodecBuilder::new(
                vec![NonZeroU64::new(2).unwrap(); 2],
                &data_type::uint8(),
            )
            .index_bytes_to_bytes_codecs(vec![])
            .index_location(location)
            .build_arc();
            let mut builder = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint8(), 0u8);
            builder.array_to_bytes_codec(codec);
            let store = Arc::new(MemoryStore::new());
            let array = builder.build(store.clone(), "/").unwrap();
            let encoded: Vec<u8> = index.into_iter().flat_map(u64::to_le_bytes).collect();
            store
                .set(&array.chunk_key(&[0, 0]).unwrap(), encoded.into())
                .unwrap();
            arrays.push(array);
        }
    }
    arrays
}

#[test]
fn decode_rejects_overflowing_shard_index() {
    for array in arrays() {
        assert!(
            array
                .retrieve_chunk::<Vec<u8>>(&[0, 0], &zarrs_codec::Resources::default())
                .is_err()
        );
    }
}

#[test]
fn decode_into_rejects_overflowing_shard_index() {
    for array in arrays() {
        let shape = vec![4, 4];
        let mut output = vec![0u8; 16];
        let output_slice = unsafe_cell_slice::UnsafeCellSlice::new(&mut output);
        let mut view = unsafe {
            // SAFETY: this is the only view over output and covers it exactly.
            ArrayBytesFixedDisjointView::new(
                output_slice,
                1,
                &shape,
                ArraySubset::new_with_shape(shape.clone()),
            )
            .unwrap()
        };
        assert!(
            array
                .retrieve_chunk_into(
                    &[0, 0],
                    ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                    &zarrs_codec::Resources::default()
                )
                .is_err()
        );
    }
}

#[test]
fn partial_decode_rejects_overflowing_shard_index() {
    // The span of subchunk [1, 1] overflows
    for array in arrays_with(&corrupt_indexes()[..2]) {
        assert!(
            array
                .retrieve_array_subset::<Vec<u8>>(
                    &ArraySubset::new_with_ranges(&[2..4, 2..4]),
                    &zarrs_codec::Resources::default()
                )
                .is_err()
        );
    }
}

#[test]
fn retrieve_subchunk_encoded_rejects_overflowing_shard_index() {
    // The span of subchunk [1, 1] overflows
    for array in arrays_with(&corrupt_indexes()[..2]) {
        let codecs_bound = array.codecs_bound();
        let sharding_codec = codecs_bound
            .array_to_bytes_codec()
            .as_any()
            .downcast_ref::<ShardingCodecBound>()
            .unwrap();
        let input_handle = Arc::new((
            Arc::new(StorageHandle::new(array.storage())),
            array.chunk_key(&[0, 0]).unwrap(),
        ));
        let decoder = ShardingPartialDecoder::new(
            input_handle,
            &array.chunk_shape(&[0, 0]).unwrap(),
            sharding_codec.subchunk_shape().clone(),
            sharding_codec.inner_codecs().clone(),
            sharding_codec.index_codecs(),
            sharding_codec.index_location(),
            array.codec_options(),
            &zarrs_codec::Resources::default(),
            sharding_codec.options().clone(),
        )
        .unwrap();
        assert!(
            decoder
                .retrieve_subchunk_encoded(
                    &[1, 1],
                    &zarrs_codec::CodecOptions::default(),
                    &zarrs_codec::Resources::default()
                )
                .is_err()
        );
        assert!(decoder.subchunk_byte_range(&[1, 1]).is_err());
    }
}

#[test]
fn compact_rejects_overflowing_shard_index() {
    for array in arrays() {
        assert!(
            array
                .compact_chunk(&[0, 0], &zarrs_codec::Resources::default())
                .is_err()
        );
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_rejects_overflowing_shard_index() {
    use zarrs::array::codec::array_to_bytes::sharding::AsyncShardingPartialDecoder;
    use zarrs::storage::AsyncWritableStorageTraits;
    use zarrs::storage::store::AsyncMemoryStore;

    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        // The span of subchunk [1, 1] overflows
        for index in &corrupt_indexes()[..2] {
            let codec = ShardingCodecBuilder::new(
                vec![NonZeroU64::new(2).unwrap(); 2],
                &data_type::uint8(),
            )
            .index_bytes_to_bytes_codecs(vec![])
            .index_location(location)
            .build_arc();
            let mut builder = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint8(), 0u8);
            builder.array_to_bytes_codec(codec);
            let store = Arc::new(AsyncMemoryStore::new());
            let array = builder.build(store.clone(), "/").unwrap();
            let encoded: Vec<u8> = index.iter().copied().flat_map(u64::to_le_bytes).collect();
            store
                .set(&array.chunk_key(&[0, 0]).unwrap(), encoded.into())
                .await
                .unwrap();

            assert!(
                array
                    .async_retrieve_array_subset::<Vec<u8>>(
                        &ArraySubset::new_with_ranges(&[2..4, 2..4]),
                        &zarrs_codec::Resources::default()
                    )
                    .await
                    .is_err()
            );
            assert!(
                array
                    .async_compact_chunk(&[0, 0], &zarrs_codec::Resources::default())
                    .await
                    .is_err()
            );

            let codecs_bound = array.codecs_bound();
            let sharding_codec = codecs_bound
                .array_to_bytes_codec()
                .as_any()
                .downcast_ref::<ShardingCodecBound>()
                .unwrap();
            let input_handle = Arc::new((
                Arc::new(StorageHandle::new(array.storage())),
                array.chunk_key(&[0, 0]).unwrap(),
            ));
            let decoder = AsyncShardingPartialDecoder::new(
                input_handle,
                &array.chunk_shape(&[0, 0]).unwrap(),
                sharding_codec.subchunk_shape().clone(),
                sharding_codec.inner_codecs().clone(),
                sharding_codec.index_codecs(),
                sharding_codec.index_location(),
                array.codec_options(),
                &zarrs_codec::Resources::default(),
                sharding_codec.options().clone(),
            )
            .await
            .unwrap();
            assert!(
                decoder
                    .retrieve_subchunk_encoded(
                        &[1, 1],
                        &zarrs_codec::CodecOptions::default(),
                        &zarrs_codec::Resources::default()
                    )
                    .await
                    .is_err()
            );
        }
    }
}
