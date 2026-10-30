//! A corrupt shard index must return an error rather than panic or return the wrong data.
#![allow(missing_docs)]

use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::codec::array_to_bytes::sharding::{ShardingCodecBound, ShardingPartialDecoder};
use zarrs::array::codec::{GzipCodec, ShardingCodec, ShardingCodecBuilder};
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArraySubset,
    data_type,
};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{StorageHandle, WritableStorageTraits};

const LOCATIONS: [ShardingIndexLocation; 2] =
    [ShardingIndexLocation::Start, ShardingIndexLocation::End];

/// The encoded size of the shard index of a 4x4 shard with 2x2 subchunks.
const INDEX_SIZE: u64 = 64;

/// The encoded data of the shard.
const DATA: [u8; 4] = [10, 20, 30, 40];

/// The encoded size of the shard.
const SHARD_SIZE: u64 = INDEX_SIZE + DATA.len() as u64;

/// The inner codecs of the sharding codec.
#[derive(Clone, Copy, Debug)]
enum Inner {
    /// The `bytes` codec, with a fixed encoded size.
    Bytes,
    /// The `bytes` and `gzip` codecs, with an unbounded encoded size.
    Gzip,
    /// A nested sharding codec with its index at the end, which partially decodes a suffix.
    Sharding,
}

const INNERS: [Inner; 3] = [Inner::Bytes, Inner::Gzip, Inner::Sharding];

fn sharding_codec(location: ShardingIndexLocation, inner: Inner) -> Arc<ShardingCodec> {
    let mut builder =
        ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type::uint8());
    builder
        .index_bytes_to_bytes_codecs(vec![])
        .index_location(location);
    match inner {
        Inner::Bytes => {}
        Inner::Gzip => {
            builder.bytes_to_bytes_codecs(vec![Arc::new(GzipCodec::new(5).unwrap())]);
        }
        Inner::Sharding => {
            let inner_codec = ShardingCodecBuilder::new(
                vec![NonZeroU64::new(1).unwrap(); 2],
                &data_type::uint8(),
            )
            .index_bytes_to_bytes_codecs(vec![])
            .index_location(ShardingIndexLocation::End)
            .build_arc();
            builder.array_to_bytes_codec(inner_codec);
        }
    }
    builder.build_arc()
}

/// The offset of the shard data.
fn data_start(location: ShardingIndexLocation) -> u64 {
    match location {
        ShardingIndexLocation::Start => INDEX_SIZE,
        ShardingIndexLocation::End => 0,
    }
}

/// Encode a shard of [`DATA`] with subchunk [1, 1] at `offset` with `size` and the other subchunks empty.
fn encoded_shard(location: ShardingIndexLocation, offset: u64, size: u64) -> Vec<u8> {
    let mut index = [u64::MAX; 8];
    index[6] = offset;
    index[7] = size;
    let index = index.into_iter().flat_map(u64::to_le_bytes);
    match location {
        ShardingIndexLocation::Start => index.chain(DATA).collect(),
        ShardingIndexLocation::End => DATA.into_iter().chain(index).collect(),
    }
}

/// Corrupt (offset, size) entries for subchunk [1, 1].
fn corrupt_entries(location: ShardingIndexLocation) -> Vec<(u64, u64)> {
    let data_start = data_start(location);
    let mut entries = vec![
        // The subchunk end overflows
        (u64::MAX - 1, 2),
        (2, u64::MAX - 1),
        // The subchunk is beyond the shard
        (u64::MAX - 1, 1),
        (data_start, u64::MAX - 1),
        // The subchunk is smaller than its encoded size
        (data_start, 1),
    ];
    if location == ShardingIndexLocation::Start {
        // The subchunk overlaps the shard index
        entries.push((0, 4));
        entries.push((INDEX_SIZE - 1, 4));
    }
    entries
}

fn array(
    location: ShardingIndexLocation,
    inner: Inner,
    encoded_shard: Vec<u8>,
) -> Array<MemoryStore> {
    let mut builder = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint8(), 0u8);
    builder.array_to_bytes_codec(sharding_codec(location, inner));
    let store = Arc::new(MemoryStore::new());
    let array = builder.build(store.clone(), "/").unwrap();
    store
        .set(&array.chunk_key(&[0, 0]).unwrap(), encoded_shard.into())
        .unwrap();
    array
}

struct CorruptArray {
    name: String,
    /// Whether subchunk [1, 1] is within the shard data.
    in_data: bool,
    array: Array<MemoryStore>,
}

/// Arrays with a corrupt shard index.
fn corrupt_arrays() -> Vec<CorruptArray> {
    let mut arrays = Vec::new();
    for location in LOCATIONS {
        for inner in INNERS {
            for (offset, size) in corrupt_entries(location) {
                let data_start = data_start(location);
                let data_end = data_start + DATA.len() as u64;
                arrays.push(CorruptArray {
                    name: format!("{location:?} {inner:?} ({offset}, {size})"),
                    in_data: offset >= data_start
                        && offset.checked_add(size).is_some_and(|end| end <= data_end),
                    array: array(location, inner, encoded_shard(location, offset, size)),
                });
            }
        }
    }
    arrays
}

/// Array subsets intersecting subchunk [1, 1].
fn subsets() -> Vec<ArraySubset> {
    vec![
        ArraySubset::new_with_ranges(&[0..4, 0..4]),
        ArraySubset::new_with_ranges(&[2..4, 2..4]),
        ArraySubset::new_with_ranges(&[2..3, 2..3]),
        ArraySubset::new_with_ranges(&[3..4, 3..4]),
        ArraySubset::new_with_ranges(&[1..4, 1..4]),
    ]
}

/// Element indexers intersecting subchunk [1, 1].
fn indexers() -> Vec<Vec<Vec<u64>>> {
    vec![
        vec![vec![3, 3]],
        vec![vec![2, 2], vec![3, 3]],
        vec![vec![3, 3], vec![0, 0], vec![2, 3]],
    ]
}

/// Returns whether retrieving `subset` into an output view fails.
fn retrieve_into_fails(array: &Array<MemoryStore>, subset: &ArraySubset) -> bool {
    let shape = subset.shape();
    let mut output = vec![0u8; subset.num_elements_usize()];
    let output_slice = unsafe_cell_slice::UnsafeCellSlice::new(&mut output);
    let mut view = unsafe {
        // SAFETY: this is the only view over output and covers it exactly.
        ArrayBytesFixedDisjointView::new(output_slice, 1, shape, ArraySubset::new_with_shape(shape))
            .unwrap()
    };
    array
        .retrieve_array_subset_into(
            subset,
            ArrayBytesDecodeIntoTarget::Fixed(&mut view),
            &zarrs_codec::Resources::default(),
        )
        .is_err()
}

fn sharding_partial_decoder(array: &Array<MemoryStore>) -> Option<ShardingPartialDecoder> {
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
    ShardingPartialDecoder::new(
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
    .ok()
}

#[test]
fn decode_rejects_corrupt_shard_index() {
    for CorruptArray { name, array, .. } in corrupt_arrays() {
        assert!(
            array
                .retrieve_chunk::<Vec<u8>>(&[0, 0], &zarrs_codec::Resources::default())
                .is_err(),
            "{name}"
        );
        assert!(
            retrieve_into_fails(&array, &ArraySubset::new_with_ranges(&[0..4, 0..4])),
            "{name}"
        );
    }
}

#[test]
fn partial_decode_rejects_corrupt_shard_index() {
    for CorruptArray { name, array, .. } in corrupt_arrays() {
        for subset in subsets() {
            assert!(
                array
                    .retrieve_array_subset::<Vec<u8>>(&subset, &zarrs_codec::Resources::default())
                    .is_err(),
                "{name} {subset:?}"
            );
            assert!(retrieve_into_fails(&array, &subset), "{name} {subset:?}");
        }
        for indexer in indexers() {
            assert!(
                array
                    .retrieve_partial_chunk::<Vec<u8>>(
                        &[0, 0],
                        &indexer,
                        &zarrs_codec::Resources::default()
                    )
                    .is_err(),
                "{name} {indexer:?}"
            );
        }
    }
}

#[test]
fn compact_rejects_corrupt_shard_index() {
    for CorruptArray {
        name,
        in_data,
        array,
    } in corrupt_arrays()
    {
        // A subchunk smaller than its encoded size is within the shard data, so it can be compacted
        assert_eq!(
            array
                .compact_chunk(&[0, 0], &zarrs_codec::Resources::default())
                .is_err(),
            !in_data,
            "{name}"
        );
    }
}

#[test]
fn retrieve_subchunk_encoded_rejects_corrupt_shard_index() {
    for CorruptArray { name, array, .. } in corrupt_arrays() {
        let Some(decoder) = sharding_partial_decoder(&array) else {
            // The shard index is rejected when it is decoded
            continue;
        };
        // The encoded bytes are not decoded, so only bytes beyond the shard are rejected
        let byte_range = decoder.subchunk_byte_range(&[1, 1]).unwrap().unwrap();
        let in_shard = byte_range.checked_range(SHARD_SIZE).is_some();
        assert_eq!(
            decoder
                .retrieve_subchunk_encoded(
                    &[1, 1],
                    &zarrs_codec::CodecOptions::default(),
                    &zarrs_codec::Resources::default()
                )
                .is_err(),
            !in_shard,
            "{name}"
        );
    }
}

#[test]
fn decode_rejects_subchunk_in_shard_index_at_end() {
    for inner in INNERS {
        // Subchunk [1, 1] references the shard index after the data
        let array = array(
            ShardingIndexLocation::End,
            inner,
            encoded_shard(ShardingIndexLocation::End, 4, 4),
        );
        assert!(
            array
                .retrieve_chunk::<Vec<u8>>(&[0, 0], &zarrs_codec::Resources::default())
                .is_err(),
            "{inner:?}"
        );
        assert!(
            retrieve_into_fails(&array, &ArraySubset::new_with_ranges(&[0..4, 0..4])),
            "{inner:?}"
        );
        assert!(
            array
                .compact_chunk(&[0, 0], &zarrs_codec::Resources::default())
                .is_err(),
            "{inner:?}"
        );
    }
}

#[test]
fn compact_subchunks_sharing_bytes() {
    for location in LOCATIONS {
        let data_start = data_start(location);
        let mut index = [u64::MAX; 8];
        // Subchunks [0, 0] and [1, 1] share the same bytes
        index[0] = data_start;
        index[1] = 4;
        index[6] = data_start;
        index[7] = 4;
        let index = index.into_iter().flat_map(u64::to_le_bytes);
        let shard = match location {
            ShardingIndexLocation::Start => index.chain(DATA).collect(),
            ShardingIndexLocation::End => DATA.into_iter().chain(index).collect(),
        };
        let array = array(location, Inner::Bytes, shard);
        let expected = vec![10, 20, 0, 0, 30, 40, 0, 0, 0, 0, 10, 20, 0, 0, 30, 40];
        assert_eq!(
            array
                .retrieve_chunk::<Vec<u8>>(&[0, 0], &zarrs_codec::Resources::default())
                .unwrap(),
            expected
        );
        array
            .compact_chunk(&[0, 0], &zarrs_codec::Resources::default())
            .unwrap();
        assert_eq!(
            array
                .retrieve_chunk::<Vec<u8>>(&[0, 0], &zarrs_codec::Resources::default())
                .unwrap(),
            expected
        );
        assert_eq!(
            array
                .retrieve_encoded_chunk(&[0, 0])
                .unwrap()
                .unwrap()
                .len(),
            usize::try_from(INDEX_SIZE).unwrap() + DATA.len()
        );
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_rejects_corrupt_shard_index() {
    use zarrs::array::codec::array_to_bytes::sharding::AsyncShardingPartialDecoder;
    use zarrs::storage::AsyncWritableStorageTraits;
    use zarrs::storage::store::AsyncMemoryStore;

    for location in LOCATIONS {
        for inner in INNERS {
            for (offset, size) in corrupt_entries(location) {
                let name = format!("{location:?} {inner:?} ({offset}, {size})");
                let mut builder =
                    ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint8(), 0u8);
                builder.array_to_bytes_codec(sharding_codec(location, inner));
                let store = Arc::new(AsyncMemoryStore::new());
                let array = builder.build(store.clone(), "/").unwrap();
                store
                    .set(
                        &array.chunk_key(&[0, 0]).unwrap(),
                        encoded_shard(location, offset, size).into(),
                    )
                    .await
                    .unwrap();

                for subset in subsets() {
                    assert!(
                        array
                            .async_retrieve_array_subset::<Vec<u8>>(
                                &subset,
                                &zarrs_codec::Resources::default()
                            )
                            .await
                            .is_err(),
                        "{name} {subset:?}"
                    );
                }
                for indexer in indexers() {
                    assert!(
                        array
                            .async_retrieve_partial_chunk::<Vec<u8>>(
                                &[0, 0],
                                &indexer,
                                &zarrs_codec::Resources::default()
                            )
                            .await
                            .is_err(),
                        "{name} {indexer:?}"
                    );
                }

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
                .await;
                if let Ok(decoder) = decoder {
                    let byte_range = decoder.subchunk_byte_range(&[1, 1]).unwrap().unwrap();
                    let in_shard = byte_range.checked_range(SHARD_SIZE).is_some();
                    assert_eq!(
                        decoder
                            .retrieve_subchunk_encoded(
                                &[1, 1],
                                &zarrs_codec::CodecOptions::default(),
                                &zarrs_codec::Resources::default()
                            )
                            .await
                            .is_err(),
                        !in_shard,
                        "{name}"
                    );
                }
            }
        }
    }
}
