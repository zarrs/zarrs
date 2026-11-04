#![allow(missing_docs)]
//! Storing and erasing encoded subchunks.
//!
//! Encoded subchunks retrieved from one array are stored in another array with the same codecs, through array-to-array
//! codecs, nested sharding, bytes-to-bytes codecs, chunk caches, and stores with and without partial writes.

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::chunk_cache::ChunkCacheDecodedLruChunkLimit;
use zarrs::array::codec::array_to_array::reshape::ReshapeShape;
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::codec::{
    Crc32cCodec, GzipCodec, ReshapeCodec, SqueezeCodec, TransposeCodec, TransposeOrder,
};
use zarrs::array::{ArrayBuilder, ArrayCached, CowBytes, Resources, data_type};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{
    MaybeBytes, MaybeBytesIterator, OffsetBytesIterator, ReadableStorageTraits, StorageError,
    StoreKey, StorePrefix, WritableStorageTraits,
};
use zarrs_storage::byte_range::ByteRangeIterator;

type TestResult = Result<(), Box<dyn Error>>;

fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}

/// A memory store that does not support partial writes.
#[derive(Debug, Default)]
struct NoPartialWriteStore(MemoryStore);

impl ReadableStorageTraits for NoPartialWriteStore {
    fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
        self.0.get(key)
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> Result<MaybeBytesIterator<'a>, StorageError> {
        self.0.get_partial_many(key, byte_ranges)
    }

    fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
        self.0.size_key(key)
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

impl WritableStorageTraits for NoPartialWriteStore {
    fn set(&self, key: &StoreKey, value: CowBytes<'_>) -> Result<(), StorageError> {
        self.0.set(key, value)
    }

    fn set_partial_many<'a>(
        &'a self,
        _key: &StoreKey,
        _offset_values: OffsetBytesIterator<'a>,
    ) -> Result<(), StorageError> {
        Err(StorageError::Unsupported(
            "the store does not support partial writes".to_string(),
        ))
    }

    fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
        self.0.erase(key)
    }

    fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
        self.0.erase_prefix(prefix)
    }

    fn supports_set_partial(&self) -> bool {
        false
    }
}

/// A single chunk array with a transpose codec in front of nested sharding, with non-square subchunks.
fn transposed_nested_sharding_builder() -> ArrayBuilder {
    let data_type = data_type::uint16();
    let inner_sharding = ShardingCodecBuilder::new(vec![nz(2), nz(4)], &data_type).build_arc();
    let mut outer_sharding = ShardingCodecBuilder::new(vec![nz(4), nz(8)], &data_type);
    outer_sharding.array_to_bytes_codec(inner_sharding);
    let mut builder = ArrayBuilder::new(vec![8, 8], vec![8, 8], data_type, 0u16);
    builder.array_to_array_codecs(vec![Arc::new(TransposeCodec::new(
        TransposeOrder::new(&[1, 0]).unwrap(),
    ))]);
    builder.array_to_bytes_codec(outer_sharding.build_arc());
    builder
}

/// Builders of arrays whose subchunks are mapped through array-to-array codecs, with the level, indices of a subchunk
/// and the number of elements of the array.
type MappedBuilder = (ArrayBuilder, usize, Vec<u64>, u16);

fn mapped_builders() -> Result<Vec<MappedBuilder>, Box<dyn Error>> {
    let data_type = data_type::uint16();
    let mut reshape = ArrayBuilder::new(vec![4, 6], vec![2, 6], data_type.clone(), 0u16);
    reshape.array_to_array_codecs(vec![Arc::new(ReshapeCodec::new(ReshapeShape::new([nz(
        12,
    )
    .into()])?))]);
    reshape.array_to_bytes_codec(ShardingCodecBuilder::new(vec![nz(3)], &data_type).build_arc());
    let mut squeeze = ArrayBuilder::new(vec![2, 4], vec![1, 4], data_type.clone(), 0u16);
    squeeze.array_to_array_codecs(vec![Arc::new(SqueezeCodec::new())]);
    squeeze.array_to_bytes_codec(ShardingCodecBuilder::new(vec![nz(2)], &data_type).build_arc());
    Ok(vec![
        (reshape, 0, vec![3, 1], 24),
        (squeeze, 0, vec![1, 1], 8),
        (transposed_nested_sharding_builder(), 0, vec![0, 1], 64),
        (transposed_nested_sharding_builder(), 1, vec![1, 3], 64),
    ])
}

#[test]
fn array_store_encoded_subchunks_through_mappings() -> TestResult {
    let resources = Resources::default();
    for (builder, level, indices, len) in mapped_builders()? {
        let store = Arc::new(MemoryStore::default());
        let source = builder.build(store.clone(), "/source")?;
        let data: Vec<u16> = (1..=len).collect();
        source.store_array_subset(&source.subset_all(), data, &resources)?;
        let target = builder.build_arc(store, "/target")?;
        let cached = ArrayCached::new(target.clone(), ChunkCacheDecodedLruChunkLimit::new(2));
        let encoded = source
            .retrieve_encoded_subchunk_at_level(level, &indices, &resources)?
            .expect("the subchunk is stored");

        // The subchunk is stored verbatim, and a cached read of the array sees it
        cached.retrieve_array_subset::<Vec<u16>>(&cached.subset_all(), &resources)?;
        // SAFETY: Both arrays have the same codecs and shape.
        unsafe {
            cached.store_encoded_subchunk_at_level(
                level,
                &indices,
                encoded.bytes().clone().into(),
                &resources,
            )?;
        }
        assert_eq!(
            target
                .retrieve_encoded_subchunk_at_level(level, &indices, &resources)?
                .expect("the subchunk is stored"),
            encoded
        );
        assert_eq!(
            cached.retrieve_subchunk_at_level::<Vec<u16>>(level, &indices, &resources)?,
            source.retrieve_subchunk_at_level::<Vec<u16>>(level, &indices, &resources)?
        );
        let subchunk_subset = target
            .subchunk_grid_at_level(level)
            .as_chunk_grid()
            .unwrap()
            .subset(&indices)?
            .unwrap();
        let stored = cached.retrieve_array_subset::<Vec<u16>>(&cached.subset_all(), &resources)?;
        let expected =
            source.retrieve_array_subset::<Vec<u16>>(&source.subset_all(), &resources)?;
        for (index, (stored, expected)) in std::iter::zip(stored, expected).enumerate() {
            let indices = zarrs::array::unravel_index(index as u64, target.shape()).unwrap();
            if subchunk_subset.contains(&indices) {
                assert_eq!(stored, expected);
            } else {
                assert_eq!(stored, 0);
            }
        }

        // An unsupported level fails without changing the array
        assert!(
            unsafe {
                cached.store_encoded_subchunk_at_level(9, &indices, vec![0; 4].into(), &resources)
            }
            .is_err()
        );
        assert!(
            cached
                .erase_subchunk_at_level(9, &indices, &resources)
                .is_err()
        );

        // Erasing the only stored subchunk erases the chunk
        cached.erase_subchunk_at_level(level, &indices, &resources)?;
        assert!(
            target
                .retrieve_encoded_subchunk_at_level(level, &indices, &resources)?
                .is_none()
        );
        assert!(
            cached
                .retrieve_array_subset::<Vec<u16>>(&cached.subset_all(), &resources)?
                .iter()
                .all(|&value| value == 0)
        );
        assert!(
            (0..target.chunk_grid_shape().iter().product::<u64>()).all(|chunk| {
                let chunk_indices =
                    zarrs::array::unravel_index(chunk, target.chunk_grid_shape()).unwrap();
                target
                    .retrieve_encoded_chunk(&chunk_indices)
                    .unwrap()
                    .is_none()
            })
        );
        // Erasing a subchunk that is not stored succeeds
        cached.erase_subchunk_at_level(level, &indices, &resources)?;
    }
    Ok(())
}

#[test]
fn array_store_encoded_subchunks_three_levels() -> TestResult {
    let resources = Resources::default();
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        for partial_writes in [true, false] {
            let data_type = data_type::uint8();
            let mut inner = ShardingCodecBuilder::new(vec![nz(4)], &data_type);
            inner.index_location(location);
            let mut middle = ShardingCodecBuilder::new(vec![nz(8)], &data_type);
            middle
                .index_location(location)
                .array_to_bytes_codec(inner.build_arc());
            middle.bytes_to_bytes_codecs(vec![Arc::new(GzipCodec::new(5)?)]);
            let mut outer = ShardingCodecBuilder::new(vec![nz(16)], &data_type);
            outer
                .index_location(location)
                .array_to_bytes_codec(middle.build_arc());
            let mut builder = ArrayBuilder::new(vec![32], vec![32], data_type, 0u8);
            builder.array_to_bytes_codec(outer.build_arc());
            builder.bytes_to_bytes_codecs(vec![Arc::new(Crc32cCodec::new())]);

            let source = builder.build(Arc::new(MemoryStore::default()), "/source")?;
            source.store_chunk(&[0], (1..=32u8).collect::<Vec<_>>(), &resources)?;
            let check = |target: &dyn Fn(usize, &[u64]) -> TestResult| -> TestResult {
                for (level, indices) in [(2, [3]), (2, [6]), (1, [0]), (0, [1]), (2, [0])] {
                    target(level, &indices)?;
                }
                Ok(())
            };
            macro_rules! store_and_check {
                ($target:expr) => {{
                    let target = $target;
                    check(&|level, indices| {
                        let encoded = source
                            .retrieve_encoded_subchunk_at_level(level, indices, &resources)?
                            .expect("the subchunk is stored");
                        // SAFETY: Both arrays have the same codecs and shape.
                        unsafe {
                            target.store_encoded_subchunk_at_level(
                                level,
                                indices,
                                encoded.bytes().clone().into(),
                                &resources,
                            )?;
                        }
                        assert_eq!(
                            target
                                .retrieve_encoded_subchunk_at_level(level, indices, &resources)?
                                .expect("the subchunk is stored"),
                            encoded
                        );
                        assert_eq!(
                            target.retrieve_subchunk_at_level::<Vec<u8>>(
                                level, indices, &resources
                            )?,
                            source.retrieve_subchunk_at_level::<Vec<u8>>(
                                level, indices, &resources
                            )?
                        );
                        Ok(())
                    })?;
                    // The subchunks stored above are in the chunk, and the rest is the fill value
                    let mut expected: Vec<u8> = (1..=32).collect();
                    expected[8..12].fill(0);
                    assert_eq!(
                        target.retrieve_chunk::<Vec<u8>>(&[0], &resources)?,
                        expected
                    );
                    target.erase_subchunk_at_level(2, &[3], &resources)?;
                    expected[12..16].fill(0);
                    assert_eq!(
                        target.retrieve_chunk::<Vec<u8>>(&[0], &resources)?,
                        expected
                    );
                    // Erasing every stored subchunk erases the chunk
                    for (level, indices) in [(0, [0]), (1, [3]), (1, [2])] {
                        target.erase_subchunk_at_level(level, &indices, &resources)?;
                    }
                    assert!(target.retrieve_encoded_chunk(&[0])?.is_none());
                }};
            }
            if partial_writes {
                store_and_check!(builder.build(Arc::new(MemoryStore::default()), "/target")?);
            } else {
                store_and_check!(
                    builder.build(Arc::new(NoPartialWriteStore::default()), "/target")?
                );
            }
        }
    }
    Ok(())
}

#[test]
fn array_erase_subchunks() -> TestResult {
    let resources = Resources::default();
    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        let data_type = data_type::uint8();
        let mut sharding = ShardingCodecBuilder::new(vec![nz(2)], &data_type);
        sharding.index_location(location);
        let mut builder = ArrayBuilder::new(vec![8], vec![8], data_type, 0u8);
        builder.array_to_bytes_codec(sharding.build_arc());
        let array = builder.build(Arc::new(MemoryStore::default()), "/")?;

        // Erasing a subchunk of a missing chunk succeeds
        array.erase_subchunk(&[1], &resources)?;
        assert!(array.retrieve_encoded_chunk(&[0])?.is_none());

        for indices in [[0], [1], [3]] {
            // SAFETY: The subchunk codec is native uint8 bytes with shape [2].
            unsafe { array.store_encoded_subchunk(&indices, vec![7u8; 2].into(), &resources)? };
        }
        array.erase_subchunk(&[1], &resources)?;
        array.erase_subchunk(&[1], &resources)?;
        assert!(array.retrieve_encoded_subchunk(&[1], &resources)?.is_none());
        assert_eq!(
            array.retrieve_chunk::<Vec<u8>>(&[0], &resources)?,
            vec![7, 7, 0, 0, 0, 0, 7, 7]
        );
        for indices in [vec![], vec![4], vec![0, 0]] {
            assert!(array.erase_subchunk(&indices, &resources).is_err());
        }
        array.erase_subchunk(&[0], &resources)?;
        array.erase_subchunk(&[3], &resources)?;
        assert!(array.retrieve_encoded_chunk(&[0])?.is_none());
    }
    Ok(())
}

#[test]
fn array_without_subchunks_cannot_store_encoded_subchunks() -> TestResult {
    let resources = Resources::default();
    let array = ArrayBuilder::new(vec![8], vec![4], data_type::uint8(), 0u8)
        .build(Arc::new(MemoryStore::default()), "/")?;
    assert!(
        unsafe { array.store_encoded_subchunk(&[0], vec![0u8; 4].into(), &resources) }.is_err()
    );
    assert!(array.erase_subchunk(&[0], &resources).is_err());
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_array_store_encoded_subchunks() -> TestResult {
    use zarrs::array::chunk_cache::AsyncChunkCacheDecodedLruChunkLimit;
    use zarrs::storage::store::AsyncMemoryStore;

    let resources = Resources::default();
    let store = Arc::new(AsyncMemoryStore::default());
    let builder = transposed_nested_sharding_builder();
    let source = builder.build(store.clone(), "/source")?;
    source
        .async_store_chunk(&[0, 0], &(1..=64u16).collect::<Vec<_>>(), &resources)
        .await?;
    let target = builder.build_arc(store, "/target")?;
    let cached = ArrayCached::new(target.clone(), AsyncChunkCacheDecodedLruChunkLimit::new(2));
    for (level, indices) in [(1, [1, 3]), (0, [0, 0])] {
        cached
            .async_retrieve_chunk::<Vec<u16>>(&[0, 0], &resources)
            .await?;
        let encoded = source
            .async_retrieve_encoded_subchunk_at_level(level, &indices, &resources)
            .await?
            .expect("the subchunk is stored");
        // SAFETY: Both arrays have the same codecs and shape.
        unsafe {
            cached
                .async_store_encoded_subchunk_at_level(
                    level,
                    &indices,
                    encoded.bytes().clone().into(),
                    &resources,
                )
                .await?;
        }
        assert_eq!(
            target
                .async_retrieve_encoded_subchunk_at_level(level, &indices, &resources)
                .await?
                .expect("the subchunk is stored"),
            encoded
        );
        assert_eq!(
            cached
                .async_retrieve_subchunk_at_level::<Vec<u16>>(level, &indices, &resources)
                .await?,
            source
                .async_retrieve_subchunk_at_level::<Vec<u16>>(level, &indices, &resources)
                .await?
        );
        cached
            .async_erase_subchunk_at_level(level, &indices, &resources)
            .await?;
        assert!(
            target
                .async_retrieve_encoded_subchunk_at_level(level, &indices, &resources)
                .await?
                .is_none()
        );
        assert!(
            cached
                .async_retrieve_chunk::<Vec<u16>>(&[0, 0], &resources)
                .await?
                .iter()
                .all(|&value| value == 0)
        );
    }
    Ok(())
}
