#![allow(missing_docs)]
#![allow(clippy::single_range_in_vec_init)]

//! Shards written as they are encoded, with multipart writes, match shards encoded whole.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::Crc32cCodec;
use zarrs::array::codec::array_to_bytes::sharding::{
    ShardingCodecBuilder, ShardingCodecOptions, ShardingIndexLocation, SubchunkWriteOrder,
};
use zarrs::array::{
    Array, ArrayBuilder, ArraySubset, BytesToBytesCodecTraits, CodecOptions, CodecSpecificOptions,
    MultipartWrite, Resources, data_type,
};
use zarrs_filesystem::FilesystemStore;
use zarrs_storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs_storage::store::MemoryStore;
use zarrs_storage::{ListableStorageTraits, ReadableStorageTraits, StorePrefix};

fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

/// The configuration of a sharded array.
#[derive(Debug, Clone, Copy)]
struct Config {
    index_location: ShardingIndexLocation,
    order: SubchunkWriteOrder,
    /// Bytes-to-bytes codecs of the subchunks.
    inner_crc32c: bool,
    /// Bytes-to-bytes codecs of the shards (which need the whole shard, so it is not streamed).
    outer_crc32c: bool,
    store_empty_chunks: bool,
    /// The part size of multipart writes.
    part_size: usize,
}

/// An array of 3 x 2 shards of 16 x 16 elements with subchunks of 2 x 2 elements in `store`.
fn array<TStorage: ?Sized>(
    store: Arc<TStorage>,
    config: Config,
    multipart_write: MultipartWrite,
) -> Array<TStorage> {
    let crc32c = || -> Vec<Arc<dyn BytesToBytesCodecTraits>> { vec![Arc::new(Crc32cCodec::new())] };
    let mut builder = ArrayBuilder::new(vec![48, 32], vec![16, 16], data_type::uint16(), 7u16);
    builder
        .array_to_bytes_codec(Arc::new(
            ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type::uint16())
                .index_location(config.index_location)
                .bytes_to_bytes_codecs(if config.inner_crc32c {
                    crc32c()
                } else {
                    vec![]
                })
                .build(),
        ))
        .bytes_to_bytes_codecs(if config.outer_crc32c {
            crc32c()
        } else {
            vec![]
        })
        .codec_specific_options(
            CodecSpecificOptions::default().with_option(
                ShardingCodecOptions::default().with_subchunk_write_order(config.order),
            ),
        );
    let array = builder.build_arc(store, "/array").unwrap();
    let options = CodecOptions::default()
        .with_store_empty_chunks(config.store_empty_chunks)
        .with_multipart_write(multipart_write)
        .with_multipart_part_size(config.part_size);
    array.with_codec_options(options)
}

/// The elements of the array: the first shard is the fill value, and the subchunks of the first row of the second shard
/// are the fill value.
fn data() -> Vec<u16> {
    (0..48 * 32)
        .map(|i| {
            let (row, column) = (i / 32, i % 32);
            if row < 16 && column < 16 || (row < 2 && column >= 16) {
                7
            } else {
                u16::try_from(i).unwrap()
            }
        })
        .collect()
}

/// The stored values of `store`, by key.
fn values(store: &MemoryStore) -> Vec<(String, Vec<u8>)> {
    store
        .list_prefix(&StorePrefix::root())
        .unwrap()
        .into_iter()
        .filter(|key| key.as_str().contains("/c/"))
        .map(|key| (key.to_string(), store.get(&key).unwrap().unwrap().to_vec()))
        .collect()
}

/// Write the data to arrays with `config`, with and without multipart writes, with `write` (and `resources`), and
/// check the shards decode identically, and are byte-identical in C order.
fn check_streamed_matches_buffered(
    config: Config,
    resources: &Resources,
    write: impl Fn(&Array<PerformanceMetricsStorageAdapter<MemoryStore>>, &[u16], &Resources),
) {
    let data = data();
    let buffered_store = Arc::new(MemoryStore::new());
    let streamed_store = Arc::new(MemoryStore::new());
    let buffered_metrics = Arc::new(PerformanceMetricsStorageAdapter::new(
        buffered_store.clone(),
    ));
    let streamed_metrics = Arc::new(PerformanceMetricsStorageAdapter::new(
        streamed_store.clone(),
    ));
    let buffered = array(buffered_metrics.clone(), config, MultipartWrite::Never);
    let streamed = array(streamed_metrics.clone(), config, MultipartWrite::Always);
    buffered.store_metadata().unwrap();
    streamed.store_metadata().unwrap();
    buffered_metrics.reset();
    streamed_metrics.reset();

    write(&buffered, &data, resources);
    write(&streamed, &data, resources);
    let all = buffered.subset_all();
    assert_eq!(
        buffered
            .retrieve_array_subset::<Vec<u16>>(&all, resources)
            .unwrap(),
        data,
        "{config:?}"
    );
    assert_eq!(
        streamed
            .retrieve_array_subset::<Vec<u16>>(&all, resources)
            .unwrap(),
        data,
        "{config:?}"
    );

    let (buffered_values, streamed_values) = (values(&buffered_store), values(&streamed_store));
    let keys = |values: &[(String, Vec<u8>)]| -> Vec<String> {
        values.iter().map(|(key, _)| key.clone()).collect()
    };
    assert_eq!(keys(&buffered_values), keys(&streamed_values), "{config:?}");
    let expected_shards = if config.store_empty_chunks { 6 } else { 5 };
    assert_eq!(buffered_values.len(), expected_shards, "{config:?}");
    if matches!(config.order, SubchunkWriteOrder::C) {
        assert_eq!(buffered_values, streamed_values, "{config:?}");
    }
    for ((_, buffered), (_, streamed)) in buffered_values.iter().zip(&streamed_values) {
        assert_eq!(buffered.len(), streamed.len(), "{config:?}");
    }

    // Each shard is one write when encoded whole, and several parts when streamed
    let shards_written = buffered_values.len();
    assert_eq!(buffered_metrics.writes(), shards_written, "{config:?}");
    let streams = config.index_location == ShardingIndexLocation::End && !config.outer_crc32c;
    if streams {
        assert!(streamed_metrics.writes() > shards_written, "{config:?}");
    } else {
        assert_eq!(streamed_metrics.writes(), shards_written, "{config:?}");
    }
    assert_eq!(
        buffered_metrics.bytes_written(),
        streamed_metrics.bytes_written(),
        "{config:?}"
    );
}

fn configs() -> Vec<Config> {
    let mut configs = vec![];
    for index_location in [ShardingIndexLocation::End, ShardingIndexLocation::Start] {
        for order in [SubchunkWriteOrder::C, SubchunkWriteOrder::Unordered] {
            for inner_crc32c in [false, true] {
                for outer_crc32c in [false, true] {
                    for store_empty_chunks in [false, true] {
                        // A part per block, and parts of several blocks
                        for part_size in [1, 100] {
                            configs.push(Config {
                                index_location,
                                order,
                                inner_crc32c,
                                outer_crc32c,
                                store_empty_chunks,
                                part_size,
                            });
                        }
                    }
                }
            }
        }
    }
    configs
}

#[test]
fn sharding_multipart_store_chunk() {
    for config in configs() {
        // One task per subchunk (64 blocks) and blocks of 8 subchunks (8 blocks)
        for compute_target in [8, 1] {
            let resources = Resources::default().with_compute_target(compute_target);
            check_streamed_matches_buffered(config, &resources, |array, data, resources| {
                for (i, j) in (0..3).flat_map(|i| (0..2).map(move |j| (i, j))) {
                    let chunk_data: Vec<u16> = (0..16)
                        .flat_map(|row| {
                            (0..16).map(move |column| (i * 16 + row) * 32 + j * 16 + column)
                        })
                        .map(|index| data[usize::try_from(index).unwrap()])
                        .collect();
                    array.store_chunk(&[i, j], chunk_data, resources).unwrap();
                }
            });
        }
    }
}

#[test]
fn sharding_multipart_store_array_subset() {
    for config in configs() {
        let resources = Resources::default().with_compute_target(8);
        check_streamed_matches_buffered(config, &resources, |array, data, resources| {
            array
                .store_array_subset(&array.subset_all(), data, resources)
                .unwrap();
        });
    }
}

/// A filesystem store streams shards by default.
#[test]
fn sharding_multipart_filesystem() {
    let path = tempfile::TempDir::new().unwrap();
    let store = Arc::new(FilesystemStore::new(path.path()).unwrap());
    let config = Config {
        index_location: ShardingIndexLocation::End,
        order: SubchunkWriteOrder::C,
        inner_crc32c: true,
        outer_crc32c: false,
        store_empty_chunks: false,
        part_size: 100,
    };
    let resources = Resources::default().with_compute_target(8);
    let filesystem = array(store.clone(), config, MultipartWrite::Auto);
    let memory_store = Arc::new(MemoryStore::new());
    let memory = array(memory_store.clone(), config, MultipartWrite::Never);
    let data = data();
    filesystem
        .store_array_subset(&filesystem.subset_all(), &data, &resources)
        .unwrap();
    memory
        .store_array_subset(&memory.subset_all(), &data, &resources)
        .unwrap();
    for (key, value) in values(&memory_store) {
        let key = key.as_str().try_into().unwrap();
        assert_eq!(store.get(&key).unwrap().unwrap().to_vec(), value);
    }
    // No staging files are left
    let mut files = 0;
    for entry in walk(path.path()) {
        assert!(!entry.ends_with(".zarrs-part"), "{entry}");
        files += 1;
    }
    // The 5 shards that are not the fill value
    assert_eq!(files, 5);
}

fn walk(dir: &std::path::Path) -> Vec<String> {
    let mut files = vec![];
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            files.extend(walk(&path));
        } else {
            files.push(path.to_string_lossy().into_owned());
        }
    }
    files
}

#[cfg(feature = "async")]
#[test]
fn sharding_multipart_async() {
    use zarrs_storage::AsyncReadableStorageTraits;
    use zarrs_storage::store::AsyncMemoryStore;

    futures::executor::block_on(async {
        let config = Config {
            index_location: ShardingIndexLocation::End,
            order: SubchunkWriteOrder::C,
            inner_crc32c: true,
            outer_crc32c: false,
            store_empty_chunks: false,
            part_size: 100,
        };
        let resources = Resources::default().with_compute_target(8);
        let data = data();
        let buffered_store = Arc::new(AsyncMemoryStore::new());
        let streamed_store = Arc::new(AsyncMemoryStore::new());
        let buffered = array(buffered_store.clone(), config, MultipartWrite::Never);
        let streamed = array(streamed_store.clone(), config, MultipartWrite::Always);
        for array in [&buffered, &streamed] {
            // Owned bytes, so compute is offloaded (rather than running on the polling task, which encodes shards whole)
            array
                .async_store_array_subset(&array.subset_all(), data.clone(), &resources)
                .await
                .unwrap();
        }
        for chunk in ArraySubset::new_with_shape(vec![3, 2]).indices() {
            let key = buffered.chunk_key(&chunk).unwrap();
            assert_eq!(
                buffered_store.get(&key).await.unwrap(),
                streamed_store.get(&key).await.unwrap()
            );
        }
        assert_eq!(
            streamed
                .async_retrieve_array_subset::<Vec<u16>>(&streamed.subset_all(), &resources)
                .await
                .unwrap(),
            data
        );
    });
}
