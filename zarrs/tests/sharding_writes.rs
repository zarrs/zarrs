//! Partial writes preserve unselected values and only retrieve affected subchunks.
#![cfg(feature = "sharding")]
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use zarrs::array::codec::{ShardingCodec, ShardingCodecBuilder};
use zarrs::array::{
    ArrayBytes, ArrayBytesRaw, ArraySubset, DataType, FillValue, IntoArrayBytes, data_type,
};
use zarrs::metadata_ext::codec::sharding::ShardingIndexLocation;
use zarrs_codec::{
    ArrayToBytesCodecTraits, BytesPartialDecoderTraits, BytesPartialEncoderTraits, CodecError,
    CodecOptions, update_array_bytes,
};
use zarrs_storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs_storage::{OffsetBytesIterator, StorageError};

#[derive(Default)]
struct Handle {
    bytes: Mutex<Option<Vec<u8>>>,
    reads: Mutex<Vec<ByteRange>>,
    writes: AtomicUsize,
    fail_write: AtomicBool,
    fail_erase: AtomicBool,
}
impl BytesPartialDecoderTraits for Handle {
    fn exists(&self) -> Result<bool, StorageError> {
        self.bytes.exists()
    }
    fn size_held(&self) -> usize {
        self.bytes.size_held()
    }
    fn supports_partial_decode(&self) -> bool {
        true
    }
    fn partial_decode_many(
        &self,
        ranges: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError> {
        let ranges: Vec<_> = ranges.collect();
        self.reads.lock().unwrap().extend(&ranges);
        self.bytes
            .partial_decode_many(Box::new(ranges.into_iter()), options)
    }
}
impl BytesPartialEncoderTraits for Handle {
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn BytesPartialDecoderTraits> {
        self
    }
    fn supports_partial_encode(&self) -> bool {
        true
    }
    fn erase(&self) -> Result<(), CodecError> {
        if self.fail_erase.load(Ordering::SeqCst) {
            return Err(StorageError::Other("erase failure".into()).into());
        }
        self.writes.fetch_add(1, Ordering::SeqCst);
        self.bytes.erase()
    }
    fn partial_encode_many(
        &self,
        values: OffsetBytesIterator<ArrayBytesRaw<'_>>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if self.fail_write.load(Ordering::SeqCst) {
            return Err(StorageError::Other("write failure".into()).into());
        }
        self.writes.fetch_add(1, Ordering::SeqCst);
        self.bytes.partial_encode_many(values, options)
    }
}
fn codec(
    dtype: &DataType,
    location: ShardingIndexLocation,
    compressed: bool,
) -> Arc<ShardingCodec> {
    let mut builder = ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], dtype);
    builder
        .index_location(location)
        .index_bytes_to_bytes_codecs(vec![]);
    #[cfg(feature = "gzip")]
    if compressed {
        builder.bytes_to_bytes_codecs(vec![Arc::new(
            zarrs::array::codec::GzipCodec::new(1).unwrap(),
        )]);
    }
    #[cfg(not(feature = "gzip"))]
    let _ = compressed;
    builder.build_arc()
}
fn cases() -> Vec<(
    DataType,
    FillValue,
    ArrayBytes<'static>,
    ArrayBytes<'static>,
)> {
    let mut cases = vec![(
        data_type::uint8(),
        0u8.into(),
        (1..=16u8).collect::<Vec<_>>().into(),
        vec![90, 91, 92, 93, 94].into(),
    )];
    let optional = data_type::uint8().to_optional().to_optional();
    cases.push((
        optional.clone(),
        FillValue::new_optional_null(),
        (1..=16u8)
            .map(|i| Some(Some(i)))
            .collect::<Vec<_>>()
            .into_array_bytes(&optional)
            .unwrap(),
        vec![
            Some(Some(90u8)),
            None,
            Some(None),
            Some(Some(93)),
            Some(Some(94)),
        ]
        .into_array_bytes(&optional)
        .unwrap(),
    ));
    let strings = data_type::string();
    cases.push((
        strings.clone(),
        "".into(),
        (1..=16)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .into_array_bytes(&strings)
            .unwrap(),
        vec!["first", "", "third", "fourth", "last"]
            .into_array_bytes(&strings)
            .unwrap(),
    ));
    cases
}

// Run the same selections and storage assertions through both encoder traits.
macro_rules! write_cases {
    ($new:ident, $wait:ident) => {{

    for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
        for target in [0, 1, 2, 8] {
            for compressed in [false, true] {
                for populated in [false, true] {
                    for (dtype, fill, original, update) in cases() {
                        let codec = codec(&dtype, location, compressed);
                        let shape = vec![NonZeroU64::new(4).unwrap(); 2];
                        let options = CodecOptions::default().with_concurrent_target(target);
                        let handle = Arc::new(Handle::default());
                        let mut expected = if populated {
                            original
                        } else {
                            ArrayBytes::new_fill_value(&dtype, 16, &fill).unwrap()
                        };
                        if populated {
                            *handle.bytes.lock().unwrap() = Some(
                                codec
                                    .encode(expected.clone(), &shape, &dtype, &fill, &options)
                                    .unwrap()
                                    .into_owned(),
                            );
                        }
                        let initial_encoded = handle.bytes.lock().unwrap().clone();
                        let encoder = $wait!(codec
                            .clone()
                            .$new(handle.clone(), &shape, &dtype, &fill, &options))
                            .unwrap();
                        handle.reads.lock().unwrap().clear();
                        // Two affected subchunks, with interleaved duplicates in the first.
                        let indices =
                            vec![vec![0, 0], vec![3, 3], vec![0, 1], vec![3, 2], vec![0, 0]];
                        $wait!(encoder.partial_encode(&indices, &update, &options)).unwrap();
                        assert_eq!(
                            handle.reads.lock().unwrap().len(),
                            if populated { 2 } else { 0 }
                        );
                        expected =
                            update_array_bytes(expected, &[4, 4], &indices, &update, dtype.size())
                                .unwrap();
                        assert_eq!(
                            $wait!(encoder
                                .partial_decode(&ArraySubset::new_with_shape(vec![4, 4]), &options))
                                .unwrap(),
                            expected
                        );
                        let stored = handle.bytes.lock().unwrap().clone().unwrap();
                        assert_eq!(
                            codec
                                .decode(stored.into(), &shape, &dtype, &fill, &options)
                                .unwrap(),
                            expected
                        );
                        // The two untouched subchunks retain their original encoded spans.
                        if let Some(initial) = initial_encoded {
                            let stored = handle.bytes.lock().unwrap().clone().unwrap();
                            let index_start = |len: usize| match location {
                                ShardingIndexLocation::Start => 0,
                                ShardingIndexLocation::End => len - 64,
                            };
                            for chunk in [1, 2] {
                                let before = index_start(initial.len()) + chunk * 16;
                                let after = index_start(stored.len()) + chunk * 16;
                                assert_eq!(
                                    &initial[before..before + 16],
                                    &stored[after..after + 16]
                                );
                            }
                            let full = codec
                                .encode(expected.clone(), &shape, &dtype, &fill, &options)
                                .unwrap();
                            assert_eq!(
                                codec.decode(full, &shape, &dtype, &fill, &options).unwrap(),
                                expected
                            );
                        }
                        // Full replacement of one subchunk must not read its old data.
                        handle.reads.lock().unwrap().clear();
                        let subset = ArraySubset::new_with_ranges(&[0..2, 0..2]);
                        let values = expected
                            .extract_array_subset(&subset, &[4, 4], &dtype)
                            .unwrap()
                            .into_owned();
                        $wait!(encoder.partial_encode(&subset, &values, &options)).unwrap();
                        assert!(handle.reads.lock().unwrap().is_empty());
                        // Invalid selection, byte count, and empty selection do not mutate storage.
                        let before = handle.bytes.lock().unwrap().clone();
                        let writes = handle.writes.load(Ordering::SeqCst);
                        assert!(
                            $wait!(encoder
                                .partial_encode(&vec![vec![0]], &update, &options))
                                .is_err()
                        );
                        assert!(
                            $wait!(encoder
                                .partial_encode(&vec![vec![4, 0]; 5], &update, &options))
                                .is_err()
                        );
                        assert!($wait!(encoder.partial_encode(&indices, &values, &options)).is_err());
                        let empty = ArraySubset::new_with_shape(vec![0, 4]);
                        let empty_values = ArrayBytes::new_fill_value(&dtype, 0, &fill).unwrap();
                        $wait!(encoder
                            .partial_encode(&empty, &empty_values, &options))
                            .unwrap();
                        $wait!(encoder
                            .partial_encode(&vec![empty.clone()], &empty_values, &options))
                            .unwrap();
                        assert_eq!(handle.writes.load(Ordering::SeqCst), writes);
                        assert_eq!(*handle.bytes.lock().unwrap(), before);
                        // Failed writes and erases must leave the cached index usable.
                        handle.fail_write.store(true, Ordering::SeqCst);
                        assert!($wait!(encoder.partial_encode(&subset, &values, &options)).is_err());
                        handle.fail_write.store(false, Ordering::SeqCst);
                        handle.fail_erase.store(true, Ordering::SeqCst);
                        assert!($wait!(encoder.erase()).is_err());
                        handle.fail_erase.store(false, Ordering::SeqCst);
                        assert_eq!(
                            $wait!(encoder
                                .partial_decode(&ArraySubset::new_with_shape(vec![4, 4]), &options))
                                .unwrap(),
                            expected
                        );
                        $wait!(encoder.erase()).unwrap();
                        assert!(!$wait!(encoder.exists()).unwrap());
                        let empty_shard = ArrayBytes::new_fill_value(&dtype, 16, &fill).unwrap();
                        assert_eq!(
                            $wait!(encoder
                                .partial_decode(&ArraySubset::new_with_shape(vec![4, 4]), &options))
                                .unwrap(),
                            empty_shard
                        );
                        $wait!(encoder.partial_encode(&indices, &update, &options)).unwrap();
                        let expected = update_array_bytes(
                            empty_shard,
                            &[4, 4],
                            &indices,
                            &update,
                            dtype.size(),
                        )
                        .unwrap();
                        assert_eq!(
                            $wait!(encoder
                                .partial_decode(&ArraySubset::new_with_shape(vec![4, 4]), &options))
                                .unwrap(),
                            expected
                        );
                        let full = ArraySubset::new_with_shape(vec![4, 4]);
                        $wait!(encoder
                            .partial_encode(
                                &full,
                                &ArrayBytes::new_fill_value(&dtype, 16, &fill).unwrap(),
                                &options,
                            ))
                            .unwrap();
                        assert!(!$wait!(encoder.exists()).unwrap());
                        assert!(handle.bytes.lock().unwrap().is_none());
                    }
                }
            }
        }
    }

    }};
}
macro_rules! immediate {
    ($value:expr) => {
        $value
    };
}
#[test]
fn sync_partial_writes() {
    write_cases!(partial_encoder, immediate);
}

#[cfg(feature = "async")]
macro_rules! awaited {
    ($value:expr) => {
        $value.await
    };
}
#[cfg(feature = "async")]
#[tokio::test]
async fn async_partial_writes() {
    write_cases!(async_partial_encoder, awaited);
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl zarrs_codec::AsyncBytesPartialDecoderTraits for Handle {
    async fn exists(&self) -> Result<bool, StorageError> {
        BytesPartialDecoderTraits::exists(self)
    }
    fn size_held(&self) -> usize {
        BytesPartialDecoderTraits::size_held(self)
    }
    fn supports_partial_decode(&self) -> bool {
        true
    }
    async fn partial_decode_many<'a>(
        &'a self,
        ranges: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
        // Yield so competing operations can exercise the encoder's async mutex.
        tokio::task::yield_now().await;
        BytesPartialDecoderTraits::partial_decode_many(self, ranges, options)
    }
}
#[cfg(feature = "async")]
#[async_trait::async_trait]
impl zarrs_codec::AsyncBytesPartialEncoderTraits for Handle {
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn zarrs_codec::AsyncBytesPartialDecoderTraits> {
        self
    }
    fn supports_partial_encode(&self) -> bool {
        true
    }
    async fn erase(&self) -> Result<(), CodecError> {
        tokio::task::yield_now().await;
        BytesPartialEncoderTraits::erase(self)
    }
    async fn partial_encode_many<'a>(
        &'a self,
        values: OffsetBytesIterator<'a, ArrayBytesRaw<'_>>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        tokio::task::yield_now().await;
        BytesPartialEncoderTraits::partial_encode_many(self, values, options)
    }
}

#[cfg(feature = "async")]
#[tokio::test]
async fn concurrent_async_updates_share_the_cached_index() {
    let dtype = data_type::uint8();
    let shape = vec![NonZeroU64::new(4).unwrap(); 2];
    let options = CodecOptions::default().with_concurrent_target(2);
    let codec = codec(&dtype, ShardingIndexLocation::End, false);
    let handle = Arc::new(Handle::default());
    let encoder = codec
        .async_partial_encoder(handle, &shape, &dtype, &0u8.into(), &options)
        .await
        .unwrap();
    let one = vec![vec![0, 0], vec![3, 3]];
    let two = vec![vec![0, 1], vec![3, 2]];
    let bytes_one = vec![7u8, 8].into();
    let bytes_two = vec![9u8, 10].into();
    let (first, second) = tokio::join!(
        encoder.partial_encode(&one, &bytes_one, &options),
        encoder.partial_encode(&two, &bytes_two, &options)
    );
    first.unwrap();
    second.unwrap();
    let decoded = encoder
        .partial_decode(&ArraySubset::new_with_shape(vec![4, 4]), &options)
        .await
        .unwrap()
        .into_fixed()
        .unwrap();
    assert_eq!(
        decoded.as_ref(),
        &[7, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 8]
    );
}

mod full_write_fallback {
    use super::*;
    use zarrs_storage::store::MemoryStore;
    use zarrs_storage::{
        Bytes, MaybeBytesIterator, ReadableStorageTraits, StoreKey, StorePrefix,
        WritableStorageTraits,
    };
    #[derive(Default)]
    struct FullWritesOnly(MemoryStore);
    impl ReadableStorageTraits for FullWritesOnly {
        fn get_partial_many<'a>(
            &'a self,
            key: &StoreKey,
            ranges: ByteRangeIterator<'a>,
        ) -> Result<MaybeBytesIterator<'a>, StorageError> {
            self.0.get_partial_many(key, ranges)
        }
        fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
            self.0.size_key(key)
        }
        fn supports_get_partial(&self) -> bool {
            true
        }
    }
    impl WritableStorageTraits for FullWritesOnly {
        fn set(&self, key: &StoreKey, bytes: Bytes) -> Result<(), StorageError> {
            self.0.set(key, bytes)
        }
        fn set_partial_many(
            &self,
            _: &StoreKey,
            _: OffsetBytesIterator,
        ) -> Result<(), StorageError> {
            panic!("store does not support partial writes")
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
    fn builder(location: ShardingIndexLocation) -> zarrs::array::ArrayBuilder {
        let dtype = data_type::uint8();
        let mut builder =
            zarrs::array::ArrayBuilder::new(vec![4, 4], vec![4, 4], dtype.clone(), 0u8);
        builder.array_to_bytes_codec(codec(&dtype, location, false));
        builder
    }
    #[test]
    fn sync_store_without_partial_writes() {
        for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
            let array = builder(location)
                .build(Arc::new(FullWritesOnly::default()), "/")
                .unwrap();
            let options = CodecOptions::default().with_experimental_partial_encoding(true);
            array.store_chunk(&[0, 0], vec![1u8; 16]).unwrap();
            array
                .store_chunk_subset_opt(
                    &[0, 0],
                    &ArraySubset::new_with_ranges(&[1..3, 1..3]),
                    vec![2u8; 4],
                    &options,
                )
                .unwrap();
            assert_eq!(
                array.retrieve_chunk::<Vec<u8>>(&[0, 0]).unwrap(),
                [1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1]
            );
        }
    }
    #[cfg(feature = "async")]
    struct SpawnBlocking;
    #[cfg(feature = "async")]
    impl zarrs_storage::storage_adapter::sync_to_async::SyncToAsyncSpawnBlocking for SpawnBlocking {
        async fn spawn_blocking<F, R>(&self, f: F) -> R
        where
            F: FnOnce() -> R + Send + 'static,
            R: Send + 'static,
        {
            tokio::task::spawn_blocking(f).await.unwrap()
        }
    }
    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_store_without_partial_writes() {
        for location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
            let store = Arc::new(
                zarrs_storage::storage_adapter::sync_to_async::SyncToAsyncStorageAdapter::new(
                    Arc::new(FullWritesOnly::default()),
                    SpawnBlocking,
                ),
            );
            let array = builder(location).build(store, "/").unwrap();
            let options = CodecOptions::default().with_experimental_partial_encoding(true);
            array
                .async_store_chunk(&[0, 0], vec![1u8; 16])
                .await
                .unwrap();
            array
                .async_store_chunk_subset_opt(
                    &[0, 0],
                    &ArraySubset::new_with_ranges(&[1..3, 1..3]),
                    vec![2u8; 4],
                    &options,
                )
                .await
                .unwrap();
            assert_eq!(
                array
                    .async_retrieve_chunk::<Vec<u8>>(&[0, 0])
                    .await
                    .unwrap(),
                [1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1]
            );
        }
    }
}
