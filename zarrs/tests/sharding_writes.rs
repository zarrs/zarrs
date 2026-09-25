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
            return Err(CodecError::from("erase failure"));
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
            return Err(CodecError::from("write failure"));
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

#[test]
fn sync_partial_writes() {
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
                        let encoder = codec
                            .clone()
                            .partial_encoder(handle.clone(), &shape, &dtype, &fill, &options)
                            .unwrap();
                        handle.reads.lock().unwrap().clear();
                        // Two affected subchunks, with interleaved duplicates in the first.
                        let indices =
                            vec![vec![0, 0], vec![3, 3], vec![0, 1], vec![3, 2], vec![0, 0]];
                        encoder.partial_encode(&indices, &update, &options).unwrap();
                        assert_eq!(
                            handle.reads.lock().unwrap().len(),
                            if populated { 2 } else { 0 }
                        );
                        expected =
                            update_array_bytes(expected, &[4, 4], &indices, &update, dtype.size())
                                .unwrap();
                        assert_eq!(
                            encoder
                                .partial_decode(&ArraySubset::new_with_shape(vec![4, 4]), &options)
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
                        encoder.partial_encode(&subset, &values, &options).unwrap();
                        assert!(handle.reads.lock().unwrap().is_empty());
                        // Invalid selection, byte count, and empty selection do not mutate storage.
                        let before = handle.bytes.lock().unwrap().clone();
                        let writes = handle.writes.load(Ordering::SeqCst);
                        assert!(
                            encoder
                                .partial_encode(&vec![vec![0]], &update, &options)
                                .is_err()
                        );
                        assert!(
                            encoder
                                .partial_encode(&vec![vec![4, 0]; 5], &update, &options)
                                .is_err()
                        );
                        assert!(encoder.partial_encode(&indices, &values, &options).is_err());
                        let empty = ArraySubset::new_with_shape(vec![0, 4]);
                        let empty_values = ArrayBytes::new_fill_value(&dtype, 0, &fill).unwrap();
                        encoder
                            .partial_encode(&empty, &empty_values, &options)
                            .unwrap();
                        encoder
                            .partial_encode(&vec![empty.clone()], &empty_values, &options)
                            .unwrap();
                        assert_eq!(handle.writes.load(Ordering::SeqCst), writes);
                        assert_eq!(*handle.bytes.lock().unwrap(), before);
                        // Failed writes and erases must leave the cached index usable.
                        handle.fail_write.store(true, Ordering::SeqCst);
                        assert!(encoder.partial_encode(&subset, &values, &options).is_err());
                        handle.fail_write.store(false, Ordering::SeqCst);
                        handle.fail_erase.store(true, Ordering::SeqCst);
                        assert!(encoder.erase().is_err());
                        handle.fail_erase.store(false, Ordering::SeqCst);
                        assert_eq!(
                            encoder
                                .partial_decode(&ArraySubset::new_with_shape(vec![4, 4]), &options)
                                .unwrap(),
                            expected
                        );
                        encoder.erase().unwrap();
                        assert!(!encoder.exists().unwrap());
                        let empty_shard = ArrayBytes::new_fill_value(&dtype, 16, &fill).unwrap();
                        assert_eq!(
                            encoder
                                .partial_decode(&ArraySubset::new_with_shape(vec![4, 4]), &options)
                                .unwrap(),
                            empty_shard
                        );
                        encoder.partial_encode(&indices, &update, &options).unwrap();
                        let expected = update_array_bytes(
                            empty_shard,
                            &[4, 4],
                            &indices,
                            &update,
                            dtype.size(),
                        )
                        .unwrap();
                        assert_eq!(
                            encoder
                                .partial_decode(&ArraySubset::new_with_shape(vec![4, 4]), &options)
                                .unwrap(),
                            expected
                        );
                        let full = ArraySubset::new_with_shape(vec![4, 4]);
                        encoder
                            .partial_encode(
                                &full,
                                &ArrayBytes::new_fill_value(&dtype, 16, &fill).unwrap(),
                                &options,
                            )
                            .unwrap();
                        assert!(!encoder.exists().unwrap());
                        assert!(handle.bytes.lock().unwrap().is_none());
                    }
                }
            }
        }
    }
}
