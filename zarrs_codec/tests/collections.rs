#![allow(missing_docs)]

use std::sync::Mutex;
use zarrs_codec::{BytesPartialDecoderTraits, CodecOptions};
use zarrs_storage::byte_range::ByteRange;
use zarrs_storage::store::MemoryStore;
use zarrs_storage::{StoreKey, WritableStorageTraits};

#[test]
fn partial_decode_collections() {
    let store = MemoryStore::new();
    let key = StoreKey::new("data").unwrap();
    store.set(&key, vec![42, 43].into()).unwrap();
    let storage_decoder = (store, key);
    let mutex_decoder = Mutex::new(Some(vec![42, 43]));
    let options = CodecOptions::default();
    for decoder in [
        &storage_decoder as &dyn BytesPartialDecoderTraits,
        &mutex_decoder,
    ] {
        for count in [0, 1, 1024] {
            let ranges = (0..count).map(|i| ByteRange::FromStart(i % 2, Some(1)));
            let values = decoder
                .partial_decode_many(Box::new(ranges), &options)
                .unwrap()
                .unwrap();
            assert_eq!(values.len(), usize::try_from(count).unwrap());
            for (i, value) in values.iter().enumerate() {
                assert_eq!(value.as_ref(), &[if i % 2 == 0 { 42 } else { 43 }]);
            }
        }
        let ranges = [
            ByteRange::FromStart(0, Some(1)),
            ByteRange::FromStart(2, Some(1)),
        ]
        .into_iter()
        .chain(std::iter::once_with(|| {
            panic!("must stop at the first invalid range")
        }));
        assert!(
            decoder
                .partial_decode_many(Box::new(ranges), &options)
                .is_err()
        );
    }
    let missing = (MemoryStore::new(), StoreKey::new("missing").unwrap());
    let empty = Mutex::new(None);
    for decoder in [&missing as &dyn BytesPartialDecoderTraits, &empty] {
        let ranges = std::iter::once_with(|| panic!("missing values must not consume ranges"));
        assert!(
            decoder
                .partial_decode_many(Box::new(ranges), &options)
                .unwrap()
                .is_none()
        );
    }
}

#[cfg(feature = "async")]
#[test]
fn async_partial_decode_collections() {
    use zarrs_codec::AsyncBytesPartialDecoderTraits;
    use zarrs_storage::AsyncWritableStorageTraits;
    use zarrs_storage::store::AsyncMemoryStore;

    futures::executor::block_on(async {
        let store = AsyncMemoryStore::new();
        let key = StoreKey::new("data").unwrap();
        store.set(&key, vec![42, 43].into()).await.unwrap();
        let decoder = (store, key);
        let options = CodecOptions::default();
        for count in [0, 1, 1024] {
            let ranges = (0..count).map(|i| ByteRange::FromStart(i % 2, Some(1)));
            let values = decoder
                .partial_decode_many(Box::new(ranges), &options)
                .await
                .unwrap()
                .unwrap();
            assert_eq!(values.len(), usize::try_from(count).unwrap());
            for (i, value) in values.iter().enumerate() {
                assert_eq!(value.as_ref(), &[if i % 2 == 0 { 42 } else { 43 }]);
            }
        }
        let ranges = [
            ByteRange::FromStart(0, Some(1)),
            ByteRange::FromStart(2, Some(1)),
        ];
        assert!(
            decoder
                .partial_decode_many(Box::new(ranges.into_iter()), &options)
                .await
                .is_err()
        );
        let missing = (AsyncMemoryStore::new(), StoreKey::new("missing").unwrap());
        let ranges = std::iter::once_with(|| panic!("missing values must not consume ranges"));
        assert!(
            missing
                .partial_decode_many(Box::new(ranges), &options)
                .await
                .unwrap()
                .is_none()
        );
    });
}
