#![allow(missing_docs)]

use zarrs_storage::byte_range::ByteRange;
use zarrs_storage::store::MemoryStore;
use zarrs_storage::{ReadableStorageTraits, StoreKey, WritableStorageTraits};

#[test]
fn partial_decode_collections() {
    let store = MemoryStore::new();
    let key = StoreKey::new("data").unwrap();
    store.set(&key, vec![42, 43].into()).unwrap();
    {
        let decoder = &store;
        for count in [0, 1, 1024] {
            let ranges = (0..count).map(|i| ByteRange::FromStart(i % 2, Some(1)));
            let values = decoder
                .get_partial_many(&key, Box::new(ranges))
                .unwrap()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
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
                .get_partial_many(&key, Box::new(ranges))
                .unwrap()
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .is_err()
        );
    }
}

#[cfg(feature = "async")]
#[test]
fn async_partial_decode_collections() {
    use futures::TryStreamExt;
    use zarrs_storage::store::AsyncMemoryStore;
    use zarrs_storage::{AsyncReadableStorageTraits, AsyncWritableStorageTraits};

    futures::executor::block_on(async {
        let store = AsyncMemoryStore::new();
        let key = StoreKey::new("data").unwrap();
        store.set(&key, vec![42, 43].into()).await.unwrap();
        let decoder = store;

        for count in [0, 1, 1024] {
            let ranges = (0..count).map(|i| ByteRange::FromStart(i % 2, Some(1)));
            let values = decoder
                .get_partial_many(&key, Box::new(ranges))
                .await
                .unwrap()
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
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
                .get_partial_many(&key, Box::new(ranges.into_iter()))
                .await
                .unwrap()
                .unwrap()
                .try_collect::<Vec<_>>()
                .await
                .is_err()
        );
    });
}
