#![allow(missing_docs)]

//! Partial encodes that replace a whole chunk with the fill value erase it, regardless of the codec.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::{
    ArrayBytes, ArrayBytesOffsets, ArraySubset, CodecChain, CodecChainBound, DataType, FillValue,
    Resources, data_type,
};
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{ReadableStorageTraits, StoreKey};
use zarrs_codec::{ArrayToBytesCodecTraits, CodecOptions, CodecSpecificOptions};

struct Case {
    name: &'static str,
    codec: Arc<CodecChainBound>,
    /// An entire chunk that is not the fill value.
    data: ArrayBytes<'static>,
    /// An entire chunk of the fill value.
    fill: ArrayBytes<'static>,
}

fn bind(codecs: &str, data_type: DataType, fill_value: FillValue) -> Arc<CodecChainBound> {
    let metadata: Vec<MetadataV3> = serde_json::from_str(codecs).unwrap();
    CodecChain::from_metadata(&metadata)
        .unwrap()
        .with_context(
            data_type,
            Some(fill_value),
            &CodecSpecificOptions::default(),
        )
        .unwrap()
}

fn f32_bytes(elements: &[f32]) -> ArrayBytes<'static> {
    ArrayBytes::from(
        elements
            .iter()
            .flat_map(|x| x.to_ne_bytes())
            .collect::<Vec<u8>>(),
    )
}

fn string_bytes(elements: &[&str]) -> ArrayBytes<'static> {
    let mut offsets = vec![0u32];
    for element in elements {
        offsets.push(offsets.last().unwrap() + u32::try_from(element.len()).unwrap());
    }
    ArrayBytes::new_vlen(
        elements.concat().into_bytes(),
        ArrayBytesOffsets::new(offsets).unwrap(),
    )
    .unwrap()
}

fn cases() -> Vec<Case> {
    let f32_case = |name, codecs| Case {
        name,
        codec: bind(codecs, data_type::float32(), FillValue::from(0.0f32)),
        data: f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
        fill: f32_bytes(&[0.0; 4]),
    };
    vec![
        f32_case(
            "bytes",
            r#"[{"name": "bytes", "configuration": {"endian": "little"}}]"#,
        ),
        f32_case(
            "fixedscaleoffset",
            r#"[
                {"name": "numcodecs.fixedscaleoffset", "configuration": {"offset": 100.0, "scale": 2, "dtype": "f4", "astype": "f8"}},
                {"name": "bytes", "configuration": {"endian": "little"}}
            ]"#,
        ),
        Case {
            name: "vlen-utf8",
            codec: bind(
                r#"[{"name": "vlen-utf8"}]"#,
                data_type::string(),
                FillValue::from(""),
            ),
            data: string_bytes(&["a", "bb", "ccc", "dddd"]),
            fill: string_bytes(&[""; 4]),
        },
    ]
}

fn shape() -> Vec<NonZeroU64> {
    vec![NonZeroU64::new(4).unwrap()]
}

#[test]
fn whole_chunk_partial_encode_empty() {
    let whole = ArraySubset::new_with_ranges(&[0..4]);
    for case in cases() {
        let name = case.name;
        for store_empty_chunks in [false, true] {
            let options = CodecOptions::default().with_store_empty_chunks(store_empty_chunks);
            let store = Arc::new(MemoryStore::new());
            let key = StoreKey::new("c").unwrap();
            let handle = Arc::new((store.clone(), key.clone()));
            let encode = |bytes: &ArrayBytes| {
                case.codec
                    .clone()
                    .partial_encoder(handle.clone(), &shape(), &options, &Resources::default())
                    .unwrap()
                    .partial_encode(&whole, bytes, &options, &Resources::default())
                    .unwrap();
            };
            encode(&case.data);
            assert!(store.get(&key).unwrap().is_some(), "{name}");
            encode(&case.fill);
            assert_eq!(
                store.get(&key).unwrap().is_some(),
                store_empty_chunks,
                "{name}: store_empty_chunks={store_empty_chunks}"
            );
        }
    }
}

#[cfg(feature = "async")]
mod r#async {
    use super::*;
    use zarrs::storage::AsyncReadableStorageTraits;
    use zarrs::storage::storage_adapter::sync_to_async::{
        SyncToAsyncSpawnBlocking, SyncToAsyncStorageAdapter,
    };

    struct TokioSpawnBlocking;

    impl SyncToAsyncSpawnBlocking for TokioSpawnBlocking {
        async fn spawn_blocking<F, R>(&self, f: F) -> R
        where
            F: FnOnce() -> R + Send + 'static,
            R: Send + 'static,
        {
            tokio::task::spawn_blocking(f).await.unwrap()
        }
    }

    #[tokio::test]
    async fn whole_chunk_async_partial_encode_empty() {
        let whole = ArraySubset::new_with_ranges(&[0..4]);
        for case in cases() {
            let name = case.name;
            for store_empty_chunks in [false, true] {
                let options = CodecOptions::default().with_store_empty_chunks(store_empty_chunks);
                let store = Arc::new(SyncToAsyncStorageAdapter::new(
                    Arc::new(MemoryStore::new()),
                    TokioSpawnBlocking,
                ));
                let key = StoreKey::new("c").unwrap();
                let handle = Arc::new((store.clone(), key.clone()));
                let encode = async |bytes: &ArrayBytes<'_>| {
                    case.codec
                        .clone()
                        .async_partial_encoder(
                            handle.clone(),
                            &shape(),
                            &options,
                            &Resources::default(),
                        )
                        .await
                        .unwrap()
                        .partial_encode(&whole, bytes, &options, &Resources::default())
                        .await
                        .unwrap();
                };
                encode(&case.data).await;
                assert!(store.get(&key).await.unwrap().is_some(), "{name}");
                encode(&case.fill).await;
                assert_eq!(
                    store.get(&key).await.unwrap().is_some(),
                    store_empty_chunks,
                    "{name}: store_empty_chunks={store_empty_chunks}"
                );
            }
        }
    }
}
