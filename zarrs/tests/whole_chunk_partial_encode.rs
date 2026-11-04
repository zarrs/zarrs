#![allow(missing_docs)]

//! Partial encodes that replace a whole chunk do not read the existing chunk.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::{
    ArrayBytes, ArrayBytesOffsets, ArraySubset, CodecChain, DataType, FillValue, data_type,
};
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::store::MemoryStore;
use zarrs::storage::{StoreKey, WritableStorageTraits};
use zarrs_codec::{CodecOptions, CodecSpecificOptions};

/// Bytes that no case can decode.
const UNDECODABLE: &[u8] = b"garbage";

#[path = "common/planned_codec.rs"]
mod planned_codec;
use planned_codec::TestCodec;
use zarrs_codec::Resources;

struct Case {
    name: &'static str,
    codec: Arc<TestCodec>,
    /// An entire chunk that is not the fill value.
    data: ArrayBytes<'static>,
}

fn bind(codecs: &str, data_type: DataType, fill_value: FillValue) -> Arc<TestCodec> {
    let metadata: Vec<MetadataV3> = serde_json::from_str(codecs).unwrap();
    let chain = CodecChain::from_metadata(&metadata).unwrap();
    let bound = chain
        .with_context(
            data_type,
            Some(fill_value),
            &CodecSpecificOptions::default(),
        )
        .unwrap();
    Arc::new(TestCodec { chain, bound })
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
    };
    #[allow(unused_mut)]
    let mut cases = vec![
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
        },
    ];
    #[cfg(feature = "gzip")]
    cases.push(f32_case(
        "bytes-gzip",
        r#"[{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "gzip", "configuration": {"level": 1}}]"#,
    ));
    cases
}

fn shape() -> Vec<NonZeroU64> {
    vec![NonZeroU64::new(4).unwrap()]
}

#[test]
fn whole_chunk_partial_encode() {
    let options = CodecOptions::default();
    let whole = ArraySubset::new_with_ranges(&[0..4]);
    for case in cases() {
        let name = case.name;
        let store = Arc::new(MemoryStore::new());
        let key = StoreKey::new("c/0").unwrap();
        store.set(&key, UNDECODABLE.to_vec().into()).unwrap();
        let handle = Arc::new((store.clone(), key.clone()));

        // The existing chunk is replaced without being decoded
        case.codec
            .clone()
            .partial_encoder(handle.clone(), &shape(), &options)
            .unwrap()
            .partial_encode(&whole, &case.data, &options, &Resources::default())
            .unwrap_or_else(|err| panic!("{name}: {err}"));

        let decoded = case
            .codec
            .clone()
            .partial_decoder(handle, &shape(), &options)
            .unwrap()
            .partial_decode(&whole, &options, &Resources::default())
            .unwrap()
            .into_owned();
        assert_eq!(decoded, case.data, "{name}");
    }
}

#[cfg(feature = "async")]
mod r#async {
    use super::*;
    use zarrs::storage::AsyncWritableStorageTraits;
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
    async fn async_whole_chunk_partial_encode() {
        let options = CodecOptions::default();
        let whole = ArraySubset::new_with_ranges(&[0..4]);
        for case in cases() {
            let name = case.name;
            let store = Arc::new(SyncToAsyncStorageAdapter::new(
                Arc::new(MemoryStore::new()),
                TokioSpawnBlocking,
            ));
            let key = StoreKey::new("c/0").unwrap();
            store.set(&key, UNDECODABLE.to_vec().into()).await.unwrap();
            let handle = Arc::new((store.clone(), key.clone()));

            // The existing chunk is replaced without being decoded
            case.codec
                .clone()
                .async_partial_encoder(handle.clone(), &shape(), &options)
                .await
                .unwrap()
                .partial_encode(&whole, &case.data, &options, &Resources::default())
                .await
                .unwrap_or_else(|err| panic!("{name}: {err}"));

            let decoded = case
                .codec
                .clone()
                .async_partial_decoder(handle, &shape(), &options)
                .await
                .unwrap()
                .partial_decode(&whole, &options, &Resources::default())
                .await
                .unwrap()
                .into_owned();
            assert_eq!(decoded, case.data, "{name}");
        }
    }
}
