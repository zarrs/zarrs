//! Verify partial shard updates respect outer and inner codec concurrency limits.
#![allow(missing_docs)]
#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use zarrs::array::{
    ArrayBuilder, BytesRepresentation, ComputeResources, CowBytes, Resources, data_type,
};
use zarrs::metadata::Configuration;
use zarrs::storage::store::MemoryStore;
use zarrs_codec::{
    BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
use zarrs_plugin::ZarrVersion;

#[derive(Debug, Default)]
struct RecordingCodec {
    active: AtomicUsize,
    peak: AtomicUsize,
    calls: AtomicUsize,
    largest_target: AtomicUsize,
}
zarrs_plugin::impl_extension_aliases!(RecordingCodec, v3: "test.concurrency");

impl CodecTraits for RecordingCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(Configuration::default())
    }
    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }
    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}
impl RecordingCodec {
    fn reset(&self) {
        self.calls.store(0, Ordering::Relaxed);
        self.largest_target.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
    }

    fn record(&self, resources: &ComputeResources) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.largest_target
            .fetch_max(resources.compute_target(), Ordering::Relaxed);
        let active = self.active.fetch_add(1, Ordering::Relaxed) + 1;
        self.peak.fetch_max(active, Ordering::Relaxed);
        // Give other subchunk jobs an opportunity to enter this codec.
        std::thread::sleep(std::time::Duration::from_millis(1));
        self.active.fetch_sub(1, Ordering::Relaxed);
    }
    fn check(&self) {
        assert_eq!(self.calls.load(Ordering::Relaxed), 16);
        assert_eq!(self.largest_target.load(Ordering::Relaxed), 2);
        assert!(self.peak.load(Ordering::Relaxed) <= 2);
    }
}

#[test]
fn nested_sharding_preserves_encoded_bytes_and_unconstrained_compute() {
    use std::num::NonZeroU64;
    use zarrs::array::codec::array_to_bytes::sharding::{
        ShardingCodecBuilder, ShardingCodecOptions, SubchunkWriteOrder,
    };

    let codec = Arc::new(RecordingCodec::default());
    let data_type = data_type::uint8();
    let mut inner = ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type);
    inner.bytes_to_bytes_codecs(vec![codec.clone()]);
    let mut outer = ShardingCodecBuilder::new(vec![NonZeroU64::new(4).unwrap(); 2], &data_type);
    outer.array_to_bytes_codec(inner.build_arc());
    let array = ArrayBuilder::new(vec![8, 8], vec![8, 8], data_type, 0u8)
        .array_to_bytes_codec(outer.build_arc())
        .codec_specific_options(
            zarrs_codec::CodecSpecificOptions::default().with_option(
                ShardingCodecOptions::default()
                    .with_subchunk_write_order(SubchunkWriteOrder::Lexicographic),
            ),
        )
        .build(Arc::new(MemoryStore::new()), "/")
        .unwrap();
    let data: Vec<u8> = (1..=64).collect();
    let mut expected_encoding = None;
    for target in [1, 4, 0] {
        let resources = Resources::default().with_compute_target(target);
        codec.reset();
        array.store_chunk(&[0, 0], &data, &resources).unwrap();
        assert_eq!(codec.calls.load(Ordering::Relaxed), 16);
        if target == 0 {
            assert_eq!(codec.largest_target.load(Ordering::Relaxed), 0);
        }
        let encoded = array.retrieve_encoded_chunk(&[0, 0]).unwrap().unwrap();
        if let Some(expected) = &expected_encoding {
            assert_eq!(&encoded, expected);
        } else {
            expected_encoding = Some(encoded);
        }
        codec.reset();
        assert_eq!(
            array
                .retrieve_chunk::<Vec<u8>>(&[0, 0], &resources)
                .unwrap(),
            data
        );
        assert_eq!(codec.calls.load(Ordering::Relaxed), 16);
        if target == 0 {
            assert_eq!(codec.largest_target.load(Ordering::Relaxed), 0);
        }
    }
}

#[test]
fn encoded_cache_uses_the_resources_of_each_read() {
    use zarrs::array::chunk_cache::ChunkCacheEncodedLruChunkLimit;
    use zarrs::array::{ArrayCached, Resources};

    let codec = Arc::new(RecordingCodec::default());
    let array = ArrayBuilder::new(vec![4], vec![4], data_type::uint8(), 0u8)
        .bytes_to_bytes_codecs(vec![codec.clone()])
        .build_arc(Arc::new(MemoryStore::new()), "/")
        .unwrap();
    let data = vec![1u8, 2, 3, 4];
    array
        .store_chunk(&[0], &data, &Resources::default())
        .unwrap();
    let cached = ArrayCached::new(array, ChunkCacheEncodedLruChunkLimit::new(1));
    for target in [7, 3, 0] {
        codec.reset();
        assert_eq!(
            cached
                .retrieve_chunk::<Vec<u8>>(&[0], &Resources::default().with_compute_target(target))
                .unwrap(),
            data,
        );
        assert_eq!(codec.calls.load(Ordering::Relaxed), 1);
        assert_eq!(codec.largest_target.load(Ordering::Relaxed), target);
    }
}
#[cfg_attr(feature = "async", async_trait::async_trait)]
impl BytesToBytesCodecTraits for RecordingCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self
    }
    fn recommended_concurrency(
        &self,
        _representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new(2..2))
    }
    fn encoded_representation(&self, representation: &BytesRepresentation) -> BytesRepresentation {
        *representation
    }
    fn encode<'a>(
        &self,
        bytes: CowBytes<'a>,
        _options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.record(resources);
        Ok(bytes)
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        _representation: &BytesRepresentation,
        _options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.record(resources);
        Ok(bytes)
    }
}

#[test]
fn partial_encode_limits_subchunk_concurrency() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    pool.install(|| {
        let codec = Arc::new(RecordingCodec::default());
        let array = ArrayBuilder::new(vec![16, 16], vec![16, 16], data_type::uint8(), 0u8)
            .subchunk_shape(vec![4, 4])
            .bytes_to_bytes_codecs(vec![codec.clone()])
            .build(Arc::new(MemoryStore::new()), "/")
            .unwrap()
            .with_codec_options(CodecOptions::default().with_experimental_partial_encoding(true));
        array
            .store_array_subset(
                &[1..15, 1..15],
                vec![7u8; 196],
                &Resources::default().with_compute_target(4),
            )
            .unwrap();
        codec.check();
        let decoded: Vec<u8> = array
            .retrieve_array_subset(
                &[1..15, 1..15],
                &Resources::default().with_compute_target(4),
            )
            .unwrap();
        assert_eq!(decoded, vec![7u8; 196]);
    });
}

#[cfg(feature = "async")]
#[test]
fn async_partial_encode_limits_subchunk_concurrency() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    pool.install(|| {
        futures::executor::block_on(async {
            let codec = Arc::new(RecordingCodec::default());
            let array = ArrayBuilder::new(vec![16, 16], vec![16, 16], data_type::uint8(), 0u8)
                .subchunk_shape(vec![4, 4])
                .bytes_to_bytes_codecs(vec![codec.clone()])
                .build(
                    Arc::new(zarrs::storage::store::AsyncMemoryStore::new()),
                    "/",
                )
                .unwrap()
                .with_codec_options(
                    CodecOptions::default().with_experimental_partial_encoding(true),
                );
            array
                .async_store_array_subset(
                    &[1..15, 1..15],
                    vec![7u8; 196],
                    &Resources::default().with_compute_target(4),
                )
                .await
                .unwrap();
            codec.check();
            let decoded: Vec<u8> = array
                .async_retrieve_array_subset(
                    &[1..15, 1..15],
                    &Resources::default().with_compute_target(4),
                )
                .await
                .unwrap();
            assert_eq!(decoded, vec![7u8; 196]);
        });
    });
}
