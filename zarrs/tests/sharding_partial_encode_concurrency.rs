//! Verify partial shard updates respect outer and inner codec concurrency limits.
#![allow(missing_docs)]
#![cfg(not(target_arch = "wasm32"))]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use zarrs::array::{ArrayBuilder, BytesRepresentation, CowBytes, data_type};
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
    fn record(&self, options: &CodecOptions) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.largest_target
            .fetch_max(options.concurrent_target(), Ordering::Relaxed);
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
        options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.record(options);
        Ok(bytes)
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        _representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.record(options);
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
            .with_codec_options(
                CodecOptions::default()
                    .with_concurrent_target(4)
                    .with_experimental_partial_encoding(true),
            );
        array
            .store_array_subset(&[1..15, 1..15], vec![7u8; 196])
            .unwrap();
        codec.check();
        let decoded: Vec<u8> = array.retrieve_array_subset(&[1..15, 1..15]).unwrap();
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
                    CodecOptions::default()
                        .with_concurrent_target(4)
                        .with_experimental_partial_encoding(true),
                );
            array
                .async_store_array_subset(&[1..15, 1..15], vec![7u8; 196])
                .await
                .unwrap();
            codec.check();
            let decoded: Vec<u8> = array
                .async_retrieve_array_subset(&[1..15, 1..15])
                .await
                .unwrap();
            assert_eq!(decoded, vec![7u8; 196]);
        });
    });
}
