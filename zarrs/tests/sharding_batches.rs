//! Generic-indexer batching and nested option propagation backported from main.
#![cfg(feature = "sharding")]
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use zarrs::array::codec::{BytesCodec, OptionalCodec, ShardingCodecBuilder, VlenCodec};
use zarrs::array::{
    ArrayBytes, ArrayBytesRaw, ArraySubset, BytesRepresentation, CodecChain, DataType, FillValue,
    Indexer, data_type,
};
use zarrs_codec::{
    ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits,
    BytesPartialDecoderTraits, CodecError, CodecMetadataOptions, CodecOptions,
    CodecSpecificOptions, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
    RecommendedConcurrency,
};
use zarrs_metadata::Configuration;
use zarrs_plugin::ZarrVersion;
use zarrs_storage::StorageError;

#[derive(Debug, Default)]
struct Counts {
    options: AtomicUsize,
    batches: Mutex<Vec<u64>>,
    monitoring: AtomicBool,
    active: AtomicUsize,
    peak: AtomicUsize,
    encodes: AtomicUsize,
    decodes: AtomicUsize,
}
impl Counts {
    fn track<T>(&self, action: impl FnOnce() -> T) -> T {
        if !self.monitoring.load(Ordering::SeqCst) {
            return action();
        }
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak.fetch_max(active, Ordering::SeqCst);
        std::thread::sleep(std::time::Duration::from_millis(1));
        let result = action();
        self.active.fetch_sub(1, Ordering::SeqCst);
        result
    }
}
#[derive(Debug)]
struct ProbeCodec(Arc<Counts>);
zarrs_plugin::impl_extension_aliases!(ProbeCodec, v3: "test.backport_probe");
impl CodecTraits for ProbeCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn configuration(&self, _: ZarrVersion, _: &CodecMetadataOptions) -> Option<Configuration> {
        Some(serde_json::Map::new().into())
    }
    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }
    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}
impl ArrayCodecTraits for ProbeCodec {
    fn recommended_concurrency(
        &self,
        _: &[NonZeroU64],
        _: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}
#[cfg_attr(feature = "async", async_trait::async_trait)]
impl ArrayToBytesCodecTraits for ProbeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self
    }
    fn with_codec_specific_options(
        self: Arc<Self>,
        _: &CodecSpecificOptions,
    ) -> Arc<dyn ArrayToBytesCodecTraits> {
        self.0.options.fetch_add(1, Ordering::SeqCst);
        self
    }
    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
        dtype: &DataType,
        fill: &FillValue,
    ) -> Result<BytesRepresentation, CodecError> {
        BytesCodec::default().encoded_representation(shape, dtype, fill)
    }
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        dtype: &DataType,
        fill: &FillValue,
        opts: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        self.0.encodes.fetch_add(1, Ordering::SeqCst);
        self.0
            .track(|| BytesCodec::default().encode(bytes, shape, dtype, fill, opts))
    }
    fn decode<'a>(
        &self,
        bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        dtype: &DataType,
        fill: &FillValue,
        opts: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.0.decodes.fetch_add(1, Ordering::SeqCst);
        self.0
            .track(|| BytesCodec::default().decode(bytes, shape, dtype, fill, opts))
    }
    fn partial_decoder(
        self: Arc<Self>,
        input: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        dtype: &DataType,
        fill: &FillValue,
        opts: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(ProbeDecoder {
            inner: Arc::new(BytesCodec::default())
                .partial_decoder(input, shape, dtype, fill, opts)?,
            counts: self.0.clone(),
        }))
    }
}
struct ProbeDecoder {
    inner: Arc<dyn ArrayPartialDecoderTraits>,
    counts: Arc<Counts>,
}
impl ArrayPartialDecoderTraits for ProbeDecoder {
    fn data_type(&self) -> &DataType {
        self.inner.data_type()
    }
    fn exists(&self) -> Result<bool, StorageError> {
        self.inner.exists()
    }
    fn size_held(&self) -> usize {
        self.inner.size_held()
    }
    fn supports_partial_decode(&self) -> bool {
        true
    }
    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        opts: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        self.counts.batches.lock().unwrap().push(indexer.len());
        self.inner.partial_decode(indexer, opts)
    }
}

#[test]
fn generic_reads_batch_duplicates_in_indexer_order() {
    let counts = Arc::new(Counts::default());
    let dtype = data_type::uint8();
    let fill = FillValue::from(0u8);
    let shape = vec![NonZeroU64::new(4).unwrap(); 2];
    let codec = ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &dtype)
        .array_to_bytes_codec(Arc::new(ProbeCodec(counts.clone())))
        .build_arc();
    let opts = CodecOptions::default().with_concurrent_target(1);
    let encoded = codec
        .encode(
            ArrayBytes::new_flen((0u8..16).collect::<Vec<_>>()),
            &shape,
            &dtype,
            &fill,
            &opts,
        )
        .unwrap();
    let decoder = codec
        .partial_decoder(Arc::new(encoded.into_owned()), &shape, &dtype, &fill, &opts)
        .unwrap();
    let indexer = vec![vec![3, 3], vec![0, 1], vec![3, 3], vec![1, 0], vec![2, 3]];
    let decoded = decoder
        .partial_decode(&indexer, &opts)
        .unwrap()
        .into_fixed()
        .unwrap();
    assert_eq!(decoded.as_ref(), &[15, 1, 15, 4, 11]);
    let mut batches = counts.batches.lock().unwrap().clone();
    batches.sort_unstable();
    assert_eq!(batches, [2, 3]);
    assert!(decoder.partial_decode(&vec![vec![4, 0]], &opts).is_err());
    assert!(
        decoder
            .partial_decode(&ArraySubset::new_with_shape(vec![1]), &opts)
            .is_err()
    );
}

#[test]
fn nested_codecs_receive_options() {
    let counts = Arc::new(Counts::default());
    let chain = || {
        Arc::new(CodecChain::new(
            vec![],
            Arc::new(ProbeCodec(counts.clone())),
            vec![],
        ))
    };
    let optional = Arc::new(OptionalCodec::new(chain(), chain()));
    optional.with_codec_specific_options(&CodecSpecificOptions::default());
    assert_eq!(counts.options.load(Ordering::SeqCst), 2);
    let vlen = Arc::new(VlenCodec::new(
        chain(),
        chain(),
        zarrs::metadata_ext::codec::vlen::VlenIndexDataType::UInt64,
        zarrs::metadata_ext::codec::vlen::VlenIndexLocation::Start,
    ));
    vlen.with_codec_specific_options(&CodecSpecificOptions::default());
    assert_eq!(counts.options.load(Ordering::SeqCst), 4);
    let codec =
        ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type::uint8())
            .array_to_bytes_codec(Arc::new(ProbeCodec(counts.clone())))
            .build_arc();
    codec.with_codec_specific_options(&CodecSpecificOptions::default());
    assert_eq!(counts.options.load(Ordering::SeqCst), 5);
}

// Only two of four subchunks are selected, with repeated coordinates.
macro_rules! bounded_writes {
    ($encoder:expr, $wait:ident, $counts:ident, $target:ident) => {{
        let encoder = $encoder;
        $counts.encodes.store(0, Ordering::SeqCst);
        $counts.decodes.store(0, Ordering::SeqCst);
        $counts.peak.store(0, Ordering::SeqCst);
        $counts.monitoring.store(true, Ordering::SeqCst);
        let indices = vec![vec![0, 0], vec![3, 3], vec![0, 0], vec![3, 2]];
        $wait!(encoder.partial_encode(
            &indices,
            &vec![30, 31, 32, 33u8].into(),
            &CodecOptions::default().with_concurrent_target($target)
        ))
        .unwrap();
        $counts.monitoring.store(false, Ordering::SeqCst);
        assert_eq!($counts.encodes.load(Ordering::SeqCst), 2);
        assert_eq!($counts.decodes.load(Ordering::SeqCst), 2);
        let peak = $counts.peak.load(Ordering::SeqCst);
        assert!(peak > 0 && peak <= if $target == 0 { 4 } else { $target });
    }};
}
macro_rules! immediate {
    ($value:expr) => {
        $value
    };
}
#[test]
fn sync_writes_batch_and_bound_subchunk_work() {
    for target in [0, 1, 2, 4] {
        let counts = Arc::new(Counts::default());
        let dtype = data_type::uint8();
        let shape = vec![NonZeroU64::new(4).unwrap(); 2];
        let codec = ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &dtype)
            .array_to_bytes_codec(Arc::new(ProbeCodec(counts.clone())))
            .build_arc();
        let options = CodecOptions::default();
        let encoded = codec
            .encode(vec![1u8; 16].into(), &shape, &dtype, &0u8.into(), &options)
            .unwrap()
            .into_owned();
        bounded_writes!(
            codec
                .partial_encoder(
                    Arc::new(Mutex::new(Some(encoded))),
                    &shape,
                    &dtype,
                    &0u8.into(),
                    &options
                )
                .unwrap(),
            immediate,
            counts,
            target
        );
    }
}
#[cfg(feature = "async")]
macro_rules! awaited {
    ($value:expr) => {
        $value.await
    };
}
#[cfg(feature = "async")]
#[tokio::test]
async fn async_writes_batch_and_bound_subchunk_work() {
    use zarrs_storage::store::AsyncMemoryStore;
    use zarrs_storage::{AsyncWritableStorageTraits, StoreKey};
    for target in [0, 1, 2, 4] {
        let counts = Arc::new(Counts::default());
        let dtype = data_type::uint8();
        let shape = vec![NonZeroU64::new(4).unwrap(); 2];
        let codec = ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &dtype)
            .array_to_bytes_codec(Arc::new(ProbeCodec(counts.clone())))
            .build_arc();
        let options = CodecOptions::default();
        let encoded = codec
            .encode(vec![1u8; 16].into(), &shape, &dtype, &0u8.into(), &options)
            .unwrap()
            .into_owned();
        let store = Arc::new(AsyncMemoryStore::new());
        let key = StoreKey::new("shard").unwrap();
        store.set(&key, encoded.into()).await.unwrap();
        let handle = Arc::new(zarrs_codec::StoragePartialEncoder::new(
            store as zarrs_storage::AsyncReadableWritableStorage,
            key,
        ));
        bounded_writes!(
            codec
                .async_partial_encoder(handle, &shape, &dtype, &0u8.into(), &options)
                .await
                .unwrap(),
            awaited,
            counts,
            target
        );
    }
}
