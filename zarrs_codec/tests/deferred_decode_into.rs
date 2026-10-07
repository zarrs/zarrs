//! Ownership and single-use route contracts for deferred decoding.
//!
//! Run under Miri with `MIRIFLAGS=-Zmiri-ignore-leaks`: Rayon's
//! `use_current_thread` pools retain their thread-local registry allocations.

use std::{
    borrow::Cow,
    num::NonZeroU64,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::ArraySubset;
use zarrs_codec::*;
use zarrs_data_type::{DataType, FillValue};
use zarrs_plugin::{ExtensionName, ZarrVersion};

#[derive(Debug, Default)]
struct Producer {
    normal: AtomicUsize,
    direct: AtomicUsize,
    fail: bool,
}

impl ExtensionName for Producer {
    fn name(&self, _: ZarrVersion) -> Option<Cow<'static, str>> {
        None
    }
}

impl CodecTraits for Producer {
    fn configuration(
        &self,
        _: ZarrVersion,
        _: &CodecMetadataOptions,
    ) -> Option<zarrs_metadata::Configuration> {
        None
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

impl BytesToBytesCodecTraits for Producer {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self
    }
    fn recommended_concurrency(
        &self,
        _: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new(1..=1))
    }
    fn encoded_representation(&self, repr: &BytesRepresentation) -> BytesRepresentation {
        *repr
    }
    fn encode<'a>(
        &self,
        bytes: CowBytes<'a>,
        _: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        Ok(bytes)
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        repr: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.normal.fetch_add(1, Ordering::Relaxed);
        assert_eq!(*repr, BytesRepresentation::FixedSize(3));
        assert_eq!(options.concurrent_target(), 7);
        if self.fail {
            Err(CodecError::Other("producer failed".into()))
        } else {
            Ok(bytes)
        }
    }
    fn decode_into(
        &self,
        bytes: CowBytes<'_>,
        repr: &BytesRepresentation,
        output: &mut [u8],
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.direct.fetch_add(1, Ordering::Relaxed);
        assert_eq!(*repr, BytesRepresentation::FixedSize(3));
        assert_eq!(options.concurrent_target(), 7);
        if self.fail {
            return Err(CodecError::Other("producer failed".into()));
        }
        Ok(copy_decoded_bytes_into(&bytes, output)?)
    }
    fn is_decode_into_efficient(&self) -> bool {
        true
    }
}

#[derive(Debug)]
struct Receiver {
    pointer: usize,
    calls: AtomicUsize,
}
impl ArrayCodecTraits for Receiver {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn data_type(&self) -> &DataType {
        panic!("unused")
    }
    fn fill_value(&self) -> &FillValue {
        panic!("unused")
    }
    fn recommended_concurrency(
        &self,
        _: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new(1..=1))
    }
}
impl ArrayToBytesCodecNoSubchunkingTraits for Receiver {}
impl ArrayToBytesCodecTraits for Receiver {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self
    }
    fn encoded_representation(&self, _: &[NonZeroU64]) -> Result<BytesRepresentation, CodecError> {
        Ok(BytesRepresentation::FixedSize(3))
    }
    fn encode<'a>(
        &self,
        _: ArrayBytes<'a>,
        _: &[NonZeroU64],
        _: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        unreachable!()
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(shape, &[NonZeroU64::new(3).unwrap()]);
        assert_eq!(options.concurrent_target(), 7);
        // Reclaiming a unique allocation must not allocate or copy.
        let bytes = bytes.into_vec();
        assert_eq!(bytes.as_ptr() as usize, self.pointer);
        Ok(ArrayBytes::Fixed(bytes.into()))
    }
}

fn options() -> CodecOptions {
    // Default options query Rayon. Keep that query on a single-thread local
    // pool so Miri does not start persistent global worker threads.
    thread_local! {
        static POOL: rayon::ThreadPool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .use_current_thread()
            .build()
            .unwrap();
    }
    POOL.with(|pool| pool.install(|| CodecOptions::default().with_concurrent_target(7)))
}

#[test]
fn immediate_retains_borrowed_and_unique_owned_bytes() {
    let borrowed = [1, 2, 3];
    let result = ArrayBytesDecodeIntoInput::from(CowBytes::Borrowed(&borrowed))
        .into_bytes(&options())
        .unwrap();
    assert!(matches!(result, CowBytes::Borrowed(_)));
    assert_eq!(result.as_ptr(), borrowed.as_ptr());
    let owned = vec![1, 2, 3];
    let pointer = owned.as_ptr();
    let result = ArrayBytesDecodeIntoInput::from(CowBytes::from(owned))
        .into_bytes(&options())
        .unwrap()
        .into_vec();
    assert_eq!(result.as_ptr(), pointer);
}

#[test]
fn deferred_routes_forward_representation_options_and_hint() {
    let producer = Producer::default();
    let source = BytesDecodeSource::new(
        &producer,
        vec![1, 2, 3].into(),
        BytesRepresentation::FixedSize(3),
    );
    assert_eq!(
        *source.decoded_representation(),
        BytesRepresentation::FixedSize(3)
    );
    assert!(source.is_decode_into_efficient());
    assert_eq!(
        &*ArrayBytesDecodeIntoInput::Deferred(source)
            .into_bytes(&options())
            .unwrap(),
        &[1, 2, 3]
    );
    assert_eq!(producer.normal.load(Ordering::Relaxed), 1);
    assert_eq!(producer.direct.load(Ordering::Relaxed), 0);
    let source = BytesDecodeSource::new(
        &producer,
        vec![4, 5, 6].into(),
        BytesRepresentation::FixedSize(3),
    );
    let mut output = [0; 3];
    source.decode_into(&mut output, &options()).unwrap();
    assert_eq!(output, [4, 5, 6]);
    assert_eq!(producer.normal.load(Ordering::Relaxed), 1);
    assert_eq!(producer.direct.load(Ordering::Relaxed), 1);
}

#[test]
fn default_receiver_preserves_unique_allocation_and_checks_target_length() {
    for deferred in [false, true] {
        for output_len in [3, 2] {
            let bytes = vec![1, 2, 3];
            let receiver = Receiver {
                pointer: bytes.as_ptr() as usize,
                calls: AtomicUsize::new(0),
            };
            let producer = Producer::default();
            let input = if deferred {
                ArrayBytesDecodeIntoInput::Deferred(BytesDecodeSource::new(
                    &producer,
                    bytes.into(),
                    BytesRepresentation::FixedSize(3),
                ))
            } else {
                CowBytes::from(bytes).into()
            };
            let mut output = vec![0; output_len];
            let shape = [output_len as u64];
            // This is the only view of output, and its subset covers the entire buffer.
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut output),
                    1,
                    &shape,
                    ArraySubset::new_with_shape(shape.to_vec()),
                )
            }
            .unwrap();
            let result = receiver.decode_into(
                input,
                &[NonZeroU64::new(3).unwrap()],
                (&mut view).into(),
                &options(),
            );
            assert_eq!(result.is_ok(), output_len == 3);
            assert_eq!(receiver.calls.load(Ordering::Relaxed), 1);
            assert_eq!(
                producer.normal.load(Ordering::Relaxed),
                usize::from(deferred)
            );
            assert_eq!(producer.direct.load(Ordering::Relaxed), 0);
        }
    }
}

#[test]
fn producer_errors_and_output_length_errors_do_not_retry() {
    for fail in [false, true] {
        let producer = Producer {
            fail,
            ..Default::default()
        };
        let source = BytesDecodeSource::new(
            &producer,
            vec![1, 2, 3].into(),
            BytesRepresentation::FixedSize(3),
        );
        assert!(source.decode_into(&mut [0; 2], &options()).is_err());
        assert_eq!(producer.direct.load(Ordering::Relaxed), 1);
        assert_eq!(producer.normal.load(Ordering::Relaxed), 0);
    }
    let producer = Producer {
        fail: true,
        ..Default::default()
    };
    let source = BytesDecodeSource::new(
        &producer,
        vec![1, 2, 3].into(),
        BytesRepresentation::FixedSize(3),
    );
    assert!(
        ArrayBytesDecodeIntoInput::Deferred(source)
            .into_bytes(&options())
            .is_err()
    );
    assert_eq!(producer.normal.load(Ordering::Relaxed), 1);
    assert_eq!(producer.direct.load(Ordering::Relaxed), 0);
}
