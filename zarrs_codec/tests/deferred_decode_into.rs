//! Ownership and single-use route contracts for deferred decoding.
//!
//! Run under Miri with `MIRIFLAGS=-Zmiri-ignore-leaks`: Rayon's
//! `use_current_thread` pools retain their thread-local registry allocations.

use std::borrow::Cow;
use std::mem::MaybeUninit;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
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
    fn partial_decoder_capability(&self) -> zarrs_codec::PartialDecoderCapability {
        zarrs_codec::PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }
    fn partial_encoder_capability(&self) -> zarrs_codec::PartialEncoderCapability {
        zarrs_codec::PartialEncoderCapability {
            partial_encode: false,
        }
    }

    fn configuration(
        &self,
        _: ZarrVersion,
        _: &CodecMetadataOptions,
    ) -> Option<zarrs_metadata::Configuration> {
        None
    }
}

impl BytesToBytesCodecTraits for Producer {
    fn recommended_concurrency(
        &self,
        _: &zarrs_codec::BytesRepresentation,
    ) -> Result<zarrs_codec::RecommendedConcurrency, zarrs_codec::CodecError> {
        Ok(zarrs_codec::RecommendedConcurrency::new_maximum(1))
    }
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self
    }
    fn encoded_representation(&self, repr: &BytesRepresentation) -> BytesRepresentation {
        *repr
    }
    fn encode<'a>(
        &self,
        bytes: CowBytes<'a>,
        _: &CodecOptions,
        _: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        Ok(bytes)
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        repr: &BytesRepresentation,
        _options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.normal.fetch_add(1, Ordering::Relaxed);
        assert_eq!(*repr, BytesRepresentation::FixedSize(3));
        assert_eq!(resources.compute_target(), 7);
        if self.fail {
            Err(CodecError::Other("producer failed".into()))
        } else {
            Ok(bytes)
        }
    }
    fn as_decode_into(&self) -> Option<&dyn BytesToBytesCodecDecodeInto> {
        Some(self)
    }
}

// SAFETY: `copy_decoded_bytes_into` initialises all of `output` on success.
unsafe impl BytesToBytesCodecDecodeInto for Producer {
    fn decode_into(
        &self,
        bytes: CowBytes<'_>,
        repr: &BytesRepresentation,
        output: &mut [MaybeUninit<u8>],
        _options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        self.direct.fetch_add(1, Ordering::Relaxed);
        assert_eq!(*repr, BytesRepresentation::FixedSize(3));
        assert_eq!(resources.compute_target(), 7);
        if self.fail {
            return Err(CodecError::Other("producer failed".into()));
        }
        Ok(copy_decoded_bytes_into(&bytes, output)?)
    }
}

#[derive(Debug)]
struct Receiver {
    pointer: usize,
    calls: AtomicUsize,
}
impl ArrayCodecTraits for Receiver {
    fn recommended_concurrency(
        &self,
        _: &[std::num::NonZeroU64],
    ) -> Result<zarrs_codec::RecommendedConcurrency, zarrs_codec::CodecError> {
        Ok(zarrs_codec::RecommendedConcurrency::new_maximum(1))
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn data_type(&self) -> &DataType {
        panic!("unused")
    }
    fn fill_value(&self) -> Option<&FillValue> {
        panic!("unused")
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
        _: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        unreachable!()
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(shape, &[NonZeroU64::new(3).unwrap()]);
        assert_eq!(resources.compute_target(), 7);
        // Reclaiming a unique allocation must not allocate or copy.
        let bytes = bytes.into_vec();
        assert_eq!(bytes.as_ptr() as usize, self.pointer);
        Ok(ArrayBytes::Fixed(bytes.into()))
    }
}

fn options() -> CodecOptions {
    CodecOptions::default()
}

fn resources() -> ComputeResources {
    // Default resources query Rayon. Keep that query on a single-thread local
    // pool so Miri does not start persistent global worker threads.
    thread_local! {
        static POOL: rayon::ThreadPool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .use_current_thread()
            .build()
            .unwrap();
    }
    POOL.with(|pool| pool.install(|| ComputeResources::default().with_compute_target(7)))
}

#[test]
fn immediate_retains_borrowed_and_unique_owned_bytes() {
    let borrowed = [1, 2, 3];
    let result = ArrayBytesDecodeIntoInput::from(CowBytes::Borrowed(&borrowed))
        .into_bytes(&options(), &resources())
        .unwrap();
    assert!(matches!(result, CowBytes::Borrowed(_)));
    assert_eq!(result.as_ptr(), borrowed.as_ptr());
    let owned = vec![1, 2, 3];
    let pointer = owned.as_ptr();
    let result = ArrayBytesDecodeIntoInput::from(CowBytes::from(owned))
        .into_bytes(&options(), &resources())
        .unwrap()
        .into_vec();
    assert_eq!(result.as_ptr(), pointer);
}

#[test]
fn deferred_routes_forward_representation_and_options() {
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
    assert_eq!(
        &*ArrayBytesDecodeIntoInput::Deferred(source)
            .into_bytes(&options(), &resources())
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
    let mut output = [MaybeUninit::uninit(); 3];
    source
        .decode_into(&mut output, &options(), &resources())
        .unwrap();
    // SAFETY: a successful `decode_into` initialises the output.
    assert_eq!(output.map(|byte| unsafe { byte.assume_init() }), [4, 5, 6]);
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
            let view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut output),
                    1,
                    &shape,
                    ArraySubset::new_with_shape(shape),
                )
            }
            .unwrap();
            let result = receiver.decode_into(
                input,
                &[NonZeroU64::new(3).unwrap()],
                &mut view.into(),
                &options(),
                &resources(),
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
        assert!(
            source
                .decode_into(&mut [MaybeUninit::uninit(); 2], &options(), &resources())
                .is_err()
        );
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
            .into_bytes(&options(), &resources())
            .is_err()
    );
    assert_eq!(producer.normal.load(Ordering::Relaxed), 1);
    assert_eq!(producer.direct.load(Ordering::Relaxed), 0);
}
