//! Regression tests for batching generic indexer reads by subchunk.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::*;
use crate::array::codec::default_array_to_bytes_codec;
use crate::array::{
    ArrayBytesFixedDisjointView, ArraySubset, BytesRepresentation, Element, FillValue, data_type,
};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_codec::plan::{
    ByteSource, DecodeRequest, PlanIndexer, TargetViews, Task, TaskTarget, drive_memory_one,
    partial_decode_memory,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayToBytesCodecNoSubchunkingTraits,
    CodecCreateError, CodecMetadataOptions, CodecSpecificOptions, CodecTraits,
    UnboundArrayToBytesCodecTraits,
};
use zarrs_metadata::Configuration;
use zarrs_plugin::ZarrVersion;

#[derive(Debug, Default)]
struct Counts {
    decoders: AtomicUsize,
    batch_sizes: Mutex<Vec<u64>>,
}

impl Counts {
    fn reset(&self) {
        self.decoders.store(0, Ordering::SeqCst);
        self.batch_sizes.lock().unwrap().clear();
    }

    fn check(&self, expected: &[u64]) {
        let mut sizes = self.batch_sizes.lock().unwrap().clone();
        sizes.sort_unstable();
        assert_eq!(sizes, expected);
        assert_eq!(self.decoders.load(Ordering::SeqCst), expected.len());
    }
}

#[derive(Debug)]
struct CountingCodec<T: ?Sized> {
    inner: Arc<T>,
    counts: Arc<Counts>,
}

type UnboundCountingCodec = CountingCodec<dyn UnboundArrayToBytesCodecTraits>;
zarrs_plugin::impl_extension_aliases!(UnboundCountingCodec, v3: "test.counting");

impl CodecTraits for CountingCodec<dyn UnboundArrayToBytesCodecTraits> {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        None
    }
}

impl UnboundArrayToBytesCodecTraits for CountingCodec<dyn UnboundArrayToBytesCodecTraits> {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        Ok(Arc::new(CountingCodec {
            inner: self
                .inner
                .with_context(data_type, fill_value, codec_specific_options)?,
            counts: self.counts.clone(),
        }))
    }
}

impl ArrayCodecTraits for CountingCodec<dyn ArrayToBytesCodecTraits> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn data_type(&self) -> &DataType {
        self.inner.data_type()
    }
    fn fill_value(&self) -> &FillValue {
        self.inner.fill_value()
    }
}

impl ArrayToBytesCodecNoSubchunkingTraits for CountingCodec<dyn ArrayToBytesCodecTraits> {}

impl ArrayToBytesCodecTraits for CountingCodec<dyn ArrayToBytesCodecTraits> {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self
    }
    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        self.inner.encoded_representation(shape)
    }
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.inner.encode(bytes, shape, options, resources)
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.inner.decode(bytes, shape, options, resources)
    }
    fn plan_decode<'a>(
        self: Arc<Self>,
        source: ByteSource<'a>,
        shape: &[NonZeroU64],
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        // Count each request for the elements of a subchunk
        self.counts.decoders.fetch_add(1, Ordering::SeqCst);
        self.counts
            .batch_sizes
            .lock()
            .unwrap()
            .push(request.indexer.as_indexer().len());
        self.inner
            .clone()
            .plan_decode(source, shape, request, options)
    }
}

struct Fixture {
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    bytes: ArrayBytes<'static>,
    counts: Arc<Counts>,
}

fn fixture(variable: bool, nested: bool, index_location: ShardingIndexLocation) -> Fixture {
    let (data_type, fill_value, bytes) = if variable {
        let data_type = data_type::string();
        let values = [
            "missing", "missing", "", "ddd", "missing", "missing", "g", "hhhh", "ii", "j", "kkk",
            "l", "mm", "n", "ooo", "pppp",
        ];
        let bytes =
            String::into_array_bytes(&data_type, values.map(String::from).to_vec()).unwrap();
        (data_type, FillValue::from("missing"), bytes)
    } else {
        (
            data_type::uint16(),
            FillValue::from(999u16),
            crate::array::transmute_to_bytes_vec(vec![
                999u16, 999, 2, 3, 999, 999, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            ])
            .into(),
        )
    };
    let counts = Arc::new(Counts::default());
    let mut inner: Arc<dyn UnboundArrayToBytesCodecTraits> = Arc::new(CountingCodec {
        inner: default_array_to_bytes_codec(&data_type),
        counts: counts.clone(),
    });
    if nested {
        inner = ShardingCodecBuilder::new(vec![NonZeroU64::new(1).unwrap(); 2], &data_type)
            .array_to_bytes_codec(inner)
            .build_arc();
    }
    let codec = ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type)
        .array_to_bytes_codec(inner)
        .index_location(index_location)
        .build_arc()
        .with_context(data_type, fill_value, &CodecSpecificOptions::default())
        .unwrap();
    Fixture {
        codec,
        bytes,
        counts,
    }
}

fn selections(nested: bool) -> Vec<(Vec<Vec<u64>>, Vec<u64>)> {
    vec![
        (
            vec![
                vec![3, 3],
                vec![0, 1],
                vec![2, 2],
                vec![0, 2],
                vec![3, 2],
                vec![0, 2],
                vec![1, 3],
                vec![0, 0],
                vec![2, 0],
            ],
            if nested {
                vec![1, 1, 1, 1, 1, 2]
            } else {
                vec![1, 3, 3]
            },
        ),
        (
            vec![vec![0, 2], vec![1, 3], vec![0, 2]],
            if nested { vec![1, 2] } else { vec![3] },
        ),
        (vec![vec![0, 0], vec![1, 1], vec![0, 0]], vec![]),
        (vec![], vec![]),
    ]
}

/// Every (variable, nested, index location, concurrency, absent) combination.
fn cases() -> impl Iterator<Item = (bool, bool, ShardingIndexLocation, usize, bool)> {
    itertools::iproduct!(
        [false, true],
        [false, true],
        [ShardingIndexLocation::Start, ShardingIndexLocation::End],
        [1, 2],
        [false, true]
    )
}

fn expected(fixture: &Fixture, indices: &[Vec<u64>], absent: bool) -> ArrayBytes<'static> {
    if absent {
        ArrayBytes::new_fill_value(
            fixture.codec.data_type(),
            indices.len() as u64,
            fixture.codec.fill_value(),
        )
        .unwrap()
    } else {
        fixture
            .bytes
            .extract_array_subset(&indices, &[4, 4], fixture.codec.data_type())
            .unwrap()
            .into_owned()
    }
}

/// An output buffer of 2-byte elements, its shape, and the subset of `len` elements to decode into.
///
/// If `strided`, the subset is the first column of a `[len, 2]` buffer prefilled with `u8::MAX`.
fn output_case(len: usize, strided: bool) -> (Vec<u8>, Vec<u64>, ArraySubset) {
    let len_u64 = len as u64;
    if strided {
        (
            vec![u8::MAX; len * 4],
            vec![len_u64, 2],
            ArraySubset::new_with_ranges(&[0..len_u64, 0..1]),
        )
    } else {
        (
            vec![0; len * 2],
            vec![len_u64],
            ArraySubset::new_with_shape(vec![len_u64]),
        )
    }
}

/// The expected contents of an [`output_case`] buffer after decoding `expected` into it.
fn expected_output(expected: &ArrayBytes, strided: bool) -> Vec<u8> {
    let expected = expected.clone().into_fixed().unwrap();
    if strided {
        expected
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|element| [element[0], element[1], u8::MAX, u8::MAX])
            .collect()
    } else {
        expected.to_vec()
    }
}

/// A view of `subset` of `output`, which holds 2-byte elements.
pub(super) fn fixed_view<'a>(
    output: &'a mut [u8],
    output_shape: &'a [u64],
    subset: ArraySubset,
) -> ArrayBytesFixedDisjointView<'a> {
    // SAFETY: this is the only view into output.
    unsafe {
        ArrayBytesFixedDisjointView::new(UnsafeCellSlice::new(output), 2, output_shape, subset)
            .unwrap()
    }
}

/// Decode `indices` of the shard `encoded` (or a missing shard) of `shape` into `target` with a planned decode.
fn decode_into(
    codec: &Arc<dyn ArrayToBytesCodecTraits>,
    encoded: Option<&[u8]>,
    shape: &[NonZeroU64],
    indices: &[Vec<u64>],
    target: &ArrayBytesDecodeIntoTarget<'_>,
) -> Result<(), CodecError> {
    let source = encoded.map_or(
        ByteSource::external(zarrs_codec::plan::SourceKey(0)),
        |encoded| ByteSource::memory(encoded.into()),
    );
    let indices = indices.to_vec();
    let task = codec.clone().plan_decode(
        source,
        shape,
        DecodeRequest {
            indexer: PlanIndexer::Borrowed(&indices),
            target: TaskTarget::Views(unsafe { TargetViews::from_target(target) }?),
        },
        &CodecOptions::default(),
    )?;
    drive_memory_one(task, &ComputeResources::default()).map(|_| ())
}

/// Requests for the elements of generic indexers are batched by subchunk.
#[test]
fn sharding_indexer_batches() {
    let shape = vec![NonZeroU64::new(4).unwrap(); 2];
    for (variable, nested, location, concurrency, absent) in cases() {
        let fixture = fixture(variable, nested, location);
        let Fixture {
            codec,
            bytes,
            counts,
        } = &fixture;
        let options = CodecOptions::default();
        let resources = ComputeResources::default().with_compute_target(concurrency);
        let encoded = (!absent).then(|| {
            codec
                .encode(bytes.clone(), &shape, &options, &resources)
                .unwrap()
                .into_vec()
        });
        let decode = |indices: &Vec<Vec<u64>>| {
            partial_decode_memory(
                codec.clone(),
                encoded.as_deref().map(CowBytes::from),
                &shape,
                indices,
                &options,
                &resources,
            )
        };
        for (indices, sizes) in selections(nested) {
            counts.reset();
            let sizes = if absent { vec![] } else { sizes };
            let expected = expected(&fixture, &indices, absent);
            assert_eq!(decode(&indices).unwrap(), expected);
            counts.check(&sizes);
            // A missing shard is the fill value, which is not decoded
            if absent {
                continue;
            }
            for strided in [false, true].into_iter().filter(|_| !variable) {
                counts.reset();
                let (mut output, output_shape, subset) = output_case(indices.len(), strided);
                decode_into(
                    codec,
                    encoded.as_deref(),
                    &shape,
                    &indices,
                    &ArrayBytesDecodeIntoTarget::Fixed(&mut fixed_view(
                        &mut output,
                        &output_shape,
                        subset,
                    )),
                )
                .unwrap();
                assert_eq!(output, expected_output(&expected, strided));
                counts.check(&sizes);
            }
        }
        counts.reset();
        for indices in [vec![vec![0, 4]], vec![vec![0]]] {
            assert!(decode(&indices).is_err());
        }
        counts.check(&[]);
    }
}
