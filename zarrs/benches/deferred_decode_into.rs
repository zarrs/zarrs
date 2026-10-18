//! Compare deferred placement with an otherwise-identical owned-intermediate route.
//!
//! Run with `cargo bench -p zarrs --bench deferred_decode_into`. Optional compressor
//! features add cases; `--no-default-features` still exercises bytes and fallback.
//! Binding, encoding, target allocation and per-iteration view construction are
//! outside timing. Both routes include the same view-metadata destruction cost.
//! The reference suppresses the innermost producer's efficiency hint, preserving
//! owned intermediates and the receiver's existing fused copy-and-transform path.
#![allow(missing_docs)]

use std::borrow::Cow;
use std::hint::black_box;
use std::num::NonZeroU64;
use std::sync::Arc;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::{BytesCodec, CodecChain};
use zarrs::array::{ArraySubset, Endianness, data_type};
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoInput, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView,
    ArrayCodecTraits, ArrayToBytesCodecNoSubchunkingTraits, ArrayToBytesCodecTraits,
    BytesRepresentation, BytesToBytesCodecTraits, CodecCreateError, CodecError,
    CodecMetadataOptions, CodecOptions, CodecSpecificOptions, CodecTraits, ComputeResources,
    CowBytes, UnboundArrayToBytesCodecTraits,
};
use zarrs_data_type::{DataType, FillValue};
use zarrs_metadata::Configuration;
use zarrs_plugin::{ExtensionName, ZarrVersion};

/// A producer with ordinary decode only: intentionally no direct-output override
/// and no efficiency hint. Decoding always returns a uniquely owned allocation,
/// which the non-native bytes receiver can transform without another allocation.
#[derive(Debug)]
struct DecodeOnly;

impl ExtensionName for DecodeOnly {
    fn name(&self, _: ZarrVersion) -> Option<Cow<'static, str>> {
        None
    }
}

impl CodecTraits for DecodeOnly {
    fn configuration(&self, _: ZarrVersion, _: &CodecMetadataOptions) -> Option<Configuration> {
        None
    }
}

impl BytesToBytesCodecTraits for DecodeOnly {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self
    }

    fn encoded_representation(&self, decoded: &BytesRepresentation) -> BytesRepresentation {
        *decoded
    }

    fn encode<'a>(
        &self,
        decoded: CowBytes<'a>,
        _: &CodecOptions,
        _: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        Ok(decoded)
    }

    fn decode<'a>(
        &self,
        encoded: CowBytes<'a>,
        _: &BytesRepresentation,
        _: &CodecOptions,
        _: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        Ok(encoded.to_vec().into())
    }
}

/// Suppress direct output without changing the producer's ordinary decoding.
/// Only efficient producers need this wrapper; fallback controls stay identical.
#[derive(Debug)]
struct OwnedIntermediate(Arc<dyn BytesToBytesCodecTraits>);

impl ExtensionName for OwnedIntermediate {
    fn name(&self, _: ZarrVersion) -> Option<Cow<'static, str>> {
        None
    }
}

impl CodecTraits for OwnedIntermediate {
    fn configuration(
        &self,
        version: ZarrVersion,
        opts: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        self.0.configuration(version, opts)
    }
}

impl BytesToBytesCodecTraits for OwnedIntermediate {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self
    }
    fn encoded_representation(&self, repr: &BytesRepresentation) -> BytesRepresentation {
        self.0.encoded_representation(repr)
    }
    fn encode<'a>(
        &self,
        bytes: CowBytes<'a>,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.0.encode(bytes, options, resources)
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        repr: &BytesRepresentation,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.0.decode(bytes, repr, options, resources)
    }
    // Default false hint deliberately forces an owned intermediate.
}

/// Use the trait's default `decode_into`, not `BytesCodec`'s specialised receiver.
/// Its ordinary decode delegates to `BytesCodec`, preserving the producer's owned
/// buffer for native passthrough or an in-place non-native endian conversion.
#[derive(Debug)]
struct OrdinaryBytes(BytesCodec);

impl ExtensionName for OrdinaryBytes {
    fn name(&self, _: ZarrVersion) -> Option<Cow<'static, str>> {
        None
    }
}

impl CodecTraits for OrdinaryBytes {
    fn configuration(&self, _: ZarrVersion, _: &CodecMetadataOptions) -> Option<Configuration> {
        None
    }
}

impl UnboundArrayToBytesCodecTraits for OrdinaryBytes {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self
    }
    fn with_context(
        &self,
        dtype: DataType,
        fill: FillValue,
        opts: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        Ok(Arc::new(OrdinaryReceiver(
            self.0.with_context(dtype, fill, opts)?,
        )))
    }
}

#[derive(Debug)]
struct OrdinaryReceiver(Arc<dyn ArrayToBytesCodecTraits>);

impl ArrayCodecTraits for OrdinaryReceiver {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn data_type(&self) -> &DataType {
        self.0.data_type()
    }
    fn fill_value(&self) -> &FillValue {
        self.0.fill_value()
    }
}

impl ArrayToBytesCodecNoSubchunkingTraits for OrdinaryReceiver {}

impl ArrayToBytesCodecTraits for OrdinaryReceiver {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self
    }
    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        self.0.encoded_representation(shape)
    }
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        opts: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.0.encode(bytes, shape, opts, resources)
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        opts: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.0.decode(bytes, shape, opts, resources)
    }
    // Deliberately inherit default decode_into: the deferred input must become
    // owned bytes, not a borrowed temporary, before calling ordinary decode.
}

/// Each entry covers a branch or interaction; sizes and widths are not crossed.
#[derive(Clone, Copy)]
struct Scenario {
    producer: &'static str,
    element_size: usize,
    num_bytes: usize,
    non_native: bool,
    padded: bool,
    compare_owned: bool,
}

fn scenarios() -> Vec<Scenario> {
    let base = Scenario {
        producer: "bytes",
        element_size: 2,
        num_bytes: 1024 * 1024,
        non_native: false,
        padded: false,
        compare_owned: false,
    };
    let mut cases = Vec::new();
    // Immediate input: copy and byte swapping into contiguous and strided views.
    for (non_native, padded) in [(false, false), (false, true), (true, false), (true, true)] {
        cases.push(Scenario {
            non_native,
            padded,
            ..base
        });
    }
    // The trait's generic receiver, and the bytes receiver's owned-input fallback.
    cases.push(Scenario {
        producer: "decode_only_owned_fallback",
        non_native: true,
        ..base
    });
    cases.push(Scenario {
        producer: "decode_only_bytes_fallback",
        non_native: true,
        padded: true,
        ..base
    });
    #[cfg(feature = "zstd")]
    {
        let zstd = Scenario {
            producer: "zstd",
            compare_owned: true,
            ..base
        };
        // Direct placement, placement followed by swapping, and strided fallback.
        cases.push(zstd);
        cases.push(Scenario {
            non_native: true,
            ..zstd
        });
        cases.push(Scenario {
            non_native: true,
            padded: true,
            compare_owned: false,
            ..zstd
        });
        // One different swap width and one small-operation comparison.
        cases.push(Scenario {
            element_size: 4,
            non_native: true,
            compare_owned: false,
            ..zstd
        });
        cases.push(Scenario {
            num_bytes: 1024,
            ..zstd
        });
    }
    #[cfg(feature = "gzip")]
    cases.push(Scenario {
        producer: "gzip",
        ..base
    });
    #[cfg(feature = "blosc")]
    cases.push(Scenario {
        producer: "blosc",
        ..base
    });
    cases
}

fn producer(scenario: Scenario) -> Vec<Arc<dyn BytesToBytesCodecTraits>> {
    match scenario.producer {
        "bytes" => vec![],
        "decode_only_owned_fallback" | "decode_only_bytes_fallback" => vec![Arc::new(DecodeOnly)],
        #[cfg(feature = "zstd")]
        "zstd" => vec![Arc::new(zarrs::array::codec::ZstdCodec::new(3, false))],
        #[cfg(feature = "gzip")]
        "gzip" => vec![Arc::new(zarrs::array::codec::GzipCodec::new(6).unwrap())],
        #[cfg(feature = "blosc")]
        "blosc" => {
            use zarrs::array::codec::BloscCodec;
            use zarrs::array::codec::bytes_to_bytes::blosc::{BloscCompressor, BloscShuffleMode};
            vec![Arc::new(
                BloscCodec::new(
                    BloscCompressor::BloscLZ,
                    5.try_into().unwrap(),
                    None,
                    BloscShuffleMode::Shuffle,
                    Some(scenario.element_size),
                )
                .unwrap(),
            )]
        }
        _ => unreachable!("unknown scenario producer"),
    }
}

fn deferred_decode_into(c: &mut Criterion) {
    let options = CodecOptions::default();
    let mut group = c.benchmark_group("deferred_decode_into");
    for scenario in scenarios() {
        let Scenario {
            element_size,
            num_bytes,
            non_native,
            padded,
            ..
        } = scenario;
        let rows = 16u64;
        let columns = (num_bytes / element_size) as u64 / rows;
        let shape = [
            NonZeroU64::new(rows).unwrap(),
            NonZeroU64::new(columns).unwrap(),
        ];
        let target_shape = [rows, columns + u64::from(padded)];
        let endian_name = if non_native { "non_native" } else { "native" };
        let case = format!(
            "u{}/{endian_name}/{}/{}/{num_bytes}",
            element_size * 8,
            scenario.producer,
            if padded {
                "noncontiguous"
            } else {
                "contiguous"
            },
        );
        group.throughput(Throughput::Bytes(num_bytes as u64));
        for reference in [false, true]
            .into_iter()
            .take(1 + usize::from(scenario.compare_owned))
        {
            let name = if reference {
                "owned_intermediate"
            } else {
                "deferred"
            };
            // Criterion invokes the callback only for selected cases. Keep its fixture
            // across sampling callbacks without allocating or encoding while listing.
            let mut fixture = None;
            group.bench_function(BenchmarkId::new(name, &case), |b| {
                let (chain, encoded, _, output) = fixture.get_or_insert_with(|| {
                    let native = Endianness::native();
                    let endian = if non_native {
                        if native == Endianness::Little {
                            Endianness::Big
                        } else {
                            Endianness::Little
                        }
                    } else {
                        native
                    };
                    let (dtype, fill) = if element_size == 2 {
                        (data_type::uint16(), FillValue::from(0u16))
                    } else {
                        (data_type::uint32(), FillValue::from(0u32))
                    };
                    let decoded: Vec<u8> = (0..num_bytes / element_size)
                        .flat_map(|i| {
                            let i = u32::try_from(i).unwrap();
                            let value = i.wrapping_mul(37) ^ ((i >> 5) & 255);
                            if element_size == 2 {
                                u16::try_from(value & 0xffff)
                                    .unwrap()
                                    .to_ne_bytes()
                                    .to_vec()
                            } else {
                                value.to_ne_bytes().to_vec()
                            }
                        })
                        .collect();
                    let receiver: Arc<dyn UnboundArrayToBytesCodecTraits> =
                        if scenario.producer == "decode_only_owned_fallback" {
                            Arc::new(OrdinaryBytes(BytesCodec::new(Some(endian))))
                        } else {
                            Arc::new(BytesCodec::new(Some(endian)))
                        };
                    let mut producers = producer(scenario);
                    if reference {
                        assert!(producers[0].is_decode_into_efficient());
                        producers[0] = Arc::new(OwnedIntermediate(producers[0].clone()));
                    }
                    let chain = CodecChain::new(vec![], receiver, producers)
                        .with_context(dtype, fill, &CodecSpecificOptions::default())
                        .unwrap();
                    let encoded = chain
                        .encode(
                            ArrayBytes::new_flen(CowBytes::Borrowed(&decoded)),
                            &shape,
                            &options,
                            &ComputeResources::default(),
                        )
                        .unwrap()
                        .into_static();
                    let mut output =
                        vec![0xa5; target_shape.iter().product::<u64>() as usize * element_size];
                    {
                        // SAFETY: this is the only view, and output is not accessed
                        // independently while it is alive.
                        let mut view = unsafe {
                            ArrayBytesFixedDisjointView::new(
                                UnsafeCellSlice::new(&mut output),
                                element_size,
                                &target_shape,
                                ArraySubset::new_with_ranges(&[0..rows, 0..columns]),
                            )
                            .unwrap()
                        };
                        chain
                            .decode_into(
                                ArrayBytesDecodeIntoInput::from(encoded.clone()),
                                &shape,
                                ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                                &options,
                                &ComputeResources::default(),
                            )
                            .unwrap();
                    }
                    assert_target(
                        &output,
                        &decoded,
                        rows as usize,
                        columns as usize,
                        element_size,
                        padded,
                    );
                    (chain, encoded, decoded, output)
                });
                let cells = UnsafeCellSlice::new(output);
                b.iter_batched(
                    || unsafe {
                        // SAFETY: PerIteration drops the previous view before the
                        // next one is constructed; overlapping views never coexist.
                        ArrayBytesFixedDisjointView::new(
                            cells,
                            element_size,
                            &target_shape,
                            ArraySubset::new_with_ranges(&[0..rows, 0..columns]),
                        )
                        .unwrap()
                    },
                    |view| {
                        let mut view = view;
                        black_box(&view);
                        chain
                            .decode_into(
                                ArrayBytesDecodeIntoInput::from(CowBytes::Borrowed(black_box(
                                    encoded,
                                ))),
                                &shape,
                                ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                                &options,
                                &ComputeResources::default(),
                            )
                            .unwrap();
                    },
                    BatchSize::PerIteration,
                );
            });
            if let Some((_, _, decoded, output)) = fixture {
                assert_target(
                    &output,
                    &decoded,
                    rows as usize,
                    columns as usize,
                    element_size,
                    padded,
                );
            }
        }
    }
    group.finish();
}

fn assert_target(
    output: &[u8],
    decoded: &[u8],
    rows: usize,
    columns: usize,
    size: usize,
    padded: bool,
) {
    let row_bytes = columns * size;
    let stride = row_bytes + usize::from(padded) * size;
    for row in 0..rows {
        assert_eq!(
            &output[row * stride..row * stride + row_bytes],
            &decoded[row * row_bytes..(row + 1) * row_bytes]
        );
        if padded {
            assert!(
                output[row * stride + row_bytes..(row + 1) * stride]
                    .iter()
                    .all(|&byte| byte == 0xa5)
            );
        }
    }
}

criterion_group!(benches, deferred_decode_into);
criterion_main!(benches);
