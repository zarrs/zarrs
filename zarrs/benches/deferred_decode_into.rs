//! Compare deferred placement with an otherwise-identical owned-intermediate route.
//!
//! Run with `cargo bench -p zarrs --bench deferred_decode_into`. Optional compressor
//! features add cases; `--no-default-features` still exercises bytes and fallback.
//! Binding, encoding, target allocation and per-iteration view construction are
//! outside timing. Both routes include the same view-metadata destruction cost.
//! The reference suppresses the innermost producer's efficiency hint, preserving
//! owned intermediates and the receiver's existing fused copy-and-transform path.
#![allow(missing_docs)]

use std::{borrow::Cow, hint::black_box, num::NonZeroU64, sync::Arc};

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::{
    ArraySubset, Endianness,
    codec::{BytesCodec, CodecChain},
    data_type,
};
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoInput, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView,
    ArrayCodecTraits, ArrayToBytesCodecNoSubchunkingTraits, ArrayToBytesCodecTraits,
    BytesRepresentation, BytesToBytesCodecTraits, CodecCreateError, CodecError,
    CodecMetadataOptions, CodecOptions, CodecSpecificOptions, CodecTraits, CowBytes,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    UnboundArrayToBytesCodecTraits,
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

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl BytesToBytesCodecTraits for DecodeOnly {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self
    }

    fn recommended_concurrency(
        &self,
        _: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encoded_representation(&self, decoded: &BytesRepresentation) -> BytesRepresentation {
        *decoded
    }

    fn encode<'a>(
        &self,
        decoded: CowBytes<'a>,
        _: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        Ok(decoded)
    }

    fn decode<'a>(
        &self,
        encoded: CowBytes<'a>,
        _: &BytesRepresentation,
        _: &CodecOptions,
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
    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        self.0.partial_decoder_capability()
    }
    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        self.0.partial_encoder_capability()
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl BytesToBytesCodecTraits for OwnedIntermediate {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self
    }
    fn recommended_concurrency(
        &self,
        repr: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        self.0.recommended_concurrency(repr)
    }
    fn encoded_representation(&self, repr: &BytesRepresentation) -> BytesRepresentation {
        self.0.encoded_representation(repr)
    }
    fn encode<'a>(
        &self,
        bytes: CowBytes<'a>,
        options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.0.encode(bytes, options)
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        repr: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.0.decode(bytes, repr, options)
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
    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        DecodeOnly.partial_decoder_capability()
    }
    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        DecodeOnly.partial_encoder_capability()
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
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
    fn recommended_concurrency(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        self.0.recommended_concurrency(shape)
    }
}

impl ArrayToBytesCodecNoSubchunkingTraits for OrdinaryReceiver {}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
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
    ) -> Result<CowBytes<'a>, CodecError> {
        self.0.encode(bytes, shape, opts)
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        opts: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.0.decode(bytes, shape, opts)
    }
    // Deliberately inherit default decode_into: the deferred input must become
    // owned bytes, not a borrowed temporary, before calling ordinary decode.
}

fn producers(element_size: usize) -> Vec<(&'static str, Vec<Arc<dyn BytesToBytesCodecTraits>>)> {
    // Keep this mutable even with all optional compressors disabled.
    #[allow(unused_mut)]
    let mut cases: Vec<(&str, Vec<Arc<dyn BytesToBytesCodecTraits>>)> = vec![
        ("bytes", vec![]),
        ("decode_only_owned_fallback", vec![Arc::new(DecodeOnly)]),
        ("decode_only_bytes_fallback", vec![Arc::new(DecodeOnly)]),
    ];
    #[cfg(feature = "zstd")]
    cases.push((
        "zstd",
        vec![Arc::new(zarrs::array::codec::ZstdCodec::new(3, false))],
    ));
    #[cfg(feature = "gzip")]
    cases.push((
        "gzip",
        vec![Arc::new(zarrs::array::codec::GzipCodec::new(6).unwrap())],
    ));
    #[cfg(feature = "blosc")]
    {
        use zarrs::array::codec::{
            BloscCodec,
            bytes_to_bytes::blosc::{BloscCompressor, BloscShuffleMode},
        };
        cases.push((
            "blosc",
            vec![Arc::new(
                BloscCodec::new(
                    BloscCompressor::BloscLZ,
                    5.try_into().unwrap(),
                    None,
                    BloscShuffleMode::Shuffle,
                    Some(element_size),
                )
                .unwrap(),
            )],
        ));
    }
    #[cfg(not(feature = "blosc"))]
    let _ = element_size;
    cases
}

// Keep the explicit matrix and each target's lifecycle together for review.
#[allow(clippy::too_many_lines)]
fn deferred_decode_into(c: &mut Criterion) {
    let options = CodecOptions::default();
    let native = if cfg!(target_endian = "little") {
        Endianness::Little
    } else {
        Endianness::Big
    };
    let opposite = if cfg!(target_endian = "little") {
        Endianness::Big
    } else {
        Endianness::Little
    };
    let mut group = c.benchmark_group("deferred_decode_into");
    for element_size in [2usize, 4] {
        let dtype = if element_size == 2 {
            data_type::uint16()
        } else {
            data_type::uint32()
        };
        let fill = if element_size == 2 {
            FillValue::from(0u16)
        } else {
            FillValue::from(0u32)
        };
        for num_bytes in [1024usize, 1024 * 1024] {
            // Sixteen rows ensure the padded destination has multiple disjoint runs.
            let rows = 16u64;
            let columns = (num_bytes / element_size) as u64 / rows;
            let shape = [
                NonZeroU64::new(rows).unwrap(),
                NonZeroU64::new(columns).unwrap(),
            ];
            // Deterministic, nonconstant data: mixes compressible structure and values
            // that expose endian mistakes. Identical input for both measured routes.
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
            group.throughput(Throughput::Bytes(num_bytes as u64));
            for (endian_name, endian) in [("native", native), ("non_native", opposite)] {
                for (producer_name, producer) in producers(element_size) {
                    let receiver: Arc<dyn UnboundArrayToBytesCodecTraits> =
                        if producer_name == "decode_only_owned_fallback" {
                            Arc::new(OrdinaryBytes(BytesCodec::new(Some(endian))))
                        } else {
                            Arc::new(BytesCodec::new(Some(endian)))
                        };
                    let chain = CodecChain::new(vec![], receiver.clone(), producer)
                        .with_context(
                            dtype.clone(),
                            fill.clone(),
                            &CodecSpecificOptions::default(),
                        )
                        .unwrap();
                    let ordinary_producers = chain
                        .bytes_to_bytes_codecs()
                        .iter()
                        .enumerate()
                        .map(|(index, producer)| {
                            if index == 0 && producer.is_decode_into_efficient() {
                                Arc::new(OwnedIntermediate(producer.clone()))
                                    as Arc<dyn BytesToBytesCodecTraits>
                            } else {
                                producer.clone()
                            }
                        })
                        .collect();
                    let ordinary = CodecChain::new(vec![], receiver, ordinary_producers)
                        .with_context(
                            dtype.clone(),
                            fill.clone(),
                            &CodecSpecificOptions::default(),
                        )
                        .unwrap();
                    let encoded = chain
                        .encode(
                            ArrayBytes::new_flen(CowBytes::Borrowed(&decoded)),
                            &shape,
                            &options,
                        )
                        .unwrap();
                    for padded in [false, true] {
                        let target_shape = [rows, columns + u64::from(padded)];
                        let subset = ArraySubset::new_with_ranges(&[0..rows, 0..columns]);
                        let mut output = vec![
                            0xa5;
                            usize::try_from(target_shape.iter().product::<u64>())
                                .unwrap()
                                * element_size
                        ];
                        let case = format!(
                            "u{}/{endian_name}/{producer_name}/{}/{num_bytes}",
                            element_size * 8,
                            if padded {
                                "noncontiguous"
                            } else {
                                "contiguous"
                            },
                        );
                        {
                            // SAFETY: one view exists, used serially, and output is not
                            // accessed independently while the view is alive.
                            let mut view = unsafe {
                                ArrayBytesFixedDisjointView::new(
                                    UnsafeCellSlice::new(&mut output),
                                    element_size,
                                    &target_shape,
                                    subset,
                                )
                                .unwrap()
                            };
                            // Validate candidate placement before timing, including strided output.
                            chain
                                .decode_into(
                                    ArrayBytesDecodeIntoInput::from(CowBytes::Borrowed(&encoded)),
                                    &shape,
                                    ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                                    &options,
                                )
                                .unwrap();
                        }
                        assert_target(
                            &output,
                            &decoded,
                            usize::try_from(rows).unwrap(),
                            usize::try_from(columns).unwrap(),
                            element_size,
                            padded,
                        );
                        // A target borrows its view for the view's data lifetime. Move a
                        // fresh view into each iteration so that this borrow is local.
                        // PerIteration ensures overlapping views never coexist.
                        {
                            let cells = UnsafeCellSlice::new(&mut output);
                            for reference in [false, true] {
                                let name = if reference {
                                    "owned_intermediate"
                                } else {
                                    "deferred"
                                };
                                group.bench_function(BenchmarkId::new(name, &case), |b| {
                                    b.iter_batched(
                                        || unsafe {
                                            // SAFETY: PerIteration drops the prior view
                                            // before constructing this one.
                                            ArrayBytesFixedDisjointView::new(
                                                cells,
                                                element_size,
                                                &target_shape,
                                                ArraySubset::new_with_ranges(&[
                                                    0..rows,
                                                    0..columns,
                                                ]),
                                            )
                                            .unwrap()
                                        },
                                        |view| {
                                            let mut view = view;
                                            black_box(&view);
                                            let selected =
                                                if reference { &ordinary } else { &chain };
                                            selected
                                                .decode_into(
                                                    ArrayBytesDecodeIntoInput::from(
                                                        CowBytes::Borrowed(black_box(&encoded)),
                                                    ),
                                                    &shape,
                                                    ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                                                    &options,
                                                )
                                                .unwrap();
                                        },
                                        BatchSize::PerIteration,
                                    );
                                });
                            }
                        }
                        assert_target(
                            &output,
                            &decoded,
                            usize::try_from(rows).unwrap(),
                            usize::try_from(columns).unwrap(),
                            element_size,
                            padded,
                        );
                    }
                }
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
