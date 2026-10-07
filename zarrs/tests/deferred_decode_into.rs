//! Regression tests for producer placement through codec-chain boundaries.
#![allow(missing_docs)]

use std::{
    borrow::Cow,
    num::NonZeroU64,
    sync::{Arc, Mutex},
};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::{
    ArraySubset, Endianness,
    codec::{BytesCodec, CodecChain},
    data_type,
};
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoInput, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView,
    ArrayToBytesCodecTraits, BytesRepresentation, BytesToBytesCodecTraits, CodecError,
    CodecMetadataOptions, CodecOptions, CodecSpecificOptions, CodecTraits, CowBytes,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
use zarrs_data_type::FillValue;
use zarrs_metadata::Configuration;
use zarrs_plugin::{ExtensionName, ZarrVersion};

#[derive(Debug, PartialEq, Eq)]
struct Call {
    id: usize,
    direct: bool,
    representation: BytesRepresentation,
    pointer: usize,
}

#[derive(Debug)]
struct Producer {
    id: usize,
    efficient: bool,
    trailer: bool,
    fail: bool,
    calls: Arc<Mutex<Vec<Call>>>,
}

impl ExtensionName for Producer {
    fn name(&self, _: ZarrVersion) -> Option<Cow<'static, str>> {
        None
    }
}

impl CodecTraits for Producer {
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

impl Producer {
    fn record(&self, direct: bool, representation: &BytesRepresentation, pointer: usize) {
        self.calls.lock().unwrap().push(Call {
            id: self.id,
            direct,
            representation: *representation,
            pointer,
        });
    }
    fn payload<'a>(&self, bytes: &'a [u8]) -> Result<&'a [u8], CodecError> {
        if self.fail {
            return Err(CodecError::Other("producer failure".into()));
        }
        if self.trailer {
            bytes
                .strip_suffix(&[u8::try_from(self.id).unwrap()])
                .ok_or_else(|| CodecError::Other("wrong trailer".into()))
        } else {
            Ok(bytes)
        }
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl BytesToBytesCodecTraits for Producer {
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
        match (*decoded, self.trailer) {
            (BytesRepresentation::FixedSize(n), true) => BytesRepresentation::FixedSize(n + 1),
            _ => *decoded,
        }
    }
    fn encode<'a>(
        &self,
        bytes: CowBytes<'a>,
        _: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        let mut bytes = bytes.into_vec();
        if self.trailer {
            bytes.push(u8::try_from(self.id).unwrap());
        }
        Ok(bytes.into())
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        repr: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        if !options.validate_checksums() {
            self.record(false, repr, 0);
            return Err(CodecError::Other(
                "producer rejected operation options".into(),
            ));
        }
        let result = self.payload(&bytes).map(<[u8]>::to_vec);
        self.record(
            false,
            repr,
            result.as_ref().map_or(0, |v| v.as_ptr() as usize),
        );
        result.map(Into::into)
    }
    fn decode_into(
        &self,
        bytes: CowBytes<'_>,
        repr: &BytesRepresentation,
        output: &mut [u8],
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.record(true, repr, output.as_ptr() as usize);
        if !options.validate_checksums() {
            return Err(CodecError::Other(
                "producer rejected operation options".into(),
            ));
        }
        let bytes = self.payload(&bytes)?;
        if bytes.len() != output.len() {
            return Err(CodecError::Other("wrong decoded length".into()));
        }
        output.copy_from_slice(bytes);
        Ok(())
    }
    fn is_decode_into_efficient(&self) -> bool {
        self.efficient
    }
}

fn producer(
    calls: &Arc<Mutex<Vec<Call>>>,
    id: usize,
    efficient: bool,
    trailer: bool,
    fail: bool,
) -> Arc<dyn BytesToBytesCodecTraits> {
    Arc::new(Producer {
        id,
        efficient,
        trailer,
        fail,
        calls: calls.clone(),
    })
}

fn bind(chain: &CodecChain) -> Arc<dyn ArrayToBytesCodecTraits> {
    chain
        .with_context(
            data_type::uint16(),
            FillValue::from(0u16),
            &CodecSpecificOptions::default(),
        )
        .unwrap()
}

fn endian(non_native: bool) -> Endianness {
    match (cfg!(target_endian = "little"), non_native) {
        (true, false) | (false, true) => Endianness::Little,
        _ => Endianness::Big,
    }
}

fn native_bytes() -> Vec<u8> {
    [0x1234u16, 0xabcd, 1, 0xff00]
        .into_iter()
        .flat_map(u16::to_ne_bytes)
        .collect()
}

fn run(chain: &dyn ArrayToBytesCodecTraits, padded: bool) -> (Vec<u8>, usize) {
    let shape = [NonZeroU64::new(2).unwrap(); 2];
    let options = CodecOptions::default();
    let encoded = chain
        .encode(ArrayBytes::new_flen(native_bytes()), &shape, &options)
        .unwrap();
    let mut output = vec![0xa5; if padded { 12 } else { 8 }];
    let pointer = output.as_ptr() as usize;
    let target_shape = [2, if padded { 3 } else { 2 }];
    {
        // SAFETY: this is the only view and no other access occurs until it is dropped.
        let mut view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut output),
                2,
                &target_shape,
                ArraySubset::new_with_ranges(&[0..2, 0..2]),
            )
            .unwrap()
        };
        chain
            .decode_into(
                ArrayBytesDecodeIntoInput::from(encoded),
                &shape,
                ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                &options,
            )
            .unwrap();
    }
    let expected = native_bytes();
    if padded {
        assert_eq!(&output[..4], &expected[..4]);
        assert_eq!(&output[6..10], &expected[4..]);
        assert_eq!(&output[4..6], &[0xa5; 2]);
        assert_eq!(&output[10..], &[0xa5; 2]);
    } else {
        assert_eq!(output, expected);
    }
    (output, pointer)
}

#[test]
fn efficient_producer_writes_final_pointer_before_endian_conversion() {
    for non_native in [false, true] {
        let calls = Arc::default();
        let chain = bind(&CodecChain::new(
            vec![],
            Arc::new(BytesCodec::new(Some(endian(non_native)))),
            vec![producer(&calls, 1, true, false, false)],
        ));
        let (_, pointer) = run(chain.as_ref(), false);
        assert_eq!(
            *calls.lock().unwrap(),
            [Call {
                id: 1,
                direct: true,
                representation: BytesRepresentation::FixedSize(8),
                pointer
            }]
        );
    }
}

#[test]
fn full_width_packbits_forwards_deferred_placement() {
    use zarrs::array::codec::PackBitsCodec;
    use zarrs::metadata_ext::codec::packbits::PackBitsPaddingEncoding;

    let calls = Arc::default();
    let codec = PackBitsCodec::new(PackBitsPaddingEncoding::None, None, None).unwrap();
    let chain = bind(&CodecChain::new(
        vec![],
        Arc::new(codec),
        vec![producer(&calls, 1, true, false, false)],
    ));
    let (_, pointer) = run(chain.as_ref(), false);
    assert_eq!(
        *calls.lock().unwrap(),
        [Call {
            id: 1,
            direct: true,
            representation: BytesRepresentation::FixedSize(8),
            pointer,
        }]
    );
}

#[test]
fn false_hint_does_not_use_available_direct_override() {
    for non_native in [false, true] {
        let calls = Arc::default();
        let chain = bind(&CodecChain::new(
            vec![],
            Arc::new(BytesCodec::new(Some(endian(non_native)))),
            vec![producer(&calls, 1, false, false, false)],
        ));
        let (_, pointer) = run(chain.as_ref(), false);
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert!(!calls[0].direct);
        assert_ne!(calls[0].pointer, pointer);
    }
}

#[test]
fn strided_target_resolves_producer_and_preserves_sentinels() {
    let calls = Arc::default();
    let chain = bind(&CodecChain::new(
        vec![],
        Arc::new(BytesCodec::new(Some(endian(true)))),
        vec![producer(&calls, 1, true, false, false)],
    ));
    let (_, pointer) = run(chain.as_ref(), true);
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert!(!calls[0].direct);
    assert_ne!(calls[0].pointer, pointer);
}

#[test]
fn multiple_producers_reverse_order_and_use_each_decoded_representation() {
    let calls = Arc::default();
    let chain = bind(&CodecChain::new(
        vec![],
        Arc::new(BytesCodec::new(Some(endian(true)))),
        vec![
            producer(&calls, 1, true, true, false),
            producer(&calls, 2, true, true, false),
            producer(&calls, 3, true, true, false),
        ],
    ));
    let (_, pointer) = run(chain.as_ref(), false);
    let calls = calls.lock().unwrap();
    assert_eq!(
        calls
            .iter()
            .map(|c| (c.id, c.direct, c.representation))
            .collect::<Vec<_>>(),
        [
            (3, false, BytesRepresentation::FixedSize(10)),
            (2, false, BytesRepresentation::FixedSize(9)),
            (1, true, BytesRepresentation::FixedSize(8)),
        ]
    );
    assert_eq!(calls[2].pointer, pointer);
}

#[test]
fn transparent_nested_chain_forwards_deferred_placement() {
    let calls = Arc::default();
    let inner = CodecChain::new(
        vec![],
        Arc::new(BytesCodec::new(Some(endian(true)))),
        vec![],
    );
    let chain = bind(&CodecChain::new(
        vec![],
        Arc::new(inner),
        vec![producer(&calls, 1, true, false, false)],
    ));
    let (_, pointer) = run(chain.as_ref(), false);
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert!(calls[0].direct);
    assert_eq!(calls[0].pointer, pointer);
}

#[test]
fn nested_chain_with_internal_producer_resolves_incoming_source_once() {
    let calls = Arc::default();
    let inner = CodecChain::new(
        vec![],
        Arc::new(BytesCodec::new(Some(endian(true)))),
        vec![producer(&calls, 1, true, true, false)],
    );
    let chain = bind(&CodecChain::new(
        vec![],
        Arc::new(inner),
        vec![producer(&calls, 2, true, true, false)],
    ));
    let (_, pointer) = run(chain.as_ref(), false);
    let calls = calls.lock().unwrap();
    assert_eq!(
        calls.iter().map(|c| (c.id, c.direct)).collect::<Vec<_>>(),
        [(2, false), (1, true)]
    );
    assert_eq!(calls[1].pointer, pointer);
}

#[test]
fn producer_error_and_malformed_length_are_not_retried() {
    for efficient in [false, true] {
        for fail in [false, true] {
            let calls = Arc::default();
            let chain = bind(&CodecChain::new(
                vec![],
                Arc::new(BytesCodec::new(Some(endian(false)))),
                vec![producer(&calls, 1, efficient, false, fail)],
            ));
            let mut output = [0xa5; 8];
            let shape = [NonZeroU64::new(4).unwrap()];
            {
                // SAFETY: a single view is used serially, with no independent output access.
                let mut view = unsafe {
                    ArrayBytesFixedDisjointView::new(
                        UnsafeCellSlice::new(&mut output),
                        2,
                        &[4],
                        ArraySubset::new_with_ranges(&[0..4]),
                    )
                    .unwrap()
                };
                let error = chain
                    .decode_into(
                        CowBytes::Borrowed(&[1, 2, 3]).into(),
                        &shape,
                        ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                        &CodecOptions::default(),
                    )
                    .unwrap_err();
                if fail {
                    assert!(error.to_string().contains("producer failure"));
                }
            }
            assert_eq!(output, [0xa5; 8]);
            assert_eq!(calls.lock().unwrap().len(), 1);
        }
    }
}

#[cfg(feature = "zstd")]
#[test]
fn real_compressor_places_non_native_values_into_strided_subset() {
    let chain = bind(&CodecChain::new(
        vec![],
        Arc::new(BytesCodec::new(Some(endian(true)))),
        vec![Arc::new(zarrs::array::codec::ZstdCodec::new(3, false))],
    ));
    run(chain.as_ref(), true);
}

/// An unknown receiver inherits the conservative default rather than `BytesCodec`'s
/// placement override. Record both the incoming allocation and decoded allocation.
#[derive(Debug)]
struct OrdinaryReceiver {
    bytes: Arc<dyn ArrayToBytesCodecTraits>,
    pointers: Arc<Mutex<Vec<usize>>>,
}

impl zarrs_codec::ArrayCodecTraits for OrdinaryReceiver {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn data_type(&self) -> &zarrs_data_type::DataType {
        self.bytes.data_type()
    }
    fn fill_value(&self) -> &FillValue {
        self.bytes.fill_value()
    }
    fn recommended_concurrency(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        self.bytes.recommended_concurrency(shape)
    }
}

impl zarrs_codec::ArrayToBytesCodecNoSubchunkingTraits for OrdinaryReceiver {}

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
        self.bytes.encoded_representation(shape)
    }
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        self.bytes.encode(bytes, shape, options)
    }
    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.pointers.lock().unwrap().push(bytes.as_ptr() as usize);
        let decoded = self.bytes.decode(bytes, shape, options)?;
        let ArrayBytes::Fixed(ref fixed) = decoded else {
            return Err(CodecError::Other("expected fixed bytes".into()));
        };
        self.pointers.lock().unwrap().push(fixed.as_ptr() as usize);
        Ok(decoded)
    }
}

#[test]
fn unknown_receiver_default_resolves_owned_producer_without_reallocating() {
    use zarrs_codec::{BytesDecodeSource, UnboundArrayToBytesCodecTraits};
    for non_native in [false, true] {
        let calls = Arc::default();
        let pointers = Arc::default();
        let producer = producer(&calls, 1, true, false, false);
        let receiver = OrdinaryReceiver {
            bytes: BytesCodec::new(Some(endian(non_native)))
                .with_context(
                    data_type::uint16(),
                    FillValue::from(0u16),
                    &CodecSpecificOptions::default(),
                )
                .unwrap(),
            pointers: Arc::clone(&pointers),
        };
        let shape = [NonZeroU64::new(4).unwrap()];
        let options = CodecOptions::default();
        let encoded = receiver
            .encode(ArrayBytes::new_flen(native_bytes()), &shape, &options)
            .unwrap();
        let mut output = [0xa5; 8];
        {
            // SAFETY: a single view is used serially and owns exclusive output access.
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut output),
                    2,
                    &[4],
                    ArraySubset::new_with_ranges(&[0..4]),
                )
                .unwrap()
            };
            receiver
                .decode_into(
                    ArrayBytesDecodeIntoInput::Deferred(BytesDecodeSource::new(
                        producer.as_ref(),
                        encoded,
                        BytesRepresentation::FixedSize(8),
                    )),
                    &shape,
                    ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                    &options,
                )
                .unwrap();
        }
        assert_eq!(output.as_slice(), native_bytes());
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert!(!calls[0].direct);
        assert_eq!(*pointers.lock().unwrap(), [calls[0].pointer; 2]);
    }
}

#[cfg(feature = "transpose")]
#[test]
fn array_to_array_branch_uses_ordinary_producer_route() {
    use zarrs::array::codec::{TransposeCodec, TransposeOrder};
    let calls = Arc::default();
    let chain = bind(&CodecChain::new(
        vec![Arc::new(TransposeCodec::new(
            TransposeOrder::new(&[1, 0]).unwrap(),
        ))],
        Arc::new(BytesCodec::new(Some(endian(true)))),
        vec![producer(&calls, 1, true, false, false)],
    ));
    run(chain.as_ref(), true);
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert!(!calls[0].direct);
}

#[cfg(all(feature = "sharding", feature = "zstd"))]
#[test]
fn sharded_array_compressed_non_native_subset_preserves_output_border() {
    use zarrs::array::{ArrayBuilder, codec::ShardingCodecBuilder};
    use zarrs_storage::store::MemoryStore;
    let dtype = data_type::uint16();
    let sharding = ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &dtype)
        .array_to_bytes_codec(Arc::new(BytesCodec::new(Some(endian(true)))))
        .bytes_to_bytes_codecs(vec![Arc::new(zarrs::array::codec::ZstdCodec::new(
            3, false,
        ))])
        .build_arc();
    let array = ArrayBuilder::new(vec![6, 6], vec![4, 4], dtype, 0u16)
        .array_to_bytes_codec(sharding)
        .build(Arc::new(MemoryStore::default()), "/deferred")
        .unwrap();
    array
        .store_array_subset(&array.subset_all(), (0..36).collect::<Vec<u16>>())
        .unwrap();
    let mut output = vec![0xa5; 6 * 6 * 2];
    {
        // SAFETY: one non-overlapping view; no independent access until it is dropped.
        let mut view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut output),
                2,
                &[6, 6],
                ArraySubset::new_with_ranges(&[1..5, 1..5]),
            )
            .unwrap()
        };
        array
            .retrieve_array_subset_into(
                &ArraySubset::new_with_ranges(&[1..5, 1..5]),
                ArrayBytesDecodeIntoTarget::Fixed(&mut view),
            )
            .unwrap();
    }
    for row in 0..6 {
        for column in 0..6 {
            let offset = (row * 6 + column) * 2;
            let expected = if (1..5).contains(&row) && (1..5).contains(&column) {
                u16::try_from(row * 6 + column).unwrap().to_ne_bytes()
            } else {
                [0xa5; 2]
            };
            assert_eq!(&output[offset..offset + 2], &expected);
        }
    }
}

#[test]
fn optional_receiver_resolves_efficient_producer_ordinary_route() {
    use zarrs::array::ArrayBuilder;
    use zarrs_storage::store::MemoryStore;
    let calls = Arc::default();
    // ArrayBuilder selects the existing OptionalCodec receiver.
    let array = ArrayBuilder::new(
        vec![4],
        vec![4],
        data_type::uint16().to_optional(),
        FillValue::from(None::<u16>),
    )
    .bytes_to_bytes_codecs(vec![producer(&calls, 1, true, false, false)])
    .build(Arc::new(MemoryStore::default()), "/optional")
    .unwrap();
    array
        .store_array_subset(
            &array.subset_all(),
            vec![Some(0x1234u16), None, Some(7), Some(0xabcd)],
        )
        .unwrap();
    let mut data = [0xa5; 8];
    let mut mask = [0xa5; 4];
    {
        // SAFETY: one view per distinct buffer, used serially with exclusive access.
        let mut data_view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut data),
                2,
                &[4],
                ArraySubset::new_with_ranges(&[0..4]),
            )
            .unwrap()
        };
        // SAFETY: the mask buffer is distinct from the data buffer.
        let mut mask_view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut mask),
                1,
                &[4],
                ArraySubset::new_with_ranges(&[0..4]),
            )
            .unwrap()
        };
        array
            .retrieve_array_subset_into(
                &array.subset_all(),
                ArrayBytesDecodeIntoTarget::Optional(
                    Box::new(ArrayBytesDecodeIntoTarget::Fixed(&mut data_view)),
                    &mut mask_view,
                ),
            )
            .unwrap();
    }
    assert_eq!(mask, [1, 0, 1, 1]);
    assert_eq!(&data[..2], &0x1234u16.to_ne_bytes());
    assert_eq!(&data[4..6], &7u16.to_ne_bytes());
    assert_eq!(&data[6..], &0xabcdu16.to_ne_bytes());
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert!(!calls[0].direct);
}

#[test]
fn operation_options_reach_producer_and_rejection_is_not_retried() {
    for efficient in [false, true] {
        let calls = Arc::default();
        let chain = bind(&CodecChain::new(
            vec![],
            Arc::new(BytesCodec::new(Some(endian(false)))),
            vec![producer(&calls, 1, efficient, false, false)],
        ));
        let mut output = [0xa5; 8];
        {
            // SAFETY: one view, used serially with exclusive buffer access.
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut output),
                    2,
                    &[4],
                    ArraySubset::new_with_ranges(&[0..4]),
                )
                .unwrap()
            };
            let error = chain
                .decode_into(
                    CowBytes::Borrowed(&[0; 8]).into(),
                    &[NonZeroU64::new(4).unwrap()],
                    ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                    &CodecOptions::default().with_validate_checksums(false),
                )
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("producer rejected operation options")
            );
        }
        assert_eq!(calls.lock().unwrap().len(), 1);
        assert_eq!(output, [0xa5; 8]);
    }
}
