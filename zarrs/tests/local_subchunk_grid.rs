//! Local subchunk grid tests.

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serial_test::serial;
use zarrs::array::chunk_grid::RegularChunkGrid;
use zarrs::array::codec::{TransposeCodec, TransposeOrder};
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytes, ArrayBytesRaw, ArrayPartialDecoderTraits, ArrayReadOps,
    ArrayToBytesCodecTraits, BytesPartialDecoderTraits, BytesRepresentation, ChunkGrid, ChunkShape,
    ChunkShapeTraits, Codec, CodecChain, CodecChainBound, CodecCreateError, CodecError,
    CodecMetadataOptions, CodecOptions, CodecTraits, DataType, DataTypeSize, FillValue,
    RecommendedConcurrency, UnboundArrayToBytesCodecTraits, data_type,
};
use zarrs::metadata::Configuration;
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::StorageError;
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::store::MemoryStore;
use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_codec::{
    ArrayCodecTraits, ArrayToArrayCodecTraits, ChunkGridDecoded, ChunkGridDecodedRef,
    ChunkGridEncoded, ChunkGridEncodedRef, PartialDecoderCapability, PartialEncoderCapability,
    UnboundArrayToArrayCodecTraits, register_codec_v3, unregister_codec_v3,
};
use zarrs_plugin::{ExtensionName, RuntimePlugin, ZarrVersion};

#[derive(Clone, Debug)]
struct DynamicLocalSubchunkCodec;

#[derive(Debug)]
struct DynamicLocalSubchunkCodecBound {
    data_type: DataType,
    fill_value: FillValue,
}

struct DynamicLocalSubchunkPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    shape: ChunkShape,
    data_type: DataType,
}

static NEXT_SHAPE: AtomicU64 = AtomicU64::new(0);

zarrs_plugin::impl_extension_aliases!(
    DynamicLocalSubchunkCodec,
    v3: "zarrs.test.dynamic_local_subchunk"
);

fn next_subchunk_shape(shape: &[NonZeroU64]) -> ChunkShape {
    let seed = NEXT_SHAPE.fetch_add(1, Ordering::Relaxed) + 1;
    shape
        .iter()
        .enumerate()
        .map(|(dim, size)| {
            let size = size.get();
            let value = 1 + ((seed + dim as u64 * 3) % size);
            NonZeroU64::new(value).unwrap()
        })
        .collect()
}

fn header_len(dimensionality: usize) -> u64 {
    (dimensionality * size_of::<u64>()) as u64
}

fn encode_shape_header(shape: &[NonZeroU64]) -> Vec<u8> {
    shape
        .iter()
        .flat_map(|dim| dim.get().to_le_bytes())
        .collect()
}

fn decode_shape_header(bytes: &[u8], dimensionality: usize) -> Result<ChunkShape, CodecError> {
    if bytes.len() != dimensionality * size_of::<u64>() {
        return Err(CodecError::Other(
            "dynamic local subchunk header has invalid length".to_string(),
        ));
    }
    bytes
        .chunks_exact(size_of::<u64>())
        .map(|chunk| {
            let value = u64::from_le_bytes(chunk.try_into().unwrap());
            NonZeroU64::new(value).ok_or_else(|| {
                CodecError::Other("dynamic local subchunk shape contains zero".to_string())
            })
        })
        .collect()
}

fn zero_bytes(data_type: &DataType, num_elements: u64) -> Result<ArrayBytes<'static>, CodecError> {
    let DataTypeSize::Fixed(size) = data_type.size() else {
        return Err(CodecError::UnsupportedDataType(
            data_type.clone(),
            "dynamic local subchunk test codec only supports fixed-size data types".to_string(),
        ));
    };
    Ok(ArrayBytes::new_flen(vec![
        0;
        usize::try_from(num_elements)
            .unwrap()
            * size
    ]))
}

impl CodecTraits for DynamicLocalSubchunkCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(Configuration::default())
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

impl zarrs_codec::CodecTraitsV3 for DynamicLocalSubchunkCodec {
    fn create(_metadata: &MetadataV3) -> Result<Codec, CodecCreateError> {
        Ok(Codec::ArrayToBytes(Arc::new(Self)))
    }
}

impl UnboundArrayToBytesCodecTraits for DynamicLocalSubchunkCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        Ok(Arc::new(DynamicLocalSubchunkCodecBound {
            data_type,
            fill_value,
        }))
    }
}

impl zarrs_codec::ArrayCodecTraits for DynamicLocalSubchunkCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl ArrayToBytesCodecTraits for DynamicLocalSubchunkCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        Ok(BytesRepresentation::FixedSize(header_len(shape.len())))
    }

    fn decoded_subchunk_grid(
        &self,
        _decoded_chunk_grid: zarrs_codec::ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridDecoded, zarrs::array::ChunkGridCreateError> {
        Ok(ChunkGridDecoded::ChunkLocal)
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        bytes.validate(shape.num_elements_u64(), &self.data_type)?;
        Ok(Cow::Owned(encode_shape_header(&next_subchunk_shape(shape))))
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        decode_shape_header(&bytes, shape.len())?;
        zero_bytes(&self.data_type, shape.num_elements_u64()).map(ArrayBytes::into_owned)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(DynamicLocalSubchunkPartialDecoder {
            input_handle,
            shape: shape.to_vec(),
            data_type: self.data_type.clone(),
        }))
    }
}

impl ArrayPartialDecoderTraits for DynamicLocalSubchunkPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn zarrs::array::Indexer,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        zero_bytes(&self.data_type, indexer.len())
    }

    fn local_subchunk_grid(&self, options: &CodecOptions) -> Result<Option<ChunkGrid>, CodecError> {
        let Some(header) = self.input_handle.partial_decode(
            ByteRange::FromStart(0, Some(header_len(self.shape.len()))),
            options,
        )?
        else {
            return Ok(None);
        };
        let subchunk_shape = decode_shape_header(&header, self.shape.len())?;
        let chunk_shape = bytemuck::must_cast_slice(&self.shape).to_vec();
        Ok(Some(ChunkGrid::new(
            RegularChunkGrid::new(chunk_shape, subchunk_shape)
                .map_err(|err| CodecError::Other(err.to_string()))?,
        )))
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[test]
#[serial]
fn dynamic_local_subchunk_grids_can_differ_by_chunk() -> Result<(), Box<dyn std::error::Error>> {
    NEXT_SHAPE.store(0, Ordering::Relaxed);
    let handle = register_codec_v3(RuntimePlugin::new(
        |name| name == "zarrs.test.dynamic_local_subchunk",
        |_metadata| Ok(Codec::ArrayToBytes(Arc::new(DynamicLocalSubchunkCodec))),
    ));

    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![12, 6], vec![6, 6], data_type::uint16(), 0u16)
        .array_to_bytes_codec(Arc::new(DynamicLocalSubchunkCodec))
        .build(store.clone(), "/array")?;
    array.store_metadata()?;

    let data = vec![1u16; 12 * 6];
    array.store_array_subset(&array.subset_all(), &data)?;

    let reopened: Array<MemoryStore> = Array::open(store, "/array")?;
    assert!(reopened.subchunk_grid().is_none());

    let first = reopened
        .local_subchunk_grid(&[0, 0], &CodecOptions::default())?
        .unwrap();
    let second = reopened
        .local_subchunk_grid(&[1, 0], &CodecOptions::default())?
        .unwrap();
    assert_ne!(
        first.chunk_shape(&[0, 0])?.unwrap(),
        second.chunk_shape(&[0, 0])?.unwrap()
    );

    let decoded: Vec<u16> = reopened.retrieve_chunk(&[0, 0])?;
    assert_eq!(decoded, vec![0u16; 36]);

    assert!(unregister_codec_v3(&handle));
    Ok(())
}

#[test]
#[serial]
fn dynamic_local_subchunk_grid_transforms_through_transpose()
-> Result<(), Box<dyn std::error::Error>> {
    NEXT_SHAPE.store(0, Ordering::Relaxed);
    let handle = register_codec_v3(RuntimePlugin::new(
        |name| name == "zarrs.test.dynamic_local_subchunk",
        |_metadata| Ok(Codec::ArrayToBytes(Arc::new(DynamicLocalSubchunkCodec))),
    ));

    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new(vec![4, 7], vec![4, 7], data_type::uint16(), 0u16);
    builder.array_to_array_codecs(vec![Arc::new(TransposeCodec::new(
        TransposeOrder::new(&[1, 0]).unwrap(),
    ))]);
    builder.array_to_bytes_codec(Arc::new(DynamicLocalSubchunkCodec));
    let array = builder.build(store.clone(), "/array")?;
    array.store_metadata()?;

    let data = vec![1u16; 4 * 7];
    array.store_array_subset(&array.subset_all(), &data)?;

    let reopened: Array<MemoryStore> = Array::open(store, "/array")?;
    let local_grid = reopened
        .local_subchunk_grid(&[0, 0], &CodecOptions::default())?
        .unwrap();
    assert_eq!(local_grid.array_shape(), &[4, 7]);
    assert_eq!(
        local_grid.chunk_shape(&[0, 0])?.unwrap(),
        vec![NonZeroU64::new(1).unwrap(), NonZeroU64::new(2).unwrap()]
    );

    assert!(unregister_codec_v3(&handle));
    Ok(())
}

#[derive(Debug)]
struct LocalOnlyReshapeGridCodec {
    reject_chunk_local: bool,
}

#[derive(Debug)]
struct LocalOnlyReshapeGridCodecBound {
    data_type: DataType,
    fill_value: FillValue,
    reject_chunk_local: bool,
}

impl ExtensionName for LocalOnlyReshapeGridCodec {
    fn name(&self, _version: ZarrVersion) -> Option<Cow<'static, str>> {
        None
    }
}

impl CodecTraits for LocalOnlyReshapeGridCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        None
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: true,
        }
    }
}

impl UnboundArrayToArrayCodecTraits for LocalOnlyReshapeGridCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self
    }

    fn with_codec_specific_options(
        self: Arc<Self>,
        _opts: &zarrs_codec::CodecSpecificOptions,
    ) -> Result<Arc<dyn UnboundArrayToArrayCodecTraits>, CodecCreateError> {
        Ok(self)
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        Ok(Arc::new(LocalOnlyReshapeGridCodecBound {
            data_type,
            fill_value,
            reject_chunk_local: self.reject_chunk_local,
        }))
    }
}

impl ArrayCodecTraits for LocalOnlyReshapeGridCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl ArrayToArrayCodecTraits for LocalOnlyReshapeGridCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.data_type
    }

    fn encoded_fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        Ok(vec![
            NonZeroU64::new(decoded_shape.num_elements_u64()).unwrap(),
        ])
    }

    fn encoded_chunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridEncoded, ChunkGridCreateError> {
        match decoded_chunk_grid {
            ChunkGridDecodedRef::None => Ok(ChunkGridEncoded::None),
            ChunkGridDecodedRef::Array(_) => Ok(ChunkGridEncoded::ChunkLocal),
            ChunkGridDecodedRef::ChunkLocal if self.reject_chunk_local => {
                Ok(ChunkGridEncoded::None)
            }
            ChunkGridDecodedRef::ChunkLocal => Ok(ChunkGridEncoded::ChunkLocal),
        }
    }

    fn decoded_subchunk_grid(
        &self,
        _decoded_chunk_grid: ChunkGridDecodedRef<'_>,
        encoded_subchunk_grid: ChunkGridEncodedRef<'_>,
    ) -> Result<ChunkGridDecoded, ChunkGridCreateError> {
        Ok(encoded_subchunk_grid.into())
    }

    fn encode<'a>(
        &self,
        _bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        unimplemented!("test codec only exercises subchunk-grid propagation")
    }

    fn decode<'a>(
        &self,
        _bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        unimplemented!("test codec only exercises subchunk-grid propagation")
    }
}

#[derive(Debug)]
struct TestSubchunkingCodec {
    expose_subchunks: bool,
}

impl ExtensionName for TestSubchunkingCodec {
    fn name(&self, _version: ZarrVersion) -> Option<Cow<'static, str>> {
        None
    }
}

impl CodecTraits for TestSubchunkingCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        None
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: true,
        }
    }
}

impl UnboundArrayToBytesCodecTraits for TestSubchunkingCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self.clone()
    }

    fn with_codec_specific_options(
        self: Arc<Self>,
        _opts: &zarrs_codec::CodecSpecificOptions,
    ) -> Result<Arc<dyn UnboundArrayToBytesCodecTraits>, CodecCreateError> {
        Ok(self)
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        Ok(Arc::new(TestSubchunkingCodecBound {
            data_type,
            fill_value,
            expose_subchunks: self.expose_subchunks,
        }))
    }
}

#[derive(Debug)]
struct TestSubchunkingCodecBound {
    data_type: DataType,
    fill_value: FillValue,
    expose_subchunks: bool,
}

impl ArrayCodecTraits for TestSubchunkingCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl ArrayToBytesCodecTraits for TestSubchunkingCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self
    }

    fn encoded_representation(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        Ok(BytesRepresentation::UnboundedSize)
    }

    fn decoded_subchunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridDecoded, ChunkGridCreateError> {
        if !self.expose_subchunks {
            return Ok(ChunkGridDecoded::None);
        }
        match decoded_chunk_grid {
            ChunkGridDecodedRef::None => Ok(ChunkGridDecoded::None),
            ChunkGridDecodedRef::ChunkLocal => Ok(ChunkGridDecoded::ChunkLocal),
            ChunkGridDecodedRef::Array(decoded_chunk_grid) => Ok(ChunkGridDecoded::Array(
                ChunkGrid::new(RegularChunkGrid::new(
                    decoded_chunk_grid.array_shape().to_vec(),
                    vec![NonZeroU64::new(3).unwrap()],
                )?),
            )),
        }
    }

    fn encode<'a>(
        &self,
        _bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        unimplemented!("test codec only exercises subchunk-grid propagation")
    }

    fn decode<'a>(
        &self,
        _bytes: ArrayBytesRaw<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        unimplemented!("test codec only exercises subchunk-grid propagation")
    }
}

fn local_only_grid_chain(expose_subchunks: bool, reject_chunk_local: bool) -> Arc<CodecChainBound> {
    let data_type = data_type::uint8();
    let fill_value = FillValue::from(0u8);
    let mut array_to_array: Vec<Arc<dyn UnboundArrayToArrayCodecTraits>> =
        vec![Arc::new(LocalOnlyReshapeGridCodec {
            reject_chunk_local: false,
        })];
    if reject_chunk_local {
        array_to_array.push(Arc::new(LocalOnlyReshapeGridCodec {
            reject_chunk_local: true,
        }));
    }
    CodecChain::new(
        array_to_array,
        Arc::new(TestSubchunkingCodec { expose_subchunks }),
        vec![],
    )
    .with_context(data_type, fill_value)
    .unwrap()
}

#[test]
fn codec_chain_discovers_chunk_local_grid_after_global_mapping_fails() {
    let decoded_chunk_grid = ChunkGrid::new(
        RegularChunkGrid::new(
            vec![4, 6],
            vec![NonZeroU64::new(2).unwrap(), NonZeroU64::new(6).unwrap()],
        )
        .unwrap(),
    );
    let chain = local_only_grid_chain(true, false);
    assert!(matches!(
        chain
            .decoded_subchunk_grid((&decoded_chunk_grid).into())
            .unwrap(),
        ChunkGridDecoded::ChunkLocal
    ));
}

#[test]
fn codec_chain_does_not_infer_chunk_local_without_downstream_subchunks() {
    let decoded_chunk_grid = ChunkGrid::new(
        RegularChunkGrid::new(
            vec![4, 6],
            vec![NonZeroU64::new(2).unwrap(), NonZeroU64::new(6).unwrap()],
        )
        .unwrap(),
    );
    let chain = local_only_grid_chain(false, false);
    assert!(matches!(
        chain
            .decoded_subchunk_grid((&decoded_chunk_grid).into())
            .unwrap(),
        ChunkGridDecoded::None
    ));
}

#[test]
fn codec_chain_allows_later_array_codec_to_reject_chunk_local() {
    let decoded_chunk_grid = ChunkGrid::new(
        RegularChunkGrid::new(
            vec![4, 6],
            vec![NonZeroU64::new(2).unwrap(), NonZeroU64::new(6).unwrap()],
        )
        .unwrap(),
    );
    let chain = local_only_grid_chain(true, true);
    assert!(matches!(
        chain
            .decoded_subchunk_grid((&decoded_chunk_grid).into())
            .unwrap(),
        ChunkGridDecoded::None
    ));
}
