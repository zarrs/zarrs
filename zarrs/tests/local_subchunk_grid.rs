//! These tests are fully disabled for now pending investigation of "sparse chunk grids", which might be a better API
#![cfg(any())]

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
    ChunkShapeTraits, Codec, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecTraits, DataType, DataTypeSize, FillValue, RecommendedConcurrency, SubchunkGrid,
    UnboundArrayToBytesCodecTraits, data_type,
};
use zarrs::metadata::Configuration;
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::StorageError;
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::store::MemoryStore;
use zarrs_codec::{
    PartialDecoderCapability, PartialEncoderCapability, register_codec_v3, unregister_codec_v3,
};
use zarrs_plugin::{RuntimePlugin, ZarrVersion};

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
        _shape: &[u64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl ArrayToBytesCodecTraits for DynamicLocalSubchunkCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self
    }

    fn encoded_representation(&self, shape: &[u64]) -> Result<BytesRepresentation, CodecError> {
        Ok(BytesRepresentation::FixedSize(header_len(shape.len())))
    }

    fn decoded_subchunk_grid(
        &self,
        _decoded_chunk_grid: &ChunkGrid,
    ) -> Result<SubchunkGrid, zarrs::array::ChunkGridCreateError> {
        Ok(SubchunkGrid::ChunkLocalDynamic)
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        bytes.validate(shape.num_elements(), &self.data_type)?;
        Ok(Cow::Owned(encode_shape_header(&next_subchunk_shape(shape))))
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytesRaw<'a>,
        shape: &[u64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        decode_shape_header(&bytes, shape.len())?;
        zero_bytes(&self.data_type, shape.num_elements()).map(ArrayBytes::into_owned)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[u64],
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
    assert!(matches!(
        reopened.subchunk_grid_kind(),
        SubchunkGrid::ChunkLocalDynamic
    ));

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
