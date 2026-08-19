//! Generic encoded subchunk retrieval tests.
//!
//! Encoded subchunk retrieval is codec generic: it is driven by the subchunk grid hierarchy and
//! the subchunk codecs exposed by a codec and its partial decoders, not by the `sharding_indexed`
//! codec specifically. These tests exercise the API with the `sharding_indexed` codec (including
//! nested sharding) and with a test codec that implements subchunking differently.

use std::borrow::Cow;
use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::chunk_cache::{
    ChunkCacheDecodedLruChunkLimit, ChunkCachePartialDecoderLruChunkLimit,
};
use zarrs::array::chunk_grid::RegularChunkGrid;
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::codec::{BytesCodec, TransposeCodec, TransposeOrder};
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytes, ArrayBytesRaw, ArrayCached, ArrayError, ArrayOps,
    ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, BytesPartialDecoderTraits,
    BytesRepresentation, ChunkGrid, ChunkShape, ChunkShapeTraits, CodecChain, CodecChainBound,
    CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits, DataType,
    FillValue, FromArrayBytes, Indexer, RecommendedConcurrency, UnboundArrayToBytesCodecTraits,
    data_type,
};
use zarrs::metadata::Configuration;
use zarrs::storage::StorageError;
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs_codec::{
    ArrayPartialDecoderSubchunkingTraits, ArrayToBytesCodecSubchunkingTraits, ChunkGridDecoded,
    ChunkGridDecodedRef, EncodedSubchunk, PartialDecoderCapability, PartialEncoderCapability,
};
use zarrs_plugin::ZarrVersion;

type TestResult = Result<(), Box<dyn Error>>;

fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}

/// Decode an encoded subchunk with the subchunk codecs of `array` at `level`.
///
/// The subchunk codec is a property of the array, while the encoded domain shape to decode with
/// travels with the subchunk.
fn decode_subchunk<A: ArrayOps>(
    array: &A,
    level: usize,
    encoded: &EncodedSubchunk<'_>,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let codecs = array
        .subchunk_codecs_at_level(level)
        .ok_or("no subchunk codecs")?;
    let bytes = codecs.decode(
        Cow::Owned(encoded.bytes().to_vec()),
        encoded.shape(),
        &CodecOptions::default(),
    )?;
    Ok(Vec::<u16>::from_array_bytes(
        bytes,
        bytemuck::must_cast_slice(encoded.shape()),
        codecs.data_type(),
    )?)
}

fn sharded_array() -> Result<Arc<Array<MemoryStore>>, Box<dyn Error>> {
    let store = Arc::new(MemoryStore::default());
    let data_type = data_type::uint16();
    let sharding = ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build_arc();
    let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type, 0u16);
    builder.array_to_bytes_codec(sharding);
    Ok(builder.build_arc(store, "/array")?)
}

#[test]
fn sharded_array_retrieve_encoded_subchunk() -> TestResult {
    let array = sharded_array()?;
    let data: Vec<u16> = (0..64).collect();
    array.store_array_subset(&array.subset_all(), &data)?;

    // The subchunk codec hierarchy matches the subchunk grid hierarchy
    assert_eq!(array.subchunk_grids().len(), 1);
    assert_eq!(array.subchunk_codecs().len(), 1);
    assert!(array.subchunk_codecs_at_level(1).is_none());

    for subchunk_indices in [[0, 0], [1, 2], [3, 3]] {
        let encoded = array
            .retrieve_encoded_subchunk(&subchunk_indices)?
            .expect("subchunk is stored");
        assert_eq!(
            decode_subchunk(array.as_ref(), 0, &encoded)?,
            array.retrieve_subchunk::<Vec<u16>>(&subchunk_indices)?
        );
    }

    // The bare and level variants are equivalent at level zero
    assert_eq!(
        array.retrieve_encoded_subchunk(&[1, 2])?,
        array.retrieve_encoded_subchunk_at_level(0, &[1, 2])?
    );

    // Encoded subchunks are available through a chunk cache that retains a partial decoder
    let cached = ArrayCached::new(array.clone(), ChunkCachePartialDecoderLruChunkLimit::new(2));
    assert_eq!(cached.subchunk_codecs().len(), 1);
    assert_eq!(
        cached.retrieve_encoded_subchunk(&[1, 2])?,
        array.retrieve_encoded_subchunk(&[1, 2])?
    );

    // ... but not through a cache of decoded chunks
    let cached = ArrayCached::new(array.clone(), ChunkCacheDecodedLruChunkLimit::new(2));
    assert!(cached.retrieve_encoded_subchunk(&[1, 2]).is_err());

    Ok(())
}

#[test]
fn sharded_array_retrieve_encoded_subchunk_missing() -> TestResult {
    let array = sharded_array()?;

    // Nothing is stored, so no subchunk is stored
    assert_eq!(array.retrieve_encoded_subchunk(&[0, 0])?, None);

    // Store a single chunk (shard), the subchunks of other shards remain unstored
    let chunk: Vec<u16> = (0..16).collect();
    array.store_chunk(&[0, 0], &chunk)?;
    assert!(array.retrieve_encoded_subchunk(&[0, 0])?.is_some());
    assert_eq!(array.retrieve_encoded_subchunk(&[3, 3])?, None);

    Ok(())
}

#[test]
fn sharded_array_retrieve_encoded_subchunk_errors() -> TestResult {
    let array = sharded_array()?;
    let data: Vec<u16> = (0..64).collect();
    array.store_array_subset(&array.subset_all(), &data)?;

    // Out-of-bounds subchunk indices
    assert!(array.retrieve_encoded_subchunk(&[4, 0]).is_err());
    // Subchunk indices with a mismatched dimensionality
    assert!(array.retrieve_encoded_subchunk(&[1]).is_err());
    assert!(array.retrieve_encoded_subchunk(&[1, 1, 7]).is_err());

    // The partial decoder is the lowest level entry point and validates its own indices, rather
    // than silently linearising mismatched indices into a different subchunk
    let partial_decoder = array.partial_decoder(&[0, 0])?;
    let options = CodecOptions::default();
    assert!(
        partial_decoder
            .retrieve_encoded_subchunk(&[0, 0], &options)?
            .is_some()
    );
    assert!(
        partial_decoder
            .retrieve_encoded_subchunk(&[1], &options)
            .is_err()
    );
    assert!(
        partial_decoder
            .retrieve_encoded_subchunk(&[1, 1, 7], &options)
            .is_err()
    );
    // Beyond the subchunk grid hierarchy
    assert!(matches!(
        array.retrieve_encoded_subchunk_at_level(1, &[0, 0]),
        Err(ArrayError::MissingSubchunkGrid)
    ));

    // An array without subchunks
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type::uint16(), 0u16)
        .build_arc(store, "/array")?;
    array.store_array_subset(&array.subset_all(), &data)?;
    assert!(matches!(
        array.retrieve_encoded_subchunk(&[0, 0]),
        Err(ArrayError::MissingSubchunkGrid)
    ));

    Ok(())
}

#[test]
fn sharded_array_with_array_to_array_codec_has_no_encoded_subchunks() -> TestResult {
    let store = Arc::new(MemoryStore::default());
    let data_type = data_type::uint16();
    let sharding = ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build_arc();
    let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type, 0u16);
    builder.array_to_array_codecs(vec![Arc::new(TransposeCodec::new(
        TransposeOrder::new(&[1, 0]).unwrap(),
    ))]);
    builder.array_to_bytes_codec(sharding);
    let array = builder.build_arc(store, "/array")?;
    let data: Vec<u16> = (0..64).collect();
    array.store_array_subset(&array.subset_all(), &data)?;

    // The subchunk grid is resolvable through the transpose codec, but the encoded subchunks are
    // in the encoded (transposed) domain and are not exposed by the transpose partial decoder.
    assert!(array.subchunk_grid().as_chunk_grid().is_some());
    assert!(matches!(
        array.retrieve_encoded_subchunk(&[0, 0]),
        Err(ArrayError::CodecError(
            CodecError::UnsupportedEncodedSubchunk
        ))
    ));

    // The subchunk codecs are still exposed for introspection
    assert_eq!(array.subchunk_codecs().len(), 1);

    Ok(())
}

#[test]
fn nested_sharded_array_retrieve_encoded_subchunk_at_level() -> TestResult {
    let store = Arc::new(MemoryStore::default());
    let data_type = data_type::uint16();
    let inner_sharding = ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build_arc();
    let mut outer_sharding = ShardingCodecBuilder::new(vec![nz(4), nz(4)], &data_type);
    outer_sharding.array_to_bytes_codec(inner_sharding);
    let mut builder = ArrayBuilder::new(vec![8, 8], vec![8, 8], data_type, 0u16);
    builder.array_to_bytes_codec(outer_sharding.build_arc());
    let array = builder.build_arc(store, "/array")?;
    let data: Vec<u16> = (0..64).collect();
    array.store_array_subset(&array.subset_all(), &data)?;

    assert_eq!(array.subchunk_grids().len(), 2);
    assert_eq!(array.subchunk_codecs().len(), 2);

    let options = CodecOptions::default();

    // Level zero: an encoded subchunk is an encoded inner shard
    let encoded = array
        .retrieve_encoded_subchunk_at_level(0, &[1, 0])?
        .expect("subchunk is stored");
    assert_eq!(
        decode_subchunk(array.as_ref(), 0, &encoded)?,
        array.retrieve_subchunk_at_level::<Vec<u16>>(0, &[1, 0])?
    );

    // Level one: an encoded subchunk of a subchunk
    let encoded = array
        .retrieve_encoded_subchunk_at_level(1, &[2, 3])?
        .expect("subchunk is stored");
    assert_eq!(
        decode_subchunk(array.as_ref(), 1, &encoded)?,
        [38, 39, 46, 47]
    );

    // Beyond the hierarchy
    assert!(
        array
            .retrieve_encoded_subchunk_at_level(2, &[0, 0])
            .is_err()
    );

    // Deeper levels are also reachable directly from a partial decoder, whose subchunk indices are
    // relative to its chunk (here the array is a single chunk, so they are identical)
    let partial_decoder = array.partial_decoder(&[0, 0])?;
    assert_eq!(partial_decoder.subchunk_codecs().len(), 2);
    let encoded = partial_decoder
        .retrieve_encoded_subchunk_at_level(1, &[2, 3], &options)?
        .expect("subchunk is stored");
    assert_eq!(
        decode_subchunk(array.as_ref(), 1, &encoded)?,
        [38, 39, 46, 47]
    );
    assert!(
        partial_decoder
            .retrieve_encoded_subchunk_at_level(2, &[0, 0], &options)
            .is_err()
    );

    Ok(())
}

#[test]
fn nested_sharded_array_encoded_subchunks_are_read_lazily() -> TestResult {
    // The sharding codec exposes an encoded subchunk as a byte interval of the shard, so a nested
    // encoded subchunk is read without retrieving the enclosing inner shard in full.
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::default(),
    )));
    let data_type = data_type::uint16();
    let inner_sharding = ShardingCodecBuilder::new(vec![nz(16), nz(16)], &data_type).build_arc();
    let mut outer_sharding = ShardingCodecBuilder::new(vec![nz(32), nz(32)], &data_type);
    outer_sharding.array_to_bytes_codec(inner_sharding);
    let mut builder = ArrayBuilder::new(vec![64, 64], vec![64, 64], data_type, 0u16);
    builder.array_to_bytes_codec(outer_sharding.build_arc());
    let array = builder.build_arc(store.clone(), "/array")?;
    let data: Vec<u16> = (0..64 * 64).map(|i| i as u16).collect();
    array.store_array_subset(&array.subset_all(), &data)?;

    // An inner shard holds 32x32 elements, a subchunk of it holds 16x16 elements
    let inner_shard_bytes = 32 * 32 * size_of::<u16>();
    let subchunk_bytes = 16 * 16 * size_of::<u16>();

    store.reset();
    let encoded = array
        .retrieve_encoded_subchunk_at_level(1, &[3, 2])?
        .expect("subchunk is stored");
    assert_eq!(encoded.bytes().len(), subchunk_bytes);
    // Only the two shard indexes and the subchunk itself are read
    assert!(
        store.bytes_read() < inner_shard_bytes,
        "read {} bytes, expected less than an inner shard ({inner_shard_bytes} bytes)",
        store.bytes_read()
    );

    Ok(())
}

#[test]
fn array_without_subchunks_has_no_encoded_subchunks() -> TestResult {
    // The `bytes` codec partial decoder opts out of subchunking with the marker trait
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![8], vec![4], data_type::uint16(), 0u16)
        .build_arc(store, "/array")?;
    let data: Vec<u16> = (0..8).collect();
    array.store_array_subset(&array.subset_all(), &data)?;

    let options = CodecOptions::default();
    let partial_decoder = array.partial_decoder(&[0])?;
    assert!(partial_decoder.local_subchunk_grids(&options)?.is_empty());
    assert!(partial_decoder.local_subchunk_grid(&options)?.is_none());
    assert!(partial_decoder.subchunk_codecs().is_empty());
    assert!(matches!(
        partial_decoder.retrieve_encoded_subchunk(&[0], &options),
        Err(CodecError::UnsupportedEncodedSubchunk)
    ));
    assert!(matches!(
        partial_decoder.retrieve_encoded_subchunk_at_level(0, &[0], &options),
        Err(CodecError::UnsupportedEncodedSubchunk)
    ));
    assert!(
        partial_decoder
            .retrieve_encoded_subchunk_at_level(1, &[0], &options)
            .is_err()
    );

    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_sharded_array_retrieve_encoded_subchunk() -> TestResult {
    use zarrs::storage::store::AsyncMemoryStore;

    let store = Arc::new(AsyncMemoryStore::default());
    let data_type = data_type::uint16();
    let sharding = ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build_arc();
    let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type, 0u16);
    builder.array_to_bytes_codec(sharding);
    let array = builder.build_arc(store, "/array")?;
    let data: Vec<u16> = (0..64).collect();
    array
        .async_store_array_subset(&array.subset_all(), &data)
        .await?;

    let encoded = array
        .async_retrieve_encoded_subchunk(&[1, 2])
        .await?
        .expect("subchunk is stored");
    assert_eq!(
        decode_subchunk(array.as_ref(), 0, &encoded)?,
        array.async_retrieve_subchunk::<Vec<u16>>(&[1, 2]).await?
    );

    assert!(
        array
            .async_retrieve_encoded_subchunk_at_level(1, &[0, 0])
            .await
            .is_err()
    );

    Ok(())
}

/// A test array-to-bytes codec that splits a one dimensional chunk into fixed size blocks.
///
/// Each block is independently encoded with the inner codecs and the encoded blocks are
/// concatenated. Unlike `sharding_indexed`, there is no index: block offsets are implied by the
/// fixed encoded size of a block.
#[derive(Debug)]
struct BlockCodec {
    block_shape: ChunkShape,
    inner_codecs: Arc<CodecChain>,
}

#[derive(Debug)]
struct BlockCodecBound {
    block_shape: ChunkShape,
    inner_codecs: Arc<CodecChainBound>,
    data_type: DataType,
    fill_value: FillValue,
}

struct BlockCodecPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    shape: ChunkShape,
    codec: Arc<BlockCodecBound>,
}

zarrs_plugin::impl_extension_aliases!(BlockCodec, v3: "zarrs.test.block");

impl BlockCodec {
    fn new(block_shape: ChunkShape) -> Self {
        Self {
            block_shape,
            inner_codecs: Arc::new(CodecChain::new(
                vec![],
                Arc::new(BytesCodec::default()),
                vec![],
            )),
        }
    }
}

impl BlockCodecBound {
    /// The encoded size of a block, which is identical for every block.
    fn block_size(&self) -> Result<u64, CodecError> {
        match self
            .inner_codecs
            .encoded_representation(&self.block_shape)?
        {
            BytesRepresentation::FixedSize(size) => Ok(size),
            _ => Err(CodecError::Other(
                "the block codec requires fixed size blocks".to_string(),
            )),
        }
    }

    /// The number of blocks in a chunk of `shape`.
    fn num_blocks(&self, shape: &[NonZeroU64]) -> Result<u64, CodecError> {
        if shape.len() != 1 || self.block_shape.len() != 1 {
            return Err(CodecError::Other(
                "the block codec is one dimensional".to_string(),
            ));
        }
        let (shape, block) = (shape[0].get(), self.block_shape[0].get());
        if shape.is_multiple_of(block) {
            Ok(shape / block)
        } else {
            Err(CodecError::Other(
                "the block shape must evenly divide the chunk shape".to_string(),
            ))
        }
    }

    fn block_grid(&self, array_shape: Vec<u64>) -> Result<ChunkGrid, CodecError> {
        Ok(ChunkGrid::new(
            RegularChunkGrid::new(array_shape, self.block_shape.clone())
                .map_err(|err| CodecError::Other(err.to_string()))?,
        ))
    }
}

impl CodecTraits for BlockCodec {
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

impl UnboundArrayToBytesCodecTraits for BlockCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        Ok(Arc::new(BlockCodecBound {
            block_shape: self.block_shape.clone(),
            inner_codecs: self
                .inner_codecs
                .with_context(data_type.clone(), fill_value.clone())?,
            data_type,
            fill_value,
        }))
    }
}

impl zarrs_codec::ArrayCodecTraits for BlockCodecBound {
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

impl ArrayToBytesCodecSubchunkingTraits for BlockCodecBound {
    fn decoded_subchunk_grids(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<Vec<ChunkGridDecoded>, zarrs::array::ChunkGridCreateError> {
        Ok(vec![match decoded_chunk_grid {
            ChunkGridDecodedRef::Array(chunk_grid) => ChunkGridDecoded::Array(ChunkGrid::new(
                RegularChunkGrid::new(chunk_grid.array_shape().to_vec(), self.block_shape.clone())?,
            )),
            ChunkGridDecodedRef::ChunkLocal => ChunkGridDecoded::ChunkLocal,
            ChunkGridDecodedRef::None => ChunkGridDecoded::None,
        }])
    }

    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
        vec![ArrayToBytesCodecTraits::into_dyn(self.inner_codecs.clone())]
    }
}

impl ArrayToBytesCodecTraits for BlockCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        Ok(BytesRepresentation::FixedSize(
            self.num_blocks(shape)? * self.block_size()?,
        ))
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        bytes.validate(shape.num_elements_u64(), &self.data_type)?;
        let num_blocks = self.num_blocks(shape)?;
        let bytes = bytes.into_fixed()?;
        let block_elements = usize::try_from(self.block_shape[0].get()).unwrap();
        let element_size = self.data_type.fixed_size().expect("fixed size data type");
        let mut encoded = Vec::with_capacity(bytes.len());
        for block in 0..usize::try_from(num_blocks).unwrap() {
            let start = block * block_elements * element_size;
            let end = start + block_elements * element_size;
            let block_bytes = ArrayBytes::new_flen(bytes[start..end].to_vec());
            encoded.extend_from_slice(&self.inner_codecs.encode(
                block_bytes,
                &self.block_shape,
                options,
            )?);
        }
        Ok(Cow::Owned(encoded))
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let num_blocks = usize::try_from(self.num_blocks(shape)?).unwrap();
        let block_size = usize::try_from(self.block_size()?).unwrap();
        if bytes.len() != num_blocks * block_size {
            return Err(CodecError::Other(
                "unexpected encoded block codec chunk size".to_string(),
            ));
        }
        let mut decoded = Vec::new();
        for block in 0..num_blocks {
            let block_bytes = Cow::Borrowed(&bytes[block * block_size..(block + 1) * block_size]);
            let block_bytes = self
                .inner_codecs
                .decode(block_bytes, &self.block_shape, options)?;
            decoded.extend_from_slice(&block_bytes.into_fixed()?);
        }
        Ok(ArrayBytes::new_flen(decoded))
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(BlockCodecPartialDecoder {
            input_handle,
            shape: shape.to_vec(),
            codec: self,
        }))
    }
}

impl ArrayPartialDecoderSubchunkingTraits for BlockCodecPartialDecoder {
    fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        Ok(vec![Some(self.codec.block_grid(
            bytemuck::must_cast_slice(&self.shape).to_vec(),
        )?)])
    }

    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
        ArrayToBytesCodecSubchunkingTraits::subchunk_codecs(self.codec.as_ref())
    }

    fn retrieve_encoded_subchunk_bytes(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytesRaw<'_>>, CodecError> {
        let num_blocks = self.codec.num_blocks(&self.shape)?;
        if subchunk_indices.len() != 1 || subchunk_indices[0] >= num_blocks {
            return Err(CodecError::Other(format!(
                "block {subchunk_indices:?} is out of bounds of {num_blocks} blocks"
            )));
        }
        let block_size = self.codec.block_size()?;
        self.input_handle.partial_decode(
            ByteRange::new(
                subchunk_indices[0] * block_size..(subchunk_indices[0] + 1) * block_size,
            ),
            options,
        )
    }
}

impl ArrayPartialDecoderTraits for BlockCodecPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.codec.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let array_shape: Vec<u64> = bytemuck::must_cast_slice(&self.shape).to_vec();
        let bytes = if let Some(bytes) = self.input_handle.decode(options)? {
            self.codec.decode(bytes, &self.shape, options)?.into_owned()
        } else {
            ArrayBytes::new_fill_value(
                &self.codec.data_type,
                self.shape.num_elements_u64(),
                &self.codec.fill_value,
            )?
        };
        Ok(bytes
            .extract_array_subset(indexer, &array_shape, &self.codec.data_type)?
            .into_owned())
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[test]
fn block_codec_retrieve_encoded_subchunk() -> TestResult {
    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new(vec![12], vec![6], data_type::uint16(), 0u16);
    builder.array_to_bytes_codec(Arc::new(BlockCodec::new(vec![nz(3)])));
    let array = builder.build_arc(store, "/array")?;
    let data: Vec<u16> = (0..12).collect();
    array.store_array_subset(&array.subset_all(), &data)?;

    // The test codec exposes a single subchunk grid level and a single subchunk codec level
    assert_eq!(array.subchunk_shape(), Some(vec![nz(3)]));
    assert_eq!(
        array.subchunk_grid().as_chunk_grid().unwrap().grid_shape(),
        &[4]
    );
    assert_eq!(array.subchunk_codecs().len(), 1);

    for subchunk_indices in 0..4 {
        let encoded = array
            .retrieve_encoded_subchunk(&[subchunk_indices])?
            .expect("subchunk is stored");
        assert_eq!(encoded.bytes().len(), 3 * size_of::<u16>());
        assert_eq!(
            decode_subchunk(array.as_ref(), 0, &encoded)?,
            array.retrieve_subchunk::<Vec<u16>>(&[subchunk_indices])?
        );
    }

    assert!(array.retrieve_encoded_subchunk(&[4]).is_err());

    Ok(())
}

#[test]
fn block_codec_within_sharding_retrieve_encoded_subchunk_at_level() -> TestResult {
    // A mixed hierarchy: level zero subchunks are sharding subchunks, level one subchunks are
    // blocks of the test codec. The generic descent spans both codecs.
    let store = Arc::new(MemoryStore::default());
    let data_type = data_type::uint16();
    let mut sharding = ShardingCodecBuilder::new(vec![nz(6)], &data_type);
    sharding.array_to_bytes_codec(Arc::new(BlockCodec::new(vec![nz(3)])));
    let mut builder = ArrayBuilder::new(vec![12], vec![12], data_type, 0u16);
    builder.array_to_bytes_codec(sharding.build_arc());
    let array = builder.build_arc(store, "/array")?;
    let data: Vec<u16> = (0..12).collect();
    array.store_array_subset(&array.subset_all(), &data)?;

    assert_eq!(array.subchunk_grids().len(), 2);
    assert_eq!(array.subchunk_codecs().len(), 2);
    assert_eq!(array.subchunk_shape_at_level(0), Some(vec![nz(6)]));
    assert_eq!(array.subchunk_shape_at_level(1), Some(vec![nz(3)]));

    let options = CodecOptions::default();

    // Level zero: a sharding subchunk, encoded by the block codec
    let encoded = array
        .retrieve_encoded_subchunk_at_level(0, &[1])?
        .expect("subchunk is stored");
    assert_eq!(
        decode_subchunk(array.as_ref(), 0, &encoded)?,
        [6, 7, 8, 9, 10, 11]
    );

    // Level one: a block of a sharding subchunk
    let encoded = array
        .retrieve_encoded_subchunk_at_level(1, &[3])?
        .expect("subchunk is stored");
    assert_eq!(encoded.bytes().len(), 3 * size_of::<u16>());
    assert_eq!(decode_subchunk(array.as_ref(), 1, &encoded)?, [9, 10, 11]);

    // ... and directly from the shard partial decoder
    let partial_decoder = array.partial_decoder(&[0])?;
    let encoded = partial_decoder
        .retrieve_encoded_subchunk_at_level(1, &[3], &options)?
        .expect("subchunk is stored");
    assert_eq!(decode_subchunk(array.as_ref(), 1, &encoded)?, [9, 10, 11]);

    Ok(())
}
