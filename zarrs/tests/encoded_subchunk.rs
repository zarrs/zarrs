#![allow(missing_docs)]
//! Generic encoded subchunk retrieval tests.
//!
//! Encoded subchunk retrieval is codec generic: it is driven by the subchunk codecs of a codec and the subchunks it
//! locates, not by the `sharding_indexed` codec specifically. These tests exercise the API with the `sharding_indexed`
//! codec (including nested sharding, array-to-array codecs and bytes-to-bytes codecs) and with a test codec that
//! implements subchunking differently.

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::chunk_cache::ChunkCacheDecodedLruChunkLimit;
use zarrs::array::chunk_grid::RegularChunkGrid;
use zarrs::array::codec::array_to_array::reshape::ReshapeShape;
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::codec::{
    BytesCodec, Crc32cCodec, GzipCodec, ReshapeCodec, SqueezeCodec, TransposeCodec, TransposeOrder,
};
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytes, ArrayCached, ArrayError, ArrayOps, ArrayToBytesCodecTraits,
    ArrayUpdateOps, BytesRepresentation, ChunkGrid, ChunkShape, ChunkShapeTraits, CodecChain,
    CodecChainBound, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    CowBytes, DataType, EncodedSubchunk, FillValue, FromArrayBytes, Resources,
    UnboundArrayToBytesCodecTraits, data_type,
};
use zarrs::metadata::Configuration;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;
use zarrs_codec::plan::{ByteSource, Outcome, SubchunkLocation, Task, subchunk_location_value};
use zarrs_codec::{
    ArrayCodecTraits, ArrayToBytesCodecSubchunkingTraits, ChunkGridDecoded, ChunkGridDecodedRef,
    CodecSpecificOptions,
};
use zarrs_plugin::ZarrVersion;

type TestResult = Result<(), Box<dyn Error>>;

fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}

/// Decode an encoded subchunk with the subchunk codec of `array` at `level`.
///
/// The subchunk codec is a property of the array, while the encoded domain shape to decode with travels with the
/// subchunk.
fn decode_subchunk<A: ArrayOps>(
    array: &A,
    level: usize,
    encoded: &EncodedSubchunk,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let codec = array
        .subchunk_codecs_at_level(level)
        .ok_or("no subchunk codecs")?;
    let bytes = codec.decode(
        CowBytes::from(encoded.bytes().to_vec()),
        encoded.shape(),
        &CodecOptions::default(),
        &Resources::default(),
    )?;
    Ok(Vec::<u16>::from_array_bytes(
        bytes.into_owned(),
        bytemuck::must_cast_slice(encoded.shape()),
        codec.data_type(),
    )?)
}

fn store_all<A: ArrayUpdateOps>(array: &A, data: &[u16]) -> TestResult {
    array.store_array_subset(&array.subset_all(), data, &Resources::default())?;
    Ok(())
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
    store_all(array.as_ref(), &data)?;
    let resources = Resources::default();

    // The subchunk codec hierarchy matches the subchunk grid hierarchy
    assert_eq!(array.subchunk_grids().len(), 1);
    assert_eq!(array.subchunk_codecs().len(), 1);
    assert!(array.subchunk_codecs_at_level(1).is_none());

    for subchunk_indices in [[0, 0], [1, 2], [3, 3]] {
        let encoded = array
            .retrieve_encoded_subchunk(&subchunk_indices, &resources)?
            .expect("subchunk is stored");
        assert_eq!(encoded.shape(), [nz(2), nz(2)]);
        assert_eq!(
            decode_subchunk(array.as_ref(), 0, &encoded)?,
            array.retrieve_subchunk::<Vec<u16>>(&subchunk_indices, &resources)?
        );
    }

    // The bare and level variants are equivalent at level zero
    assert_eq!(
        array.retrieve_encoded_subchunk(&[1, 2], &resources)?,
        array.retrieve_encoded_subchunk_at_level(0, &[1, 2], &resources)?
    );

    // Encoded subchunks are read from the store through a chunk cache
    let cached = ArrayCached::new(array.clone(), ChunkCacheDecodedLruChunkLimit::new(2));
    assert_eq!(cached.subchunk_codecs().len(), 1);
    assert_eq!(
        cached.retrieve_encoded_subchunk(&[1, 2], &resources)?,
        array.retrieve_encoded_subchunk(&[1, 2], &resources)?
    );

    Ok(())
}

#[test]
fn sharded_array_retrieve_encoded_subchunk_missing() -> TestResult {
    let array = sharded_array()?;
    let resources = Resources::default();

    // Nothing is stored, so no subchunk is stored
    assert_eq!(array.retrieve_encoded_subchunk(&[0, 0], &resources)?, None);

    // Store a single chunk (shard), the subchunks of other shards remain unstored
    let chunk: Vec<u16> = (1..17).collect();
    array.store_chunk(&[0, 0], &chunk, &resources)?;
    assert!(
        array
            .retrieve_encoded_subchunk(&[0, 0], &resources)?
            .is_some()
    );
    assert_eq!(array.retrieve_encoded_subchunk(&[3, 3], &resources)?, None);

    // A subchunk of the fill value is not stored in a stored shard
    let mut chunk = vec![0u16; 16];
    chunk[0] = 1;
    array.store_chunk(&[0, 0], &chunk, &resources)?;
    assert!(
        array
            .retrieve_encoded_subchunk(&[0, 0], &resources)?
            .is_some()
    );
    assert_eq!(array.retrieve_encoded_subchunk(&[1, 1], &resources)?, None);

    Ok(())
}

#[test]
fn sharded_array_retrieve_encoded_subchunk_errors() -> TestResult {
    let array = sharded_array()?;
    let data: Vec<u16> = (0..64).collect();
    store_all(array.as_ref(), &data)?;
    let resources = Resources::default();

    // Out-of-bounds subchunk indices
    assert!(
        array
            .retrieve_encoded_subchunk(&[4, 0], &resources)
            .is_err()
    );
    // Subchunk indices with a mismatched dimensionality
    assert!(array.retrieve_encoded_subchunk(&[1], &resources).is_err());
    assert!(
        array
            .retrieve_encoded_subchunk(&[1, 1, 7], &resources)
            .is_err()
    );
    // Beyond the subchunk grid hierarchy
    assert!(matches!(
        array.retrieve_encoded_subchunk_at_level(1, &[0, 0], &resources),
        Err(ArrayError::MissingSubchunkGrid)
    ));

    // An array without subchunks
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type::uint16(), 0u16)
        .build_arc(store, "/array")?;
    store_all(array.as_ref(), &data)?;
    assert!(array.subchunk_codecs().is_empty());
    assert!(matches!(
        array.retrieve_encoded_subchunk(&[0, 0], &resources),
        Err(ArrayError::MissingSubchunkGrid)
    ));

    Ok(())
}

#[test]
fn sharded_array_with_reshape_codec_retrieve_encoded_subchunk() -> TestResult {
    // The reshape flattens a [2,6] chunk to [12], so the encoded domain is one dimensional
    let store = Arc::new(MemoryStore::default());
    let data_type = data_type::uint16();
    let sharding = ShardingCodecBuilder::new(vec![nz(3)], &data_type).build_arc();
    let mut builder = ArrayBuilder::new(vec![4, 6], vec![2, 6], data_type, 0u16);
    builder.array_to_array_codecs(vec![Arc::new(ReshapeCodec::new(
        ReshapeShape::new([nz(12).into()]).unwrap(),
    ))]);
    builder.array_to_bytes_codec(sharding);
    let array = builder.build_arc(store, "/array")?;
    let data: Vec<u16> = (0..24).collect();
    store_all(array.as_ref(), &data)?;
    let resources = Resources::default();

    // The subchunk grid is in the domain of the array, so it is two dimensional
    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap().clone();
    assert_eq!(subchunk_grid.grid_shape(), [4, 2]);

    for subchunk_indices in [[0, 0], [0, 1], [2, 1], [3, 1]] {
        let encoded = array
            .retrieve_encoded_subchunk(&subchunk_indices, &resources)?
            .expect("subchunk is stored");

        // The encoded shape is one dimensional
        let subchunk_subset = subchunk_grid.subset(&subchunk_indices)?.unwrap();
        assert_eq!(subchunk_subset.shape(), [1, 3]);
        assert_eq!(encoded.shape(), [nz(3)]);

        // A reshape does not reorder elements, so the values match the array subset
        assert_eq!(
            decode_subchunk(array.as_ref(), 0, &encoded)?,
            array.retrieve_array_subset::<Vec<u16>>(&subchunk_subset, &resources)?
        );
    }

    Ok(())
}

#[test]
fn sharded_array_with_squeeze_codec_retrieve_encoded_subchunk() -> TestResult {
    // Squeeze drops the unit axis, so the encoded domain has a lower dimensionality than the array
    let store = Arc::new(MemoryStore::default());
    let data_type = data_type::uint16();
    let sharding = ShardingCodecBuilder::new(vec![nz(2)], &data_type).build_arc();
    let mut builder = ArrayBuilder::new(vec![2, 4], vec![1, 4], data_type, 0u16);
    builder.array_to_array_codecs(vec![Arc::new(SqueezeCodec::new())]);
    builder.array_to_bytes_codec(sharding);
    let array = builder.build_arc(store, "/array")?;
    let data: Vec<u16> = (0..8).collect();
    store_all(array.as_ref(), &data)?;
    let resources = Resources::default();

    // The subchunk grid is in the domain of the array, so it retains the unit axis
    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap().clone();
    assert_eq!(subchunk_grid.grid_shape(), [2, 2]);

    for subchunk_indices in [[0, 0], [0, 1], [1, 0], [1, 1]] {
        let encoded = array
            .retrieve_encoded_subchunk(&subchunk_indices, &resources)?
            .expect("subchunk is stored");

        // The encoded shape has the unit axis dropped
        let subchunk_subset = subchunk_grid.subset(&subchunk_indices)?.unwrap();
        assert_eq!(subchunk_subset.shape(), [1, 2]);
        assert_eq!(encoded.shape(), [nz(2)]);

        // Squeeze does not reorder elements, so the values match the array subset
        assert_eq!(
            decode_subchunk(array.as_ref(), 0, &encoded)?,
            array.retrieve_array_subset::<Vec<u16>>(&subchunk_subset, &resources)?
        );
    }

    Ok(())
}

#[test]
fn sharded_array_with_transpose_codec_retrieve_encoded_subchunk() -> TestResult {
    // A non-square subchunk shape, so that a mismapped axis permutation is detectable
    let store = Arc::new(MemoryStore::default());
    let data_type = data_type::uint16();
    let sharding = ShardingCodecBuilder::new(vec![nz(2), nz(4)], &data_type).build_arc();
    let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 8], data_type, 0u16);
    builder.array_to_array_codecs(vec![Arc::new(TransposeCodec::new(
        TransposeOrder::new(&[1, 0]).unwrap(),
    ))]);
    builder.array_to_bytes_codec(sharding);
    let array = builder.build_arc(store, "/array")?;
    let data: Vec<u16> = (0..64).collect();
    store_all(array.as_ref(), &data)?;
    let resources = Resources::default();

    assert!(array.subchunk_grid().as_chunk_grid().is_some());
    assert_eq!(array.subchunk_codecs().len(), 1);

    // Encoded subchunks are exposed through the transpose codec. They are in the encoded (transposed) domain: the
    // subchunk is addressed in the domain of the array, but the shape and element order of the bytes are those of the
    // encoded domain.
    for subchunk_indices in [[0, 0], [1, 1], [1, 0], [0, 3]] {
        let encoded = array
            .retrieve_encoded_subchunk(&subchunk_indices, &resources)?
            .expect("subchunk is stored");

        // The array subchunk grid is transposed relative to the encoded domain
        let subchunk_subset = array
            .subchunk_grid()
            .as_chunk_grid()
            .unwrap()
            .subset(&subchunk_indices)?
            .unwrap();
        assert_eq!(subchunk_subset.shape(), [4, 2]);
        assert_eq!(encoded.shape(), [nz(2), nz(4)]);

        // Decoding yields the subchunk values transposed, since the transpose codec has not been applied to them
        let decoded = decode_subchunk(array.as_ref(), 0, &encoded)?;
        let expected = array.retrieve_array_subset::<Vec<u16>>(&subchunk_subset, &resources)?;
        let transposed: Vec<u16> = (0..2)
            .flat_map(|col| (0..4).map(move |row| (row, col)))
            .map(|(row, col)| expected[row * 2 + col])
            .collect();
        assert_eq!(decoded, transposed);
    }

    Ok(())
}

#[test]
fn sharded_array_with_bytes_to_bytes_codecs_retrieve_encoded_subchunk() -> TestResult {
    let data_type = data_type::uint16();
    let data: Vec<u16> = (0..64).collect();
    let resources = Resources::default();
    // A checksum is a window of its input, so subchunks are read lazily through it. A compressor decodes the whole
    // shard first.
    for (bytes_to_bytes, lazy) in [
        (vec![Arc::new(Crc32cCodec::new()) as Arc<_>], true),
        (vec![Arc::new(GzipCodec::new(5)?) as Arc<_>], false),
    ] {
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
            MemoryStore::default(),
        )));
        let sharding = ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build_arc();
        let mut builder = ArrayBuilder::new(vec![8, 8], vec![8, 8], data_type.clone(), 0u16);
        builder.array_to_bytes_codec(sharding);
        builder.bytes_to_bytes_codecs(bytes_to_bytes);
        let array = builder.build_arc(store.clone(), "/array")?;
        store_all(array.as_ref(), &data)?;

        let shard_bytes = array
            .retrieve_encoded_chunk(&[0, 0])?
            .expect("shard is stored")
            .len();
        store.reset();
        let encoded = array
            .retrieve_encoded_subchunk(&[1, 2], &resources)?
            .expect("subchunk is stored");
        assert_eq!(store.bytes_read() < shard_bytes, lazy);
        assert_eq!(
            decode_subchunk(array.as_ref(), 0, &encoded)?,
            array.retrieve_subchunk::<Vec<u16>>(&[1, 2], &resources)?
        );
        assert_eq!(
            array
                .retrieve_encoded_subchunk(&[0, 0], &resources)?
                .is_some(),
            true
        );
    }
    Ok(())
}

fn nested_sharded_array<
    TStorage: ?Sized + zarrs::storage::ReadableWritableStorageTraits + 'static,
>(
    store: Arc<TStorage>,
    outer: u64,
    inner: u64,
    shape: u64,
) -> Result<Arc<Array<TStorage>>, Box<dyn Error>> {
    let data_type = data_type::uint16();
    let inner_sharding = ShardingCodecBuilder::new(vec![nz(inner); 2], &data_type).build_arc();
    let mut outer_sharding = ShardingCodecBuilder::new(vec![nz(outer); 2], &data_type);
    outer_sharding.array_to_bytes_codec(inner_sharding);
    let mut builder = ArrayBuilder::new(vec![shape; 2], vec![shape; 2], data_type, 0u16);
    builder.array_to_bytes_codec(outer_sharding.build_arc());
    Ok(builder.build_arc(store, "/array")?)
}

#[test]
fn nested_sharded_array_retrieve_encoded_subchunk_at_level() -> TestResult {
    let array = nested_sharded_array(Arc::new(MemoryStore::default()), 4, 2, 8)?;
    let data: Vec<u16> = (0..64).collect();
    store_all(array.as_ref(), &data)?;
    let resources = Resources::default();

    assert_eq!(array.subchunk_grids().len(), 2);
    assert_eq!(array.subchunk_codecs().len(), 2);

    // Level zero: an encoded subchunk is an encoded inner shard
    let encoded = array
        .retrieve_encoded_subchunk_at_level(0, &[1, 0], &resources)?
        .expect("subchunk is stored");
    assert_eq!(encoded.shape(), [nz(4), nz(4)]);
    assert_eq!(
        decode_subchunk(array.as_ref(), 0, &encoded)?,
        array.retrieve_subchunk_at_level::<Vec<u16>>(0, &[1, 0], &resources)?
    );

    // Level one: an encoded subchunk of a subchunk
    let encoded = array
        .retrieve_encoded_subchunk_at_level(1, &[2, 3], &resources)?
        .expect("subchunk is stored");
    assert_eq!(encoded.shape(), [nz(2), nz(2)]);
    assert_eq!(
        decode_subchunk(array.as_ref(), 1, &encoded)?,
        [38, 39, 46, 47]
    );

    // Beyond the hierarchy
    assert!(
        array
            .retrieve_encoded_subchunk_at_level(2, &[0, 0], &resources)
            .is_err()
    );

    Ok(())
}

#[test]
fn nested_sharded_array_encoded_subchunks_are_read_lazily() -> TestResult {
    // The sharding codec locates an encoded subchunk as a byte interval of the shard, so a nested encoded subchunk is
    // read without retrieving the enclosing inner shard in full.
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::default(),
    )));
    let array = nested_sharded_array(store.clone(), 32, 16, 64)?;
    let data: Vec<u16> = (0..64 * 64).map(|i| i as u16).collect();
    store_all(array.as_ref(), &data)?;

    // An inner shard holds 32x32 elements, a subchunk of it holds 16x16 elements
    let inner_shard_bytes = 32 * 32 * size_of::<u16>();
    let subchunk_bytes = 16 * 16 * size_of::<u16>();

    store.reset();
    let encoded = array
        .retrieve_encoded_subchunk_at_level(1, &[3, 2], &Resources::default())?
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

/// A single chunk array with a transpose codec in front of nested sharding.
///
/// The subchunk shapes are non-square, so resolving an encoded subchunk shape in the decoded (transposed) domain
/// rather than the encoded domain is detectable.
fn transposed_nested_sharding_builder() -> ArrayBuilder {
    let data_type = data_type::uint16();
    let inner_sharding = ShardingCodecBuilder::new(vec![nz(2), nz(4)], &data_type).build_arc();
    let mut outer_sharding = ShardingCodecBuilder::new(vec![nz(4), nz(8)], &data_type);
    outer_sharding.array_to_bytes_codec(inner_sharding);
    let mut builder = ArrayBuilder::new(vec![8, 8], vec![8, 8], data_type, 0u16);
    builder.array_to_array_codecs(vec![Arc::new(TransposeCodec::new(
        TransposeOrder::new(&[1, 0]).unwrap(),
    ))]);
    builder.array_to_bytes_codec(outer_sharding.build_arc());
    builder
}

#[test]
fn nested_sharded_array_behind_transpose_retrieve_encoded_subchunk_at_level() -> TestResult {
    let store = Arc::new(MemoryStore::default());
    let array = transposed_nested_sharding_builder().build_arc(store, "/array")?;
    let data: Vec<u16> = (0..64).collect();
    store_all(array.as_ref(), &data)?;
    let resources = Resources::default();

    for (level, subchunk_indices, shape) in [
        (0, [0, 1], [nz(4), nz(8)]),
        (1, [1, 3], [nz(2), nz(4)]),
        (1, [0, 2], [nz(2), nz(4)]),
    ] {
        let encoded = array
            .retrieve_encoded_subchunk_at_level(level, &subchunk_indices, &resources)?
            .expect("subchunk is stored");
        assert_eq!(encoded.shape(), shape);

        // Decoding and transposing the subchunk gives the subchunk of the array
        let subchunk_subset = array
            .subchunk_grid_at_level(level)
            .as_chunk_grid()
            .unwrap()
            .subset(&subchunk_indices)?
            .unwrap();
        let expected = array.retrieve_array_subset::<Vec<u16>>(&subchunk_subset, &resources)?;
        let decoded = decode_subchunk(array.as_ref(), level, &encoded)?;
        let (rows, cols) = (
            usize::try_from(subchunk_subset.shape()[0]).unwrap(),
            usize::try_from(subchunk_subset.shape()[1]).unwrap(),
        );
        let transposed: Vec<u16> = (0..cols)
            .flat_map(|col| (0..rows).map(move |row| (row, col)))
            .map(|(row, col)| expected[row * cols + col])
            .collect();
        assert_eq!(decoded, transposed);
    }

    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_sharded_array_retrieve_encoded_subchunk() -> TestResult {
    use zarrs::storage::store::AsyncMemoryStore;

    let store = Arc::new(AsyncMemoryStore::default());
    let array = transposed_nested_sharding_builder().build_arc(store, "/array")?;
    let data: Vec<u16> = (0..64).collect();
    array
        .async_store_array_subset(&array.subset_all(), &data, &Resources::default())
        .await?;
    let resources = Resources::default();

    let encoded = array
        .async_retrieve_encoded_subchunk(&[0, 1], &resources)
        .await?
        .expect("subchunk is stored");
    assert_eq!(encoded.shape(), [nz(4), nz(8)]);
    let encoded = array
        .async_retrieve_encoded_subchunk_at_level(1, &[1, 3], &resources)
        .await?
        .expect("subchunk is stored");
    assert_eq!(encoded.shape(), [nz(2), nz(4)]);
    assert_eq!(
        decode_subchunk(array.as_ref(), 1, &encoded)?,
        [38, 46, 54, 62, 39, 47, 55, 63]
    );

    assert!(
        array
            .async_retrieve_encoded_subchunk_at_level(2, &[0, 0], &resources)
            .await
            .is_err()
    );

    Ok(())
}

/// A test array-to-bytes codec that splits a one dimensional chunk into fixed size blocks.
///
/// Each block is independently encoded with the inner codecs and the encoded blocks are concatenated. Unlike
/// `sharding_indexed`, there is no index: block offsets are implied by the fixed encoded size of a block.
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
        let [shape] = shape else {
            return Err(CodecError::Other(
                "the block codec is one dimensional".to_string(),
            ));
        };
        let (shape, block) = (shape.get(), self.block_shape[0].get());
        if shape.is_multiple_of(block) {
            Ok(shape / block)
        } else {
            Err(CodecError::Other(
                "the block shape must evenly divide the chunk shape".to_string(),
            ))
        }
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
}

impl UnboundArrayToBytesCodecTraits for BlockCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        Ok(Arc::new(BlockCodecBound {
            block_shape: self.block_shape.clone(),
            inner_codecs: self.inner_codecs.with_context(
                data_type.clone(),
                fill_value.clone(),
                codec_specific_options,
            )?,
            data_type,
            fill_value,
        }))
    }
}

impl ArrayCodecTraits for BlockCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> &FillValue {
        &self.fill_value
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
        vec![self.inner_codecs.clone().into_dyn()]
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
        resources: &Resources,
    ) -> Result<CowBytes<'a>, CodecError> {
        bytes.validate(shape.num_elements_u64(), &self.data_type)?;
        let num_blocks = usize::try_from(self.num_blocks(shape)?).unwrap();
        let bytes = bytes.into_fixed()?;
        let block_len = bytes.len() / num_blocks;
        let mut encoded = Vec::with_capacity(bytes.len());
        for block in bytes.chunks(block_len) {
            encoded.extend_from_slice(&self.inner_codecs.encode(
                ArrayBytes::new_flen(block.to_vec()),
                &self.block_shape,
                options,
                resources,
            )?);
        }
        Ok(CowBytes::from(encoded))
    }

    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let num_blocks = usize::try_from(self.num_blocks(shape)?).unwrap();
        let block_size = usize::try_from(self.block_size()?).unwrap();
        if bytes.len() != num_blocks * block_size {
            return Err(CodecError::Other(
                "unexpected encoded block codec chunk size".to_string(),
            ));
        }
        let mut decoded = Vec::new();
        for block in bytes.chunks(block_size) {
            let block = self.inner_codecs.decode(
                CowBytes::from(block.to_vec()),
                &self.block_shape,
                options,
                resources,
            )?;
            decoded.extend_from_slice(&block.into_fixed()?);
        }
        Ok(ArrayBytes::new_flen(decoded))
    }

    fn plan_locate_subchunk(
        self: Arc<Self>,
        source: ByteSource<'static>,
        shape: &[NonZeroU64],
        coordinates: &[u64],
        _options: &CodecOptions,
    ) -> Result<Task<'static>, CodecError> {
        // Blocks have a fixed encoded size, so a block is located without reading
        let num_blocks = self.num_blocks(shape)?;
        let block = coordinates[0] / self.block_shape[0].get();
        if coordinates.len() != 1 || block >= num_blocks {
            return Err(CodecError::Other(format!(
                "coordinates {coordinates:?} are out of bounds of {num_blocks} blocks"
            )));
        }
        let block_size = self.block_size()?;
        let location = SubchunkLocation {
            source: source.interval(block * block_size, block_size)?,
            shape: self.block_shape.clone(),
            coordinates: vec![coordinates[0] % self.block_shape[0].get()],
        };
        Ok(Task::compute(move |_| {
            Ok(Outcome::Value(subchunk_location_value(Some(location))))
        }))
    }
}

#[test]
fn block_codec_retrieve_encoded_subchunk() -> TestResult {
    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new(vec![12], vec![6], data_type::uint16(), 0u16);
    builder.array_to_bytes_codec(Arc::new(BlockCodec::new(vec![nz(3)])));
    let array = builder.build_arc(store, "/array")?;
    let data: Vec<u16> = (0..12).collect();
    store_all(array.as_ref(), &data)?;
    let resources = Resources::default();

    // The test codec exposes a single subchunk grid level and a single subchunk codec level
    assert_eq!(array.subchunk_shape(), Some(vec![nz(3)]));
    assert_eq!(
        array.subchunk_grid().as_chunk_grid().unwrap().grid_shape(),
        &[4]
    );
    assert_eq!(array.subchunk_codecs().len(), 1);

    for subchunk_indices in 0..4 {
        let encoded = array
            .retrieve_encoded_subchunk(&[subchunk_indices], &resources)?
            .expect("subchunk is stored");
        assert_eq!(encoded.bytes().len(), 3 * size_of::<u16>());
        assert_eq!(
            decode_subchunk(array.as_ref(), 0, &encoded)?,
            array.retrieve_subchunk::<Vec<u16>>(&[subchunk_indices], &resources)?
        );
    }

    assert!(array.retrieve_encoded_subchunk(&[4], &resources).is_err());

    Ok(())
}

#[test]
fn block_codec_within_sharding_retrieve_encoded_subchunk_at_level() -> TestResult {
    // A mixed hierarchy: level zero subchunks are sharding subchunks, level one subchunks are blocks of the test codec.
    // The generic descent spans both codecs.
    let store = Arc::new(MemoryStore::default());
    let data_type = data_type::uint16();
    let mut sharding = ShardingCodecBuilder::new(vec![nz(6)], &data_type);
    sharding.array_to_bytes_codec(Arc::new(BlockCodec::new(vec![nz(3)])));
    let mut builder = ArrayBuilder::new(vec![12], vec![12], data_type, 0u16);
    builder.array_to_bytes_codec(sharding.build_arc());
    let array = builder.build_arc(store, "/array")?;
    let data: Vec<u16> = (0..12).collect();
    store_all(array.as_ref(), &data)?;
    let resources = Resources::default();

    assert_eq!(array.subchunk_grids().len(), 2);
    assert_eq!(array.subchunk_codecs().len(), 2);
    assert_eq!(array.subchunk_shape_at_level(0), Some(vec![nz(6)]));
    assert_eq!(array.subchunk_shape_at_level(1), Some(vec![nz(3)]));

    // Level zero: a sharding subchunk, encoded by the block codec
    let encoded = array
        .retrieve_encoded_subchunk_at_level(0, &[1], &resources)?
        .expect("subchunk is stored");
    assert_eq!(
        decode_subchunk(array.as_ref(), 0, &encoded)?,
        [6, 7, 8, 9, 10, 11]
    );

    // Level one: a block of a sharding subchunk
    let encoded = array
        .retrieve_encoded_subchunk_at_level(1, &[3], &resources)?
        .expect("subchunk is stored");
    assert_eq!(encoded.bytes().len(), 3 * size_of::<u16>());
    assert_eq!(decode_subchunk(array.as_ref(), 1, &encoded)?, [9, 10, 11]);

    Ok(())
}
