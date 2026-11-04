#![allow(missing_docs)]
//! Encoded subchunks are retrieved without decoding a chunk, even if the codecs of an array cache a decoded chunk for
//! partial decoding.

use std::borrow::Cow;
use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use zarrs::array::chunk_cache::{
    ChunkCache, ChunkCacheDecodedLruChunkLimit, ChunkCacheEncodedLruChunkLimit,
    ChunkCachePartialDecoderLruChunkLimit,
};
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytes, ArrayCached, ArrayError, ArrayIndices, ArrayOps, ChunkGrid,
    ChunkShape, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    CowBytes, DataType, FillValue, FromArrayBytes, RecommendedConcurrency, data_type,
};
use zarrs::metadata::Configuration;
use zarrs::storage::ReadableWritableStorageTraits;
use zarrs::storage::store::MemoryStore;
use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_codec::{
    ArrayCodecTraits, ArrayToArrayCodecTraits, ChunkGridDecoded, ChunkGridDecodedRef,
    ChunkGridEncoded, ChunkGridEncodedRef, CodecSpecificOptions, EncodedSubchunk,
    PartialDecoderCapability, PartialEncoderCapability, UnboundArrayToArrayCodecTraits,
};
use zarrs_plugin::{ExtensionName, ZarrVersion};

type TestResult = Result<(), Box<dyn Error>>;

fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}

/// An identity array-to-array codec that cannot partially decode, so partial decoders of its codec chain cache the
/// decoded chunk.
///
/// Decoding panics while `forbid_decode` is set.
#[derive(Debug, Default)]
struct NoPartialDecodeCodec {
    forbid_decode: Arc<AtomicBool>,
}

#[derive(Debug)]
struct NoPartialDecodeCodecBound {
    forbid_decode: Arc<AtomicBool>,
    data_type: DataType,
    fill_value: Option<FillValue>,
}

impl ExtensionName for NoPartialDecodeCodec {
    fn name(&self, _version: ZarrVersion) -> Option<Cow<'static, str>> {
        None
    }
}

impl CodecTraits for NoPartialDecodeCodec {
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
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl UnboundArrayToArrayCodecTraits for NoPartialDecodeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: Option<FillValue>,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        Ok(Arc::new(NoPartialDecodeCodecBound {
            forbid_decode: self.forbid_decode.clone(),
            data_type,
            fill_value,
        }))
    }
}

impl ArrayCodecTraits for NoPartialDecodeCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> Option<&FillValue> {
        self.fill_value.as_ref()
    }

    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl zarrs_codec::ArrayToArrayCodecSubchunkingTraits for NoPartialDecodeCodecBound {
    fn encoded_chunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridEncoded, ChunkGridCreateError> {
        Ok(match decoded_chunk_grid {
            ChunkGridDecodedRef::Array(grid) => ChunkGridEncoded::Array(grid.clone()),
            ChunkGridDecodedRef::ChunkLocal => ChunkGridEncoded::ChunkLocal,
            ChunkGridDecodedRef::None => ChunkGridEncoded::None,
        })
    }

    fn decoded_subchunk_grid(
        &self,
        _decoded_chunk_grid: ChunkGridDecodedRef<'_>,
        encoded_subchunk_grid: ChunkGridEncodedRef<'_>,
    ) -> Result<ChunkGridDecoded, ChunkGridCreateError> {
        Ok(encoded_subchunk_grid.into())
    }

    fn decoded_local_subchunk_grid(
        &self,
        _decoded_shape: &[NonZeroU64],
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        Ok(Some(encoded_subchunk_grid.clone()))
    }

    fn encoded_coordinates(
        &self,
        _decoded_shape: &[NonZeroU64],
        decoded_coordinates: &[u64],
    ) -> Result<ArrayIndices, CodecError> {
        Ok(decoded_coordinates.into())
    }
}

impl ArrayToArrayCodecTraits for NoPartialDecodeCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.data_type
    }

    fn encoded_fill_value(&self) -> Option<&FillValue> {
        self.fill_value.as_ref()
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        Ok(decoded_shape.to_vec())
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        assert!(
            !self.forbid_decode.load(Ordering::Relaxed),
            "a chunk was decoded"
        );
        Ok(bytes)
    }
}

/// The level zero and level one subchunks retrieved by the tests.
const SUBCHUNKS: [(usize, [u64; 2]); 4] = [(0, [0, 0]), (0, [1, 0]), (1, [0, 1]), (1, [3, 2])];

/// An 8x8 single chunk array of nested shards (4x4 then 2x2 subchunks), optionally behind the identity
/// [`NoPartialDecodeCodec`].
fn nested_sharded_builder(codec: Option<Arc<NoPartialDecodeCodec>>) -> ArrayBuilder {
    let data_type = data_type::uint16();
    let inner = ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build_arc();
    let mut outer = ShardingCodecBuilder::new(vec![nz(4), nz(4)], &data_type);
    outer.array_to_bytes_codec(inner);
    let mut builder = ArrayBuilder::new(vec![8, 8], vec![8, 8], data_type, 0u16);
    builder.array_to_bytes_codec(outer.build_arc());
    if let Some(codec) = codec {
        builder.array_to_array_codecs(vec![codec]);
    }
    builder
}

/// Build [`nested_sharded_builder`] in `store` and store `0..64` in it.
fn nested_sharded_array<TStorage: ?Sized + ReadableWritableStorageTraits + 'static>(
    store: Arc<TStorage>,
    codec: Option<Arc<NoPartialDecodeCodec>>,
) -> Result<Array<TStorage>, Box<dyn Error>> {
    let array = nested_sharded_builder(codec).build(store, "/array")?;
    array.store_chunk(&[0, 0], (0..64u16).collect::<Vec<_>>())?;
    Ok(array)
}

/// A function retrieving the encoded subchunk at a level and subchunk indices.
type Retrieve<'a> = dyn Fn(usize, &[u64]) -> Result<Option<EncodedSubchunk>, ArrayError> + 'a;

/// Decode an `encoded` subchunk at `level` with the subchunk codecs of `array`.
///
/// Sharding may store nested subchunks in any order, so encoded subchunks are compared by value.
fn decode_subchunk<A: ArrayOps>(
    array: &A,
    level: usize,
    encoded: Option<EncodedSubchunk>,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let encoded = encoded.ok_or("the subchunk is not stored")?;
    let codecs = array
        .subchunk_codecs_at_level(level)
        .ok_or("no subchunk codecs")?;
    let bytes = codecs.decode(
        CowBytes::from(encoded.bytes().clone()),
        encoded.shape(),
        &CodecOptions::default(),
    )?;
    Ok(Vec::<u16>::from_array_bytes(
        bytes,
        bytemuck::must_cast_slice(encoded.shape()),
        codecs.data_type(),
    )?)
}

#[test]
fn encoded_subchunks_are_retrieved_without_decoding() -> TestResult {
    let reference = nested_sharded_array(Arc::new(MemoryStore::default()), None)?;
    let codec = Arc::new(NoPartialDecodeCodec::default());
    let array = Arc::new(nested_sharded_array(
        Arc::new(MemoryStore::default()),
        Some(codec.clone()),
    )?);
    codec.forbid_decode.store(true, Ordering::Relaxed);

    // Retrieve every subchunk with `retrieve`, and check it against the reference array
    let check = |retrieve: &Retrieve<'_>| -> TestResult {
        for (level, indices) in SUBCHUNKS {
            assert_eq!(
                decode_subchunk(&reference, level, retrieve(level, &indices)?)?,
                reference.retrieve_subchunk_at_level::<Vec<u16>>(level, &indices)?
            );
        }
        Ok(())
    };

    // The array
    check(&|level, indices| array.retrieve_encoded_subchunk_at_level(level, indices))?;

    // A cache of encoded chunks partially decodes the cached chunk without decoding it
    let cached = ArrayCached::new(array.clone(), ChunkCacheEncodedLruChunkLimit::new(2));
    check(&|level, indices| cached.retrieve_encoded_subchunk_at_level(level, indices))?;
    assert!(!cached.cache().is_empty());

    // Caches of decoded chunks and of partial decoders that cache decoded chunks bypass the cache
    let cached = ArrayCached::new(array.clone(), ChunkCacheDecodedLruChunkLimit::new(2));
    check(&|level, indices| cached.retrieve_encoded_subchunk_at_level(level, indices))?;
    assert!(cached.cache().is_empty());
    let cached = ArrayCached::new(array.clone(), ChunkCachePartialDecoderLruChunkLimit::new(2));
    check(&|level, indices| cached.retrieve_encoded_subchunk_at_level(level, indices))?;
    assert!(cached.cache().is_empty());

    // The array still decodes normally
    codec.forbid_decode.store(false, Ordering::Relaxed);
    assert_eq!(
        array.retrieve_chunk::<Vec<u16>>(&[0, 0])?,
        (0..64).collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn cached_partial_decoder_serves_encoded_subchunks_if_it_caches_no_decoded_chunk() -> TestResult {
    let array = Arc::new(nested_sharded_array(
        Arc::new(MemoryStore::default()),
        None,
    )?);
    let cached = ArrayCached::new(array.clone(), ChunkCachePartialDecoderLruChunkLimit::new(2));
    for (level, indices) in SUBCHUNKS {
        assert_eq!(
            cached.retrieve_encoded_subchunk_at_level(level, &indices)?,
            array.retrieve_encoded_subchunk_at_level(level, &indices)?
        );
    }
    // The partial decoder, holding the shard index, is cached
    assert!(!cached.cache().is_empty());
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_encoded_subchunks_are_retrieved_without_decoding() -> TestResult {
    use zarrs::array::chunk_cache::{
        AsyncChunkCache, AsyncChunkCacheDecodedLruChunkLimit, AsyncChunkCacheEncodedLruChunkLimit,
        AsyncChunkCachePartialDecoderLruChunkLimit,
    };
    use zarrs::storage::store::AsyncMemoryStore;

    let reference = nested_sharded_array(Arc::new(MemoryStore::default()), None)?;
    let codec = Arc::new(NoPartialDecodeCodec::default());
    let array = Arc::new(
        nested_sharded_builder(Some(codec.clone()))
            .build(Arc::new(AsyncMemoryStore::default()), "/array")?,
    );
    array
        .async_store_chunk(&[0, 0], (0..64u16).collect::<Vec<_>>())
        .await?;
    codec.forbid_decode.store(true, Ordering::Relaxed);

    let encoded_cache =
        ArrayCached::new(array.clone(), AsyncChunkCacheEncodedLruChunkLimit::new(2));
    let decoded_cache =
        ArrayCached::new(array.clone(), AsyncChunkCacheDecodedLruChunkLimit::new(2));
    let partial_decoder_cache = ArrayCached::new(
        array.clone(),
        AsyncChunkCachePartialDecoderLruChunkLimit::new(2),
    );
    for (level, indices) in SUBCHUNKS {
        let expected = reference.retrieve_subchunk_at_level::<Vec<u16>>(level, &indices)?;
        for encoded in [
            array
                .async_retrieve_encoded_subchunk_at_level(level, &indices)
                .await?,
            encoded_cache
                .async_retrieve_encoded_subchunk_at_level(level, &indices)
                .await?,
            decoded_cache
                .async_retrieve_encoded_subchunk_at_level(level, &indices)
                .await?,
            partial_decoder_cache
                .async_retrieve_encoded_subchunk_at_level(level, &indices)
                .await?,
        ] {
            assert_eq!(decode_subchunk(&reference, level, encoded)?, expected);
        }
    }
    assert!(!encoded_cache.cache().is_empty().await);
    assert!(decoded_cache.cache().is_empty().await);
    assert!(partial_decoder_cache.cache().is_empty().await);
    Ok(())
}
