#![allow(missing_docs)]
//! Encoded subchunks are retrieved without decoding a chunk, even if the codecs of an array cache a decoded chunk for
//! partial decoding.

mod identity_codec;

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use identity_codec::IdentityCodec;
use zarrs::array::chunk_cache::{
    ChunkCache, ChunkCacheDecodedLruChunkLimit, ChunkCacheEncodedLruChunkLimit,
    ChunkCachePartialDecoderLruChunkLimit,
};
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::{
    Array, ArrayBuilder, ArrayCached, ArrayError, ArrayOps, CodecOptions, CowBytes, FromArrayBytes,
    data_type,
};
use zarrs::storage::ReadableWritableStorageTraits;
use zarrs::storage::store::MemoryStore;
use zarrs_codec::EncodedSubchunk;

type TestResult = Result<(), Box<dyn Error>>;

fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}

/// The level zero and level one subchunks retrieved by the tests.
const SUBCHUNKS: [(usize, [u64; 2]); 4] = [(0, [0, 0]), (0, [1, 0]), (1, [0, 1]), (1, [3, 2])];

/// An 8x8 single chunk array of nested shards (4x4 then 2x2 subchunks), optionally behind the identity
/// [`IdentityCodec`], which cannot partially decode.
fn nested_sharded_builder(codec: Option<Arc<IdentityCodec>>) -> ArrayBuilder {
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
    codec: Option<Arc<IdentityCodec>>,
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
    let codec = Arc::new(IdentityCodec::new(false));
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
    let codec = Arc::new(IdentityCodec::new(false));
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
