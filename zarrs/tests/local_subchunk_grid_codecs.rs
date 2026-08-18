#![allow(missing_docs)]

//! Chunk-local subchunk grid propagation through codec partial decoders/encoders.

mod subchunk_grid_cases;

use std::error::Error;
use std::sync::Arc;

use subchunk_grid_cases::{Case, cases, nz, plain_sharding};
use zarrs::array::chunk_cache::{
    ChunkCache, ChunkCacheDecodedLruChunkLimit, ChunkCacheEncodedLruChunkLimit,
    ChunkCachePartialDecoderLruChunkLimit,
};
use zarrs::array::codec::array_to_bytes::sharding::{ShardingCodecBound, ShardingPartialDecoder};
use zarrs::array::{
    Array, ArrayBuilder, ArrayCached, ArrayPartialDecoderSubchunkingTraits,
    ArrayPartialEncoderTraits, CodecOptions, data_type,
};
use zarrs::storage::StorageHandle;
use zarrs::storage::store::MemoryStore;

type TestResult = Result<(), Box<dyn Error>>;

/// Build the array for `case` in a fresh store, with the chunk under test written.
fn build(case: &Case) -> Result<Arc<Array<MemoryStore>>, Box<dyn Error>> {
    let store = Arc::new(MemoryStore::default());
    let array = case.builder().build_arc(store, "/array")?;
    array.store_chunk(&case.chunk_indices, case.zero_chunk_bytes())?;
    Ok(array)
}

/// Build `case` with the default, non-subchunking array-to-bytes codec.
fn build_without_sharding(case: &Case) -> Result<Arc<Array<MemoryStore>>, Box<dyn Error>> {
    let store = Arc::new(MemoryStore::default());
    Ok(case.builder_without_sharding().build_arc(store, "/array")?)
}

fn sharding_partial_decoder(
    array: &Array<MemoryStore>,
    chunk_indices: &[u64],
) -> Result<ShardingPartialDecoder, Box<dyn Error>> {
    let codecs_bound = array.codecs_bound();
    let sharding_codec = codecs_bound
        .array_to_bytes_codec()
        .as_any()
        .downcast_ref::<ShardingCodecBound>()
        .ok_or("array-to-bytes codec is not sharding")?;

    let mut encoded_shape = array.chunk_shape(chunk_indices)?;
    for codec in codecs_bound.array_to_array_codecs() {
        encoded_shape = codec.encoded_shape(&encoded_shape)?;
    }

    let storage_handle = Arc::new(StorageHandle::new(array.storage()));
    let storage_transformer = array
        .storage_transformers()
        .create_readable_transformer(storage_handle)?;
    let input_handle = Arc::new((storage_transformer, array.chunk_key(chunk_indices)));

    Ok(ShardingPartialDecoder::new(
        input_handle,
        encoded_shape,
        sharding_codec.subchunk_shape().clone(),
        sharding_codec.inner_codecs().clone(),
        sharding_codec.index_codecs(),
        sharding_codec.index_location(),
        array.codec_options(),
        sharding_codec.options().clone(),
    )?)
}

#[test]
fn local_subchunk_grid_propagates_through_partial_decoders() -> TestResult {
    for case in cases() {
        let array = build(&case)?;
        let grid = array.local_subchunk_grid(&case.chunk_indices)?;
        case.assert_local_grid(grid.as_ref());

        // There is only one level of subchunking, so the next level down is absent.
        assert!(
            array
                .local_subchunk_grid_at_level(1, &case.chunk_indices)?
                .is_none(),
            "{}: expected no second subchunk level",
            case.name
        );

        // The partial decoder under test also has to report the decoded data type, and whether
        // the chunk it was created for exists.
        let written = array.partial_decoder(&case.chunk_indices)?;
        assert_eq!(written.data_type(), &case.data_type, "{}", case.name);
        assert!(written.exists()?, "{}: written chunk exists", case.name);
        let codecs = written.subchunk_codecs();
        assert_eq!(codecs.len(), 1, "{}: one subchunk codec", case.name);
        assert_eq!(
            codecs[0].data_type(),
            &case.encoded_data_type,
            "{}",
            case.name
        );

        let unwritten = array.partial_decoder(&case.absent_chunk_indices)?;
        assert_eq!(unwritten.data_type(), &case.data_type, "{}", case.name);
        assert!(
            !unwritten.exists()?,
            "{}: unwritten chunk does not exist",
            case.name
        );
    }
    Ok(())
}

#[test]
fn rearranging_codecs_do_not_create_subchunks() -> TestResult {
    for case in cases().into_iter().filter(|case| case.codec.is_some()) {
        let array = build_without_sharding(&case)?;
        let decoder = array.partial_decoder(&case.chunk_indices)?;
        assert!(
            decoder
                .local_subchunk_grids(&CodecOptions::default())?
                .is_empty(),
            "{}: no subchunk grids without sharding",
            case.name
        );
        assert!(
            decoder.subchunk_codecs().is_empty(),
            "{}: no subchunk codecs without sharding",
            case.name
        );
    }
    Ok(())
}

#[test]
fn transpose_preserves_subchunk_codec_encoded_domain() -> TestResult {
    let case = cases()
        .into_iter()
        .find(|case| case.name == "transpose")
        .unwrap();
    let array = case
        .builder()
        .build_arc(Arc::new(MemoryStore::default()), "/array")?;
    let values = (0..case.chunk_shape.iter().product::<u64>())
        .map(|value| u16::try_from(value).unwrap())
        .collect::<Vec<_>>();
    array.store_chunk(&case.chunk_indices, &values)?;

    let partial_decoder = array.partial_decoder(&case.chunk_indices)?;
    let codec = partial_decoder.subchunk_codecs().into_iter().next().unwrap();
    let sharding_decoder = sharding_partial_decoder(&array, &case.chunk_indices)?;
    let encoded = sharding_decoder
        .retrieve_encoded_subchunk(&[0, 0], &CodecOptions::default())?
        .unwrap();
    let decoded_bytes = codec
        .decode(
            encoded,
            &case.encoded_subchunk_shape,
            &CodecOptions::default(),
        )?
        .into_fixed()?;
    let decoded = decoded_bytes
        .chunks_exact(size_of::<u16>())
        .map(|bytes| u16::from_ne_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();

    assert_eq!(case.encoded_subchunk_shape, [nz(3), nz(2)]);
    assert_eq!(case.local_subchunk_shape, [nz(2), nz(3)]);
    assert_eq!(decoded, [0, 6, 1, 7, 2, 8]);
    Ok(())
}

#[test]
fn local_subchunk_grid_exposed_by_partial_encoders() -> TestResult {
    let case = plain_sharding();
    let array = build(&case)?;
    let cached = ArrayCached::new(array.clone(), ChunkCachePartialDecoderLruChunkLimit::new(1));

    let encoders: [(&str, Arc<dyn ArrayPartialEncoderTraits>); 2] = [
        ("sharding", array.partial_encoder(&case.chunk_indices)?),
        ("cached", cached.partial_encoder(&case.chunk_indices)?),
    ];
    for (name, encoder) in encoders {
        let grids = encoder.local_subchunk_grids(&CodecOptions::default())?;
        assert_eq!(grids.len(), 1, "{name}: one subchunk level");
        case.assert_local_grid(grids[0].as_ref());
        case.assert_local_grid(
            encoder
                .local_subchunk_grid(&CodecOptions::default())?
                .as_ref(),
        );
    }
    Ok(())
}

/// A cache only exposes subchunks if it caches something that can still resolve them.
fn check_cache<C: ChunkCache + 'static>(
    case: &Case,
    array: &Arc<Array<MemoryStore>>,
    cache: C,
    expect_subchunks: bool,
) -> TestResult {
    let cached = ArrayCached::new(array.clone(), cache);
    let grid = cached.local_subchunk_grid(&case.chunk_indices)?;
    if expect_subchunks {
        case.assert_local_grid(grid.as_ref());
    } else {
        assert!(grid.is_none());
    }
    Ok(())
}

#[test]
fn local_subchunk_grid_via_chunk_caches() -> TestResult {
    let case = plain_sharding();
    let array = build(&case)?;

    // A partial decoder cache holds the decoder itself, and an encoded cache builds one over the
    // cached encoded chunk, so both keep subchunks visible. A decoded cache holds decoded chunk
    // bytes, which cannot resolve subchunks.
    check_cache(
        &case,
        &array,
        ChunkCachePartialDecoderLruChunkLimit::new(1),
        true,
    )?;
    check_cache(&case, &array, ChunkCacheEncodedLruChunkLimit::new(1), true)?;
    check_cache(&case, &array, ChunkCacheDecodedLruChunkLimit::new(1), false)
}

#[test]
fn local_subchunk_grid_absent_with_array_partial_decoder_cache() -> TestResult {
    // `vlen-utf8` does not support partial decoding, so the codec chain inserts an
    // `ArrayPartialDecoderCache` at the top of the partial decoder chain.
    let store = Arc::new(MemoryStore::default());
    let array =
        ArrayBuilder::new(vec![4], vec![4], data_type::string(), "").build(store, "/array")?;
    array.store_chunk(&[0], vec!["a", "bb", "ccc", "dddd"])?;

    let partial_decoder = array.partial_decoder(&[0])?;
    assert!(partial_decoder.supports_partial_decode());
    assert!(
        partial_decoder
            .local_subchunk_grids(&CodecOptions::default())?
            .is_empty()
    );
    assert!(array.local_subchunk_grid(&[0])?.is_none());
    Ok(())
}
