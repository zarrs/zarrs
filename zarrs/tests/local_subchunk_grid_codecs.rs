#![allow(missing_docs)]

//! Chunk-local subchunk grid propagation through codec partial decoders/encoders.

mod subchunk_grid_cases;

use std::error::Error;
use std::sync::Arc;

use subchunk_grid_cases::{Case, cases, plain_sharding};
use zarrs::array::chunk_cache::{
    ChunkCache, ChunkCacheDecodedLruChunkLimit, ChunkCacheEncodedLruChunkLimit,
    ChunkCachePartialDecoderLruChunkLimit,
};
use zarrs::array::{
    Array, ArrayBuilder, ArrayCached, ArrayPartialEncoderTraits, CodecOptions, data_type,
};
use zarrs::storage::store::MemoryStore;

type TestResult = Result<(), Box<dyn Error>>;

/// Build the array for `case` in a fresh store, with the chunk under test written.
fn build(case: &Case) -> Result<Arc<Array<MemoryStore>>, Box<dyn Error>> {
    let store = Arc::new(MemoryStore::default());
    let array = case.builder().build_arc(store, "/array")?;
    array.store_chunk(&case.chunk_indices, case.zero_chunk_bytes())?;
    Ok(array)
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
