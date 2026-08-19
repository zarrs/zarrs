#![allow(missing_docs)]
#![cfg(feature = "async")]

//! Asynchronous variant of `local_subchunk_grid_codecs.rs`, over the same scenarios.

mod subchunk_grid_cases;

use std::error::Error;
use std::sync::Arc;

use subchunk_grid_cases::{Case, cases, plain_sharding};
use zarrs::array::chunk_cache::{
    AsyncChunkCache, AsyncChunkCacheDecodedLruChunkLimit, AsyncChunkCacheEncodedLruChunkLimit,
    AsyncChunkCachePartialDecoderLruChunkLimit,
};
use zarrs::array::{
    Array, ArrayBuilder, ArrayCached, AsyncArrayPartialEncoderTraits, CodecOptions, data_type,
};
use zarrs_storage::store::AsyncMemoryStore;

type TestResult = Result<(), Box<dyn Error>>;

/// Build the array for `case` in a fresh store, with the chunk under test written.
async fn build(case: &Case) -> Result<Arc<Array<AsyncMemoryStore>>, Box<dyn Error>> {
    let store = Arc::new(AsyncMemoryStore::new());
    let array = case.builder().build_arc(store, "/array")?;
    array
        .async_store_chunk(&case.chunk_indices, case.zero_chunk_bytes())
        .await?;
    Ok(array)
}

/// Build `case` with the default, non-subchunking array-to-bytes codec.
fn build_without_sharding(case: &Case) -> Result<Arc<Array<AsyncMemoryStore>>, Box<dyn Error>> {
    let store = Arc::new(AsyncMemoryStore::new());
    Ok(case.builder_without_sharding().build_arc(store, "/array")?)
}

#[tokio::test]
async fn async_local_subchunk_grid_propagates_through_partial_decoders() -> TestResult {
    for case in cases() {
        let array = build(&case).await?;
        let grid = array.async_local_subchunk_grid(&case.chunk_indices).await?;
        case.assert_local_grid(grid.as_ref());

        // There is only one level of subchunking, so the next level down is absent.
        assert!(
            array
                .async_local_subchunk_grid_at_level(1, &case.chunk_indices)
                .await?
                .is_none(),
            "{}: expected no second subchunk level",
            case.name
        );

        // The partial decoder under test also has to report the decoded data type and existence.
        let partial_decoder = array.async_partial_decoder(&case.chunk_indices).await?;
        assert_eq!(
            partial_decoder.data_type(),
            &case.data_type,
            "{}",
            case.name
        );
        assert!(partial_decoder.exists().await?, "{}", case.name);
        let codecs = partial_decoder.subchunk_codecs();
        assert_eq!(codecs.len(), 1, "{}: one subchunk codec", case.name);
        assert_eq!(
            codecs[0].data_type(),
            &case.encoded_data_type,
            "{}",
            case.name
        );
    }
    Ok(())
}

#[tokio::test]
async fn async_rearranging_codecs_do_not_create_subchunks() -> TestResult {
    for case in cases().into_iter().filter(|case| case.codec.is_some()) {
        let array = build_without_sharding(&case)?;
        let decoder = array.async_partial_decoder(&case.chunk_indices).await?;
        assert!(
            decoder
                .local_subchunk_grids(&CodecOptions::default())
                .await?
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

#[tokio::test]
async fn async_local_subchunk_grid_exposed_by_partial_encoders() -> TestResult {
    let case = plain_sharding();
    let array = build(&case).await?;
    let cached = ArrayCached::new(
        array.clone(),
        AsyncChunkCachePartialDecoderLruChunkLimit::new(1),
    );

    let encoders: [(&str, Arc<dyn AsyncArrayPartialEncoderTraits>); 2] = [
        (
            "sharding",
            array.async_partial_encoder(&case.chunk_indices).await?,
        ),
        (
            "cached",
            cached.async_partial_encoder(&case.chunk_indices).await?,
        ),
    ];
    for (name, encoder) in encoders {
        let grids = encoder
            .local_subchunk_grids(&CodecOptions::default())
            .await?;
        assert_eq!(grids.len(), 1, "{name}: one subchunk level");
        case.assert_local_grid(grids[0].as_ref());
        case.assert_local_grid(
            encoder
                .local_subchunk_grid(&CodecOptions::default())
                .await?
                .as_ref(),
        );
    }
    Ok(())
}

/// A cache only exposes subchunks if it caches something that can still resolve them.
async fn check_cache<C: AsyncChunkCache + 'static>(
    case: &Case,
    array: &Arc<Array<AsyncMemoryStore>>,
    cache: C,
    expect_subchunks: bool,
) -> TestResult {
    let cached = ArrayCached::new(array.clone(), cache);
    let grid = cached
        .async_local_subchunk_grid(&case.chunk_indices)
        .await?;
    if expect_subchunks {
        case.assert_local_grid(grid.as_ref());
    } else {
        assert!(grid.is_none());
    }
    Ok(())
}

#[tokio::test]
async fn async_local_subchunk_grid_via_chunk_caches() -> TestResult {
    let case = plain_sharding();
    let array = build(&case).await?;

    // A partial decoder cache holds the decoder itself, and an encoded cache wraps a synchronous
    // decoder over the cached encoded chunk, so both keep subchunks visible. A decoded cache holds
    // decoded chunk bytes, which cannot resolve subchunks.
    check_cache(
        &case,
        &array,
        AsyncChunkCachePartialDecoderLruChunkLimit::new(1),
        true,
    )
    .await?;
    check_cache(
        &case,
        &array,
        AsyncChunkCacheEncodedLruChunkLimit::new(1),
        true,
    )
    .await?;
    check_cache(
        &case,
        &array,
        AsyncChunkCacheDecodedLruChunkLimit::new(1),
        false,
    )
    .await
}

#[tokio::test]
async fn async_local_subchunk_grid_absent_with_array_partial_decoder_cache() -> TestResult {
    // `vlen-utf8` does not support partial decoding, so the codec chain inserts an
    // `ArrayPartialDecoderCache` at the top of the partial decoder chain.
    let store = Arc::new(AsyncMemoryStore::new());
    let array =
        ArrayBuilder::new(vec![4], vec![4], data_type::string(), "").build(store, "/array")?;
    array
        .async_store_chunk(&[0], vec!["a", "bb", "ccc", "dddd"])
        .await?;

    let partial_decoder = array.async_partial_decoder(&[0]).await?;
    assert!(partial_decoder.supports_partial_decode());
    assert!(
        partial_decoder
            .local_subchunk_grids(&CodecOptions::default())
            .await?
            .is_empty()
    );
    assert!(array.async_local_subchunk_grid(&[0]).await?.is_none());
    Ok(())
}
