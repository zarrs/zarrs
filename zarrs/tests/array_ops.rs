#![allow(missing_docs)]

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::chunk_cache::{
    ChunkCache, ChunkCacheDecodedLruChunkLimit, ChunkCacheEncodedLruChunkLimit,
    ChunkCachePartialDecoderLruChunkLimit,
};
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArrayCached,
    ArrayMetadataOptions, ArrayOps, ArrayReadOps, ArraySubset, ArrayUpdateOps, ArrayWriteOps,
    CodecOptions, data_type,
};
use zarrs::config::MetadataEraseVersion;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;

type TestStore = PerformanceMetricsStorageAdapter<MemoryStore>;
type TestResult = Result<(), Box<dyn Error>>;

fn fixture() -> (Arc<Array<TestStore>>, Arc<TestStore>) {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::default(),
    )));
    let mut builder = ArrayBuilder::new(vec![5, 5], vec![3, 3], data_type::uint8(), 0u8);
    builder
        .subchunk_shape(vec![1, 1])
        .dimension_names(Some(["y", "x"]));
    builder
        .attributes_mut()
        .insert("purpose".to_string(), "array_ops_test".into());
    let array = builder.build_arc(store.clone(), "/array").unwrap();
    (array, store)
}

fn populate<A: ArrayWriteOps>(array: &A) -> TestResult {
    array.store_chunk(&[0, 0], &[1u8, 2, 3, 6, 7, 8, 11, 12, 13])?;
    array.store_chunk_opt(
        &[0, 1],
        &[4u8, 5, 0, 9, 10, 0, 14, 15, 0],
        &CodecOptions::default(),
    )?;
    array.store_chunks(
        &ArraySubset::new_with_ranges(&[1..2, 0..2]),
        &[
            16u8, 17, 18, 19, 20, 0, 21, 22, 23, 24, 25, 0, 0, 0, 0, 0, 0, 0,
        ],
    )?;
    Ok(())
}

fn exercise_array_ops<A: ArrayOps>(array: &A) -> TestResult {
    assert_eq!(array.path().as_str(), "/array");
    assert_eq!(*array.data_type(), data_type::uint8());
    assert_eq!(array.fill_value().as_ne_bytes(), &[0]);
    assert_eq!(array.shape(), &[5, 5]);
    assert_eq!(array.dimensionality(), 2);
    assert_eq!(array.chunk_grid_shape(), &[2, 2]);
    assert_eq!(array.dimension_names().as_ref().unwrap().len(), 2);
    assert_eq!(array.attributes()["purpose"], "array_ops_test");
    assert_eq!(
        array.metadata(),
        &array.metadata_opt(&ArrayMetadataOptions::default().with_include_zarrs_metadata(false))
    );
    let _ = array.builder().build_metadata()?;
    assert_eq!(
        array.subchunk_shape(),
        Some(vec![NonZeroU64::new(1).unwrap(); 2])
    );
    let subchunk_grid = array.subchunk_grid().unwrap();
    assert_eq!(subchunk_grid.grid_shape(), &[6, 6]);
    assert_eq!(array.chunk_origin(&[1, 1])?, [3, 3]);
    assert_eq!(array.chunk_shape(&[0, 0])?, vec![3u64; 2]);
    assert_eq!(
        subchunk_grid.chunk_shape(&[0, 0]).unwrap(),
        Some(vec![1u64; 2])
    );
    assert_eq!(array.chunk_shape_usize(&[0, 0])?, [3, 3]);
    assert_eq!(array.subset_all(), ArraySubset::new_with_shape(vec![5, 5]));
    assert_eq!(
        array.chunk_subset(&[1, 1])?,
        ArraySubset::new_with_ranges(&[3..6, 3..6])
    );
    assert_eq!(
        array.chunk_subset_bounded(&[1, 1])?,
        ArraySubset::new_with_ranges(&[3..5, 3..5])
    );
    let chunks = ArraySubset::new_with_ranges(&[0..2, 0..2]);
    assert_eq!(
        array.chunks_subset(&chunks)?,
        ArraySubset::new_with_ranges(&[0..6, 0..6])
    );
    assert_eq!(array.chunks_subset_bounded(&chunks)?, array.subset_all());
    assert_eq!(
        array
            .chunks_in_array_subset(&ArraySubset::new_with_ranges(&[2..5, 2..5]))?
            .unwrap(),
        chunks
    );
    assert!(array.chunk_origin(&[0]).is_err());
    assert!(array.chunk_shape(&[0]).is_err());

    // Exercise accessors whose concrete internals are intentionally opaque.
    let _ = array.storage();
    let _ = array.codecs();
    let _ = array.chunk_grid();
    let _ = array.subchunk_grid();
    let _ = array.chunk_key_encoding();
    let _ = array.storage_transformers();
    let _ = array.chunk_key(&[0, 0]);
    Ok(())
}

fn retrieve_into<A: ArrayReadOps>(
    array: &A,
    subset: &ArraySubset,
    chunk_indices: Option<&[u64]>,
    explicit_options: bool,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let shape = subset.shape().to_vec();
    let mut output = vec![0; subset.num_elements_usize()];
    {
        let output_slice = unsafe_cell_slice::UnsafeCellSlice::new(&mut output);
        let full_subset = ArraySubset::new_with_shape(shape.clone());
        let mut view = unsafe {
            // SAFETY: this is the only view over output and covers it exactly.
            ArrayBytesFixedDisjointView::new(output_slice, 1, &shape, full_subset)?
        };
        let target = ArrayBytesDecodeIntoTarget::Fixed(&mut view);
        if let Some(chunk_indices) = chunk_indices {
            array.retrieve_chunk_subset_into(
                chunk_indices,
                subset,
                target,
                &CodecOptions::default(),
            )?;
        } else if explicit_options {
            array.retrieve_array_subset_into_opt(subset, target, &CodecOptions::default())?;
        } else {
            array.retrieve_array_subset_into(subset, target)?;
        }
    }
    Ok(output)
}

fn retrieve_chunk_into<A: ArrayReadOps>(
    array: &A,
    chunk_indices: &[u64],
) -> Result<Vec<u8>, Box<dyn Error>> {
    let shape = array.chunk_shape(chunk_indices)?;
    let mut output = vec![0; shape.iter().product::<u64>() as usize];
    {
        let output_slice = unsafe_cell_slice::UnsafeCellSlice::new(&mut output);
        let full_subset = ArraySubset::new_with_shape(shape.clone());
        let mut view = unsafe {
            // SAFETY: this is the only view over output and covers it exactly.
            ArrayBytesFixedDisjointView::new(output_slice, 1, &shape, full_subset)?
        };
        array.retrieve_chunk_into(
            chunk_indices,
            ArrayBytesDecodeIntoTarget::Fixed(&mut view),
            &CodecOptions::default(),
        )?;
    }
    Ok(output)
}

fn exercise_array_read_ops<A: ArrayReadOps + ArrayWriteOps>(array: &A) -> TestResult {
    populate(array)?;
    let options = CodecOptions::default();
    let chunk_subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
    let chunks = ArraySubset::new_with_ranges(&[0..1, 0..2]);
    let array_subset = ArraySubset::new_with_ranges(&[1..4, 1..4]);

    assert_eq!(
        array.retrieve_chunk::<Vec<u8>>(&[0, 0])?,
        [1, 2, 3, 6, 7, 8, 11, 12, 13]
    );
    assert_eq!(
        array.retrieve_chunk_opt::<Vec<u8>>(&[0, 1], &options)?,
        [4, 5, 0, 9, 10, 0, 14, 15, 0]
    );
    assert_eq!(
        retrieve_chunk_into(array, &[0, 0])?,
        [1, 2, 3, 6, 7, 8, 11, 12, 13]
    );
    assert_eq!(
        array.retrieve_chunk_if_exists::<Vec<u8>>(&[1, 0])?,
        Some(vec![16, 17, 18, 21, 22, 23, 0, 0, 0])
    );
    array.erase_chunk(&[1, 1])?;
    assert_eq!(
        array.retrieve_chunk_if_exists_opt::<Vec<u8>>(&[1, 1], &options)?,
        None
    );
    assert_eq!(
        array.retrieve_chunk_subset::<Vec<u8>>(&[0, 0], &chunk_subset)?,
        [7, 8, 12, 13]
    );
    assert_eq!(
        array.retrieve_chunk_subset_opt::<Vec<u8>>(&[0, 0], &chunk_subset, &options)?,
        [7, 8, 12, 13]
    );
    assert_eq!(
        retrieve_into(array, &chunk_subset, Some(&[0, 0]), true)?,
        [7, 8, 12, 13]
    );
    assert_eq!(
        array.retrieve_chunks::<Vec<u8>>(&chunks)?,
        [1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 15, 0]
    );
    assert_eq!(
        array.retrieve_chunks_opt::<Vec<u8>>(&chunks, &options)?,
        [1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 15, 0]
    );
    assert_eq!(
        array.retrieve_array_subset::<Vec<u8>>(&array_subset)?,
        [7, 8, 9, 12, 13, 14, 17, 18, 0]
    );
    assert_eq!(
        array.retrieve_array_subset_opt::<Vec<u8>>(&array_subset, &options)?,
        [7, 8, 9, 12, 13, 14, 17, 18, 0]
    );
    assert_eq!(
        retrieve_into(array, &array_subset, None, false)?,
        [7, 8, 9, 12, 13, 14, 17, 18, 0]
    );
    assert_eq!(
        retrieve_into(array, &array_subset, None, true)?,
        [7, 8, 9, 12, 13, 14, 17, 18, 0]
    );
    assert_eq!(
        array.retrieve_subchunk_opt::<Vec<u8>>(&[1, 1], &options)?,
        [7]
    );
    assert_eq!(
        array.retrieve_subchunks_opt::<Vec<u8>>(
            &ArraySubset::new_with_ranges(&[1..3, 1..3]),
            &options,
        )?,
        [7, 8, 12, 13]
    );
    assert!(array.retrieve_encoded_chunk(&[0, 0])?.is_some());
    assert_eq!(
        array.retrieve_encoded_chunks_opt(&chunks, &options)?.len(),
        chunks.num_elements_usize()
    );
    assert!(array.retrieve_encoded_subchunk(&[1, 1])?.is_some());
    let decoder = array.partial_decoder(&[0, 0])?;
    assert!(decoder.exists()?);
    assert_eq!(
        decoder
            .partial_decode(&chunk_subset, &options)?
            .into_fixed()?
            .as_ref(),
        &[7, 8, 12, 13]
    );
    assert!(array.partial_decoder_opt(&[0, 0], &options)?.exists()?);
    Ok(())
}

fn exercise_array_write_update_ops<A: ArrayUpdateOps>(array: &A) -> TestResult {
    let options = CodecOptions::default();
    array.store_metadata()?;
    array.store_metadata_opt(&ArrayMetadataOptions::default())?;
    array.erase_metadata()?;
    array.store_metadata()?;
    array.erase_metadata_opt(MetadataEraseVersion::All)?;
    array.store_metadata()?;

    array.store_chunk(&[0, 0], &[1u8; 9])?;
    array.store_chunk_opt(&[0, 1], &[2u8; 9], &options)?;
    array.store_chunks_opt(
        &ArraySubset::new_with_ranges(&[1..2, 0..2]),
        &[3u8; 18],
        &options,
    )?;
    assert_eq!(array.retrieve_chunk::<Vec<u8>>(&[1, 1])?, [3u8; 9]);

    array.store_chunk_subset(
        &[0, 0],
        &ArraySubset::new_with_ranges(&[1..2, 1..3]),
        &[4u8, 5],
    )?;
    array.store_chunk_subset_opt(
        &[0, 0],
        &ArraySubset::new_with_ranges(&[2..3, 0..1]),
        &[6u8],
        &options,
    )?;
    array.store_array_subset(&ArraySubset::new_with_ranges(&[0..1, 0..2]), &[7u8, 8])?;
    array.store_array_subset_opt(
        &ArraySubset::new_with_ranges(&[4..5, 4..5]),
        &[9u8],
        &options,
    )?;
    assert_eq!(
        array.retrieve_chunk::<Vec<u8>>(&[0, 0])?,
        [7, 8, 1, 1, 4, 5, 6, 1, 1]
    );
    assert_eq!(
        array.readable().retrieve_chunk::<Vec<u8>>(&[1, 1])?,
        [3, 3, 3, 3, 9, 3, 3, 3, 3]
    );

    let encoder = array.partial_encoder(&[0, 0], &options)?;
    assert!(encoder.supports_partial_encode());
    encoder.partial_encode(
        &ArraySubset::new_with_ranges(&[0..1, 2..3]),
        &vec![10u8].into(),
        &options,
    )?;
    assert_eq!(array.retrieve_chunk::<Vec<u8>>(&[0, 0])?[2], 10);

    let encoded = array.retrieve_encoded_chunk(&[0, 0])?.unwrap();
    unsafe {
        // SAFETY: bytes were produced by the same array and chunk configuration.
        array.store_encoded_chunk(&[0, 1], encoded.into())?;
    }
    assert_eq!(
        array.retrieve_chunk::<Vec<u8>>(&[0, 1])?,
        [7, 8, 10, 1, 4, 5, 6, 1, 1]
    );

    let _ = array.compact_chunk(&[0, 0], &options)?;
    array.erase_chunk(&[0, 1])?;
    assert!(
        array
            .retrieve_chunk_if_exists::<Vec<u8>>(&[0, 1])?
            .is_none()
    );
    array.erase_chunks(&ArraySubset::new_with_ranges(&[1..2, 0..2]))?;
    assert!(
        array
            .retrieve_chunk_if_exists::<Vec<u8>>(&[1, 0])?
            .is_none()
    );
    Ok(())
}

fn exercise_all<A: ArrayUpdateOps>(array: &A) -> TestResult {
    exercise_array_ops(array)?;
    exercise_array_read_ops(array)?;
    exercise_array_write_update_ops(array)
}

fn exercise_variable_length_ops<A: ArrayUpdateOps>(array: &A) -> TestResult {
    let first_chunk = ["a", "bb", "ccc", "dddd"];
    let second_chunk = ["eeeee", "ffffff", "ggggggg", "hhhhhhhh"];
    array.store_chunk(&[0, 0], &first_chunk)?;
    array.store_chunk(&[0, 1], &second_chunk)?;

    assert_eq!(array.retrieve_chunk::<Vec<String>>(&[0, 0])?, first_chunk);
    assert_eq!(
        array.retrieve_array_subset::<Vec<String>>(&ArraySubset::new_with_ranges(&[1..3, 1..3]))?,
        ["dddd", "ggggggg", "", ""]
    );

    array.store_array_subset(
        &ArraySubset::new_with_ranges(&[1..3, 1..3]),
        &["updated", "values", "across", "chunks"],
    )?;
    assert_eq!(
        array.retrieve_array_subset::<Vec<String>>(&ArraySubset::new_with_ranges(&[1..3, 1..3]))?,
        ["updated", "values", "across", "chunks"]
    );
    Ok(())
}

fn exercise_optional_ops<A: ArrayUpdateOps>(array: &A) -> TestResult {
    let first_chunk = [Some(1u8), None, Some(3), Some(4)];
    let second_chunk = [None, Some(6u8), None, Some(8)];
    array.store_chunk(&[0, 0], &first_chunk)?;
    array.store_chunk(&[0, 1], &second_chunk)?;

    assert_eq!(
        array.retrieve_chunk::<Vec<Option<u8>>>(&[0, 0])?,
        first_chunk
    );
    assert_eq!(
        array.retrieve_array_subset::<Vec<Option<u8>>>(&ArraySubset::new_with_ranges(&[
            0..2,
            1..3
        ]))?,
        [None, None, Some(4), None]
    );

    array.store_array_subset(
        &ArraySubset::new_with_ranges(&[0..2, 1..3]),
        &[Some(9u8), None, None, Some(10)],
    )?;
    assert_eq!(
        array.retrieve_array_subset::<Vec<Option<u8>>>(&ArraySubset::new_with_ranges(&[
            0..2,
            1..3
        ]))?,
        [Some(9), None, None, Some(10)]
    );
    Ok(())
}

#[test]
fn array_implements_complete_sync_operation_suite() -> TestResult {
    let (array, _) = fixture();
    exercise_all(array.as_ref())
}

#[test]
fn array_cached_implements_complete_sync_operation_suite() -> TestResult {
    let (array, _) = fixture();
    let cached = ArrayCached::new(array, ChunkCacheDecodedLruChunkLimit::new(16));
    exercise_all(&cached)
}

#[test]
fn array_and_cached_support_variable_length_operation_suite() -> TestResult {
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::string(), "")
        .build_arc(store, "/array")?;
    exercise_variable_length_ops(array.as_ref())?;

    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::string(), "")
        .build_arc(store, "/array")?;
    let cached = ArrayCached::new(array, ChunkCacheDecodedLruChunkLimit::new(4));
    exercise_variable_length_ops(&cached)
}

#[test]
fn array_and_cached_support_optional_operation_suite() -> TestResult {
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(
        vec![4, 4],
        vec![2, 2],
        data_type::uint8().to_optional(),
        zarrs::array::FillValue::from(None::<u8>),
    )
    .build_arc(store, "/array")?;
    exercise_optional_ops(array.as_ref())?;

    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(
        vec![4, 4],
        vec![2, 2],
        data_type::uint8().to_optional(),
        zarrs::array::FillValue::from(None::<u8>),
    )
    .build_arc(store, "/array")?;
    let cached = ArrayCached::new(array, ChunkCacheDecodedLruChunkLimit::new(4));
    exercise_optional_ops(&cached)
}

fn assert_cache_hits_and_invalidation<C>(cache: C, caches_full_chunk_reads: bool) -> TestResult
where
    C: ChunkCache + 'static,
{
    let (array, store) = fixture();
    array.store_chunk(&[0, 0], &[1u8; 9])?;
    array.store_chunk(&[0, 1], &[2u8; 9])?;
    let cached = ArrayCached::new(array, cache);

    store.reset();
    assert_eq!(cached.retrieve_chunk::<Vec<u8>>(&[0, 0])?, [1u8; 9]);
    let reads_after_miss = store.reads();
    assert!(reads_after_miss > 0);
    assert_eq!(cached.retrieve_chunk::<Vec<u8>>(&[0, 0])?, [1u8; 9]);
    if caches_full_chunk_reads {
        assert_eq!(store.reads(), reads_after_miss);
    } else {
        assert!(store.reads() > reads_after_miss);
    }

    assert_eq!(cached.retrieve_chunk::<Vec<u8>>(&[0, 1])?, [2u8; 9]);
    assert_eq!(cached.cache().len(), 2);
    cached.store_chunk(&[0, 0], &[3u8; 9])?;
    assert_eq!(cached.cache().len(), 1);
    assert_eq!(cached.retrieve_chunk::<Vec<u8>>(&[0, 0])?, [3u8; 9]);

    cached.store_array_subset(&ArraySubset::new_with_ranges(&[0..1, 3..4]), &[4u8])?;
    assert_eq!(cached.cache().len(), 1);
    assert_eq!(cached.retrieve_chunk::<Vec<u8>>(&[0, 1])?[0], 4);

    cached.store_metadata()?;
    assert!(cached.cache().is_empty());
    Ok(())
}

#[test]
fn array_cached_cache_hits_and_invalidation_cover_all_value_types() -> TestResult {
    assert_cache_hits_and_invalidation(ChunkCacheEncodedLruChunkLimit::new(16), true)?;
    assert_cache_hits_and_invalidation(ChunkCacheDecodedLruChunkLimit::new(16), true)?;
    assert_cache_hits_and_invalidation(ChunkCachePartialDecoderLruChunkLimit::new(16), false)
}

fn assert_cache_hits_for_array_subset_into<C>(cache: C) -> TestResult
where
    C: ChunkCache + 'static,
{
    let (array, store) = fixture();
    populate(array.as_ref())?;
    let cached = ArrayCached::new(array, cache);
    let subset = ArraySubset::new_with_ranges(&[1..4, 1..4]);

    store.reset();
    assert_eq!(
        retrieve_into(&cached, &subset, None, true)?,
        [7, 8, 9, 12, 13, 14, 17, 18, 19]
    );
    let reads_after_miss = store.reads();
    assert!(reads_after_miss > 0);
    assert_eq!(
        retrieve_into(&cached, &subset, None, true)?,
        [7, 8, 9, 12, 13, 14, 17, 18, 19]
    );
    assert_eq!(store.reads(), reads_after_miss);
    Ok(())
}

#[test]
fn array_cached_array_subset_into_uses_cache() -> TestResult {
    assert_cache_hits_for_array_subset_into(ChunkCacheEncodedLruChunkLimit::new(16))?;
    assert_cache_hits_for_array_subset_into(ChunkCacheDecodedLruChunkLimit::new(16))
}

// TODO: ChunkCacheType could be adjusted so that ChunkCacheEncoded could handle caching of encoded chunks, but seems low value
#[test]
fn array_cached_encoded_reads_bypass_cache() -> TestResult {
    let (array, store) = fixture();
    array.store_chunk(&[0, 0], &[1u8; 9])?;
    let cached = ArrayCached::new(array, ChunkCacheDecodedLruChunkLimit::new(4));

    store.reset();
    assert!(cached.retrieve_encoded_chunk(&[0, 0])?.is_some());
    let reads = store.reads();
    assert!(reads > 0);
    assert!(cached.cache().is_empty());
    assert!(cached.retrieve_encoded_chunk(&[0, 0])?.is_some());
    assert!(store.reads() > reads);
    assert!(cached.cache().is_empty());
    Ok(())
}
