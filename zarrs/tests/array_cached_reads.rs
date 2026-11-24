#![allow(missing_docs)]

//! Baseline tests for cached reads across cache types, out-of-bounds subset validation, and
//! non-subset custom indexers.

use std::fmt::Debug;
use std::sync::Arc;

use zarrs::array::chunk_cache::{
    ChunkCache, ChunkCacheDecodedLruChunkLimit, ChunkCacheEncodedLruChunkLimit,
    ChunkCachePartialDecoderLruChunkLimit, SyncChunkCacheType,
};
use zarrs::array::{
    Array, ArrayBuilder, ArrayCached, ArrayError, ArrayIndices, ArraySubset, DataType, Element,
    ElementOwned, FillValue, Indexer, IndexerError, data_type,
};
use zarrs_chunk_grid::IndexerIterator;
use zarrs_codec::Resources;
use zarrs_storage::store::MemoryStore;

const SHAPE: [u64; 2] = [7, 9];
const CHUNK_SHAPE: [u64; 2] = [4, 4];

fn builder(data_type: DataType, fill_value: impl Into<FillValue>) -> ArrayBuilder {
    ArrayBuilder::new(
        SHAPE.to_vec(),
        CHUNK_SHAPE.to_vec(),
        data_type,
        fill_value.into(),
    )
}

fn build(builder: &ArrayBuilder) -> Array<MemoryStore> {
    builder
        .build(Arc::new(MemoryStore::default()), "/array")
        .unwrap()
}

fn u16_builders() -> Vec<(&'static str, ArrayBuilder)> {
    #[allow(unused_mut)]
    let mut builders = vec![("uncompressed", builder(data_type::uint16(), 0u16))];

    #[cfg(feature = "zstd")]
    {
        let mut b = builder(data_type::uint16(), 0u16);
        b.bytes_to_bytes_codecs(vec![Arc::new(zarrs::array::codec::ZstdCodec::new(
            1, false,
        ))]);
        builders.push(("zstd", b));
    }

    #[cfg(feature = "transpose")]
    {
        let mut b = builder(data_type::uint16(), 0u16);
        b.array_to_array_codecs(vec![Arc::new(zarrs::array::codec::TransposeCodec::new(
            zarrs::array::codec::TransposeOrder::new(&[1, 0]).unwrap(),
        ))]);
        builders.push(("transpose", b));
    }

    #[cfg(feature = "sharding")]
    {
        let mut b = builder(data_type::uint16(), 0u16);
        b.subchunk_shape(vec![2, 2]);
        builders.push(("sharded", b));
    }

    builders
}

fn check_cached_reads<T, C>(
    cache_name: &str,
    name: &str,
    array: &Array<MemoryStore>,
    cache: C,
    elements: &[T],
) where
    T: Element + ElementOwned + Clone + PartialEq + Debug,
    C: ChunkCache,
    C::Value: SyncChunkCacheType,
{
    let resources = Resources::default();
    let all = ArraySubset::new_with_shape(SHAPE);
    array
        .store_array_subset(&all, elements.to_vec(), &resources)
        .unwrap();
    let cached = ArrayCached::new(array.clone(), cache);

    for subset in [
        all.clone(),
        ArraySubset::new_with_ranges(&[1..6, 2..8]),
        ArraySubset::new_with_ranges(&[4..8, 4..8]),
        ArraySubset::new_with_ranges(&[1..3, 5..8]),
    ] {
        let expected: Vec<T> = array.retrieve_array_subset(&subset, &resources).unwrap();
        // Retrieve twice to cover both cache fill and cache hit paths.
        for pass in 0..2 {
            let actual: Vec<T> = cached.retrieve_array_subset(&subset, &resources).unwrap();
            assert_eq!(actual, expected, "{cache_name} {name} pass={pass} {subset}");
        }
    }

    for chunk_indices in [[0, 0], [1, 2]] {
        let expected: Vec<T> = array.retrieve_chunk(&chunk_indices, &resources).unwrap();
        let actual: Vec<T> = cached.retrieve_chunk(&chunk_indices, &resources).unwrap();
        assert_eq!(
            actual, expected,
            "{cache_name} {name} chunk {chunk_indices:?}"
        );
        let actual_if_exists: Option<Vec<T>> = cached
            .retrieve_chunk_if_exists(&chunk_indices, &resources)
            .unwrap();
        assert_eq!(
            actual_if_exists,
            Some(expected),
            "{cache_name} {name} chunk_if_exists {chunk_indices:?}"
        );
    }

    for partial in [
        ArraySubset::new_with_ranges(&[1..4, 0..3]),
        ArraySubset::new_with_ranges(&[0..4, 0..4]),
    ] {
        let expected: Vec<T> = array
            .retrieve_partial_chunk(&[0, 1], &partial, &resources)
            .unwrap();
        let actual: Vec<T> = cached
            .retrieve_partial_chunk(&[0, 1], &partial, &resources)
            .unwrap();
        assert_eq!(actual, expected, "{cache_name} {name} partial {partial}");
    }

    // Missing chunk fill-value path (chunk, partial chunk, and multi-chunk subset)
    cached.erase_chunk(&[1, 1]).unwrap();
    let expected: Vec<T> = array.retrieve_chunk(&[1, 1], &resources).unwrap();
    assert_eq!(
        cached
            .retrieve_chunk::<Vec<T>>(&[1, 1], &resources)
            .unwrap(),
        expected,
        "{cache_name} {name} missing chunk"
    );
    let partial = ArraySubset::new_with_ranges(&[1..3, 0..3]);
    let expected: Vec<T> = array
        .retrieve_partial_chunk(&[1, 1], &partial, &resources)
        .unwrap();
    assert_eq!(
        cached
            .retrieve_partial_chunk::<Vec<T>>(&[1, 1], &partial, &resources)
            .unwrap(),
        expected,
        "{cache_name} {name} missing partial chunk"
    );
    let expected: Vec<T> = array.retrieve_array_subset(&all, &resources).unwrap();
    assert_eq!(
        cached
            .retrieve_array_subset::<Vec<T>>(&all, &resources)
            .unwrap(),
        expected,
        "{cache_name} {name} missing subset"
    );
}

#[test]
fn cached_reads_match_uncached() {
    let u16_elements: Vec<u16> = (0..63u16).map(|i| i * 3 + 1).collect();
    let optional_elements: Vec<Option<u16>> =
        (0..63u16).map(|i| (i % 4 != 0).then_some(i * 3)).collect();
    let string_elements: Vec<String> = (0..63).map(|i| "s".repeat(i % 5)).collect();

    for (name, b) in u16_builders() {
        check_cached_reads(
            "decoded",
            name,
            &build(&b),
            ChunkCacheDecodedLruChunkLimit::new(4),
            &u16_elements,
        );
        check_cached_reads(
            "encoded",
            name,
            &build(&b),
            ChunkCacheEncodedLruChunkLimit::new(4),
            &u16_elements,
        );
        check_cached_reads(
            "partial_decoder",
            name,
            &build(&b),
            ChunkCachePartialDecoderLruChunkLimit::new(4),
            &u16_elements,
        );
    }

    let opt_builder = builder(
        data_type::uint16().to_optional(),
        FillValue::new_optional_null(),
    );
    check_cached_reads(
        "decoded",
        "optional",
        &build(&opt_builder),
        ChunkCacheDecodedLruChunkLimit::new(4),
        &optional_elements,
    );
    check_cached_reads(
        "encoded",
        "optional",
        &build(&opt_builder),
        ChunkCacheEncodedLruChunkLimit::new(4),
        &optional_elements,
    );
    check_cached_reads(
        "partial_decoder",
        "optional",
        &build(&opt_builder),
        ChunkCachePartialDecoderLruChunkLimit::new(4),
        &optional_elements,
    );

    let str_builder = builder(data_type::string(), "");
    check_cached_reads(
        "decoded",
        "string",
        &build(&str_builder),
        ChunkCacheDecodedLruChunkLimit::new(4),
        &string_elements,
    );
    check_cached_reads(
        "encoded",
        "string",
        &build(&str_builder),
        ChunkCacheEncodedLruChunkLimit::new(4),
        &string_elements,
    );
    check_cached_reads(
        "partial_decoder",
        "string",
        &build(&str_builder),
        ChunkCachePartialDecoderLruChunkLimit::new(4),
        &string_elements,
    );
}

/// An array subset that is not exposed via `as_array_subset()`, like a custom indexer.
struct OpaqueSubset(ArraySubset);

impl Indexer for OpaqueSubset {
    fn dimensionality(&self) -> usize {
        Indexer::dimensionality(&self.0)
    }

    fn len(&self) -> u64 {
        Indexer::len(&self.0)
    }

    fn output_shape(&self) -> Vec<u64> {
        Indexer::output_shape(&self.0)
    }

    fn iter_indices(&self) -> Box<dyn IndexerIterator<Item = ArrayIndices>> {
        Indexer::iter_indices(&self.0)
    }

    fn iter_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = u64>>, IndexerError> {
        Indexer::iter_linearised_indices(&self.0, array_shape)
    }

    fn iter_contiguous_linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<Box<dyn IndexerIterator<Item = (u64, u64)>>, IndexerError> {
        Indexer::iter_contiguous_linearised_indices(&self.0, array_shape)
    }

    fn validate(&self, array_shape: &[u64]) -> Result<(), IndexerError> {
        Indexer::validate(&self.0, array_shape)
    }
}

#[test]
fn opaque_indexer_partial_chunk_roundtrip() {
    let resources = Resources::default();
    let array = build(&builder(data_type::uint16(), 0u16));
    let cached = ArrayCached::new(array.clone(), ChunkCacheDecodedLruChunkLimit::new(4));
    let subset = ArraySubset::new_with_ranges(&[1..3, 0..3]);
    let indexer = OpaqueSubset(subset.clone());
    assert_eq!(indexer.output_shape(), [2, 3]);

    let written: Vec<u16> = (1..=6).collect();
    array
        .store_partial_chunk(&[0, 0], &indexer, written.clone(), &resources)
        .unwrap();
    let stored: Vec<u16> = array
        .retrieve_partial_chunk(&[0, 0], &subset, &resources)
        .unwrap();
    assert_eq!(stored, written);
    let read: Vec<u16> = array
        .retrieve_partial_chunk(&[0, 0], &indexer, &resources)
        .unwrap();
    assert_eq!(read, written);
    let read: Vec<u16> = cached
        .retrieve_partial_chunk(&[0, 0], &indexer, &resources)
        .unwrap();
    assert_eq!(read, written);
}

/// Out-of-bounds subsets with huge element counts are rejected before any output is allocated.
#[test]
fn retrieve_out_of_bounds_rejected_before_allocation() {
    let resources = Resources::default();
    let array = build(&builder(data_type::uint16(), 0u16));
    let cached = ArrayCached::new(array.clone(), ChunkCacheDecodedLruChunkLimit::new(4));

    // 2^64 elements
    let partial_oob = ArraySubset::new_with_ranges(&[0..1 << 32, 0..1 << 32]);
    assert!(matches!(
        array.retrieve_partial_chunk::<Vec<u16>>(&[0, 0], &partial_oob, &resources),
        Err(ArrayError::InvalidArraySubset(..))
    ));
    assert!(matches!(
        cached.retrieve_partial_chunk::<Vec<u16>>(&[0, 0], &partial_oob, &resources),
        Err(ArrayError::InvalidArraySubset(..))
    ));

    // Wrong dimensionality with huge element count
    let subset_oob = ArraySubset::new_with_ranges(&[0..1 << 32, 0..1 << 32, 0..2]);
    assert!(matches!(
        array.retrieve_array_subset::<Vec<u16>>(&subset_oob, &resources),
        Err(ArrayError::InvalidArraySubset(..))
    ));
    assert!(matches!(
        cached.retrieve_array_subset::<Vec<u16>>(&subset_oob, &resources),
        Err(ArrayError::InvalidArraySubset(..))
    ));
}

#[cfg(feature = "async")]
mod async_tests {
    use super::*;
    use zarrs::array::chunk_cache::{
        AsyncChunkCache, AsyncChunkCacheDecodedLruChunkLimit, AsyncChunkCacheEncodedLruChunkLimit,
        AsyncChunkCachePartialDecoderLruChunkLimit, AsyncChunkCacheType,
    };
    use zarrs_storage::store::AsyncMemoryStore;

    async fn check_async_cached_reads<T, C>(
        cache_name: &str,
        name: &str,
        builder: &ArrayBuilder,
        cache: C,
        elements: &[T],
    ) where
        T: Element + ElementOwned + Clone + PartialEq + Debug + Send + Sync,
        C: AsyncChunkCache,
        C::Value: AsyncChunkCacheType,
    {
        let resources = Resources::default();
        let array = builder
            .build(Arc::new(AsyncMemoryStore::new()), "/array")
            .unwrap();
        let all = ArraySubset::new_with_shape(SHAPE);
        array
            .async_store_array_subset(&all, elements.to_vec(), &resources)
            .await
            .unwrap();
        let cached = ArrayCached::new(array.clone(), cache);

        for subset in [
            all.clone(),
            ArraySubset::new_with_ranges(&[1..6, 2..8]),
            ArraySubset::new_with_ranges(&[4..8, 4..8]),
            ArraySubset::new_with_ranges(&[1..3, 5..8]),
        ] {
            let expected: Vec<T> = array
                .async_retrieve_array_subset(&subset, &resources)
                .await
                .unwrap();
            for pass in 0..2 {
                let actual: Vec<T> = cached
                    .async_retrieve_array_subset(&subset, &resources)
                    .await
                    .unwrap();
                assert_eq!(actual, expected, "{cache_name} {name} pass={pass} {subset}");
            }
        }

        for chunk_indices in [[0, 0], [1, 2]] {
            let expected: Vec<T> = array
                .async_retrieve_chunk(&chunk_indices, &resources)
                .await
                .unwrap();
            let actual: Vec<T> = cached
                .async_retrieve_chunk(&chunk_indices, &resources)
                .await
                .unwrap();
            assert_eq!(
                actual, expected,
                "{cache_name} {name} chunk {chunk_indices:?}"
            );
            let actual_if_exists: Option<Vec<T>> = cached
                .async_retrieve_chunk_if_exists(&chunk_indices, &resources)
                .await
                .unwrap();
            assert_eq!(
                actual_if_exists,
                Some(expected),
                "{cache_name} {name} chunk_if_exists {chunk_indices:?}"
            );
        }

        for partial in [
            ArraySubset::new_with_ranges(&[1..4, 0..3]),
            ArraySubset::new_with_ranges(&[0..4, 0..4]),
        ] {
            let expected: Vec<T> = array
                .async_retrieve_partial_chunk(&[0, 1], &partial, &resources)
                .await
                .unwrap();
            let actual: Vec<T> = cached
                .async_retrieve_partial_chunk(&[0, 1], &partial, &resources)
                .await
                .unwrap();
            assert_eq!(actual, expected, "{cache_name} {name} partial {partial}");
        }

        // Missing chunk fill-value path
        cached.async_erase_chunk(&[1, 1]).await.unwrap();
        let expected: Vec<T> = array
            .async_retrieve_chunk(&[1, 1], &resources)
            .await
            .unwrap();
        assert_eq!(
            cached
                .async_retrieve_chunk::<Vec<T>>(&[1, 1], &resources)
                .await
                .unwrap(),
            expected,
            "{cache_name} {name} missing chunk"
        );
    }

    #[tokio::test]
    async fn async_cached_reads_match_uncached() {
        let u16_elements: Vec<u16> = (0..63u16).map(|i| i * 3 + 1).collect();
        let optional_elements: Vec<Option<u16>> =
            (0..63u16).map(|i| (i % 4 != 0).then_some(i * 3)).collect();
        let string_elements: Vec<String> = (0..63).map(|i| "s".repeat(i % 5)).collect();

        for (name, b) in u16_builders() {
            check_async_cached_reads(
                "decoded",
                name,
                &b,
                AsyncChunkCacheDecodedLruChunkLimit::new(4),
                &u16_elements,
            )
            .await;
            check_async_cached_reads(
                "encoded",
                name,
                &b,
                AsyncChunkCacheEncodedLruChunkLimit::new(4),
                &u16_elements,
            )
            .await;
            check_async_cached_reads(
                "partial_decoder",
                name,
                &b,
                AsyncChunkCachePartialDecoderLruChunkLimit::new(4),
                &u16_elements,
            )
            .await;
        }

        let opt_builder = builder(
            data_type::uint16().to_optional(),
            FillValue::new_optional_null(),
        );
        check_async_cached_reads(
            "encoded",
            "optional",
            &opt_builder,
            AsyncChunkCacheEncodedLruChunkLimit::new(4),
            &optional_elements,
        )
        .await;
        check_async_cached_reads(
            "partial_decoder",
            "optional",
            &opt_builder,
            AsyncChunkCachePartialDecoderLruChunkLimit::new(4),
            &optional_elements,
        )
        .await;

        let str_builder = builder(data_type::string(), "");
        check_async_cached_reads(
            "encoded",
            "string",
            &str_builder,
            AsyncChunkCacheEncodedLruChunkLimit::new(4),
            &string_elements,
        )
        .await;
        check_async_cached_reads(
            "partial_decoder",
            "string",
            &str_builder,
            AsyncChunkCachePartialDecoderLruChunkLimit::new(4),
            &string_elements,
        )
        .await;
    }

    #[tokio::test]
    async fn async_retrieve_out_of_bounds_rejected_before_allocation() {
        let resources = Resources::default();
        let array = builder(data_type::uint16(), 0u16)
            .build(Arc::new(AsyncMemoryStore::new()), "/array")
            .unwrap();
        let cached = ArrayCached::new(array.clone(), AsyncChunkCacheDecodedLruChunkLimit::new(4));

        let partial_oob = ArraySubset::new_with_ranges(&[0..1 << 32, 0..1 << 32]);
        assert!(matches!(
            array
                .async_retrieve_partial_chunk::<Vec<u16>>(&[0, 0], &partial_oob, &resources)
                .await,
            Err(ArrayError::InvalidArraySubset(..))
        ));
        assert!(matches!(
            cached
                .async_retrieve_partial_chunk::<Vec<u16>>(&[0, 0], &partial_oob, &resources)
                .await,
            Err(ArrayError::InvalidArraySubset(..))
        ));

        let subset_oob = ArraySubset::new_with_ranges(&[0..1 << 32, 0..1 << 32, 0..2]);
        assert!(matches!(
            array
                .async_retrieve_array_subset::<Vec<u16>>(&subset_oob, &resources)
                .await,
            Err(ArrayError::InvalidArraySubset(..))
        ));
        assert!(matches!(
            cached
                .async_retrieve_array_subset::<Vec<u16>>(&subset_oob, &resources)
                .await,
            Err(ArrayError::InvalidArraySubset(..))
        ));
    }
}
