#![allow(missing_docs)]

//! Array operations in F order match C order operations with the elements reordered.

use std::fmt::Debug;
use std::sync::Arc;

use zarrs::array::chunk_cache::ChunkCacheDecodedLruChunkLimit;
use zarrs::array::{
    Array, ArrayBuilder, ArrayCached, ArrayError, ArrayIndices, ArrayMetadata, ArraySubset,
    Element, ElementOrder, ElementOwned, FillValue, Indexer, IndexerError, Resources, Tensor,
    data_type,
};
use zarrs::storage::store::MemoryStore;
use zarrs_chunk_grid::IndexerIterator;

const SHAPE: [u64; 2] = [7, 9];

/// Reorder the C-order `elements` of an array with `shape` into F order.
fn c_to_f<T: Clone>(elements: &[T], shape: &[u64]) -> Vec<T> {
    (0..elements.len() as u64)
        .map(|mut position| {
            // Unravel the F-order position, with the first dimension fastest
            let mut c_index = 0;
            let mut c_stride = 1;
            let mut indices = vec![0; shape.len()];
            for (index, &size) in indices.iter_mut().zip(shape) {
                *index = position % size;
                position /= size;
            }
            for (index, &size) in indices.iter().zip(shape).rev() {
                c_index += index * c_stride;
                c_stride *= size;
            }
            elements[usize::try_from(c_index).unwrap()].clone()
        })
        .collect()
}

fn builder(data_type: zarrs::array::DataType, fill_value: impl Into<FillValue>) -> ArrayBuilder {
    ArrayBuilder::new(SHAPE.to_vec(), vec![4, 4], data_type, fill_value.into())
}

fn build(builder: &ArrayBuilder) -> Array<MemoryStore> {
    builder
        .build(Arc::new(MemoryStore::new()), "/array")
        .unwrap()
}

/// Arrays of `u16` elements with various codecs, in C order.
fn u16_arrays() -> Vec<(&'static str, Array<MemoryStore>)> {
    let mut arrays = vec![
        ("bytes", build(&builder(data_type::uint16(), 0u16))),
        (
            "sharded",
            build(builder(data_type::uint16(), 0u16).subchunk_shape(vec![2, 2])),
        ),
    ];
    #[cfg(feature = "transpose")]
    arrays.push((
        "transpose",
        build(
            builder(data_type::uint16(), 0u16).array_to_array_codecs(vec![Arc::new(
                zarrs::array::codec::TransposeCodec::new(
                    zarrs::array::codec::TransposeOrder::new(&[1, 0]).unwrap(),
                ),
            )]),
        ),
    ));
    #[cfg(feature = "transpose")]
    {
        let metadata: ArrayMetadata = serde_json::from_str(
            r#"{"zarr_format": 2, "shape": [7, 9], "chunks": [4, 4], "dtype": "<u2", "fill_value": 0, "order": "F", "compressor": null, "filters": null}"#,
        )
        .unwrap();
        arrays.push((
            "v2_order_f",
            Array::new_with_metadata(Arc::new(MemoryStore::new()), "/array", metadata).unwrap(),
        ));
    }
    arrays
}

/// Check that every generic read of `array` in F order is the C order read reordered.
fn check_reads<T>(name: &str, array: &Array<MemoryStore>, elements: &[T])
where
    T: Element + ElementOwned + Clone + PartialEq + Debug,
{
    let resources = Resources::default();
    let c = array.with_element_order(ElementOrder::C);
    let f = array.with_element_order(ElementOrder::F);
    assert_eq!(f.element_order(), ElementOrder::F);
    let all = ArraySubset::new_with_shape(SHAPE);
    c.store_array_subset(&all, elements.to_vec(), &resources)
        .unwrap();

    let cached = ArrayCached::new(Arc::new(f.clone()), ChunkCacheDecodedLruChunkLimit::new(4));
    let cached_c = ArrayCached::new(Arc::new(c.clone()), ChunkCacheDecodedLruChunkLimit::new(4));
    assert_eq!(cached.element_order(), ElementOrder::F);
    assert_eq!(
        cached_c.with_element_order(ElementOrder::F).element_order(),
        ElementOrder::F
    );

    // Array subsets spanning several chunks, one chunk, and part of a chunk
    for subset in [
        all.clone(),
        ArraySubset::new_with_ranges(&[1..6, 2..8]),
        ArraySubset::new_with_ranges(&[4..8, 4..8]),
        ArraySubset::new_with_ranges(&[1..3, 5..8]),
        ArraySubset::new_with_ranges(&[2..3, 1..8]),
    ] {
        let expected = c_to_f(
            &c.retrieve_array_subset::<Vec<T>>(&subset, &resources)
                .unwrap(),
            subset.shape(),
        );
        let actual: Vec<T> = f.retrieve_array_subset(&subset, &resources).unwrap();
        assert_eq!(actual, expected, "{name} {subset}");
        let actual: Vec<T> = cached.retrieve_array_subset(&subset, &resources).unwrap();
        assert_eq!(actual, expected, "{name} cached {subset}");
    }

    // Chunks
    for chunk_indices in [[0, 0], [1, 2]] {
        let chunk_shape =
            zarrs::array::chunk_shape_to_array_shape(&f.chunk_shape(&chunk_indices).unwrap());
        let expected = c_to_f(
            &c.retrieve_chunk::<Vec<T>>(&chunk_indices, &resources)
                .unwrap(),
            &chunk_shape,
        );
        let actual: Vec<T> = f.retrieve_chunk(&chunk_indices, &resources).unwrap();
        assert_eq!(actual, expected, "{name} {chunk_indices:?}");
        let actual: Vec<T> = cached.retrieve_chunk(&chunk_indices, &resources).unwrap();
        assert_eq!(actual, expected, "{name} cached {chunk_indices:?}");
        let actual: Option<Vec<T>> = f
            .retrieve_chunk_if_exists(&chunk_indices, &resources)
            .unwrap();
        assert_eq!(actual.as_ref(), Some(&expected), "{name} {chunk_indices:?}");
        let actual: Option<Vec<T>> = cached
            .retrieve_chunk_if_exists(&chunk_indices, &resources)
            .unwrap();
        assert_eq!(actual, Some(expected), "{name} cached {chunk_indices:?}");
    }

    // Partial chunks
    let partial = ArraySubset::new_with_ranges(&[1..4, 0..3]);
    let expected = c_to_f(
        &c.retrieve_partial_chunk::<Vec<T>>(&[0, 1], &partial, &resources)
            .unwrap(),
        partial.shape(),
    );
    let actual: Vec<T> = f
        .retrieve_partial_chunk(&[0, 1], &partial, &resources)
        .unwrap();
    assert_eq!(actual, expected, "{name} partial");
    let actual: Vec<T> = cached
        .retrieve_partial_chunk(&[0, 1], &partial, &resources)
        .unwrap();
    assert_eq!(actual, expected, "{name} cached partial");

    // Element indexers have one output dimension, so they are unaffected
    let indices = vec![vec![3u64, 2], vec![0, 1], vec![2, 3]];
    let actual: Vec<T> = f
        .retrieve_partial_chunk(&[0, 0], &indices, &resources)
        .unwrap();
    let expected: Vec<T> = c
        .retrieve_partial_chunk(&[0, 0], &indices, &resources)
        .unwrap();
    assert_eq!(actual, expected, "{name} indices");

    // Chunks of the chunk grid
    let chunks = ArraySubset::new_with_ranges(&[0..2, 1..3]);
    let chunks_subset = f.chunks_subset(&chunks).unwrap();
    let expected = c_to_f(
        &c.retrieve_chunks::<Vec<T>>(&chunks, &resources).unwrap(),
        chunks_subset.shape(),
    );
    let actual: Vec<T> = f.retrieve_chunks(&chunks, &resources).unwrap();
    assert_eq!(actual, expected, "{name} chunks");

    // A missing chunk
    cached.erase_chunk(&[1, 1]).unwrap();
    assert!(
        f.retrieve_chunk_if_exists::<Vec<T>>(&[1, 1], &resources)
            .unwrap()
            .is_none()
    );
    let expected = c_to_f(
        &c.retrieve_chunk::<Vec<T>>(&[1, 1], &resources).unwrap(),
        &[4, 4],
    );
    assert_eq!(
        f.retrieve_chunk::<Vec<T>>(&[1, 1], &resources).unwrap(),
        expected
    );

    // Arrays with layouts are logically equal in either order, including when at most one dimension has more than one element
    #[cfg(feature = "ndarray")]
    for subset in [
        ArraySubset::new_with_ranges(&[1..6, 2..8]),
        ArraySubset::new_with_ranges(&[2..3, 1..7]),
        ArraySubset::new_with_ranges(&[1..7, 2..3]),
        ArraySubset::new_with_ranges(&[1..1, 2..8]),
    ] {
        let c_array: ndarray::ArrayD<T> = c.retrieve_array_subset(&subset, &resources).unwrap();
        let f_array: ndarray::ArrayD<T> = f.retrieve_array_subset(&subset, &resources).unwrap();
        let cached_f_array: ndarray::ArrayD<T> =
            cached.retrieve_array_subset(&subset, &resources).unwrap();
        assert_eq!(c_array, f_array, "{name} ndarray {subset}");
        assert_eq!(c_array, cached_f_array, "{name} cached ndarray {subset}");
        assert!(c_array.is_standard_layout());
        assert!(f_array.t().is_standard_layout());
        assert!(cached_f_array.t().is_standard_layout());
    }
}

#[test]
fn element_order_reads_u16() {
    let elements: Vec<u16> = (0..63u16).map(|i| i * 3 + 1).collect();
    for (name, array) in u16_arrays() {
        check_reads(name, &array, &elements);
    }
    // Tensors carry the order of their elements, including when at most one dimension has more than one element
    let resources = Resources::default();
    let (_, array) = u16_arrays().remove(0);
    array
        .store_array_subset(
            &ArraySubset::new_with_shape(SHAPE),
            elements.clone(),
            &resources,
        )
        .unwrap();
    let f = array.with_element_order(ElementOrder::F);
    let cached = ArrayCached::new(Arc::new(f.clone()), ChunkCacheDecodedLruChunkLimit::new(4));
    let tensor: Tensor<'static> = f.retrieve_chunk(&[0, 0], &resources).unwrap();
    assert_eq!(tensor.order(), ElementOrder::F);
    for subset in [
        ArraySubset::new_with_ranges(&[2..3, 1..7]),
        ArraySubset::new_with_ranges(&[2..3, 1..3]),
        ArraySubset::new_with_ranges(&[1..1, 2..8]),
    ] {
        let tensor: Tensor<'static> = f.retrieve_array_subset(&subset, &resources).unwrap();
        assert_eq!(tensor.order(), ElementOrder::F);
        let tensor: Tensor<'static> = cached.retrieve_array_subset(&subset, &resources).unwrap();
        assert_eq!(tensor.order(), ElementOrder::F);
    }
    let partial = ArraySubset::new_with_ranges(&[1..2, 0..3]);
    let tensor: Tensor<'static> = f
        .retrieve_partial_chunk(&[0, 0], &partial, &resources)
        .unwrap();
    assert_eq!(tensor.order(), ElementOrder::F);
    let tensor: Tensor<'static> = cached
        .retrieve_partial_chunk(&[0, 0], &partial, &resources)
        .unwrap();
    assert_eq!(tensor.order(), ElementOrder::F);
}

#[test]
fn element_order_reads_optional() {
    let elements: Vec<Option<u16>> = (0..63u16).map(|i| (i % 4 != 0).then_some(i * 3)).collect();
    let array = build(&builder(
        data_type::uint16().to_optional(),
        FillValue::new_optional_null(),
    ));
    check_reads("optional", &array, &elements);
    let array = build(
        builder(
            data_type::uint16().to_optional(),
            FillValue::new_optional_null(),
        )
        .subchunk_shape(vec![2, 2]),
    );
    check_reads("optional_sharded", &array, &elements);
}

#[test]
fn element_order_reads_string() {
    let elements: Vec<String> = (0..63).map(|i| "s".repeat(i % 5)).collect();
    let array = build(&builder(data_type::string(), ""));
    check_reads("string", &array, &elements);
}

/// An array subset that is not exposed as one, like a custom indexer.
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

/// The elements of indexers other than array subsets are in C order.
#[test]
fn element_order_indexer_is_c_order() {
    let resources = Resources::default();
    let c = build(&builder(data_type::uint16(), 0u16));
    let f = c.with_element_order(ElementOrder::F);
    let cached = ArrayCached::new(Arc::new(f.clone()), ChunkCacheDecodedLruChunkLimit::new(4));
    let subset = ArraySubset::new_with_ranges(&[1..3, 0..3]);
    let indexer = OpaqueSubset(subset.clone());
    assert_eq!(indexer.output_shape(), [2, 3]);

    let written: Vec<u16> = (1..=6).collect();
    c.store_partial_chunk(&[0, 0], &subset, written.clone(), &resources)
        .unwrap();
    let read: Vec<u16> = f
        .retrieve_partial_chunk(&[0, 0], &indexer, &resources)
        .unwrap();
    assert_eq!(read, written);
    let read: Vec<u16> = cached
        .retrieve_partial_chunk(&[0, 0], &indexer, &resources)
        .unwrap();
    assert_eq!(read, written);
}

/// A partial chunk subset outside the chunk is rejected before its F-ordered output is allocated.
#[test]
fn element_order_partial_chunk_out_of_bounds() {
    let resources = Resources::default();
    let f = build(&builder(data_type::uint16(), 0u16)).with_element_order(ElementOrder::F);
    let cached = ArrayCached::new(Arc::new(f.clone()), ChunkCacheDecodedLruChunkLimit::new(4));
    // The output of this subset has 2^64 elements
    let subset = ArraySubset::new_with_ranges(&[0..1 << 32, 0..1 << 32]);
    assert!(matches!(
        f.retrieve_partial_chunk::<Vec<u16>>(&[0, 0], &subset, &resources),
        Err(ArrayError::InvalidArraySubset(..))
    ));
    assert!(matches!(
        cached.retrieve_partial_chunk::<Vec<u16>>(&[0, 0], &subset, &resources),
        Err(ArrayError::InvalidArraySubset(..))
    ));
}

#[cfg(feature = "async")]
mod async_reads {
    use super::*;
    use zarrs::array::chunk_cache::AsyncChunkCacheDecodedLruChunkLimit;
    use zarrs_storage::store::AsyncMemoryStore;

    async fn check_async_reads<T>(name: &str, builder: &ArrayBuilder, elements: &[T])
    where
        T: Element + ElementOwned + Clone + PartialEq + Debug + Send + Sync,
    {
        let resources = Resources::default();
        let c = builder
            .build(Arc::new(AsyncMemoryStore::new()), "/array")
            .unwrap();
        let f = c.with_element_order(ElementOrder::F);
        let all = ArraySubset::new_with_shape(SHAPE);
        c.async_store_chunks(
            &ArraySubset::new_with_shape(c.chunk_grid_shape()),
            {
                // Store whole chunks (in C order) by padding the elements to the chunk grid
                let padded_shape = [8u64, 12];
                let padded: Vec<T> = (0..padded_shape[0])
                    .flat_map(|i| (0..padded_shape[1]).map(move |j| (i, j)))
                    .map(|(i, j)| {
                        elements[usize::try_from(i.min(6) * 9 + j.min(8)).unwrap()].clone()
                    })
                    .collect();
                padded
            },
            &resources,
        )
        .await
        .unwrap();
        let cached = ArrayCached::new(
            Arc::new(f.clone()),
            AsyncChunkCacheDecodedLruChunkLimit::new(4),
        );
        for subset in [
            all.clone(),
            ArraySubset::new_with_ranges(&[1..6, 2..8]),
            ArraySubset::new_with_ranges(&[4..8, 4..8]),
            ArraySubset::new_with_ranges(&[1..3, 5..8]),
        ] {
            let expected = c_to_f(
                &c.async_retrieve_array_subset::<Vec<T>>(&subset, &resources)
                    .await
                    .unwrap(),
                subset.shape(),
            );
            let actual: Vec<T> = f
                .async_retrieve_array_subset(&subset, &resources)
                .await
                .unwrap();
            assert_eq!(actual, expected, "{name} {subset}");
            let actual: Vec<T> = cached
                .async_retrieve_array_subset(&subset, &resources)
                .await
                .unwrap();
            assert_eq!(actual, expected, "{name} cached {subset}");
        }
        for chunk_indices in [[0, 0], [1, 2]] {
            let expected = c_to_f(
                &c.async_retrieve_chunk::<Vec<T>>(&chunk_indices, &resources)
                    .await
                    .unwrap(),
                &[4, 4],
            );
            let actual: Vec<T> = f
                .async_retrieve_chunk(&chunk_indices, &resources)
                .await
                .unwrap();
            assert_eq!(actual, expected, "{name} {chunk_indices:?}");
            let actual: Vec<T> = cached
                .async_retrieve_chunk(&chunk_indices, &resources)
                .await
                .unwrap();
            assert_eq!(actual, expected, "{name} cached {chunk_indices:?}");
            let actual: Option<Vec<T>> = f
                .async_retrieve_chunk_if_exists(&chunk_indices, &resources)
                .await
                .unwrap();
            assert_eq!(actual.as_ref(), Some(&expected), "{name} {chunk_indices:?}");
            let actual: Option<Vec<T>> = cached
                .async_retrieve_chunk_if_exists(&chunk_indices, &resources)
                .await
                .unwrap();
            assert_eq!(actual, Some(expected), "{name} cached {chunk_indices:?}");
        }
        let partial = ArraySubset::new_with_ranges(&[1..4, 0..3]);
        let expected = c_to_f(
            &c.async_retrieve_partial_chunk::<Vec<T>>(&[0, 1], &partial, &resources)
                .await
                .unwrap(),
            partial.shape(),
        );
        let actual: Vec<T> = f
            .async_retrieve_partial_chunk(&[0, 1], &partial, &resources)
            .await
            .unwrap();
        assert_eq!(actual, expected, "{name} partial");
        let actual: Vec<T> = cached
            .async_retrieve_partial_chunk(&[0, 1], &partial, &resources)
            .await
            .unwrap();
        assert_eq!(actual, expected, "{name} cached partial");
    }

    #[tokio::test]
    async fn element_order_async_partial_chunk_out_of_bounds() {
        let resources = Resources::default();
        let f = builder(data_type::uint16(), 0u16)
            .build(Arc::new(AsyncMemoryStore::new()), "/array")
            .unwrap()
            .with_element_order(ElementOrder::F);
        let cached = ArrayCached::new(
            Arc::new(f.clone()),
            AsyncChunkCacheDecodedLruChunkLimit::new(4),
        );
        let subset = ArraySubset::new_with_ranges(&[0..1 << 32, 0..1 << 32]);
        assert!(matches!(
            f.async_retrieve_partial_chunk::<Vec<u16>>(&[0, 0], &subset, &resources)
                .await,
            Err(ArrayError::InvalidArraySubset(..))
        ));
        assert!(matches!(
            cached
                .async_retrieve_partial_chunk::<Vec<u16>>(&[0, 0], &subset, &resources)
                .await,
            Err(ArrayError::InvalidArraySubset(..))
        ));
    }

    #[tokio::test]
    async fn element_order_async_reads() {
        let u16_elements: Vec<u16> = (0..63u16).map(|i| i * 3 + 1).collect();
        check_async_reads("bytes", &builder(data_type::uint16(), 0u16), &u16_elements).await;
        check_async_reads(
            "sharded",
            builder(data_type::uint16(), 0u16).subchunk_shape(vec![2, 2]),
            &u16_elements,
        )
        .await;
        let optional: Vec<Option<u16>> =
            (0..63u16).map(|i| (i % 4 != 0).then_some(i * 3)).collect();
        check_async_reads(
            "optional",
            &builder(
                data_type::uint16().to_optional(),
                FillValue::new_optional_null(),
            ),
            &optional,
        )
        .await;
        let strings: Vec<String> = (0..63).map(|i| "s".repeat(i % 5)).collect();
        check_async_reads("string", &builder(data_type::string(), ""), &strings).await;
    }
}
