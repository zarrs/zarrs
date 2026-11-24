#![allow(missing_docs)]

//! Storing inputs with an inherent layout (`ndarray` arrays and `Tensor`) validates their shape
//! against the target, and empty chunk selections validate dimensionality and shape.

use std::sync::Arc;

use zarrs::array::{
    Array, ArrayBuilder, ArrayError, ArrayIndices, ArraySubset, ElementError, Tensor, data_type,
};
use zarrs_codec::Resources;
use zarrs_storage::store::MemoryStore;

const SHAPE: [u64; 2] = [7, 9];
const CHUNK_SHAPE: [u64; 2] = [4, 4];

fn build_array() -> Array<MemoryStore> {
    ArrayBuilder::new(
        SHAPE.to_vec(),
        CHUNK_SHAPE.to_vec(),
        data_type::uint16(),
        0u16,
    )
    .build(Arc::new(MemoryStore::default()), "/array")
    .unwrap()
}

fn u16_bytes(elements: &[u16]) -> Vec<u8> {
    elements.iter().flat_map(|e| e.to_ne_bytes()).collect()
}

#[test]
fn store_tensor_shape_validation() {
    let resources = Resources::default();
    let array = build_array();
    let all = ArraySubset::new_with_shape(SHAPE);

    // store_chunk: [4, 4] target rejects [16] and [2, 8]
    let chunk_elements: Vec<u16> = (0..16u16).collect();
    for bad_shape in [vec![16], vec![2, 8]] {
        let bad_tensor = Tensor::new(u16_bytes(&chunk_elements), data_type::uint16(), bad_shape);
        assert!(matches!(
            array.store_chunk(&[0, 0], &bad_tensor, &resources),
            Err(ArrayError::ElementError(
                ElementError::IncompatibleShape { .. }
            ))
        ));
    }

    // store_chunks (multi-chunk): [8, 8] target rejects [64]
    let multi_chunk_elements: Vec<u16> = (0..64u16).collect();
    let chunks = ArraySubset::new_with_ranges(&[0..2, 0..2]);
    let bad_chunks_tensor = Tensor::new(
        u16_bytes(&multi_chunk_elements),
        data_type::uint16(),
        vec![64],
    );
    assert!(matches!(
        array.store_chunks(&chunks, &bad_chunks_tensor, &resources),
        Err(ArrayError::ElementError(
            ElementError::IncompatibleShape { .. }
        ))
    ));

    // store_array_subset (multi-chunk): [7, 9] target rejects [63] and [9, 7]
    let all_elements: Vec<u16> = (0..63u16).collect();
    for bad_shape in [vec![63], vec![9, 7]] {
        let bad_tensor = Tensor::new(u16_bytes(&all_elements), data_type::uint16(), bad_shape);
        assert!(matches!(
            array.store_array_subset(&all, &bad_tensor, &resources),
            Err(ArrayError::ElementError(
                ElementError::IncompatibleShape { .. }
            ))
        ));
    }

    // store_partial_chunk (subset): [2, 3] target rejects [6] and [3, 2]
    let partial_elements: Vec<u16> = (0..6u16).collect();
    let partial_subset = ArraySubset::new_with_ranges(&[1..3, 0..3]);
    for bad_shape in [vec![6], vec![3, 2]] {
        let bad_tensor = Tensor::new(u16_bytes(&partial_elements), data_type::uint16(), bad_shape);
        assert!(matches!(
            array.store_partial_chunk(&[0, 0], &partial_subset, &bad_tensor, &resources),
            Err(ArrayError::ElementError(
                ElementError::IncompatibleShape { .. }
            ))
        ));
    }

    // store_partial_chunk (1D element-list indexer covering the entire 2D chunk):
    // output_shape() is [16], so [16] succeeds and [4, 4] is rejected.
    let all_chunk_indices: Vec<ArrayIndices> = (0..4)
        .flat_map(|i| (0..4).map(move |j| ArrayIndices::from([i, j].as_slice())))
        .collect();
    let flat_chunk_tensor = Tensor::new(u16_bytes(&chunk_elements), data_type::uint16(), vec![16]);
    let grid_chunk_tensor =
        Tensor::new(u16_bytes(&chunk_elements), data_type::uint16(), vec![4, 4]);
    assert!(matches!(
        array.store_partial_chunk(&[0, 0], &all_chunk_indices, &grid_chunk_tensor, &resources),
        Err(ArrayError::ElementError(
            ElementError::IncompatibleShape { .. }
        ))
    ));
    array
        .store_partial_chunk(&[0, 0], &all_chunk_indices, &flat_chunk_tensor, &resources)
        .unwrap();
    let stored: Vec<u16> = array.retrieve_chunk(&[0, 0], &resources).unwrap();
    assert_eq!(stored, chunk_elements);

    // Matching Tensor stores cleanly across store_array_subset
    let good_tensor = Tensor::new(
        u16_bytes(&all_elements),
        data_type::uint16(),
        SHAPE.to_vec(),
    );
    array
        .store_array_subset(&all, &good_tensor, &resources)
        .unwrap();
    let stored: Vec<u16> = array.retrieve_array_subset(&all, &resources).unwrap();
    assert_eq!(stored, all_elements);
}

#[cfg(feature = "ndarray")]
#[test]
fn store_ndarray_shape_validation() {
    let resources = Resources::default();
    let array = build_array();
    let all = ArraySubset::new_with_shape(SHAPE);
    let chunk_elements: Vec<u16> = (0..16u16).collect();
    let all_elements: Vec<u16> = (0..63u16).collect();
    let all_chunk_indices: Vec<ArrayIndices> = (0..4)
        .flat_map(|i| (0..4).map(move |j| ArrayIndices::from([i, j].as_slice())))
        .collect();

    let flat_array = ndarray::Array1::from_vec(chunk_elements.clone());
    assert!(matches!(
        array.store_chunk(&[0, 0], &flat_array, &resources),
        Err(ArrayError::ElementError(
            ElementError::IncompatibleShape { .. }
        ))
    ));
    assert!(matches!(
        array.store_chunk(&[0, 0], flat_array.clone(), &resources),
        Err(ArrayError::ElementError(
            ElementError::IncompatibleShape { .. }
        ))
    ));

    let transposed_array = ndarray::Array2::from_shape_vec((9, 7), all_elements.clone()).unwrap();
    assert!(matches!(
        array.store_array_subset(&all, &transposed_array, &resources),
        Err(ArrayError::ElementError(
            ElementError::IncompatibleShape { .. }
        ))
    ));

    // 1D element-list indexer covering the chunk accepts Array1([16]) and rejects Array2([4, 4])
    let grid_array = ndarray::Array2::from_shape_vec((4, 4), chunk_elements).unwrap();
    assert!(matches!(
        array.store_partial_chunk(&[0, 0], &all_chunk_indices, &grid_array, &resources),
        Err(ArrayError::ElementError(
            ElementError::IncompatibleShape { .. }
        ))
    ));
    array
        .store_partial_chunk(&[0, 0], &all_chunk_indices, &flat_array, &resources)
        .unwrap();

    // Matching C-order and F-order ndarray arrays (borrowed and owned) store identical elements
    let c_array = ndarray::Array2::from_shape_vec((7, 9), all_elements).unwrap();
    let f_array = c_array
        .t()
        .as_standard_layout()
        .into_owned()
        .reversed_axes();
    for input in [&c_array, &f_array] {
        array
            .store_array_subset(&all, vec![0u16; 63], &resources)
            .unwrap();
        array.store_array_subset(&all, input, &resources).unwrap();
        let stored: ndarray::Array2<u16> = array.retrieve_array_subset(&all, &resources).unwrap();
        assert_eq!(&stored, input);

        array
            .store_array_subset(&all, vec![0u16; 63], &resources)
            .unwrap();
        array
            .store_array_subset(&all, input.clone(), &resources)
            .unwrap();
        let stored: ndarray::Array2<u16> = array.retrieve_array_subset(&all, &resources).unwrap();
        assert_eq!(&stored, input);
    }
}

#[test]
fn store_empty_chunks_validates_dimensionality_and_shape() {
    let resources = Resources::default();
    let array = build_array();

    // Empty chunk selection with wrong dimensionality is rejected
    assert!(matches!(
        array.chunks_subset(&[0..0]),
        Err(ArrayError::IncompatibleDimensionalityError(..))
    ));
    let bad_dims_tensor = Tensor::new(vec![], data_type::uint16(), vec![0]);
    assert!(matches!(
        array.store_chunks(&[0..0], &bad_dims_tensor, &resources),
        Err(ArrayError::IncompatibleDimensionalityError(..))
    ));
    assert!(matches!(
        array.store_chunks(&[0..0], &[0u16; 0], &resources),
        Err(ArrayError::IncompatibleDimensionalityError(..))
    ));

    // Empty chunk selection validates tensor / ndarray shapes against [0, 0]
    let empty_chunks = [0..0, 0..0];
    for bad_shape in [vec![0], vec![1, 0], vec![0, 1]] {
        let bad_tensor = Tensor::new(vec![], data_type::uint16(), bad_shape);
        assert!(matches!(
            array.store_chunks(&empty_chunks, &bad_tensor, &resources),
            Err(ArrayError::ElementError(
                ElementError::IncompatibleShape { .. }
            ))
        ));
    }
    let non_empty_bad_shape = Tensor::new(vec![0, 0], data_type::uint16(), vec![1]);
    assert!(matches!(
        array.store_chunks(&empty_chunks, &non_empty_bad_shape, &resources),
        Err(ArrayError::ElementError(
            ElementError::IncompatibleShape { .. }
        ))
    ));
    let good_tensor = Tensor::new(vec![], data_type::uint16(), vec![0, 0]);
    array
        .store_chunks(&empty_chunks, &good_tensor, &resources)
        .unwrap();
    array
        .store_chunks(&empty_chunks, &[0u16; 0], &resources)
        .unwrap();

    #[cfg(feature = "ndarray")]
    {
        assert!(matches!(
            array.store_chunks(&empty_chunks, ndarray::Array1::<u16>::zeros(0), &resources),
            Err(ArrayError::ElementError(
                ElementError::IncompatibleShape { .. }
            ))
        ));
        assert!(matches!(
            array.store_chunks(
                &empty_chunks,
                ndarray::Array2::<u16>::zeros((1, 0)),
                &resources,
            ),
            Err(ArrayError::ElementError(
                ElementError::IncompatibleShape { .. }
            ))
        ));
        array
            .store_chunks(
                &empty_chunks,
                ndarray::Array2::<u16>::zeros((0, 0)),
                &resources,
            )
            .unwrap();
    }
}

#[cfg(feature = "async")]
mod async_tests {
    use super::*;
    use zarrs_storage::store::MemoryStore;

    fn build_async_array() -> Array<MemoryStore> {
        ArrayBuilder::new(
            SHAPE.to_vec(),
            CHUNK_SHAPE.to_vec(),
            data_type::uint16(),
            0u16,
        )
        .build(Arc::new(MemoryStore::new()), "/array")
        .unwrap()
    }

    #[tokio::test]
    async fn async_store_rejects_incompatible_shape() {
        let resources = Resources::default();
        let array = build_async_array();
        let all = ArraySubset::new_with_shape(SHAPE);

        let chunk_elements: Vec<u16> = (0..16u16).collect();
        let flat_chunk_tensor =
            Tensor::new(u16_bytes(&chunk_elements), data_type::uint16(), vec![16]);
        assert!(matches!(
            array
                .async_store_chunk(&[0, 0], &flat_chunk_tensor, &resources)
                .await,
            Err(ArrayError::ElementError(
                ElementError::IncompatibleShape { .. }
            ))
        ));

        let all_elements: Vec<u16> = (0..63u16).collect();
        let bad_subset_tensor =
            Tensor::new(u16_bytes(&all_elements), data_type::uint16(), vec![63]);
        assert!(matches!(
            array
                .async_store_array_subset(&all, &bad_subset_tensor, &resources)
                .await,
            Err(ArrayError::ElementError(
                ElementError::IncompatibleShape { .. }
            ))
        ));

        let all_chunk_indices: Vec<ArrayIndices> = (0..4)
            .flat_map(|i| (0..4).map(move |j| ArrayIndices::from([i, j].as_slice())))
            .collect();
        let grid_chunk_tensor =
            Tensor::new(u16_bytes(&chunk_elements), data_type::uint16(), vec![4, 4]);
        assert!(matches!(
            array
                .async_store_partial_chunk(
                    &[0, 0],
                    &all_chunk_indices,
                    &grid_chunk_tensor,
                    &resources
                )
                .await,
            Err(ArrayError::ElementError(
                ElementError::IncompatibleShape { .. }
            ))
        ));
        array
            .async_store_partial_chunk(&[0, 0], &all_chunk_indices, &flat_chunk_tensor, &resources)
            .await
            .unwrap();
        let stored: Vec<u16> = array
            .async_retrieve_chunk(&[0, 0], &resources)
            .await
            .unwrap();
        assert_eq!(stored, chunk_elements);

        #[cfg(feature = "ndarray")]
        {
            let flat_array = ndarray::Array1::from_vec(chunk_elements);
            assert!(matches!(
                array
                    .async_store_chunk(&[0, 0], &flat_array, &resources)
                    .await,
                Err(ArrayError::ElementError(
                    ElementError::IncompatibleShape { .. }
                ))
            ));
        }
    }

    #[tokio::test]
    async fn async_store_empty_chunks_validates_dimensionality_and_shape() {
        let resources = Resources::default();
        let array = build_async_array();

        let bad_dims_tensor = Tensor::new(vec![], data_type::uint16(), vec![0]);
        assert!(matches!(
            array
                .async_store_chunks(&[0..0], &bad_dims_tensor, &resources)
                .await,
            Err(ArrayError::IncompatibleDimensionalityError(..))
        ));
        assert!(matches!(
            array
                .async_store_chunks(&[0..0], &[0u16; 0], &resources)
                .await,
            Err(ArrayError::IncompatibleDimensionalityError(..))
        ));

        let empty_chunks = [0..0, 0..0];
        for bad_shape in [vec![0], vec![1, 0], vec![0, 1]] {
            let bad_tensor = Tensor::new(vec![], data_type::uint16(), bad_shape);
            assert!(matches!(
                array
                    .async_store_chunks(&empty_chunks, &bad_tensor, &resources)
                    .await,
                Err(ArrayError::ElementError(
                    ElementError::IncompatibleShape { .. }
                ))
            ));
        }
        let non_empty_bad_shape = Tensor::new(vec![0, 0], data_type::uint16(), vec![1]);
        assert!(matches!(
            array
                .async_store_chunks(&empty_chunks, &non_empty_bad_shape, &resources)
                .await,
            Err(ArrayError::ElementError(
                ElementError::IncompatibleShape { .. }
            ))
        ));
        let good_tensor = Tensor::new(vec![], data_type::uint16(), vec![0, 0]);
        array
            .async_store_chunks(&empty_chunks, &good_tensor, &resources)
            .await
            .unwrap();
        array
            .async_store_chunks(&empty_chunks, &[0u16; 0], &resources)
            .await
            .unwrap();

        #[cfg(feature = "ndarray")]
        {
            assert!(matches!(
                array
                    .async_store_chunks(&empty_chunks, ndarray::Array1::<u16>::zeros(0), &resources)
                    .await,
                Err(ArrayError::ElementError(
                    ElementError::IncompatibleShape { .. }
                ))
            ));
            assert!(matches!(
                array
                    .async_store_chunks(
                        &empty_chunks,
                        ndarray::Array2::<u16>::zeros((1, 0)),
                        &resources,
                    )
                    .await,
                Err(ArrayError::ElementError(
                    ElementError::IncompatibleShape { .. }
                ))
            ));
            array
                .async_store_chunks(
                    &empty_chunks,
                    ndarray::Array2::<u16>::zeros((0, 0)),
                    &resources,
                )
                .await
                .unwrap();
        }
    }
}
