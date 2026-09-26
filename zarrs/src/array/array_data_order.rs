use zarrs_codec::CodecError;

use crate::array::{
    ArrayBytes, ArrayIndices, ArraySubset, ArraySubsetTraits, DataType, Indexer, ravel_indices,
};

/// The memory layout of data returned by array read operations and consumed by array write operations.
///
/// Array subsets, chunk indices, and indexers are always expressed in the array's own dimension order.
/// The data order only changes the layout of the elements in the data buffers.
///
/// An F-order (column-major) buffer of an array subset with shape `[a, b, c]` has the same layout as a C-order (row-major) buffer of the transposed subset with shape `[c, b, a]`.
/// Reading and writing in F order is efficient if the array has a leading `transpose` codec, such as a Zarr V2 array with `"order": "F"`, since the `transpose` is fused or skipped entirely.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ArrayDataOrder {
    /// C order (row-major): the last dimension varies fastest.
    #[default]
    C,
    /// F order (Fortran, column-major): the first dimension varies fastest.
    F,
}

/// Reverse the order of the elements in `v`.
pub(crate) fn reverse_axes<T: Copy>(v: &[T]) -> Vec<T> {
    v.iter().rev().copied().collect()
}

/// Reverse the axes of C-order `bytes` with `shape`, producing F-order bytes.
pub(crate) fn reverse_axes_of_bytes(
    bytes: &ArrayBytes<'_>,
    shape: &[u64],
    data_type: &DataType,
) -> Result<ArrayBytes<'static>, CodecError> {
    #[cfg(feature = "transpose")]
    {
        let order: Vec<usize> = (0..shape.len()).rev().collect();
        Ok(
            crate::array::codec::array_to_array::transpose::apply_permutation(
                bytes, shape, &order, data_type,
            )?
            .into_owned(),
        )
    }
    #[cfg(not(feature = "transpose"))]
    {
        let _ = (bytes, shape, data_type);
        Err(CodecError::Other(
            "the transpose feature is required for F order data".to_string(),
        ))
    }
}

/// Reverse the axes of an array subset.
pub(crate) fn reverse_subset(subset: &dyn ArraySubsetTraits) -> ArraySubset {
    ArraySubset::from(
        std::iter::zip(subset.start().iter().rev(), subset.shape().iter().rev())
            .map(|(&start, &size)| start..start + size),
    )
}

/// Reverse the axes of an indexer.
///
/// The returned indexer selects the same elements as `indexer` in an array with reversed axes.
/// Elements are ordered such that the output is the F-order output of `indexer`.
pub(crate) fn reverse_indexer(indexer: &dyn Indexer) -> Box<dyn Indexer> {
    if let Some(subset) = indexer.as_array_subset() {
        return Box::new(reverse_subset(subset));
    }

    let indices: Vec<ArrayIndices> = indexer
        .iter_indices()
        .map(|indices| reverse_axes(&indices))
        .collect();
    let output_shape = indexer.output_shape();
    if output_shape.len() < 2 {
        return Box::new(indices);
    }

    // Reorder the indices from the C order to the F order of the output shape
    let output_shape_reversed = reverse_axes(&output_shape);
    let indices_f: Vec<ArrayIndices> = ArraySubset::new_with_shape(output_shape_reversed)
        .indices()
        .into_iter()
        .map(|output_indices_reversed| {
            let output_indices = reverse_axes(&output_indices_reversed);
            let linear = ravel_indices(&output_indices, &output_shape)
                .expect("output indices are within the output shape");
            indices[usize::try_from(linear).unwrap()].clone()
        })
        .collect();
    Box::new(indices_f)
}

#[cfg(all(test, feature = "transpose"))]
mod tests {
    use std::sync::Arc;

    use zarrs_storage::store::MemoryStore;

    use super::*;
    use crate::array::{Array, ArrayMetadata};

    #[test]
    fn data_order_skips_transpose_for_v2_f_order() {
        let metadata: zarrs_metadata::v2::ArrayMetadataV2 = serde_json::from_str(
            r#"{
                "zarr_format": 2,
                "shape": [4, 6],
                "chunks": [2, 3],
                "dtype": "<u2",
                "compressor": null,
                "fill_value": 0,
                "order": "F",
                "filters": null
            }"#,
        )
        .unwrap();
        let mut array = Array::new_with_metadata(
            Arc::new(MemoryStore::new()),
            "/array",
            ArrayMetadata::V2(metadata),
        )
        .unwrap();
        assert_eq!(array.codecs_bound().array_to_array_codecs().len(), 1);

        array.set_data_order(ArrayDataOrder::F).unwrap();
        assert!(array.reverses_axes());
        assert!(
            array
                .codecs_bound_in_data_order()
                .array_to_array_codecs()
                .is_empty()
        );

        array.set_data_order(ArrayDataOrder::C).unwrap();
        assert!(!array.reverses_axes());
        assert_eq!(
            array
                .codecs_bound_in_data_order()
                .array_to_array_codecs()
                .len(),
            1
        );
    }
}
