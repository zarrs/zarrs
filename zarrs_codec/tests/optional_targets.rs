#![allow(missing_docs)]

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::ArraySubset;
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView,
    IncompatibleOptionalMaskError, decode_into_array_bytes_target,
};

const SHAPE: [u64; 2] = [3, 4];

/// Build a `u16` target covering `subset`, with `masks` ordered from outermost to innermost.
fn target<'a>(
    data: &'a mut [u8; 24],
    masks: &'a mut [[u8; 12]],
    subset: &ArraySubset,
) -> ArrayBytesDecodeIntoTarget<'a> {
    // SAFETY: each view is the only view of its backing buffer.
    let data_view = unsafe {
        ArrayBytesFixedDisjointView::new(UnsafeCellSlice::new(data), 2, &SHAPE, subset.clone())
    }
    .unwrap();
    let mask_views: Vec<_> = masks
        .iter_mut()
        .map(|mask| {
            // SAFETY: each view is the only view of its backing buffer.
            unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(mask),
                    1,
                    &SHAPE,
                    subset.clone(),
                )
            }
            .unwrap()
        })
        .collect();
    ArrayBytesDecodeIntoTarget::from(data_view)
        .with_optional_masks(mask_views)
        .unwrap()
}

/// Build optional array bytes with `masks` ordered from outermost to innermost.
fn source<'a>(data: &'a [u8], masks: &'a [[u8; 4]]) -> ArrayBytes<'a> {
    masks
        .iter()
        .rev()
        .fold(ArrayBytes::new_flen(data), |bytes, mask| {
            bytes.with_optional_mask(mask.as_slice())
        })
}

fn depth(target: &ArrayBytesDecodeIntoTarget) -> usize {
    match target {
        ArrayBytesDecodeIntoTarget::Fixed(_) => 0,
        ArrayBytesDecodeIntoTarget::Optional { data, .. } => 1 + depth(data),
    }
}

/// Copy the elements of a 2x2 source into `(rows, cols)` of the expected data and masks.
fn expect_block(
    expected_data: &mut [u8; 24],
    expected_masks: &mut [[u8; 12]],
    (rows, cols): (std::ops::Range<usize>, std::ops::Range<usize>),
    data: &[u8],
    masks: &[[u8; 4]],
) {
    let indices = rows.flat_map(|row| cols.clone().map(move |col| row * 4 + col));
    for (k, index) in indices.enumerate() {
        expected_data[index * 2..index * 2 + 2].copy_from_slice(&data[k * 2..k * 2 + 2]);
        for (expected_mask, mask) in expected_masks.iter_mut().zip(masks) {
            expected_mask[index] = mask[k];
        }
    }
}

#[test]
fn optional_targets_nest_masks_outer_to_inner() {
    let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
    let source_data = [10, 11, 12, 13, 14, 15, 16, 17];
    let source_masks = [[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]];

    for depth_ in [0, 1, 3] {
        let mut data = [0xa5; 24];
        let mut masks = vec![[0xa5; 12]; depth_];
        {
            let target = target(&mut data, &mut masks, &subset);
            assert_eq!(depth(&target), depth_);
            assert_eq!(target.subset(), &subset);
            assert_eq!(target.num_elements(), 4);
            let source = source(&source_data, &source_masks[..depth_]);
            decode_into_array_bytes_target(&source, target).unwrap();
        }

        let mut expected_data = [0xa5; 24];
        let mut expected_masks = vec![[0xa5; 12]; depth_];
        expect_block(
            &mut expected_data,
            &mut expected_masks,
            (1..3, 1..3),
            &source_data,
            &source_masks,
        );
        assert_eq!(data, expected_data);
        assert_eq!(masks, expected_masks);
    }
}

#[test]
fn subdivided_targets_preserve_structure() {
    let subset = ArraySubset::new_with_ranges(&[1..3, 0..4]);
    let left = ArraySubset::new_with_ranges(&[1..3, 0..2]);
    let right = ArraySubset::new_with_ranges(&[1..3, 2..4]);
    let left_data = [1, 2, 3, 4, 5, 6, 7, 8];
    let right_data = [11, 12, 13, 14, 15, 16, 17, 18];
    let left_masks = [[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]];
    let right_masks = [[0, 1, 0, 1], [1, 0, 0, 1], [0, 0, 1, 1]];

    for depth_ in [0, 1, 3] {
        let mut data = [0xa5; 24];
        let mut masks = vec![[0xa5; 12]; depth_];
        {
            let target = target(&mut data, &mut masks, &subset);
            for (sub_subset, sub_data, sub_masks) in [
                (&left, &left_data, &left_masks),
                (&right, &right_data, &right_masks),
            ] {
                // SAFETY: the left and right subtargets are disjoint.
                let sub = unsafe { target.subdivide(sub_subset.clone()) }.unwrap();
                assert_eq!(depth(&sub), depth_);
                assert_eq!(sub.subset(), sub_subset);
                let source = source(sub_data, &sub_masks[..depth_]);
                decode_into_array_bytes_target(&source, sub).unwrap();
            }
        }

        let mut expected_data = [0xa5; 24];
        let mut expected_masks = vec![[0xa5; 12]; depth_];
        for (cols, sub_data, sub_masks) in [
            (0..2, &left_data, &left_masks),
            (2..4, &right_data, &right_masks),
        ] {
            expect_block(
                &mut expected_data,
                &mut expected_masks,
                (1..3, cols),
                sub_data,
                sub_masks,
            );
        }
        assert_eq!(data, expected_data);
        assert_eq!(masks, expected_masks);
    }
}

#[test]
fn subdivide_rejects_subsets_outside_the_target() {
    let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
    for depth_ in [0, 1] {
        let mut data = [0xa5; 24];
        let mut masks = vec![[0xa5; 12]; depth_];
        let target = target(&mut data, &mut masks, &subset);
        for outside in [
            ArraySubset::new_with_ranges(&[0..2, 1..3]),
            ArraySubset::new_with_ranges(&[1..3, 2..4]),
            ArraySubset::new_with_ranges(&[1..3]),
        ] {
            // SAFETY: no subtarget is created.
            assert!(unsafe { target.subdivide(outside) }.is_err());
        }
        let empty = ArraySubset::new_with_start_shape(vec![99, 99], vec![0, 0]).unwrap();
        // SAFETY: an empty subtarget references no elements.
        let sub = unsafe { target.subdivide(empty) }.unwrap();
        assert_eq!(sub.num_elements(), 0);
        assert_eq!(depth(&sub), depth_);
    }
}

#[test]
fn with_optional_mask_validates_the_mask() {
    let data_subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
    let mut data = [0u8; 24];
    let mut mask = [0u8; 12];
    let mut wide_mask = [0u8; 24];
    let transposed_shape = [4, 3];
    let data_slice = UnsafeCellSlice::new(&mut data);
    let mask_slice = UnsafeCellSlice::new(&mut mask);
    let wide_mask_slice = UnsafeCellSlice::new(&mut wide_mask);
    let data_target = |subset: ArraySubset| -> ArrayBytesDecodeIntoTarget<'_> {
        // SAFETY: each target is used in isolation.
        unsafe { ArrayBytesFixedDisjointView::new(data_slice, 2, &SHAPE, subset) }
            .unwrap()
            .into()
    };
    // SAFETY: each view is used in isolation.
    let mask_view = |size, shape, subset| unsafe {
        let slice = if size == 1 {
            mask_slice
        } else {
            wide_mask_slice
        };
        ArrayBytesFixedDisjointView::new(slice, size, shape, subset).unwrap()
    };

    let err = |target: ArrayBytesDecodeIntoTarget<'_>, mask| {
        target.with_optional_mask(mask).map(|_| ()).unwrap_err()
    };
    assert!(matches!(
        err(
            data_target(data_subset.clone()),
            mask_view(2, &SHAPE, data_subset.clone())
        ),
        IncompatibleOptionalMaskError::ElementSize(2)
    ));
    assert!(matches!(
        err(
            data_target(data_subset.clone()),
            mask_view(1, &transposed_shape, data_subset.clone())
        ),
        IncompatibleOptionalMaskError::ArrayShape { .. }
    ));
    for mask_subset in [
        ArraySubset::new_with_ranges(&[0..2, 0..2]),
        ArraySubset::new_with_ranges(&[1..3, 1..2]),
        ArraySubset::new_with_start_shape(vec![1, 1], vec![0, 2]).unwrap(),
    ] {
        assert!(matches!(
            err(
                data_target(data_subset.clone()),
                mask_view(1, &SHAPE, mask_subset)
            ),
            IncompatibleOptionalMaskError::Subset { .. }
        ));
    }

    // Nested masks are validated against the innermost data
    let nested = data_target(data_subset.clone())
        .with_optional_mask(mask_view(1, &SHAPE, data_subset.clone()))
        .unwrap();
    assert!(matches!(
        err(
            nested,
            mask_view(1, &SHAPE, ArraySubset::new_with_ranges(&[0..2, 1..3]))
        ),
        IncompatibleOptionalMaskError::Subset { .. }
    ));

    // Empty subsets match irrespective of their start
    let empty_data = ArraySubset::new_with_start_shape(vec![0, 0], vec![0, 4]).unwrap();
    let empty_mask = ArraySubset::new_with_start_shape(vec![3, 0], vec![0, 4]).unwrap();
    let target = data_target(empty_data)
        .with_optional_mask(mask_view(1, &SHAPE, empty_mask))
        .unwrap();
    assert_eq!(target.num_elements(), 0);
    assert!(matches!(
        err(
            data_target(ArraySubset::new_with_start_shape(vec![0, 0], vec![0, 4]).unwrap()),
            mask_view(
                1,
                &SHAPE,
                ArraySubset::new_with_start_shape(vec![0, 0], vec![0, 3]).unwrap()
            )
        ),
        IncompatibleOptionalMaskError::Subset { .. }
    ));
}
