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
    ArrayBytes::new_flen(data).with_optional_masks(masks.iter().map(<[u8; 4]>::as_slice))
}

fn depth(target: &ArrayBytesDecodeIntoTarget) -> usize {
    match target {
        ArrayBytesDecodeIntoTarget::Fixed(_) => 0,
        ArrayBytesDecodeIntoTarget::Optional(optional) => {
            assert_eq!(optional.mask().shape(), optional.data().shape());
            1 + depth(optional.data())
        }
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
            let mut target = target(&mut data, &mut masks, &subset);
            assert_eq!(depth(&target), depth_);
            assert_eq!(target.shape(), subset.shape());
            assert_eq!(target.num_elements(), 4);
            let source = source(&source_data, &source_masks[..depth_]);
            decode_into_array_bytes_target(&source, &mut target).unwrap();
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
    // Subsets of a target are relative to the target
    let left = ArraySubset::new_with_ranges(&[0..2, 0..2]);
    let right = ArraySubset::new_with_ranges(&[0..2, 2..4]);
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
                let mut sub = unsafe { target.subdivide(sub_subset.clone()) }.unwrap();
                assert_eq!(depth(&sub), depth_);
                assert_eq!(sub.shape(), sub_subset.shape());
                let source = source(sub_data, &sub_masks[..depth_]);
                decode_into_array_bytes_target(&source, &mut sub).unwrap();
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
        // Subsets are relative to the 2x2 target
        for outside in [
            ArraySubset::new_with_ranges(&[0..3, 0..2]),
            ArraySubset::new_with_ranges(&[0..2, 1..3]),
            ArraySubset::new_with_ranges(&[0..2]),
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
    let mut outer_mask = [0u8; 12];
    let transposed_shape = [4, 3];
    let data_slice = UnsafeCellSlice::new(&mut data);
    let mask_slice = UnsafeCellSlice::new(&mut mask);
    let wide_mask_slice = UnsafeCellSlice::new(&mut wide_mask);
    let outer_mask_slice = UnsafeCellSlice::new(&mut outer_mask);
    let data_target = |subset: ArraySubset| -> ArrayBytesDecodeIntoTarget<'_> {
        // SAFETY: views of the same buffer are never live at the same time.
        unsafe { ArrayBytesFixedDisjointView::new(data_slice, 2, &SHAPE, subset) }
            .unwrap()
            .into()
    };
    // SAFETY: views of the same buffer are never live at the same time.
    let mask_view = |size, shape, subset| unsafe {
        let slice = if size == 1 {
            mask_slice
        } else {
            wide_mask_slice
        };
        ArrayBytesFixedDisjointView::new(slice, size, shape, subset).unwrap()
    };

    // Return the error kind, checking that the target and mask are returned unchanged
    let err = |target: ArrayBytesDecodeIntoTarget<'_>, mask: ArrayBytesFixedDisjointView<'_>| {
        let (target_depth, target_shape) = (depth(&target), target.shape().to_vec());
        let (mask_shape, mask_strides) = (mask.shape().to_vec(), mask.strides().to_vec());
        let error = target.with_optional_mask(mask).map(|_| ()).unwrap_err();
        let kind = error.error().clone();
        let (target, mask) = error.into_parts();
        assert_eq!(depth(&target), target_depth);
        assert_eq!(target.shape(), target_shape);
        assert_eq!(
            (mask.shape(), mask.strides()),
            (&*mask_shape, &*mask_strides)
        );
        kind
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
        IncompatibleOptionalMaskError::Layout { .. }
    ));
    let offset_err = err(
        data_target(data_subset.clone()),
        mask_view(1, &SHAPE, ArraySubset::new_with_ranges(&[0..2, 0..2])),
    );
    assert!(matches!(
        offset_err,
        IncompatibleOptionalMaskError::Layout {
            mask_offset: 0,
            data_offset: 5,
            ..
        }
    ));
    assert_eq!(
        offset_err.to_string(),
        "the optional mask layout (offset 0, strides [4, 1]) does not address the same elements as the data layout (offset 5, strides [4, 1])"
    );
    for mask_subset in [
        ArraySubset::new_with_ranges(&[1..3, 1..2]),
        ArraySubset::new_with_start_shape(vec![1, 1], vec![0, 2]).unwrap(),
    ] {
        assert!(matches!(
            err(
                data_target(data_subset.clone()),
                mask_view(1, &SHAPE, mask_subset)
            ),
            IncompatibleOptionalMaskError::Shape { .. }
        ));
    }
    // A mask of a different array with the same layout is compatible
    let mut padded_mask = [0u8; 15];
    // SAFETY: the mask is the only view of its buffer.
    let padded_mask_view = unsafe {
        ArrayBytesFixedDisjointView::new(
            UnsafeCellSlice::new(&mut padded_mask),
            1,
            &[5, 3],
            ArraySubset::new_with_ranges(&[1..2, 0..3]),
        )
    }
    .unwrap();
    // SAFETY: the target is the only view of its buffer.
    let mut row = [0u8; 24];
    let row_target: ArrayBytesDecodeIntoTarget<'_> = unsafe {
        ArrayBytesFixedDisjointView::new(
            UnsafeCellSlice::new(&mut row),
            2,
            &[4, 3],
            ArraySubset::new_with_ranges(&[1..2, 0..3]),
        )
    }
    .unwrap()
    .into();
    assert!(row_target.with_optional_mask(padded_mask_view).is_ok());

    // Nested masks are validated against the innermost data
    let nested = data_target(data_subset.clone())
        .with_optional_mask(mask_view(1, &SHAPE, data_subset.clone()))
        .unwrap();
    // SAFETY: the outer mask is the only view of its buffer.
    let outer_mask_view = unsafe {
        ArrayBytesFixedDisjointView::new(
            outer_mask_slice,
            1,
            &SHAPE,
            ArraySubset::new_with_ranges(&[0..2, 1..3]),
        )
    }
    .unwrap();
    assert!(matches!(
        err(nested, outer_mask_view),
        IncompatibleOptionalMaskError::Layout { .. }
    ));

    // Empty views reference no elements, so only their shapes must match
    let empty =
        |start: Vec<u64>, shape: Vec<u64>| ArraySubset::new_with_start_shape(start, shape).unwrap();
    let num_elements = data_target(empty(vec![0, 0], vec![0, 4]))
        .with_optional_mask(mask_view(1, &SHAPE, empty(vec![0, 0], vec![0, 4])))
        .unwrap()
        .num_elements();
    assert_eq!(num_elements, 0);
    assert!(
        data_target(empty(vec![0, 0], vec![0, 4]))
            .with_optional_mask(mask_view(1, &SHAPE, empty(vec![3, 0], vec![0, 4])))
            .is_ok()
    );
    assert!(matches!(
        err(
            data_target(empty(vec![0, 0], vec![0, 4])),
            mask_view(1, &SHAPE, empty(vec![0, 0], vec![0, 3]))
        ),
        IncompatibleOptionalMaskError::Shape { .. }
    ));
}

#[test]
fn with_optional_masks_returns_inputs_on_error() {
    let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
    let other = ArraySubset::new_with_ranges(&[0..2, 0..1]);
    let mut data = [0u8; 24];
    let mut masks = [[0u8; 12]; 2];
    // SAFETY: each view is the only view of its backing buffer.
    let data_view = unsafe {
        ArrayBytesFixedDisjointView::new(UnsafeCellSlice::new(&mut data), 2, &SHAPE, subset.clone())
    }
    .unwrap();
    let mask_views: Vec<_> = masks
        .iter_mut()
        .zip([&subset, &other])
        .map(|(mask, mask_subset)| {
            // SAFETY: each view is the only view of its backing buffer.
            unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(mask),
                    1,
                    &SHAPE,
                    mask_subset.clone(),
                )
            }
            .unwrap()
        })
        .collect();

    let error = ArrayBytesDecodeIntoTarget::from(data_view)
        .with_optional_masks(mask_views)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(
        error.error(),
        IncompatibleOptionalMaskError::Shape { .. }
    ));
    let (target, mask_views) = error.into_parts();
    assert_eq!(depth(&target), 0);
    assert_eq!(target.shape(), subset.shape());
    let mask_shapes: Vec<_> = mask_views.iter().map(|mask| mask.shape()).collect();
    assert_eq!(mask_shapes, [subset.shape(), other.shape()]);
}

#[test]
fn transformed_optional_targets_keep_masks_aligned() {
    let subset = ArraySubset::new_with_ranges(&[1..3, 0..4]);
    let source_data: Vec<u8> = (10..26).collect();
    let source_masks = [[1, 0, 1, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0, 0, 1]];
    // The (row, col) of each element of a C-order source in the transformed target
    let permuted: Vec<(usize, usize)> = (0..4)
        .flat_map(|col| (1..3).map(move |row| (row, col)))
        .collect();
    let reshaped: Vec<(usize, usize)> = (1..3)
        .flat_map(|row| (0..4).map(move |col| (row, col)))
        .collect();
    for (transform, positions) in [("permuted", permuted), ("reshaped", reshaped)] {
        for depth_ in [0, 1, 2] {
            let mut data = [0xa5; 24];
            let mut masks = vec![[0xa5; 12]; depth_];
            {
                let mut target = target(&mut data, &mut masks, &subset);
                let mut transformed = match transform {
                    "permuted" => target.permuted_axes(&[1, 0]).unwrap(),
                    _ => target.reshaped(&[4, 2]).unwrap(),
                };
                assert_eq!(depth(&transformed), depth_);
                assert_eq!(transformed.shape(), [4, 2]);
                let source = ArrayBytes::new_flen(&source_data)
                    .with_optional_masks(source_masks[..depth_].iter().map(<[u8; 8]>::as_slice));
                decode_into_array_bytes_target(&source, &mut transformed).unwrap();
            }
            let mut expected_data = [0xa5; 24];
            let mut expected_masks = vec![[0xa5; 12]; depth_];
            for (k, &(row, col)) in positions.iter().enumerate() {
                let index = row * 4 + col;
                expected_data[index * 2..index * 2 + 2]
                    .copy_from_slice(&source_data[k * 2..k * 2 + 2]);
                for (expected_mask, mask) in expected_masks.iter_mut().zip(&source_masks) {
                    expected_mask[index] = mask[k];
                }
            }
            assert_eq!(data, expected_data, "{transform} {depth_}");
            assert_eq!(masks, expected_masks, "{transform} {depth_}");
        }
    }

    // Transforms that are not expressible return `None`
    let mut data = [0u8; 24];
    let mut masks = vec![[0u8; 12]; 1];
    let mut target = target(
        &mut data,
        &mut masks,
        &ArraySubset::new_with_ranges(&[1..3, 1..4]),
    );
    assert!(target.permuted_axes(&[0, 0]).is_none());
    assert!(target.reshaped(&[5]).is_none());
    // The rows of a region narrower than the array are not uniformly spaced
    assert!(target.reshaped(&[6]).is_none());
    assert!(target.reshaped(&[2, 1, 3]).is_some());
    let mut transposed = target.permuted_axes(&[1, 0]).unwrap();
    assert!(transposed.reshaped(&[6]).is_none());
    assert!(transposed.reshaped(&[3, 1, 2]).is_some());
}
