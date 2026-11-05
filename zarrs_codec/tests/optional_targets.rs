#![allow(missing_docs)]

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::ArraySubset;
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView,
    build_nested_optional_target, decode_into_array_bytes_target, extract_target_views,
};

#[test]
fn optional_targets_preserve_data_and_outer_to_inner_masks() {
    let shape = [3, 4];
    let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
    let source_data = [10, 11, 12, 13, 14, 15, 16, 17];
    let source_masks = [[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]];

    for depth in [0, 1, 3] {
        let mut data = [0xa5; 24];
        let mut masks = vec![[0xa5; 12]; depth];
        {
            // SAFETY: each view exclusively references its own backing buffer.
            let mut data_view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut data),
                    2,
                    &shape,
                    subset.clone(),
                )
                .unwrap()
            };
            let mut mask_views: Vec<_> = masks
                .iter_mut()
                .map(|mask| unsafe {
                    // SAFETY: each mask has exactly one view.
                    ArrayBytesFixedDisjointView::new(
                        UnsafeCellSlice::new(mask),
                        1,
                        &shape,
                        subset.clone(),
                    )
                    .unwrap()
                })
                .collect();
            let expected_data = std::ptr::from_ref(&data_view);
            let expected_masks: Vec<_> = mask_views.iter().map(std::ptr::from_ref).collect();
            let target = build_nested_optional_target(&mut data_view, &mut mask_views);

            // Check the nesting independently of the extraction helper.
            let mut inner = &target;
            for expected_mask in &expected_masks {
                let ArrayBytesDecodeIntoTarget::Optional(next, mask) = inner else {
                    panic!("missing optional layer");
                };
                assert_eq!(std::ptr::from_ref(&**mask), *expected_mask);
                inner = next;
            }
            let ArrayBytesDecodeIntoTarget::Fixed(view) = inner else {
                panic!("expected innermost data view");
            };
            assert_eq!(std::ptr::from_ref(&**view), expected_data);

            let (view, extracted_masks) = extract_target_views(&target);
            assert_eq!(std::ptr::from_ref(view), expected_data);
            assert_eq!(
                extracted_masks
                    .into_iter()
                    .map(std::ptr::from_ref)
                    .collect::<Vec<_>>(),
                expected_masks
            );

            let mut source = ArrayBytes::new_flen(&source_data);
            for mask in source_masks[..depth].iter().rev() {
                source = source.with_optional_mask(mask.as_slice());
            }
            decode_into_array_bytes_target(&source, target).unwrap();
        }

        let mut expected_data = [0xa5; 24];
        expected_data[10..14].copy_from_slice(&source_data[..4]);
        expected_data[18..22].copy_from_slice(&source_data[4..]);
        assert_eq!(data, expected_data);
        for (mask, source) in masks.iter().zip(source_masks) {
            let mut expected = [0xa5; 12];
            expected[5..7].copy_from_slice(&source[..2]);
            expected[9..11].copy_from_slice(&source[2..]);
            assert_eq!(*mask, expected);
        }
    }
}
