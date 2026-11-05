#![allow(missing_docs)]

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::ArraySubset;
use zarrs_codec::{ArrayBytesFixedDisjointView, CodecError, InvalidBytesLengthError};

#[test]
fn contiguous_and_strided_copies_preserve_surrounding_bytes() {
    let shape = [4, 5];
    for (rows, columns, region_lengths) in [(1..3, 0..5, vec![20]), (0..4, 1..4, vec![6, 6, 6, 6])]
    {
        let elements = (rows.end - rows.start) * (columns.end - columns.start);
        let source: Vec<_> = (0..u8::try_from(elements * 2).unwrap()).collect();
        for transform in [false, true] {
            let mut bytes = [0xa5; 40];
            let mut copied_regions = Vec::new();
            {
                // SAFETY: this is the only view of bytes.
                let mut view = unsafe {
                    ArrayBytesFixedDisjointView::new(
                        UnsafeCellSlice::new(&mut bytes),
                        2,
                        &shape,
                        ArraySubset::new_with_ranges(&[rows.clone(), columns.clone()]),
                    )
                    .unwrap()
                };
                if transform {
                    view.try_copy_from_slice_with(&source, |source, destination| {
                        copied_regions.push(source.to_vec());
                        assert_eq!(source.len(), destination.len());
                        assert_eq!(source.len() % 2, 0);
                        for (source, destination) in source
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .zip(destination.as_chunks_mut::<2>().0)
                        {
                            destination.copy_from_slice(&[source[1], source[0]]);
                        }
                        Ok::<_, InvalidBytesLengthError>(())
                    })
                    .unwrap();
                } else {
                    view.copy_from_slice(&source).unwrap();
                }
            }
            let mut expected = [0xa5; 40];
            let mut source_elements = source.as_chunks::<2>().0.iter();
            for row in rows.clone() {
                for column in columns.clone() {
                    let offset = usize::try_from((row * 5 + column) * 2).unwrap();
                    let source = source_elements.next().unwrap();
                    let swapped = [source[1], source[0]];
                    expected[offset..offset + 2].copy_from_slice(if transform {
                        &swapped
                    } else {
                        source
                    });
                }
            }
            assert_eq!(bytes, expected);
            if transform {
                assert_eq!(
                    copied_regions.iter().map(Vec::len).collect::<Vec<_>>(),
                    region_lengths
                );
                assert_eq!(copied_regions.concat(), source);
            }
        }
    }
}

#[test]
fn malformed_lengths_are_rejected_before_callbacks() {
    let shape = [4, 5];
    for subset in [
        ArraySubset::new_with_ranges(&[1..3, 0..5]),
        ArraySubset::new_with_ranges(&[0..4, 1..4]),
        ArraySubset::new_with_start_shape(vec![99, 99], vec![0, 2]).unwrap(),
    ] {
        let expected_len = subset.num_elements_usize() * 2;
        let mut bytes = [0xa5; 40];
        {
            // SAFETY: this is the only view of bytes.
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut bytes),
                    2,
                    &shape,
                    subset,
                )
                .unwrap()
            };
            for length in [
                0,
                expected_len.saturating_sub(1),
                expected_len + 1,
                expected_len + 2,
            ] {
                if length == expected_len {
                    continue;
                }
                let error = view
                    .try_copy_from_slice_with(
                        &vec![0; length],
                        |_, _| -> Result<(), InvalidBytesLengthError> {
                            panic!("invalid lengths must not invoke the callback")
                        },
                    )
                    .unwrap_err();
                assert_eq!(
                    error.to_string(),
                    InvalidBytesLengthError::new(length, expected_len).to_string()
                );
            }
        }
        assert_eq!(bytes, [0xa5; 40]);
    }
}

#[test]
fn empty_views_do_not_invoke_callbacks() {
    let shape = [4, 5];
    for subset in [
        ArraySubset::new_with_ranges(&[1..1, 1..4]),
        ArraySubset::new_with_start_shape(vec![99, 99], vec![0, 0]).unwrap(),
    ] {
        let mut bytes = [0xa5; 40];
        {
            // SAFETY: this is the only view of bytes.
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut bytes),
                    2,
                    &shape,
                    subset,
                )
                .unwrap()
            };
            view.try_copy_from_slice_with(&[], |_, _| -> Result<(), InvalidBytesLengthError> {
                panic!("empty views must not invoke the callback")
            })
            .unwrap();
        }
        assert_eq!(bytes, [0xa5; 40]);
    }
}

#[test]
fn callback_errors_stop_subsequent_regions() {
    let shape = [4, 5];
    let source: Vec<u8> = (0..16).collect();
    for failing_region in [0, 1, 3] {
        let mut bytes = [0xa5; 40];
        let mut calls = 0;
        {
            // SAFETY: this is the only view of bytes.
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut bytes),
                    2,
                    &shape,
                    ArraySubset::new_with_ranges(&[0..4, 1..3]),
                )
                .unwrap()
            };
            let error = view
                .try_copy_from_slice_with(&source, |source, destination| {
                    let region = calls;
                    calls += 1;
                    destination.copy_from_slice(source);
                    if region == failing_region {
                        Err(CodecError::Other(format!("failed region {region}")))
                    } else {
                        Ok(())
                    }
                })
                .unwrap_err();
            assert!(
                matches!(error, CodecError::Other(message) if message == format!("failed region {failing_region}"))
            );
        }
        assert_eq!(calls, failing_region + 1);
        // A callback may write before failing; callers cannot rely on rollback.
        let mut expected = [0xa5; 40];
        for region in 0..=failing_region {
            let destination_offset = region * 10 + 2;
            expected[destination_offset..destination_offset + 4]
                .copy_from_slice(&source[region * 4..region * 4 + 4]);
        }
        assert_eq!(bytes, expected);
    }
}
