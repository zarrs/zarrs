//! Array subset iterators.
//!
//! The iterators are:
//!  - [`Indices`]: iterate over the multidimensional indices of the elements in a region.
//!  - [`LinearisedIndices`]: iterate over linearised indices of the elements in a region.
//!  - [`ContiguousIndices`]: iterate over contiguous sets of elements in the region with the start a multidimensional index.
//!  - [`ContiguousLinearisedIndices`]: iterate over contiguous sets of elements in the region with the start a linearised index.
//!
//! These can be created with the following concrete [`ArraySubset`](crate::ArraySubset) methods or [`Indexer`](crate::Indexer) trait methods:
//! [`indices`](crate::ArraySubset::indices) or [`iter_indices`](crate::Indexer::iter_indices) ,
//! [`linearised_indices`](crate::ArraySubset::linearised_indices) or [`iter_linearised_indices`](crate::Indexer::iter_linearised_indices),
//! [`contiguous_linearised_indices`](crate::ArraySubset::contiguous_linearised_indices) or [`contiguous_linearised_indices`](crate::Indexer::iter_contiguous_linearised_indices).
//!
//! All iterators support [`into_iter()`](IntoIterator::into_iter) ([`IntoIterator`]).
//! The [`Indices`] iterator also supports [`rayon`]'s [`into_par_iter()`](rayon::iter::IntoParallelIterator::into_par_iter) ([`IntoParallelIterator`](rayon::iter::IntoParallelIterator)).

mod contiguous_indices_iterator;
mod contiguous_linearised_indices_iterator;
mod indices_iterator;
mod linearised_indices_iterator;
mod strided_indices;

pub use contiguous_indices_iterator::{
    ContiguousIndices, ContiguousIndicesIntoIterator, ContiguousIndicesIterator,
};
pub use contiguous_linearised_indices_iterator::{
    ContiguousLinearisedIndices, ContiguousLinearisedIndicesIntoIterator,
    ContiguousLinearisedIndicesIterator,
};
pub use indices_iterator::{
    Indices, IndicesIntoIterator, IndicesIterator, ParIndicesIntoIterator, ParIndicesIterator,
};
pub use linearised_indices_iterator::{
    LinearisedIndices, LinearisedIndicesIntoIterator, LinearisedIndicesIterator,
};

#[cfg(test)]
mod tests {
    use crate::ArrayIndices;
    use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

    use crate::{ArraySubset, IndexerError};

    #[test]
    fn array_subset_iter_indices() {
        let region = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        let indices = region.indices();

        let mut iter = indices.iter();
        assert_eq!(iter.size_hint(), (4, Some(4)));
        assert_eq!(iter.next(), Some(ArrayIndices::Heap(vec![1, 1])));
        assert_eq!(iter.next_back(), Some(ArrayIndices::Heap(vec![2, 2])));
        assert_eq!(iter.next(), Some(ArrayIndices::Heap(vec![1, 2])));
        assert_eq!(iter.next(), Some(ArrayIndices::Heap(vec![2, 1])));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);

        let expected = vec![
            ArrayIndices::Heap(vec![1, 1]),
            ArrayIndices::Heap(vec![1, 2]),
            ArrayIndices::Heap(vec![2, 1]),
            ArrayIndices::Heap(vec![2, 2]),
        ];
        assert_eq!(indices.iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.clone().into_iter().collect::<Vec<_>>(), expected);
        assert_eq!((&indices).into_par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.into_par_iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn array_subset_iter_linearised_indices() {
        let region = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        assert!(region.linearised_indices(&[4, 4, 4]).is_err());
        let indices = region.linearised_indices(&[4, 4]).unwrap();
        //  0  1  2  3
        //  4  5  6  7
        //  8  9 10 11
        // 12 13 14 15

        let mut iter = indices.iter();
        assert_eq!(iter.size_hint(), (4, Some(4)));
        assert_eq!(iter.next_back(), Some(10));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), Some(6));
        assert_eq!(iter.next(), Some(9));
        assert_eq!(iter.next(), None);

        let expected = vec![5, 6, 9, 10];
        assert_eq!(indices.iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.into_iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.into_par_iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn array_subset_iter_contiguous_indices1() {
        let region = ArraySubset::new_with_shape(vec![2, 2]);
        let indices = region.contiguous_indices(&[2, 2]).unwrap();
        let mut iter = indices.into_iter();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        assert_eq!(iter.contiguous_elements(), 4);
        assert_eq!(iter.next(), Some((ArrayIndices::Heap(vec![0, 0]), 4)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn array_subset_iter_contiguous_indices2() {
        let region = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        let indices = region.contiguous_indices(&[4, 4]).unwrap();
        assert_eq!(indices.len(), 2);
        assert!(!indices.is_empty());
        assert_eq!(indices.contiguous_elements_usize(), 2);

        let mut iter = indices.iter();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        assert_eq!(iter.contiguous_elements(), 2);
        assert_eq!(iter.next_back(), Some((ArrayIndices::Heap(vec![2, 1]), 2)));
        assert_eq!(iter.next(), Some((ArrayIndices::Heap(vec![1, 1]), 2)));
        assert_eq!(iter.next(), None);

        let expected = vec![
            (ArrayIndices::Heap(vec![1, 1]), 2),
            (ArrayIndices::Heap(vec![2, 1]), 2),
        ];
        assert_eq!(indices.iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.clone().into_iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.into_par_iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn array_subset_iter_contiguous_indices3() {
        let region = ArraySubset::new_with_ranges(&[1..3, 0..1, 0..2, 0..2]);
        let indices = region.contiguous_indices(&[3, 1, 2, 2]).unwrap();

        let expected = vec![(ArrayIndices::Heap(vec![1, 0, 0, 0]), 8)];
        assert_eq!(indices.iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.into_iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.into_par_iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn array_subset_iter_continuous_linearised_indices() {
        let region = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        let indices = region.contiguous_linearised_indices(&[4, 4]).unwrap();
        assert_eq!(indices.len(), 2);
        assert!(!indices.is_empty());
        assert_eq!(indices.contiguous_elements_usize(), 2);

        let mut iter = indices.iter();
        //  0  1  2  3
        //  4  5  6  7
        //  8  9 10 11
        // 12 13 14 15
        assert_eq!(iter.size_hint(), (2, Some(2)));
        assert_eq!(iter.contiguous_elements(), 2);
        assert_eq!(iter.next_back(), Some((9, 2)));
        assert_eq!(iter.next(), Some((5, 2)));
        assert_eq!(iter.next(), None);

        let expected = vec![(5, 2), (9, 2)];
        assert_eq!(indices.iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.clone().into_iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.into_par_iter().collect::<Vec<_>>(), expected);
    }

    /// Check iterating forwards, backwards, and alternating between the front and the back.
    fn check_double_ended<I>(iter: I, expected: &[I::Item], mut take_front: impl FnMut() -> bool)
    where
        I: DoubleEndedIterator + ExactSizeIterator + Clone,
        I::Item: PartialEq + std::fmt::Debug + Clone,
    {
        assert_eq!(iter.len(), expected.len());
        assert_eq!(iter.clone().collect::<Vec<_>>(), expected);
        assert_eq!(
            iter.clone().rev().collect::<Vec<_>>(),
            expected.iter().rev().cloned().collect::<Vec<_>>()
        );
        let mut iter = iter;
        let (mut front, mut back) = (0, expected.len());
        while front < back {
            if take_front() {
                assert_eq!(iter.next().as_ref(), Some(&expected[front]));
                front += 1;
            } else {
                back -= 1;
                assert_eq!(iter.next_back().as_ref(), Some(&expected[back]));
            }
            assert_eq!(iter.len(), back - front);
        }
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }

    /// Every iterator matches enumerating the elements of random subsets without an iterator.
    #[test]
    fn array_subset_iters_match_brute_force() {
        use rayon::iter::IndexedParallelIterator;

        // A linear congruential generator, for reproducible pseudo-random cases
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut random = |bound: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % bound
        };

        for _ in 0..2000 {
            let dimensionality = usize::try_from(random(6)).unwrap();
            let array_shape: Vec<u64> = (0..dimensionality).map(|_| 1 + random(5)).collect();
            let ranges: Vec<_> = array_shape
                .iter()
                .map(|&size| {
                    let start = random(size + 1);
                    start..start + random(size - start + 1)
                })
                .collect();
            let region = ArraySubset::new_with_ranges(&ranges);

            // Enumerate the elements of the region without an iterator
            let num_elements = region.num_elements_usize();
            let expected_indices: Vec<ArrayIndices> = (0..num_elements as u64)
                .map(|mut element| {
                    let mut indices: ArrayIndices = region.start().iter().copied().collect();
                    for (index, &size) in std::iter::zip(&mut indices, region.shape()).rev() {
                        *index += element % size;
                        element /= size;
                    }
                    indices
                })
                .collect();
            let expected_linearised: Vec<u64> = expected_indices
                .iter()
                .map(|indices| crate::ravel_indices(indices, &array_shape).unwrap())
                .collect();

            let indices = region.indices();
            check_double_ended(indices.iter(), &expected_indices, || random(2) == 0);
            assert_eq!(
                indices.clone().into_iter().collect::<Vec<_>>(),
                expected_indices
            );
            assert_eq!(
                indices.par_iter().with_max_len(1).collect::<Vec<_>>(),
                expected_indices
            );

            let linearised = region.linearised_indices(&array_shape).unwrap();
            check_double_ended(linearised.iter(), &expected_linearised, || random(2) == 0);
            assert_eq!(
                linearised.into_iter().collect::<Vec<_>>(),
                expected_linearised
            );

            // Contiguous runs partition the linearised indices into runs of consecutive indices
            let contiguous = region.contiguous_linearised_indices(&array_shape).unwrap();
            let run_length = contiguous.contiguous_elements_usize();
            let expected_runs: Vec<(u64, u64)> = if num_elements == 0 {
                vec![]
            } else {
                expected_linearised
                    .chunks(run_length)
                    .map(|run| {
                        assert!(run.windows(2).all(|pair| pair[1] == pair[0] + 1));
                        (run[0], run_length as u64)
                    })
                    .collect()
            };
            check_double_ended(contiguous.iter(), &expected_runs, || random(2) == 0);
            assert_eq!(contiguous.into_iter().collect::<Vec<_>>(), expected_runs);

            let contiguous = region.contiguous_indices(&array_shape).unwrap();
            let expected_runs: Vec<(ArrayIndices, u64)> = if num_elements == 0 {
                vec![]
            } else {
                expected_indices
                    .chunks(run_length)
                    .map(|run| (run[0].clone(), run_length as u64))
                    .collect()
            };
            check_double_ended(contiguous.iter(), &expected_runs, || random(2) == 0);
            assert_eq!(contiguous.into_iter().collect::<Vec<_>>(), expected_runs);
        }
    }

    /// An empty array subset references no elements, so it is in-bounds irrespective of its start.
    /// `LinearisedIndices` and `ContiguousIndices` must agree on this.
    #[test]
    fn array_subset_iter_empty_subset_bounds() {
        for (region, array_shape) in [
            // ... at the origin
            (ArraySubset::new_with_ranges(&[0..0, 0..2]), vec![4, 4]),
            // ... at the end of a dimension
            (ArraySubset::new_with_ranges(&[4..4]), vec![4]),
            (ArraySubset::new_with_ranges(&[4..4, 1..3]), vec![4, 4]),
            // ... beyond the end of a dimension
            (ArraySubset::new_with_ranges(&[5..5]), vec![4]),
            (ArraySubset::new_with_ranges(&[9..9, 1..3]), vec![4, 4]),
            // ... of a zero-sized array
            (ArraySubset::new_with_ranges(&[0..0, 0..2]), vec![0, 4]),
        ] {
            let linearised = region.linearised_indices(&array_shape).unwrap();
            assert_eq!(linearised.into_iter().count(), 0);
            let contiguous = region.contiguous_indices(&array_shape).unwrap();
            assert_eq!(contiguous.into_iter().count(), 0);
            let contiguous = region.contiguous_linearised_indices(&array_shape).unwrap();
            assert_eq!(contiguous.into_iter().count(), 0);
            assert!(region.inbounds_shape(&array_shape));
        }

        // ... but an empty subset with an incompatible dimensionality is not in-bounds
        let region = ArraySubset::new_with_ranges(&[0..0]);
        assert!(matches!(
            region.linearised_indices(&[4, 4]),
            Err(IndexerError::IncompatibleDimensionality(_))
        ));
        assert!(matches!(
            region.contiguous_indices(&[4, 4]),
            Err(IndexerError::IncompatibleDimensionality(_))
        ));
        assert!(!region.inbounds_shape(&[4, 4]));
    }
}
