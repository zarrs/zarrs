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
    use crate::ArrayIndicesTinyVec;
    use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

    use crate::ArraySubset;

    #[test]
    fn array_subset_iter_indices() {
        let region = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        let indices = region.indices();

        let mut iter = indices.iter();
        assert_eq!(iter.size_hint(), (4, Some(4)));
        assert_eq!(iter.next(), Some(ArrayIndicesTinyVec::Heap(vec![1, 1])));
        assert_eq!(
            iter.next_back(),
            Some(ArrayIndicesTinyVec::Heap(vec![2, 2]))
        );
        assert_eq!(iter.next(), Some(ArrayIndicesTinyVec::Heap(vec![1, 2])));
        assert_eq!(iter.next(), Some(ArrayIndicesTinyVec::Heap(vec![2, 1])));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);

        let expected = vec![
            ArrayIndicesTinyVec::Heap(vec![1, 1]),
            ArrayIndicesTinyVec::Heap(vec![1, 2]),
            ArrayIndicesTinyVec::Heap(vec![2, 1]),
            ArrayIndicesTinyVec::Heap(vec![2, 2]),
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
        assert_eq!(indices.clone().into_iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.into_par_iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn array_subset_iter_contiguous_indices1() {
        let region = ArraySubset::new_with_shape(vec![2, 2]);
        let indices = region.contiguous_indices(&[2, 2]).unwrap();
        let mut iter = indices.into_iter();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        assert_eq!(iter.contiguous_elements(), 4);
        assert_eq!(
            iter.next(),
            Some((ArrayIndicesTinyVec::Heap(vec![0, 0]), 4))
        );
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
        assert_eq!(
            iter.next_back(),
            Some((ArrayIndicesTinyVec::Heap(vec![2, 1]), 2))
        );
        assert_eq!(
            iter.next(),
            Some((ArrayIndicesTinyVec::Heap(vec![1, 1]), 2))
        );
        assert_eq!(iter.next(), None);

        let expected = vec![
            (ArrayIndicesTinyVec::Heap(vec![1, 1]), 2),
            (ArrayIndicesTinyVec::Heap(vec![2, 1]), 2),
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

        let expected = vec![(ArrayIndicesTinyVec::Heap(vec![1, 0, 0, 0]), 8)];
        assert_eq!(indices.iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.clone().into_iter().collect::<Vec<_>>(), expected);
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
}
