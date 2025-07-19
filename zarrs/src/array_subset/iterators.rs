//! Array subset iterators.
//!
//! The iterators are:
//!  - [`Indices`]: iterate over the multidimensional indices of the elements in the subset.
//!  - [`LinearisedIndices`]: iterate over linearised indices of the elements in the subset.
//!  - [`ContiguousIndices`]: iterate over contiguous sets of elements in the subset with the start a multidimensional index.
//!  - [`ContiguousLinearisedIndices`]: iterate over contiguous sets of elements in the subset with the start a linearised index.
//!  - [`Chunks`]: iterate over regular sized chunks in the array subset.
//!
//! These can be created with the appropriate [`ArraySubset`](super::ArraySubset) methods including
//! [`indices`](super::ArraySubset::indices),
//! [`linearised_indices`](super::ArraySubset::linearised_indices),
//! [`contiguous_indices`](super::ArraySubset::contiguous_indices),
//! [`contiguous_linearised_indices`](super::ArraySubset::contiguous_linearised_indices), and
//! [`chunks`](super::ArraySubset::chunks).
//!
//! All iterators support [`into_iter()`](IntoIterator::into_iter) ([`IntoIterator`]).
//! The [`Indices`] and [`Chunks`] iterators also support [`rayon`]'s [`into_par_iter()`](rayon::iter::IntoParallelIterator::into_par_iter) ([`IntoParallelIterator`](rayon::iter::IntoParallelIterator)).

mod chunks_iterator;
mod contiguous_indices_iterator;
mod contiguous_linearised_indices_iterator;
mod indices_iterator;
mod linearised_indices_iterator;

pub use chunks_iterator::{
    Chunks, ChunksIntoIterator, ChunksIterator, ParChunksIntoIterator, ParChunksIterator,
};
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
    use std::num::NonZeroU64;

    use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

    use crate::array_subset::ArraySubset;

    #[test]
    fn array_subset_iter_indices() {
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        let indices = subset.indices();

        let mut iter = indices.iter();
        assert_eq!(iter.size_hint(), (4, Some(4)));
        assert_eq!(iter.next(), Some(vec![1, 1]));
        assert_eq!(iter.next_back(), Some(vec![2, 2]));
        assert_eq!(iter.next(), Some(vec![1, 2]));
        assert_eq!(iter.next(), Some(vec![2, 1]));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);

        let expected = vec![vec![1, 1], vec![1, 2], vec![2, 1], vec![2, 2]];
        assert_eq!(indices.iter().collect::<Vec<_>>(), expected);
        assert_eq!((&indices).par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.clone().into_iter().collect::<Vec<_>>(), expected);
        assert_eq!((&indices).into_par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.into_par_iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn array_subset_iter_linearised_indices() {
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        assert!(subset.linearised_indices(&[4, 4, 4]).is_err());
        let indices = subset.linearised_indices(&[4, 4]).unwrap();
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
        let subset = ArraySubset::new_with_shape(vec![2, 2]);
        let indices = subset.contiguous_indices(&[2, 2]).unwrap();
        let mut iter = indices.into_iter();
        assert_eq!(iter.size_hint(), (1, Some(1)));
        assert_eq!(iter.contiguous_elements(), 4);
        assert_eq!(iter.next(), Some((vec![0, 0], 4)));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn array_subset_iter_contiguous_indices2() {
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        let indices = subset.contiguous_indices(&[4, 4]).unwrap();
        assert_eq!(indices.len(), 2);
        assert!(!indices.is_empty());
        assert_eq!(indices.contiguous_elements_usize(), 2);

        let mut iter = indices.iter();
        assert_eq!(iter.size_hint(), (2, Some(2)));
        assert_eq!(iter.contiguous_elements(), 2);
        assert_eq!(iter.next_back(), Some((vec![2, 1], 2)));
        assert_eq!(iter.next(), Some((vec![1, 1], 2)));
        assert_eq!(iter.next(), None);

        let expected = vec![(vec![1, 1], 2), (vec![2, 1], 2)];
        assert_eq!(indices.iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.clone().into_iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.into_par_iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn array_subset_iter_contiguous_indices3() {
        let subset = ArraySubset::new_with_ranges(&[1..3, 0..1, 0..2, 0..2]);
        let indices = subset.contiguous_indices(&[3, 1, 2, 2]).unwrap();

        let expected = vec![(vec![1, 0, 0, 0], 8)];
        assert_eq!(indices.iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(indices.clone().into_iter().collect::<Vec<_>>(), expected);
        // assert_eq!(indices.into_par_iter().collect::<Vec<_>>(), expected);
    }

    #[test]
    fn array_subset_iter_continuous_linearised_indices() {
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        let indices = subset.contiguous_linearised_indices(&[4, 4]).unwrap();
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

    #[test]
    #[rustfmt::skip]
    fn array_subset_iter_chunks1() {
        let subset = ArraySubset::new_with_ranges(&[1..5, 1..5]);
        let chunk_shape_invalid = [NonZeroU64::new(2).unwrap()];
        assert!(subset.chunks(&chunk_shape_invalid).is_err());
        let chunk_shape = [NonZeroU64::new(2).unwrap(), NonZeroU64::new(2).unwrap()];
        let chunks = subset.chunks(&chunk_shape).unwrap();
        assert!(!chunks.is_empty());

        let mut iter = chunks.iter();
        assert_eq!(iter.size_hint(), (9, Some(9)));
        assert_eq!(iter.next().unwrap(), (vec![0, 0], ArraySubset::new_with_ranges(&[0..2, 0..2])));
        assert_eq!(iter.next_back().unwrap(), (vec![2, 2], ArraySubset::new_with_ranges(&[4..6, 4..6])));
        assert_eq!(iter.next().unwrap(), (vec![0, 1], ArraySubset::new_with_ranges(&[0..2, 2..4])));
        assert_eq!(iter.next().unwrap(), (vec![0, 2], ArraySubset::new_with_ranges(&[0..2, 4..6])));
        assert_eq!(iter.next().unwrap(), (vec![1, 0], ArraySubset::new_with_ranges(&[2..4, 0..2])));
        assert_eq!(iter.next().unwrap(), (vec![1, 1], ArraySubset::new_with_ranges(&[2..4, 2..4])));
        assert_eq!(iter.next().unwrap(), (vec![1, 2], ArraySubset::new_with_ranges(&[2..4, 4..6])));
        assert_eq!(iter.next().unwrap(), (vec![2, 0], ArraySubset::new_with_ranges(&[4..6, 0..2])));
        assert_eq!(iter.next().unwrap(), (vec![2, 1], ArraySubset::new_with_ranges(&[4..6, 2..4])));
        assert!(iter.next().is_none());
    }

    #[test]
    #[rustfmt::skip]
    fn array_subset_iter_chunks2() {
        let subset = ArraySubset::new_with_ranges(&[2..5, 2..6]);
        let chunk_shape = [NonZeroU64::new(2).unwrap(), NonZeroU64::new(3).unwrap()];
        let chunks = subset.chunks(&chunk_shape).unwrap();

        let expected = vec![
            (vec![1, 0], ArraySubset::new_with_ranges(&[2..4, 0..3])),
            (vec![1, 1], ArraySubset::new_with_ranges(&[2..4, 3..6])),
            (vec![2, 0], ArraySubset::new_with_ranges(&[4..6, 0..3])),
            (vec![2, 1], ArraySubset::new_with_ranges(&[4..6, 3..6])),
        ];
        assert_eq!(chunks.iter().collect::<Vec<_>>(), expected);
        assert_eq!((&chunks).par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(chunks.par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(chunks.clone().into_iter().collect::<Vec<_>>(), expected);
        assert_eq!((&chunks).into_par_iter().collect::<Vec<_>>(), expected);
        assert_eq!(chunks.into_par_iter().collect::<Vec<_>>(), expected);
    }
}
