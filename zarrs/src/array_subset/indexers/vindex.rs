use std::fmt::{Debug, Display};

use crate::array::{ArrayIndices, ArrayShape};

use crate::array_subset::indexers::{Indexer, IndexerEnum};
use crate::array_subset::iterators::{
    ContiguousIndices, ContiguousLinearisedIndices, LinearisedIndices,
};
use crate::array_subset::{
    ArraySubset, IncompatibleArraySubsetAndShapeError, IncompatibleDimensionalityError,
};
use derive_more::From;
use itertools::{izip, Itertools};
use thiserror::Error;
use zarrs_storage::byte_range::ByteRange;

// TODO: sorted for now assumed

/// A vindex array subset
#[derive(Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Default)]
pub struct VIndex {
    /// The start of the array subset.
    start: ArrayIndices,
    /// The shape of the array subset.
    shape: ArrayShape,
    /// The indices themselves
    indices: Vec<Vec<u64>>,
    /// How to interpret the indices
    are_dimension_first_indices: bool,
}

fn transpose(v: &Vec<Vec<u64>>) -> Vec<Vec<u64>> {
    (0..v[0].len())
        .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<_>>())
        .collect()
}

impl Display for VIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.indices.fmt(f)
    }
}

impl Indexer for VIndex {
    fn num_elements(&self) -> u64 {
        if self.are_dimension_first_indices {
            return self.indices[0].len() as u64;
        }
        return self.indices.len() as u64;
    }

    fn is_compatible_shape(&self, array_shape: &[u64]) -> bool {
        let is_compat_array_shape = array_shape.iter().skip(1).all_equal_value() != Ok(&1);
        if self.are_dimension_first_indices {
            return self.indices[0].len() <= (array_shape[0] as usize) && is_compat_array_shape;
        }
        self.indices.len() <= (array_shape[0] as usize) && is_compat_array_shape
    }

    fn shape(&self) -> &[u64] {
        &self.shape
    }

    fn find_linearised_index(&self, index: usize) -> ArrayIndices {
        if self.are_dimension_first_indices {
            return self.indices.iter().map(|i| i[index]).collect::<Vec<_>>();
        }
        self.indices[index].clone()
    }

    fn end_exc(&self) -> ArrayIndices {
        if self.are_dimension_first_indices {
            return self
                .indices
                .iter()
                .map(|i| i[i.len() - 1] + 1)
                .collect::<Vec<_>>();
        }
        self.indices[self.indices.len() - 1].clone()
    }

    fn end_inc(&self) -> Option<ArrayIndices> {
        if self.is_empty() {
            None
        } else {
            if self.are_dimension_first_indices {
                return Some(
                    self.indices
                        .iter()
                        .map(|i| i[i.len() - 1])
                        .collect::<Vec<_>>(),
                );
            }
            Some(self.indices[self.indices.len() - 1].clone())
        }
    }

    fn start(&self) -> &[u64] {
        &self.start
    }

    fn dimensionality(&self) -> usize {
        if self.are_dimension_first_indices {
            return self.indices.len();
        }
        return self.indices[0].len();
    }

    fn byte_ranges(
        &self,
        array_shape: &[u64],
        element_size: usize,
    ) -> Result<Vec<ByteRange>, IncompatibleArraySubsetAndShapeError> {
        let mut byte_ranges: Vec<ByteRange> = Vec::new();
        let linearised_indices = self.linearised_indices(&array_shape)?;
        for array_index in &linearised_indices {
            let byte_index = array_index * element_size as u64;
            byte_ranges.push(ByteRange::FromStart(byte_index, Some(element_size as u64)));
        }
        Ok(byte_ranges)
    }

    fn contains(&self, indices: &[u64]) -> bool {
        if self.are_dimension_first_indices {
            return indices
                .iter()
                .zip(&self.indices)
                .all(|(index, vindex)| vindex.contains(index));
        }
        self.indices.contains(&indices.to_vec())
    }

    fn overlap(
        &self,
        subset_other: &ArraySubset,
    ) -> Result<ArraySubset, IncompatibleDimensionalityError> {
        if let IndexerEnum::RangeSubset(range_subset) = &subset_other.indexer {
            if range_subset.dimensionality() == self.dimensionality() {
                let indices: Vec<Vec<u64>>;
                if self.are_dimension_first_indices {
                    indices = transpose(&self.indices)
                        .into_iter()
                        .filter(|row| {
                            izip!(row.iter(), subset_other.start(), subset_other.end_exc())
                                .all(|(i, s, e)| i >= s && i < &e)
                        })
                        .collect::<Vec<Vec<u64>>>();
                } else {
                    indices = self.indices.clone();
                }
                return Ok(IndexerEnum::VIndex(
                    Self::new_from_selection_first_indices(indices).unwrap(),
                )
                .into()); // TODO: handle error better
            } else {
                return Err(IncompatibleDimensionalityError::new(
                    range_subset.dimensionality(),
                    self.dimensionality(),
                ));
            }
        } else {
            todo!("Intersect other types or handle error more gracefully (ugh)")
        }
    }

    fn relative_to(&self, start: &[u64]) -> Result<ArraySubset, IncompatibleDimensionalityError> {
        if start.len() == self.dimensionality() {
            let iter: std::vec::IntoIter<Vec<u64>>;
            if self.are_dimension_first_indices {
                iter = transpose(&self.indices).into_iter();
            } else {
                iter = self.indices.clone().into_iter();
            }
            let indices = iter
                .filter(|row| row.iter().zip(start).all(|(i, s)| i >= s))
                .collect::<Vec<Vec<u64>>>();
            let shape = vec![indices[0].len() as u64];
            Ok(IndexerEnum::VIndex(Self {
                indices,
                start: start.to_vec(),
                shape,
                are_dimension_first_indices: false,
            })
            .into())
        } else {
            Err(IncompatibleDimensionalityError::new(
                start.len(),
                self.dimensionality(),
            ))
        }
    }
}

impl VIndex {
    fn linearised_indices(
        &self,
        array_shape: &[u64],
    ) -> Result<LinearisedIndices, IncompatibleArraySubsetAndShapeError> {
        LinearisedIndices::new(
            IndexerEnum::VIndex(self.clone()).into(),
            array_shape.to_vec(),
        ) // TODO: better handling of these iterators
    }
}

/// An incompatible array and array shape error.
#[derive(Clone, Debug, Error, From)]
#[error("At least one of the indices was not equal to the others in length: {0:?}")]
pub struct UnequalVIndexLengthsError(Vec<usize>);

impl UnequalVIndexLengthsError {
    /// Create a new incompatible array subset and shape error.
    #[must_use]
    pub fn new(indices_lengths: Vec<usize>) -> Self {
        Self(indices_lengths)
    }
}

/// An incompatible array and array shape error.
#[derive(Clone, Debug, Error, From, Default)]
#[error("Empty indices")]
pub struct EmptyVIndexError;

/// An incompatible VIndex argument
#[derive(Debug, Error)]
pub enum VIndexError {
    /// An incompatible array and array shape error.
    #[error("At least one of the indices was not equal to the others in length: {0}")]
    UnequalVIndexLengths(#[from] UnequalVIndexLengthsError),
    /// An incompatible array and array shape error.
    #[error("Empty indices")]
    EmptyVIndex(#[from] EmptyVIndexError),
}
fn check_indices(indices: &[Vec<u64>]) -> Result<(), VIndexError> {
    if indices.len() == 0 || indices[0].len() == 0 {
        return Err(EmptyVIndexError.into());
    }
    if !indices.iter().map(|x: &Vec<u64>| x.len()).all_equal() {
        return Err(
            UnequalVIndexLengthsError::new(indices.iter().map(|x| x.len()).collect()).into(),
        );
    }
    Ok(())
}

impl VIndex {
    pub fn new_from_dimension_first_indices(
        indices: Vec<ArrayIndices>,
    ) -> Result<Self, VIndexError> {
        check_indices(&indices)?;
        let shape = vec![indices[0].len() as u64];
        let start = indices.iter().map(|i| i[0]).collect::<Vec<_>>();
        Ok(Self {
            shape,
            start,
            indices,
            are_dimension_first_indices: true,
        })
    }

    pub fn new_from_selection_first_indices(
        indices: Vec<ArrayIndices>,
    ) -> Result<Self, VIndexError> {
        check_indices(&indices)?;
        let shape = vec![indices.len() as u64];
        let start = indices[0].clone();
        Ok(Self {
            shape,
            start,
            indices,
            are_dimension_first_indices: false,
        })
    }
}

#[cfg(test)]
mod tests {

    use crate::{array_subset::indexers::vindex::VIndexError, array_subset::indexers::Indexer};

    use super::VIndex;

    #[test]
    fn vindex_new_ok() {
        assert!(
            VIndex::new_from_dimension_first_indices(vec![vec![0, 1, 2, 5], vec![1, 0, 2, 5]])
                .is_ok()
        )
    }
    #[test]
    fn vindex_new_unequal() {
        assert!(
            VIndex::new_from_dimension_first_indices(vec![vec![0, 1, 2, 5], vec![1, 0, 2]])
                .is_err_and(|x| matches!(x, VIndexError::UnequalVIndexLengths(_)))
        )
    }

    #[test]
    fn vindex_new_empty() {
        assert!(VIndex::new_from_dimension_first_indices(vec![])
            .is_err_and(|x| matches!(x, VIndexError::EmptyVIndex(_))))
    }

    #[test]
    fn vindex_byte_ranges_dimension_first() {
        let indexer =
            VIndex::new_from_dimension_first_indices(vec![vec![0, 1, 2, 5], vec![1, 0, 2, 5]])
                .unwrap();
        indexer
            .byte_ranges(vec![10, 10].as_slice(), 4)
            .unwrap()
            .iter()
            .zip(vec![4, 40, 88, 220])
            .for_each(|(byte_range, expected)| {
                assert_eq!(byte_range.start(0), expected);
                assert_eq!(byte_range.end(0), expected + 4);
            });
    }

    #[test]
    fn vindex_byte_ranges_selection_first() {
        let indexer = VIndex::new_from_selection_first_indices(vec![
            vec![0, 1],
            vec![1, 0],
            vec![2, 2],
            vec![5, 5],
        ])
        .unwrap();
        indexer
            .byte_ranges(vec![10, 10].as_slice(), 4)
            .unwrap()
            .iter()
            .zip(vec![4, 40, 88, 220])
            .for_each(|(byte_range, expected)| {
                assert_eq!(byte_range.start(0), expected);
                assert_eq!(byte_range.end(0), expected + 4);
            });
    }
}
