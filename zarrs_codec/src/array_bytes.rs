use std::ops::IndexMut;

use derive_more::derive::Display;
use thiserror::Error;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::iterators::ContiguousLinearisedIndices;
use zarrs_chunk_grid::{ArraySubsetTraits, ravel_indices};
use zarrs_data_type::DataTypeFillValueError;
use zarrs_storage::byte_range::{ByteRange, InvalidByteRangeError, extract_byte_ranges_concat};

use crate::small_copy::with_small_len;
use crate::{
    ArrayBytesFixedDisjointView, ArrayBytesTarget, ArraySubset, CodecError, DataType, FillValue,
    Indexer, IndexerError, InvalidBytesLengthError, InvalidNumberOfElementsError,
};

mod array_bytes_offsets;
pub use array_bytes_offsets::{
    ArrayBytesOffsets, ArrayBytesOffsetsCreateError, ArrayBytesOffsetsElement,
    ArrayBytesOffsetsIter, ArrayBytesOffsetsRangesIter, ArrayBytesOffsetsSlice,
};

fn offsets_from_usize(
    offsets: Vec<usize>,
) -> Result<ArrayBytesOffsets, ArrayBytesOffsetsCreateError> {
    if u32::try_from(offsets.iter().copied().max().unwrap_or(0)).is_ok() {
        #[cfg(target_pointer_width = "32")]
        let offsets = bytemuck::allocation::cast_vec::<usize, u32>(offsets);
        #[cfg(not(target_pointer_width = "32"))]
        let offsets = offsets
            .into_iter()
            .map(|offset| u32::try_from(offset).unwrap())
            .collect::<Vec<_>>();
        ArrayBytesOffsets::new(offsets)
    } else {
        #[cfg(target_pointer_width = "64")]
        let offsets = bytemuck::allocation::cast_vec::<usize, u64>(offsets);
        #[cfg(not(target_pointer_width = "64"))]
        let offsets = offsets
            .into_iter()
            .map(|offset| u64::try_from(offset).unwrap())
            .collect::<Vec<_>>();
        ArrayBytesOffsets::new(offsets)
    }
}

pub use zarrs_storage::CowBytes;

mod array_bytes_variable_length;
pub use array_bytes_variable_length::ArrayBytesVariableLength;

mod array_bytes_optional;
pub use array_bytes_optional::ArrayBytesOptional;

mod array_bytes_mask;
pub use array_bytes_mask::{ArrayBytesMask, ArrayBytesMaskCreateError};

/// Fixed or variable length array bytes.
#[derive(Clone, Debug, PartialEq, Eq, derive_more::From)]
#[non_exhaustive]
pub enum ArrayBytes<'a> {
    /// Bytes for a fixed length array.
    ///
    /// These represent elements in C-contiguous order (i.e. row-major order) where the last dimension varies the fastest.
    Fixed(CowBytes<'a>),
    /// Bytes and element byte offsets for a variable length array.
    Variable(ArrayBytesVariableLength<'a>),
    /// Bytes for an optional array (data with an [`ArrayBytesMask`]).
    Optional(ArrayBytesOptional<'a>),
}

/// An error raised if variable length array bytes offsets are out of bounds.
#[derive(Clone, Debug, Display, Error)]
#[display("Offset {offset} is out of bounds for bytes of length {len}")]
pub struct ArrayBytesOffsetsOutOfBoundsError {
    offset: usize,
    len: usize,
}

/// Errors related to [`ArrayBytes<'_>`] and [`ArrayBytes`].
#[derive(Clone, Debug, Error)]
pub enum ArrayBytesError {
    /// Invalid use of a fixed length method.
    #[error("Used a fixed length (flen) method on a variable length (vlen) array")]
    UsedFixedLengthMethodOnVariableLengthArray,
}

/// Expected fixed length array bytes but found variable or optional.
#[derive(Clone, Copy, Debug, Display, Error)]
#[display("Expected fixed length array bytes")]
pub struct ExpectedFixedLengthBytesError;

/// Expected variable length array bytes but found fixed or optional.
#[derive(Clone, Copy, Debug, Display, Error)]
#[display("Expected variable length array bytes")]
pub struct ExpectedVariableLengthBytesError;

/// Expected optional array bytes but found fixed or variable.
#[derive(Clone, Copy, Debug, Display, Error)]
#[display("Expected optional array bytes")]
pub struct ExpectedOptionalBytesError;

impl<'a> ArrayBytes<'a> {
    /// Create a new fixed length array bytes from `bytes`.
    ///
    /// `bytes` must be C-contiguous.
    pub fn new_flen(bytes: impl Into<CowBytes<'a>>) -> Self {
        Self::Fixed(bytes.into())
    }

    /// Create a new variable length array bytes from `bytes` and `offsets`.
    ///
    /// # Errors
    /// Returns a [`ArrayBytesOffsetsOutOfBoundsError`] if the last offset is out of bounds of the bytes.
    pub fn new_vlen(
        bytes: impl Into<CowBytes<'a>>,
        offsets: ArrayBytesOffsets,
    ) -> Result<Self, ArrayBytesOffsetsOutOfBoundsError> {
        ArrayBytesVariableLength::new(bytes, offsets).map(Self::Variable)
    }

    /// Create a new variable length array bytes from `bytes` and `offsets` without checking the offsets.
    ///
    /// # Safety
    /// The last offset must be less than or equal to the length of the bytes.
    pub unsafe fn new_vlen_unchecked(
        bytes: impl Into<CowBytes<'a>>,
        offsets: ArrayBytesOffsets,
    ) -> Self {
        Self::Variable(unsafe { ArrayBytesVariableLength::new_unchecked(bytes, offsets) })
    }

    /// Wrap the array bytes with an [`ArrayBytesMask`].
    #[must_use]
    pub fn with_optional_mask(self, mask: impl Into<ArrayBytesMask<'a>>) -> Self {
        Self::Optional(ArrayBytesOptional::new(self, mask))
    }

    /// Wrap the array bytes in one optional layer per mask, with `masks` ordered from outermost to innermost.
    ///
    /// Like [`ArrayBytesTarget::with_optional_masks`](crate::ArrayBytesTarget::with_optional_masks).
    #[must_use]
    pub fn with_optional_masks<I>(self, masks: I) -> Self
    where
        I: IntoIterator,
        I::Item: Into<ArrayBytesMask<'a>>,
        I::IntoIter: DoubleEndedIterator,
    {
        masks
            .into_iter()
            .rev()
            .fold(self, |bytes, mask| bytes.with_optional_mask(mask))
    }

    /// Create a new [`ArrayBytes`] with `num_elements` composed entirely of the `fill_value`.
    ///
    /// # Errors
    /// Returns [`DataTypeFillValueError`] if the fill value is incompatible with the data type,
    /// the element count exceeds [`usize::MAX`], or a required buffer exceeds [`isize::MAX`] bytes.
    pub fn new_fill_value(
        data_type: &DataType,
        num_elements: u64,
        fill_value: &FillValue,
    ) -> Result<Self, DataTypeFillValueError> {
        let num_elements = usize::try_from(num_elements).map_err(|_| DataTypeFillValueError)?;
        // Optional masks have one byte per element, including for zero-width data.
        if num_elements > isize::MAX as usize && data_type.is_optional() {
            return Err(DataTypeFillValueError);
        }
        Self::new_fill_value_bytes(data_type, num_elements, fill_value.as_ne_bytes())
    }

    fn new_null_fill_value_bytes(
        data_type: &DataType,
        num_elements: usize,
    ) -> Result<Self, DataTypeFillValueError> {
        match data_type {
            DataType::Fixed(fixed) => {
                let len = fill_bytes_len(num_elements, fixed.size())?;
                Ok(Self::new_flen(vec![0; len]))
            }
            DataType::Variable(_) => {
                let num_offsets = fill_offsets_len::<u32>(num_elements)?;
                // SAFETY: The offset buffer is nonempty and every offset is zero.
                let offsets = unsafe { ArrayBytesOffsets::new_unchecked(vec![0u32; num_offsets]) };
                Ok(unsafe {
                    // SAFETY: The last offset is 0, equal to the empty bytes length.
                    Self::new_vlen_unchecked(Vec::new(), offsets)
                })
            }
            DataType::Optional(optional) => Ok(Self::new_null_fill_value_bytes(
                optional.data_type(),
                num_elements,
            )?
            .with_optional_mask(vec![false; num_elements])),
            _ => Err(DataTypeFillValueError),
        }
    }

    fn new_fill_value_bytes(
        data_type: &DataType,
        num_elements: usize,
        fill_value_bytes: &[u8],
    ) -> Result<Self, DataTypeFillValueError> {
        match data_type {
            DataType::Fixed(fixed) => {
                let size = fixed.size();
                if fill_value_bytes.len() != size {
                    return Err(DataTypeFillValueError);
                }
                fill_bytes_len(num_elements, size)?;
                Ok(Self::new_flen(fill_value_bytes.repeat(num_elements)))
            }
            DataType::Variable(_) => {
                let num_offsets = fill_offsets_len::<usize>(num_elements)?;
                fill_bytes_len(num_elements, fill_value_bytes.len())?;
                let mut offsets = Vec::with_capacity(num_offsets);
                for i in 0..=num_elements {
                    offsets.push(i * fill_value_bytes.len());
                }
                let offsets = offsets_from_usize(offsets).map_err(|_| DataTypeFillValueError)?;
                Ok(unsafe {
                    // SAFETY: The last offset is equal to the length of the bytes.
                    Self::new_vlen_unchecked(fill_value_bytes.repeat(num_elements), offsets)
                })
            }
            DataType::Optional(optional) => match parse_optional_fill_value(fill_value_bytes)? {
                None => Ok(
                    Self::new_null_fill_value_bytes(optional.data_type(), num_elements)?
                        .with_optional_mask(vec![false; num_elements]),
                ),
                Some(inner_bytes) => Ok(Self::new_fill_value_bytes(
                    optional.data_type(),
                    num_elements,
                    inner_bytes,
                )?
                .with_optional_mask(vec![true; num_elements])),
            },
            _ => Err(DataTypeFillValueError),
        }
    }

    pub(crate) fn new_empty(data_type: &DataType) -> Result<Self, CodecError> {
        match data_type {
            DataType::Fixed(_) => Ok(Self::new_flen(vec![])),
            DataType::Variable(_) => {
                Ok(Self::new_vlen(vec![], ArrayBytesOffsets::new(vec![0u32])?)?)
            }
            DataType::Optional(optional) => {
                Ok(Self::new_empty(optional.data_type())?.with_optional_mask(Vec::<bool>::new()))
            }
            _ => Err("unsupported data type layout".into()),
        }
    }

    /// Convert the array bytes into fixed length bytes.
    ///
    /// # Errors
    /// Returns an [`ExpectedFixedLengthBytesError`] if the bytes are not fixed.
    pub fn into_fixed(self) -> Result<CowBytes<'a>, ExpectedFixedLengthBytesError> {
        match self {
            Self::Fixed(bytes) => Ok(bytes),
            Self::Variable(..) | Self::Optional(..) => Err(ExpectedFixedLengthBytesError),
        }
    }

    /// Convert the array bytes into variable length bytes and element byte offsets.
    ///
    /// # Errors
    /// Returns an [`ExpectedVariableLengthBytesError`] if the bytes are not variable.
    pub fn into_variable(
        self,
    ) -> Result<ArrayBytesVariableLength<'a>, ExpectedVariableLengthBytesError> {
        match self {
            Self::Fixed(..) | Self::Optional(..) => Err(ExpectedVariableLengthBytesError),
            Self::Variable(variable_length_bytes) => Ok(variable_length_bytes),
        }
    }

    /// Convert the array bytes into optional data and validity mask.
    ///
    /// # Errors
    /// Returns an [`ExpectedOptionalBytesError`] if the bytes are not optional.
    pub fn into_optional(self) -> Result<ArrayBytesOptional<'a>, ExpectedOptionalBytesError> {
        match self {
            Self::Optional(optional_bytes) => Ok(optional_bytes),
            Self::Fixed(..) | Self::Variable(..) => Err(ExpectedOptionalBytesError),
        }
    }

    /// Returns the size (in bytes) of the underlying element bytes.
    ///
    /// This only considers the size of the element bytes, and does not include the element offsets for a variable sized array or the mask for optional arrays.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::Fixed(bytes) | Self::Variable(ArrayBytesVariableLength { bytes, offsets: _ }) => {
                bytes.len()
            }
            Self::Optional(optional_bytes) => optional_bytes.data().size(),
        }
    }

    /// Return the byte offsets for variable sized bytes. Returns [`None`] for fixed size bytes.
    #[must_use]
    pub fn offsets(&self) -> Option<&ArrayBytesOffsets> {
        match self {
            Self::Fixed(..) => None,
            Self::Variable(ArrayBytesVariableLength { offsets, .. }) => Some(offsets),
            Self::Optional(optional_bytes) => optional_bytes.data().offsets(),
        }
    }

    /// Convert into owned [`ArrayBytes<'static>`].
    #[must_use]
    pub fn into_owned(self) -> ArrayBytes<'static> {
        match self {
            Self::Fixed(bytes) => ArrayBytes::Fixed(bytes.into_static()),
            Self::Variable(ArrayBytesVariableLength { bytes, offsets }) => {
                ArrayBytes::Variable(ArrayBytesVariableLength {
                    bytes: bytes.into_static(),
                    offsets,
                })
            }
            Self::Optional(optional_bytes) => ArrayBytes::Optional(optional_bytes.into_owned()),
        }
    }

    /// Validate that the array has a valid encoding.
    ///
    /// For a fixed-length array, check it matches the expected size.
    /// For a variable-length array, check that there is one more offset than elements and the last offset is equal to the length of the bytes.
    /// The offsets are monotonically increasing by construction, so this does not iterate over them.
    /// For an optional array, check that the mask has one byte per element and validate the inner array.
    ///
    /// # Errors
    /// Returns an error if the array is not valid.
    pub fn validate(&self, num_elements: u64, data_type: &DataType) -> Result<(), CodecError> {
        validate_bytes(self, num_elements, data_type)
    }

    /// Returns [`true`] if the array is empty for the given fill value.
    ///
    /// The array bytes are expected to be [validated](Self::validate).
    ///
    /// For variable-length arrays, every element must exactly equal the fill value,
    /// respecting element boundaries and ignoring bytes outside the element ranges.
    /// A variable-length array with no elements matches any fill value.
    #[must_use]
    pub fn is_fill_value(&self, fill_value: &FillValue) -> bool {
        self.is_fill_value_bytes(fill_value.as_ne_bytes())
    }

    fn is_fill_value_bytes(&self, fill_value_bytes: &[u8]) -> bool {
        match self {
            Self::Fixed(bytes) => FillValue::equals_all_bytes(fill_value_bytes, bytes),
            Self::Variable(ArrayBytesVariableLength { bytes, offsets }) => offsets
                .element_ranges()
                .all(|range| &bytes[range] == fill_value_bytes),
            Self::Optional(optional_bytes) => match fill_value_bytes.split_last() {
                Some((&0, &[])) => optional_bytes.mask().iter().all(|&b| b == 0),
                Some((&1, inner)) => {
                    optional_bytes.mask().iter().all(|&b| b == 1)
                        && optional_bytes.data().is_fill_value_bytes(inner)
                }
                _ => false,
            },
        }
    }

    /// Extract a subset of the array bytes.
    ///
    /// # Errors
    /// Returns a [`CodecError::IncompatibleIndexer`] if the `indexer` has an incompatible dimensionality with, or is out-of-bounds of, `array_shape`.
    /// Returns another [`CodecError`] if `self` is incompatible with `data_type`.
    ///
    /// # Panics
    /// Panics if indices in the subset exceed [`usize::MAX`].
    pub fn extract_array_subset(
        &self,
        indexer: &dyn Indexer,
        array_shape: &[u64],
        data_type: &DataType,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if let Some(subset) = indexer.as_array_subset() {
            // The runs are computed once, and shared by the data and mask of optional bytes
            let runs = subset.contiguous_linearised_indices(array_shape)?;
            return self.extract_runs(&runs, data_type);
        }
        match (self, data_type) {
            (
                ArrayBytes::Variable(ArrayBytesVariableLength { bytes, offsets }),
                DataType::Variable(_),
            ) => {
                let num_elements = indexer.len();
                let indices: Vec<_> = indexer.iter_linearised_indices(array_shape)?.collect();
                let mut bytes_length = 0;
                for index in &indices {
                    let index = usize::try_from(*index).unwrap();
                    bytes_length += offsets.element_range(index).len();
                }
                let mut ss_bytes = Vec::with_capacity(bytes_length);
                let mut ss_offsets = Vec::with_capacity(usize::try_from(1 + num_elements).unwrap());
                for index in &indices {
                    let index = usize::try_from(*index).unwrap();
                    ss_offsets.push(ss_bytes.len());
                    ss_bytes.extend_from_slice(&bytes[offsets.element_range(index)]);
                }
                ss_offsets.push(ss_bytes.len());
                let ss_offsets = offsets_from_usize(ss_offsets).unwrap();
                let array_bytes = unsafe {
                    // SAFETY: The last offset is equal to the length of the bytes
                    ArrayBytes::new_vlen_unchecked(ss_bytes, ss_offsets)
                };
                Ok(array_bytes)
            }
            (ArrayBytes::Fixed(bytes), DataType::Fixed(fixed)) => {
                let bytes = extract_elements(bytes, indexer, array_shape, fixed.size())?;
                Ok(ArrayBytes::new_flen(bytes))
            }
            (ArrayBytes::Optional(optional_bytes), DataType::Optional(opt)) => {
                // Extract subset of the inner data recursively
                let subset_data = optional_bytes.data().extract_array_subset(
                    indexer,
                    array_shape,
                    opt.data_type(),
                )?;

                // Extract subset of the mask (1 byte per element)
                let subset_mask = extract_elements(optional_bytes.mask(), indexer, array_shape, 1)?;
                // SAFETY: extraction preserves the source mask's values
                let subset_mask = unsafe { ArrayBytesMask::new_unchecked(subset_mask) };

                Ok(subset_data.with_optional_mask(subset_mask))
            }
            (_, DataType::Fixed(_)) => Err(ExpectedFixedLengthBytesError.into()),
            (_, DataType::Variable(_)) => Err(ExpectedVariableLengthBytesError.into()),
            (_, DataType::Optional(_)) => Err(ExpectedOptionalBytesError.into()),
            _ => Err("unsupported data type layout".into()),
        }
    }

    /// Extract the elements in the contiguous `runs` of element indices, in order.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `self` is incompatible with `data_type` or `runs` address elements out of bounds.
    ///
    /// # Panics
    /// Panics if indices or byte lengths exceed [`usize::MAX`].
    pub fn extract_runs(
        &self,
        runs: &ContiguousLinearisedIndices,
        data_type: &DataType,
    ) -> Result<ArrayBytes<'static>, CodecError> {
        match (self, data_type) {
            (ArrayBytes::Fixed(bytes), DataType::Fixed(fixed)) => {
                let size = fixed.size();
                Ok(ArrayBytes::new_flen(extract_runs_fixed(bytes, runs, size)?))
            }
            (ArrayBytes::Variable(variable), DataType::Variable(_)) => Ok(ArrayBytes::Variable(
                extract_runs_variable(variable.bytes(), variable.offsets(), runs)?,
            )),
            (ArrayBytes::Optional(optional), DataType::Optional(opt)) => {
                let data = optional.data().extract_runs(runs, opt.data_type())?;
                let mask = extract_runs_fixed(optional.mask(), runs, 1)?;
                // SAFETY: the bytes are extracted from a mask of 0 or 1
                let mask = unsafe { ArrayBytesMask::new_unchecked(mask) };
                Ok(data.with_optional_mask(mask))
            }
            (_, DataType::Fixed(_)) => Err(ExpectedFixedLengthBytesError.into()),
            (_, DataType::Variable(_)) => Err(ExpectedVariableLengthBytesError.into()),
            (_, DataType::Optional(_)) => Err(ExpectedOptionalBytesError.into()),
            _ => Err("unsupported data type layout".into()),
        }
    }

    /// Copy the elements in the contiguous `runs` of element indices into `output_target`, in order.
    ///
    /// This is equivalent to [`decode_into_array_bytes_target`] with the elements [extracted](Self::extract_runs) from `runs`, without an intermediate buffer.
    /// The elements have the data type size of `output_target`.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if
    /// - `self` is incompatible with `output_target`,
    /// - `runs` address elements beyond `self`, or
    /// - `runs` do not have as many elements as `output_target`.
    ///
    /// # Panics
    /// Panics if indices or byte lengths exceed [`usize::MAX`].
    pub fn extract_runs_into(
        &self,
        runs: &ContiguousLinearisedIndices,
        output_target: &mut ArrayBytesTarget<'_>,
    ) -> Result<(), CodecError> {
        match (self, output_target) {
            (ArrayBytes::Fixed(bytes), ArrayBytesTarget::Fixed(view)) => {
                view.copy_runs_from_slice(bytes, runs)
            }
            (ArrayBytes::Optional(optional), ArrayBytesTarget::Optional(target)) => {
                optional.data().extract_runs_into(runs, &mut target.data)?;
                target.mask.copy_runs_from_slice(optional.mask(), runs)
            }
            (ArrayBytes::Variable(..), _) => Err(ExpectedFixedLengthBytesError.into()),
            (ArrayBytes::Fixed(_), ArrayBytesTarget::Optional(_)) => {
                Err(ExpectedOptionalBytesError.into())
            }
            (ArrayBytes::Optional(..), ArrayBytesTarget::Fixed(_)) => {
                Err(ExpectedFixedLengthBytesError.into())
            }
        }
    }
}

fn fill_bytes_len(
    num_elements: usize,
    element_size: usize,
) -> Result<usize, DataTypeFillValueError> {
    num_elements
        .checked_mul(element_size)
        .filter(|&len| len <= isize::MAX as usize)
        .ok_or(DataTypeFillValueError)
}

fn fill_offsets_len<T>(num_elements: usize) -> Result<usize, DataTypeFillValueError> {
    let count = num_elements.checked_add(1).ok_or(DataTypeFillValueError)?;
    std::alloc::Layout::array::<T>(count).map_err(|_| DataTypeFillValueError)?;
    Ok(count)
}

fn parse_optional_fill_value(
    fill_value_bytes: &[u8],
) -> Result<Option<&[u8]>, DataTypeFillValueError> {
    match fill_value_bytes.split_last() {
        Some((&0, &[])) => Ok(None),
        Some((&1, inner)) => Ok(Some(inner)),
        _ => Err(DataTypeFillValueError),
    }
}

/// Extract variable-length elements in `runs` into `extracted`, constructing `O`-width offsets directly.
///
/// Returns [`None`] if a run addresses elements beyond `offsets` or `bytes`, or an extracted offset exceeds `O`.
fn extract_vlen_runs<O: ArrayBytesOffsetsElement + TryFrom<usize>>(
    bytes: &[u8],
    offsets: &ArrayBytesOffsets,
    runs: &ContiguousLinearisedIndices,
    run_elements: usize,
    num_out: usize,
    extracted: &mut Vec<u8>,
) -> Option<ArrayBytesOffsets> {
    let mut out_offsets = Vec::with_capacity(num_out.saturating_add(1));
    for (index, _) in runs {
        let start = usize::try_from(index).ok()?;
        let end = start.checked_add(run_elements)?;
        let first_offset = offsets.get(start)?;
        let source = bytes.get(first_offset..offsets.get(end)?)?;
        let base = extracted.len();
        for element in start..end {
            // Offsets increase, so the offset within the source is not negative.
            // It is at most `source.len()`, so adding it to `base` (both at most `isize::MAX`) does not overflow.
            let offset = base + (offsets.get(element)? - first_offset);
            out_offsets.push(O::try_from(offset).ok()?);
        }
        extracted.extend_from_slice(source);
    }
    out_offsets.push(O::try_from(extracted.len()).ok()?);
    // SAFETY: `out_offsets` is non-empty and monotonically increasing, since each run continues from the end of the previous run, and its offsets are from `usize`.
    Some(unsafe { ArrayBytesOffsets::new_unchecked(out_offsets) })
}

/// Extract variable-length elements in `runs` from `bytes` and `offsets`.
fn extract_runs_variable(
    bytes: &[u8],
    offsets: &ArrayBytesOffsets,
    runs: &ContiguousLinearisedIndices,
) -> Result<ArrayBytesVariableLength<'static>, CodecError> {
    // Offsets are non-empty
    let num_elements = offsets.len() - 1;
    let run_elements = match runs.iter().next_back() {
        None => 0,
        Some((index, run_elements)) => {
            // Runs are ordered, so the last run bounds the runs
            if index
                .checked_add(run_elements)
                .is_none_or(|end| end > num_elements as u64)
            {
                return Err(CodecError::InvalidVariableSizedArrayOffsets);
            }
            // The run is within the offsets, so its length fits in `usize`
            usize::try_from(run_elements).unwrap()
        }
    };
    // Runs that repeat elements can select more elements than there are
    let num_out = runs
        .len()
        .checked_mul(run_elements)
        .filter(|&num_out| num_out <= num_elements)
        .ok_or_else(|| {
            IndexerError::IncompatibleLength(
                (runs.len() as u64).saturating_mul(run_elements as u64),
                num_elements as u64,
            )
        })?;
    // The extracted byte length, unless runs repeat elements
    let bytes_len = if num_out == num_elements {
        // The runs cover as many elements as the bytes, e.g. reordering all elements
        offsets.last() - offsets.get(0).unwrap_or(0)
    } else {
        runs.iter()
            .try_fold(0usize, |bytes_len, (index, _)| {
                let start = usize::try_from(index).ok()?;
                let len = offsets.get(start.checked_add(run_elements)?)? - offsets.get(start)?;
                bytes_len.checked_add(len)
            })
            .ok_or(CodecError::InvalidVariableSizedArrayOffsets)?
    };
    let mut extracted = Vec::with_capacity(bytes_len);
    // The offsets are u64 if the extracted bytes exceed u32, which `bytes_len` underestimates if runs repeat elements
    let out_offsets = u32::try_from(bytes_len)
        .ok()
        .and_then(|_| {
            extract_vlen_runs::<u32>(bytes, offsets, runs, run_elements, num_out, &mut extracted)
        })
        .or_else(|| {
            extracted.clear();
            extract_vlen_runs::<u64>(bytes, offsets, runs, run_elements, num_out, &mut extracted)
        })
        .ok_or(CodecError::InvalidVariableSizedArrayOffsets)?;
    // SAFETY: the last offset is `extracted.len()`.
    Ok(unsafe { ArrayBytesVariableLength::new_unchecked(extracted, out_offsets) })
}

/// Return an error if the fixed-length elements of `element_size` bytes in `runs` are not within `bytes`.
///
/// Runs are ordered and have equal lengths, so the last run bounds all selected bytes.
/// The offsets are checked so that an overflowing end cannot wrap within `bytes`.
pub(crate) fn validate_runs_within(
    bytes: &[u8],
    runs: &ContiguousLinearisedIndices,
    element_size: usize,
) -> Result<(), CodecError> {
    let element_size = element_size as u64;
    if let Some((index, elements)) = runs.iter().next_back() {
        let end = index
            .checked_add(elements)
            .and_then(|end| end.checked_mul(element_size));
        if end.is_none_or(|end| end > bytes.len() as u64) {
            let range = index.saturating_mul(element_size)..end.unwrap_or(u64::MAX);
            return Err(
                InvalidByteRangeError::new(ByteRange::new(range), bytes.len() as u64).into(),
            );
        }
    }
    Ok(())
}

/// Extract fixed-length elements of `element_size` bytes in `runs` from `bytes`.
pub(crate) fn extract_runs_fixed(
    bytes: &[u8],
    runs: &ContiguousLinearisedIndices,
    element_size: usize,
) -> Result<Vec<u8>, CodecError> {
    // Validate the runs before allocating the output, which may be large even for truncated input.
    validate_runs_within(bytes, runs, element_size)?;
    let element_size = element_size as u64;
    if runs.is_empty() || element_size == 0 {
        return Ok(Vec::new());
    }
    // Runs that repeat elements can select more elements than `bytes` holds
    let num_available = bytes.len() as u64 / element_size;
    let num_selected = (runs.len() as u64)
        .checked_mul(runs.contiguous_elements())
        .filter(|&num_selected| num_selected <= num_available)
        .ok_or_else(|| {
            IndexerError::IncompatibleLength(
                (runs.len() as u64).saturating_mul(runs.contiguous_elements()),
                num_available,
            )
        })?;
    // The selected bytes are at most `bytes.len()`, so their lengths do not overflow
    let run_len = usize::try_from(runs.contiguous_elements() * element_size).unwrap();
    let mut out = Vec::with_capacity(usize::try_from(num_selected * element_size).unwrap());
    with_small_len!(run_len, |run_len| {
        for (index, _) in runs {
            let start = index.checked_mul(element_size);
            let Some(run) = start
                .and_then(|start| usize::try_from(start).ok())
                .and_then(|start| bytes.get(start..)?.get(..run_len))
            else {
                let start = start.unwrap_or(u64::MAX);
                let range = start..start.saturating_add(run_len as u64);
                return Err(
                    InvalidByteRangeError::new(ByteRange::new(range), bytes.len() as u64).into(),
                );
            };
            out.extend_from_slice(run);
        }
    });
    Ok(out)
}

/// The bytes of the elements of `indexer` in `bytes`, an array of `array_shape` with elements of `element_size` bytes.
fn extract_elements(
    bytes: &[u8],
    indexer: &dyn Indexer,
    array_shape: &[u64],
    element_size: usize,
) -> Result<Vec<u8>, CodecError> {
    let bytes = if let Some(subset) = indexer.as_array_subset() {
        // The contiguous runs of an array subset are iterated without boxing an iterator, and copied into an output of the known size
        let runs = subset.contiguous_linearised_indices(array_shape)?;
        extract_runs_fixed(bytes, &runs, element_size)?
    } else {
        let byte_ranges = indexer.iter_contiguous_byte_ranges(array_shape, element_size)?;
        extract_byte_ranges_concat(bytes, byte_ranges)?
    };
    Ok(bytes)
}

/// Validate fixed length array bytes for a given array size.
fn validate_bytes_flen(bytes: &CowBytes, array_size: usize) -> Result<(), InvalidBytesLengthError> {
    if bytes.len() == array_size {
        Ok(())
    } else {
        Err(InvalidBytesLengthError::new(bytes.len(), array_size))
    }
}

/// Validate variable length array bytes for an array with `num_elements`.
///
/// The offsets are monotonically increasing (an invariant of [`ArrayBytesOffsets`]), and the last does not exceed the length of the bytes (an invariant of [`ArrayBytesVariableLength`]).
/// So every offset is within the bytes, and only the number of offsets and the last offset need checking.
fn validate_bytes_vlen(
    bytes: &CowBytes,
    offsets: &ArrayBytesOffsets,
    num_elements: u64,
) -> Result<(), CodecError> {
    debug_assert!(offsets.as_slice().iter().is_sorted() && offsets.last() <= bytes.len());
    if offsets.num_elements() as u64 == num_elements && offsets.last() == bytes.len() {
        Ok(())
    } else {
        Err(CodecError::InvalidVariableSizedArrayOffsets)
    }
}

/// Validate array bytes.
fn validate_bytes(
    bytes: &ArrayBytes<'_>,
    num_elements: u64,
    data_type: &DataType,
) -> Result<(), CodecError> {
    match (bytes, data_type) {
        (ArrayBytes::Fixed(bytes), DataType::Fixed(fixed)) => {
            let len = num_elements
                .checked_mul(fixed.size() as u64)
                .and_then(|len| usize::try_from(len).ok())
                .ok_or_else(|| {
                    CodecError::Other("array byte length overflows usize".to_string())
                })?;
            Ok(validate_bytes_flen(bytes, len)?)
        }
        (
            ArrayBytes::Variable(ArrayBytesVariableLength { bytes, offsets }),
            DataType::Variable(_),
        ) => validate_bytes_vlen(bytes, offsets, num_elements),
        (ArrayBytes::Optional(optional_bytes), DataType::Optional(optional)) => {
            let mask_len = optional_bytes.mask().len();
            if mask_len as u64 != num_elements {
                return Err(match usize::try_from(num_elements) {
                    Ok(expected_len) => InvalidBytesLengthError::new(mask_len, expected_len).into(),
                    Err(_) => {
                        InvalidNumberOfElementsError::new(mask_len as u64, num_elements).into()
                    }
                });
            }
            validate_bytes(optional_bytes.data(), num_elements, optional.data_type())
        }
        (_, DataType::Fixed(_)) => Err(ExpectedFixedLengthBytesError.into()),
        (_, DataType::Variable(_)) => Err(ExpectedVariableLengthBytesError.into()),
        (_, DataType::Optional(_)) => Err(ExpectedOptionalBytesError.into()),
        _ => Err("unsupported data type layout".into()),
    }
}

fn update_bytes_vlen_array_subset<'a>(
    input: &ArrayBytesVariableLength<'_>,
    input_shape: &[u64],
    update: &ArrayBytesVariableLength<'_>,
    update_subset: &dyn ArraySubsetTraits,
) -> Result<ArrayBytesVariableLength<'a>, CodecError> {
    let input_offsets = input.offsets();
    let input_bytes = input.bytes();
    let update_offsets = update.offsets();
    let update_bytes = update.bytes();

    // Get the current and new length of the bytes in the chunk subset
    let size_subset_new = update
        .offsets()
        .element_ranges()
        .map(|range| range.len())
        .sum::<usize>();
    let size_subset_old = {
        let chunk_indices = update_subset.linearised_indices(input_shape)?;
        chunk_indices
            .iter()
            .map(|index| {
                let index = usize::try_from(index).unwrap();
                input_offsets.element_range(index).len()
            })
            .sum::<usize>()
    };

    // Populate new offsets and bytes
    let mut offsets_new = Vec::with_capacity(input_offsets.len());
    let bytes_new_len = input_bytes
        .len()
        .checked_sub(size_subset_old)
        .and_then(|kept| kept.checked_add(size_subset_new))
        .ok_or_else(|| {
            CodecError::Other("variable length array bytes overflow usize".to_string())
        })?;
    let mut bytes_new = Vec::with_capacity(bytes_new_len);
    let indices = ArraySubset::new_with_shape(input_shape).indices();
    let update_subset_start = update_subset.start();
    let update_subset_shape = update_subset.shape();
    for (chunk_index, indices) in indices.iter().enumerate() {
        offsets_new.push(bytes_new.len());
        if update_subset.contains(&indices) {
            let subset_indices = indices
                .iter()
                .zip(update_subset_start.iter())
                .map(|(i, s)| i - s)
                .collect::<Vec<_>>();
            let subset_index =
                ravel_indices(&subset_indices, &update_subset_shape).expect("inbounds indices");
            let subset_index = usize::try_from(subset_index).unwrap();
            bytes_new.extend_from_slice(&update_bytes[update_offsets.element_range(subset_index)]);
        } else {
            bytes_new.extend_from_slice(&input_bytes[input_offsets.element_range(chunk_index)]);
        }
    }
    offsets_new.push(bytes_new.len());
    let offsets_new = offsets_from_usize(offsets_new).unwrap();
    let array_bytes = unsafe {
        // SAFETY: The last offset is equal to the length of the bytes
        ArrayBytesVariableLength::new_unchecked(bytes_new, offsets_new)
    };
    Ok(array_bytes)
}

fn update_bytes_vlen_indexer<'a>(
    input: &ArrayBytesVariableLength<'_>,
    input_shape: &[u64],
    update: &ArrayBytesVariableLength<'_>,
    update_indexer: &dyn Indexer,
) -> Result<ArrayBytesVariableLength<'a>, CodecError> {
    let input_bytes = input.bytes();
    let input_offsets = input.offsets();
    let update_bytes = update.bytes();
    let update_offsets = update.offsets();

    // Get the indices of elements to update and the size of the old and new bytes
    let num_elements = input_offsets.num_elements();
    let update_indices = update_indexer.iter_linearised_indices(input_shape)?;
    let mut element_indices_update: Vec<Option<usize>> = vec![None; num_elements];
    let mut updated_size_old = 0usize;
    let mut updated_size_new = 0usize;
    for (update_index, input_index) in update_indices.enumerate() {
        let input_index = usize::try_from(input_index).unwrap();
        let new_len = update_offsets.element_range(update_index).len();
        if let Some(prev_update_index) = element_indices_update[input_index].replace(update_index) {
            updated_size_new -= update_offsets.element_range(prev_update_index).len();
        } else {
            updated_size_old += input_offsets.element_range(input_index).len();
        }
        updated_size_new += new_len;
    }

    // Populate new offsets and bytes
    let mut offsets_new = Vec::with_capacity(input_offsets.len());
    let bytes_new_len = input_bytes
        .len()
        .checked_sub(updated_size_old)
        .and_then(|kept| kept.checked_add(updated_size_new))
        .ok_or_else(|| {
            CodecError::Other("variable length array bytes overflow usize".to_string())
        })?;
    let mut bytes_new = Vec::with_capacity(bytes_new_len);
    for input_index in 0..num_elements {
        offsets_new.push(bytes_new.len());
        if let Some(update_index) = element_indices_update[input_index] {
            bytes_new.extend_from_slice(&update_bytes[update_offsets.element_range(update_index)]);
        } else {
            bytes_new.extend_from_slice(&input_bytes[input_offsets.element_range(input_index)]);
        }
    }
    offsets_new.push(bytes_new.len());
    let offsets_new = offsets_from_usize(offsets_new).unwrap();
    // SAFETY: The last offset is equal to the length of the bytes.
    Ok(unsafe { ArrayBytesVariableLength::new_unchecked(bytes_new, offsets_new) })
}

/// Update array bytes. Specialised for `ArraySubset`.
///
/// # Errors
/// Returns a [`CodecError`] if
/// - `bytes` are not compatible with the `shape` and `data_type`,
/// - `update_bytes` are not compatible with the `update_subset` and `data_type`,
/// - `update_subset` has an incompatible dimensionality with, or is not within the bounds of, `shape`
fn update_array_bytes_array_subset<'a>(
    bytes: ArrayBytes,
    shape: &[u64],
    update_subset: &dyn ArraySubsetTraits,
    update_bytes: &ArrayBytes,
    data_type: &DataType,
) -> Result<ArrayBytes<'a>, CodecError> {
    match (bytes, update_bytes, data_type) {
        (ArrayBytes::Variable(bytes), ArrayBytes::Variable(update), DataType::Variable(_)) => {
            Ok(ArrayBytes::Variable(update_bytes_vlen_array_subset(
                &bytes,
                shape,
                update,
                update_subset,
            )?))
        }
        (ArrayBytes::Fixed(bytes), ArrayBytes::Fixed(update_bytes), DataType::Fixed(fixed)) => {
            let mut bytes = bytes.into_vec();
            let mut output_view: ArrayBytesFixedDisjointView<'_> = unsafe {
                // SAFETY: Only one view is created, so it is disjoint
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut bytes),
                    fixed.size(),
                    shape,
                    update_subset.to_array_subset(),
                )
            }
            .map_err(CodecError::from)?;
            output_view.copy_from_slice(update_bytes)?;
            Ok(ArrayBytes::new_flen(bytes))
        }
        (
            ArrayBytes::Optional(optional_bytes),
            ArrayBytes::Optional(update_optional_bytes),
            DataType::Optional(optional),
        ) => {
            // Update the data recursively
            let (data, mask) = optional_bytes.into_parts();
            let data_after_update = update_array_bytes_array_subset(
                *data,
                shape,
                update_subset,
                update_optional_bytes.data(),
                optional.data_type(),
            )?;

            // Update the mask (it's a fixed-size bool array, 1 byte per element)
            let mut mask = mask.into_bytes().into_vec();
            let mut mask_view: ArrayBytesFixedDisjointView<'_> = unsafe {
                // SAFETY: Only one view is created, so it is disjoint
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(&mut mask),
                    1, // bool is 1 byte per element
                    shape,
                    update_subset.to_array_subset(),
                )
            }
            .map_err(CodecError::from)?;
            mask_view.copy_from_slice(update_optional_bytes.mask())?;
            // SAFETY: the update copies entries from another valid mask
            let mask = unsafe { ArrayBytesMask::new_unchecked(mask) };

            Ok(ArrayBytes::Optional(ArrayBytesOptional::new(
                data_after_update,
                mask,
            )))
        }
        (_, _, DataType::Variable(_)) => Err(ExpectedVariableLengthBytesError.into()),
        (_, _, DataType::Fixed(_)) => Err(ExpectedFixedLengthBytesError.into()),
        (_, _, DataType::Optional(_)) => Err(ExpectedOptionalBytesError.into()),
        _ => Err("unsupported data type layout".into()),
    }
}

/// Update array bytes. Specialised for `Indexer`.
///
/// # Errors
/// Returns a [`CodecError`] if
/// - `bytes` are not compatible with the `shape` and `data_type`,
/// - `update_bytes` are not compatible with the `update_indexer` and `data_type`,
/// - `update_indexer` has an incompatible dimensionality with, or is not within the bounds of, `shape`
fn update_array_bytes_indexer<'a>(
    bytes: ArrayBytes,
    shape: &[u64],
    update_indexer: &dyn Indexer,
    update_bytes: &ArrayBytes,
    data_type: &DataType,
) -> Result<ArrayBytes<'a>, CodecError> {
    match (bytes, update_bytes, data_type) {
        (
            ArrayBytes::Variable(bytes),
            ArrayBytes::Variable(update_bytes),
            DataType::Variable(_),
        ) => Ok(ArrayBytes::Variable(update_bytes_vlen_indexer(
            &bytes,
            shape,
            update_bytes,
            update_indexer,
        )?)),
        (ArrayBytes::Fixed(bytes), ArrayBytes::Fixed(update_bytes), DataType::Fixed(fixed)) => {
            let mut bytes = bytes.into_vec();
            let byte_ranges = update_indexer.iter_contiguous_byte_ranges(shape, fixed.size())?;
            let mut offset: usize = 0;
            for byte_range in byte_ranges {
                let start = usize::try_from(byte_range.start).unwrap();
                let end = usize::try_from(byte_range.end).unwrap();
                let byte_range_len = end.saturating_sub(start);
                bytes
                    .index_mut(start..end)
                    .copy_from_slice(&update_bytes[offset..offset + byte_range_len]);
                offset += byte_range_len;
            }
            Ok(ArrayBytes::new_flen(bytes))
        }
        (
            ArrayBytes::Optional(optional_bytes),
            ArrayBytes::Optional(update_optional_bytes),
            DataType::Optional(optional),
        ) => {
            // Update the data recursively
            let (data, mask) = optional_bytes.into_parts();
            let data_after_update = update_array_bytes_indexer(
                *data,
                shape,
                update_indexer,
                update_optional_bytes.data(),
                optional.data_type(),
            )?;

            // Update the mask (it's a fixed-size bool array, 1 byte per element)
            let mut mask = mask.into_bytes().into_vec();
            let byte_ranges = update_indexer.iter_contiguous_byte_ranges(shape, 1)?; // 1 byte per bool
            let mut offset: usize = 0;
            for byte_range in byte_ranges {
                let start = usize::try_from(byte_range.start).unwrap();
                let end = usize::try_from(byte_range.end).unwrap();
                let byte_range_len = end.saturating_sub(start);
                mask.index_mut(start..end).copy_from_slice(
                    &update_optional_bytes.mask()[offset..offset + byte_range_len],
                );
                offset += byte_range_len;
            }
            // SAFETY: the update copies entries from another valid mask
            let mask = unsafe { ArrayBytesMask::new_unchecked(mask) };

            Ok(ArrayBytes::Optional(ArrayBytesOptional::new(
                data_after_update,
                mask,
            )))
        }
        (_, _, DataType::Variable(_)) => Err(ExpectedVariableLengthBytesError.into()),
        (_, _, DataType::Fixed(_)) => Err(ExpectedFixedLengthBytesError.into()),
        (_, _, DataType::Optional(_)) => Err(ExpectedOptionalBytesError.into()),
        _ => Err("unsupported data type layout".into()),
    }
}

/// Update array bytes.
///
/// # Errors
/// Returns a [`CodecError`] if
/// - `bytes` are not compatible with the `shape` and `data_type`,
/// - `update_bytes` are not compatible with the `update_indexer` and `data_type`,
/// - `update_indexer` has an incompatible dimensionality with, or is not within the bounds of, `shape`
///
/// # Panics
/// Panics if the indexer references bytes beyond [`usize::MAX`].
pub fn update_array_bytes<'a>(
    bytes: ArrayBytes,
    shape: &[u64],
    update_indexer: &dyn Indexer,
    update_bytes: &ArrayBytes,
    data_type: &DataType,
) -> Result<ArrayBytes<'a>, CodecError> {
    let num_elements = shape
        .iter()
        .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| CodecError::Other("array shape element count overflows u64".to_string()))?;
    bytes.validate(num_elements, data_type)?;
    update_indexer.validate(shape)?;
    update_bytes.validate(update_indexer.len(), data_type)?;
    if let Some(output_subset) = update_indexer.as_array_subset() {
        update_array_bytes_array_subset(bytes, shape, output_subset, update_bytes, data_type)
    } else {
        update_array_bytes_indexer(bytes, shape, update_indexer, update_bytes, data_type)
    }
}

/// The elements of an [`ArrayBytesTarget`] to write.
#[derive(Clone, Copy)]
enum TargetElements<'p> {
    /// Every element of the target.
    All,
    /// The elements at these positions, which index the elements of the target in C order.
    Positions(&'p [usize]),
}

impl TargetElements<'_> {
    fn fill(
        self,
        view: &mut ArrayBytesFixedDisjointView<'_>,
        value: &[u8],
    ) -> Result<(), CodecError> {
        match self {
            Self::All => Ok(view.fill(value)?),
            Self::Positions(positions) => view.fill_elements(positions, value),
        }
    }

    fn fill_zero(self, view: &mut ArrayBytesFixedDisjointView<'_>) -> Result<(), CodecError> {
        match self {
            Self::All => {
                view.fill_byte(0);
                Ok(())
            }
            Self::Positions(positions) => {
                view.fill_elements(positions, &vec![0; view.data_type_size()])
            }
        }
    }

    fn copy(
        self,
        view: &mut ArrayBytesFixedDisjointView<'_>,
        bytes: &[u8],
    ) -> Result<(), CodecError> {
        match self {
            Self::All => Ok(view.copy_from_slice(bytes)?),
            Self::Positions(positions) => view.copy_elements_from_slice(positions, bytes),
        }
    }
}

/// Decode the fill value into a preallocated output target.
///
/// This method is intended for internal use by Array.
/// Supports fixed length data types, including nested optional types with fixed length data.
///
/// # Errors
/// Returns a [`CodecError`] if
///  - `data_type` is variable-sized,
///  - the structure of `output_target` does not match `data_type` (one [`ArrayBytesTarget::Optional`] layer per optional nesting level), or
///  - `fill_value` is incompatible with `data_type`.
pub fn copy_fill_value_into(
    data_type: &DataType,
    fill_value: &FillValue,
    output_target: &mut ArrayBytesTarget<'_>,
) -> Result<(), CodecError> {
    copy_fill_value_bytes_into(
        data_type,
        fill_value.as_ne_bytes(),
        output_target,
        TargetElements::All,
    )
}

/// Decode the fill value into the elements of a preallocated output target at `positions`.
///
/// Positions index the elements of the target in C order.
///
/// # Errors
/// Returns a [`CodecError`] if [`copy_fill_value_into`] would, or a position is out-of-bounds of the target.
pub fn copy_fill_value_into_elements(
    data_type: &DataType,
    fill_value: &FillValue,
    positions: &[usize],
    output_target: &mut ArrayBytesTarget<'_>,
) -> Result<(), CodecError> {
    copy_fill_value_bytes_into(
        data_type,
        fill_value.as_ne_bytes(),
        output_target,
        TargetElements::Positions(positions),
    )
}

fn copy_null_fill_value_into(
    data_type: &DataType,
    output_target: &mut ArrayBytesTarget<'_>,
    elements: TargetElements,
) -> Result<(), CodecError> {
    match (data_type, output_target) {
        (DataType::Optional(optional), ArrayBytesTarget::Optional(optional_target)) => {
            copy_null_fill_value_into(optional.data_type(), &mut optional_target.data, elements)?;
            elements.fill_zero(&mut optional_target.mask)
        }
        (DataType::Fixed(fixed), ArrayBytesTarget::Fixed(view)) => {
            if view.data_type_size() != fixed.size() {
                return Err(
                    InvalidBytesLengthError::new(fixed.size(), view.data_type_size()).into(),
                );
            }
            elements.fill_zero(view)
        }
        (DataType::Fixed(_), ArrayBytesTarget::Optional(_)) => {
            Err(ExpectedOptionalBytesError.into())
        }
        (DataType::Optional(_), ArrayBytesTarget::Fixed(_)) | (DataType::Variable(_), _) => {
            Err(ExpectedFixedLengthBytesError.into())
        }
        _ => Err("unsupported data type layout".into()),
    }
}

fn copy_fill_value_bytes_into(
    data_type: &DataType,
    fill_value_bytes: &[u8],
    output_target: &mut ArrayBytesTarget<'_>,
    elements: TargetElements,
) -> Result<(), CodecError> {
    match (data_type, output_target) {
        (DataType::Optional(optional), ArrayBytesTarget::Optional(optional_target)) => {
            match parse_optional_fill_value(fill_value_bytes)? {
                None => {
                    copy_null_fill_value_into(
                        optional.data_type(),
                        &mut optional_target.data,
                        elements,
                    )?;
                    elements.fill(&mut optional_target.mask, &[0])?;
                }
                Some(inner_bytes) => {
                    copy_fill_value_bytes_into(
                        optional.data_type(),
                        inner_bytes,
                        &mut optional_target.data,
                        elements,
                    )?;
                    elements.fill(&mut optional_target.mask, &[1])?;
                }
            }
            Ok(())
        }
        (DataType::Fixed(fixed), ArrayBytesTarget::Fixed(view)) => {
            if fill_value_bytes.len() != fixed.size() {
                return Err(DataTypeFillValueError.into());
            }
            elements.fill(view, fill_value_bytes)
        }
        (DataType::Fixed(_), ArrayBytesTarget::Optional(_)) => {
            Err(ExpectedOptionalBytesError.into())
        }
        (DataType::Optional(_), ArrayBytesTarget::Fixed(_)) | (DataType::Variable(_), _) => {
            Err(ExpectedFixedLengthBytesError.into())
        }
        _ => Err("unsupported data type layout".into()),
    }
}

/// Helper function to decode `ArrayBytes` into a target, handling mask and data separately.
///
/// This function handles the common pattern of decoding `ArrayBytes` into an `ArrayBytesTarget`,
/// properly handling optional data types.
///
/// # Errors
/// Returns a [`CodecError`] if the bytes are incompatible with the target.
pub fn decode_into_array_bytes_target(
    bytes: &ArrayBytes,
    target: &mut ArrayBytesTarget<'_>,
) -> Result<(), CodecError> {
    decode_into_target_elements(bytes, target, TargetElements::All)
}

/// Decode `bytes` into the elements of `target` at `positions`.
///
/// Element `i` of `bytes` is written to element `positions[i]` of the target, where positions index the elements of the target in C order.
///
/// # Errors
/// Returns a [`CodecError`] if the bytes are incompatible with the target, the number of elements in `bytes` does not match the number of `positions`, or a position is out-of-bounds of the target.
pub fn decode_into_array_bytes_target_elements(
    bytes: &ArrayBytes,
    positions: &[usize],
    target: &mut ArrayBytesTarget<'_>,
) -> Result<(), CodecError> {
    decode_into_target_elements(bytes, target, TargetElements::Positions(positions))
}

fn decode_into_target_elements(
    bytes: &ArrayBytes,
    target: &mut ArrayBytesTarget<'_>,
    elements: TargetElements,
) -> Result<(), CodecError> {
    match (bytes, target) {
        (ArrayBytes::Fixed(data_bytes), ArrayBytesTarget::Fixed(data)) => {
            elements.copy(data, data_bytes)
        }
        (ArrayBytes::Optional(optional_bytes), ArrayBytesTarget::Optional(optional_target)) => {
            decode_into_target_elements(
                optional_bytes.data(),
                &mut optional_target.data,
                elements,
            )?;
            elements.copy(&mut optional_target.mask, optional_bytes.mask())
        }
        (ArrayBytes::Variable(..), _) => Err(ExpectedFixedLengthBytesError.into()),
        (ArrayBytes::Fixed(_), ArrayBytesTarget::Optional(_)) => {
            Err(ExpectedOptionalBytesError.into())
        }
        (ArrayBytes::Optional(..), ArrayBytesTarget::Fixed(_)) => {
            Err(ExpectedFixedLengthBytesError.into())
        }
    }
}

// impl<'a, 'b> From<&ArrayBytes<'a>> for ArrayBytes<'b> {
//     fn from(bytes: &ArrayBytes<'a>) -> Self {
//         match bytes {
//             Self::Fixed(bytes) => {
//                 let bytes = bytes.to_vec();
//                 ArrayBytes::<'b>::new_flen(bytes)
//             },
//             Self::Variable(bytes, offsets) => {
//                 let bytes: CowBytes<'b> = bytes.to_vec().into();
//                 let offsets: ArrayBytesOffsets<'b> = offsets.to_vec().into();
//                 ArrayBytes::new_vlen(bytes, offsets)
//             }
//         }
//     }
// }

// impl<'a> From<ArrayBytes<'_>> for ArrayBytes<'a> {
//     fn from(bytes: ArrayBytes<'_>) -> Self {
//         match bytes {
//             ArrayBytes::Fixed(bytes) => ArrayBytes::new_flen(bytes)
//             ArrayBytes::Variable(bytes, offsets) => ArrayBytes::new_vlen(bytes, offsets)
//         }
//     }
// }

impl<'a> From<&'a [u8]> for ArrayBytes<'a> {
    fn from(bytes: &'a [u8]) -> Self {
        ArrayBytes::new_flen(bytes)
    }
}

impl From<Vec<u8>> for ArrayBytes<'_> {
    fn from(bytes: Vec<u8>) -> Self {
        ArrayBytes::new_flen(bytes)
    }
}

impl<'a> From<std::borrow::Cow<'a, [u8]>> for ArrayBytes<'a> {
    fn from(bytes: std::borrow::Cow<'a, [u8]>) -> Self {
        ArrayBytes::new_flen(CowBytes::from(bytes))
    }
}

impl<'a, const N: usize> From<&'a [u8; N]> for ArrayBytes<'a> {
    fn from(bytes: &'a [u8; N]) -> Self {
        ArrayBytes::new_flen(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct FixedTestDataType(usize);

    zarrs_plugin::impl_extension_aliases!(FixedTestDataType, v3: "test.fixed");

    impl zarrs_data_type::DataTypeTraits for FixedTestDataType {
        fn into_data_type(self: std::sync::Arc<Self>) -> DataType {
            DataType::Fixed(self)
        }

        fn configuration(&self, _: zarrs_plugin::ZarrVersion) -> zarrs_metadata::Configuration {
            serde_json::Map::from_iter([("size".to_string(), self.0.into())]).into()
        }

        fn fill_value(
            &self,
            metadata: &zarrs_metadata::FillValueMetadata,
            _: zarrs_plugin::ZarrVersion,
        ) -> Result<FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
            metadata
                .as_bytes()
                .filter(|bytes| bytes.len() == self.0)
                .map(FillValue::new)
                .ok_or(zarrs_data_type::DataTypeFillValueMetadataError)
        }

        fn metadata_fill_value(
            &self,
            fill_value: &FillValue,
        ) -> Result<zarrs_metadata::FillValueMetadata, DataTypeFillValueError> {
            (fill_value.size() == self.0)
                .then(|| fill_value.as_ne_bytes().to_vec().into())
                .ok_or(DataTypeFillValueError)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl zarrs_data_type::FixedDataTypeTraits for FixedTestDataType {
        fn size(&self) -> usize {
            self.0
        }
    }

    #[derive(Debug)]
    struct VariableTestDataType;

    zarrs_plugin::impl_extension_aliases!(VariableTestDataType, v3: "test.variable");

    impl zarrs_data_type::DataTypeTraits for VariableTestDataType {
        fn into_data_type(self: std::sync::Arc<Self>) -> DataType {
            DataType::Variable(self)
        }

        fn configuration(&self, _: zarrs_plugin::ZarrVersion) -> zarrs_metadata::Configuration {
            zarrs_metadata::Configuration::default()
        }

        fn fill_value(
            &self,
            _: &zarrs_metadata::FillValueMetadata,
            _: zarrs_plugin::ZarrVersion,
        ) -> Result<FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
            Ok(FillValue::new(vec![]))
        }

        fn metadata_fill_value(
            &self,
            _: &FillValue,
        ) -> Result<zarrs_metadata::FillValueMetadata, DataTypeFillValueError> {
            Ok(zarrs_metadata::FillValueMetadata::Null)
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl zarrs_data_type::VariableDataTypeTraits for VariableTestDataType {}

    #[test]
    fn validate_fixed_bytes_checks_byte_length_before_narrowing() {
        let bytes = ArrayBytes::new_flen(vec![]);
        let zero_width = DataType::new(FixedTestDataType(0));
        let nonzero_width = DataType::new(FixedTestDataType(2));

        bytes.validate(u64::MAX, &zero_width).unwrap();
        assert!(bytes.validate(u64::MAX, &nonzero_width).is_err());
    }

    #[test]
    fn fill_value_rejects_unrepresentable_buffers() {
        let fixed = DataType::new(FixedTestDataType(1));
        let zero_width = DataType::new(FixedTestDataType(0));
        let variable = DataType::new(VariableTestDataType);
        let null = FillValue::new_optional_null();
        let empty = FillValue::from(());
        let first_oversized = isize::MAX as u64 + 1;
        let payload_width = 2 * size_of::<usize>();

        let cases = [
            (fixed, first_oversized, FillValue::from(0u8)),
            (
                DataType::new(FixedTestDataType(2)).to_optional(),
                first_oversized / 2,
                null.clone(),
            ),
            (
                variable.clone(),
                isize::MAX as u64 / size_of::<usize>() as u64,
                empty.clone(),
            ),
            (
                variable.clone(),
                isize::MAX as u64 / payload_width as u64 + 1,
                FillValue::new(vec![0; payload_width]),
            ),
            (
                variable.to_optional(),
                isize::MAX as u64 / size_of::<u32>() as u64,
                null.clone(),
            ),
            (zero_width.to_optional(), first_oversized, null),
            (
                zero_width.to_optional().to_optional(),
                first_oversized,
                empty.clone().into_optional().into_optional(),
            ),
        ];
        for (data_type, count, fill) in cases {
            assert!(matches!(
                ArrayBytes::new_fill_value(&data_type, count, &fill),
                Err(DataTypeFillValueError)
            ));
        }

        let bytes = ArrayBytes::new_fill_value(&zero_width, first_oversized, &empty).unwrap();
        assert!(bytes.into_fixed().unwrap().is_empty());
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn optional_validation_rejects_unrepresentable_count() {
        let data_type = DataType::new(FixedTestDataType(0)).to_optional();
        let empty = ArrayBytes::new_empty(&data_type).unwrap();
        let count = usize::MAX as u64 + 1;
        assert!(matches!(
            empty.validate(count, &data_type),
            Err(CodecError::InvalidNumberOfElements(_))
        ));
        assert!(matches!(
            ArrayBytes::new_fill_value(&data_type, count, &FillValue::new_optional_null()),
            Err(DataTypeFillValueError)
        ));
    }

    #[test]
    fn empty_bytes_preserve_optional_layers() {
        let data_type = DataType::new(FixedTestDataType(0))
            .to_optional()
            .to_optional();
        let bytes = ArrayBytes::new_empty(&data_type).unwrap();
        bytes.validate(0, &data_type).unwrap();
        let outer = bytes.into_optional().unwrap();
        assert!(outer.mask().is_empty());
        let (inner, _) = outer.into_parts();
        let inner = inner.into_optional().unwrap();
        assert!(inner.mask().is_empty());
        assert!(inner.data().clone().into_fixed().unwrap().is_empty());
    }

    #[test]
    fn extract_elements_validates_bounds_before_allocation() {
        let shape = [1u64 << 63];
        let subset = ArraySubset::new_with_shape(shape);
        assert!(matches!(
            extract_elements(&[], &subset, &shape, 1),
            Err(CodecError::InvalidByteRangeError(_))
        ));
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn extract_elements_rejects_overflowing_offsets() {
        // The end of the last run is 2 * 2^63 bytes, which wraps to 0 if unchecked
        let subset = ArraySubset::new_with_shape(vec![2]);
        assert!(matches!(
            extract_elements(&[], &subset, &[2], 1 << 63),
            Err(CodecError::InvalidByteRangeError(_))
        ));
    }

    #[test]
    fn extract_elements_validates_last_run() {
        let subset = ArraySubset::new_with_ranges(&[0..2, 1..3]);
        // The first run (bytes 2..6) exists, but the last run (bytes 10..14) is truncated.
        assert!(matches!(
            extract_elements(&[0; 12], &subset, &[3, 4], 2),
            Err(CodecError::InvalidByteRangeError(_))
        ));
    }

    #[test]
    fn extract_elements_only_requires_selected_bytes() {
        let subset = ArraySubset::new_with_ranges(&[0..2, 1..3]);
        let bytes: Vec<_> = (0..14).collect();
        assert_eq!(
            extract_elements(&bytes, &subset, &[3, 4], 2).unwrap(),
            [2, 3, 4, 5, 10, 11, 12, 13]
        );
        let empty_subset = ArraySubset::new_with_shape(vec![0, 4]);
        assert_eq!(
            extract_elements(&[], &empty_subset, &[3, 4], 2).unwrap(),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn validate_bytes_vlen_checks_count_and_last_offset() {
        let bytes = CowBytes::from(b"abcd".to_vec());
        let offsets = ArrayBytesOffsets::new(vec![0u32, 1, 4]).unwrap();
        assert!(validate_bytes_vlen(&bytes, &offsets, 2).is_ok());
        // The wrong number of elements
        assert!(validate_bytes_vlen(&bytes, &offsets, 1).is_err());
        assert!(validate_bytes_vlen(&bytes, &offsets, 3).is_err());
        // Trailing bytes after the last element
        let offsets = ArrayBytesOffsets::new(vec![0u32, 1, 3]).unwrap();
        assert!(validate_bytes_vlen(&bytes, &offsets, 2).is_err());
    }

    #[test]
    fn offsets_from_usize_uses_narrowest_width() {
        let input = vec![0, u32::MAX as usize];
        #[cfg(target_pointer_width = "32")]
        let input_ptr = input.as_ptr().cast::<u32>();
        let offsets = offsets_from_usize(input).unwrap();
        assert!(offsets.is_u32());
        #[cfg(target_pointer_width = "32")]
        assert_eq!(offsets.as_u32().unwrap().as_ptr(), input_ptr);

        #[cfg(target_pointer_width = "64")]
        {
            let input = vec![0, u32::MAX as usize + 1];
            let input_ptr = input.as_ptr().cast::<u64>();
            let offsets = offsets_from_usize(input).unwrap();
            assert!(offsets.is_u64());
            assert_eq!(offsets.as_u64().unwrap().as_ptr(), input_ptr);
        }
    }

    #[test]
    fn variable_is_fill_value() {
        let cases: &[(&[u8], &[u32], &str, bool)] = &[
            (b"abab", &[0, 2, 4], "ab", true),
            (b"abac", &[0, 2, 4], "ab", false),
            (b"ab", &[0, 1, 2], "ab", false),
            (b"abab", &[0, 4, 4], "ab", false),
            (b"", &[0, 0, 0], "", true),
            (b"a", &[0, 0, 1], "", false),
            (b"", &[0, 0], "ab", false),
            (b"xababz", &[1, 3, 5], "ab", true),
            (b"", &[0], "ab", true),
            (b"", &[0], "", true),
        ];
        for &(bytes, offsets, fill, expected) in cases {
            for offsets in [
                ArrayBytesOffsets::new(offsets.to_vec()).unwrap(),
                ArrayBytesOffsets::new(offsets.iter().copied().map(u64::from).collect::<Vec<_>>())
                    .unwrap(),
            ] {
                let mask = vec![true; offsets.num_elements()];
                let array_bytes = ArrayBytes::new_vlen(bytes, offsets).unwrap();
                assert_eq!(
                    array_bytes.is_fill_value(&FillValue::from(fill)),
                    expected,
                    "{array_bytes:?}, fill {fill:?}"
                );
                assert_eq!(
                    array_bytes
                        .with_optional_mask(mask)
                        .is_fill_value(&FillValue::from(Some(fill))),
                    expected
                );
            }
        }
    }

    #[test]
    fn optional_is_fill_value() {
        for depth in 1..=3 {
            let mut fill_value = FillValue::from(0x1234u16);
            let mut bytes = ArrayBytes::new_flen(0x1234u16.to_ne_bytes().repeat(3));
            let mut different_bytes = ArrayBytes::new_flen(0x1235u16.to_ne_bytes().repeat(3));
            let mut inner_null =
                ArrayBytes::new_flen(vec![0xff; 6]).with_optional_mask(vec![false; 3]);
            let mut inner_null_fill = FillValue::new_optional_null();
            for level in 0..depth {
                fill_value = fill_value.into_optional();
                different_bytes = different_bytes.with_optional_mask(vec![true; 3]);
                if level != 0 {
                    bytes = bytes.with_optional_mask(vec![true; 3]);
                    inner_null = inner_null.with_optional_mask(vec![true; 3]);
                    inner_null_fill = inner_null_fill.into_optional();
                }
            }
            assert!(
                bytes
                    .clone()
                    .with_optional_mask(vec![true; 3])
                    .is_fill_value(&fill_value)
            );
            assert!(
                !bytes
                    .with_optional_mask(vec![true, false, true])
                    .is_fill_value(&fill_value)
            );
            assert!(!different_bytes.is_fill_value(&fill_value));
            assert!(inner_null.is_fill_value(&inner_null_fill));
        }

        let null_fill = FillValue::new_optional_null();
        assert!(
            ArrayBytes::new_flen(vec![0xff; 6])
                .with_optional_mask(vec![false; 3])
                .is_fill_value(&null_fill)
        );
        assert!(
            !ArrayBytes::new_flen(vec![0xff; 6])
                .with_optional_mask(vec![false, true, false])
                .is_fill_value(&null_fill)
        );
        let empty = ArrayBytes::new_flen(vec![]).with_optional_mask(Vec::<bool>::new());
        assert!(empty.is_fill_value(&null_fill));
        assert!(!empty.is_fill_value(&FillValue::from(Some(1u16))));
        assert!(empty.is_fill_value(&FillValue::from(Some(""))));
        let strings = ArrayBytes::new_vlen(
            b"abcabc".as_slice(),
            ArrayBytesOffsets::new(vec![0u32, 3, 6]).unwrap(),
        )
        .unwrap()
        .with_optional_mask(vec![true; 2]);
        assert!(strings.is_fill_value(&FillValue::from(Some("abc"))));
        assert!(!strings.is_fill_value(&FillValue::from(Some("abd"))));
    }

    #[test]
    fn array_bytes_vlen() {
        let data = [0u8, 1, 2, 3, 4];
        assert!(ArrayBytes::new_vlen(&data, vec![0u32].try_into().unwrap()).is_ok());
        assert!(ArrayBytes::new_vlen(&data, vec![0u32, 5].try_into().unwrap()).is_ok());
        assert!(ArrayBytes::new_vlen(&data, vec![0u32, 5, 5].try_into().unwrap()).is_ok());
        assert!(ArrayBytes::new_vlen(&data, vec![0u32, 5, 6].try_into().unwrap()).is_err());
        assert!(ArrayBytes::new_vlen(&data, vec![0u32, 1, 3, 5].try_into().unwrap()).is_ok());
        assert!(ArrayBytes::new_vlen(&data, vec![0u32, 1, 3, 6].try_into().unwrap()).is_err());
    }

    #[test]
    fn test_flen_update_subset() {
        let mut bytes_array = vec![0u8; 4 * 4];
        {
            let bytes_array = UnsafeCellSlice::new(&mut bytes_array);
            let mut output_non_overlapping_0 = unsafe {
                // SAFETY: Only one view is created, so it is disjoint
                ArrayBytesFixedDisjointView::new(
                    bytes_array,
                    size_of::<u8>(),
                    &[4, 4],
                    ArraySubset::new_with_ranges(&[1..2, 1..3]),
                )
                .unwrap()
            };
            output_non_overlapping_0.copy_from_slice(&[1u8, 2]).unwrap();

            let mut output_non_overlapping_1 = unsafe {
                // SAFETY: Only one view is created, so it is disjoint
                ArrayBytesFixedDisjointView::new(
                    bytes_array,
                    size_of::<u8>(),
                    &[4, 4],
                    ArraySubset::new_with_ranges(&[3..4, 0..2]),
                )
                .unwrap()
            };
            output_non_overlapping_1.copy_from_slice(&[3u8, 4]).unwrap();
        }

        debug_assert_eq!(
            bytes_array,
            vec![0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 3, 4, 0, 0]
        );
    }

    #[test]
    fn update_array_bytes_array_subset_optional() -> Result<(), Box<dyn std::error::Error>> {
        // Create initial 4x4 array with optional u8 data
        // Layout (row-major):
        // [1  2  3  4 ]
        // [5  6  N  8 ]
        // [9  N  11 12]
        // [N  14 15 16]
        // where N = None
        let initial_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let initial_mask = vec![1u8, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]; // 0 = None
        let initial_bytes = ArrayBytes::new_flen(initial_data)
            .with_optional_mask(ArrayBytesMask::new(initial_mask)?);

        // Create 2x2 update for subset [1..3, 1..3]
        // This will update positions:
        // [1,1]=6, [1,2]=N, [2,1]=N, [2,2]=11
        // With new values:
        // [99 N ]
        // [97 96]
        let update_data = vec![99u8, 98, 97, 96];
        let update_mask = vec![1u8, 0, 1, 1]; // Second element is None
        let update_bytes =
            ArrayBytes::new_flen(update_data).with_optional_mask(ArrayBytesMask::new(update_mask)?);

        // Update using ArraySubset
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        let result = update_array_bytes(
            initial_bytes,
            &[4, 4],
            &subset,
            &update_bytes,
            &DataType::new(FixedTestDataType(1)).to_optional(),
        )?;

        // Update using indexer
        let indexer = vec![vec![3, 2], vec![3, 3]];
        let update_bytes =
            ArrayBytes::new_flen(vec![0u8, 255u8]).with_optional_mask(vec![false, true]);
        let result = update_array_bytes(
            result,
            &[4, 4],
            &indexer,
            &update_bytes,
            &DataType::new(FixedTestDataType(1)).to_optional(),
        )?;

        // Verify result
        let (result_data, result_mask) = result.into_optional()?.into_parts();
        let ArrayBytes::Fixed(result_data) = *result_data else {
            panic!("Expected fixed bytes")
        };

        // Expected layout after update:
        // [1  2  3  4 ]
        // [5  99 N  8 ]
        // [9  97 96 12]
        // [N  14 N 255]

        // Verify updated positions
        assert_eq!(result_data[5], 99); // [1,1]
        assert_eq!(result_mask[5], 1);

        assert_eq!(result_mask[6], 0); // [1,2] - should be None

        assert_eq!(result_data[9], 97); // [2,1]
        assert_eq!(result_mask[9], 1);

        assert_eq!(result_data[10], 96); // [2,2]
        assert_eq!(result_mask[10], 1);

        // Verify unchanged positions
        assert_eq!(result_data[0], 1); // [0,0]
        assert_eq!(result_mask[0], 1);

        assert_eq!(result_mask[14], 0); // [3,2]

        assert_eq!(result_data[15], 255); // [3,3]
        assert_eq!(result_mask[15], 1);

        assert_eq!(result_mask[12], 0); // [3,0] - still None

        Ok(())
    }

    #[test]
    fn update_array_bytes_array_subset_nested_optional_2_level()
    -> Result<(), Box<dyn std::error::Error>> {
        // Create initial 4x4 array with Option<Option<u8>> data
        // Layout (row-major, S=Some, N=None, outer/inner):
        // [SS(1)  SS(2)  SN     NN   ]
        // [SS(5)  SS(6)  SS(7)  SS(8)]
        // [SN     SS(10) SS(11) SN   ]
        // [NN     SS(14) SS(15) SS(16)]
        let initial_data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let initial_inner_mask = vec![1u8, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1]; // inner Some/None
        let initial_outer_mask = vec![1u8, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]; // outer Some/None

        let initial_bytes = ArrayBytes::new_flen(initial_data)
            .with_optional_mask(ArrayBytesMask::new(initial_inner_mask)?)
            .with_optional_mask(ArrayBytesMask::new(initial_outer_mask)?);

        // Create 2x2 update for subset [1..3, 1..3]
        // Update positions [1,1], [1,2], [2,1], [2,2]
        // New values:
        // [SS(99) SN   ]
        // [NN     SS(96)]
        let update_data = vec![99u8, 98, 97, 96];
        let update_inner_mask = vec![1u8, 0, 1, 1]; // [99 is valid, 98 is inner None, ...]
        let update_outer_mask = vec![1u8, 1, 0, 1]; // [outer valid, outer valid, outer None, outer valid]

        let update_bytes = ArrayBytes::new_flen(update_data)
            .with_optional_mask(ArrayBytesMask::new(update_inner_mask)?)
            .with_optional_mask(ArrayBytesMask::new(update_outer_mask)?);

        // Update using ArraySubset
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        let result = update_array_bytes(
            initial_bytes,
            &[4, 4],
            &subset,
            &update_bytes,
            &DataType::new(FixedTestDataType(1))
                .to_optional()
                .to_optional(),
        )?;

        // Verify result - extract outer optional layer
        let (result_middle, result_outer_mask) = result.into_optional()?.into_parts();

        // Extract inner optional layer
        let (result_data, result_inner_mask) = result_middle.into_optional()?.into_parts();
        let ArrayBytes::Fixed(result_data) = *result_data else {
            panic!("Expected fixed bytes")
        };

        // Verify updated positions
        // [1,1] = Some(Some(99))
        assert_eq!(result_data[5], 99);
        assert_eq!(result_inner_mask[5], 1);
        assert_eq!(result_outer_mask[5], 1);

        // [1,2] = Some(None)
        assert_eq!(result_inner_mask[6], 0); // inner None
        assert_eq!(result_outer_mask[6], 1); // outer Some

        // [2,1] = None (outer)
        assert_eq!(result_outer_mask[9], 0);

        // [2,2] = Some(Some(96))
        assert_eq!(result_data[10], 96);
        assert_eq!(result_inner_mask[10], 1);
        assert_eq!(result_outer_mask[10], 1);

        // Verify unchanged positions
        // [0,0] = Some(Some(1))
        assert_eq!(result_data[0], 1);
        assert_eq!(result_inner_mask[0], 1);
        assert_eq!(result_outer_mask[0], 1);

        // [0,2] = Some(None) - unchanged
        assert_eq!(result_inner_mask[2], 0);
        assert_eq!(result_outer_mask[2], 1);

        // [3,0] = None - unchanged
        assert_eq!(result_outer_mask[12], 0);

        Ok(())
    }

    #[test]
    fn extract_runs_variable_multi_element_runs() {
        // 3x4 variable-length elements; extracting rows 0..2, cols 1..3 gives two 2-element runs
        let data = b"aabbbccddddeefffgghhiijjjkkkllll";
        let lens = [2usize, 3, 2, 4, 2, 3, 2, 2, 2, 3, 3, 4];
        let mut raw_offsets = vec![0u64];
        for len in lens {
            raw_offsets.push(raw_offsets.last().unwrap() + len as u64);
        }
        let offsets = ArrayBytesOffsets::new(raw_offsets).unwrap();
        let subset = ArraySubset::new_with_ranges(&[0..2, 1..3]);
        let runs = subset.contiguous_linearised_indices(&[3, 4]).unwrap();
        assert_eq!(runs.contiguous_elements(), 2);

        let extracted = extract_runs_variable(data, &offsets, &runs).unwrap();
        assert!(extracted.offsets().is_u32());
        assert_eq!(extracted.bytes().as_ref(), b"bbbccfffgg");
        assert_eq!(
            extracted.offsets().iter().collect::<Vec<_>>(),
            vec![0, 3, 5, 8, 10]
        );

        let empty_runs = ArraySubset::new_with_ranges(&[0..0, 1..3])
            .contiguous_linearised_indices(&[3, 4])
            .unwrap();
        let empty = extract_runs_variable(data, &offsets, &empty_runs).unwrap();
        assert!(empty.offsets().is_u32());
        assert_eq!(empty.bytes().as_ref(), b"");
        assert_eq!(empty.offsets().iter().collect::<Vec<_>>(), vec![0]);

        let short_offsets = ArrayBytesOffsets::new(vec![0u32, 2, 5, 7, 11, 13]).unwrap();
        assert!(matches!(
            extract_runs_variable(&data[..13], &short_offsets, &runs),
            Err(CodecError::InvalidVariableSizedArrayOffsets)
        ));

        // Offsets beyond the bytes are an error
        assert!(matches!(
            extract_runs_variable(&data[..4], &offsets, &runs),
            Err(CodecError::InvalidVariableSizedArrayOffsets)
        ));

        assert_eq!(extract_runs_fixed(&[], &runs, 0).unwrap(), [] as [u8; 0]);
    }

    #[test]
    fn optional_fill_value_validates_framing_and_preserves_nested_null_masks() {
        fn view(bytes: &mut [u8], element_size: usize) -> ArrayBytesFixedDisjointView<'_> {
            // SAFETY: This is the only view into `bytes`.
            unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(bytes),
                    element_size,
                    &[1],
                    ArraySubset::new_with_shape([1]),
                )
                .unwrap()
            }
        }

        let opt_u8 = DataType::new(FixedTestDataType(1)).to_optional();
        let opt_fixed0 = DataType::new(FixedTestDataType(0)).to_optional();
        let opt_var = DataType::new(VariableTestDataType).to_optional();

        for bad in [vec![], vec![42, 2], vec![99, 0]] {
            let fill = FillValue::new(bad);
            assert!(matches!(
                ArrayBytes::new_fill_value(&opt_u8, 1, &fill),
                Err(DataTypeFillValueError)
            ));
            let mut data = [0];
            let mut mask = [0];
            let mut target = ArrayBytesTarget::from(view(&mut data, 1))
                .with_optional_mask(view(&mut mask, 1))
                .unwrap();
            assert!(copy_fill_value_into(&opt_u8, &fill, &mut target).is_err());

            let present = ArrayBytes::new_flen(vec![42u8]).with_optional_mask(vec![true]);
            let null = ArrayBytes::new_flen(vec![0u8]).with_optional_mask(vec![false]);
            assert!(!present.is_fill_value(&fill));
            assert!(!null.is_fill_value(&fill));
        }

        for empty_inner in [&opt_fixed0, &opt_var] {
            assert!(matches!(
                ArrayBytes::new_fill_value(empty_inner, 1, &FillValue::new(vec![])),
                Err(DataTypeFillValueError)
            ));
            let nested = empty_inner.to_optional();
            let bytes =
                ArrayBytes::new_fill_value(&nested, 2, &FillValue::new_optional_null()).unwrap();
            let outer = bytes.into_optional().unwrap();
            assert_eq!(outer.mask().as_ref(), &[0, 0]);
            let (inner, _) = outer.into_parts();
            let inner = inner.into_optional().unwrap();
            assert_eq!(inner.mask().as_ref(), &[0, 0]);
        }

        let fill = FillValue::new(vec![1]);
        let bytes = ArrayBytes::new_fill_value(&opt_fixed0, 1, &fill).unwrap();
        assert!(bytes.is_fill_value(&fill));
        assert_eq!(bytes.into_optional().unwrap().mask().as_ref(), &[1]);

        let mut data = [];
        let mut mask = [0];
        let mut target = ArrayBytesTarget::from(view(&mut data, 0))
            .with_optional_mask(view(&mut mask, 1))
            .unwrap();
        copy_fill_value_into(&opt_fixed0, &fill, &mut target).unwrap();
        assert_eq!(mask, [1]);
    }

    #[test]
    fn update_array_bytes_validates_lengths_for_non_subset_indexers() {
        let fixed_dt = DataType::new(FixedTestDataType(1));
        let opt_dt = fixed_dt.to_optional();
        let nested_opt_dt = opt_dt.to_optional();
        let var_dt = DataType::new(VariableTestDataType);

        let shape = [2, 2];
        let indexer = vec![vec![0, 0], vec![1, 1]];

        // Fixed input and update length mismatches with non-subset indexer
        let valid_fixed = ArrayBytes::new_flen(vec![1u8, 2, 3, 4]);
        let valid_fixed_update = ArrayBytes::new_flen(vec![10u8, 20]);
        for bad_bytes in [vec![0u8; 3], vec![0u8; 5]] {
            assert!(
                update_array_bytes(
                    ArrayBytes::new_flen(bad_bytes),
                    &shape,
                    &indexer,
                    &valid_fixed_update,
                    &fixed_dt,
                )
                .is_err()
            );
        }
        for bad_update in [vec![10u8], vec![10u8, 20, 30]] {
            assert!(
                update_array_bytes(
                    valid_fixed.clone(),
                    &shape,
                    &indexer,
                    &ArrayBytes::new_flen(bad_update),
                    &fixed_dt,
                )
                .is_err()
            );
        }
        assert!(
            update_array_bytes(
                valid_fixed,
                &shape,
                &vec![vec![2, 0], vec![1, 1]],
                &valid_fixed_update,
                &fixed_dt,
            )
            .is_err()
        );

        // Optional mask length mismatches with non-subset indexer
        let valid_opt = ArrayBytes::new_flen(vec![1u8, 2, 3, 4])
            .with_optional_mask(vec![true, true, true, true]);
        let valid_opt_update =
            ArrayBytes::new_flen(vec![10u8, 20]).with_optional_mask(vec![false, true]);
        for bad_mask in [vec![true; 3], vec![true; 5]] {
            assert!(
                update_array_bytes(
                    ArrayBytes::new_flen(vec![1u8, 2, 3, 4]).with_optional_mask(bad_mask),
                    &shape,
                    &indexer,
                    &valid_opt_update,
                    &opt_dt,
                )
                .is_err()
            );
        }
        for bad_update_mask in [vec![true], vec![true; 3]] {
            assert!(
                update_array_bytes(
                    valid_opt.clone(),
                    &shape,
                    &indexer,
                    &ArrayBytes::new_flen(vec![10u8, 20]).with_optional_mask(bad_update_mask),
                    &opt_dt,
                )
                .is_err()
            );
        }

        // Nested optional valid update succeeds and invalid inner layer is rejected
        let valid_nested = valid_opt
            .clone()
            .with_optional_mask(vec![true, false, true, true]);
        let valid_nested_update = valid_opt_update
            .clone()
            .with_optional_mask(vec![true, true]);
        let updated = update_array_bytes(
            valid_nested.clone(),
            &shape,
            &indexer,
            &valid_nested_update,
            &nested_opt_dt,
        )
        .unwrap();
        let outer = updated.into_optional().unwrap();
        assert_eq!(outer.mask().as_ref(), &[1, 0, 1, 1]);
        let (inner, _) = outer.into_parts();
        let inner = inner.into_optional().unwrap();
        assert_eq!(inner.mask().as_ref(), &[0, 1, 1, 1]);
        assert_eq!(
            inner.data().clone().into_fixed().unwrap().as_ref(),
            &[10, 2, 3, 20]
        );
        assert!(
            update_array_bytes(
                valid_nested,
                &shape,
                &indexer,
                &ArrayBytes::new_flen(vec![10u8, 20])
                    .with_optional_mask(vec![true, true, true])
                    .with_optional_mask(vec![true, true]),
                &nested_opt_dt,
            )
            .is_err()
        );

        // Variable length input/update offset count mismatches for both subset and indexer
        let valid_vlen = ArrayBytes::new_vlen(
            b"abc".to_vec(),
            ArrayBytesOffsets::new(vec![0u32, 1, 2, 3]).unwrap(),
        )
        .unwrap();
        let bad_vlen_4 = ArrayBytes::new_vlen(
            b"abcd".to_vec(),
            ArrayBytesOffsets::new(vec![0u32, 1, 2, 3, 4]).unwrap(),
        )
        .unwrap();
        let valid_vlen_update_2 = ArrayBytes::new_vlen(
            b"xy".to_vec(),
            ArrayBytesOffsets::new(vec![0u32, 1, 2]).unwrap(),
        )
        .unwrap();
        let bad_vlen_update_3 = ArrayBytes::new_vlen(
            b"xyz".to_vec(),
            ArrayBytesOffsets::new(vec![0u32, 1, 2, 3]).unwrap(),
        )
        .unwrap();
        let subset_2 = ArraySubset::new_with_ranges(&[0..2]);
        let indexer_2 = vec![vec![0], vec![2]];
        assert!(
            update_array_bytes(
                bad_vlen_4.clone(),
                &[3],
                &subset_2,
                &valid_vlen_update_2,
                &var_dt
            )
            .is_err()
        );
        assert!(
            update_array_bytes(bad_vlen_4, &[3], &indexer_2, &valid_vlen_update_2, &var_dt)
                .is_err()
        );
        assert!(
            update_array_bytes(
                valid_vlen.clone(),
                &[3],
                &subset_2,
                &bad_vlen_update_3,
                &var_dt
            )
            .is_err()
        );
        assert!(
            update_array_bytes(valid_vlen, &[3], &indexer_2, &bad_vlen_update_3, &var_dt).is_err()
        );
    }

    #[test]
    fn update_array_bytes_vlen_handles_duplicate_indexer_elements() {
        let var_dt = DataType::new(VariableTestDataType);
        let initial = ArrayBytes::new_vlen(
            b"hello!".to_vec(),
            ArrayBytesOffsets::new(vec![0u32, 5, 6]).unwrap(),
        )
        .unwrap();
        let indexer = vec![vec![0], vec![0]];

        // Shrinking duplicate updates should not underflow capacity arithmetic
        let shrink_update = ArrayBytes::new_vlen(
            b"abc".to_vec(),
            ArrayBytesOffsets::new(vec![0u32, 2, 3]).unwrap(),
        )
        .unwrap();
        let updated = update_array_bytes(initial, &[2], &indexer, &shrink_update, &var_dt).unwrap();
        let ArrayBytes::Variable(vlen) = updated else {
            panic!("expected variable bytes")
        };
        assert_eq!(vlen.bytes().as_ref(), b"c!");
        assert_eq!(vlen.offsets().iter().collect::<Vec<_>>(), vec![0, 1, 2]);

        // Growing then shrinking duplicate updates keeps only the final write per index
        let short = ArrayBytes::new_vlen(
            b"a!".to_vec(),
            ArrayBytesOffsets::new(vec![0u32, 1, 2]).unwrap(),
        )
        .unwrap();
        let mixed_update = ArrayBytes::new_vlen(
            b"longestx".to_vec(),
            ArrayBytesOffsets::new(vec![0u32, 7, 8]).unwrap(),
        )
        .unwrap();
        let updated = update_array_bytes(short, &[2], &indexer, &mixed_update, &var_dt).unwrap();
        let ArrayBytes::Variable(vlen) = updated else {
            panic!("expected variable bytes")
        };
        assert_eq!(vlen.bytes().as_ref(), b"x!");
        assert_eq!(vlen.offsets().iter().collect::<Vec<_>>(), vec![0, 1, 2]);
    }

    #[test]
    fn copy_fill_value_into_reports_target_layout_mismatches() {
        fn view(buffer: &mut [u8; 1]) -> ArrayBytesFixedDisjointView<'_> {
            // SAFETY: the view exclusively borrows the buffer.
            unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(buffer),
                    1,
                    &[1],
                    ArraySubset::new_with_shape([1]),
                )
                .unwrap()
            }
        }

        let fixed = DataType::new(FixedTestDataType(1));
        let optional = fixed.to_optional();
        let nested_optional = optional.to_optional();
        let null = FillValue::new_optional_null();
        let mut data = [0];
        let mut inner_mask = [0];
        let mut outer_mask = [0];

        let mut target = ArrayBytesTarget::Fixed(view(&mut data));
        assert!(matches!(
            copy_fill_value_into(&optional, &null, &mut target),
            Err(CodecError::ExpectedFixedLengthBytes(_))
        ));

        let mut target = target.with_optional_mask(view(&mut inner_mask)).unwrap();
        assert!(matches!(
            copy_fill_value_into(&fixed, &FillValue::from(0u8), &mut target),
            Err(CodecError::ExpectedOptionalBytes(_))
        ));
        assert!(matches!(
            copy_fill_value_into(&nested_optional, &null, &mut target),
            Err(CodecError::ExpectedFixedLengthBytes(_))
        ));

        let mut target = target.with_optional_mask(view(&mut outer_mask)).unwrap();
        assert!(matches!(
            copy_fill_value_into(&optional, &null, &mut target),
            Err(CodecError::ExpectedOptionalBytes(_))
        ));
    }

    #[test]
    fn copy_null_fill_value_into_zeros_strided_views_and_checks_element_size() {
        let opt_dt = DataType::new(FixedTestDataType(4)).to_optional();
        let nested_dt = opt_dt.to_optional();
        let fill = FillValue::new_optional_null();

        // 2x2 array, strided 2x1 subset on column 1 -> two disjoint 4-byte regions
        let subset = ArraySubset::new_with_ranges(&[0..2, 1..2]);
        let mut data = vec![0xFFu8; 4 * 4];
        let mut inner_mask = vec![1u8; 4];
        let mut outer_mask = vec![1u8; 4];

        let data_view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut data),
                4,
                &[2, 2],
                subset.clone(),
            )
            .unwrap()
        };
        let inner_mask_view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut inner_mask),
                1,
                &[2, 2],
                subset.clone(),
            )
            .unwrap()
        };
        let outer_mask_view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut outer_mask),
                1,
                &[2, 2],
                subset.clone(),
            )
            .unwrap()
        };
        let mut target = ArrayBytesTarget::from(data_view)
            .with_optional_masks([outer_mask_view, inner_mask_view])
            .unwrap();
        copy_fill_value_into(&nested_dt, &fill, &mut target).unwrap();

        // Selected column [*, 1] (elements 1 and 3) is zeroed, unselected [*, 0] is untouched
        assert_eq!(&data[0..4], &[0xFF; 4]);
        assert_eq!(&data[4..8], &[0; 4]);
        assert_eq!(&data[8..12], &[0xFF; 4]);
        assert_eq!(&data[12..16], &[0; 4]);
        assert_eq!(inner_mask, vec![1, 0, 1, 0]);
        assert_eq!(outer_mask, vec![1, 0, 1, 0]);

        // Mismatched target element size is still rejected
        let mut bad_data = vec![0xFFu8; 2 * 4];
        let bad_view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut bad_data),
                2,
                &[2, 2],
                subset.clone(),
            )
            .unwrap()
        };
        let bad_mask_view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(&mut inner_mask),
                1,
                &[2, 2],
                subset,
            )
            .unwrap()
        };
        let mut bad_target = ArrayBytesTarget::from(bad_view)
            .with_optional_mask(bad_mask_view)
            .unwrap();
        assert!(copy_fill_value_into(&opt_dt, &fill, &mut bad_target).is_err());
    }

    #[test]
    fn extract_runs_rejects_selecting_more_elements_than_exist() {
        // A zero stride repeats the first element
        let repeated = |count: u64| {
            ContiguousLinearisedIndices::new_strided(0, std::iter::once((count, 0))).unwrap()
        };
        let bytes = [1u8, 2, 3, 4];
        assert_eq!(
            extract_runs_fixed(&bytes, &repeated(2), 2).unwrap(),
            [1, 2, 1, 2]
        );
        assert!(matches!(
            extract_runs_fixed(&bytes, &repeated(3), 2),
            Err(CodecError::IncompatibleIndexer(_))
        ));
        // The extracted length would overflow
        assert!(matches!(
            extract_runs_fixed(&bytes, &repeated(u64::MAX), 2),
            Err(CodecError::IncompatibleIndexer(_))
        ));

        // The extracted bytes exceed the bytes of the elements, which size the output
        let offsets = ArrayBytesOffsets::new(vec![0u32, 2, 3]).unwrap();
        let extracted = extract_runs_variable(b"abc", &offsets, &repeated(2)).unwrap();
        assert_eq!(extracted.bytes().as_ref(), b"abab");
        assert_eq!(
            extracted.offsets().iter().collect::<Vec<_>>(),
            vec![0, 2, 4]
        );
        assert!(matches!(
            extract_runs_variable(b"abc", &offsets, &repeated(3)),
            Err(CodecError::IncompatibleIndexer(_))
        ));
        assert!(matches!(
            extract_runs_variable(b"abc", &offsets, &repeated(u64::MAX)),
            Err(CodecError::IncompatibleIndexer(_))
        ));
    }
}
