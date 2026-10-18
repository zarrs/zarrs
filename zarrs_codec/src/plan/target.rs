//! Output targets and requests of planned decoding.

use std::num::NonZeroU64;
use std::sync::Arc;

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_chunk_grid::{ArraySubset, ArraySubsetTraits, Indexer, IndexerError};
use zarrs_data_type::DataType;

use crate::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, CodecError,
    build_nested_optional_target,
};

/// The views of an output that a task decodes into.
///
/// The data view is accompanied by one validity mask view per level of optional nesting, ordered from outermost to innermost.
pub struct TargetViews<'a> {
    data: ArrayBytesFixedDisjointView<'a>,
    masks: Vec<ArrayBytesFixedDisjointView<'a>>,
    // Declared last so it is dropped after the views that reference it.
    keepalive: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl std::fmt::Debug for TargetViews<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TargetViews")
            .field("subset", self.data.subset())
            .field("masks", &self.masks.len())
            .field("shared", &self.keepalive.is_some())
            .finish()
    }
}

impl<'a> TargetViews<'a> {
    /// Create target views from a data view and its mask views (outermost to innermost).
    #[must_use]
    pub fn new(
        data: ArrayBytesFixedDisjointView<'a>,
        masks: Vec<ArrayBytesFixedDisjointView<'a>>,
    ) -> Self {
        Self {
            data,
            masks,
            keepalive: None,
        }
    }

    /// Create target views of the whole of `target`.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the views of `target` are invalid.
    ///
    /// # Safety
    /// The views of `target` must not be used while the returned views are used, except through the returned views.
    pub unsafe fn from_target(
        target: &'a ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<Self, CodecError> {
        let (data, masks) = crate::extract_target_views(target);
        Ok(Self::new(
            unsafe { data.subdivide(data.subset().clone()) }?,
            masks
                .iter()
                .map(|mask| unsafe { mask.subdivide(mask.subset().clone()) })
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }

    /// The subset of the output covered by the views.
    #[must_use]
    pub fn subset(&self) -> &ArraySubset {
        self.data.subset()
    }

    /// The number of elements in the views.
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.data.num_elements()
    }

    /// Create views of `subset` of the output, which must be within [`subset`](Self::subset).
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `subset` is not within the subset of these views.
    ///
    /// # Safety
    /// The views of `subset` must not overlap with any other views of the same output that are used concurrently,
    /// including these views.
    pub unsafe fn subdivide(&self, subset: ArraySubset) -> Result<Self, CodecError> {
        let masks = self
            .masks
            .iter()
            .map(|mask| unsafe { mask.subdivide(subset.clone()) })
            .collect::<Result<Vec<_>, _>>()?;
        let data = unsafe { self.data.subdivide(subset) }?;
        Ok(Self {
            data,
            masks,
            keepalive: self.keepalive.clone(),
        })
    }

    /// Run `f` with views of `subset` of the output, which must be within [`subset`](Self::subset).
    ///
    /// Unlike [`subdivide`](Self::subdivide), the views borrow these views rather than holding the output, so creating
    /// them does not touch the reference count of a shared output.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `subset` is not within the subset of these views.
    ///
    /// # Safety
    /// The views of `subset` must not overlap with any other views of the same output that are used concurrently,
    /// including these views.
    pub unsafe fn with_subdivided<'s, R>(
        &'s self,
        subset: ArraySubset,
        f: impl FnOnce(TargetViews<'s>) -> R,
    ) -> Result<R, CodecError> {
        let masks = self
            .masks
            .iter()
            .map(|mask| unsafe { mask.subdivide(subset.clone()) })
            .collect::<Result<Vec<_>, _>>()?;
        let data = unsafe { self.data.subdivide(subset) }?;
        Ok(f(TargetViews {
            data,
            masks,
            keepalive: None,
        }))
    }

    /// Hold the output of the views through a new reference count.
    ///
    /// Views subdivided from the returned views then share a reference count with each other, rather than with
    /// every view of the output, so tasks on different threads that subdivide different views do not contend on it.
    #[must_use]
    pub fn with_local_keepalive(self) -> Self {
        let Self {
            data,
            masks,
            keepalive,
        } = self;
        Self {
            data,
            masks,
            keepalive: keepalive
                .map(|keepalive| Arc::new(keepalive) as Arc<dyn std::any::Any + Send + Sync>),
        }
    }

    /// Run `f` with the views as an [`ArrayBytesDecodeIntoTarget`].
    pub fn with_target<R>(self, f: impl FnOnce(ArrayBytesDecodeIntoTarget<'_>) -> R) -> R {
        let Self {
            data,
            masks,
            keepalive,
        } = self;
        // Moving the views shortens their lifetime to this scope, and they are dropped before `keepalive`.
        let _keepalive = keepalive;
        let mut data: ArrayBytesFixedDisjointView<'_> = data;
        let mut masks: Vec<ArrayBytesFixedDisjointView<'_>> = masks;
        f(build_nested_optional_target(&mut data, &mut masks))
    }

    /// Partially decode the elements of `indexer` with `decoder` into the views.
    ///
    /// This is the asynchronous counterpart of decoding into [`with_target`](Self::with_target), as the target is
    /// held across the decode.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the decode fails.
    #[cfg(feature = "async")]
    pub async fn async_partial_decode_into(
        self,
        decoder: &dyn crate::AsyncArrayPartialDecoderTraits,
        indexer: &dyn Indexer,
        options: &crate::CodecOptions,
        resources: &crate::Resources,
    ) -> Result<(), CodecError> {
        let Self {
            data,
            masks,
            keepalive,
        } = self;
        // As in `with_target`, the views are dropped before `keepalive`.
        let _keepalive = keepalive;
        let mut data: ArrayBytesFixedDisjointView<'_> = data;
        let mut masks: Vec<ArrayBytesFixedDisjointView<'_>> = masks;
        decoder
            .partial_decode_into(
                indexer,
                build_nested_optional_target(&mut data, &mut masks),
                options,
                resources,
            )
            .await
    }
}

/// An output buffer shared by the tasks of a drive that decode into it.
///
/// Shared outputs let tasks own their views (`'static`), so that they can run on other threads
/// without borrowing from the caller.
/// The buffer is allocated once, and every task writes its region directly into it.
pub struct SharedOutput {
    data: Vec<u8>,
    masks: Vec<Vec<u8>>,
    len: usize,
    // Referenced by views, so heap allocated with a stable address.
    _shape: Box<[u64]>,
}

impl std::fmt::Debug for SharedOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedOutput")
            .field("shape", &self._shape)
            .finish_non_exhaustive()
    }
}

impl SharedOutput {
    /// Allocate an output of `shape` elements of `data_type_size` bytes with `optional_nesting_depth` validity masks,
    /// and return views of the entire output.
    ///
    /// The output is not initialised: every element must be written through the views before
    /// [`into_array_bytes`](Self::into_array_bytes).
    ///
    /// # Panics
    /// Panics if the size of the output exceeds [`usize::MAX`].
    #[must_use]
    pub fn new(
        data_type_size: usize,
        shape: &[u64],
        optional_nesting_depth: usize,
    ) -> (Arc<Self>, TargetViews<'static>) {
        let len = usize::try_from(shape.iter().product::<u64>()).unwrap();
        let mut output = Self {
            data: Vec::with_capacity(len * data_type_size),
            masks: (0..optional_nesting_depth)
                .map(|_| Vec::with_capacity(len))
                .collect(),
            len,
            _shape: shape.into(),
        };
        let subset = ArraySubset::new_with_shape(shape);
        // SAFETY: the slices and shape reference heap allocations owned by the output, which do not
        // move when the output is moved into the `Arc`. Every view holds a clone of the `Arc` that is
        // dropped after the view, so the allocations outlive the views. Each allocation has exactly one
        // view of the entire output, and further views are only created by `subdivide`.
        let shape: &'static [u64] = unsafe { &*std::ptr::from_ref::<[u64]>(&output._shape) };
        let data_slice: UnsafeCellSlice<'static, u8> = unsafe {
            std::mem::transmute(UnsafeCellSlice::new_from_vec_with_spare_capacity(
                &mut output.data,
            ))
        };
        let mask_slices: Vec<UnsafeCellSlice<'static, u8>> = output
            .masks
            .iter_mut()
            .map(|mask| unsafe {
                std::mem::transmute(UnsafeCellSlice::new_from_vec_with_spare_capacity(mask))
            })
            .collect();
        let data = unsafe {
            ArrayBytesFixedDisjointView::new(data_slice, data_type_size, shape, subset.clone())
        }
        .expect("the view covers the entire output");
        let masks = mask_slices
            .into_iter()
            .map(|mask| {
                unsafe { ArrayBytesFixedDisjointView::new(mask, 1, shape, subset.clone()) }
                    .expect("the view covers the entire output")
            })
            .collect();
        let output = Arc::new(output);
        let views = TargetViews {
            data,
            masks,
            keepalive: Some(output.clone() as Arc<dyn std::any::Any + Send + Sync>),
        };
        (output, views)
    }

    /// Allocate an output of `shape` elements of `data_type`, and return views of the entire output.
    ///
    /// Optional data types have a validity mask per level of nesting. Returns [`None`] if the innermost data type is
    /// variable-length. See [`new`](Self::new).
    ///
    /// # Panics
    /// Panics if the size of the output exceeds [`usize::MAX`].
    #[must_use]
    pub fn for_data_type(
        data_type: &DataType,
        shape: &[u64],
    ) -> Option<(Arc<Self>, TargetViews<'static>)> {
        let mut innermost = data_type;
        let mut optional_nesting_depth = 0;
        while let Some(inner) = innermost.optional_inner() {
            innermost = inner;
            optional_nesting_depth += 1;
        }
        Some(Self::new(
            innermost.fixed_size()?,
            shape,
            optional_nesting_depth,
        ))
    }

    /// Return the output as array bytes.
    ///
    /// The data is not copied.
    ///
    /// # Errors
    /// Returns `self` if views of the output still exist.
    ///
    /// # Safety
    /// Every element of the output must have been written through its views.
    pub unsafe fn into_array_bytes(self: Arc<Self>) -> Result<ArrayBytes<'static>, Arc<Self>> {
        let Self {
            mut data,
            mut masks,
            len,
            _shape,
        } = Arc::try_unwrap(self)?;
        unsafe { data.set_len(data.capacity()) };
        let mut bytes = ArrayBytes::new_flen(data);
        for mask in &mut masks {
            unsafe { mask.set_len(len) };
        }
        for mask in masks.into_iter().rev() {
            bytes = bytes.with_optional_mask(mask);
        }
        Ok(bytes)
    }
}

/// Where a decoding task places its output.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)] // boxing would allocate per task
pub enum TaskTarget<'a> {
    /// Decode into views of an output with a fixed-length data type.
    Views(TargetViews<'a>),
    /// Return the decoded array bytes as the value of the task ([`TaskValue::Array`](super::TaskValue::Array)).
    ///
    /// Used for variable-length data types.
    Value,
}

/// The elements a decoding task decodes.
#[derive(Clone)]
pub enum PlanIndexer<'a> {
    /// A borrowed indexer.
    Borrowed(&'a dyn Indexer),
    /// An array subset.
    Subset(ArraySubset),
    /// A shared indexer.
    Shared(Arc<dyn Indexer>),
}

impl std::fmt::Debug for PlanIndexer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanIndexer")
            .field("output_shape", &self.as_indexer().output_shape())
            .finish_non_exhaustive()
    }
}

impl PlanIndexer<'_> {
    /// The indexer.
    #[must_use]
    pub fn as_indexer(&self) -> &dyn Indexer {
        match self {
            Self::Borrowed(indexer) => *indexer,
            Self::Subset(subset) => subset,
            Self::Shared(indexer) => indexer.as_ref(),
        }
    }

    /// Returns true if the indexer selects every element of a chunk of `shape` in C order.
    #[must_use]
    pub fn is_whole(&self, shape: &[NonZeroU64]) -> bool {
        match self {
            Self::Subset(subset) => subset.is_whole(bytemuck::must_cast_slice(shape)),
            _ => super::is_whole_chunk(self.as_indexer(), shape),
        }
    }

    /// Validate the indexer against a chunk of `shape`.
    ///
    /// # Errors
    /// Returns [`IndexerError`] if the indexer has an incompatible dimensionality or is out of bounds.
    pub fn validate(&self, shape: &[NonZeroU64]) -> Result<(), IndexerError> {
        match self {
            Self::Subset(subset) if subset.is_whole(bytemuck::must_cast_slice(shape)) => Ok(()),
            _ => self.as_indexer().validate(bytemuck::must_cast_slice(shape)),
        }
    }
}

impl From<ArraySubset> for PlanIndexer<'_> {
    fn from(subset: ArraySubset) -> Self {
        Self::Subset(subset)
    }
}

/// A request to decode elements of a chunk.
#[derive(Debug)]
pub struct DecodeRequest<'a> {
    /// The elements of the chunk to decode.
    pub indexer: PlanIndexer<'a>,
    /// Where to place the decoded elements, in the order of the indexer.
    pub target: TaskTarget<'a>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_output_subdivide_and_unwrap() {
        let (output, views) = SharedOutput::new(2, &[4, 4], 1);
        let data_ptr = output.data.as_ptr();
        // Subdivide into four disjoint quadrants, written by different threads
        let quadrants: Vec<_> = [[0..2, 0..2], [0..2, 2..4], [2..4, 0..2], [2..4, 2..4]]
            .into_iter()
            .map(|ranges| {
                unsafe { views.subdivide(ArraySubset::new_with_ranges(&ranges)) }.unwrap()
            })
            .collect();
        drop(views);
        assert!(unsafe { output.clone().into_array_bytes() }.is_err());
        std::thread::scope(|scope| {
            for (i, quadrant) in quadrants.into_iter().enumerate() {
                scope.spawn(move || {
                    quadrant.with_target(|target| {
                        let ArrayBytesDecodeIntoTarget::Optional(data, mask) = target else {
                            panic!()
                        };
                        let ArrayBytesDecodeIntoTarget::Fixed(data) = *data else {
                            panic!()
                        };
                        data.fill(&[i as u8; 2]).unwrap();
                        mask.fill(&[1]).unwrap();
                    });
                });
            }
        });
        let bytes = unsafe { output.into_array_bytes() }.unwrap();
        let (data, mask) = bytes.into_optional().unwrap().into_parts();
        assert_eq!(mask, vec![1; 16]);
        let data = data.into_fixed().unwrap();
        // The output is not copied
        assert_eq!(data.as_ptr(), data_ptr);
        assert_eq!(
            data.as_ref(),
            &[
                0, 0, 0, 0, 1, 1, 1, 1, //
                0, 0, 0, 0, 1, 1, 1, 1, //
                2, 2, 2, 2, 3, 3, 3, 3, //
                2, 2, 2, 2, 3, 3, 3, 3, //
            ]
        );
    }

    #[test]
    fn subdivide_rejects_out_of_bounds() {
        let (_output, views) = SharedOutput::new(1, &[4], 0);
        let half = unsafe { views.subdivide(ArraySubset::new_with_ranges(&[0..2])) }.unwrap();
        assert!(unsafe { half.subdivide(ArraySubset::new_with_ranges(&[1..3])) }.is_err());
    }
}
