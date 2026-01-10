//! The chunk grid API for the [`zarrs`](https://docs.rs/zarrs/latest/zarrs/index.html) crate.
//!
//! ## Licence
//! `zarrs_chunk_grid` is licensed under either of
//!  - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_chunk_grid/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//!  - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_chunk_grid/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

mod array_subset_traits;
pub use array_subset_traits::ArraySubsetTraits;

mod array_subset;
pub use array_subset::{ArraySubset, ArraySubsetError};

mod indexer;
pub use indexer::{Indexer, IndexerError, IndexerIterator};

mod chunk_shape_traits;
pub use chunk_shape_traits::ChunkShapeTraits;

pub mod iterators;

use std::sync::{Arc, LazyLock};

use derive_more::{Deref, From};

pub use zarrs_metadata::{ArrayShape, ChunkShape};

/// An ND index to an element in an array or chunk.
pub type ArrayIndices = Vec<u64>;

/// An ND index to an element in an array or chunk.
/// Uses [`TinyVec`](tinyvec::TinyVec) for stack allocation up to 4 dimensions.
pub type ArrayIndicesTinyVec = tinyvec::TinyVec<[u64; 4]>;

use iterators::{IndicesIntoIterator, ParIndicesIntoIterator};

use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::Configuration;
use zarrs_plugin::{
    ExtensionAliases, ExtensionName, MaybeSend, MaybeSync, Plugin2, PluginCreateError,
    PluginUnsupportedError, RuntimePlugin2, RuntimeRegistry, ZarrVersion3, ZarrVersions,
};

// Note: Chunk grids are V3-only, so the version parameter is not needed in match functions

/// An incompatible dimensionality error.
#[derive(Copy, Clone, Debug, thiserror::Error)]
#[error("incompatible dimensionality {0}, expected {1}")]
pub struct IncompatibleDimensionalityError(usize, usize);

impl IncompatibleDimensionalityError {
    /// Create a new incompatible dimensionality error.
    #[must_use]
    pub const fn new(got: usize, expected: usize) -> Self {
        Self(got, expected)
    }
}

/// A chunk grid implementing [`ChunkGridTraits`].
#[derive(Debug, Clone, Deref, From)]
pub struct ChunkGrid(Arc<dyn ChunkGridTraits>);

impl<T: ChunkGridTraits + 'static> From<Arc<T>> for ChunkGrid {
    fn from(chunk_grid: Arc<T>) -> Self {
        Self(chunk_grid)
    }
}

impl ExtensionName for ChunkGrid {
    fn name(&self, version: ZarrVersions) -> Option<std::borrow::Cow<'static, str>> {
        self.0.name(version)
    }
}

/// A chunk grid plugin.
#[derive(derive_more::Deref)]
pub struct ChunkGridPlugin(Plugin2<ChunkGrid, MetadataV3, ArrayShape>);
inventory::collect!(ChunkGridPlugin);

impl ChunkGridPlugin {
    /// Create a new [`ChunkGridPlugin`] for a type implementing [`ExtensionAliases<ZarrVersion3>`].
    ///
    /// The `match_name_fn` is automatically derived from `T::matches_name`.
    pub const fn new<T: ExtensionAliases<ZarrVersion3>>(
        create_fn: fn(
            metadata: &MetadataV3,
            array_shape: &ArrayShape,
        ) -> Result<ChunkGrid, PluginCreateError>,
    ) -> Self {
        Self(Plugin2::new(|name| T::matches_name(name), create_fn))
    }
}

/// A runtime chunk grid plugin for dynamic registration.
pub type ChunkGridRuntimePlugin = RuntimePlugin2<ChunkGrid, MetadataV3, ArrayShape>;

/// A handle to a registered chunk grid plugin.
pub type ChunkGridRuntimeRegistryHandle = Arc<ChunkGridRuntimePlugin>;

/// Global runtime registry for chunk grid plugins.
pub static CHUNK_GRID_RUNTIME_REGISTRY: LazyLock<RuntimeRegistry<ChunkGridRuntimePlugin>> =
    LazyLock::new(RuntimeRegistry::new);

/// Register a chunk grid plugin at runtime.
///
/// Runtime-registered plugins take precedence over compile-time registered plugins.
///
/// # Returns
///
/// A handle that can be used to unregister the plugin later.
pub fn register_chunk_grid(plugin: ChunkGridRuntimePlugin) -> ChunkGridRuntimeRegistryHandle {
    CHUNK_GRID_RUNTIME_REGISTRY.register(plugin)
}

/// Unregister a runtime chunk grid plugin.
///
/// # Returns
///
/// `true` if the plugin was found and removed, `false` otherwise.
pub fn unregister_chunk_grid(handle: &ChunkGridRuntimeRegistryHandle) -> bool {
    CHUNK_GRID_RUNTIME_REGISTRY.unregister(handle)
}

impl ChunkGrid {
    /// Create a chunk grid.
    pub fn new<T: ChunkGridTraits + 'static>(chunk_grid: T) -> Self {
        let chunk_grid: Arc<dyn ChunkGridTraits> = Arc::new(chunk_grid);
        chunk_grid.into()
    }

    /// Create the metadata for the chunk grid.
    ///
    /// # Panics
    ///
    /// Panics if the chunk grid has no name for V3.
    #[must_use]
    pub fn metadata(&self) -> MetadataV3 {
        let name = self.name_v3().expect("chunk grid must have a V3 name");
        let configuration = self.configuration();
        if configuration.is_empty() {
            MetadataV3::new(name.into_owned())
        } else {
            MetadataV3::new_with_configuration(name.into_owned(), configuration)
        }
    }

    /// Create a chunk grid from metadata and an array shape.
    ///
    /// # Errors
    ///
    /// Returns a [`PluginCreateError`] if the metadata is invalid or not associated with a registered chunk grid plugin.
    pub fn from_metadata(
        metadata: &MetadataV3,
        array_shape: &[u64],
    ) -> Result<Self, PluginCreateError> {
        let name = metadata.name();

        // Check runtime registry first (higher priority)
        {
            let result = CHUNK_GRID_RUNTIME_REGISTRY.with_plugins(|plugins| {
                for plugin in plugins {
                    if plugin.match_name(name) {
                        return Some(plugin.create(metadata, &array_shape.to_vec()));
                    }
                }
                None
            });
            if let Some(result) = result {
                return result;
            }
        }

        // Fall back to compile-time registered plugins
        for plugin in inventory::iter::<ChunkGridPlugin> {
            if plugin.match_name(name) {
                return plugin.create(metadata, &array_shape.to_vec());
            }
        }
        Err(
            PluginUnsupportedError::new(metadata.name().to_string(), "chunk grid".to_string())
                .into(),
        )
    }
}

/// Chunk grid traits.
///
/// # Safety
/// - Chunks must be disjoint.
/// - Methods must check the dimensionality of arguments and returned indices/shapes/subsets must match the chunk grid dimensionality.
pub unsafe trait ChunkGridTraits:
    ExtensionName + core::fmt::Debug + MaybeSend + MaybeSync
{
    /// The configuration of the chunk grid.
    fn configuration(&self) -> Configuration;

    /// The dimensionality of the chunk grid.
    fn dimensionality(&self) -> usize;

    /// The array shape (i.e. number of elements).
    ///
    /// If supported by the chunk grid, zero sized dimensions are considered "unlimited".
    fn array_shape(&self) -> &[u64];

    /// The grid shape (i.e. number of chunks).
    ///
    /// If supported by the chunk grid, the grid will have zero sized dimensions where the array shape is zero, which is considered "unlimited".
    fn grid_shape(&self) -> &[u64];

    /// The shape of the chunk at `chunk_indices`.
    ///
    /// Returns [`None`] if the shape of the chunk at `chunk_indices` cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `chunk_indices` do not match the dimensionality of the chunk grid.
    fn chunk_shape(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkShape>, IncompatibleDimensionalityError>;

    /// The shape of the chunk at `chunk_indices`.
    ///
    /// Returns [`None`] if the shape of the chunk at `chunk_indices` cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `chunk_indices` do not match the dimensionality of the chunk grid.
    fn chunk_shape_u64(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayShape>, IncompatibleDimensionalityError>;

    /// The origin of the chunk at `chunk_indices`.
    ///
    /// Returns [`None`] if the chunk origin cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if the length of `chunk_indices` do not match the dimensionality of the chunk grid.
    fn chunk_origin(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError>;

    /// Return the [`ArraySubset`] of the chunk at `chunk_indices`.
    ///
    /// Returns [`None`] if the chunk subset cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `chunk_indices` do not match the dimensionality of the chunk grid.
    fn subset(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        let chunk_origin = self.chunk_origin(chunk_indices)?;
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        if let (Some(chunk_origin), Some(chunk_shape)) = (chunk_origin, chunk_shape) {
            let ranges = chunk_origin
                .into_iter()
                .zip(chunk_shape)
                .map(|(o, s)| o..(o + s.get()));
            Ok(Some(ArraySubset::from(ranges)))
        } else {
            Ok(None)
        }
    }

    /// Return the [`ArraySubset`] of the chunks in `chunks`.
    ///
    /// Returns [`None`] if the chunk subset cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `chunks` do not match the dimensionality of the chunk grid.
    fn chunks_subset(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        if chunks.dimensionality() != self.dimensionality() {
            Err(IncompatibleDimensionalityError::new(
                chunks.dimensionality(),
                self.dimensionality(),
            ))
        } else if let Some(end) = chunks.end_inc() {
            let chunk0 = self.subset(&chunks.start())?;
            let chunk1 = self.subset(&end)?;
            if let (Some(chunk0), Some(chunk1)) = (chunk0, chunk1) {
                let ranges = std::iter::zip(chunk0.start(), chunk1.end_exc())
                    .map(|(&s, e)| s..e)
                    .collect::<Vec<_>>();
                Ok(Some(ArraySubset::new_with_ranges(&ranges)))
            } else {
                Ok(None)
            }
        } else {
            Ok(Some(ArraySubset::new_empty(chunks.dimensionality())))
        }
    }

    /// The indices of a chunk which has the element at `array_indices`.
    ///
    /// Returns [`None`] if the chunk indices cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `array_indices` do not match the dimensionality of the chunk grid.
    fn chunk_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError>;

    /// The indices within the chunk of the element at `array_indices`.
    ///
    /// Returns [`None`] if the chunk element indices cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if `array_indices` do not match the dimensionality of the chunk grid.
    fn chunk_element_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError>;

    /// Check if array indices are in-bounds.
    ///
    /// Ensures array indices are within the array shape.
    /// Zero sized array dimensions are considered "unlimited" and always in-bounds.
    #[must_use]
    fn array_indices_inbounds(&self, array_indices: &[u64]) -> bool {
        array_indices.len() == self.dimensionality()
            && std::iter::zip(array_indices, self.array_shape())
                .all(|(&index, &shape)| shape == 0 || index < shape)
    }

    /// Check if chunk indices are in-bounds.
    ///
    /// Ensures chunk grid indices are within the chunk grid shape.
    /// Zero sized array dimensions are considered "unlimited" and always in-bounds.
    #[must_use]
    fn chunk_indices_inbounds(&self, chunk_indices: &[u64]) -> bool {
        chunk_indices.len() == self.dimensionality()
            && std::iter::zip(chunk_indices, self.grid_shape())
                .all(|(&index, &shape)| shape == 0 || index < shape)
    }

    /// Return an array subset indicating the chunks intersecting the given `region`.
    ///
    /// Returns [`None`] if the intersecting chunks cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if the region has an incorrect dimensionality.
    fn chunks_in_array_subset(
        &self,
        region: &dyn ArraySubsetTraits,
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        match region.end_inc() {
            Some(end) => {
                let chunks_start = self.chunk_indices(&region.start())?;
                let chunks_end = self.chunk_indices(&end)?;
                // .unwrap_or_else(|| self.grid_shape());

                Ok(
                    if let (Some(chunks_start), Some(chunks_end)) = (chunks_start, chunks_end) {
                        let ranges = std::iter::zip(&chunks_start, chunks_end)
                            .map(|(&s, e)| s..e + 1)
                            .collect::<Vec<_>>();
                        Some(ArraySubset::new_with_ranges(&ranges))
                    } else {
                        None
                    },
                )
            }
            None => Ok(Some(ArraySubset::new_empty(self.dimensionality()))),
        }
    }

    /// Return a serial iterator over the chunk indices of the chunk grid.
    fn iter_chunk_indices(&self) -> IndicesIntoIterator {
        let shape = self.grid_shape().to_vec();
        let n_chunks = shape.iter().product::<u64>();
        let n_chunks = usize::try_from(n_chunks).unwrap();
        IndicesIntoIterator {
            subset: ArraySubset::new_with_shape(shape),
            range: 0..n_chunks,
        }
    }

    /// Return a parallel iterator over the chunk indices of the chunk grid.
    fn par_iter_chunk_indices(&self) -> ParIndicesIntoIterator {
        let shape = self.grid_shape().to_vec();
        let n_chunks = shape.iter().product::<u64>();
        let n_chunks = usize::try_from(n_chunks).unwrap();
        ParIndicesIntoIterator {
            subset: ArraySubset::new_with_shape(shape),
            range: 0..n_chunks,
        }
    }
}

unsafe impl ChunkGridTraits for ChunkGrid {
    fn configuration(&self) -> Configuration {
        self.0.configuration()
    }

    fn dimensionality(&self) -> usize {
        self.0.dimensionality()
    }

    fn array_shape(&self) -> &[u64] {
        self.0.array_shape()
    }

    fn grid_shape(&self) -> &[u64] {
        self.0.grid_shape()
    }

    fn chunk_shape(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkShape>, IncompatibleDimensionalityError> {
        self.0.chunk_shape(chunk_indices)
    }

    fn chunk_shape_u64(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayShape>, IncompatibleDimensionalityError> {
        self.0.chunk_shape_u64(chunk_indices)
    }

    fn chunk_origin(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.0.chunk_origin(chunk_indices)
    }

    fn chunk_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.0.chunk_indices(array_indices)
    }

    fn chunk_element_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.0.chunk_element_indices(array_indices)
    }
}

unsafe impl ChunkGridTraits for Arc<dyn ChunkGridTraits> {
    fn configuration(&self) -> Configuration {
        self.as_ref().configuration()
    }

    fn dimensionality(&self) -> usize {
        self.as_ref().dimensionality()
    }

    fn array_shape(&self) -> &[u64] {
        self.as_ref().array_shape()
    }

    fn grid_shape(&self) -> &[u64] {
        self.as_ref().grid_shape()
    }

    fn chunk_shape(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkShape>, IncompatibleDimensionalityError> {
        self.as_ref().chunk_shape(chunk_indices)
    }

    fn chunk_shape_u64(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayShape>, IncompatibleDimensionalityError> {
        self.as_ref().chunk_shape_u64(chunk_indices)
    }

    fn chunk_origin(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.as_ref().chunk_origin(chunk_indices)
    }

    fn chunk_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.as_ref().chunk_indices(array_indices)
    }

    fn chunk_element_indices(
        &self,
        array_indices: &[u64],
    ) -> Result<Option<ArrayIndices>, IncompatibleDimensionalityError> {
        self.as_ref().chunk_element_indices(array_indices)
    }
}

/// Chunk grid iterators.
pub trait ChunkGridTraitsIterators: ChunkGridTraits {
    /// Return a serial iterator over the chunk subsets of the chunk grid.
    fn iter_chunk_subsets(&self) -> Box<dyn Iterator<Item = ArraySubset> + '_> {
        Box::new(self.iter_chunk_indices().map(|chunk_indices| {
            self.subset(&chunk_indices)
                .expect("matching dimensionality")
                .expect("inbounds chunk")
        }))
    }

    /// Return a serial iterator over the chunk indices and subsets of the chunk grid.
    fn iter_chunk_indices_and_subsets(
        &self,
    ) -> Box<dyn Iterator<Item = (ArrayIndicesTinyVec, ArraySubset)> + '_> {
        Box::new(self.iter_chunk_indices().map(|chunk_indices| {
            let chunk_subset = self
                .subset(&chunk_indices)
                .expect("matching dimensionality")
                .expect("inbounds chunk");
            (chunk_indices, chunk_subset)
        }))
    }
}

impl<T> ChunkGridTraitsIterators for T where T: ChunkGridTraits {}

/// Ravel ND indices to a linearised index.
///
/// Returns [`None`] if any `indices` are out-of-bounds of `shape`.
#[must_use]
fn ravel_indices(indices: &[u64], shape: &[u64]) -> Option<u64> {
    let mut index: u64 = 0;
    let mut count = 1;
    for (i, s) in std::iter::zip(indices, shape).rev() {
        if i >= s {
            return None;
        }
        index += i * count;
        count *= s;
    }
    Some(index)
}

/// Unravel a linearised index to ND indices.
#[must_use]
fn unravel_index(mut index: u64, shape: &[u64]) -> Option<ArrayIndicesTinyVec> {
    let total_size: u64 = shape
        .iter()
        .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))?;
    if index >= total_size {
        return None;
    }

    // Specialised routines for dimensions <=4, unrolled and no dynamic allocation
    match shape.len() {
        0 => Some(ArrayIndicesTinyVec::new()),
        1 => Some(tinyvec::tiny_vec!([u64; 4] => index % shape[0])),
        2 => {
            let i1 = index % shape[1];
            index /= shape[1];
            let i0 = index % shape[0];
            Some(tinyvec::tiny_vec!([u64; 4] => i0, i1))
        }
        3 => {
            let i2 = index % shape[2];
            index /= shape[2];
            let i1 = index % shape[1];
            index /= shape[1];
            let i0 = index % shape[0];
            Some(tinyvec::tiny_vec!([u64; 4] => i0, i1, i2))
        }
        4 => {
            let i3 = index % shape[3];
            index /= shape[3];
            let i2 = index % shape[2];
            index /= shape[2];
            let i1 = index % shape[1];
            index /= shape[1];
            let i0 = index % shape[0];
            Some(tinyvec::tiny_vec!([u64; 4] => i0, i1, i2, i3))
        }
        len => {
            // For 5+ dimensions, use Vec path with spare_capacity_mut
            let mut vec = Vec::with_capacity(len);

            {
                // SAFETY: `indices` are initialised and never read below
                let indices = unsafe { vec_spare_capacity_to_mut_slice(&mut vec) };

                // Fill in reverse order
                for i in (0..len).rev() {
                    indices[i] = index % shape[i];
                    index /= shape[i];
                }
            }

            // SAFETY: all `len` elements are initialised
            unsafe { vec.set_len(len) };

            Some(ArrayIndicesTinyVec::Heap(vec))
        }
    }
}

/// Get a mutable slice of the spare capacity in a vector.
///
/// # Safety
/// The caller must not read from the returned slice before it has been initialised.
unsafe fn vec_spare_capacity_to_mut_slice<T>(vec: &mut Vec<T>) -> &mut [T] {
    let spare_capacity = vec.spare_capacity_mut();
    // SAFETY: `spare_capacity` is valid for both reads and writes for len * size_of::<T>() many bytes, and it is properly aligned
    unsafe {
        std::slice::from_raw_parts_mut(
            spare_capacity.as_mut_ptr().cast::<T>(),
            spare_capacity.len(),
        )
    }
}
