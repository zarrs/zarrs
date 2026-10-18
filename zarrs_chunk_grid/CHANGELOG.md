# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `IncompatibleDimensionError`
- Add `unravel_index`
- Add `ChunkGridCreateError` for chunk grid creation and configuration failures

### Changed
- Reserve capacity for fallible indexer collections and optional chunk-shape conversion
- `ArraySubset` stores its start and shape inline for up to four dimensions, so creating, cloning, offsetting and intersecting subsets does not allocate
- `ContiguousIndices::new` (and so `contiguous_[linearised_]indices`) reuses the subset it is given and checks bounds without allocating
- `ContiguousLinearisedIndices` iterates forwards by adding strides, rather than unravelling and ravelling the indices of each contiguous run
  - **Breaking**: `ContiguousLinearisedIndices::new` takes the subset as a `&impl ArraySubsetTraits` rather than an `ArraySubset`, and the array shape as a `&[u64]` rather than a `Vec<u64>`
- `Indices`, `LinearisedIndices` and `ContiguousIndices` also iterate forwards by stepping their indices, unravelling only the first index of an iterator
  - All of these iterators step through only the dimensions in which they have more than one index
  - **Breaking**: `LinearisedIndices::{new,new_unchecked}` take the subset as a `&impl ArraySubsetTraits` rather than an `ArraySubset`, and the array shape as a `&[u64]` rather than an `ArrayShape`
- `ArraySubsetTraits::{linearised_indices,contiguous_linearised_indices}` no longer copy the subset
- Implement `Clone` for `ContiguousIndices[Into]Iterator`, `LinearisedIndices[Into]Iterator` and `ContiguousLinearisedIndices[Into]Iterator`
- `ArraySubset::{new_with_shape,new_with_start_shape}` accept any `impl AsRef<[u64]>` (e.g. a slice) rather than a `Vec<u64>`, so a subset can be created without allocating
  - **Breaking**: So do `ArraySubset::{new_with_start_end_inc,new_with_start_end_exc}`, rather than `ArrayIndices`
- **Breaking**: `ArrayIndices` is a `TinyVec<[u64; 4]>` rather than a `Vec<u64>`, and `ArrayIndicesTinyVec` is removed
  - Indices of up to four dimensions are held inline, so `ChunkGridTraits::{chunk_origin,chunk_indices,chunk_element_indices}` and the indices of iterators do not allocate
  - `Indexer` is implemented for lists of `Vec<u64>` as well as of `ArrayIndices` (e.g. `vec![vec![0, 1], vec![2, 3]]`)
- Bump `zarrs_metadata` to 0.7.6
- Zero sized dimensions are no longer considered _unlimited_
  - *Behavioural Change*: `array_indices_inbounds` and `chunk_indices_inbounds` now always return `false` for zero-sized arrays
- **Breaking**: Add the required `Indexer::validate` method for validating an indexer against an array shape
- **Breaking**: Add `chunk_edge_lengths()` to `ChunkGridTraits`
- **Breaking**: Change chunk grid plugin creation APIs to return `ChunkGridCreateError`
- Bump `itertools` to 0.15.0
- **Breaking**: Bump MSRV to 1.92 (11 December, 2025)

### Fixed
- Remove redundant slice conversions in the `Vec<ArrayIndices>` indexer implementation to satisfy Clippy
- An empty array subset is now always in-bounds, since it references no elements (unless it has an incompatible dimensionality)
- `ravel_indices` now returns `None` if `indices` and `shape` have a different length
- `Indexer::as_array_subset()` now returns `Some` for `Vec<Range<u64>>` and `&[Range<u64>]`
- `ArraySubset[Traits]::inbounds_shape` now returns `false` for a subset whose end exceeds `u64::MAX`, and `LinearisedIndices::new`, `ContiguousIndices::new` and `ContiguousLinearisedIndices::new` return an error for it, rather than wrapping in release builds
- `Indices` iterators over a subset with five or more dimensions no longer yield nothing if the number of elements in the subset exceeds `u64::MAX`
- `LinearisedIndices` and `ContiguousLinearisedIndices` panic if a linearised index exceeds `u64::MAX`, rather than wrapping in release builds, and no longer overflow in debug builds for an array with more than `u64::MAX` elements whose subset's linearised indices fit

## [0.5.1] - 2026-02-08

### Added
- Add `ArraySubset::offset` for offsetting by an `offset` (i.e., a new start at `self.start() + offset`).

## [0.5.0] - 2026-02-02

### Changed
- **Breaking**: Add `create()` method to `ChunkGridTraits` trait
- **Breaking**: Remove `create_fn` parameter from `ChunkGridPlugin::new()` and add `T: ChunkGridTraits` bound
- **Breaking**: Bump the MSRV to 1.91 and Rust 2024 edition

## [0.4.1] - 2026-01-14

### Added
- Add `ravel_indices()` utility function

## [0.4.0] - 2026-01-13

### Added
- Add `ChunkGrid::metadata()`

### Changed
- **Breaking**: Change `DataType[Runtime]Plugin` to the new `Plugin` system from `zarrs_plugin`
- **Breaking**: Revise `ChunkGridTraits`:
  - Replace `create_metadata()` with `configuration()`
- **Breaking**: Add `ExtensionName` supertrait to `ChunkGridTraits`

## [0.3.0] - 2026-01-09

### Added
- Add sealed `ArraySubsetTraits` trait
  - Implemented by `ArraySubset`, `[Range<u64>; N]`, `[Range<u64>]`, and `Vec<Range<u64>>`.
- Add sealed `ChunkShapeTraits` trait
  - Implemented by `T: AsRef<[NonZeroU64]>`

### Changed
- **Breaking**: Move various items to the crate root:
  - `ArraySubset` and `ArraySubsetError`
  - `Indexer`, `IndexerError`, and `IndexerIterator` (renamed from `IncompatibleIndexerError`)
  - `IncompatibleDimensionalityError`
  - `iterators` module
- **Breaking**: Change `ChunkGridTraits::chunks_in_array_subset()` to accept `&dyn ArraySubsetTraits` instead of `&ArraySubset`

### Removed
- **Breaking**: Remove `IncompatibleStartEndIndicesError`
- **Breaking**: Remove `IncompatibleOffsetError`
- **Breaking**: Make `indexer` and `array_subset` modules private

## [0.2.0] - 2025-12-31

### Added
- Add runtime chunk grid registration: `[un]register_chunk_grid` and `ChunkGridRuntime{Plugin,RegistryHandle}`, 

### Changed
- **Breaking**: `ChunkGridPlugin::new()` now takes a `default_name_fn` parameter
- **Breaking**: `ChunkGridPlugin::match_name()` now takes a `ZarrVersions` parameter (use `Plugin2::match_name()`)
- **Breaking**: Bump MSRV to 1.88

### Fixed
- Remove unused `half` dependency

## [0.1.0] - 2025-12-26

### Added
- Initial release
- Split from the `zarrs::array::chunk_grid`, `zarrs:array_subset` and `zarrs::indexer` modules of `zarrs` 0.23.0-dev
- Add `ArrayIndicesTinyVec` type

### Changed
- **Breaking**: Return `ArrayIndicesTinyVec` instead of `ArrayIndices` from:
  - `Indexer::iter_indices()`
  - `ChunkGridTraitsIterators::iter_chunk_indices_and_subsets()`
- **Breaking**: Change `Item` associated type of `[Par]Indices[Into]Iterator` from `ArrayIndices` to `ArrayIndicesTinyVec`
- **Breaking**: Change `Item` associated type of `[Par]ContiguousIndices[Into]Iterator` from `(ArrayIndices, u64)` to `(ArrayIndicesTinyVec, u64)`
- **Breaking**: Change return type `ChunkGridTraits::{array_shape,grid_shape}()` to `&[u64]` instead of `&ArrayShape`
- **Breaking**: Change `ChunkGridPlugin::new()` `create_fn` signature from `fn(&(MetadataV3, ArrayShape))` to `fn(&MetadataV3, &ArrayShape)`

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_chunk_grid-v0.5.1...HEAD
[0.5.1]: https://github.com/zarrs/zarrs/releases/tag/zarrs_chunk_grid-v0.5.1
[0.5.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_chunk_grid-v0.5.0
[0.4.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_chunk_grid-v0.4.1
[0.4.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_chunk_grid-v0.4.0
[0.3.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_chunk_grid-v0.3.0
[0.2.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_chunk_grid-v0.2.0
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_chunk_grid-v0.1.0
