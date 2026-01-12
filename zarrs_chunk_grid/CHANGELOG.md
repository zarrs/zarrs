# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_chunk_grid-v0.4.0...HEAD
[0.4.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_chunk_grid-v0.4.0
[0.3.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_chunk_grid-v0.3.0
[0.2.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_chunk_grid-v0.2.0
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_chunk_grid-v0.1.0
