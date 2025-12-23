# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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


[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_chunk_grid-v0.1.0...HEAD
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_chunk_grid-v0.1.0
