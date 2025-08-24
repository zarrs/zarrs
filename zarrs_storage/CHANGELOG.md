# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Implement `Clone` for `Error` structs
- Add `MaybeSend`/`MaybeSync` for WASM compatibility ([#245] by [@keller-mark])
- Add missing `AsyncReadableWritableStorage`

### Changed
- **Breaking**: Add upcasting methods to storage traits (`readable()`, `writable()`, etc.)
- **Breaking**: Change `byte_ranges: &[ByteRange]` parameter to `byte_ranges: impl Iterator<Item = ByteRange>` for
  - `extract_byte_ranges[{_concat,_read_seek,_read}]`
  - `[Async]ReadableStorageTraits::get_partial_values_key`

### Removed
- **Breaking**: Remove `extract_byte_ranges_unchecked`
- **Breaking**: Remove `extract_byte_ranges_concat_unchecked`
- **Breaking**: Remove `extract_byte_ranges_read`, no longer needed by `zarrs_zip`

[#245]: https://github.com/zarrs/zarrs/pull/245

## [0.3.4] - 2025-05-16

### Changed
- Update URLs to point to new `zarrs` GitHub organisation

## [0.3.3] - 2025-04-26

### Added
- Add additional tests

### Changed
- auto derive storage traits for Arc stores

## [0.3.2] - 2025-03-10

### Changed
- Bump `derive_more` to 2.0.0
- Bump `thiserror` to 2.0.12
- Fix clippy lints

## [0.3.1] - 2025-01-10

### Changed
- Bump `itertools` to 0.14

### Fixed
- Fix `unsafe_op_in_unsafe_fn` in lint

## [0.3.0] - 2024-11-15

### Added
 - Add `ByteRange::new` and `From` for `RangeBounds<u64>`
 - Add `PerformanceMetricsStorageAdapter::{keys_erased,reset}()`
 - Implement `Ord` and `PartialOrd` for `ByteRange`

### Changed
 - Bump `unsafe_cell_slice` to 0.2.0
 - **Breaking**: Change `ByteRange::FromEnd` to `ByteRange::Suffix`
 - **Breaking**: implement `AsyncReadableWritableStorageTraits` for `T: AsyncReadableStorageTraits + AsyncWritableStorageTraits`
 - **Breaking**: Bump MSRV to 1.77 (21 March, 2024)
 - **Breaking**: Rename `StoreKeyStartValue` to `StoreKeyOffsetValue`
   - Adds `offset` method and removes `start` and `end`
 - Count missing values as reads in `PerformanceMetricsStorageAdapter`
 - Print value lengths rather than values in `UsageLogStorageAdapter::set_partial_values()`

### Removed
 - **Breaking**: Remove `ByteRange::offset()`

## [0.2.2] - 2024-10-17

### Changed
 - Validate that chunk keys do not contain '//'

### Fixed
 - Fix new clippy warnings

## [0.2.1] - 2024-09-22

### Added
 - Add `storage_adapter::usage_log::UsageLogStorageAdapter`
 - Add `storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter`

## [0.2.0] - 2024-09-15

### Changed
 - Remove unused code related to store plugins
 - **Breaking**: Move filesystem/http/zip store implementations into separate crates:
   - `zarrs_filesystem`
   - `zarrs_http`
   - `zarrs_zip`

### Removed
 - **Breaking**: remove `http` and `zip` features

## [0.1.2] - 2024-09-03

### Changed
 - Use `doc_auto_cfg` on [docs.rs](https://docs.rs/)

## [0.1.1] - 2024-09-03

### Changed
 - Build with all features on [docs.rs](https://docs.rs/)

## [0.1.0] - 2024-09-02

### Added
 - Initial release
 - Split from the `storage` module of `zarrs` 0.17.0-dev

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_storage-v0.3.4...HEAD
[0.3.4]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_storage-v0.3.4
[0.3.3]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_storage-v0.3.3
[0.3.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_storage-v0.3.2
[0.3.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_storage-v0.3.1
[0.3.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_storage-v0.3.0
[0.2.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_storage-v0.2.2
[0.2.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_storage-v0.2.1
[0.2.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_storage-v0.2.0
[0.1.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_storage-v0.1.2
[0.1.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_storage-v0.1.1
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_storage-v0.1.0

[@keller-mark]: https://github.com/keller-mark
