# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Remove `parking_lot` dependency (use `std::sync::RwLock` instead)

## [0.3.6] - 2025-12-26

### Fixed
- Improve read performance

## [0.3.5] - 2025-11-14

### Fixed
- Fix test compilation on non-linux platforms since 0.22.2 ([#300] by [@clbarnes])

[#300]: https://github.com/zarrs/zarrs/pull/300

## [0.3.4] - 2025-10-31

### Fixed
- Raise the MSRV from 1.77 to 1.82
  - 1.82 has been the true MSRV since 0.3.0

## [0.3.3] - 2025-10-30

### Changed
- Failed write operations on a read-only filesystem return a more specific `std::io::ErrorKind::PermissionDenied` or `ReadOnlyFilesystem` instead of `StorageError::ReadOnly`

### Fixed
- Avoid brittle read-only check on `FilesystemStore` initialisation
  - This could fail when initialising from multiple threads/processes

## [0.3.2] - 2025-10-07

### Fixed
- Fix compilation on 32-bit systems

## [0.3.1] - 2025-10-05

### Added
- Add direct I/O read support ([#249] by [@ilan-gold])

[#249]: https://github.com/zarrs/zarrs/pull/249

## [0.3.0] - 2025-09-18

### Changed
- **Breaking**: Bump `zarrs_storage` to 0.4.0

## [0.2.3] - 2025-06-16

## Changed
- Update URLs to point to new `zarrs` GitHub organisation

## [0.2.2] - 2025-05-04

### Changed
- Move integration tests into `tests/`

### Fixed
- Skip `direct_io` test on linux systems that do not support direct IO

## [0.2.1] - 2025-04-26

### Changed
- Bump `itertools` to 0.14
- Bump `derive_more` to 2.0.0
- Bump `thiserror` to 2.0.12
- Bump `zarrs_storage` to 0.3.3

### Fixed
- Fix `clippy::single_char_pattern` lint

## [0.2.0] - 2024-11-15

### Changed
 - Bump `zarrs_storage` to 0.3.0
 - **Breaking**: Bump MSRV to 1.77 (21 March, 2024)

## [0.1.0] - 2024-09-15

### Added
 - Split from the `zarrs_storage` crate

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_filesystem-v0.3.6...HEAD
[0.3.6]: https://github.com/zarrs/zarrs/releases/tag/zarrs_filesystem-v0.3.6
[0.3.5]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.3.5
[0.3.4]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.3.4
[0.3.3]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.3.3
[0.3.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.3.2
[0.3.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.3.1
[0.3.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.3.0
[0.2.3]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.2.3
[0.2.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.2.2
[0.2.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.2.1
[0.2.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.2.0
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.1.0

[@ilan-gold]: https://github.com/ilan-gold
[@clbarnes]: https://github.com/clbarnes
