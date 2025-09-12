# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## Changed
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

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_filesystem-v0.2.3...HEAD
[0.2.3]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.2.3
[0.2.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.2.2
[0.2.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.2.1
[0.2.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.2.0
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_filesystem-v0.1.0
