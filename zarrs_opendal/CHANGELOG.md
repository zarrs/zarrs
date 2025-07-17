# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## Changed
- **Breaking**: Bump `opendal` to 0.54

## Removed
- **Breaking**: Remove `OpendalStore`
  - Use `AsyncOpendalStore` with the `AsyncToSyncStorageAdapter` instead

## [0.7.2] - 2025-05-16

## Changed
- Update URLs to point to new `zarrs` GitHub organisation

## [0.7.1] - 2025-05-04

### Changed
- Move integration tests into `tests/`

## [0.7.0] - 2025-04-06

### Changed
- Bump `thiserror` to 2.0.12
 - **Breaking**: Bump `opendal` supported range to 0.51-0.53
 - **Breaking**: Bump MSRV to 1.80 (25 July, 2024)

## [0.6.0] - 2025-02-24

### Changed
 - **Breaking**: Bump `opendal` supported range to 0.51-0.52

## [0.5.0] - 2024-12-24

### Changed
 - **Breaking**: Bump `opendal` to 0.51

## [0.4.0] - 2024-11-15

### Added
 - Add docs about precise version selection

### Changed
 - Bump `zarrs_storage` to 0.3.0-dev
 - **Breaking**: Bump MSRV to 1.77 (21 March, 2024)

## [0.3.1] - 2024-09-23

### Added
 - Add version compatibility matrix

## [0.3.0] - 2024-09-18

### Changed
 - **Breaking**: Bump maximum supported `opendal` to 0.50

## [0.2.0] - 2024-09-16

### Added
 - Add example to readme and root docs

### Changed
 - **Breaking**: Bump `zarrs_storage` to 0.2.0

## [0.1.0] - 2024-09-02

### Added
 - Initial release
 - Split from the `storage` module of `zarrs` 0.17.0-dev

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_opendal-v0.7.2...HEAD
[0.7.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_opendal-v0.7.2
[0.7.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_opendal-v0.7.1
[0.7.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_opendal-v0.7.0
[0.6.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_opendal-v0.6.0
[0.5.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_opendal-v0.5.0
[0.4.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_opendal-v0.4.0
[0.3.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_opendal-v0.3.1
[0.3.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_opendal-v0.3.0
[0.2.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_opendal-v0.2.0
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_opendal-v0.1.0
