# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.2] - 2026-02-02

### Changed
- Change to Rust 2024 edition
- Bump `zarrs_storage` to 0.4.2

## [0.6.1] - 2025-12-31

### Changed
- Bump MSRV to 1.85 (matches `object_store` MSRV)

## [0.6.0] - 2025-12-26

### Changed
- **Breaking**: Bump `object_store` to 0.13

## [0.5.0] - 2025-09-18

### Changed
- **Breaking**: Bump `zarrs_storage` to 0.4.0

## [0.4.3] - 2025-06-20

### Added
- Add `fs`, `aws`, `azure`, `gcp` and `http` features to enable `object_store` features of the same name ([#211] by [@jder])

[#211]: https://github.com/zarrs/zarrs/pull/211

## [0.4.2] - 2025-05-16

## Changed
- Update URLs to point to new `zarrs` GitHub organisation

## [0.4.1] - 2025-05-04

### Changed
- Move integration tests into `tests/`

## [0.4.0] - 2025-03-10

### Changed
- **Breaking**: Bump `object_store` to 0.12.0
- Bump `thiserror` to 2.0.12

## [0.3.0] - 2024-11-15

### Added
 - Add docs about precise version selection

### Changed
 - Bump `zarrs_storage` to 0.3.0-dev
 - **Breaking**: Bump MSRV to 1.77 (21 March, 2024)

## [0.2.1] - 2024-09-23

### Added
 - Add version compatibility matrix

## [0.2.0] - 2024-09-15

### Added
 - Add example to readme and root docs

### Changed
 - **Breaking**: Bump `zarrs_storage` to 0.2.0

## [0.1.0] - 2024-09-02

### Added
 - Initial release
 - Split from the `storage` module of `zarrs` 0.17.0-dev

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_object_store-v0.6.2...HEAD
[0.6.2]: https://github.com/zarrs/zarrs/releases/tag/zarrs_object_store-v0.6.2
[0.6.1]: https://github.com/zarrs/zarrs/releases/tag/zarrs_object_store-v0.6.1
[0.6.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_object_store-v0.6.0
[0.5.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_object_store-v0.5.0
[0.4.3]: https://github.com/zarrs/zarrs/releases/tag/zarrs_object_store-v0.4.3
[0.4.2]: https://github.com/zarrs/zarrs/releases/tag/zarrs_object_store-v0.4.2
[0.4.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_object_store-v0.4.1
[0.4.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_object_store-v0.4.0
[0.3.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_object_store-v0.3.0
[0.2.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_object_store-v0.2.1
[0.2.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_object_store-v0.2.0
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_object_store-v0.1.0

[@jder]: https://github.com/jder
