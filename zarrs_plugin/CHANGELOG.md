# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2025-09-18

### Added
- Implement `Clone` for `Error` structs
- Add `MaybeSend`/`MaybeSync` for WASM compatibility in dependent crates ([#245] by [@keller-mark])

[#245]: https://github.com/zarrs/zarrs/pull/245

## [0.2.1] - 2025-05-16

### Added
- Add licence info to crate root docs

### Changed
- Update URLs to point to new `zarrs` GitHub organisation

## [0.2.0] - 2025-04-26

### Added
- Add additional tests

### Changed
- Bump `thiserror` to 2.0.12
- **Breaking**: `PluginUnsupportedError` no longer has a `configuration` parameter
- **Breaking**: `PluginMetadataInvalidError` now uses a `String` representation of metadata

### Removed
- Dependency on `zarrs_metadata`

### Fixed
- Broken Zarr spec URLs

## [0.1.0] - 2025-03-02

### Added
 - Initial release
 - Split from the `plugin` module of `zarrs` 0.20.0-dev

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_plugin-v0.2.2...HEAD
[0.2.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_plugin-v0.2.2
[0.2.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_plugin-v0.2.1
[0.2.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_plugin-v0.2.0
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_plugin-v0.1.0

[@keller-mark]: https://github.com/keller-mark
