# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `CodecTraitsV2` and  `CodecTraitsV3` traits

### Changed
- **Breaking**: Remove `create_fn` parameter from `CodecPluginV2::create()` and add `T: CodecTraitsV2` bound
- **Breaking**: Remove `create_fn` parameter from `CodecPluginV3::create()` and add `T: CodecTraitsV3` bound
- **Breaking**: Rename `ArrayRawBytesOffsetsOutOfBoundsError` to `ArrayBytesRawOffsetsOutOfBoundsError`
- **Breaking**: Rename `ArrayRawBytesOffsetsCreateError` to `ArrayBytesRawOffsetsCreateError`

## [0.1.0] - 2026-01-14

### Added
- Split from the `zarrs::array::codec` module of `zarrs` 0.23.0-beta.5

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_data_type-v0.1.0...HEAD
[0.1.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_data_type-v0.1.0
