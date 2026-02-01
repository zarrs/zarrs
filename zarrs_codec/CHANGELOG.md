# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-02-02

### Added
- Add `CodecTraitsV2` and  `CodecTraitsV3` traits
- Add `CodecError::UnsupportedDataTypeCodec` variant for data type codec support errors
- Add `ExpectedFixedLengthBytesError`, `ExpectedVariableLengthBytesError`, and `ExpectedOptionalBytesError` error types

### Changed
- **Breaking**: Remove `create_fn` parameter from `CodecPluginV2::create()` and add `T: CodecTraitsV2` bound
- **Breaking**: Remove `create_fn` parameter from `CodecPluginV3::create()` and add `T: CodecTraitsV3` bound
- **Breaking**: Rename `ArrayRawBytesOffsetsOutOfBoundsError` to `ArrayBytesRawOffsetsOutOfBoundsError`
- **Breaking**: Rename `ArrayRawBytesOffsetsCreateError` to `ArrayBytesRawOffsetsCreateError`
- **Breaking**: `ArrayBytes::into_fixed()` now returns `Result<_, ExpectedFixedLengthBytesError>` instead of `Result<_, CodecError>`
- **Breaking**: `ArrayBytes::into_variable()` now returns `Result<_, ExpectedVariableLengthBytesError>` instead of `Result<_, CodecError>`
- **Breaking**: `ArrayBytes::into_optional()` now returns `Result<_, ExpectedOptionalBytesError>` instead of `Result<_, CodecError>`
- **Breaking**: `CodecError::ExpectedFixedLengthBytes`, `CodecError::ExpectedVariableLengthBytes`, and `CodecError::ExpectedOptionalBytes` now wrap their respective dedicated error types

### Removed
- **Breaking**: Remove `CodecError::ExpectedNonOptionalBytes` (replaced with `CodecError::ExpectedOptionalBytes`)
- **Breaking**: Remove `ArrayBytes::into_optional_bytes()` method (use `into_optional()` instead)
- **Breaking**: Remove `optional_nesting_depth`, `build_nested_optional_target`, `merge_chunks_vlen`, `merge_chunks_vlen_optional`, and `extract_decoded_regions_vlen` (moved to `zarrs` as private functions)

## [0.1.0] - 2026-01-14

### Added
- Split from the `zarrs::array::codec` module of `zarrs` 0.23.0-beta.5

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_codec-v0.2.0...HEAD
[0.2.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_codec-v0.2.0
[0.1.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_codec-v0.1.0
