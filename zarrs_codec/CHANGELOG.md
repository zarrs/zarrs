# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `CodecTraitsV2` and  `CodecTraitsV3` traits
- Add `CodecError::UnsupportedDataTypeCodec` variant for data type codec support errors
- Add `CodecError::ArrayBytesValidateError` variant
- Add `ArrayBytesExt` extension trait with `extract_array_subset()` method

### Changed
- **Breaking**: Remove `create_fn` parameter from `CodecPluginV2::create()` and add `T: CodecTraitsV2` bound
- **Breaking**: Remove `create_fn` parameter from `CodecPluginV3::create()` and add `T: CodecTraitsV3` bound
- **Breaking**: Move core `ArrayBytes` types to `zarrs_data_type` (re-exported for compatibility)
- **Breaking**: `ArrayBytes::into_fixed()` now returns `Result<_, ExpectedFixedLengthBytesError>` instead of `Result<_, CodecError>`
- **Breaking**: `ArrayBytes::into_variable()` now returns `Result<_, ExpectedVariableLengthBytesError>` instead of `Result<_, CodecError>`
- **Breaking**: `ArrayBytes::into_optional()` now returns `Result<_, ExpectedOptionalBytesError>` instead of `Result<_, CodecError>`
- Bump `zarrs_data_type` to 0.9.0

### Removed
- **Breaking**: Remove `optional_nesting_depth`, `build_nested_optional_target`, `merge_chunks_vlen`, `merge_chunks_vlen_optional`, and `extract_decoded_regions_vlen` (moved to `zarrs` as private functions)

## [0.1.0] - 2026-01-14

### Added
- Split from the `zarrs::array::codec` module of `zarrs` 0.23.0-beta.5

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_data_type-v0.1.0...HEAD
[0.1.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_data_type-v0.1.0
