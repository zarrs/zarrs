# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `CodecCreateError` for codec creation, reconfiguration, and binding failures
- Add `UnboundArrayTo{Array,Bytes}CodecTraits`
- Implement `[Async]BytesPartial{Encoder,Decoder}Traits` for `(Tstorage: *StorageTraits, StoreKey)`
- Add `ChunkGrid{Encoded,Decoded}Ref` and `[Async]ArrayPartialDecoderTraits::local_subchunk_grid[s]` for chunk-local subchunk grids

### Changed
- Use `ambisync` to share sync and async partial codec traits, default partial codecs, byte interval decoders, and codec partial factory methods
- **Breaking**: Decouple partial codec iterator lifetimes from decoder and encoder lifetimes
  - `AsyncBytesPartialDecoderTraits::partial_decode_many` no longer requires its `ByteRangeIterator` to live as long as the decoder borrow; returned bytes remain tied only to the decoder
  - `AsyncBytesPartialEncoderTraits::partial_encode_many` likewise accepts an independently lived `OffsetBytesIterator`
  - Downstream async trait implementations must change iterator arguments from `ByteRangeIterator<'a>` / `OffsetBytesIterator<'a, _>` to `ByteRangeIterator<'_>` / `OffsetBytesIterator<'_, _>` respectively
- **Breaking**: Refactor `ArrayTo{Array,Bytes}CodecTraits`
  - These traits are now associated with codecs that are _bound_ to a data type and fill value and validated at array creation time
  - **Breaking**: Add `data_type()`, `fill_value()`, `encoded_chunk_grid()` and `decoded_subchunk_grid[s]()` methods
  - **Breaking**: Remove `decoded_shape()` and `partial_decode_granularity()` methods
  - **Breaking**: Remove `data_type` and `fill_value` parameters from various methods
  - **Breaking**: Add `ArrayTo{Array,Bytes}CodecSubchunkingTraits` supertraits for resolving subchunk grids
    - `ArrayToArrayCodecSubchunkingIdentityTraits` and `ArrayToBytesCodecNoSubchunkingTraits` marker traits are available for common codecs

### Removed
- **Breaking**: Remove `ArrayCodecTraits::partial_decode_granularity`
- **Breaking**: Remove `[Async]StoragePartial{Encoder,Decoder}`
- **Breaking**: Remove `[Async]ArrayPartialEncoderTraits::into_dyn_decoder()`

## [0.2.1] - 2026-03-21

### Added
- Add `CodecSpecificOptions` for codec-specific runtime configuration
- Add `with_codec_specific_options` default method to `ArrayToArrayCodecTraits`, `ArrayToBytesCodecTraits`, and `BytesToBytesCodecTraits`

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

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_codec-v0.2.1...HEAD
[0.2.1]: https://github.com/zarrs/zarrs/releases/tag/zarrs_codec-v0.2.1
[0.2.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_codec-v0.2.0
[0.1.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_codec-v0.1.0
