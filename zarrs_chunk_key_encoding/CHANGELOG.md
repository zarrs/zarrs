# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-02-01

### Changed
- **Breaking**: Add `create()` method to `ChunkKeyEncodingTraits` trait
- **Breaking**: Remove `create_fn` parameter from `ChunkKeyEncodingPlugin::new()` and add `T: ChunkKeyEncodingTraits` bound

## [0.1.0] - 2026-01-14

### Added
- Split from the `zarrs::array::chunk_key_encoding` module of `zarrs` 0.23.0-beta.5

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_chunk_key_encoding-v0.2.0...HEAD
[0.2.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_chunk_key_encoding-v0.2.0
[0.1.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_chunk_key_encoding-v0.1.0
