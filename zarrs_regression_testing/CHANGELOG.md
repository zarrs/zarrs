# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This crate is not published. Releases are `zarrs_regression_testing-vX.Y.Z` tags, and its interface is described in `PROTOCOL.md`.

## [Unreleased]

### Added
- Report non-conformant `numcodecs.fletcher32` checksums written by releases, and add `schema::REJECTS_CONFORMANT` for releases that reject the conformant equivalent of the non-conformant data they write (known issues rather than regressions)
- Compare lossy data read by other subjects approximately (`schema::Data::approx_eq`), as decodings may differ slightly between implementations, and add `schema::DataType::numeric`
- Sample `numcodecs.fixedscaleoffset` data that is encoded in the range of the encoded data type, as out-of-range values are unspecified
- Add `schema::describe` to describe a case from its metadata, shape and chunk shape
- Add `schema::FLETCHER32_ODD_LENGTH`, the non-conformance in `schema::REJECTS_CONFORMANT`

## [0.1.0] - Unreleased

### Added
- Test data compatibility between the current `zarrs` and previous releases, with a text summary
- Write cases and results as JSON with `--json`
- Add the `cases`, `serve` and `build` subcommands
- Add the `schema` module: the helper protocol, case manifest and results without a `zarrs` dependency (with `default-features = false`)

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_regression_testing-v0.1.0...HEAD
[0.1.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_regression_testing-v0.1.0
