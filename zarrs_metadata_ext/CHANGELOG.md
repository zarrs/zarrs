# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `index_location` field to `vlen` codec metadata
  - Adds `VlenIndexLocation` and `VlenCodecConfigurationV0_1` structs
  - Adds `V0_1` variant to `VlenCodecConfiguration` enum

### Changed
- **Breaking**: add `data_type` parameter to `v2_to_v3::fill_value_metadata_v2_to_v3()` ([#207] by [@jder])
- Map Zarr V2 `null` fill values to a default for more data types ([#207] by [@jder])
  - This matches the behaviour of `zarr-python` 3.0.3+
- Bump minimum `half` to 2.4.1

[#207]: https://github.com/zarrs/zarrs/pull/207

## [0.1.1] - 2025-06-16

### Added
- Add `data_type::numpy_datetime64::NumpyDateTime64DataTypeConfigurationV1`
- Add `data_type::numpy_timedelta64::NumpyTimeDelta64DataTypeConfigurationV1`
- Add `data_type::NumpyTimeUnit`

### Changed
- Bump minimum `serde` to 1.0.203

## [0.1.0] - 2025-05-16

### Added
- Split from the `zarrs_metadata` module of `zarrs_metadata` 0.4.0

### Changed
- rename `ArrayMetadataV2ToV3ConversionError` to `ArrayMetadataV2ToV3Error`
- rename `InvalidPermutationError` to `TransposeOrderError`
- change the suffix of experimental codec configurations from V1 to V0 (`gdeflate`, `squeeze`, `vlen`, `vlen_v2`)

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_metadata_ext-v0.1.1...HEAD
[0.1.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_metadata_ext-v0.1.1
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_metadata_ext-v0.1.0

[@jder]: https://github.com/jder
