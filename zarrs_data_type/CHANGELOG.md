# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.2] - 2025-06-19

### Changed
- Implement `From<num::complex::Complex<T>>` for `FillValue: From<T>`
- Bump minimum `half` to 2.4.1

## [0.3.1] - 2025-06-08

### Added
- Implement `From<num::complex::Complex<half::{bf16,f16}>>` for `FillValue`

## [0.3.0] - 2025-05-16

### Changed
- **Breaking**: Bump `zarrs_metadata` to 0.5.0
- Update URLs to point to new `zarrs` GitHub organisation

## [0.2.0] - 2025-05-03

### Added
- Add support for data type extensions
  - Add `DataTypePlugin` and `DataTypeExtension`
  - Add `DataTypeExtensionBytesCodec`, `DataTypeExtensionBytesCodecError`
  - Add `DataTypeExtensionPackBitsCodec`
  - This crate no longer defines explicit data types

### Changed
- **Breaking**: Rename `IncompatibleFillValueError` to `DataTypeFillValueError`
- **Breaking**: Rename `IncompatibleFillValueMetadataError` to `DataTypeFillValueMetadataError`
- Bump `derive_more` to 2.0.0
- Bump `half` to 2.3.1
- Bump `thiserror` to 2.0.12

### Removed
- **Breaking**: Move `DataType` to `zarrs::array[::data_type]::DataType`
- **Breaking**: Remove `UnsupportedDataTypeError`

## [0.1.0] - 2025-01-24

### Added
- Initial release
- Split from the `zarrs::array::{data_type,fill_value}` modules of `zarrs` 0.20.0-dev

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_data_type-v0.3.2...HEAD
[0.3.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_data_type-v0.3.2
[0.3.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_data_type-v0.3.1
[0.3.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_data_type-v0.3.0
[0.2.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_data_type-v0.2.0
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_data_type-v0.1.0
