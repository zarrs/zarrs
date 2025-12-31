# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **Breaking**: Change `DataType` from an alias of `Arc<dyn DataTypeTraits>` to a newtype
- **Breaking**: Rename `DataTypeExtension` to `DataTypeTraits`

### Removed
- **Breaking**: Remove `DataTypeExtensionError` type

## [0.6.0] - 2025-12-31

### Added
- Add `DataType` type alias (`Arc<dyn DataTypeExtension>`)
- Add `DataTypeExtension::default_name()` method
- Add `DataTypeExtension::eq()` method with default implementation
- **Breaking**: Add `DataTypeExtension::as_any()` method
- Add runtime data type registration: `[un]register_data_type` and `DataTypeRuntime{Plugin,RegistryHandle}`

### Changed
- **Breaking**: `DataTypePlugin::new()` now takes a `default_name_fn` parameter
- **Breaking**: `DataTypePlugin::match_name()` now takes a `ZarrVersions` parameter
- Bump the MSRV to 1.88

### Removed
- **Breaking**: Remove `DataTypeExtensionBytesCodec` trait and `DataTypeExtensionBytesCodecError` enum
- **Breaking**: Remove `DataTypeExtensionPackBitsCodec` trait
- **Breaking**: Remove `DataTypeExtension::codec_bytes()` method
- **Breaking**: Remove `DataTypeExtension::codec_packbits()` method
- **Breaking**: Remove `DataTypeExtensionError::BytesCodec` variant

## [0.5.0] - 2025-12-26

### Added
- Add `FillValue::into_optional()`
- Implement `From<Option<T>>` for `FillValue` where `FillValue: From<T>`

### Changed
- **Breaking**: Rename `FillValue::new_null()` to `new_optional_null()`
- **Breaking**: bump `zarrs_metadata` to 0.7.0
- Bump `zarrs_plugin` to 0.2.3

### Removed
- **Breaking**: Remove `FillValue::is_null()`

## [0.4.2] - 2025-10-31

## Fixed
- Fix unintended MSRV increase to 1.80 from 1.77

## [0.4.1] - 2025-10-26

### Added
- Add support for `null` fill values: `FillValue::new_null()` and `is_null()`

## [0.4.0] - 2025-09-19

### Added
- Implement `Clone` for `Error` structs

### Changed
- **Breaking**: bump `zarrs_metadata` to 0.6.0

## [0.3.3] - 2025-09-18

*This release was yanked.*

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

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_data_type-v0.6.0...HEAD
[0.6.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_data_type-v0.6.0
[0.5.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_data_type-v0.5.0
[0.4.2]: https://github.com/zarrs/zarrs/releases/tag/zarrs_data_type-v0.4.2
[0.4.1]: https://github.com/zarrs/zarrs/releases/tag/zarrs_data_type-v0.4.1
[0.4.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_data_type-v0.4.0
[0.3.3]: https://github.com/zarrs/zarrs/releases/tag/zarrs_data_type-v0.3.3
[0.3.2]: https://github.com/zarrs/zarrs/releases/tag/zarrs_data_type-v0.3.2
[0.3.1]: https://github.com/zarrs/zarrs/releases/tag/zarrs_data_type-v0.3.1
[0.3.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_data_type-v0.3.0
[0.2.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_data_type-v0.2.0
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_data_type-v0.1.0
