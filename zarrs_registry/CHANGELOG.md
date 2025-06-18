# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `data_type::COMPLEX_FLOAT4_E2M1FN`
- Add `data_type::COMPLEX_FLOAT6_{E2M3FN,E3M2FN}`
- Add `data_type::FLOAT8_{E3M4,E4M3,E4M3B11FNUZ,E4M3FNUZ,E5M2,E5M2FNUZ,E8M0FNU}`

## [0.1.3] - 2025-06-16

### Added
- Add `data_type::{NUMPY_DATETIME64,NUMPY_TIMEDELTA64}`

## [0.1.2] - 2025-06-08

### Added
- Add `[U]INT{2,4}`
- Add `FLOAT4_E2M1FN`, `FLOAT6_{E2M3FN,E3M2FN}`, `FLOAT8_{E3M4,E4M3,E4M3B11FNUZ,E4M3FNUZ,E5M2,E5M2FNUZ,E8M0FNU}`

### Changed
- Give `COMPLEX_FLOAT32` a unique identifier instead of `COMPLEX64`
- Give `COMPLEX_FLOAT64` a unique identifier instead of `COMPLEX128`

## [0.1.1] - 2025-05-16

### Added
- Add licence info to crate root docs

### Changed
- Update URLs to point to new `zarrs` GitHub organisation

## [0.1.0] - 2025-05-03

### Added
- Initial release (split from `zarrs_metadata` 0.4.0 during development)

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_registry-v0.1.3...HEAD
[0.1.3]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_registry-v0.1.3
[0.1.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_registry-v0.1.2
[0.1.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_registry-v0.1.1
[0.1.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs_registry-v0.1.0
