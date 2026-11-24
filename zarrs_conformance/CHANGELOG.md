# Changelog

## [Unreleased]

### Changed
- Use `SyncFilesystemStore` for local Zarr data
- Match data types to their array byte layouts when printing elements, and return errors for unsupported layouts or zero-sized elements

### Fixed
- Report fill-buffer size errors from the array reader when printing array elements
