# Changelog

## [Unreleased]

### Changed
- Match data types to their array byte layouts when printing elements, and reject unsupported zero-sized elements instead of panicking

### Fixed
- Report fill-buffer size errors from the array reader when printing array elements
