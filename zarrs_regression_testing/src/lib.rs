//! Data compatibility testing between the current `zarrs` and previous releases.
//!
//! The library is the machine-readable interface of the `zarrs_regression_testing` binary (see [`schema`]), so other tools can consume its output and drive it as an implementation adapter.
//! It does not depend on `zarrs`; use `default-features = false` to not build the binary's dependencies.
//!
//! This crate is not published. It is versioned with semantic versioning and released as `zarrs_regression_testing-vX.Y.Z` tags of the `zarrs` repository.

pub mod schema;
