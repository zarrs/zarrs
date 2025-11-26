//! Zarr versions.
//!
//! - [Zarr Version 3 Specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
//! - [Zarr Version 2 Specification](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html)

use std::fmt::Debug;

/// Marker trait for Zarr versions.
pub trait ZarrVersion: Debug + Default {}

/// Zarr Version 3.
#[derive(Debug, Copy, Clone, Default)]
pub struct ZarrVersion3;

/// Zarr Version 2.
#[derive(Debug, Copy, Clone, Default)]
pub struct ZarrVersion2;

impl ZarrVersion for ZarrVersion3 {}
impl ZarrVersion for ZarrVersion2 {}
