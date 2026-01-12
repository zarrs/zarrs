//! Zarr versions.
//!
//! - [Zarr Version 3 Specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
//! - [Zarr Version 2 Specification](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html)

use std::fmt::Debug;

/// Marker trait for Zarr versions.
pub trait ZarrVersionTraits: Debug + Default {}

/// Zarr Version 3.
#[derive(Debug, Copy, Clone, Default)]
pub struct ZarrVersion3;

/// Zarr Version 2.
#[derive(Debug, Copy, Clone, Default)]
pub struct ZarrVersion2;

impl ZarrVersionTraits for ZarrVersion3 {}
impl ZarrVersionTraits for ZarrVersion2 {}

/// Zarr versions.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ZarrVersion {
    /// Zarr Version 2.
    V2,
    /// Zarr Version 3.
    V3,
}

impl From<ZarrVersion2> for ZarrVersion {
    fn from(_: ZarrVersion2) -> Self {
        ZarrVersion::V2
    }
}

impl From<ZarrVersion3> for ZarrVersion {
    fn from(_: ZarrVersion3) -> Self {
        ZarrVersion::V3
    }
}
