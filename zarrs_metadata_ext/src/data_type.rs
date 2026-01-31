//! Zarr array data type metadata.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata-data-type>.

pub mod optional;

/// Re-export `numpy.datetime64` data type metadata from `zarrs_metadata_numpy_time`.
pub mod numpy_datetime64 {
    pub use zarrs_metadata_numpy_time::NumpyDateTime64DataTypeConfigurationV1;
}

/// Re-export `numpy.timedelta64` data type metadata from `zarrs_metadata_numpy_time`.
pub mod numpy_timedelta64 {
    pub use zarrs_metadata_numpy_time::NumpyTimeDelta64DataTypeConfigurationV1;
}

pub use zarrs_metadata_numpy_time::NumpyTimeUnit;
