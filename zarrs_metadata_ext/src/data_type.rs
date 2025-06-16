//! Zarr array data type metadata.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata-data-type>.

pub mod numpy_datetime64;
pub mod numpy_timedelta64;

mod numpy_time_unit;
pub use numpy_time_unit::NumpyTimeUnit;
