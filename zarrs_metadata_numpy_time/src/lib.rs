//! `NumPy` `datetime64`/`timedelta64` types for the `zarrs` crate.
//!
//! This crate provides the shared type definitions used by both `zarrs_metadata_ext` and `zarrs_data_type`
//! for `NumPy` temporal data types.

#![warn(missing_docs)]

mod numpy_datetime64;
mod numpy_time_unit;
mod numpy_timedelta64;

pub use numpy_datetime64::NumpyDateTime64DataTypeConfigurationV1;
pub use numpy_time_unit::NumpyTimeUnit;
pub use numpy_timedelta64::NumpyTimeDelta64DataTypeConfigurationV1;
