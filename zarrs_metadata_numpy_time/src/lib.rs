//! `NumPy` `datetime64`/`timedelta64` types for the `zarrs` crate.
//!
//! This crate provides the shared type definitions used by both `zarrs_metadata_ext` and `zarrs_data_type`
//! for `NumPy` temporal data types.
//!
//! ## Licence
//! `zarrs_metadata_numpy_time` is licensed under either of
//!  - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_metadata_numpy_time/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//!  - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_metadata_numpy_time/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

#![warn(missing_docs)]

mod numpy_datetime64;
mod numpy_time_unit;
mod numpy_timedelta64;

pub use numpy_datetime64::NumpyDateTime64DataTypeConfigurationV1;
pub use numpy_time_unit::NumpyTimeUnit;
pub use numpy_timedelta64::NumpyTimeDelta64DataTypeConfigurationV1;
