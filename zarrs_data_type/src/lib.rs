//! The data type API for the [`zarrs`](https://docs.rs/zarrs/latest/zarrs/index.html) crate.
//!
//! ## Licence
//! `zarrs_data_type` is licensed under either of
//!  - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_data_type/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//!  - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_data_type/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

mod data_type;
mod data_type_plugin;
mod fill_value;

pub use data_type::{DataType, DataTypeTraits};
pub use data_type_plugin::{
    register_data_type, unregister_data_type, DataTypePlugin, DataTypeRuntimePlugin,
    DataTypeRuntimeRegistryHandle, DATA_TYPE_RUNTIME_REGISTRY,
};
pub use fill_value::{DataTypeFillValueError, DataTypeFillValueMetadataError, FillValue};
