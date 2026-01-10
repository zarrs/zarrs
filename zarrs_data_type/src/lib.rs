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
    register_data_type_v2, register_data_type_v3, unregister_data_type_v2, unregister_data_type_v3,
    DataTypePluginV2, DataTypePluginV3, DataTypeRuntimePluginV2, DataTypeRuntimePluginV3,
    DataTypeRuntimeRegistryHandleV2, DataTypeRuntimeRegistryHandleV3,
    DATA_TYPE_RUNTIME_REGISTRY_V2, DATA_TYPE_RUNTIME_REGISTRY_V3,
};
pub use fill_value::{DataTypeFillValueError, DataTypeFillValueMetadataError, FillValue};
