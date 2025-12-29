//! [Zarr](https://zarr-specs.readthedocs.io/) extensions metadata support for the [`zarrs`](https://docs.rs/zarrs/latest/zarrs/index.html) crate.
//!
//! This crate supports serialisation and deserialisation of Zarr V2 and V3 extensions metadata.
//!
//! This crate includes known metadata for Zarr V3 extension points (chunk grids, chunk key encodings, codecs, and data types), including:
//! - _Core_ extensions defined in the [Zarr V3 specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html),
//! - _Registered_ extensions defined at [zarr-developers/zarr-extensions](https://github.com/zarr-developers/zarr-extensions/), and
//! - `numcodecs` codecs and _experimental_ extensions in `zarrs` that have yet to be registered.
//!
//! ## Licence
//! `zarrs_metadata_ext` is licensed under either of
//!  - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_metadata_ext/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//!  - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_metadata_ext/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

pub mod array;
pub mod group;

pub mod array_storage_transformer;
pub mod chunk_grid;
pub mod chunk_key_encoding;
pub mod codec;
pub mod data_type;
