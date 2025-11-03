//! The Zarr extension point registry for the `zarrs` crate.
//!
//! Zarr V3 extension points include data types, codecs, chunk grids, chunk key encodings, and storage transformers.
//! Additionally, [`ZEP0009`] introduces a new `extension` metadata field for arbitrary array and group extensions.
//!
//! Zarr V2 extension points are limited to data types and codecs.
//!
//! ### Extension Aliasing
//!
//! An extension point `name` (or `id` in Zarr V2) may change over time as the Zarr specification and extensions evolve.
//! For example, the `bytes` codec was originally called `endian` in the provisionally accepted Zarr V3 specification.
//! Aliasing of `name`s enables seamless backward compatibility and interoperability between different Zarr versions, Zarr implementations, and extension versions (where compatible).
//!
//! This crate defines a *unique extension identifier* for each known extension in each extension point.
//! Known aliases for an extension can be mapped to their unique extension identifier that can be mapped to a default `name` for metadata serialisation.
//! Extension alias maps can be mutated to support custom extensions or to override the defaults.
//!
//! *Unique extension identifiers* are an implementation detail and may not match extension `name`s.
//! However, they are publicly exposed to permit manipulation of aliases and default names.
//!
//! ### Extension Name Conventions
//! Prior to [`ZEP0009`], a Zarr V3 extension point `name` was encouraged to be a unique URI pointing to a specification of the extension.
//! [`ZEP0009`] revises conventions for extension point naming:
//! - private or experimental extensions must use a *namespaced name* (e.g. `numcodecs.adler32`, `zarrs.vlen`, etc.), and
//! - extensions registered  at [`zarr-extensions`] can use a *raw name*, such as `bfloat16`, or a *namespaced name*.
//!
//! ### Alias Maps
//! This crate provides [`Default`] alias maps for Zarr V2 and V3 extension points.
//! The alias maps define the list of known compatible extension point aliases and a *default* `name` for each extension point.
//! The default `name` is what will be serialised when creating a new array or group, unless overriden.
//! Alias maps are currently limited to codecs and data types, but they may be extended to other extension types in the future.
//!
//! In the [`zarrs`] crate, the extension point aliases can be configured to support custom extensions or to override the defaults, such as if [`zarrs_metadata`](`crate`) is lagging [`zarr-extensions`].
//! This crate will be continually updated to include new compatible extension point aliases as they are developed in other implementations and registered at [`zarr-extensions`].
//!
//! ## Licence
//! `zarrs_registry` is licensed under either of
//!  - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_registry/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//!  - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_registry/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
//!
//! [`ZEP0009`]: https://zarr.dev/zeps/draft/ZEP0009.html
//! [`zarrs`]: https://docs.rs/zarrs/latest/zarrs/index.html
//! [`zarr-extensions`]: https://github.com/zarr-developers/zarr-extensions

mod zarr_version;
pub use zarr_version::{ZarrVersion, ZarrVersion2, ZarrVersion3};

mod extension_type;
pub use extension_type::{
    ExtensionType, ExtensionTypeChunkGrid, ExtensionTypeChunkKeyEncoding, ExtensionTypeCodec,
    ExtensionTypeDataType, ExtensionTypeStorageTransformer,
};

mod extension_aliases;
pub use extension_aliases::{
    ExtensionAliasMapRegex, ExtensionAliasMapString, ExtensionAliases, ExtensionNameMap,
};

mod extension_aliases_chunk_grid;
pub use extension_aliases_chunk_grid::ExtensionAliasesChunkGridV3;

mod extension_aliases_chunk_key_encoding;
pub use extension_aliases_chunk_key_encoding::ExtensionAliasesChunkKeyEncodingV3;

mod extension_aliases_codec;
pub use extension_aliases_codec::{ExtensionAliasesCodecV2, ExtensionAliasesCodecV3};

mod extension_aliases_data_type;
pub use extension_aliases_data_type::{ExtensionAliasesDataTypeV2, ExtensionAliasesDataTypeV3};

mod extension_aliases_storage_transformer;
pub use extension_aliases_storage_transformer::ExtensionAliasesStorageTransformerV3;

/// Unique identifiers for _chunk grid_ extensions.
pub mod chunk_grid;

/// Unique identifiers for _chunk key encoding_ extensions.
pub mod chunk_key_encoding;

/// Unique identifiers for _codec_ extensions.
pub mod codec;

/// Unique identifiers for _data type_ extensions.
pub mod data_type;

/// Unique identifiers for _storage transformer_ extensions.
pub mod storage_transformer;
