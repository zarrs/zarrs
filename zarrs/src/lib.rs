//! `zarrs` is Rust library for the [Zarr](https://zarr.dev) storage format for multidimensional arrays and metadata.
//!
//! If you are a Python user, check out [`zarrs-python`](https://github.com/zarrs/zarrs-python).
//! It includes a high-performance codec pipeline for the reference [`zarr-python`](https://github.com/zarr-developers/zarr-python) implementation.
//!
//! `zarrs` supports [Zarr V3](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html) and a V3 compatible subset of [Zarr V2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html).
//! It is fully up-to-date and conformant with the Zarr 3.1 specification with support for:
//! - all *core extensions* (data types, codecs, chunk grids, chunk key encodings, storage transformers),
//! - all accepted [Zarr Enhancement Proposals (ZEPs)](https://zarr.dev/zeps/) and several draft ZEPs:
//!   - ZEP 0003: Variable chunking
//!   - ZEP 0007: Strings
//!   - ZEP 0009: Zarr Extension Naming
//! - various registered extensions from [zarr-developers/zarr-extensions/](https://github.com/zarr-developers/zarr-extensions/),
//! - experimental codecs and data types intended for future registration, and
//! - user-defined custom extensions and stores.
//!
//! A changelog can be found [here](https://github.com/zarrs/zarrs/blob/main/CHANGELOG.md).
//! Correctness issues with past versions are [detailed here](https://github.com/zarrs/zarrs/blob/main/doc/correctness_issues.md).
//!
//! Developed at the [Department of Materials Physics](https://physics.anu.edu.au/research/mp/), Australian National University, Canberra, Australia.
//!
//! ## Getting Started
//! - Review the [implementation status](#implementation-status) which summarises zarr version support, array support (codecs, data types, etc.) and storage support.
//! - Read [The `zarrs` Book].
//! - View the [examples](https://github.com/zarrs/zarrs/tree/main/zarrs/examples) and [the example below](#examples).
//! - Read the [documentation](https://docs.rs/zarrs/latest/zarrs/).
//! - Check out the [`zarrs` ecosystem](#zarrs-ecosystem).
//!
//! ## Implementation Status
//!
//! #### Zarr Version Support
//!
//! `zarrs` has first-class Zarr V3 support and additionally supports a *compatible subset* of Zarr V2 data that:
//! - can be converted to V3 with only a metadata change, and
//! - uses array metadata that is recognised and supported for encoding/decoding.
//!
//! `zarrs` supports forward conversion from Zarr V2 to V3. See ["Converting Zarr V2 to V3"](https://book.zarrs.dev/v2_to_v3.html) in [The `zarrs` Book], or try the [`zarrs_reencode`](https://github.com/zarrs/zarrs_tools/blob/main/docs/zarrs_reencode.md) CLI tool.
//!
//! #### Array Support
//!
//! <details><summary>Data Types</summary>
//!
#![doc = include_str!("../doc/status/data_types.md")]
//! </details>
//!
//! <details><summary>Codecs</summary>
//!
#![doc = include_str!("../doc/status/codecs.md")]
//! </details>
//!
//! <details><summary>Chunk Grids</summary>
//!
#![doc = include_str!("../doc/status/chunk_grids.md")]
//! </details>
//!
//! <details><summary>Chunk Key Encodings</summary>
//!
#![doc = include_str!("../doc/status/chunk_key_encodings.md")]
//! </details>
//!
//! <details><summary>Storage Transformers</summary>
//!
#![doc = include_str!("../doc/status/storage_transformers.md")]
//! </details>
//!
//! #### Storage Support
//!
//! `zarrs` supports a huge range of stores (including custom stores) via the [`zarrs_storage`] API.
//!
//! <details><summary>Stores</summary>
//!
#![doc = include_str!("../doc/status/stores.md")]
//! </details>
//!
//! [`opendal`]: https://docs.rs/opendal/latest/opendal/
//! [`object_store`]: https://docs.rs/object_store/latest/object_store/
//! [`object_store`]: https://docs.rs/object_store/latest/object_store/
//! [`zarrs_icechunk`]: https://docs.rs/zarrs_icechunk/latest/zarrs_icechunk/
//! [`zarrs_object_store`]: https://docs.rs/zarrs_object_store/latest/zarrs_object_store/
//! [`zarrs_opendal`]: https://docs.rs/zarrs_opendal/latest/zarrs_opendal/
//!
//!
//! The [`opendal`] and [`object_store`] crates are popular Rust storage backends that are fully supported via [`zarrs_opendal`] and [`zarrs_object_store`].
//! These backends provide more feature complete HTTP stores than [`zarrs_http`].
//!
//! [`zarrs_icechunk`] implements the [Icechunk](https://icechunk.io) transactional storage engine, a storage specification for Zarr that supports [`object_store`] stores.
//!
//! The [`AsyncToSyncStorageAdapter`](crate::storage::storage_adapter::async_to_sync::AsyncToSyncStorageAdapter) enables some async stores to be used in a sync context.
//!
//! ## Examples
//! ### Create and Read a Zarr Hierarchy
#![cfg_attr(feature = "ndarray", doc = "```rust")]
#![cfg_attr(not(feature = "ndarray"), doc = "```rust,ignore")]
//! # use std::{path::PathBuf, sync::Arc};
//! use zarrs::group::GroupBuilder;
//! use zarrs::array::{ArrayBuilder, DataType, FillValue, ZARR_NAN_F32};
//! # #[cfg(feature = "gzip")]
//! use zarrs::array::codec::GzipCodec; // requires gzip feature
//! use zarrs::array_subset::ArraySubset;
//! use zarrs::storage::ReadableWritableListableStorage;
//! use zarrs::filesystem::FilesystemStore; // requires filesystem feature
//!
//! // Create a filesystem store
//! let store_path: PathBuf = "/path/to/hierarchy.zarr".into();
//! # let store_path: PathBuf = "tests/data/array_write_read.zarr".into();
//! let store: ReadableWritableListableStorage =
//!     Arc::new(FilesystemStore::new(&store_path)?);
//! # let store = Arc::new(zarrs::storage::store::MemoryStore::new());
//!
//! // Write the root group metadata
//! GroupBuilder::new()
//!     .build(store.clone(), "/")?
//!     // .attributes(...)
//!     .store_metadata()?;
//!
//! // Create a new V3 array using the array builder
//! let array = ArrayBuilder::new(
//!     vec![3, 4], // array shape
//!     DataType::Float32,
//!     vec![2, 2].try_into()?, // regular chunk shape (non-zero elements)
//!     FillValue::from(ZARR_NAN_F32),
//! )
//! .bytes_to_bytes_codecs(vec![
//! #     #[cfg(feature = "gzip")]
//!     Arc::new(GzipCodec::new(5)?),
//! ])
//! .dimension_names(["y", "x"].into())
//! .attributes(serde_json::json!({"Zarr V3": "is great"}).as_object().unwrap().clone())
//! .build(store.clone(), "/array")?; // /path/to/hierarchy.zarr/array
//!
//! // Store the array metadata
//! array.store_metadata()?;
//! println!("{}", serde_json::to_string_pretty(array.metadata())?);
//! // {
//! //     "zarr_format": 3,
//! //     "node_type": "array",
//! //     ...
//! // }
//!
//! // Perform some operations on the chunks
//! array.store_chunk_elements::<f32>(
//!     &[0, 1], // chunk index
//!     &[0.2, 0.3, 1.2, 1.3]
//! )?;
//! array.store_array_subset_ndarray::<f32, _>(
//!     &[1, 1], // array index (start of subset)
//!     ndarray::array![[-1.1, -1.2], [-2.1, -2.2]]
//! )?;
//! array.erase_chunk(&[1, 1])?;
//!
//! // Retrieve all array elements as an ndarray
//! let array_ndarray = array.retrieve_array_subset_ndarray::<f32>(&array.subset_all())?;
//! println!("{array_ndarray:4}");
//! // [[ NaN,  NaN,  0.2,  0.3],
//! //  [ NaN, -1.1, -1.2,  1.3],
//! //  [ NaN, -2.1,  NaN,  NaN]]
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### More examples
//! Various examples can be found in the [examples](https://github.com/zarrs/zarrs/blob/main/zarrs/examples) directory that demonstrate:
//! - creating and manipulating zarr hierarchies with various stores (sync and async), codecs, etc,
//! - converting between Zarr V2 and V3, and
//! - creating custom data types.
//!
//! Examples can be run with `cargo run --example <EXAMPLE_NAME>`.
//!  - Some examples require non-default features, which can be enabled with `--all-features` or `--features <FEATURES>`.
//!  - Some examples support a `-- --usage-log` argument to print storage API calls during execution.
//!
//! ## Crate Features
//! #### Default
//!  - `filesystem`: Re-export [`zarrs_filesystem`] as [`zarrs::filesystem`](crate::filesystem`).
//!  - `ndarray`: [`ndarray`] utility functions for [`Array`](crate::array::Array).
//!  - Codecs: `blosc`, `crc32c`, `gzip`, `sharding`, `transpose`, `zstd`.
//!
//! #### Non-Default
//!  - `async`: an **experimental** asynchronous API for [`stores`](storage), [`Array`](crate::array::Array), and [`Group`](group::Group).
//!    - The async API is runtime-agnostic. This has some limitations that are detailed in the [`Array`](crate::array::Array) docs.
//!    - The async API is not as performant as the sync API.
//!  - Codecs: `bitround`, `bz2`, `fletcher32`, `gdeflate`, `pcodec`, `zfp`, `zlib`.
//!  - `dlpack`: adds convenience methods for [`DLPack`](https://arrow.apache.org/docs/python/dlpack.html) tensor interop to [`Array`](crate::array::Array).
//!  - Additional [`Element`](crate::array::Element)/[`ElementOwned`](crate::array::ElementOwned) implementations:
//!    - `float8`: add support for [`float8`] subfloat data types.
//!    - `jiff`: add support for [`jiff`] time data types.
//!    - `chrono`: add support for [`chrono`] time data types.
//!
//! ## `zarrs` Ecosystem
#![doc = include_str!("../doc/ecosystem.md")]
//!
//! ## Licence
//! `zarrs` is licensed under either of
//!  - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//!  - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
//!
//! [The `zarrs` Book]: https://book.zarrs.dev
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![warn(clippy::wildcard_enum_match_arm)]

pub mod array;
pub mod array_subset;
pub mod config;
pub mod group;
pub mod node;
pub mod version;

pub use zarrs_metadata as metadata;
pub use zarrs_metadata_ext as metadata_ext;
pub use zarrs_plugin as plugin;
pub use zarrs_registry as registry;
pub use zarrs_storage as storage;

#[cfg(feature = "filesystem")]
pub use zarrs_filesystem as filesystem;

// Re-export byte_range for compat with <17.0.0
pub use storage::byte_range;

/// Get a mutable slice of the spare capacity in a vector.
fn vec_spare_capacity_to_mut_slice<T>(vec: &mut Vec<T>) -> &mut [T] {
    let spare_capacity = vec.spare_capacity_mut();
    // SAFETY: `spare_capacity` is valid for both reads and writes for len * size_of::<T>() many bytes, and it is properly aligned
    unsafe {
        std::slice::from_raw_parts_mut(
            spare_capacity.as_mut_ptr().cast::<T>(),
            spare_capacity.len(),
        )
    }
}
