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
//! - experimental codecs intended for future registration, and
//! - user-defined custom extensions and stores.
//!
//! A changelog can be found [here](https://github.com/zarrs/zarrs/blob/main/CHANGELOG.md).
//! Correctness issues with past versions are [detailed here](https://github.com/zarrs/zarrs/blob/main/doc/correctness_issues.md).
//!
//! Developed at the [Department of Materials Physics](https://physics.anu.edu.au/research/mp/), Australian National University, Canberra, Australia.
//!
//! ## Getting Started
//! - Review the [Zarr version support](#zarr-version-support), [array extension support](#array-extension-support) (codecs, data types, etc.), [storage support](#storage-support), and the [`zarrs` ecosystem](#zarrs-ecosystem).
//! - View the [the examples](#examples) below.
//! - Read the [documentation](https://docs.rs/zarrs/latest/zarrs/) and [The `zarrs` Book].
//!
//! ### Zarr Version Support
//!
//! `zarrs` has first-class Zarr V3 support and additionally supports a *compatible subset* of Zarr V2 data that:
//! - can be converted to V3 with only a metadata change, and
//! - uses array metadata that is recognised and supported for encoding/decoding.
//!
//! `zarrs` supports forward conversion from Zarr V2 to V3. See [Converting Zarr V2 to V3](https://book.zarrs.dev/v2_to_v3.html) in [The `zarrs` Book], or try the [`zarrs_reencode`](https://github.com/zarrs/zarrs_tools/blob/main/docs/zarrs_reencode.md) CLI tool.
//!
//! ### Array Extension Support
//!
//! Extensions are grouped into three categories:
//! - *Core*: defined in the Zarr V3 specification and are fully supported.
//! - *Registered*: specified at <https://github.com/zarr-developers/zarr-extensions/>
//!   - Registered extensions listed in the below tables are fully supported unless otherwise indicated.
//! - *Experimental*: indicated by 🚧 in the tables below and **recommended for evaluation only**.
//!   - Experimental extensions are either pending registration or have no formal specification outside of the `zarrs` docs.
//!   - Experimental extensions may be unrecognised or incompatible with other Zarr implementations.
//!   - Experimental extensions may change in future releases without maintaining backwards compatibility.
//! - *Deprecated*: indicated by ~~strikethrough~~ in the tables below
//!   - Deprecated aliases will not be removed, but are not recommended for use in new arrays.
//!   - Deprecated extensions may be removed in future releases.
//!
//! Extension names and aliases are configurable with [`Config::codec_aliases_v3_mut`](config::Config::codec_aliases_v3_mut) and similar methods for data types and Zarr V2.
//! `zarrs` will persist extension names if opening an existing array of creating an array from metadata.
//!
//! #### Data Types
//!
#![doc = include_str!("../doc/status/data_types.md")]
//!
//! #### Codecs
//!
#![doc = include_str!("../doc/status/codecs.md")]
//!
//! #### Chunk Grids
//!
#![doc = include_str!("../doc/status/chunk_grids.md")]
//!
//! #### Chunk Key Encodings
//!
#![doc = include_str!("../doc/status/chunk_key_encodings.md")]
//!
//! #### Storage Transformers
//!
#![doc = include_str!("../doc/status/storage_transformers.md")]
//!
//! ### Storage Support
//!
//! `zarrs` supports a huge range of stores (including custom stores) via the [`zarrs_storage`] API.
//!
#![doc = include_str!("../doc/status/stores.md")]
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
//! ## Logging
//! `zarrs` logs information and warnings using the [`log`] crate.
//! A logging implementation must be enabled to capture logs.
//! See the [`log`] crate documentation for more details.
//!
//! ## Examples
//! ### Create and Read a Zarr Hierarchy
#![cfg_attr(feature = "ndarray", doc = "```rust")]
#![cfg_attr(not(feature = "ndarray"), doc = "```rust,ignore")]
//! # use std::{path::PathBuf, sync::Arc};
//!
//! // Create a filesystem store
//! let store_path: PathBuf = "/path/to/hierarchy.zarr".into();
//! # let store_path: PathBuf = "tests/data/array_write_read.zarr".into();
//! let store: zarrs::storage::ReadableWritableListableStorage = Arc::new(
//!     // zarrs::filesystem requires the filesystem feature
//!     zarrs::filesystem::FilesystemStore::new(&store_path)?
//! );
//! # let store = Arc::new(zarrs::storage::store::MemoryStore::new());
//!
//! // Write the root group metadata
//! zarrs::group::GroupBuilder::new()
//!     .build(store.clone(), "/")?
//!     // .attributes(...)
//!     .store_metadata()?;
//!
//! // Create a new sharded V3 array using the array builder
//! let array = zarrs::array::ArrayBuilder::new(
//!     vec![3, 4], // array shape
//!     vec![2, 2], // regular chunk (shard) shape
//!     zarrs::array::DataType::Float32,
//!     0.0f32, // fill value
//! )
//! .array_to_bytes_codec(Arc::new(
//!     // The sharding codec requires the sharding feature
//!     zarrs::array::codec::ShardingCodecBuilder::new(
//!         [2, 1].try_into()? // inner chunk shape
//!     )
//!     .bytes_to_bytes_codecs(vec![
//!         // GzipCodec requires the gzip feature
//! #       #[cfg(feature = "gzip")]
//!         Arc::new(zarrs::array::codec::GzipCodec::new(5)?),
//!     ])
//!     .build()
//! ))
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
//! // Perform some write operations on the chunks
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
//! let array_all = array.retrieve_array_subset_ndarray::<f32>(&array.subset_all())?;
//! println!("{array_all:4}");
//! // [[ NaN,  NaN,  0.2,  0.3],
//! //  [ NaN, -1.1, -1.2,  1.3],
//! //  [ NaN, -2.1,  NaN,  NaN]]
//!
//! // Retrieve a chunk directly
//! let array_chunk = array.retrieve_chunk_ndarray::<f32>(
//!     &[0, 1], // chunk index
//! )?;
//! println!("{array_chunk:4}");
//! // [[  0.2,  0.3],
//! //  [ -1.2,  1.3]]
//!
//! // Retrieve an inner chunk
//! use zarrs::array::ArrayShardedReadableExt;
//! let shard_index_cache = zarrs::array::ArrayShardedReadableExtCache::new(&array);
//! let array_inner_chunk = array.retrieve_inner_chunk_ndarray_opt::<f32>(
//!     &shard_index_cache,
//!     &[0, 3], // inner chunk index
//!     &zarrs::array::codec::CodecOptions::default(),
//! )?;
//! println!("{array_inner_chunk:4}");
//! // [[ 0.3],
//! //  [ 1.3]]
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ### Additional Examples
//! Various examples can be found in the [`examples/`](https://github.com/zarrs/zarrs/blob/main/zarrs/examples) directory of the `zarrs` repository that demonstrate:
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
//!  - Codecs: `adler32`, `bitround`, `bz2`, `fletcher32`, `gdeflate`, `pcodec`, `zfp`, `zlib`.
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
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(html_logo_url = "https://zarrs.dev/zarrs-logo-400x400.png")]
#![warn(clippy::wildcard_enum_match_arm)]

pub mod array;
pub mod array_subset;
pub mod config;
pub mod group;
pub mod hierarchy;
pub mod indexer;
pub mod node;
pub mod version;

#[cfg(feature = "filesystem")]
pub use zarrs_filesystem as filesystem;
pub use zarrs_metadata as metadata;
pub use zarrs_metadata_ext as metadata_ext;
pub use zarrs_plugin as plugin;
pub use zarrs_registry as registry;
pub use zarrs_storage as storage;

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

/// Log a warning that an extension is experimental and may be incompatible with other Zarr V3 implementations.
///
/// # Arguments
/// * `name` - The name of the extension (e.g., `vlen`, `regular_bounded`)
/// * `extension_type` - The type of extension (e.g., `codec`, `chunk grid`, `chunk key encoding`)
pub(crate) fn warn_experimental_extension(name: &str, extension_type: &str) {
    log::warn!(
        "The `{name}` {extension_type} is experimental and may be incompatible with other Zarr V3 implementations.",
    );
}

/// Log a warning that a deprecated extension name is being used.
///
/// # Arguments
/// * `deprecated_name` - The deprecated name being used (e.g., `binary`)
/// * `extension_type` - The type of extension (e.g., `codec`, `chunk grid`, `chunk key encoding`)
/// * `current_name` - The current/preferred name (e.g., `bytes`)
pub(crate) fn warn_deprecated_extension(
    deprecated_name: &str,
    extension_type: &str,
    current_name: Option<&str>,
) {
    if let Some(current_name) = current_name {
        log::warn!(
            "The `{deprecated_name}` {extension_type} alias is deprecated, use `{current_name}` instead.",
        );
    } else {
        log::warn!("The `{deprecated_name}` {extension_type} is deprecated.",);
    }
}

#[cfg(not(target_arch = "wasm32"))]
use rayon_iter_concurrent_limit::iter_concurrent_limit;

#[cfg(target_arch = "wasm32")]
/// A serial equivalent of [`rayon_iter_concurrent_limit::iter_concurrent_limit`] for WASM compatibility.
#[macro_export]
macro_rules! iter_concurrent_limit {
    ( $concurrent_limit:expr, $iterator:expr, $fn:tt, $op:expr ) => {{
        let _concurrent_limit = $concurrent_limit; // fixes unused lint
        $iterator.into_iter().$fn($op)
    }};
}
