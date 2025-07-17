//! [`opendal`] store support for the [`zarrs`](https://docs.rs/zarrs/latest/zarrs/index.html) crate.
//!
//! ```
//! # use std::sync::Arc;
//! use zarrs_storage::AsyncReadableStorage;
//! use zarrs_opendal::AsyncOpendalStore;
//!
//! let builder = opendal::services::Http::default().endpoint("http://...");
//! let operator = opendal::Operator::new(builder)?.finish();
//! let store: AsyncReadableStorage = Arc::new(AsyncOpendalStore::new(operator));
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Version Compatibility Matrix
//!
#![doc = include_str!("../doc/version_compatibility_matrix.md")]
//!
//! `opendal` is re-exported as a dependency of this crate, so it does not need to be specified as a direct dependency.
//!
//! However, if `opendal` is a direct dependency, it is necessary to ensure that the version used by this crate is compatible.
//! This crate can depend on a range of semver-incompatible versions of `opendal`, and Cargo will not automatically choose a single version of `opendal` that satisfies all dependencies.
//! Use a precise cargo update to ensure compatibility.
//! For example, if this crate resolves to `opendal` 0.50.2 and your code uses 0.49.2:
//! ```shell
//! cargo update --package opendal:0.50.2 --precise 0.49.2
//! ```
//!
//! ## Licence
//! `zarrs_opendal` is licensed under either of
//! - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_opendal/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//! - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_opendal/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.

mod r#async;

pub use r#async::AsyncOpendalStore;

pub use opendal;

use zarrs_storage::StorageError;

/// Map [`opendal::ErrorKind::NotFound`] to None, pass through other errors
fn handle_result_notfound<T>(result: Result<T, opendal::Error>) -> Result<Option<T>, StorageError> {
    match result {
        Ok(result) => Ok(Some(result)),
        Err(err) => {
            if err.kind() == opendal::ErrorKind::NotFound {
                Ok(None)
            } else {
                Err(StorageError::Other(err.to_string()))
            }
        }
    }
}

fn handle_result<T>(result: Result<T, opendal::Error>) -> Result<T, StorageError> {
    result.map_err(|err| StorageError::Other(err.to_string()))
}
