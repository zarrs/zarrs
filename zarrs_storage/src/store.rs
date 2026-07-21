//! Zarr stores.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#stores>

mod memory_store;
pub use memory_store::MemoryStore;

#[cfg(feature = "async")]
mod async_memory_store;
#[cfg(feature = "async")]
pub use async_memory_store::AsyncMemoryStore;
