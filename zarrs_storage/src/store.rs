//! Zarr stores.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#stores>

mod memory_store;
pub use memory_store::MemoryStore;
