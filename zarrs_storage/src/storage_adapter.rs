//! Storage adapters.
//!
//! Storage adapters can be layered on stores.

pub mod atomic_write;

#[cfg(feature = "async")]
pub mod async_to_sync;
#[cfg(feature = "async")]
pub mod sync_to_async;

pub mod performance_metrics;
pub mod usage_log;
