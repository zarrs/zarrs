//! Drivers that execute planned tasks.

#[cfg(feature = "async")]
mod asynchronous;
mod coalesce;
mod core;
#[cfg(not(target_family = "wasm"))]
mod io_pool;
mod serial;
#[cfg(not(target_family = "wasm"))]
mod sync;
#[cfg(test)]
mod tests;

#[cfg(all(feature = "async", not(target_arch = "wasm32")))]
pub use asynchronous::{OFFLOAD_THRESHOLD, drive_async_offload, drive_async_offload_one};
#[cfg(feature = "async")]
pub use asynchronous::{drive_async, drive_async_one};
#[cfg(not(target_family = "wasm"))]
pub use io_pool::{IoPool, IoScope};
pub use serial::{drive_serial, drive_serial_one};
#[cfg(not(target_family = "wasm"))]
pub use sync::{drive_sync, drive_sync_one};

/// Drive `task`, whose byte sources are all in memory: on Rayon, or serially on wasm.
///
/// Codecs use this to decode or encode in memory with their planners (e.g. the subchunks of a shard).
///
/// # Errors
/// Returns a [`CodecError`](crate::CodecError) if a task fails or reads a byte source that is not in memory.
pub fn drive_memory_one<'a>(
    task: super::Task<'a>,
    resources: &crate::Resources,
) -> Result<super::TaskValue<'a>, crate::CodecError> {
    #[cfg(not(target_family = "wasm"))]
    return drive_sync_one(task, &NoIo, resources);
    #[cfg(target_family = "wasm")]
    return drive_serial_one(task, &NoIo, resources);
}

/// The I/O backend of drives of tasks whose byte sources are all in memory.
struct NoIo;

impl super::SyncIo for NoIo {
    fn read(
        &self,
        _key: super::SourceKey,
        _ranges: &[zarrs_storage::byte_range::ByteRange],
    ) -> Result<Option<super::ReadBytes>, zarrs_storage::StorageError> {
        Err(zarrs_storage::StorageError::Other(
            "a task of an in-memory drive read a byte source that is not in memory".to_string(),
        ))
    }
}
