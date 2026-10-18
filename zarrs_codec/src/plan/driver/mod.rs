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
pub use asynchronous::OFFLOAD_THRESHOLD;
#[cfg(feature = "async")]
pub use asynchronous::{
    drive_async, drive_async_offload, drive_async_offload_one, drive_async_one,
};
#[cfg(not(target_family = "wasm"))]
pub use io_pool::{IoPool, IoScope};
pub use serial::{drive_serial, drive_serial_one};
#[cfg(not(target_family = "wasm"))]
pub use sync::{drive_sync, drive_sync_one};

/// Drive `task` with the backend `io`, blocking until it completes: on Rayon with [`drive_sync_one`], or serially on wasm.
///
/// # Errors
/// Returns a [`CodecError`](crate::CodecError) if a task fails.
pub fn drive_blocking_one<'a>(
    task: super::Task<'a>,
    io: &dyn super::SyncIo,
    resources: &crate::Resources,
) -> Result<super::TaskValue<'a>, crate::CodecError> {
    #[cfg(not(target_family = "wasm"))]
    return drive_sync_one(task, io, resources);
    #[cfg(target_family = "wasm")]
    return drive_serial_one(task, io, resources);
}

/// Drive `task`, whose byte sources are all in memory: on Rayon, or serially on wasm.
///
/// Codecs use this to decode or encode in memory with their planners (e.g. the subchunks of a shard).
///
/// # Errors
/// Returns a [`CodecError`](crate::CodecError) if a task fails or reads a byte source that is not in memory.
pub fn drive_memory_one<'a>(
    task: super::Task<'a>,
    resources: &crate::ComputeResources,
) -> Result<super::TaskValue<'a>, crate::CodecError> {
    drive_blocking_one(task, &NoIo, &memory_resources(resources))
}

/// The resources of a drive whose reads are all in memory.
///
/// Reads are inline, so the drive never starts an I/O pool.
pub(crate) fn memory_resources(resources: &crate::ComputeResources) -> crate::Resources {
    crate::Resources::from(resources.clone()).with_read_dispatch(Some(crate::ReadDispatch::Inline))
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
