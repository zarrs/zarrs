//! Planned (sans-I/O) encoding and decoding.
//!
//! Codecs describe an operation as a graph of tasks.
//! Each task reads byte ranges of a [`ByteSource`] and then runs a compute step on the bytes read.
//! A driver executes the graph, performing I/O and compute with separate concurrency limits and
//! interleaving them, so that a task computes as soon as its bytes are available.

mod decode;
mod driver;
mod encode;
mod io;
mod memory;
mod prepare;
mod source;
mod subchunk;
mod target;
mod task;

pub use decode::{
    ArrayToArrayDecodePlan, decode_whole, is_whole_chunk, local_subchunk_grids,
    local_subchunk_grids_value, place_decoded, place_fill_value, plan_decode_request_elementwise,
    plan_decode_whole, whole_chunk_hint,
};
#[cfg(not(target_family = "wasm"))]
pub use driver::{IoPool, IoScope, drive_sync, drive_sync_one};
#[cfg(all(feature = "async", not(target_arch = "wasm32")))]
pub use driver::{OFFLOAD_THRESHOLD, drive_async_offload, drive_async_offload_one};
#[cfg(feature = "async")]
pub use driver::{drive_async, drive_async_one};
pub use driver::{drive_blocking_one, drive_memory_one, drive_serial, drive_serial_one};
pub use encode::{
    PartialEncodeState, decode_and_update, encode_and_write, encode_hint, encoded_value,
    partial_encode_request_elementwise, partial_encode_state, partial_encode_value,
    plan_encode_whole, plan_partial_encode_whole,
};
#[cfg(feature = "async")]
pub use io::AsyncIo;
pub use io::{ReadBytes, SyncIo};
pub use memory::{partial_decode_memory, partial_encode_memory};
pub use prepare::{
    DecodedDecoder, MissingDecoder, PrepareDecode, PreparedDecoder, SourceDecoder,
    prepare_decode_whole, prepared,
};
pub(crate) use source::SourceRoot;
pub use source::{ByteSource, RangeDecoder, SourceKey, SourceStep};
pub use subchunk::{
    EncodedSubchunk, SubchunkLocation, encoded_subchunk, plan_retrieve_encoded_subchunk,
    subchunk_location, subchunk_location_value,
};
pub use target::{DecodeRequest, PlanIndexer, SharedOutput, TargetViews, TaskTarget};
pub use task::{
    AnyValue, ComputeCtx, Generator, Join, Outcome, Reads, Task, TaskHint, TaskValue, WriteOp,
};
pub(crate) use task::{ComputeStep, read_memory};
