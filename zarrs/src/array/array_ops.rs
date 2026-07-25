use std::sync::Arc;

use super::chunk_cache::ChunkCache;
use super::{
    Array, ArrayBuilder, ArrayCached, ArrayCreateError, ArrayError, ArrayIndices, ArrayMetadata,
    ArrayMetadataOptions, ArrayShape, ArraySubset, ArraySubsetTraits, ChunkGrid, ChunkGridDecoded,
    ChunkGridDecodedRef, ChunkKeyEncoding, ChunkShape, CodecChain, CodecChainBound,
    CodecCreateError, CodecOptions, CodecSpecificOptions, DataType, DimensionName, FillValue,
    FromArrayBytes, IncompatibleDimensionalityError, IntoArrayBytes, NodePath,
    StorageTransformerChain,
};
use crate::config::MetadataEraseVersion;
#[cfg(feature = "async")]
use zarrs_storage::{
    AsyncReadableStorageTraits, AsyncReadableWritableStorageTraits, AsyncWritableStorageTraits,
    MaybeSend,
};
use zarrs_storage::{
    ReadableStorageTraits, ReadableWritableStorageTraits, StorageError, StoreKey,
    WritableStorageTraits,
};

mod array_mut_ops;
mod array_mut_ops_array;
#[allow(clippy::module_inception)]
mod array_ops;
mod array_ops_array;
mod array_ops_array_cached;
mod array_read_ops;
mod array_read_ops_array;
mod array_read_ops_array_cached;
mod array_read_ops_common;
mod array_update_ops;
mod array_update_ops_array;
mod array_update_ops_array_cached;
mod array_write_ops;
mod array_write_ops_array;
mod array_write_ops_array_cached;

pub use array_mut_ops::ArrayMutOps;
pub use array_ops::ArrayOps;
pub use array_read_ops::ArrayReadOps;
#[cfg(feature = "async")]
pub use array_read_ops::AsyncArrayReadOps;
pub use array_update_ops::ArrayUpdateOps;
#[cfg(feature = "async")]
pub use array_update_ops::AsyncArrayUpdateOps;
pub use array_write_ops::ArrayWriteOps;
#[cfg(feature = "async")]
pub use array_write_ops::AsyncArrayWriteOps;
