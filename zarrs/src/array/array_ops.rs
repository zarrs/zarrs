use std::sync::Arc;

use super::{
    Array, ArrayBuilder, ArrayCreateError, ArrayError, ArrayIndices, ArrayMetadata,
    ArrayMetadataOptions, ArrayShape, ArraySubset, ArraySubsetTraits, ChunkGrid, ChunkKeyEncoding,
    ChunkShape, CodecChain, CodecOptions, CodecSpecificOptions, DataType, DimensionName, FillValue,
    FromArrayBytes, IncompatibleDimensionalityError, IntoArrayBytes, NodePath,
    StorageTransformerChain,
};
use crate::config::MetadataEraseVersion;
use zarrs_storage::{
    ReadableStorageTraits, ReadableWritableStorageTraits, StorageError, StoreKey,
    WritableStorageTraits,
};

mod array_mut_ops;
#[allow(clippy::module_inception)]
mod array_ops;
mod array_read_ops;
mod array_update_ops;
mod array_write_ops;

pub use array_mut_ops::ArrayMutOps;
pub use array_ops::ArrayOps;
pub use array_read_ops::ArrayReadOps;
pub use array_update_ops::ArrayUpdateOps;
pub use array_write_ops::ArrayWriteOps;
