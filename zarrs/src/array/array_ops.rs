use std::sync::Arc;

use super::{
    Array, ArrayBuilder, ArrayCreateError, ArrayError, ArrayIndices, ArrayMetadata,
    ArrayMetadataOptions, ArrayShape, ArraySubset, ArraySubsetTraits, ChunkGrid, ChunkKeyEncoding,
    ChunkShape, CodecChain, CodecOptions, CodecSpecificOptions, DataType, DimensionName, FillValue,
    IncompatibleDimensionalityError, NodePath, StorageTransformerChain,
};
use zarrs_storage::StoreKey;

mod array_mut_ops;
#[allow(clippy::module_inception)]
mod array_ops;

pub use array_mut_ops::ArrayMutOps;
pub use array_ops::ArrayOps;
