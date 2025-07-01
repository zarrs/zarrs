use std::sync::Arc;

use crate::array::{chunk_grid::ChunkGridTraits, ChunkGrid};

/// An input that can be mapped to a chunk grid.
#[derive(Debug)]
pub struct ArrayBuilderChunkGrid(ChunkGrid);

impl ArrayBuilderChunkGrid {
    pub(crate) fn as_chunk_grid(&self) -> &ChunkGrid {
        &self.0
    }
}

impl From<ChunkGrid> for ArrayBuilderChunkGrid {
    fn from(value: ChunkGrid) -> Self {
        Self(value)
    }
}

impl<T: ChunkGridTraits + 'static> From<T> for ArrayBuilderChunkGrid {
    fn from(value: T) -> Self {
        Self(ChunkGrid::new(value))
    }
}

impl From<Arc<dyn ChunkGridTraits>> for ArrayBuilderChunkGrid {
    fn from(value: Arc<dyn ChunkGridTraits>) -> Self {
        Self(value.into())
    }
}
