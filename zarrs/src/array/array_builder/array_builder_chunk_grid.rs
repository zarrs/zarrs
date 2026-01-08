use crate::array::ChunkGrid;
use crate::array::chunk_grid::ChunkGridTraits;

/// An input that can be mapped to a chunk grid.
#[derive(Debug)]
pub struct ArrayBuilderChunkGrid(ChunkGrid);

impl ArrayBuilderChunkGrid {
    pub(crate) fn as_chunk_grid(&self) -> &ChunkGrid {
        &self.0
    }
}

impl<T: ChunkGridTraits + 'static> From<T> for ArrayBuilderChunkGrid {
    fn from(value: T) -> Self {
        Self(ChunkGrid::new(value))
    }
}
