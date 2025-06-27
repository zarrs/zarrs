use std::{num::NonZeroU64, sync::Arc};

use derive_more::From;
use serde::Serialize;
use zarrs_metadata::{v3::MetadataV3, ArrayShape, ChunkShape};
use zarrs_plugin::PluginMetadataInvalidError;

use crate::array::{
    chunk_grid::{ChunkGridTraits, RegularChunkGrid},
    ArrayCreateError, ChunkGrid,
};

/// An input that can be mapped to a chunk grid.
#[derive(Debug, From)]
pub struct ArrayBuilderChunkGrid(ArrayBuilderChunkGridImpl);

#[derive(Debug, From)]
enum ArrayBuilderChunkGridImpl {
    ChunkGrid(ChunkGrid),
    Metadata(MetadataV3),
    MetadataString(String),
    ArrayShape(ArrayShape),
    ChunkShape(ChunkShape),
}

impl ArrayBuilderChunkGrid {
    pub(crate) fn to_chunk_grid(&self, shape: &ArrayShape) -> Result<ChunkGrid, ArrayCreateError> {
        match &self.0 {
            ArrayBuilderChunkGridImpl::ChunkGrid(chunk_grid) => Ok(chunk_grid.clone()),
            ArrayBuilderChunkGridImpl::Metadata(metadata) => {
                let chunk_grid = ChunkGrid::from_metadata(metadata)
                    .map_err(ArrayCreateError::ChunkGridCreateError)?;
                debug_assert_eq!(chunk_grid.dimensionality(), shape.len());
                Ok(chunk_grid)
            }
            ArrayBuilderChunkGridImpl::MetadataString(metadata) => {
                let metadata = MetadataV3::try_from(metadata.as_str()).map_err(|_| {
                    ArrayCreateError::ChunkGridCreateError(zarrs_plugin::PluginCreateError::from(
                        "chunk grid string cannot be parsed as metadata",
                    ))
                })?;
                let chunk_grid = ChunkGrid::from_metadata(&metadata)
                    .map_err(ArrayCreateError::ChunkGridCreateError)?;
                debug_assert_eq!(chunk_grid.dimensionality(), shape.len());
                Ok(chunk_grid)
            }
            ArrayBuilderChunkGridImpl::ArrayShape(chunk_shape) => {
                let chunk_shape: ChunkShape = chunk_shape.clone().try_into().map_err(|_| {
                    #[derive(Serialize)]
                    struct RegularChunkGridConfigurationInvalid {
                        chunk_shape: ArrayShape,
                    }
                    let metadata = MetadataV3::new_with_serializable_configuration(
                        "regular".to_string(),
                        &RegularChunkGridConfigurationInvalid {
                            chunk_shape: chunk_shape.clone(),
                        },
                    )
                    .expect("RegularChunkGridConfigurationInvalid is serialisable");
                    ArrayCreateError::ChunkGridCreateError(
                        PluginMetadataInvalidError::new(
                            "regular",
                            "chunk_grid",
                            metadata.to_string(),
                        )
                        .into(),
                    )
                })?;
                Ok(ChunkGrid::new(RegularChunkGrid::new(chunk_shape)))
            }
            ArrayBuilderChunkGridImpl::ChunkShape(chunk_shape) => {
                Ok(ChunkGrid::new(RegularChunkGrid::new(chunk_shape.clone())))
            }
        }
    }
}

impl From<ChunkGrid> for ArrayBuilderChunkGrid {
    fn from(value: ChunkGrid) -> Self {
        Self(ArrayBuilderChunkGridImpl::ChunkGrid(value))
    }
}

impl<T: ChunkGridTraits + 'static> From<T> for ArrayBuilderChunkGrid {
    fn from(value: T) -> Self {
        Self(ArrayBuilderChunkGridImpl::ChunkGrid(ChunkGrid::new(value)))
    }
}

impl From<Arc<dyn ChunkGridTraits>> for ArrayBuilderChunkGrid {
    fn from(value: Arc<dyn ChunkGridTraits>) -> Self {
        Self(ArrayBuilderChunkGridImpl::ChunkGrid(value.into()))
    }
}

impl From<MetadataV3> for ArrayBuilderChunkGrid {
    fn from(value: MetadataV3) -> Self {
        Self(ArrayBuilderChunkGridImpl::Metadata(value))
    }
}

impl From<String> for ArrayBuilderChunkGrid {
    fn from(value: String) -> Self {
        Self(ArrayBuilderChunkGridImpl::MetadataString(value))
    }
}

impl From<&str> for ArrayBuilderChunkGrid {
    fn from(value: &str) -> Self {
        Self(ArrayBuilderChunkGridImpl::MetadataString(value.to_string()))
    }
}

impl<const N: usize> From<[u64; N]> for ArrayBuilderChunkGrid {
    fn from(chunk_shape: [u64; N]) -> Self {
        Self(ArrayBuilderChunkGridImpl::ArrayShape(chunk_shape.to_vec()))
    }
}

impl<const N: usize> From<&[u64; N]> for ArrayBuilderChunkGrid {
    fn from(chunk_shape: &[u64; N]) -> Self {
        Self(ArrayBuilderChunkGridImpl::ArrayShape(chunk_shape.to_vec()))
    }
}

impl From<&[u64]> for ArrayBuilderChunkGrid {
    fn from(chunk_shape: &[u64]) -> Self {
        Self(ArrayBuilderChunkGridImpl::ArrayShape(chunk_shape.to_vec()))
    }
}

impl<const N: usize> From<[NonZeroU64; N]> for ArrayBuilderChunkGrid {
    fn from(chunk_shape: [NonZeroU64; N]) -> Self {
        Self(ArrayBuilderChunkGridImpl::ChunkShape(
            chunk_shape.to_vec().into(),
        ))
    }
}

impl<const N: usize> From<&[NonZeroU64; N]> for ArrayBuilderChunkGrid {
    fn from(chunk_shape: &[NonZeroU64; N]) -> Self {
        Self(ArrayBuilderChunkGridImpl::ChunkShape(
            chunk_shape.to_vec().into(),
        ))
    }
}

impl From<&[NonZeroU64]> for ArrayBuilderChunkGrid {
    fn from(chunk_shape: &[NonZeroU64]) -> Self {
        Self(ArrayBuilderChunkGridImpl::ChunkShape(
            chunk_shape.to_vec().into(),
        ))
    }
}

impl From<Vec<NonZeroU64>> for ArrayBuilderChunkGrid {
    fn from(chunk_shape: Vec<NonZeroU64>) -> Self {
        Self(ArrayBuilderChunkGridImpl::ChunkShape(chunk_shape.into()))
    }
}

impl From<Vec<u64>> for ArrayBuilderChunkGrid {
    fn from(chunk_shape: Vec<u64>) -> Self {
        Self(ArrayBuilderChunkGridImpl::ArrayShape(chunk_shape))
    }
}
