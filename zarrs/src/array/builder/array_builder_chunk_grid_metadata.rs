use std::num::NonZeroU64;

use derive_more::From;
use serde::Serialize;

use crate::array::{ArrayCreateError, ArrayShape, ChunkShape};
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata_ext::chunk_grid::regular::RegularChunkGridConfiguration;

/// An input that can be mapped to a chunk grid.
#[derive(Debug, Clone, From)]
pub struct ArrayBuilderChunkGridMetadata(ArrayBuilderChunkGridMetadataImpl);

#[derive(Debug, Clone, From)]
enum ArrayBuilderChunkGridMetadataImpl {
    Metadata(MetadataV3),
    MetadataString(String),
    ArrayShape(ArrayShape),
    ChunkShape(ChunkShape),
}

impl ArrayBuilderChunkGridMetadata {
    pub(crate) fn to_metadata(&self) -> Result<MetadataV3, ArrayCreateError> {
        match &self.0 {
            ArrayBuilderChunkGridMetadataImpl::Metadata(metadata) => Ok(metadata.clone()),
            ArrayBuilderChunkGridMetadataImpl::MetadataString(metadata) => {
                let metadata = MetadataV3::try_from(metadata.as_str()).map_err(|_| {
                    ArrayCreateError::ChunkGridCreateError(zarrs_plugin::PluginCreateError::from(
                        "chunk grid string cannot be parsed as metadata",
                    ))
                })?;
                Ok(metadata)
            }
            ArrayBuilderChunkGridMetadataImpl::ArrayShape(chunk_shape) => {
                #[derive(Serialize)]
                struct RegularChunkGridConfiguration {
                    chunk_shape: ArrayShape,
                }
                let metadata = MetadataV3::new_with_serializable_configuration(
                    "regular".to_string(),
                    &RegularChunkGridConfiguration {
                        chunk_shape: chunk_shape.clone(),
                    },
                )
                .expect("RegularChunkGridConfigurationInvalid is serialisable");
                Ok(metadata)
            }
            ArrayBuilderChunkGridMetadataImpl::ChunkShape(chunk_shape) => {
                let metadata = MetadataV3::new_with_serializable_configuration(
                    "regular".to_string(),
                    &RegularChunkGridConfiguration {
                        chunk_shape: chunk_shape.clone(),
                    },
                )
                .expect("RegularChunkGridConfigurationInvalid is serialisable");
                Ok(metadata)
            }
        }
    }
}

impl From<MetadataV3> for ArrayBuilderChunkGridMetadata {
    fn from(value: MetadataV3) -> Self {
        Self(ArrayBuilderChunkGridMetadataImpl::Metadata(value))
    }
}

impl From<String> for ArrayBuilderChunkGridMetadata {
    fn from(value: String) -> Self {
        Self(ArrayBuilderChunkGridMetadataImpl::MetadataString(value))
    }
}

impl From<&str> for ArrayBuilderChunkGridMetadata {
    fn from(value: &str) -> Self {
        Self(ArrayBuilderChunkGridMetadataImpl::MetadataString(
            value.to_string(),
        ))
    }
}

impl<const N: usize> From<[u64; N]> for ArrayBuilderChunkGridMetadata {
    fn from(chunk_shape: [u64; N]) -> Self {
        Self(ArrayBuilderChunkGridMetadataImpl::ArrayShape(
            chunk_shape.to_vec(),
        ))
    }
}

impl<const N: usize> From<&[u64; N]> for ArrayBuilderChunkGridMetadata {
    fn from(chunk_shape: &[u64; N]) -> Self {
        Self(ArrayBuilderChunkGridMetadataImpl::ArrayShape(
            chunk_shape.to_vec(),
        ))
    }
}

impl From<&[u64]> for ArrayBuilderChunkGridMetadata {
    fn from(chunk_shape: &[u64]) -> Self {
        Self(ArrayBuilderChunkGridMetadataImpl::ArrayShape(
            chunk_shape.to_vec(),
        ))
    }
}

impl<const N: usize> From<[NonZeroU64; N]> for ArrayBuilderChunkGridMetadata {
    fn from(chunk_shape: [NonZeroU64; N]) -> Self {
        Self(ArrayBuilderChunkGridMetadataImpl::ChunkShape(
            chunk_shape.to_vec(),
        ))
    }
}

impl<const N: usize> From<&[NonZeroU64; N]> for ArrayBuilderChunkGridMetadata {
    fn from(chunk_shape: &[NonZeroU64; N]) -> Self {
        Self(ArrayBuilderChunkGridMetadataImpl::ChunkShape(
            chunk_shape.to_vec(),
        ))
    }
}

impl From<&[NonZeroU64]> for ArrayBuilderChunkGridMetadata {
    fn from(chunk_shape: &[NonZeroU64]) -> Self {
        Self(ArrayBuilderChunkGridMetadataImpl::ChunkShape(
            chunk_shape.to_vec(),
        ))
    }
}

impl From<Vec<NonZeroU64>> for ArrayBuilderChunkGridMetadata {
    fn from(chunk_shape: Vec<NonZeroU64>) -> Self {
        Self(ArrayBuilderChunkGridMetadataImpl::ChunkShape(chunk_shape))
    }
}

impl From<Vec<u64>> for ArrayBuilderChunkGridMetadata {
    fn from(chunk_shape: Vec<u64>) -> Self {
        Self(ArrayBuilderChunkGridMetadataImpl::ArrayShape(chunk_shape))
    }
}
