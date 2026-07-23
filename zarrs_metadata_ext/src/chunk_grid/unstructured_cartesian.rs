//! `zarrs.unstructured_cartesian` chunk grid metadata.

use std::num::NonZeroU64;

use derive_more::Display;
use serde::{Deserialize, Serialize};

use zarrs_metadata::{ChunkShapeNonEmpty, ConfigurationSerialize};

/// Configuration parameters for a `zarrs.unstructured_cartesian` chunk grid.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub enum UnstructuredCartesianChunkGridConfiguration {
    /// Inline chunk definitions.
    Inline {
        /// The chunks in the grid.
        chunks: Vec<UnstructuredCartesianChunk>,
    },
}

impl ConfigurationSerialize for UnstructuredCartesianChunkGridConfiguration {}

/// A chunk in an unstructured cartesian chunk grid.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
pub struct UnstructuredCartesianChunk {
    /// The origin of the chunk in array coordinates.
    pub origin: Vec<u64>,
    /// The shape of the chunk.
    pub shape: ChunkShapeNonEmpty,
}

impl UnstructuredCartesianChunk {
    /// Create a new unstructured cartesian chunk.
    #[must_use]
    pub fn new(origin: Vec<u64>, shape: Vec<NonZeroU64>) -> Self {
        Self { origin, shape }
    }
}
