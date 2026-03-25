//! `rectangular` chunk grid metadata.

use std::num::NonZeroU64;

use derive_more::Display;
use serde::{Deserialize, Serialize};
use zarrs_metadata::{ChunkShape, ConfigurationSerialize};

/// Configuration parameters for a `rectangular` chunk grid.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct RectangularChunkGridConfiguration {
    /// The chunk shape.
    pub chunk_shape: Vec<RectangularChunkGridDimensionConfiguration>,
}

impl ConfigurationSerialize for RectangularChunkGridConfiguration {}

/// A chunk element in the `chunk_shape` field of `rectangular` chunk grid netadata.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, derive_more::From)]
#[serde(untagged)]
pub enum RectangularChunkGridDimensionConfiguration {
    /// A fixed chunk size.
    Fixed(NonZeroU64),
    /// A varying chunk size.
    Varying(ChunkShape),
}
