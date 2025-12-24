//! `regular` chunk grid metadata.

use std::num::NonZeroU64;

use crate::ChunkShape;
use derive_more::Display;
use serde::{Deserialize, Serialize};
use zarrs_metadata::ConfigurationSerialize;

/// Configuration parameters for a `regular` chunk grid.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display(
    "regular chunk grid {}",
    serde_json::to_string(self).unwrap_or_default()
)]
pub struct RegularChunkGridConfiguration {
    /// The chunk shape.
    pub chunk_shape: ChunkShape,
}

impl ConfigurationSerialize for RegularChunkGridConfiguration {}

macro_rules! from_chunkgrid_regular_configuration {
    ( $t:ty ) => {
        impl From<$t> for RegularChunkGridConfiguration {
            fn from(value: $t) -> Self {
                Self {
                    chunk_shape: value.into(),
                }
            }
        }
    };
    ( $t:ty, $g:ident ) => {
        impl<const $g: usize> From<$t> for RegularChunkGridConfiguration {
            fn from(value: $t) -> Self {
                Self {
                    chunk_shape: value.into(),
                }
            }
        }
    };
}

macro_rules! try_from_chunkgrid_regular_configuration {
    ( $t:ty ) => {
        impl TryFrom<$t> for RegularChunkGridConfiguration {
            type Error = std::num::TryFromIntError;
            fn try_from(value: $t) -> Result<Self, Self::Error> {
                value.try_into()
            }
        }
    };
    ( $t:ty, $g:ident ) => {
        impl<const $g: usize> TryFrom<$t> for RegularChunkGridConfiguration {
            type Error = std::num::TryFromIntError;
            fn try_from(value: $t) -> Result<Self, Self::Error> {
                value.try_into()
            }
        }
    };
}

from_chunkgrid_regular_configuration!(Vec<NonZeroU64>);
from_chunkgrid_regular_configuration!(&[NonZeroU64]);
from_chunkgrid_regular_configuration!([NonZeroU64; N], N);
from_chunkgrid_regular_configuration!(&[NonZeroU64; N], N);
try_from_chunkgrid_regular_configuration!(Vec<u64>);
try_from_chunkgrid_regular_configuration!(&[u64]);
try_from_chunkgrid_regular_configuration!([u64; N], N);
try_from_chunkgrid_regular_configuration!(&[u64; N], N);
