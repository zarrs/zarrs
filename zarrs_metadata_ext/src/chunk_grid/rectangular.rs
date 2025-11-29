//! `rectangular` chunk grid metadata.

use std::num::NonZeroU64;

use derive_more::{Display, From};
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
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, From)]
#[serde(untagged)]
pub enum RectangularChunkGridDimensionConfiguration {
    /// A fixed chunk size.
    Fixed(NonZeroU64),
    /// A varying chunk size.
    Varying(ChunkShape),
}

impl TryFrom<u64> for RectangularChunkGridDimensionConfiguration {
    type Error = std::num::TryFromIntError;
    fn try_from(value: u64) -> Result<Self, Self::Error> {
        let value = NonZeroU64::try_from(value)?;
        Ok(Self::Fixed(value))
    }
}

macro_rules! from_chunkgrid_rectangular {
    ( $t:ty ) => {
        impl From<$t> for RectangularChunkGridDimensionConfiguration {
            fn from(value: $t) -> Self {
                Self::Varying(value.to_vec().into())
            }
        }
    };
    ( $t:ty, $g:ident ) => {
        impl<const $g: usize> From<$t> for RectangularChunkGridDimensionConfiguration {
            fn from(value: $t) -> Self {
                Self::Varying(value.to_vec().into())
            }
        }
    };
}

macro_rules! try_from_chunkgrid_rectangular_configuration {
    ( $t:ty ) => {
        impl TryFrom<$t> for RectangularChunkGridDimensionConfiguration {
            type Error = std::num::TryFromIntError;
            fn try_from(value: $t) -> Result<Self, Self::Error> {
                let vec = value.try_into()?;
                Ok(Self::Varying(vec))
            }
        }
    };
    ( $t:ty, $g:ident ) => {
        impl<const $g: usize> TryFrom<$t> for RectangularChunkGridDimensionConfiguration {
            type Error = std::num::TryFromIntError;
            fn try_from(value: $t) -> Result<Self, Self::Error> {
                let vec = value.try_into()?;
                Ok(Self::Varying(vec))
            }
        }
    };
}

from_chunkgrid_rectangular!(Vec<NonZeroU64>);
from_chunkgrid_rectangular!(&[NonZeroU64]);
from_chunkgrid_rectangular!([NonZeroU64; N], N);
from_chunkgrid_rectangular!(&[NonZeroU64; N], N);
try_from_chunkgrid_rectangular_configuration!(Vec<u64>);
try_from_chunkgrid_rectangular_configuration!(&[u64]);
try_from_chunkgrid_rectangular_configuration!([u64; N], N);
try_from_chunkgrid_rectangular_configuration!(&[u64; N], N);
