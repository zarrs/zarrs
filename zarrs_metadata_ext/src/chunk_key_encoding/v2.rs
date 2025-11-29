//! `v2` chunk key encoding metadata.

use derive_more::Display;
use serde::{Deserialize, Serialize};
use zarrs_metadata::{ChunkKeySeparator, ConfigurationSerialize};

/// A `v2` chunk key encoding configuration.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct V2ChunkKeyEncodingConfiguration {
    /// The chunk key separator.
    #[serde(default = "v2_separator")]
    pub separator: ChunkKeySeparator,
}

impl ConfigurationSerialize for V2ChunkKeyEncodingConfiguration {}

const fn v2_separator() -> ChunkKeySeparator {
    ChunkKeySeparator::Dot
}
