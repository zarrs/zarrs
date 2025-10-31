//! `suffix` chunk key encoding metadata.

use serde::{Deserialize, Serialize};

use derive_more::Display;

use zarrs_metadata::{v3::MetadataV3, ConfigurationSerialize};

/// A `suffix` chunk key encoding configuration.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct SuffixChunkKeyEncodingConfiguration {
    /// The suffix to append to chunk keys.
    pub suffix: String,
    /// The base chunk key encoding.
    #[serde(rename = "base-encoding")]
    pub base_encoding: MetadataV3,
}

impl ConfigurationSerialize for SuffixChunkKeyEncodingConfiguration {}
