//! `rectilinear` chunk grid metadata.

use std::num::NonZeroU64;

use derive_more::Display;
use serde::{Deserialize, Serialize};

use zarrs_metadata::ConfigurationSerialize;

/// Configuration parameters for a `rectilinear` chunk grid.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(tag = "kind", rename_all = "lowercase")]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub enum RectilinearChunkGridConfiguration {
    /// Inline chunk shape specification.
    Inline {
        /// The chunk shape.
        chunk_shapes: Vec<RunLengthElements>,
    },
}

impl ConfigurationSerialize for RectilinearChunkGridConfiguration {}

/// An element in run-length encoded chunk configuration.
///
/// This is a run-length encoded representation of chunk sizes for a dimension.
/// Each element is either a single value or a pair [value, count].
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, derive_more::From)]
#[serde(untagged)]
pub enum RunLengthElement {
    /// A pair [value, count] representing `count` repetitions of `value`.
    Repeated([NonZeroU64; 2]),
    /// A single value.
    Single(NonZeroU64),
}

/// An element in the `chunk_shape` field of `rectilinear` chunk grid metadata.
pub type RunLengthElements = Vec<RunLengthElement>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectilinear_spec_example() {
        let json = r#"
        {
            "kind": "inline",
            "chunk_shapes": [
                [[2, 3]],
                [[1, 6]],
                [1, [2, 1], 3],
                [[1, 3], 3],
                [6]
            ]
        }"#;
        let _config: RectilinearChunkGridConfiguration = serde_json::from_str(json).unwrap();
    }
}
