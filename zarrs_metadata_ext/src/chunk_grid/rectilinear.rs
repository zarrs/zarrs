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
        /// The chunk edge lengths per dimension.
        chunk_shapes: Vec<ChunkEdgeLengths>,
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

/// Chunk edge lengths for a dimension in a rectilinear chunk grid.
///
/// Can be specified as either:
/// - A scalar integer representing a regular grid of fixed-size chunks
/// - An array of integers (potentially run-length encoded) representing varying chunk sizes
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, derive_more::From)]
#[serde(untagged)]
pub enum ChunkEdgeLengths {
    /// An array of chunk edge lengths (potentially run-length encoded).
    Varying(Vec<RunLengthElement>),
    /// A scalar value representing a regular grid with fixed-size chunks.
    Scalar(NonZeroU64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rectilinear_spec_example() {
        let json = r#"
        {
            "kind": "inline",
            "chunk_shapes": [
                4,
                [1, 2, 3],
                [[4, 2]],
                [[1, 3], 3],
                [4, 4, 4]
            ]
        }"#;
        let _config: RectilinearChunkGridConfiguration = serde_json::from_str(json).unwrap();
    }

    #[test]
    fn rectilinear_scalar_chunk_shapes() {
        // Test from the updated spec showing scalar integers
        let json = r#"
        {
            "kind": "inline",
            "chunk_shapes": [
                4,
                [1, 2, 3],
                [[4, 2]],
                [[1, 3], 3],
                [4, 4, 4]
            ]
        }"#;
        let config: RectilinearChunkGridConfiguration = serde_json::from_str(json).unwrap();

        let RectilinearChunkGridConfiguration::Inline { chunk_shapes } = &config;
        assert_eq!(chunk_shapes.len(), 5);

        // First dimension should be scalar
        assert!(matches!(&chunk_shapes[0], ChunkEdgeLengths::Scalar(v) if v.get() == 4));

        // Second dimension should be array
        assert!(matches!(&chunk_shapes[1], ChunkEdgeLengths::Varying(_)));
    }

    #[test]
    fn rectilinear_mixed_scalar_and_array() {
        let json = r#"
        {
            "kind": "inline",
            "chunk_shapes": [10, [5, 5]]
        }"#;
        let config: RectilinearChunkGridConfiguration = serde_json::from_str(json).unwrap();

        let RectilinearChunkGridConfiguration::Inline { chunk_shapes } = &config;
        assert_eq!(chunk_shapes.len(), 2);
        assert!(matches!(&chunk_shapes[0], ChunkEdgeLengths::Scalar(v) if v.get() == 10));
        assert!(matches!(&chunk_shapes[1], ChunkEdgeLengths::Varying(_)));
    }
}
