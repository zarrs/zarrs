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
    /// A scalar value representing a regular grid with fixed-size chunks.
    Scalar(NonZeroU64),
    /// An array of chunk edge lengths (potentially run-length encoded).
    Varying(Vec<RunLengthElement>),
}

/// Expand run-length encoded chunk edge lengths.
///
/// # Panics
/// Panics if a run-length count does not fit into `usize`.
#[must_use]
pub fn decode_rle_edge_lengths(elements: &[RunLengthElement]) -> Vec<NonZeroU64> {
    let mut result = Vec::new();
    for element in elements {
        match element {
            RunLengthElement::Single(value) => result.push(*value),
            RunLengthElement::Repeated([value, count]) => {
                let count = usize::try_from(count.get()).unwrap();
                result.reserve(count);
                for _ in 0..count {
                    result.push(*value);
                }
            }
        }
    }
    result
}

/// Encode explicit chunk edge lengths as scalar or run-length encoded metadata.
#[must_use]
pub fn encode_rle_edge_lengths(edge_lengths: &[NonZeroU64]) -> ChunkEdgeLengths {
    if let Some(first) = edge_lengths.first().copied() {
        if edge_lengths.iter().all(|edge_length| *edge_length == first) {
            return ChunkEdgeLengths::Scalar(first);
        }
    }
    ChunkEdgeLengths::Varying(compress_run_length(edge_lengths))
}

/// Compress a sequence of chunk edge lengths into run-length encoded form.
fn compress_run_length(sizes: &[NonZeroU64]) -> Vec<RunLengthElement> {
    let mut result = Vec::new();
    let Some(mut current) = sizes.first().copied() else {
        return result;
    };
    let mut count = 0u64;
    for size in sizes {
        if *size == current {
            count += 1;
        } else {
            push_run_length(&mut result, current, count);
            current = *size;
            count = 1;
        }
    }
    push_run_length(&mut result, current, count);
    result
}

fn push_run_length(result: &mut Vec<RunLengthElement>, value: NonZeroU64, count: u64) {
    if count == 1 {
        result.push(RunLengthElement::Single(value));
    } else {
        result.push(RunLengthElement::Repeated([
            value,
            NonZeroU64::new(count).unwrap(),
        ]));
    }
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

    #[test]
    fn encode_rle_edge_lengths_scalar() {
        let edge_lengths = [NonZeroU64::new(4).unwrap(); 3];
        assert!(matches!(
            encode_rle_edge_lengths(&edge_lengths),
            ChunkEdgeLengths::Scalar(value) if value.get() == 4
        ));
    }

    #[test]
    fn encode_rle_edge_lengths_varying() {
        let edge_lengths = [5, 5, 15, 15, 20]
            .into_iter()
            .map(|value| NonZeroU64::new(value).unwrap())
            .collect::<Vec<_>>();
        let chunk_edge_lengths = encode_rle_edge_lengths(&edge_lengths);
        let ChunkEdgeLengths::Varying(elements) = chunk_edge_lengths else {
            panic!("expected varying chunk edge lengths");
        };
        assert_eq!(
            elements,
            vec![
                RunLengthElement::Repeated([
                    NonZeroU64::new(5).unwrap(),
                    NonZeroU64::new(2).unwrap(),
                ]),
                RunLengthElement::Repeated([
                    NonZeroU64::new(15).unwrap(),
                    NonZeroU64::new(2).unwrap(),
                ]),
                RunLengthElement::Single(NonZeroU64::new(20).unwrap()),
            ]
        );
        assert_eq!(decode_rle_edge_lengths(&elements), edge_lengths);
    }
}
