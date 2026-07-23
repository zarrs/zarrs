//! Zarr array chunk grid metadata.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-grid>.

pub mod rectangular;
pub mod rectilinear;
pub mod regular;
pub mod unstructured_cartesian;

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::unstructured_cartesian::{
        UnstructuredCartesianChunk, UnstructuredCartesianChunkGridConfiguration,
    };

    #[test]
    fn unstructured_cartesian_inline_configuration() {
        let json = r#"
        {
            "kind": "inline",
            "chunks": [
                { "origin": [0, 0], "shape": [10, 20] },
                { "origin": [10, 0], "shape": [5, 20] }
            ]
        }"#;
        let config: UnstructuredCartesianChunkGridConfiguration =
            serde_json::from_str(json).unwrap();

        assert_eq!(
            config,
            UnstructuredCartesianChunkGridConfiguration::Inline {
                chunks: vec![
                    UnstructuredCartesianChunk {
                        origin: vec![0, 0],
                        shape: vec![NonZeroU64::new(10).unwrap(), NonZeroU64::new(20).unwrap()],
                    },
                    UnstructuredCartesianChunk {
                        origin: vec![10, 0],
                        shape: vec![NonZeroU64::new(5).unwrap(), NonZeroU64::new(20).unwrap()],
                    },
                ]
            }
        );
        assert_eq!(
            serde_json::to_string(&config).unwrap(),
            r#"{"kind":"inline","chunks":[{"origin":[0,0],"shape":[10,20]},{"origin":[10,0],"shape":[5,20]}]}"#
        );
    }

    #[test]
    fn unstructured_cartesian_rejects_unknown_fields() {
        let json = r#"
        {
            "kind": "inline",
            "chunks": [
                { "origin": [0, 0], "shape": [10, 20], "extra": true }
            ]
        }"#;

        assert!(serde_json::from_str::<UnstructuredCartesianChunkGridConfiguration>(json).is_err());
    }
}
