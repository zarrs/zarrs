//! An explicit coordinate-to-key chunk key encoding (`zarrs.experimental.explicit`).
//!
//! Maps explicit chunk grid coordinates to arbitrary relative store keys via `[indices, key]` pairs
//! in Zarr V3 metadata, e.g.:
//! ```json
//! {
//!   "name": "zarrs.experimental.explicit",
//!   "configuration": {
//!     "chunks": [
//!       [[0, 0, 0], "part_a.h5"],
//!       [[1, 0, 0], "nested/part_b.h5"]
//!     ]
//!   }
//! }
//! ```

use std::collections::HashMap;

use derive_more::Display;
use serde::{Deserialize, Serialize};
use zarrs::array::chunk_key_encoding::api::{
    ChunkKeyEncoding, ChunkKeyEncodingError, ChunkKeyEncodingPlugin, ChunkKeyEncodingTraits,
};
use zarrs::metadata::v3::MetadataV3;
use zarrs::metadata::{Configuration, ConfigurationSerialize};
use zarrs_plugin::PluginCreateError;

/// Configuration for [`ExplicitChunkKeyEncoding`].
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub(crate) struct ExplicitChunkKeyEncodingConfiguration {
    /// Explicit mapping of `[chunk_grid_indices, relative_store_key]` pairs.
    pub(crate) chunks: Vec<(Vec<u64>, String)>,
}

impl ConfigurationSerialize for ExplicitChunkKeyEncodingConfiguration {}

/// An explicit mapping from chunk grid coordinates to relative store keys.
#[derive(Debug, Clone)]
pub(crate) struct ExplicitChunkKeyEncoding {
    dimensionality: Option<usize>,
    chunks: HashMap<Vec<u64>, String>,
}

zarrs_plugin::impl_extension_aliases!(ExplicitChunkKeyEncoding, v3: "zarrs.experimental.explicit");

inventory::submit! {
    ChunkKeyEncodingPlugin::new::<ExplicitChunkKeyEncoding>()
}

impl ExplicitChunkKeyEncoding {
    /// Create a new [`ExplicitChunkKeyEncoding`] from `(chunk_grid_indices, key)` pairs.
    pub(crate) fn new<I, K>(entries: I) -> Result<Self, PluginCreateError>
    where
        I: IntoIterator<Item = (Vec<u64>, K)>,
        K: Into<String>,
    {
        let mut dimensionality: Option<usize> = None;
        let mut chunks = HashMap::new();

        for (indices, key) in entries {
            let key = key.into();
            if key.is_empty() || key.starts_with('/') || key.ends_with('/') || key.contains("//") {
                return Err(PluginCreateError::Other(format!(
                    "explicit chunk key {key:?} must be a non-empty relative key without leading/trailing '/' or '//'"
                )));
            }
            if let Some(dim) = dimensionality {
                if indices.len() != dim {
                    return Err(PluginCreateError::Other(format!(
                        "explicit chunk indices {indices:?} have dimensionality {}, expected {dim}",
                        indices.len()
                    )));
                }
            } else {
                dimensionality = Some(indices.len());
            }
            if chunks.insert(indices.clone(), key).is_some() {
                return Err(PluginCreateError::Other(format!(
                    "duplicate explicit chunk key mapping for indices {indices:?}"
                )));
            }
        }

        Ok(Self {
            dimensionality,
            chunks,
        })
    }
}

impl ChunkKeyEncodingTraits for ExplicitChunkKeyEncoding {
    fn create(metadata: &MetadataV3) -> Result<ChunkKeyEncoding, PluginCreateError> {
        let config: ExplicitChunkKeyEncodingConfiguration = metadata.to_typed_configuration()?;
        Ok(Self::new(config.chunks)?.into())
    }

    fn configuration(&self) -> Configuration {
        let mut chunks: Vec<(Vec<u64>, String)> = self
            .chunks
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        chunks.sort_by(|a, b| a.0.cmp(&b.0));
        ExplicitChunkKeyEncodingConfiguration { chunks }.into()
    }

    fn encode(&self, chunk_grid_indices: &[u64]) -> Result<String, ChunkKeyEncodingError> {
        let mut key = String::new();
        self.encode_into(chunk_grid_indices, &mut key)?;
        Ok(key)
    }

    fn encode_into(
        &self,
        chunk_grid_indices: &[u64],
        key: &mut String,
    ) -> Result<(), ChunkKeyEncodingError> {
        if let Some(dim) = self.dimensionality
            && chunk_grid_indices.len() != dim
        {
            return Err(ChunkKeyEncodingError::IncompatibleDimensionality(
                chunk_grid_indices.len(),
                format!("explicit chunk key encoding expects {dim}D indices"),
            ));
        }
        let mapped = self.chunks.get(chunk_grid_indices).ok_or_else(|| {
            ChunkKeyEncodingError::Other(format!(
                "chunk indices {chunk_grid_indices:?} are not present in explicit chunk key mapping"
            ))
        })?;
        key.push_str(mapped);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_mapping_roundtrip() {
        let encoding = ExplicitChunkKeyEncoding::new([
            (vec![0, 0, 0], "part_a.h5"),
            (vec![1, 0, 0], "nested/part_b.h5"),
        ])
        .unwrap();
        assert_eq!(encoding.encode(&[0, 0, 0]).unwrap(), "part_a.h5");
        assert_eq!(encoding.encode(&[1, 0, 0]).unwrap(), "nested/part_b.h5");
        assert!(encoding.encode(&[2, 0, 0]).is_err());
        assert!(encoding.encode(&[0, 0]).is_err());

        let wrapped: ChunkKeyEncoding = encoding.into();
        let metadata = wrapped.metadata();
        let reopened = ChunkKeyEncoding::from_metadata(&metadata).unwrap();
        assert_eq!(reopened.encode(&[0, 0, 0]).unwrap(), "part_a.h5");
        assert_eq!(reopened.encode(&[1, 0, 0]).unwrap(), "nested/part_b.h5");
    }

    #[test]
    fn explicit_mapping_validation() {
        assert!(
            ExplicitChunkKeyEncoding::new([(vec![0], "a.h5"), (vec![0], "b.h5")]).is_err(),
            "duplicate indices must be rejected"
        );
        assert!(
            ExplicitChunkKeyEncoding::new([(vec![0], "a.h5"), (vec![0, 0], "b.h5")]).is_err(),
            "mixed dimensionality must be rejected"
        );
        assert!(
            ExplicitChunkKeyEncoding::new([(vec![0], "/leading.h5")]).is_err(),
            "leading slash must be rejected"
        );
        assert!(
            ExplicitChunkKeyEncoding::new([(vec![0], "a//b.h5")]).is_err(),
            "double slash must be rejected"
        );
    }
}
