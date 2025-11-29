//! The `default_suffix` chunk key encoding.

use derive_more::Display;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use super::{ChunkKeyEncoding, ChunkKeyEncodingTraits, ChunkKeySeparator};
use crate::metadata::ConfigurationSerialize;
use crate::{
    array::chunk_key_encoding::ChunkKeyEncodingPlugin,
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
    storage::StoreKey,
};

/// Unique identifier for the `default_suffix` chunk key encoding (extension).
const DEFAULT_SUFFIX: &str = "zarrs.default_suffix"; // TODO: Move to zarrs_registry on stabilisation

/// Configuration parameters for a `default_suffix` chunk key encoding.
// TODO: move to zarrs_metadata_ex on stabilisation
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct DefaultSuffixChunkKeyEncodingConfiguration {
    /// The chunk key separator.
    #[serde(default = "default_separator")]
    pub separator: ChunkKeySeparator,
    /// The chunk key suffix.
    pub suffix: String,
}

impl ConfigurationSerialize for DefaultSuffixChunkKeyEncodingConfiguration {}

const fn default_separator() -> ChunkKeySeparator {
    ChunkKeySeparator::Slash
}

// Register the chunk key encoding.
inventory::submit! {
    ChunkKeyEncodingPlugin::new(DEFAULT_SUFFIX, is_name_default_suffix, create_chunk_key_encoding_default_suffix)
}

fn is_name_default_suffix(name: &str) -> bool {
    name.eq(DEFAULT_SUFFIX)
}

pub(crate) fn create_chunk_key_encoding_default_suffix(
    metadata: &MetadataV3,
) -> Result<ChunkKeyEncoding, PluginCreateError> {
    crate::warn_experimental_extension(metadata.name(), "chunk key encoding");
    let configuration: DefaultSuffixChunkKeyEncodingConfiguration =
        metadata.to_configuration().map_err(|_| {
            PluginMetadataInvalidError::new(
                DEFAULT_SUFFIX,
                "chunk key encoding",
                metadata.to_string(),
            )
        })?;
    let default = DefaultSuffixChunkKeyEncoding::new(configuration.separator, configuration.suffix);
    Ok(ChunkKeyEncoding::new(default))
}

/// A `default_suffix` chunk key encoding.
///
/// <div class="warning">
/// This chunk key encoding is experimental and may be incompatible with other Zarr V3 implementations.
/// </div>
///
/// This matches the functionality of the `default` chunk key encoding, but a suffix is appended to each key.
#[derive(Debug, Clone)]
pub struct DefaultSuffixChunkKeyEncoding {
    separator: ChunkKeySeparator,
    suffix: String,
}

impl DefaultSuffixChunkKeyEncoding {
    /// Create a new `default_suffix` chunk key encoding.
    #[must_use]
    pub const fn new(separator: ChunkKeySeparator, suffix: String) -> Self {
        Self { separator, suffix }
    }
}

impl ChunkKeyEncodingTraits for DefaultSuffixChunkKeyEncoding {
    fn create_metadata(&self) -> MetadataV3 {
        let configuration = DefaultSuffixChunkKeyEncodingConfiguration {
            separator: self.separator,
            suffix: self.suffix.clone(),
        };
        MetadataV3::new_with_serializable_configuration(DEFAULT_SUFFIX.to_string(), &configuration)
            .unwrap()
    }

    fn encode(&self, chunk_grid_indices: &[u64]) -> StoreKey {
        const PREFIX: &str = "c";
        let suffix: &str = &self.suffix;

        let key = if chunk_grid_indices.is_empty() {
            format!("{PREFIX}{suffix}")
        } else {
            // Avoid a heap allocation of the chunk key separator
            let mut separator_str: [u8; 4] = [0; 4];
            let separator_char: char = self.separator.into();
            let separator_str: &str = separator_char.encode_utf8(&mut separator_str);

            // Use itoa for integer conversion, faster than format!
            let mut buffers = vec![itoa::Buffer::new(); chunk_grid_indices.len()];

            let iter = chunk_grid_indices
                .iter()
                .zip(&mut buffers)
                .map(|(&n, buffer)| buffer.format(n));
            #[allow(clippy::let_and_return)]
            let out = [PREFIX].into_iter().chain(iter).join(separator_str) + suffix;
            out
        };
        unsafe { StoreKey::new_unchecked(key) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{data_key, NodePath};

    #[test]
    fn slash_nd() {
        let chunk_key_encoding: ChunkKeyEncoding =
            DefaultSuffixChunkKeyEncoding::new(ChunkKeySeparator::Slash, ".tiff".to_string())
                .into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[1, 23, 45]));
        assert_eq!(key, StoreKey::new("c/1/23/45.tiff").unwrap());
    }

    #[test]
    fn dot_nd() {
        let chunk_key_encoding: ChunkKeyEncoding =
            DefaultSuffixChunkKeyEncoding::new(ChunkKeySeparator::Dot, ".tiff".to_string()).into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[1, 23, 45]));
        assert_eq!(key, StoreKey::new("c.1.23.45.tiff").unwrap());
    }

    #[test]
    fn slash_scalar() {
        let chunk_key_encoding: ChunkKeyEncoding =
            DefaultSuffixChunkKeyEncoding::new(ChunkKeySeparator::Slash, ".tiff".to_string())
                .into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[]));
        assert_eq!(key, StoreKey::new("c.tiff").unwrap());
    }

    #[test]
    fn dot_scalar() {
        let chunk_key_encoding: ChunkKeyEncoding =
            DefaultSuffixChunkKeyEncoding::new(ChunkKeySeparator::Dot, ".tiff".to_string()).into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[]));
        assert_eq!(key, StoreKey::new("c.tiff").unwrap());
    }
}
