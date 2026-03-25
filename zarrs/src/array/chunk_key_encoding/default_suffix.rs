//! The `default_suffix` chunk key encoding.

use derive_more::Display;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use zarrs_chunk_key_encoding::{ChunkKeyEncoding, ChunkKeyEncodingPlugin, ChunkKeyEncodingTraits};
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::{ChunkKeySeparator, Configuration, ConfigurationSerialize};
use zarrs_plugin::PluginCreateError;
use zarrs_storage::StoreKey;

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

zarrs_plugin::impl_extension_aliases!(DefaultSuffixChunkKeyEncoding,
    v3: "zarrs.default_suffix", []
);

// Register the chunk key encoding.
inventory::submit! {
    ChunkKeyEncodingPlugin::new::<DefaultSuffixChunkKeyEncoding>()
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
    fn create(metadata: &MetadataV3) -> Result<ChunkKeyEncoding, PluginCreateError> {
        crate::warn_experimental_extension(metadata.name(), "chunk key encoding");
        let configuration: DefaultSuffixChunkKeyEncodingConfiguration =
            metadata.to_typed_configuration()?;
        let default =
            DefaultSuffixChunkKeyEncoding::new(configuration.separator, configuration.suffix);
        Ok(default.into())
    }

    fn configuration(&self) -> Configuration {
        DefaultSuffixChunkKeyEncodingConfiguration {
            separator: self.separator,
            suffix: self.suffix.clone(),
        }
        .into()
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
    use crate::node::{NodePath, data_key};

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
