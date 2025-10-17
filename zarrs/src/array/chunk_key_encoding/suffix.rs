//! The `suffix` chunk key encoding.

pub use zarrs_metadata_ext::chunk_key_encoding::suffix::SuffixChunkKeyEncodingConfiguration;
use zarrs_registry::chunk_key_encoding::SUFFIX;

use crate::{
    array::chunk_key_encoding::ChunkKeyEncodingPlugin,
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
    storage::StoreKey,
};

use super::{ChunkKeyEncoding, ChunkKeyEncodingTraits};

// Register the chunk key encoding.
inventory::submit! {
    ChunkKeyEncodingPlugin::new(SUFFIX, is_name_suffix, create_chunk_key_encoding_suffix)
}

fn is_name_suffix(name: &str) -> bool {
    name.eq(SUFFIX)
}

pub(crate) fn create_chunk_key_encoding_suffix(
    metadata: &MetadataV3,
) -> Result<ChunkKeyEncoding, PluginCreateError> {
    crate::warn_experimental_extension(metadata.name(), "chunk key encoding");
    let configuration: SuffixChunkKeyEncodingConfiguration =
        metadata.to_configuration().map_err(|_| {
            PluginMetadataInvalidError::new(SUFFIX, "chunk key encoding", metadata.to_string())
        })?;

    // Create the base encoding from the required base-encoding field
    let base_encoding = ChunkKeyEncoding::from_metadata(&configuration.base_encoding)?;

    let suffix = SuffixChunkKeyEncoding::new(base_encoding, configuration.suffix);
    Ok(ChunkKeyEncoding::new(suffix))
}

/// A `suffix` chunk key encoding.
///
/// <div class="warning">
/// This chunk key encoding is experimental and may be incompatible with other Zarr V3 implementations.
/// </div>
///
/// This encoding composes over a base chunk key encoding and appends a suffix to each generated key.
/// It enables direct file access and maintains Zarr protocol compatibility while supporting
/// interoperability with external tools.
///
/// See <https://github.com/mkitti/zarr-extensions/blob/main/chunk-key-encodings/suffix/>.
#[derive(Debug, Clone)]
pub struct SuffixChunkKeyEncoding {
    base_encoding: ChunkKeyEncoding,
    suffix: String,
}

impl SuffixChunkKeyEncoding {
    /// Create a new `suffix` chunk key encoding.
    #[must_use]
    pub fn new(base_encoding: ChunkKeyEncoding, suffix: String) -> Self {
        Self {
            base_encoding,
            suffix,
        }
    }
}

impl ChunkKeyEncodingTraits for SuffixChunkKeyEncoding {
    fn create_metadata(&self) -> MetadataV3 {
        let configuration = SuffixChunkKeyEncodingConfiguration {
            suffix: self.suffix.clone(),
            base_encoding: self.base_encoding.create_metadata(),
        };
        MetadataV3::new_with_serializable_configuration(SUFFIX.to_string(), &configuration).unwrap()
    }

    fn encode(&self, chunk_grid_indices: &[u64]) -> StoreKey {
        // Get the base encoding result
        let base_key = self.base_encoding.encode(chunk_grid_indices);

        // Append the suffix
        let key_with_suffix = format!("{}{}", base_key.as_str(), self.suffix);

        unsafe { StoreKey::new_unchecked(key_with_suffix) }
    }
}

#[cfg(test)]
mod tests {
    use crate::node::{data_key, NodePath};

    use super::*;

    #[test]
    fn suffix_with_default_base() {
        // Test with default base encoding (slash separator)
        let base = super::super::default::DefaultChunkKeyEncoding::default();
        let chunk_key_encoding: ChunkKeyEncoding =
            SuffixChunkKeyEncoding::new(base.into(), ".tiff".to_string()).into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[1, 23, 45]));
        assert_eq!(key, StoreKey::new("c/1/23/45.tiff").unwrap());
    }

    #[test]
    fn suffix_with_v2_base() {
        // Test with v2 base encoding
        let base = super::super::v2::V2ChunkKeyEncoding::default();
        let chunk_key_encoding: ChunkKeyEncoding =
            SuffixChunkKeyEncoding::new(base.into(), ".shard.zip".to_string()).into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[1, 23, 45]));
        assert_eq!(key, StoreKey::new("1.23.45.shard.zip").unwrap());
    }

    #[test]
    fn suffix_scalar() {
        // Test with scalar (no indices)
        let base = super::super::default::DefaultChunkKeyEncoding::default();
        let chunk_key_encoding: ChunkKeyEncoding =
            SuffixChunkKeyEncoding::new(base.into(), ".tiff".to_string()).into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[]));
        assert_eq!(key, StoreKey::new("c.tiff").unwrap());
    }

    #[test]
    fn suffix_with_dot_separator() {
        // Test with dot separator
        let base = super::super::default::DefaultChunkKeyEncoding::new_dot();
        let chunk_key_encoding: ChunkKeyEncoding =
            SuffixChunkKeyEncoding::new(base.into(), ".zarr".to_string()).into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[0, 1]));
        assert_eq!(key, StoreKey::new("c.0.1.zarr").unwrap());
    }

    #[test]
    fn roundtrip_metadata() {
        // Test metadata serialization and deserialization
        let base = super::super::default::DefaultChunkKeyEncoding::default();
        let encoding = SuffixChunkKeyEncoding::new(base.into(), ".tiff".to_string());

        let metadata = encoding.create_metadata();
        assert_eq!(metadata.name(), SUFFIX);

        // Verify we can recreate the encoding from metadata
        let recreated = create_chunk_key_encoding_suffix(&metadata).unwrap();
        let key1 = encoding.encode(&[1, 2, 3]);
        let key2 = recreated.encode(&[1, 2, 3]);
        assert_eq!(key1, key2);
    }
}
