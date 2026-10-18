//! The `v2` chunk key encoding.

use zarrs_chunk_key_encoding::{
    ChunkKeyEncoding, ChunkKeyEncodingError, ChunkKeyEncodingPlugin, ChunkKeyEncodingTraits,
};
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata::{ChunkKeySeparator, Configuration};
pub use zarrs_metadata_ext::chunk_key_encoding::v2::V2ChunkKeyEncodingConfiguration;
use zarrs_plugin::PluginCreateError;

zarrs_plugin::impl_extension_aliases!(V2ChunkKeyEncoding, v3: "v2");

// Register the chunk key encoding.
inventory::submit! {
    ChunkKeyEncodingPlugin::new::<V2ChunkKeyEncoding>()
}

/// A `v2` chunk key encoding.
///
/// The identifier for chunk with at least one dimension is formed by concatenating for each dimension:
/// - the ASCII decimal string representation of the chunk index within that dimension, followed by
/// - the separator character, except that it is omitted for the last dimension.
///
/// See <https://zarr-specs.readthedocs.io/en/latest/v3/chunk-key-encodings/v2/index.html>.
#[derive(Debug, Clone)]
pub struct V2ChunkKeyEncoding {
    separator: ChunkKeySeparator,
}

impl V2ChunkKeyEncoding {
    /// Create a new `v2` chunk key encoding with separator `separator`.
    #[must_use]
    pub const fn new(separator: ChunkKeySeparator) -> Self {
        Self { separator }
    }

    /// Create a new `v2` chunk key encoding with separator `.`.
    #[must_use]
    pub const fn new_dot() -> Self {
        Self {
            separator: ChunkKeySeparator::Dot,
        }
    }

    /// Create a new `v2` chunk key encoding with separator `/`.
    #[must_use]
    pub const fn new_slash() -> Self {
        Self {
            separator: ChunkKeySeparator::Slash,
        }
    }
}

impl Default for V2ChunkKeyEncoding {
    /// Create a `v2` chunk key encoding with default separator: `.`.
    fn default() -> Self {
        Self {
            separator: ChunkKeySeparator::Dot,
        }
    }
}

impl ChunkKeyEncodingTraits for V2ChunkKeyEncoding {
    fn create(metadata: &MetadataV3) -> Result<ChunkKeyEncoding, PluginCreateError> {
        let configuration: V2ChunkKeyEncodingConfiguration = metadata.to_typed_configuration()?;
        let v2 = V2ChunkKeyEncoding::new(configuration.separator);
        Ok(v2.into())
    }

    fn configuration(&self) -> Configuration {
        V2ChunkKeyEncodingConfiguration {
            separator: self.separator,
        }
        .into()
    }

    fn encode(&self, chunk_grid_indices: &[u64]) -> Result<String, ChunkKeyEncodingError> {
        Ok(if chunk_grid_indices.is_empty() {
            '0'.to_string()
        } else {
            super::join_chunk_key("", chunk_grid_indices, self.separator, "")
        })
    }

    fn encode_into(
        &self,
        chunk_grid_indices: &[u64],
        key: &mut String,
    ) -> Result<(), ChunkKeyEncodingError> {
        if chunk_grid_indices.is_empty() {
            key.push('0');
        } else {
            key.reserve(super::chunk_key_capacity("", chunk_grid_indices, ""));
            super::push_chunk_key(key, "", chunk_grid_indices, self.separator, "");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{NodePath, data_key};
    use zarrs_storage::StoreKey;

    #[test]
    fn slash_nd() {
        let chunk_key_encoding: ChunkKeyEncoding = V2ChunkKeyEncoding::new_slash().into();
        let key = data_key(
            &NodePath::root(),
            &chunk_key_encoding.encode(&[1, 23, 45]).unwrap(),
        )
        .unwrap();
        assert_eq!(key, StoreKey::new("1/23/45").unwrap());
    }

    #[test]
    fn dot_nd() {
        let chunk_key_encoding: ChunkKeyEncoding = V2ChunkKeyEncoding::new_dot().into();
        let key = data_key(
            &NodePath::root(),
            &chunk_key_encoding.encode(&[1, 23, 45]).unwrap(),
        )
        .unwrap();
        assert_eq!(key, StoreKey::new("1.23.45").unwrap());
    }

    #[test]
    fn slash_scalar() {
        let chunk_key_encoding: ChunkKeyEncoding = V2ChunkKeyEncoding::new_slash().into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[]).unwrap()).unwrap();
        assert_eq!(key, StoreKey::new("0").unwrap());
    }

    #[test]
    fn dot_scalar() {
        let chunk_key_encoding: ChunkKeyEncoding = V2ChunkKeyEncoding::new_dot().into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[]).unwrap()).unwrap();
        assert_eq!(key, StoreKey::new("0").unwrap());
    }
}
