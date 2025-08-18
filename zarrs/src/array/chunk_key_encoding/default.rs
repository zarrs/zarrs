//! The `default` chunk key encoding.

use itertools::Itertools;
pub use zarrs_metadata_ext::chunk_key_encoding::default::DefaultChunkKeyEncodingConfiguration;
use zarrs_registry::chunk_key_encoding::DEFAULT;

use crate::{
    array::chunk_key_encoding::ChunkKeyEncodingPlugin,
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
    storage::StoreKey,
};

use super::{ChunkKeyEncoding, ChunkKeyEncodingTraits, ChunkKeySeparator};

// Register the chunk key encoding.
inventory::submit! {
    ChunkKeyEncodingPlugin::new(DEFAULT, is_name_default, create_chunk_key_encoding_default)
}

fn is_name_default(name: &str) -> bool {
    name.eq(DEFAULT)
}

pub(crate) fn create_chunk_key_encoding_default(
    metadata: &MetadataV3,
) -> Result<ChunkKeyEncoding, PluginCreateError> {
    let configuration: DefaultChunkKeyEncodingConfiguration =
        metadata.to_configuration().map_err(|_| {
            PluginMetadataInvalidError::new(DEFAULT, "chunk key encoding", metadata.to_string())
        })?;
    let default = DefaultChunkKeyEncoding::new(configuration.separator);
    Ok(ChunkKeyEncoding::new(default))
}

/// A `default` chunk key encoding.
///
/// The key for a chunk with grid index (k, j, i, …) is formed by taking the initial prefix c, and appending for each dimension:
/// - the separator character, followed by,
/// - the ASCII decimal string representation of the chunk index within that dimension.
///
/// See <https://zarr-specs.readthedocs.io/en/latest/v3/chunk-key-encodings/default/index.html>.
#[derive(Debug, Clone)]
pub struct DefaultChunkKeyEncoding {
    separator: ChunkKeySeparator,
}

impl DefaultChunkKeyEncoding {
    /// Create a new `default` chunk key encoding with separator `separator`.
    #[must_use]
    pub const fn new(separator: ChunkKeySeparator) -> Self {
        Self { separator }
    }

    /// Create a new `default` chunk key encoding with separator `.`.
    #[must_use]
    pub const fn new_dot() -> Self {
        Self {
            separator: ChunkKeySeparator::Dot,
        }
    }

    /// Create a new `default` chunk key encoding with separator `/`.
    #[must_use]
    pub const fn new_slash() -> Self {
        Self {
            separator: ChunkKeySeparator::Slash,
        }
    }
}

impl Default for DefaultChunkKeyEncoding {
    /// Create a `default` chunk key encoding with default separator: `/`.
    fn default() -> Self {
        Self {
            separator: ChunkKeySeparator::Slash,
        }
    }
}

impl ChunkKeyEncodingTraits for DefaultChunkKeyEncoding {
    fn create_metadata(&self) -> MetadataV3 {
        let configuration = DefaultChunkKeyEncodingConfiguration {
            separator: self.separator,
        };
        MetadataV3::new_with_serializable_configuration(DEFAULT.to_string(), &configuration)
            .unwrap()
    }

    fn encode(&self, chunk_grid_indices: &[u64]) -> StoreKey {
        const PREFIX: &str = "c";

        let key = if chunk_grid_indices.is_empty() {
            PREFIX.to_string()
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
            let out = [PREFIX].into_iter().chain(iter).join(separator_str);
            out
        };
        unsafe { StoreKey::new_unchecked(key) }
    }
}

#[cfg(test)]
mod tests {
    use crate::node::{data_key, NodePath};

    use super::*;

    #[test]
    fn slash_nd() {
        let chunk_key_encoding: ChunkKeyEncoding = DefaultChunkKeyEncoding::new_slash().into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[1, 23, 45]));
        assert_eq!(key, StoreKey::new("c/1/23/45").unwrap());
    }

    #[test]
    fn dot_nd() {
        let chunk_key_encoding: ChunkKeyEncoding = DefaultChunkKeyEncoding::new_dot().into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[1, 23, 45]));
        assert_eq!(key, StoreKey::new("c.1.23.45").unwrap());
    }

    #[test]
    fn slash_scalar() {
        let chunk_key_encoding: ChunkKeyEncoding = DefaultChunkKeyEncoding::new_slash().into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[]));
        assert_eq!(key, StoreKey::new("c").unwrap());
    }

    #[test]
    fn dot_scalar() {
        let chunk_key_encoding: ChunkKeyEncoding = DefaultChunkKeyEncoding::new_dot().into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[]));
        assert_eq!(key, StoreKey::new("c").unwrap());
    }
}
