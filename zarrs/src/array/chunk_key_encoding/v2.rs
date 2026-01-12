//! The `v2` chunk key encoding.

use itertools::Itertools;

use super::{ChunkKeyEncoding, ChunkKeyEncodingTraits, ChunkKeySeparator};
use crate::array::chunk_key_encoding::ChunkKeyEncodingPlugin;
use crate::metadata::Configuration;
use crate::metadata::v3::MetadataV3;
pub use crate::metadata_ext::chunk_key_encoding::v2::V2ChunkKeyEncodingConfiguration;
use crate::plugin::{PluginConfigurationInvalidError, PluginCreateError};
use crate::storage::StoreKey;

zarrs_plugin::impl_extension_aliases!(V2ChunkKeyEncoding, v3: "v2");

// Register the chunk key encoding.
inventory::submit! {
    ChunkKeyEncodingPlugin::new::<V2ChunkKeyEncoding>(create_chunk_key_encoding_v2)
}

pub(crate) fn create_chunk_key_encoding_v2(
    metadata: &MetadataV3,
) -> Result<ChunkKeyEncoding, PluginCreateError> {
    let configuration: V2ChunkKeyEncodingConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginConfigurationInvalidError::new(metadata.to_string()))?;
    let v2 = V2ChunkKeyEncoding::new(configuration.separator);
    Ok(v2.into())
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
    fn configuration(&self) -> Configuration {
        V2ChunkKeyEncodingConfiguration {
            separator: self.separator,
        }
        .into()
    }

    fn encode(&self, chunk_grid_indices: &[u64]) -> StoreKey {
        let key = if chunk_grid_indices.is_empty() {
            '0'.to_string()
        } else {
            // Avoid a heap allocation of the chunk key separator
            let mut separator_str: [u8; 4] = [0; 4];
            let separator_char: char = self.separator.into();
            let separator_str: &str = separator_char.encode_utf8(&mut separator_str);

            // Use itoa for integer conversion, faster than format!
            let mut buffers = vec![itoa::Buffer::new(); chunk_grid_indices.len()];

            chunk_grid_indices
                .iter()
                .zip(&mut buffers)
                .map(|(&n, buffer)| buffer.format(n))
                .join(separator_str)
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
        let chunk_key_encoding: ChunkKeyEncoding = V2ChunkKeyEncoding::new_slash().into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[1, 23, 45]));
        assert_eq!(key, StoreKey::new("1/23/45").unwrap());
    }

    #[test]
    fn dot_nd() {
        let chunk_key_encoding: ChunkKeyEncoding = V2ChunkKeyEncoding::new_dot().into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[1, 23, 45]));
        assert_eq!(key, StoreKey::new("1.23.45").unwrap());
    }

    #[test]
    fn slash_scalar() {
        let chunk_key_encoding: ChunkKeyEncoding = V2ChunkKeyEncoding::new_slash().into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[]));
        assert_eq!(key, StoreKey::new("0").unwrap());
    }

    #[test]
    fn dot_scalar() {
        let chunk_key_encoding: ChunkKeyEncoding = V2ChunkKeyEncoding::new_dot().into();
        let key = data_key(&NodePath::root(), &chunk_key_encoding.encode(&[]));
        assert_eq!(key, StoreKey::new("0").unwrap());
    }
}
