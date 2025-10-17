//! The `optional` array to bytes codec (Experimental).
//!
//! Encodes optional (nullable) data by separating the mask and data encoding.
//! This codec is designed for the `Optional` data type, which represents nullable values.
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ### Compatible Implementations
//! None
//!
//! ### Specification
//! - <https://codec.zarrs.dev/array_to_bytes/optional>
//!
//! The `optional` codec separates encoding of the mask (flattened bool array)
//! and the data (flattened bytes, excluding missing elements).
//! This allows for efficient storage when many elements are missing, and enables
//! independent compression strategies for mask and data.
//!
//! Missing elements are not encoded at all in the flattened data, saving space.
//! The mask and data are encoded through independent codec chains specified
//! by `mask_codecs` and `data_codecs` configuration parameters.
//!
//! The encoded format is:
//! - 8 bytes: mask length (u64 little-endian)
//! - 8 bytes: data length (u64 little-endian)
//! - N bytes: encoded mask
//! - M bytes: encoded data (only valid elements)
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `optional`
//! - `https://codec.zarrs.dev/array_to_bytes/optional`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! None
//!
//! ### Codec `configuration` Example - [`OptionalCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "mask_codecs": [
//!         {
//!             "name": "packbits",
//!             "configuration": {}
//!         },
//!         {
//!             "name": "gzip",
//!             "configuration": {"level": 5}
//!         }
//!     ],
//!     "data_codecs": [
//!         {
//!             "name": "bytes",
//!             "configuration": {"endian": "little"}
//!         },
//!         {
//!             "name": "gzip",
//!             "configuration": {"level": 5}
//!         }
//!     ]
//! }
//! # "#;
//! # use zarrs_metadata_ext::codec::optional::OptionalCodecConfiguration;
//! # let configuration: OptionalCodecConfiguration = serde_json::from_str(JSON).unwrap();
//! ```

mod optional_codec;

use std::sync::Arc;

pub use optional_codec::OptionalCodec;
pub use zarrs_metadata_ext::codec::optional::{
    OptionalCodecConfiguration, OptionalCodecConfigurationV1,
};
use zarrs_registry::codec::OPTIONAL;

use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(OPTIONAL, is_identifier_optional, create_codec_optional)
}

fn is_identifier_optional(identifier: &str) -> bool {
    identifier == OPTIONAL
}

pub(crate) fn create_codec_optional(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    crate::warn_experimental_extension(metadata.name(), "codec");
    let configuration: OptionalCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(OPTIONAL, "codec", metadata.to_string()))?;
    let codec = Arc::new(OptionalCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}
