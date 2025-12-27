//! The `zfpy` array to bytes codec (Experimental).
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! [zfp](https://zfp.io/) is a compressed number format for 1D to 4D arrays of 32/64-bit floating point or integer data.
//! 8/16-bit integer types are supported through promotion to 32-bit in accordance with the [zfp utility functions](https://zfp.readthedocs.io/en/release1.0.1/low-level-api.html#utility-functions).
//!
//! This codec requires the `zfp` feature, which is disabled by default.
//!
//! ### Compatible Implementations
//! This codec is fully compatible with the `numcodecs.zfpy` codec in `zarr-python`.
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.zfpy>
//! - <https://codec.zarrs.dev/array_to_bytes/zfpy>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `numcodecs.zfpy`
//! - `https://codec.zarrs.dev/array_to_bytes/zfpy`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `zfpy`
//!
//! ### Codec `configuration` Example - [`ZfpyCodecConfiguration`]:
//! #### Encode in fixed rate mode with 10.5 compressed bits per value
//! ```rust
//! # let JSON = r#"
//! {
//!     "mode": 2,
//!     "rate": 10.5
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::zfpy::ZfpyCodecConfiguration;
//! # let configuration: ZfpyCodecConfiguration = serde_json::from_str(JSON).unwrap();
//! ```
//!
//! #### Encode in fixed precision mode with 19 uncompressed bits per value
//! ```rust
//! # let JSON = r#"
//! {
//!     "mode": 3,
//!     "precision": 19
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::zfpy::ZfpyCodecConfiguration;
//! # let configuration: ZfpyCodecConfiguration = serde_json::from_str(JSON).unwrap();
//! ```
//!
//! #### Encode in fixed accuracy mode with a tolerance of 0.05
//! ```rust
//! # let JSON = r#"
//! {
//!     "mode": 4,
//!     "tolerance": 0.05
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::zfpy::ZfpyCodecConfiguration;
//! # let configuration: ZfpyCodecConfiguration = serde_json::from_str(JSON).unwrap();
//! ```

mod zfpy_codec;

use std::sync::Arc;

use zarrs_plugin::ExtensionIdentifier;
use zarrs_plugin::{PluginCreateError, PluginMetadataInvalidError};

use crate::array::codec::{Codec, CodecPlugin};
use crate::metadata::v3::MetadataV3;
pub use crate::metadata_ext::codec::zfpy::{
    ZfpyCodecConfiguration, ZfpyCodecConfigurationNumcodecs,
};
pub use zfpy_codec::ZfpyCodec;

// Register the codec.
inventory::submit! {
    CodecPlugin::new(ZfpyCodec::IDENTIFIER, ZfpyCodec::matches_name, ZfpyCodec::default_name, create_codec_zfpy)
}

pub(crate) fn create_codec_zfpy(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: ZfpyCodecConfiguration = metadata.to_configuration().map_err(|_| {
        PluginMetadataInvalidError::new(ZfpyCodec::IDENTIFIER, "codec", metadata.to_string())
    })?;
    let codec = Arc::new(ZfpyCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}
