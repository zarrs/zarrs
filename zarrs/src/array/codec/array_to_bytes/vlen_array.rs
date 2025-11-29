//! The `vlen-array` array to bytes codec (Experimental).
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ### Compatible Implementations
//! This codec is fully compatible with the `numcodecs.vlen-array` codec in `zarr-python`.
//!
//! ### Specification
//! - <https://codec.zarrs.dev/array_to_bytes/vlen_array>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `numcodecs.vlen-array`
//! - `https://codec.zarrs.dev/array_to_bytes/vlen_array`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `vlen-array`
//!
//! ### Codec `configuration` Example - [`VlenArrayCodecConfiguration`]:
//! ```json
//! {}
//! ```

use crate::array::codec::array_to_bytes::vlen_v2::vlen_v2_macros;
pub use crate::metadata_ext::codec::vlen_array::{
    VlenArrayCodecConfiguration, VlenArrayCodecConfigurationV1,
};

vlen_v2_macros::vlen_v2_module!(vlen_array, vlen_array_codec, VlenArrayCodec, VLEN_ARRAY);
