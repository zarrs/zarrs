//! The `vlen_v2` array to bytes codec (Experimental).
//!
//! This codec is the same as `vlen-utf8`, `vlen-array`, `vlen-bytes` from Zarr V2, except that it is decoupled from the data type.
//! It can operate on any variable-sized data type.
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ### Compatible Implementations
//! None
//!
//! ### Specification
//! - <https://codec.zarrs.dev/array_to_bytes/vlen_v2>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `zarrs.vlen_v2`
//! - `https://codec.zarrs.dev/array_to_bytes/vlen_v2`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! None
//!
//! ### Codec `configuration` Example - [`VlenV2CodecConfiguration`]:
//! ```json
//! {}
//! ```

mod vlen_v2_codec;
mod vlen_v2_partial_decoder;

pub(crate) mod vlen_v2_macros;

use std::sync::Arc;

pub use vlen_v2::{VlenV2CodecConfiguration, VlenV2CodecConfigurationV0};
pub use vlen_v2_codec::VlenV2Codec;
use zarrs_plugin::ExtensionIdentifier;

use super::vlen_array::VlenArrayCodec;
use super::vlen_bytes::VlenBytesCodec;
use super::vlen_utf8::VlenUtf8Codec;
use crate::array::{
    ArrayBytesRaw,
    codec::{CodecError, InvalidBytesLengthError},
};
use crate::metadata_ext::codec::vlen_v2::{self};
use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

// Register the codec.
inventory::submit! {
    CodecPlugin::new(VlenV2Codec::IDENTIFIER, VlenV2Codec::matches_name, VlenV2Codec::default_name, create_codec_vlen_v2)
}
inventory::submit! {
    CodecPlugin::new(VlenArrayCodec::IDENTIFIER, VlenArrayCodec::matches_name, VlenArrayCodec::default_name, create_codec_vlen_v2)
}
inventory::submit! {
    CodecPlugin::new(VlenBytesCodec::IDENTIFIER, VlenBytesCodec::matches_name, VlenBytesCodec::default_name, create_codec_vlen_v2)
}
inventory::submit! {
    CodecPlugin::new(VlenUtf8Codec::IDENTIFIER, VlenUtf8Codec::matches_name, VlenUtf8Codec::default_name, create_codec_vlen_v2)
}

pub(crate) fn create_codec_vlen_v2(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    match metadata.name() {
        name if name == VlenV2Codec::IDENTIFIER || name == VlenArrayCodec::IDENTIFIER => {
            crate::warn_experimental_extension(metadata.name(), "codec");
        }
        _ => {}
    }
    if metadata.configuration_is_none_or_empty() {
        let codec = Arc::new(VlenV2Codec::new());
        Ok(Codec::ArrayToBytes(codec))
    } else {
        Err(
            PluginMetadataInvalidError::new(VlenV2Codec::IDENTIFIER, "codec", metadata.to_string())
                .into(),
        )
    }
}

fn get_interleaved_bytes_and_offsets(
    num_elements: usize,
    bytes: &ArrayBytesRaw,
) -> Result<(Vec<u8>, Vec<usize>), CodecError> {
    // Validate the bytes is long enough to contain header and element lengths
    let header_length = size_of::<u32>() * (1 + num_elements);
    if bytes.len() < header_length {
        return Err(InvalidBytesLengthError::new(bytes.len(), header_length).into());
    }

    // Validate the number of elements from the header
    let header_num_elements = u32::from_le_bytes((&bytes[0..size_of::<u32>()]).try_into().unwrap());
    if u32::try_from(num_elements).unwrap() != header_num_elements {
        return Err(CodecError::Other(format!(
            "Expected header with {num_elements} elements, got {header_num_elements}"
        )));
    }

    let bytes_len = bytes.len() - header_length;
    let mut bytes_out = Vec::with_capacity(bytes_len);
    let mut offsets_out = Vec::with_capacity(num_elements + 1);
    let mut offset = size_of::<u32>();
    for _element in 0..num_elements {
        let length =
            u32::from_le_bytes(bytes[offset..offset + size_of::<u32>()].try_into().unwrap());
        offset += size_of::<u32>();
        offsets_out.push(bytes_out.len());
        if length != 0 {
            bytes_out.extend_from_slice(&bytes[offset..offset + length as usize]);
            offset += length as usize;
        }
    }
    offsets_out.push(bytes_out.len());

    Ok((bytes_out, offsets_out))
}
