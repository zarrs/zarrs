//! The `vlen` array to bytes codec (Experimental).
//!
//! Encodes the offsets and bytes of variable-sized data through independent codec chains.
//! This codec is compatible with any variable-sized data type.
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ### Compatible Implementations
//! None
//!
//! ### Specification
//! - <https://codec.zarrs.dev/array_to_bytes/vlen>
//!
//! Based on <https://github.com/zarr-developers/zeps/pull/47#issuecomment-1710505141> by Jeremy Maitin-Shepard.
//! Additional discussion:
//! - <https://github.com/zarr-developers/zeps/pull/47#issuecomment-2238480835>
//! - <https://github.com/zarr-developers/zarr-python/pull/2036#discussion_r1788465492>
//!
//! This is an alternative `vlen` codec to the `vlen-utf8`, `vlen-bytes`, and `vlen-array` codecs that were introduced in Zarr V2.
//! Rather than interleaving element bytes and lengths, element bytes (data) and offsets (indexes) are encoded separately and concatenated.
//! Unlike the legacy `vlen-*` codecs, this new `vlen` codec is suited to partial decoding.
//! Additionally, it it is not coupled to the array data type and can utilise the full potential of the Zarr V3 codec system.
//!
//! Before encoding, the index is structured using the Apache arrow variable-size binary layout with the validity bitmap elided.
//! The index has `length + 1` offsets which are monotonically increasing such that
//! ```rust,ignore
//! element_position = offsets[j]
//! element_length = offsets[j + 1] - offsets[j]  // (for 0 <= j < length)
//! ```
//! where `length` is the number of chunk elements.
//! The index can be encoded with either `uint32` or `uint64` offsets dependent on the `index_data_type` configuration parameter.
//!
//! The data and index can use their own independent codec chain with support for any Zarr V3 codecs.
//! The codecs are specified by `data_codecs` and `index_codecs` parameters in the codec configuration.
//!
//! The index length and index can be encoded at the start or end of each chunk.
//! If `index_location` is `start`:
//! - The first 8 bytes hold a u64 little-endian indicating the length of the encoded index.
//! - This is followed by the encoded index and then the encoded bytes with no padding.
//!
//! If `index_location` is `end`:
//! - The last 8 bytes hold the length of the encoded index.
//! - The encoded index lies between the encoded data and the index length.
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `zarrs.vlen`
//! - `https://codec.zarrs.dev/array_to_bytes/vlen`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! None
//!
//! ### Codec `configuration` Example - [`VlenCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "data_codecs": [
//!             {
//!                     "name": "bytes"
//!             },
//!             {
//!                     "name": "blosc",
//!                     "configuration": {
//!                             "cname": "zstd",
//!                             "clevel": 5,
//!                             "shuffle": "bitshuffle",
//!                             "typesize": 1,
//!                             "blocksize": 0
//!                     }
//!             }
//!     ],
//!     "index_codecs": [
//!             {
//!                     "name": "bytes",
//!                     "configuration": {
//!                             "endian": "little"
//!                     }
//!             },
//!             {
//!                     "name": "blosc",
//!                     "configuration": {
//!                             "cname": "zstd",
//!                             "clevel": 5,
//!                             "shuffle": "shuffle",
//!                             "typesize": 4,
//!                             "blocksize": 0
//!                     }
//!             }
//!     ],
//!     "index_data_type": "uint32",
//!     "index_location": "end"
//! }
//! # "#;
//! # use zarrs_metadata_ext::codec::vlen::VlenCodecConfiguration;
//! # let configuration: VlenCodecConfiguration = serde_json::from_str(JSON).unwrap();

mod vlen_codec;
mod vlen_partial_decoder;

use std::{num::NonZeroU64, sync::Arc};

use itertools::Itertools;
use zarrs_metadata_ext::codec::vlen::VlenIndexLocation;

use crate::array::{
    codec::{ArrayToBytesCodecTraits, CodecError, CodecOptions, InvalidBytesLengthError},
    convert_from_bytes_slice, ChunkRepresentation, CodecChain, DataType, Endianness, RawBytes,
};
pub use zarrs_metadata_ext::codec::vlen::{
    VlenCodecConfiguration, VlenCodecConfigurationV0, VlenCodecConfigurationV0_1,
};
use zarrs_registry::codec::VLEN;

pub use vlen_codec::VlenCodec;

use crate::{
    array::codec::{Codec, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

use super::bytes::reverse_endianness;

// Register the codec.
inventory::submit! {
    CodecPlugin::new(VLEN, is_identifier_vlen, create_codec_vlen)
}

fn is_identifier_vlen(identifier: &str) -> bool {
    identifier == VLEN
}

pub(crate) fn create_codec_vlen(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    log::warn!("The `vlen` codec is experimental and may be incompatible with other Zarr V3 implementations.");
    let configuration: VlenCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(VLEN, "codec", metadata.to_string()))?;
    let codec = Arc::new(VlenCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}

fn get_vlen_bytes_and_offsets(
    index_chunk_representation: &ChunkRepresentation,
    bytes: &RawBytes,
    index_codecs: &CodecChain,
    data_codecs: &CodecChain,
    index_location: VlenIndexLocation,
    options: &CodecOptions,
) -> Result<(Vec<u8>, Vec<usize>), CodecError> {
    // Get the index length
    if bytes.len() < size_of::<u64>() {
        return Err(InvalidBytesLengthError::new(bytes.len(), size_of::<u64>()).into());
    }
    let (bytes_index_len, bytes_main) = match index_location {
        VlenIndexLocation::Start => bytes.split_at(size_of::<u64>()),
        VlenIndexLocation::End => {
            let (bytes_main, bytes_index_len) = bytes.split_at(bytes.len() - size_of::<u64>());
            (bytes_index_len, bytes_main)
        }
    };
    let index_len = u64::from_le_bytes(bytes_index_len.try_into().unwrap());
    let index_len = usize::try_from(index_len)
        .map_err(|_| CodecError::Other("index length exceeds usize::MAX".to_string()))?;

    // Get the encoded index and data
    let (index_enc, data_enc) = match index_location {
        VlenIndexLocation::Start => bytes_main.split_at(index_len),
        VlenIndexLocation::End => {
            let (bytes_data, bytes_index) = bytes_main.split_at(bytes_main.len() - index_len);
            (bytes_index, bytes_data)
        }
    };

    // Decode the index
    let mut index = index_codecs
        .decode(index_enc.into(), index_chunk_representation, options)?
        .into_fixed()?;
    if Endianness::Big.is_native() {
        reverse_endianness(index.to_mut(), &DataType::UInt64);
    }
    #[allow(clippy::wildcard_enum_match_arm)]
    let index = match index_chunk_representation.data_type() {
        // DataType::UInt8 => {
        //     let index = convert_from_bytes_slice::<u8>(&index);
        //     offsets_u8_to_usize(index)
        // }
        // DataType::UInt16 => {
        //     let index = convert_from_bytes_slice::<u16>(&index);
        //     offsets_u16_to_usize(index)
        // }
        DataType::UInt32 => {
            let index = convert_from_bytes_slice::<u32>(&index);
            offsets_u32_to_usize(index)
        }
        DataType::UInt64 => {
            let index = convert_from_bytes_slice::<u64>(&index);
            offsets_u64_to_usize(index)
        }
        _ => unreachable!("other data types are not part of VlenIndexDataType"),
    };

    // Get the data length
    let Some(&data_len_expected) = index.last() else {
        return Err(CodecError::Other(
            "Index is empty? It should have at least one element".to_string(),
        ));
    };

    // Decode the data
    let data = if let Ok(data_len_expected) = NonZeroU64::try_from(data_len_expected as u64) {
        data_codecs
            .decode(
                data_enc.into(),
                &unsafe {
                    // SAFETY: data type and fill value are compatible
                    ChunkRepresentation::new_unchecked(
                        vec![data_len_expected],
                        DataType::UInt8,
                        0u8,
                    )
                },
                options,
            )?
            .into_fixed()?
            .into_owned()
    } else {
        vec![]
    };

    // Check the data length is as expected
    let data_len = data.len();
    if data_len != data_len_expected {
        return Err(CodecError::Other(format!(
            "Expected data length {data_len_expected} does not match data length {data_len}"
        )));
    }

    // Validate the offsets
    for (curr, next) in index.iter().tuple_windows() {
        if next < curr || *next > data_len {
            return Err(CodecError::Other(
                "Invalid bytes offsets in vlen Offset64 encoded chunk".to_string(),
            ));
        }
    }

    Ok((data, index))
}

// /// Convert u8 offsets to usize
// ///
// /// # Panics if the offsets exceed [`usize::MAX`].
// fn offsets_u8_to_usize(offsets: Vec<u8>) -> Vec<usize> {
//     if size_of::<u8>() == size_of::<usize>() {
//         bytemuck::allocation::cast_vec(offsets)
//     } else {
//         offsets
//             .into_iter()
//             .map(|offset| usize::from(offset))
//             .collect()
//     }
// }

// /// Convert u16 offsets to usize
// ///
// /// # Panics if the offsets exceed [`usize::MAX`].
// fn offsets_u16_to_usize(offsets: Vec<u16>) -> Vec<usize> {
//     if size_of::<u16>() == size_of::<usize>() {
//         bytemuck::allocation::cast_vec(offsets)
//     } else {
//         offsets
//             .into_iter()
//             .map(|offset| usize::from(offset))
//             .collect()
//     }
// }

/// Convert u32 offsets to usize
///
/// # Panics if the offsets exceed [`usize::MAX`].
fn offsets_u32_to_usize(offsets: Vec<u32>) -> Vec<usize> {
    if size_of::<u32>() == size_of::<usize>() {
        bytemuck::allocation::cast_vec(offsets)
    } else {
        offsets
            .into_iter()
            .map(|offset| usize::try_from(offset).unwrap())
            .collect()
    }
}

/// Convert u64 offsets to usize
///
/// # Panics if the offsets exceed [`usize::MAX`].
fn offsets_u64_to_usize(offsets: Vec<u64>) -> Vec<usize> {
    if size_of::<u64>() == size_of::<usize>() {
        bytemuck::allocation::cast_vec(offsets)
    } else {
        offsets
            .into_iter()
            .map(|offset| usize::try_from(offset).unwrap())
            .collect()
    }
}
