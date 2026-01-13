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
//!   "data_codecs": [
//!     {
//!       "name": "bytes"
//!     },
//!     {
//!       "name": "blosc",
//!       "configuration": {
//!         "cname": "zstd",
//!         "clevel": 5,
//!         "shuffle": "bitshuffle",
//!         "typesize": 1,
//!         "blocksize": 0
//!       }
//!     }
//!   ],
//!   "index_codecs": [
//!     {
//!       "name": "bytes",
//!       "configuration": {
//!         "endian": "little"
//!       }
//!     },
//!     {
//!       "name": "blosc",
//!       "configuration": {
//!         "cname": "zstd",
//!         "clevel": 5,
//!         "shuffle": "shuffle",
//!         "typesize": 4,
//!         "blocksize": 0
//!       }
//!     }
//!   ],
//!   "index_data_type": "uint32",
//!   "index_location": "end"
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::vlen::VlenCodecConfiguration;
//! # let configuration: VlenCodecConfiguration = serde_json::from_str(JSON).unwrap();

mod vlen_codec;
mod vlen_partial_decoder;

use std::num::NonZeroU64;
use std::sync::Arc;

use super::bytes::reverse_endianness;
use crate::array::{
    ArrayBytesRaw, ChunkShape, ChunkShapeTraits, CodecChain, Endianness, convert_from_bytes_slice,
    data_type,
};
use itertools::Itertools;
pub use vlen_codec::VlenCodec;
use zarrs_codec::{
    ArrayToBytesCodecTraits, Codec, CodecError, CodecOptions, CodecPluginV3,
    InvalidBytesLengthError,
};
use zarrs_data_type::FillValue;
use zarrs_metadata::v3::MetadataV3;
pub use zarrs_metadata_ext::codec::vlen::{
    VlenCodecConfiguration, VlenCodecConfigurationV0, VlenCodecConfigurationV0_1,
};
use zarrs_metadata_ext::codec::vlen::{VlenIndexDataType, VlenIndexLocation};
use zarrs_plugin::{PluginConfigurationInvalidError, PluginCreateError};

zarrs_plugin::impl_extension_aliases!(VlenCodec,
    v3: "zarrs.vlen", ["https://codec.zarrs.dev/array_to_bytes/vlen"]
);

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<VlenCodec>(create_codec_vlen_v3)
}

pub(crate) fn create_codec_vlen_v3(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    crate::warn_experimental_extension(metadata.name(), "codec");
    let configuration: VlenCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginConfigurationInvalidError::new(metadata.to_string()))?;
    let codec = Arc::new(VlenCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}

fn get_vlen_bytes_and_offsets(
    bytes: &ArrayBytesRaw,
    shape: &[NonZeroU64],
    index_data_type: VlenIndexDataType,
    index_codecs: &CodecChain,
    data_codecs: &CodecChain,
    index_location: VlenIndexLocation,
    options: &CodecOptions,
) -> Result<(Vec<u8>, Vec<usize>), CodecError> {
    let index_shape = ChunkShape::from(vec![
        NonZeroU64::try_from(shape.num_elements_u64() + 1).unwrap(),
    ]);
    let (data_type, fill_value) = match index_data_type {
        VlenIndexDataType::UInt32 => (data_type::uint32(), FillValue::from(0u32)),
        VlenIndexDataType::UInt64 => (data_type::uint64(), FillValue::from(0u64)),
    };

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
        .decode(
            index_enc.into(),
            &index_shape,
            &data_type,
            &fill_value,
            options,
        )?
        .into_fixed()?;
    if Endianness::Big.is_native() {
        reverse_endianness(index.to_mut(), &data_type::uint64());
    }
    let index = match index_data_type {
        VlenIndexDataType::UInt32 => {
            let index = convert_from_bytes_slice::<u32>(&index);
            offsets_u32_to_usize(index)
        }
        VlenIndexDataType::UInt64 => {
            let index = convert_from_bytes_slice::<u64>(&index);
            offsets_u64_to_usize(index)
        }
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
                &[data_len_expected],
                &data_type::uint8(),
                &0u8.into(),
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
