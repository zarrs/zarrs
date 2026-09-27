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

use bytes::Bytes;

use super::bytes::reverse_endianness;
use crate::array::{
    ArrayBytesOffsets, ChunkShape, ChunkShapeTraits, CodecChainBound, CowBytes, Endianness,
    data_type,
};
pub use vlen_codec::VlenCodec;
use zarrs_codec::{
    ArrayCodecTraits, ArrayToBytesCodecTraits, Codec, CodecError, CodecOptions, CodecPluginV3,
    CodecTraitsV3, InvalidBytesLengthError,
};
use zarrs_metadata::v3::MetadataV3;
use zarrs_metadata_ext::codec::vlen::VlenIndexLocation;
pub use zarrs_metadata_ext::codec::vlen::{
    VlenCodecConfiguration, VlenCodecConfigurationV0, VlenCodecConfigurationV0_1,
};

zarrs_plugin::impl_extension_aliases!(VlenCodec,
    v3: "zarrs.vlen", ["https://codec.zarrs.dev/array_to_bytes/vlen"]
);

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<VlenCodec>()
}

impl CodecTraitsV3 for VlenCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        crate::warn_experimental_extension(metadata.name(), "codec");
        let configuration: VlenCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(VlenCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToBytes(codec))
    }
}

fn get_vlen_bytes_and_offsets(
    bytes: &CowBytes,
    shape: &[NonZeroU64],
    index_codecs: &CodecChainBound,
    data_codecs: &CodecChainBound,
    index_location: VlenIndexLocation,
    options: &CodecOptions,
) -> Result<(Bytes, ArrayBytesOffsets), CodecError> {
    let index_shape = ChunkShape::from(vec![
        NonZeroU64::try_from(shape.num_elements_u64() + 1).unwrap(),
    ]);
    // Get the index length
    if bytes.len() < size_of::<u64>() {
        return Err(InvalidBytesLengthError::new(bytes.len(), size_of::<u64>()).into());
    }
    let len = bytes.len();
    let index_len_range = match index_location {
        VlenIndexLocation::Start => 0..size_of::<u64>(),
        VlenIndexLocation::End => len - size_of::<u64>()..len,
    };
    let index_len = u64::from_le_bytes(bytes[index_len_range].try_into().unwrap());
    let index_len = usize::try_from(index_len)
        .map_err(|_| CodecError::Other("index length exceeds usize::MAX".to_string()))?;
    let main_len = len - size_of::<u64>();
    if index_len > main_len {
        return Err(CodecError::Other(format!(
            "index length {index_len} exceeds the available encoded length {main_len}"
        )));
    }

    // Get the encoded index and data without copying
    let (index_enc, data_enc) = match index_location {
        VlenIndexLocation::Start => {
            let data_start = size_of::<u64>() + index_len;
            (
                bytes.slice(size_of::<u64>()..data_start),
                bytes.slice(data_start..),
            )
        }
        VlenIndexLocation::End => {
            let index_start = main_len - index_len;
            (
                bytes.slice(index_start..main_len),
                bytes.slice(..index_start),
            )
        }
    };

    // Decode the index
    let mut index = index_codecs
        .decode(index_enc, &index_shape, options)?
        .into_fixed()?;
    let index_data_type = index_codecs.data_type();
    if Endianness::Big.is_native() {
        index.with_mut(|index| reverse_endianness(index, index_data_type));
    }
    // The index is retained without copying if it is shared and aligned
    let index = index.into_static();
    let index = if *index_data_type == data_type::uint32() {
        ArrayBytesOffsets::from_ne_bytes::<u32>(index)?
    } else if *index_data_type == data_type::uint64() {
        ArrayBytesOffsets::from_ne_bytes::<u64>(index)?
    } else {
        return Err(CodecError::Other(
            "unsupported vlen index data type, expected uint32 or uint64".to_string(),
        ));
    };

    // Decode the data
    let data_len_expected = index.last();
    let data = if let Ok(data_len_expected) = NonZeroU64::try_from(data_len_expected as u64) {
        data_codecs
            .decode(data_enc, &[data_len_expected], options)?
            .into_fixed()?
            .into_bytes()
    } else {
        Bytes::new()
    };

    // Check the data length is as expected
    let data_len = data.len();
    if data_len != data_len_expected {
        return Err(CodecError::Other(format!(
            "Expected data length {data_len_expected} does not match data length {data_len}"
        )));
    }

    Ok((data, index))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use itertools::Itertools;
    use zarrs_codec::{CodecOptions, CowBytes, UnboundArrayToBytesCodecTraits};
    use zarrs_data_type::FillValue;

    use super::{VlenCodec, VlenCodecConfiguration};
    use crate::array::data_type;
    use crate::array::element::Element;

    #[test]
    fn codec_vlen_offsets_width() -> Result<(), Box<dyn std::error::Error>> {
        for ((index_data_type, is_u32), index_location) in [("uint32", true), ("uint64", false)]
            .into_iter()
            .cartesian_product(["start", "end"])
        {
            let configuration: VlenCodecConfiguration = serde_json::from_str(&format!(
                r#"{{
                    "data_codecs": [{{"name": "bytes"}}],
                    "index_codecs": [{{"name": "bytes","configuration": {{ "endian": "little" }}}}],
                    "index_data_type": "{index_data_type}",
                    "index_location": "{index_location}"
                }}"#
            ))?;
            let data_type = data_type::string();
            let codec = Arc::new(VlenCodec::new_with_configuration(&configuration)?)
                .with_context(data_type.clone(), FillValue::from(""))?;

            let elements = vec!["a", "bb", "", "dddd"];
            let bytes = <&str>::into_array_bytes(&data_type, elements)?.into_owned();
            let shape = [NonZeroU64::new(4).unwrap()];
            let encoded = codec.encode(bytes.clone(), &shape, &CodecOptions::default())?;
            let encoded = CowBytes::Shared(encoded.into_bytes());
            let encoded_range = encoded.as_ptr_range();
            let decoded = codec.decode(encoded.clone(), &shape, &CodecOptions::default())?;
            assert_eq!(decoded, bytes);
            let offsets = decoded.offsets().unwrap();
            assert_eq!(offsets.is_u32(), is_u32);

            // Shared, aligned and little-endian indexes are not copied
            let offsets_ptr = offsets.as_ne_bytes().as_ptr();
            let index_start = if index_location == "start" {
                8
            } else {
                encoded.len() - 8 - offsets.as_ne_bytes().len()
            };
            let index_aligned =
                encoded[index_start..]
                    .as_ptr()
                    .align_offset(if is_u32 { 4 } else { 8 })
                    == 0;
            if index_aligned && cfg!(target_endian = "little") {
                assert!(encoded_range.contains(&offsets_ptr));
            } else {
                assert!(!encoded_range.contains(&offsets_ptr));
            }
            // The data is never copied with a passthrough data codec chain
            let data_ptr = decoded.into_variable()?.bytes().as_ptr();
            assert!(encoded_range.contains(&data_ptr));
        }
        Ok(())
    }
}
