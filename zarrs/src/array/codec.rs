//! Zarr codecs.
//!
//! Array chunks can be encoded using a sequence of codecs, each of which specifies a bidirectional transform (an encode transform and a decode transform).
//! A codec can map array to an array, an array to bytes, or bytes to bytes.
//! A codec may support partial decoding to extract a byte range or array subset without needing to decode the entire input.
//!
//! A [`CodecChain`] represents a codec sequence consisting of any number of array to array and bytes to bytes codecs, and one array to bytes codec.
//! A codec chain is itself an array to bytes codec.
//! A cache may be inserted into a codec chain to optimise partial decoding where appropriate.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-encoding>.
//!
#![doc = include_str!("../../doc/status/codecs.md")]

pub mod array_to_array;
pub mod array_to_bytes;
pub mod bytes_to_bytes;

use std::sync::Arc;

// Array to array
#[cfg(feature = "bitround")]
pub use array_to_array::bitround::*;
pub use array_to_array::fixedscaleoffset::*;
pub use array_to_array::reshape::*;
pub use array_to_array::squeeze::*;
#[cfg(feature = "transpose")]
pub use array_to_array::transpose::*;
// Array to bytes
pub use array_to_bytes::bytes::*;
pub use array_to_bytes::codec_chain::CodecChain;
pub use array_to_bytes::optional::*;
pub use array_to_bytes::packbits::*;
#[cfg(feature = "pcodec")]
pub use array_to_bytes::pcodec::*;
#[cfg(feature = "sharding")]
pub use array_to_bytes::sharding::*;
pub use array_to_bytes::vlen::*;
pub use array_to_bytes::vlen_array::*;
pub use array_to_bytes::vlen_bytes::*;
pub use array_to_bytes::vlen_utf8::*;
pub use array_to_bytes::vlen_v2::*;
#[cfg(feature = "zfp")]
pub use array_to_bytes::zfp::*;
#[cfg(feature = "zfp")]
pub use array_to_bytes::zfpy::*;

// Bytes to bytes
#[cfg(feature = "adler32")]
pub use bytes_to_bytes::adler32::*;
#[cfg(feature = "blosc")]
pub use bytes_to_bytes::blosc::*;
#[cfg(feature = "bz2")]
pub use bytes_to_bytes::bz2::*;
#[cfg(feature = "crc32c")]
pub use bytes_to_bytes::crc32c::*;
#[cfg(feature = "fletcher32")]
pub use bytes_to_bytes::fletcher32::*;
#[cfg(feature = "gdeflate")]
pub use bytes_to_bytes::gdeflate::*;
#[cfg(feature = "gzip")]
pub use bytes_to_bytes::gzip::*;
pub use bytes_to_bytes::shuffle::*;
#[cfg(feature = "zlib")]
pub use bytes_to_bytes::zlib::*;
#[cfg(feature = "zstd")]
pub use bytes_to_bytes::zstd::*;

mod array_partial_decoder_cache;
mod bytes_partial_decoder_cache;
pub(crate) use array_partial_decoder_cache::ArrayPartialDecoderCache;
pub(crate) use bytes_partial_decoder_cache::BytesPartialDecoderCache;

/// Re-export the `zarrs_codec` API.
///
/// The API is mostly useful to implementors of custom codecs.
/// Users can import less used types (e.g. errors) from this crate if needed.
pub use zarrs_codec as api;

/// Returns the default array-to-bytes codec for a given data type.
///
/// The default codec is dependent on the data type:
///  - [`bytes`](array_to_bytes::bytes) for fixed-length data types,
///  - [`vlen-utf8`](array_to_bytes::vlen_utf8) for the [`StringDataType`](crate::array::data_type::StringDataType) variable-length data type,
///  - [`vlen-bytes`](array_to_bytes::vlen_bytes) for the [`BytesDataType`](crate::array::data_type::BytesDataType) variable-length data type,
///  - [`vlen`](array_to_bytes::vlen) for any other variable-length data type, and
///  - [`optional`](array_to_bytes::optional) wrapping the appropriate inner codec for optional data types.
#[must_use]
pub fn default_array_to_bytes_codec(
    data_type: &zarrs_data_type::DataType,
) -> Arc<dyn zarrs_codec::ArrayToBytesCodecTraits> {
    // Special handling for optional types
    if let Some(opt) = data_type.as_optional() {
        // Create mask codec chain using PackBitsCodec
        let mask_codec_chain = Arc::new(CodecChain::new(
            vec![],
            Arc::new(PackBitsCodec::default()),
            vec![],
        ));

        // For data codec chain, recursively handle nested data types
        let data_codec_chain = Arc::new(CodecChain::new(
            vec![],
            default_array_to_bytes_codec(opt.data_type()), // Recursive call handles nested Optional types
            vec![],
        ));

        return Arc::new(array_to_bytes::optional::OptionalCodec::new(
            mask_codec_chain,
            data_codec_chain,
        ));
    }

    // Handle non-optional types based on size
    if data_type.is_fixed() {
        Arc::<BytesCodec>::default()
    } else {
        // FIXME: Default to VlenCodec if ever stabilised
        // Variable-sized types
        use std::any::TypeId;
        let type_id = data_type.as_any().type_id();
        if type_id == TypeId::of::<crate::array::data_type::StringDataType>() {
            Arc::new(VlenUtf8Codec::new())
        } else if type_id == TypeId::of::<crate::array::data_type::BytesDataType>() {
            Arc::new(VlenBytesCodec::new())
        } else {
            Arc::new(VlenCodec::default())
        }
    }
}
