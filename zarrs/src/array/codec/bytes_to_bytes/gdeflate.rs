//! The `gdeflate` bytes to bytes codec (Experimental).
//!
//! Applies [GDeflate](https://docs.nvidia.com/cuda/nvcomp/gdeflate.html) compression.
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ### Compatible Implementations
//! None
//!
//! ### Specification
//! - <https://codec.zarrs.dev/bytes_to_bytes/gdeflate>
//!
//! `gdeflate` encoded data sequentially encodes a static header, a dynamic header, and the compressed bytes.
//!
//! The static header is composed of the following:
//!  - `UNCOMPRESSED_INPUT_LENGTH`: a little-endian 64-bit unsigned integer holding the total uncompressed length of the input bytes.
//!  - `NUMBER_OF_PAGES`: a little-endian 64-bit unsigned integer holding the number of compressed pages.
//!
//! The dynamic header is composed of the following:
//!  - `COMPRESSED_PAGE_SIZES`: `NUMBER_OF_PAGES` little-endian 64-bit unsigned integers holding the compressed sizes of each page.
//!
//! The remaining bytes are the `gdeflate` encoded pages of total length equal to the sum of all `COMPRESSED_PAGE_SIZES`.
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `zarrs.gdeflate`
//! - `https://codec.zarrs.dev/bytes_to_bytes/gdeflate`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! None
//!
//! ### Codec `configuration` Example - [`GDeflateCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "level": 9
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::gdeflate::GDeflateCodecConfiguration;
//! # serde_json::from_str::<GDeflateCodecConfiguration>(JSON).unwrap();
//! ```

mod gdeflate_codec;

use std::sync::Arc;

pub use gdeflate_codec::GDeflateCodec;
use zarrs_metadata::v3::MetadataV3;

use crate::array::ArrayBytesRaw;
use zarrs_codec::{Codec, CodecError, CodecPluginV3, InvalidBytesLengthError};
pub use zarrs_metadata_ext::codec::gdeflate::{
    GDeflateCodecConfiguration, GDeflateCodecConfigurationV0, GDeflateCompressionLevel,
    GDeflateCompressionLevelError,
};
use zarrs_plugin::{PluginConfigurationInvalidError, PluginCreateError};

zarrs_plugin::impl_extension_aliases!(GDeflateCodec, v3: "zarrs.gdeflate");

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<GDeflateCodec>(create_codec_gdeflate_v3)
}

pub(crate) fn create_codec_gdeflate_v3(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    crate::warn_experimental_extension(metadata.name(), "codec");
    let configuration: GDeflateCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginConfigurationInvalidError::new(metadata.to_string()))?;
    let codec = Arc::new(GDeflateCodec::new_with_configuration(&configuration)?);
    Ok(Codec::BytesToBytes(codec))
}

const GDEFLATE_PAGE_SIZE_UNCOMPRESSED: usize = 65536;
const GDEFLATE_STATIC_HEADER_LENGTH: usize = 2 * size_of::<u64>();

fn gdeflate_decode(encoded_value: &ArrayBytesRaw<'_>) -> Result<Vec<u8>, CodecError> {
    if encoded_value.len() < GDEFLATE_STATIC_HEADER_LENGTH {
        return Err(InvalidBytesLengthError::new(
            encoded_value.len(),
            GDEFLATE_STATIC_HEADER_LENGTH,
        )
        .into());
    }

    // Decode the static header
    let as_u64 = |bytes: &[u8]| -> u64 { u64::from_le_bytes(bytes.try_into().unwrap()) };
    let decoded_value_len = as_u64(&encoded_value[0..size_of::<u64>()]);
    let decoded_value_len = usize::try_from(decoded_value_len).unwrap();
    let num_pages = as_u64(&encoded_value[size_of::<u64>()..2 * size_of::<u64>()]);
    let num_pages = usize::try_from(num_pages).unwrap();

    // Check length of dynamic header
    let dynamic_header_length = num_pages * size_of::<u64>();
    if encoded_value.len() < GDEFLATE_STATIC_HEADER_LENGTH + dynamic_header_length {
        return Err(InvalidBytesLengthError::new(
            encoded_value.len(),
            GDEFLATE_STATIC_HEADER_LENGTH + dynamic_header_length,
        )
        .into());
    }

    // Decode the pages
    let decompressor = GDeflateDecompressor::new()?;
    let mut decoded_value = Vec::with_capacity(decoded_value_len);
    let mut page_offset = GDEFLATE_STATIC_HEADER_LENGTH + dynamic_header_length;
    for page in 0..num_pages {
        // Get the compressed page length
        let page_size_compressed_offset = GDEFLATE_STATIC_HEADER_LENGTH + page * size_of::<u64>();
        let page_size_compressed = as_u64(
            &encoded_value
                [page_size_compressed_offset..page_size_compressed_offset + size_of::<u64>()],
        );
        let page_size_compressed = usize::try_from(page_size_compressed).unwrap();

        // Get the compressed page data
        let page_data = &encoded_value[page_offset..page_offset + page_size_compressed];
        let in_page = gdeflate_sys::libdeflate_gdeflate_in_page {
            data: page_data.as_ptr().cast(),
            nbytes: page_data.len(),
        };

        // Decompress the page
        let data_out = decoded_value.spare_capacity_mut();
        let page_size_uncompressed =
            decompressor.decompress_page(in_page, data_out.as_mut_ptr().cast(), data_out.len())?;

        unsafe {
            decoded_value.set_len(decoded_value.len() + page_size_uncompressed);
        }
        page_offset += page_size_compressed;
    }

    Ok(decoded_value)
}

struct GDeflateCompressor(*mut gdeflate_sys::libdeflate_gdeflate_compressor);

impl GDeflateCompressor {
    pub(crate) fn new(compression_level: GDeflateCompressionLevel) -> Result<Self, CodecError> {
        let compressor = unsafe {
            gdeflate_sys::libdeflate_alloc_gdeflate_compressor(compression_level.as_i32())
        };
        if compressor.is_null() {
            Err(CodecError::Other(
                "Failed to create gdeflate compressor".to_string(),
            ))
        } else {
            Ok(Self(compressor))
        }
    }

    fn get_npages_compress_bound(&self, input_length: usize) -> (usize, usize) {
        let mut out_npages = 0;
        let compress_bound = unsafe {
            gdeflate_sys::libdeflate_gdeflate_compress_bound(
                self.0,
                input_length,
                &raw mut out_npages,
            )
        };
        (out_npages, compress_bound)
    }

    pub(crate) fn compress(
        &self,
        uncompressed_bytes: &[u8],
    ) -> Result<(Vec<usize>, Vec<u8>), CodecError> {
        let (out_npages, compress_bound) = self.get_npages_compress_bound(uncompressed_bytes.len());
        // let compress_bound_page = compress_bound / out_npages;

        let mut compressed_bytes = Vec::with_capacity(compress_bound);
        let mut page_sizes = Vec::with_capacity(out_npages);
        for i in 0..out_npages {
            let page_offset = i * GDEFLATE_PAGE_SIZE_UNCOMPRESSED;

            let data_out = compressed_bytes.spare_capacity_mut();
            let mut out_page = gdeflate_sys::libdeflate_gdeflate_out_page {
                data: data_out.as_mut_ptr().cast(),
                nbytes: data_out.len(),
            };

            let data_in = &uncompressed_bytes[page_offset
                ..(page_offset + GDEFLATE_PAGE_SIZE_UNCOMPRESSED).min(uncompressed_bytes.len())];
            let compressed_size = unsafe {
                gdeflate_sys::libdeflate_gdeflate_compress(
                    self.0,
                    data_in.as_ptr().cast(),
                    data_in.len(),
                    &raw mut out_page,
                    1,
                )
            };
            if compressed_size == 0 {
                return Err(CodecError::Other("gdeflate compression failed".to_string()));
            }
            page_sizes.push(compressed_size);
            unsafe {
                compressed_bytes.set_len(compressed_bytes.len() + compressed_size);
            }
        }

        Ok((page_sizes, compressed_bytes))
    }
}

impl Drop for GDeflateCompressor {
    fn drop(&mut self) {
        unsafe { gdeflate_sys::libdeflate_free_gdeflate_compressor(self.0) }
    }
}

struct GDeflateDecompressor(*mut gdeflate_sys::libdeflate_gdeflate_decompressor);

impl GDeflateDecompressor {
    pub(crate) fn new() -> Result<Self, CodecError> {
        let decompressor = unsafe { gdeflate_sys::libdeflate_alloc_gdeflate_decompressor() };
        if decompressor.is_null() {
            Err(CodecError::Other(
                "Failed to create gdeflate compressor".to_string(),
            ))
        } else {
            Ok(Self(decompressor))
        }
    }

    pub(crate) fn decompress_page(
        &self,
        mut in_page: gdeflate_sys::libdeflate_gdeflate_in_page,
        out: *mut u8,
        out_nbytes_avail: usize,
    ) -> Result<usize, CodecError> {
        let mut actual_out_nbytes: usize = 0;
        let result = unsafe {
            gdeflate_sys::libdeflate_gdeflate_decompress(
                self.0,
                &raw mut in_page,
                1,
                out.cast(),
                out_nbytes_avail,
                &raw mut actual_out_nbytes,
            )
        };
        assert_eq!(actual_out_nbytes, out_nbytes_avail);
        if result == 0 {
            Ok(actual_out_nbytes)
        } else {
            Err(CodecError::Other(
                "gdeflate page decompression failed".to_string(),
            ))
        }
    }
}

impl Drop for GDeflateDecompressor {
    fn drop(&mut self) {
        unsafe { gdeflate_sys::libdeflate_free_gdeflate_decompressor(self.0) }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::sync::Arc;

    use super::*;
    use crate::array::BytesRepresentation;
    use zarrs_codec::{BytesPartialDecoderTraits, BytesToBytesCodecTraits, CodecOptions};
    use zarrs_storage::byte_range::ByteRange;

    const JSON_VALID: &str = r#"{
        "level": 1
    }"#;

    #[test]
    fn codec_gdeflate_configuration_valid() {
        assert!(serde_json::from_str::<GDeflateCodecConfiguration>(JSON_VALID).is_ok());
    }

    #[test]
    fn codec_gdeflate_configuration_invalid1() {
        const JSON_INVALID1: &str = r#"{
        "level": -1
    }"#;
        assert!(serde_json::from_str::<GDeflateCodecConfiguration>(JSON_INVALID1).is_err());
    }

    #[test]
    fn codec_gdeflate_configuration_invalid2() {
        const JSON_INVALID2: &str = r#"{
        "level": 13
    }"#;
        assert!(serde_json::from_str::<GDeflateCodecConfiguration>(JSON_INVALID2).is_err());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_gdeflate_round_trip1() {
        let elements: Vec<u16> = (0..32).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: GDeflateCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = GDeflateCodec::new_with_configuration(&configuration).unwrap();

        let encoded = codec
            .encode(Cow::Borrowed(&bytes), &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(encoded, &bytes_representation, &CodecOptions::default())
            .unwrap();
        assert_eq!(bytes, decoded.to_vec());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_gdeflate_partial_decode() {
        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: GDeflateCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = Arc::new(GDeflateCodec::new_with_configuration(&configuration).unwrap());

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions = [
            ByteRange::FromStart(4, Some(4)),
            ByteRange::FromStart(10, Some(2)),
        ];

        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &bytes_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size_held(), input_handle.size_held()); // gdeflate partial decoder does not hold bytes
        let decoded_partial_chunk = partial_decoder
            .partial_decode_many(
                Box::new(decoded_regions.into_iter()),
                &CodecOptions::default(),
            )
            .unwrap()
            .unwrap()
            .concat();

        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .clone()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        let answer: Vec<u16> = vec![2, 3, 5];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    #[cfg_attr(miri, ignore)]
    async fn codec_gdeflate_async_partial_decode() {
        let elements: Vec<u16> = (0..8).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes_representation = BytesRepresentation::FixedSize(bytes.len() as u64);

        let configuration: GDeflateCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = Arc::new(GDeflateCodec::new_with_configuration(&configuration).unwrap());

        let encoded = codec
            .encode(Cow::Owned(bytes), &CodecOptions::default())
            .unwrap();
        let decoded_regions = [
            ByteRange::FromStart(4, Some(4)),
            ByteRange::FromStart(10, Some(2)),
        ];

        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .async_partial_decoder(
                input_handle,
                &bytes_representation,
                &CodecOptions::default(),
            )
            .await
            .unwrap();
        let decoded_partial_chunk = partial_decoder
            .partial_decode_many(
                Box::new(decoded_regions.into_iter()),
                &CodecOptions::default(),
            )
            .await
            .unwrap()
            .unwrap()
            .concat();

        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .clone()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        let answer: Vec<u16> = vec![2, 3, 5];
        assert_eq!(answer, decoded_partial_chunk);
    }
}
