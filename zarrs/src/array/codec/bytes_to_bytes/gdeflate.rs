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

use crate::array::BytesRepresentation;
use zarrs_codec::{Codec, CodecError, CodecPluginV3, CodecTraitsV3, InvalidBytesLengthError};
pub use zarrs_metadata_ext::codec::gdeflate::{
    GDeflateCodecConfiguration, GDeflateCodecConfigurationV0, GDeflateCompressionLevel,
    GDeflateCompressionLevelError,
};

zarrs_plugin::impl_extension_aliases!(GDeflateCodec, v3: "zarrs.gdeflate");

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<GDeflateCodec>()
}

impl CodecTraitsV3 for GDeflateCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        crate::warn_experimental_extension(metadata.name(), "codec");
        let configuration: GDeflateCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(GDeflateCodec::new_with_configuration(&configuration)?);
        Ok(Codec::BytesToBytes(codec))
    }
}

const GDEFLATE_PAGE_SIZE_UNCOMPRESSED: usize = 65536;
const GDEFLATE_STATIC_HEADER_LENGTH: usize = 2 * size_of::<u64>();

fn read_u64_le(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().unwrap())
}

/// The validated static header of `gdeflate` encoded bytes.
struct GDeflateHeader {
    decoded_len: usize,
    num_pages: usize,
}

/// Decode the static header.
///
/// The page sizes must fit in the encoded bytes and the pages must be able to hold the decoded length.
/// This does not bound the decoded length by much, since a page decodes to up to 64 KiB from only 8 bytes of page size.
/// Do not allocate the decoded length of an untrusted header before it is checked against the decoded representation or the pages are decoded.
fn gdeflate_decode_header(encoded_value: &[u8]) -> Result<GDeflateHeader, CodecError> {
    if encoded_value.len() < GDEFLATE_STATIC_HEADER_LENGTH {
        return Err(InvalidBytesLengthError::new(
            encoded_value.len(),
            GDEFLATE_STATIC_HEADER_LENGTH,
        )
        .into());
    }
    let decoded_len = usize::try_from(read_u64_le(&encoded_value[0..size_of::<u64>()]))
        .map_err(|err| CodecError::Other(err.to_string()))?;
    let num_pages = usize::try_from(read_u64_le(
        &encoded_value[size_of::<u64>()..2 * size_of::<u64>()],
    ))
    .map_err(|err| CodecError::Other(err.to_string()))?;

    // Check length of dynamic header
    num_pages
        .checked_mul(size_of::<u64>())
        .and_then(|length| length.checked_add(GDEFLATE_STATIC_HEADER_LENGTH))
        .filter(|length| *length <= encoded_value.len())
        .ok_or_else(|| {
            InvalidBytesLengthError::new(
                encoded_value.len(),
                num_pages
                    .saturating_mul(size_of::<u64>())
                    .saturating_add(GDEFLATE_STATIC_HEADER_LENGTH),
            )
        })?;

    // Each page decodes to at most one page of bytes
    if decoded_len.div_ceil(GDEFLATE_PAGE_SIZE_UNCOMPRESSED) > num_pages {
        return Err(CodecError::Other(
            "the gdeflate decoded length exceeds the capacity of the pages".to_string(),
        ));
    }
    Ok(GDeflateHeader {
        decoded_len,
        num_pages,
    })
}

/// Decodes the pages of `gdeflate` encoded bytes in order.
struct GDeflatePageDecoder<'a> {
    encoded_value: &'a [u8],
    decompressor: GDeflateDecompressor,
    num_pages: usize,
    page: usize,
    page_offset: usize,
}

impl<'a> GDeflatePageDecoder<'a> {
    fn new(encoded_value: &'a [u8], header: &GDeflateHeader) -> Result<Self, CodecError> {
        Ok(Self {
            encoded_value,
            decompressor: GDeflateDecompressor::new()?,
            num_pages: header.num_pages,
            page: 0,
            page_offset: GDEFLATE_STATIC_HEADER_LENGTH + header.num_pages * size_of::<u64>(),
        })
    }

    /// Decode the next page into `out`, which must be the decoded length of the page.
    ///
    /// This is a full page of decoded bytes except for the last page.
    fn decode_next(&mut self, out: &mut [u8]) -> Result<(), CodecError> {
        debug_assert!(self.page < self.num_pages);
        let encoded_value = self.encoded_value;

        // Get the compressed page length
        let page_size_compressed_offset =
            GDEFLATE_STATIC_HEADER_LENGTH + self.page * size_of::<u64>();
        let page_size_compressed = usize::try_from(read_u64_le(
            &encoded_value
                [page_size_compressed_offset..page_size_compressed_offset + size_of::<u64>()],
        ))
        .map_err(|err| CodecError::Other(err.to_string()))?;

        // Get the compressed page data
        let page_offset = self.page_offset;
        let page_data = page_offset
            .checked_add(page_size_compressed)
            .and_then(|page_end| encoded_value.get(page_offset..page_end))
            .ok_or_else(|| {
                InvalidBytesLengthError::new(
                    encoded_value.len(),
                    page_offset.saturating_add(page_size_compressed),
                )
            })?;
        let in_page = gdeflate_sys::libdeflate_gdeflate_in_page {
            data: page_data.as_ptr().cast(),
            nbytes: page_data.len(),
        };

        self.decompressor
            .decompress_page(in_page, out.as_mut_ptr(), out.len())?;
        self.page += 1;
        self.page_offset += page_size_compressed;
        Ok(())
    }
}

/// Decode `encoded_value`, which must decode to at most the size of `decoded_representation`.
fn gdeflate_decode(
    encoded_value: &[u8],
    decoded_representation: &BytesRepresentation,
) -> Result<Vec<u8>, CodecError> {
    let header = gdeflate_decode_header(encoded_value)?;
    let mut decoded_value = if let Some(size) = decoded_representation.size() {
        // The decoded length of the header is untrusted, so it must not be larger than expected
        if !u64::try_from(header.decoded_len).is_ok_and(|len| len <= size) {
            return Err(InvalidBytesLengthError::new(
                header.decoded_len,
                usize::try_from(size).unwrap_or(usize::MAX),
            )
            .into());
        }
        Vec::with_capacity(header.decoded_len)
    } else {
        // Allocate as pages are decoded, so that a header that is larger than the pages decode to is an error rather than a large allocation
        Vec::new()
    };

    let mut pages = GDeflatePageDecoder::new(encoded_value, &header)?;
    let mut remaining = header.decoded_len;
    for _ in 0..header.num_pages {
        let page_len = remaining.min(GDEFLATE_PAGE_SIZE_UNCOMPRESSED);
        let page_start = decoded_value.len();
        decoded_value.resize(page_start + page_len, 0);
        pages.decode_next(&mut decoded_value[page_start..])?;
        remaining -= page_len;
    }
    if remaining == 0 {
        Ok(decoded_value)
    } else {
        Err(InvalidBytesLengthError::new(decoded_value.len(), header.decoded_len).into())
    }
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
        if result == 0 && actual_out_nbytes == out_nbytes_avail {
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
    use std::sync::Arc;

    use super::*;
    use crate::array::{BytesRepresentation, CowBytes};
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

    /// Chunks of more than one page, which are decoded page by page.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_gdeflate_round_trip_pages() {
        let configuration: GDeflateCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = GDeflateCodec::new_with_configuration(&configuration).unwrap();
        for len in [
            1,
            GDEFLATE_PAGE_SIZE_UNCOMPRESSED,
            GDEFLATE_PAGE_SIZE_UNCOMPRESSED + 1,
            3 * GDEFLATE_PAGE_SIZE_UNCOMPRESSED + 1000,
        ] {
            let bytes: Vec<u8> = (0..len).map(|i| u8::try_from(i % 251).unwrap()).collect();
            let encoded = codec
                .encode(CowBytes::Borrowed(&bytes), &CodecOptions::default())
                .unwrap();
            let decoded = codec
                .decode(
                    encoded,
                    &BytesRepresentation::FixedSize(len as u64),
                    &CodecOptions::default(),
                )
                .unwrap();
            assert_eq!(bytes, decoded.to_vec());
        }
    }

    /// Headers that are inconsistent with the encoded bytes are an error rather than a panic or a
    /// large allocation.
    #[test]
    fn codec_gdeflate_decode_invalid_header() {
        let configuration: GDeflateCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = GDeflateCodec::new_with_configuration(&configuration).unwrap();
        let decode = |decoded_len: u64, num_pages: u64| {
            let mut encoded = Vec::new();
            encoded.extend_from_slice(&decoded_len.to_le_bytes());
            encoded.extend_from_slice(&num_pages.to_le_bytes());
            encoded.extend_from_slice(&[0u8; 32]);
            codec.decode(
                CowBytes::from(encoded),
                &BytesRepresentation::UnboundedSize,
                &CodecOptions::default(),
            )
        };
        // The length of the page sizes overflows when added to the length of the static header
        assert!(decode(10, u64::MAX / 8).is_err());
        // The length of the page sizes overflows
        assert!(decode(10, u64::MAX).is_err());
        // More page sizes than the encoded bytes hold
        assert!(decode(10, 5).is_err());
        // A decoded length that the pages cannot hold
        assert!(decode(u64::MAX, 1).is_err());
        assert!(decode(3 * GDEFLATE_PAGE_SIZE_UNCOMPRESSED as u64, 2).is_err());
        // A decoded length that exceeds the decoded representation
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&(GDEFLATE_PAGE_SIZE_UNCOMPRESSED as u64).to_le_bytes());
        encoded.extend_from_slice(&1u64.to_le_bytes());
        encoded.extend_from_slice(&8u64.to_le_bytes());
        encoded.extend_from_slice(&[0u8; 8]);
        for representation in [
            BytesRepresentation::FixedSize(10),
            BytesRepresentation::BoundedSize(10),
        ] {
            assert!(matches!(
                codec.decode(
                    CowBytes::from(encoded.clone()),
                    &representation,
                    &CodecOptions::default()
                ),
                Err(CodecError::UnexpectedChunkDecodedSize(_))
            ));
        }
        // Page sizes that exceed the encoded bytes
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&10u64.to_le_bytes());
        encoded.extend_from_slice(&1u64.to_le_bytes());
        encoded.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(
            codec
                .decode(
                    CowBytes::from(encoded),
                    &BytesRepresentation::UnboundedSize,
                    &CodecOptions::default(),
                )
                .is_err()
        );
    }

    /// A large decoded length in a header does not cause a large allocation before page validation.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_gdeflate_decode_large_header() {
        let configuration: GDeflateCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = GDeflateCodec::new_with_configuration(&configuration).unwrap();
        // The pages can hold a decoded length of 8 GiB, but they are empty.
        let num_pages: usize = 1 << 17;
        let decoded_len = (num_pages as u64) * GDEFLATE_PAGE_SIZE_UNCOMPRESSED as u64;
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&decoded_len.to_le_bytes());
        encoded.extend_from_slice(&(num_pages as u64).to_le_bytes());
        encoded.resize(
            GDEFLATE_STATIC_HEADER_LENGTH + num_pages * size_of::<u64>(),
            0,
        );
        assert!(
            codec
                .decode(
                    CowBytes::from(encoded),
                    &BytesRepresentation::UnboundedSize,
                    &CodecOptions::default(),
                )
                .is_err()
        );
    }

    /// Invalid encoded bytes are an error rather than a panic.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_gdeflate_decode_invalid() {
        let configuration: GDeflateCodecConfiguration = serde_json::from_str(JSON_VALID).unwrap();
        let codec = GDeflateCodec::new_with_configuration(&configuration).unwrap();
        let bytes: Vec<u8> = (0..2 * GDEFLATE_PAGE_SIZE_UNCOMPRESSED)
            .map(|i| u8::try_from(i % 251).unwrap())
            .collect();
        let encoded = codec
            .encode(CowBytes::Borrowed(&bytes), &CodecOptions::default())
            .unwrap()
            .to_vec();
        let decode = |encoded: Vec<u8>| {
            codec.decode(
                CowBytes::from(encoded),
                &BytesRepresentation::FixedSize(bytes.len() as u64),
                &CodecOptions::default(),
            )
        };
        assert!(decode(encoded.clone()).is_ok());

        // Truncated in the header, the page sizes, and the pages
        for len in [4, GDEFLATE_STATIC_HEADER_LENGTH + 4, encoded.len() - 10] {
            assert!(decode(encoded[..len].to_vec()).is_err());
        }
        // Corrupt pages
        let header_length = GDEFLATE_STATIC_HEADER_LENGTH + 2 * size_of::<u64>();
        let mut corrupt = encoded.clone();
        corrupt[header_length..].fill(0xFF);
        assert!(decode(corrupt).is_err());
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
            .encode(CowBytes::Borrowed(&bytes), &CodecOptions::default())
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
            .encode(CowBytes::from(bytes), &CodecOptions::default())
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
            .encode(CowBytes::from(bytes), &CodecOptions::default())
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
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect();
        let answer: Vec<u16> = vec![2, 3, 5];
        assert_eq!(answer, decoded_partial_chunk);
    }
}
