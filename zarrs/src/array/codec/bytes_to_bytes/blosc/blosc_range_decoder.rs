//! Decoding byte ranges of blosc encoded values.

use super::{
    blosc_decompress_bytes, blosc_decompress_bytes_partial, blosc_nbytes, blosc_typesize,
    blosc_validate,
};
use crate::array::CowBytes;
use zarrs_codec::CodecError;
use zarrs_codec::plan::RangeDecoder;
use zarrs_storage::byte_range::{ByteRange, ByteRangeIterator, InvalidByteRangeError};

/// Decode byte ranges of a blosc encoded value.
///
/// Byte ranges aligned to the blosc typesize are decoded with `blosc_getitem`.
/// Otherwise, the entire value is decoded, as `blosc_getitem` can only retrieve whole items.
/// This is the case if the decoded size is not a multiple of the typesize (e.g. optional data).
fn blosc_partial_decode<'a>(
    encoded_value: &[u8],
    decoded_regions: ByteRangeIterator,
) -> Result<Vec<CowBytes<'a>>, CodecError> {
    let invalid = || CodecError::from("blosc encoded value is invalid");
    blosc_validate(encoded_value).ok_or_else(invalid)?;
    let nbytes = blosc_nbytes(encoded_value).ok_or_else(invalid)?;
    let typesize = blosc_typesize(encoded_value).ok_or_else(invalid)?;
    let blosc_error = |err: super::BloscError| CodecError::from(err.to_string());
    let mut decoded_value: Option<Vec<u8>> = None;
    decoded_regions
        .map(|byte_range| {
            let std::ops::Range { start, end } = byte_range
                .checked_range_usize(nbytes)
                .ok_or_else(|| InvalidByteRangeError::new(byte_range, nbytes as u64))?;
            if start == end {
                // `blosc_getitem` fails to retrieve zero items
                Ok(CowBytes::from(Vec::new()))
            } else if start.is_multiple_of(typesize) && end.is_multiple_of(typesize) {
                let decoded =
                    blosc_decompress_bytes_partial(encoded_value, start, end - start, typesize);
                decoded.map(CowBytes::from).map_err(blosc_error)
            } else {
                if decoded_value.is_none() {
                    decoded_value = Some(
                        blosc_decompress_bytes(encoded_value, nbytes, 1).map_err(blosc_error)?,
                    );
                }
                let decoded_value = decoded_value.as_ref().expect("decoded above");
                Ok(CowBytes::from(decoded_value[start..end].to_vec()))
            }
        })
        .collect()
}

/// Decodes byte ranges of a blosc encoded value with `blosc_getitem` where possible.
#[derive(Debug)]
pub(crate) struct BloscRangeDecoder;

impl RangeDecoder for BloscRangeDecoder {
    fn decoded_len(&self, encoded: &[u8]) -> Result<u64, CodecError> {
        let invalid = || CodecError::from("blosc encoded value is invalid");
        blosc_validate(encoded).ok_or_else(invalid)?;
        Ok(blosc_nbytes(encoded).ok_or_else(invalid)? as u64)
    }

    fn decode_ranges(
        &self,
        encoded: &[u8],
        ranges: &[ByteRange],
    ) -> Result<Vec<CowBytes<'static>>, CodecError> {
        blosc_partial_decode(encoded, Box::new(ranges.iter().copied()))
    }
}
