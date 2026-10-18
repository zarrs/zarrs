//! Bytes to bytes codecs.

#[cfg(feature = "adler32")]
pub mod adler32;
#[cfg(feature = "blosc")]
pub mod blosc;
#[cfg(feature = "bz2")]
pub mod bz2;
#[cfg(feature = "crc32c")]
pub mod crc32c;
#[cfg(feature = "fletcher32")]
pub mod fletcher32;
#[cfg(feature = "gdeflate")]
pub mod gdeflate;
#[cfg(feature = "gzip")]
pub mod gzip;
pub mod shuffle;
#[cfg(feature = "zlib")]
pub mod zlib;
#[cfg(feature = "zstd")]
pub mod zstd;

#[cfg(test)]
pub mod test_unbounded;

/// Check [`decode_into`](zarrs_codec::BytesToBytesCodecTraits::decode_into) of `codec`.
///
/// The encoded `decoded` bytes must decode into an output of their length, and an output `length_step` bytes longer
/// or shorter must be reported as an unexpected decoded size. Each of the `invalid` encoded values, made from the
/// encoded bytes (e.g. a truncated stream), must fail to decode into an output of the decoded length.
#[cfg(test)]
pub(crate) fn check_bytes_decode_into(
    codec: &dyn zarrs_codec::BytesToBytesCodecTraits,
    decoded: &[u8],
    length_step: usize,
    invalid: impl Fn(&[u8]) -> Vec<Vec<u8>>,
) {
    use zarrs_codec::{BytesRepresentation, CodecError, CodecOptions, ComputeResources, CowBytes};

    let options = CodecOptions::default();
    let resources = ComputeResources::default();
    let len = decoded.len();
    let encoded = codec
        .encode(CowBytes::Borrowed(decoded), &options, &resources)
        .unwrap();
    let decode_into = |encoded: CowBytes<'_>, output: &mut [u8]| {
        codec.decode_into(
            encoded,
            &BytesRepresentation::FixedSize(len as u64),
            output,
            &options,
            &resources,
        )
    };

    let mut output = vec![0; len];
    decode_into(encoded.clone(), &mut output).unwrap();
    assert_eq!(output, decoded);

    // An output of the wrong length is reported as an invalid length
    assert!(matches!(
        decode_into(encoded.clone(), &mut vec![0; len + length_step]),
        Err(CodecError::UnexpectedChunkDecodedSize(_))
    ));
    assert!(matches!(
        decode_into(encoded.clone(), &mut output[..len - length_step]),
        Err(CodecError::UnexpectedChunkDecodedSize(_))
    ));

    for invalid in invalid(&encoded) {
        assert!(decode_into(CowBytes::from(invalid), &mut output).is_err());
    }
}

/// Convert `bytes` into an owned [`Vec`] with at least `spare` bytes of spare capacity.
///
/// This lets a codec that appends or prepends a fixed-size prefix/suffix write those bytes
/// without a reallocation, whether or not the input is already owned.
#[cfg(any(feature = "adler32", feature = "crc32c", feature = "fletcher32"))]
pub(crate) fn into_owned_with_spare_capacity(
    bytes: crate::array::CowBytes<'_>,
    spare: usize,
) -> Vec<u8> {
    match bytes {
        // One allocation, sized for the suffix up front.
        crate::array::CowBytes::Borrowed(bytes) => {
            let mut owned = Vec::with_capacity(bytes.len().saturating_add(spare));
            owned.extend_from_slice(bytes);
            owned
        }
        bytes @ crate::array::CowBytes::Shared(_) => {
            // Reuses the buffer if it is not shared, otherwise copies.
            let mut bytes = bytes.into_vec();
            bytes.reserve_exact(spare);
            bytes
        }
    }
}
