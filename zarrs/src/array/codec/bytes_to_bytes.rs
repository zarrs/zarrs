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

mod strip_prefix_partial_decoder;

mod strip_suffix_partial_decoder;

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

#[cfg(test)]
pub(crate) fn check_bytes_decode_into(
    codec: &dyn zarrs_codec::BytesToBytesCodecDecodeInto,
    decoded: &[u8],
    length_step: usize,
    invalid: impl Fn(&[u8]) -> Vec<Vec<u8>>,
) {
    use std::mem::MaybeUninit;
    use zarrs_codec::{BytesRepresentation, CodecError, CodecOptions, ComputeResources, CowBytes};

    let options = CodecOptions::default();
    let resources = ComputeResources::default();
    let len = decoded.len();
    assert!(codec.as_decode_into().is_some());
    let encoded = codec
        .encode(CowBytes::Borrowed(decoded), &options, &resources)
        .unwrap();
    let decode_into = |encoded: CowBytes<'_>, output: &mut [MaybeUninit<u8>]| {
        codec.decode_into(
            encoded,
            &BytesRepresentation::FixedSize(len as u64),
            output,
            &options,
            &resources,
        )
    };

    let mut output = vec![MaybeUninit::uninit(); len];
    decode_into(encoded.clone(), &mut output).unwrap();
    // SAFETY: a successful `decode_into` initialises the output.
    assert_eq!(
        unsafe { crate::slice_assume_init_mut(&mut output) },
        decoded
    );

    // An output of the wrong length is reported as an invalid length
    assert!(matches!(
        decode_into(
            encoded.clone(),
            &mut vec![MaybeUninit::uninit(); len + length_step]
        ),
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
