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

/// The progress of one call of a streaming decompressor.
#[cfg(any(feature = "bz2", feature = "zlib"))]
pub(crate) struct DecompressProgress {
    /// The number of input bytes consumed.
    pub(crate) consumed: usize,
    /// The number of output bytes initialised.
    pub(crate) produced: usize,
    /// Whether the end of the stream was reached.
    pub(crate) stream_end: bool,
}

/// Decompress the stream `input` into `output`, which may be uninitialised, with a streaming decompressor.
///
/// `decompress` decompresses a prefix of its input into a prefix of its output, initialising the first [`DecompressProgress::produced`] bytes of its output.
/// It stops at the first decoded byte past the end of `output` rather than decoding the remainder, so a stream that is much larger than `output` is not expanded.
/// Bytes after the end of the stream are ignored.
///
/// Every byte of `output` is initialised on success.
///
/// # Errors
/// Returns an [`InvalidBytesLengthError`](zarrs_codec::InvalidBytesLengthError) if the length of the decoded bytes is not the length of `output`, an IO error if the stream is incomplete, or the first error of `decompress`.
/// If the decoded bytes are too long, the reported length is one more than the length of `output`.
#[cfg(any(feature = "bz2", feature = "zlib"))]
pub(crate) fn decompress_stream_into(
    input: &[u8],
    output: &mut [std::mem::MaybeUninit<u8>],
    mut decompress: impl FnMut(
        &[u8],
        &mut [std::mem::MaybeUninit<u8>],
    ) -> Result<DecompressProgress, zarrs_codec::CodecError>,
) -> Result<(), zarrs_codec::CodecError> {
    use zarrs_codec::InvalidBytesLengthError;

    let len = output.len();
    let (mut consumed, mut produced) = (0, 0);
    loop {
        let progress = if produced < len {
            decompress(&input[consumed..], &mut output[produced..])?
        } else {
            // The output is full, so only the end of the stream or an extra byte remains
            decompress(&input[consumed..], &mut [std::mem::MaybeUninit::uninit()])?
        };
        consumed += progress.consumed;
        produced += progress.produced;
        if produced > len {
            return Err(InvalidBytesLengthError::new(len.saturating_add(1), len).into());
        } else if progress.stream_end {
            return if produced == len {
                Ok(())
            } else {
                Err(InvalidBytesLengthError::new(produced, len).into())
            };
        } else if progress.consumed == 0 && progress.produced == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "the compressed stream is incomplete",
            )
            .into());
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

/// Check that `codec` reports a stream that decodes to far more than the output as a length error, and ignores bytes after the end of a stream.
#[cfg(all(test, any(feature = "bz2", feature = "gzip", feature = "zlib")))]
pub(crate) fn check_stream_decode_into_bounds(
    codec: &dyn zarrs_codec::BytesToBytesCodecDecodeInto,
) {
    use std::mem::MaybeUninit;
    use zarrs_codec::{BytesRepresentation, CodecError, CodecOptions, ComputeResources, CowBytes};

    let options = CodecOptions::default();
    let resources = ComputeResources::default();
    let decode_into = |encoded: &[u8], output: &mut [MaybeUninit<u8>]| {
        codec.decode_into(
            CowBytes::from(encoded),
            &BytesRepresentation::FixedSize(output.len() as u64),
            output,
            &options,
            &resources,
        )
    };

    // A stream that decodes to 8 MiB is rejected for an 8 byte output
    let oversized = codec
        .encode(CowBytes::from(vec![0u8; 8 << 20]), &options, &resources)
        .unwrap();
    assert!(matches!(
        decode_into(&oversized, &mut [MaybeUninit::uninit(); 8]),
        Err(CodecError::UnexpectedChunkDecodedSize(_))
    ));

    // Bytes after the end of the stream are ignored
    let decoded = b"bytes after the end of the stream".repeat(4);
    let mut encoded = codec
        .encode(CowBytes::from(decoded.as_slice()), &options, &resources)
        .unwrap()
        .into_vec();
    encoded.extend_from_slice(b"trailing");
    let mut output = vec![MaybeUninit::uninit(); decoded.len()];
    decode_into(&encoded, &mut output).unwrap();
    // SAFETY: a successful `decode_into` initialises the output.
    assert_eq!(
        unsafe { crate::slice_assume_init_mut(&mut output) },
        decoded
    );
}
