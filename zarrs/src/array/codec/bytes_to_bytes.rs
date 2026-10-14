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
    bytes: crate::array::ArrayBytesRaw<'_>,
    spare: usize,
) -> Vec<u8> {
    match bytes {
        std::borrow::Cow::Owned(mut bytes) => {
            bytes.reserve_exact(spare);
            bytes
        }
        std::borrow::Cow::Borrowed(bytes) => {
            let mut owned = Vec::with_capacity(bytes.len().saturating_add(spare));
            owned.extend_from_slice(bytes);
            owned
        }
    }
}
