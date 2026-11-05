#[cfg(not(target_arch = "wasm32"))]
#[path = "blosc_codec_via_blosc_src.rs"]
mod blosc_codec_impl;

#[cfg(target_arch = "wasm32")]
#[path = "blosc_codec_via_blusc.rs"]
mod blosc_codec_impl;

pub use blosc_codec_impl::BloscCodec;

use std::mem::MaybeUninit;

use zarrs_codec::{
    BytesRepresentation, BytesToBytesCodecDecodeInto, CodecError, CodecOptions, ComputeResources,
    CowBytes, InvalidBytesLengthError,
};

use super::blosc_impl::{blosc_decompress_bytes_into, blosc_validate};

// SAFETY: `blosc_decompress_bytes_into` fails unless it initialises exactly `output.len()` bytes.
unsafe impl BytesToBytesCodecDecodeInto for BloscCodec {
    fn decode_into(
        &self,
        encoded_value: CowBytes<'_>,
        _decoded_representation: &BytesRepresentation,
        output: &mut [MaybeUninit<u8>],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        let decoded_len = blosc_validate(&encoded_value)
            .ok_or_else(|| CodecError::from("blosc encoded value is invalid"))?;
        if decoded_len != output.len() {
            return Err(InvalidBytesLengthError::new(decoded_len, output.len()).into());
        }
        let n_threads = 1;
        blosc_decompress_bytes_into(&encoded_value, output, n_threads)
            .map_err(|e| CodecError::from(e.to_string()))
    }
}
