/// The input length needed to to run `blosc_compress_bytes` in parallel,
/// and the output length needed to run `blosc_decompress_bytes` in parallel.
/// Otherwise, these functions will use one thread regardless of the `numinternalthreads` parameter.
const MIN_PARALLEL_LENGTH: usize = 4_000_000;

use std::ffi::{c_char, c_int, c_void};
use std::sync::Arc;

use blosc_src::{
    BLOSC_MAX_OVERHEAD, BLOSC_MAX_THREADS, blosc_cbuffer_metainfo, blosc_cbuffer_sizes,
    blosc_cbuffer_validate, blosc_compress_ctx, blosc_decompress_ctx, blosc_getitem,
};
use derive_more::From;
use thiserror::Error;
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;

use super::blosc_codec::BloscCodec;
use zarrs_codec::{Codec, CodecTraitsV2, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::blosc::{
    BloscCodecConfiguration, BloscCodecConfigurationNumcodecs, BloscCodecConfigurationV1,
    BloscCompressionLevel, BloscCompressor, BloscShuffleMode, BloscShuffleModeNumcodecs,
};
use zarrs_plugin::PluginCreateError;

impl CodecTraitsV3 for BloscCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
        let configuration: BloscCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(BloscCodec::new_with_configuration(&configuration)?);
        Ok(Codec::BytesToBytes(codec))
    }
}

impl CodecTraitsV2 for BloscCodec {
    fn create(metadata: &MetadataV2) -> Result<Codec, PluginCreateError> {
        let configuration: BloscCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(BloscCodec::new_with_configuration(&configuration)?);
        Ok(Codec::BytesToBytes(codec))
    }
}

#[derive(Clone, Debug, Error, From)]
#[error("{0}")]
/// An error from the blosc codec.
pub struct BloscError(String);

impl From<&str> for BloscError {
    fn from(err: &str) -> Self {
        Self(err.to_string())
    }
}

pub(super) const fn compressor_as_cstr(compressor: BloscCompressor) -> *const u8 {
    match compressor {
        BloscCompressor::BloscLZ => blosc_src::BLOSC_BLOSCLZ_COMPNAME.as_ptr(),
        BloscCompressor::LZ4 => blosc_src::BLOSC_LZ4_COMPNAME.as_ptr(),
        BloscCompressor::LZ4HC => blosc_src::BLOSC_LZ4HC_COMPNAME.as_ptr(),
        BloscCompressor::Snappy => blosc_src::BLOSC_SNAPPY_COMPNAME.as_ptr(),
        BloscCompressor::Zlib => blosc_src::BLOSC_ZLIB_COMPNAME.as_ptr(),
        BloscCompressor::Zstd => blosc_src::BLOSC_ZSTD_COMPNAME.as_ptr(),
    }
}

/// Compress bytes using blosc.
pub fn blosc_compress_bytes(
    src: &[u8],
    clevel: BloscCompressionLevel,
    shuffle_mode: BloscShuffleMode,
    typesize: usize,
    compressor: BloscCompressor,
    blocksize: usize,
    numinternalthreads: usize,
) -> Result<Vec<u8>, BloscError> {
    let numinternalthreads = if src.len() >= MIN_PARALLEL_LENGTH {
        std::cmp::min(numinternalthreads, BLOSC_MAX_THREADS as usize)
    } else {
        1
    };

    // let mut dest = vec![0; src.len() + BLOSC_MAX_OVERHEAD as usize];
    let destsize = src.len() + BLOSC_MAX_OVERHEAD as usize;
    let mut dest: Vec<u8> = Vec::with_capacity(destsize);
    let destsize = unsafe {
        let clevel: u8 = clevel.into();
        blosc_compress_ctx(
            c_int::from(clevel),
            shuffle_mode as c_int,
            std::cmp::max(1, typesize), // 0 is an error, even with noshuffle?
            src.len(),
            src.as_ptr().cast::<c_void>(),
            dest.as_mut_ptr().cast::<c_void>(),
            destsize,
            compressor_as_cstr(compressor).cast::<c_char>(),
            blocksize,
            i32::try_from(numinternalthreads).unwrap(),
        )
    };
    if destsize > 0 {
        unsafe {
            #[allow(clippy::cast_sign_loss)]
            dest.set_len(destsize as usize);
        }
        Ok(dest)
    } else {
        let clevel: u8 = clevel.into();
        Err(BloscError::from(format!(
            "blosc_compress_ctx(clevel: {}, doshuffle: {shuffle_mode:?}, typesize: {typesize}, nbytes: {}, destsize {destsize}, compressor {compressor:?}, bloscksize: {blocksize}) -> {destsize} (failure)",
            clevel,
            src.len()
        )))
    }
}

/// Validate a blosc buffer and return the decompressed size, or `None` if invalid.
pub fn blosc_validate(src: &[u8]) -> Option<usize> {
    let mut destsize: usize = 0;
    let valid = unsafe {
        blosc_cbuffer_validate(src.as_ptr().cast::<c_void>(), src.len(), &raw mut destsize)
    } == 0;
    valid.then_some(destsize)
}

/// # Safety
///
/// Validate first
pub fn blosc_typesize(src: &[u8]) -> Option<usize> {
    let mut typesize: usize = 0;
    let mut flags: i32 = 0;
    unsafe {
        blosc_cbuffer_metainfo(
            src.as_ptr().cast::<c_void>(),
            &raw mut typesize,
            &raw mut flags,
        );
    };
    (typesize != 0).then_some(typesize)
}

/// Returns the length of the uncompress bytes of a `blosc` buffer.
///
/// # Safety
///
/// Validate first
pub fn blosc_nbytes(src: &[u8]) -> Option<usize> {
    let mut uncompressed_bytes: usize = 0;
    let mut cbytes: usize = 0;
    let mut blocksize: usize = 0;
    unsafe {
        blosc_cbuffer_sizes(
            src.as_ptr().cast::<c_void>(),
            &raw mut uncompressed_bytes,
            &raw mut cbytes,
            &raw mut blocksize,
        );
    };
    (uncompressed_bytes > 0 && cbytes > 0 && blocksize > 0).then_some(uncompressed_bytes)
}

/// Decompress a blosc buffer.
pub fn blosc_decompress_bytes(
    src: &[u8],
    destsize: usize,
    numinternalthreads: usize,
) -> Result<Vec<u8>, BloscError> {
    let numinternalthreads = if destsize >= MIN_PARALLEL_LENGTH {
        std::cmp::min(numinternalthreads, BLOSC_MAX_THREADS as usize)
    } else {
        1
    };

    let mut dest: Vec<u8> = Vec::with_capacity(destsize);
    let destsize = unsafe {
        blosc_decompress_ctx(
            src.as_ptr().cast::<c_void>(),
            dest.as_mut_ptr().cast::<c_void>(),
            destsize,
            i32::try_from(numinternalthreads).unwrap(),
        )
    };
    if destsize > 0 {
        unsafe {
            #[allow(clippy::cast_sign_loss)]
            dest.set_len(destsize as usize);
        }
        Ok(dest)
    } else {
        Err(BloscError::from("blosc_decompress_ctx failed"))
    }
}

/// Partially decompress a blosc buffer at a given byte offset and length.
pub fn blosc_decompress_bytes_partial(
    src: &[u8],
    offset: usize,
    length: usize,
    typesize: usize,
) -> Result<Vec<u8>, BloscError> {
    let start = i32::try_from(offset / typesize).unwrap();
    let nitems = i32::try_from(length / typesize).unwrap();
    let mut dest: Vec<u8> = Vec::with_capacity(length);
    let destsize = unsafe {
        blosc_getitem(
            src.as_ptr().cast::<c_void>(),
            start,
            nitems,
            dest.as_mut_ptr().cast::<c_void>(),
        )
    };
    if destsize <= 0 {
        Err(BloscError::from(format!(
            "blosc_getitem(src: len {}, start: {start}, nitems: {nitems}) -> {destsize} (failure)",
            src.len()
        )))
    } else {
        unsafe {
            #[allow(clippy::cast_sign_loss)]
            dest.set_len(destsize as usize);
        }
        Ok(dest)
    }
}
