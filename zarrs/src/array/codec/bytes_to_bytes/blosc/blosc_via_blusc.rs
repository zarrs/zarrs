/// The input length needed to to run `blosc_compress_bytes` in parallel,
/// and the output length needed to run `blosc_decompress_bytes` in parallel.
/// Otherwise, these functions will use one thread regardless of the `numinternalthreads` parameter.
const MIN_PARALLEL_LENGTH: usize = 4_000_000;
use std::sync::Arc;

pub use super::blosc_codec::BloscCodec;
use blusc::{
    BLOSC_MAX_THREADS,
    // For compression
    BLOSC2_CPARAMS_DEFAULTS,
    BLOSC2_DPARAMS_DEFAULTS,
    BLOSC2_MAX_OVERHEAD,
    // For decompression
    blosc1_cbuffer_metainfo,
    blosc1_cbuffer_sizes,
    blosc1_cbuffer_validate,
    blosc1_compress_ctx,
    blosc1_getitem,
    blosc2_create_cctx,
    blosc2_create_dctx,
    blosc2_decompress_ctx,
};
use derive_more::From;
use thiserror::Error;
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;

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
pub struct BloscError(String);

impl From<&str> for BloscError {
    fn from(err: &str) -> Self {
        Self(err.to_string())
    }
}

pub fn compressor_as_str(compressor: BloscCompressor) -> &'static str {
    match compressor {
        BloscCompressor::BloscLZ => blusc::BLOSC_BLOSCLZ_COMPNAME,
        BloscCompressor::LZ4 => blusc::BLOSC_LZ4_COMPNAME,
        BloscCompressor::LZ4HC => blusc::BLOSC_LZ4HC_COMPNAME,
        BloscCompressor::Snappy => blusc::BLOSC_SNAPPY_COMPNAME,
        BloscCompressor::Zlib => blusc::BLOSC_ZLIB_COMPNAME,
        BloscCompressor::Zstd => blusc::BLOSC_ZSTD_COMPNAME,
    }
}

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
    let destsize = src.len() + BLOSC2_MAX_OVERHEAD as usize;
    let mut dest: Vec<u8> = vec![0; destsize];
    let destsize = {
        let mut cparams = BLOSC2_CPARAMS_DEFAULTS;
        cparams.typesize = typesize as i32;
        cparams.clevel = clevel.into();
        cparams.nthreads = numinternalthreads as i16;
        cparams.blocksize = blocksize as i32;
        cparams.compcode = match compressor {
            BloscCompressor::BloscLZ => 0,
            BloscCompressor::LZ4 => 1,
            BloscCompressor::LZ4HC => 2,
            BloscCompressor::Snappy => 3,
            BloscCompressor::Zlib => 4,
            BloscCompressor::Zstd => 5,
        };
        cparams.filters[5] = match shuffle_mode {
            BloscShuffleMode::NoShuffle => 0,
            BloscShuffleMode::Shuffle => 1,
            BloscShuffleMode::BitShuffle => 2,
        };
        let context = blosc2_create_cctx(cparams);

        blosc1_compress_ctx(&context, src, &mut dest)
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

pub fn blosc_validate(src: &[u8]) -> Option<usize> {
    blosc1_cbuffer_validate(src, src.len()).ok()
}

/// # Safety
///
/// Validate first
pub fn blosc_typesize(src: &[u8]) -> Option<usize> {
    let (typesize, _flags) = blosc1_cbuffer_metainfo(src)?;
    (typesize != 0).then_some(typesize)
}

/// Returns the length of the uncompress bytes of a `blosc` buffer.
///
/// # Safety
///
/// Validate first
pub fn blosc_nbytes(src: &[u8]) -> Option<usize> {
    let (uncompressed_bytes, cbytes, blocksize) = blosc1_cbuffer_sizes(src);
    (uncompressed_bytes > 0 && cbytes > 0 && blocksize > 0).then_some(uncompressed_bytes)
}

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

    let mut dest: Vec<u8> = vec![0; destsize];
    let destsize = {
        let mut dparams = BLOSC2_DPARAMS_DEFAULTS;
        dparams.nthreads = numinternalthreads as i16;
        let context = blosc2_create_dctx(dparams);
        blosc2_decompress_ctx(&context, src, &mut dest)
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

pub fn blosc_decompress_bytes_partial(
    src: &[u8],
    offset: usize,
    length: usize,
    typesize: usize,
) -> Result<Vec<u8>, BloscError> {
    let start = i32::try_from(offset / typesize).unwrap();
    let nitems = i32::try_from(length / typesize).unwrap();
    let mut dest: Vec<u8> = vec![0; length];
    let destsize = blosc1_getitem(src, start, nitems, &mut dest);
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
