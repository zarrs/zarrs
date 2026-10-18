use std::borrow::Cow;
use std::sync::Arc;

use blusc::{BLOSC2_MAX_OVERHEAD, blosc2_get_complib_info};
use zarrs_metadata::Configuration;
use zarrs_plugin::ZarrVersion;

use super::super::blosc_impl::{blosc_decompress_bytes_into, compressor_as_str};
use super::super::{
    BloscCodecConfiguration, BloscCodecConfigurationNumcodecs, BloscCodecConfigurationV1,
    BloscCompressionLevel, BloscCompressor, BloscError, BloscShuffleMode,
    BloscShuffleModeNumcodecs, blosc_compress_bytes, blosc_decompress_bytes, blosc_range_decoder,
    blosc_validate,
};
use crate::array::{BytesRepresentation, CowBytes};
use zarrs_codec::{
    BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    ComputeResources, InvalidBytesLengthError,
};
use zarrs_plugin::PluginCreateError;

/// A `blosc` codec implementation.
#[derive(Clone, Debug)]
pub struct BloscCodec {
    cname: BloscCompressor,
    clevel: BloscCompressionLevel,
    blocksize: usize,
    shuffle_mode: Option<BloscShuffleMode>,
    typesize: Option<usize>,
}

impl BloscCodec {
    /// Create a new `blosc` codec.
    ///
    /// The block size is chosen automatically if `blocksize` is none or zero.
    /// `typesize` must be a positive integer if shuffling is enabled.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if
    ///  - the compressor is not supported, or
    ///  - `typesize` is [`None`] and shuffling is enabled.
    pub fn new(
        cname: BloscCompressor,
        clevel: BloscCompressionLevel,
        blocksize: Option<usize>,
        shuffle_mode: BloscShuffleMode,
        typesize: Option<usize>,
    ) -> Result<Self, PluginCreateError> {
        if shuffle_mode != BloscShuffleMode::NoShuffle
            && (typesize.is_none() || typesize == Some(0))
        {
            return Err(PluginCreateError::from(
                "typesize is a positive integer required if shuffling is enabled.",
            ));
        }

        // Check that the compressor is available
        let support = blosc2_get_complib_info(compressor_as_str(cname))
            .map(|(_, _, support)| support)
            .unwrap_or(-1);
        if support < 0 {
            return Err(PluginCreateError::from(format!(
                "compressor {cname:?} is not supported."
            )));
        }

        Ok(Self {
            cname,
            clevel,
            blocksize: blocksize.unwrap_or_default(),
            shuffle_mode: Some(shuffle_mode),
            typesize,
        })
    }

    /// Create a new `blosc` codec from configuration.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &BloscCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            BloscCodecConfiguration::V1(configuration) => Self::new(
                configuration.cname,
                configuration.clevel,
                Some(configuration.blocksize),
                configuration.shuffle,
                configuration.typesize,
            ),
            BloscCodecConfiguration::Numcodecs(_) => {
                // Note: this situation is avoided with codec_metadata_v2_to_v3
                Err(PluginCreateError::Other(
                    "the blosc codec cannot be created from numcodecs.blosc metadata directly"
                        .to_string(),
                ))?
            }
            _ => Err(PluginCreateError::Other(
                "this blosc codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    fn do_encode(&self, decoded_value: &[u8], n_threads: usize) -> Result<Vec<u8>, CodecError> {
        let typesize = self.typesize.unwrap_or_default();
        blosc_compress_bytes(
            decoded_value,
            self.clevel,
            self.shuffle_mode.unwrap_or(if typesize > 0 {
                BloscShuffleMode::BitShuffle
            } else {
                BloscShuffleMode::NoShuffle
            }),
            typesize,
            self.cname,
            self.blocksize,
            n_threads,
        )
        .map_err(|err: BloscError| CodecError::Other(err.to_string()))
    }

    fn do_decode(encoded_value: &[u8], n_threads: usize) -> Result<Vec<u8>, CodecError> {
        blosc_validate(encoded_value).map_or_else(
            || Err(CodecError::from("blosc encoded value is invalid")),
            |destsize| {
                blosc_decompress_bytes(encoded_value, destsize, n_threads)
                    .map_err(|e| CodecError::from(e.to_string()))
            },
        )
    }

    fn do_decode_into(
        encoded_value: &[u8],
        output: &mut [u8],
        n_threads: usize,
    ) -> Result<(), CodecError> {
        let decoded_len = blosc_validate(encoded_value)
            .ok_or_else(|| CodecError::from("blosc encoded value is invalid"))?;
        if decoded_len != output.len() {
            return Err(InvalidBytesLengthError::new(decoded_len, output.len()).into());
        }
        blosc_decompress_bytes_into(encoded_value, output, n_threads)
            .map_err(|e| CodecError::from(e.to_string()))
    }
}

impl CodecTraits for BloscCodec {
    fn configuration(
        &self,
        version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let shuffle = self.shuffle_mode.unwrap_or_else(|| {
            if self.typesize.unwrap_or_default() > 0 {
                BloscShuffleMode::BitShuffle
            } else {
                BloscShuffleMode::NoShuffle
            }
        });
        match version {
            ZarrVersion::V2 => {
                let shuffle = match shuffle {
                    BloscShuffleMode::NoShuffle => BloscShuffleModeNumcodecs::NoShuffle,
                    BloscShuffleMode::Shuffle => BloscShuffleModeNumcodecs::Shuffle,
                    BloscShuffleMode::BitShuffle => BloscShuffleModeNumcodecs::BitShuffle,
                };
                let configuration =
                    BloscCodecConfiguration::Numcodecs(BloscCodecConfigurationNumcodecs {
                        cname: self.cname,
                        clevel: self.clevel,
                        shuffle,
                        blocksize: self.blocksize,
                    });
                Some(configuration.into())
            }
            ZarrVersion::V3 => {
                let configuration = BloscCodecConfiguration::V1(BloscCodecConfigurationV1 {
                    cname: self.cname,
                    clevel: self.clevel,
                    shuffle,
                    typesize: self.typesize,
                    blocksize: self.blocksize,
                });
                Some(configuration.into())
            }
        }
    }
}

impl BytesToBytesCodecTraits for BloscCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        decoded_value: CowBytes<'a>,
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        // let n_threads = std::cmp::min(
        //     options.concurrent_limit(),
        //     std::thread::available_parallelism().unwrap(),
        // )
        // .get();
        let n_threads = 1;
        Ok(CowBytes::from(self.do_encode(&decoded_value, n_threads)?))
    }

    fn decode<'a>(
        &self,
        encoded_value: CowBytes<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        // let n_threads = std::cmp::min(
        //     options.concurrent_limit(),
        //     std::thread::available_parallelism().unwrap(),
        // )
        // .get();
        let n_threads = 1;
        Ok(CowBytes::from(Self::do_decode(&encoded_value, n_threads)?))
    }

    fn is_decode_into_efficient(&self) -> bool {
        true
    }

    fn decode_into(
        &self,
        encoded_value: CowBytes<'_>,
        _decoded_representation: &BytesRepresentation,
        output: &mut [u8],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        let n_threads = 1;
        Self::do_decode_into(&encoded_value, output, n_threads)
    }

    fn source_step(&self) -> zarrs_codec::plan::SourceStep {
        zarrs_codec::plan::SourceStep::RangeDecode(Arc::new(blosc_range_decoder::BloscRangeDecoder))
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        decoded_representation
            .size()
            .map_or(BytesRepresentation::UnboundedSize, |size| {
                BytesRepresentation::BoundedSize(size + u64::from(BLOSC2_MAX_OVERHEAD as u32))
            })
    }
}
