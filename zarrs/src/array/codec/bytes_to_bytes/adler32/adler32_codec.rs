use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};

use super::{Adler32CodecConfiguration, Adler32CodecConfigurationV1, CHECKSUM_SIZE};
use crate::array::codec::bytes_to_bytes::into_owned_with_spare_capacity;
use crate::array::{BytesRepresentation, CowBytes};
use zarrs_codec::plan::SourceStep;
use zarrs_codec::{
    BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits, Resources,
};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::adler32::Adler32CodecConfigurationChecksumLocation;

/// A `adler32` codec implementation.
#[derive(Clone, Debug, Default)]
pub struct Adler32Codec {
    location: Adler32CodecConfigurationChecksumLocation,
}

impl Adler32Codec {
    /// Create a new `adler32` codec.
    #[must_use]
    pub const fn new(location: Adler32CodecConfigurationChecksumLocation) -> Self {
        Self { location }
    }

    /// Create a new `adler32` codec.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &Adler32CodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            Adler32CodecConfiguration::V1(configuration) => Ok(Self {
                location: configuration.location,
            }),
            _ => Err(PluginCreateError::Other(
                "this adler32 codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for Adler32Codec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = Adler32CodecConfiguration::V1(Adler32CodecConfigurationV1 {
            location: self.location,
        });
        Some(configuration.into())
    }
}

impl BytesToBytesCodecTraits for Adler32Codec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        decoded_value: CowBytes<'a>,
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let mut adler = simd_adler32::Adler32::new();
        adler.write(&decoded_value);
        let checksum = adler.finish().to_le_bytes();

        let encoded_value = match self.location {
            Adler32CodecConfigurationChecksumLocation::Start => {
                let mut encoded_value =
                    into_owned_with_spare_capacity(decoded_value, CHECKSUM_SIZE);
                let data_len = encoded_value.len();
                encoded_value.resize(data_len + CHECKSUM_SIZE, 0);
                encoded_value.copy_within(..data_len, CHECKSUM_SIZE);
                encoded_value[..CHECKSUM_SIZE].copy_from_slice(&checksum);
                encoded_value
            }
            Adler32CodecConfigurationChecksumLocation::End => {
                let mut encoded_value =
                    into_owned_with_spare_capacity(decoded_value, CHECKSUM_SIZE);
                encoded_value.extend_from_slice(&checksum);
                encoded_value
            }
        };
        Ok(CowBytes::from(encoded_value))
    }

    fn decode<'a>(
        &self,
        encoded_value: CowBytes<'a>,
        _decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<CowBytes<'a>, CodecError> {
        if encoded_value.len() >= CHECKSUM_SIZE {
            let (decoded_value, checksum) = match self.location {
                Adler32CodecConfigurationChecksumLocation::Start => {
                    let checksum: [u8; CHECKSUM_SIZE] =
                        encoded_value[..CHECKSUM_SIZE].try_into().unwrap();
                    let data_len = encoded_value.len() - CHECKSUM_SIZE;
                    (
                        encoded_value.slice(CHECKSUM_SIZE..CHECKSUM_SIZE + data_len),
                        checksum,
                    )
                }
                Adler32CodecConfigurationChecksumLocation::End => {
                    let checksum_start = encoded_value.len() - CHECKSUM_SIZE;
                    let checksum: [u8; CHECKSUM_SIZE] =
                        encoded_value[checksum_start..].try_into().unwrap();
                    (encoded_value.slice(0..checksum_start), checksum)
                }
            };

            if options.validate_checksums() {
                let mut adler = simd_adler32::Adler32::new();
                adler.write(&decoded_value);
                if adler.finish().to_le_bytes() != checksum {
                    return Err(CodecError::InvalidChecksum);
                }
            }

            Ok(decoded_value)
        } else {
            Err(CodecError::Other(
                "adler32 decoder expects a 32 bit input".to_string(),
            ))
        }
    }

    fn source_step(&self) -> SourceStep {
        let size = CHECKSUM_SIZE as u64;
        match self.location {
            Adler32CodecConfigurationChecksumLocation::Start => SourceStep::Window {
                prefix: size,
                suffix: 0,
            },
            Adler32CodecConfigurationChecksumLocation::End => SourceStep::Window {
                prefix: 0,
                suffix: size,
            },
        }
    }

    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        match decoded_representation {
            BytesRepresentation::FixedSize(size) => {
                BytesRepresentation::FixedSize(size + CHECKSUM_SIZE as u64)
            }
            BytesRepresentation::BoundedSize(size) => {
                BytesRepresentation::BoundedSize(size + CHECKSUM_SIZE as u64)
            }
            BytesRepresentation::UnboundedSize => BytesRepresentation::UnboundedSize,
        }
    }
}
