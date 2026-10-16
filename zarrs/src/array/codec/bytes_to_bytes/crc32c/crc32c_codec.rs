use std::sync::Arc;

use zarrs_metadata_ext::codec::crc32c::Crc32cCodecConfigurationLocation;
use zarrs_plugin::ZarrVersion;

use super::{CHECKSUM_SIZE, Crc32cCodecConfiguration, Crc32cCodecConfigurationV1};
use crate::array::codec::bytes_to_bytes::into_owned_with_spare_capacity;
use crate::array::{BytesRepresentation, CowBytes};
use zarrs_codec::plan::SourceStep;
use zarrs_codec::{
    BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, Resources,
};
use zarrs_metadata::Configuration;

/// A `crc32c` codec implementation.
#[derive(Clone, Debug, Default)]
pub struct Crc32cCodec(Crc32cCodecConfigurationLocation);

impl Crc32cCodec {
    /// Create a new `crc32c` codec.
    #[must_use]
    pub const fn new() -> Self {
        Self(Crc32cCodecConfigurationLocation::End)
    }

    /// Create a new `crc32c` codec.
    #[must_use]
    #[allow(clippy::wildcard_enum_match_arm)]
    pub const fn new_with_configuration(configuration: &Crc32cCodecConfiguration) -> Self {
        let location = match configuration {
            Crc32cCodecConfiguration::Numcodecs(cfg) => cfg.location,
            Crc32cCodecConfiguration::V1(_) | _ => Crc32cCodecConfigurationLocation::End,
        };
        Self(location)
    }
}

impl CodecTraits for Crc32cCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = Crc32cCodecConfiguration::V1(Crc32cCodecConfigurationV1 {});
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,   // TODO
            partial_decode: false, // TODO
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl BytesToBytesCodecTraits for Crc32cCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        decoded_value: CowBytes<'a>,
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let checksum = crc32c::crc32c(&decoded_value).to_le_bytes();
        let encoded_value = match self.0 {
            Crc32cCodecConfigurationLocation::End => {
                let mut encoded_value =
                    into_owned_with_spare_capacity(decoded_value, CHECKSUM_SIZE);
                encoded_value.extend_from_slice(&checksum);
                encoded_value
            }
            Crc32cCodecConfigurationLocation::Start => {
                let mut encoded_value =
                    into_owned_with_spare_capacity(decoded_value, CHECKSUM_SIZE);
                let data_len = encoded_value.len();
                encoded_value.resize(data_len + CHECKSUM_SIZE, 0);
                encoded_value.copy_within(..data_len, CHECKSUM_SIZE);
                encoded_value[..CHECKSUM_SIZE].copy_from_slice(&checksum);
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
            let data_len = encoded_value.len() - CHECKSUM_SIZE;
            let (data, checksum_stored): (&[u8], [u8; CHECKSUM_SIZE]) = match self.0 {
                Crc32cCodecConfigurationLocation::End => (
                    &encoded_value[..data_len],
                    encoded_value[data_len..].try_into().unwrap(),
                ),
                Crc32cCodecConfigurationLocation::Start => (
                    &encoded_value[CHECKSUM_SIZE..],
                    encoded_value[..CHECKSUM_SIZE].try_into().unwrap(),
                ),
            };

            if options.validate_checksums() {
                let checksum = crc32c::crc32c(data).to_le_bytes();
                if checksum != checksum_stored {
                    return Err(CodecError::InvalidChecksum);
                }
            }

            // Strip the checksum, which is free for borrowed and shared bytes.
            Ok(match self.0 {
                Crc32cCodecConfigurationLocation::End => encoded_value.slice(0..data_len),
                Crc32cCodecConfigurationLocation::Start => {
                    encoded_value.slice(CHECKSUM_SIZE..CHECKSUM_SIZE + data_len)
                }
            })
        } else {
            Err(CodecError::Other(
                "crc32c decoder expects a 32 bit input".to_string(),
            ))
        }
    }

    fn source_step(&self) -> SourceStep {
        let size = CHECKSUM_SIZE as u64;
        match self.0 {
            Crc32cCodecConfigurationLocation::End => SourceStep::Window {
                prefix: 0,
                suffix: size,
            },
            Crc32cCodecConfigurationLocation::Start => SourceStep::Window {
                prefix: size,
                suffix: 0,
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
