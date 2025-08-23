#![allow(clippy::similar_names)]

// FIXME: This codec was really hacked together.
// It can probably be written much cleaner and with simpler logic.

use std::{borrow::Cow, sync::Arc};

use num::Integer;
use zarrs_metadata::{Configuration, Endianness};
use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;
use zarrs_plugin::PluginCreateError;
use zarrs_registry::codec::PACKBITS;

use crate::array::{
    codec::{
        array_to_bytes::{bytes::BytesPartialDecoder, packbits::div_rem_8bit},
        ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, BytesCodec,
        BytesPartialDecoderTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
        InvalidBytesLengthError, RecommendedConcurrency,
    },
    ArrayBytes, BytesRepresentation, ChunkRepresentation, RawBytes,
};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};

#[cfg(feature = "async")]
use super::packbits_partial_decoder::AsyncPackBitsPartialDecoder;

#[cfg(feature = "async")]
use crate::array::codec::array_to_bytes::bytes::AsyncBytesPartialDecoder;

use super::{
    pack_bits_components, packbits_partial_decoder::PackBitsPartialDecoder,
    DataTypeExtensionPackBitsCodecComponents, PackBitsCodecConfiguration,
    PackBitsCodecConfigurationV1,
};

/// A `packbits` codec implementation.
#[derive(Debug, Clone)]
pub struct PackBitsCodec {
    padding_encoding: PackBitsPaddingEncoding,
    first_bit: Option<u64>,
    last_bit: Option<u64>,
}

impl Default for PackBitsCodec {
    fn default() -> Self {
        Self::new(PackBitsPaddingEncoding::default(), None, None)
            .expect("this configuration is supported")
    }
}

fn padding_bits(num_elements: u64, element_size_bits: u64) -> u8 {
    let rem = ((num_elements * element_size_bits) % 8) as u8;
    if rem == 0 {
        0
    } else {
        8 - rem
    }
}

impl PackBitsCodec {
    /// Create a new `packbits` codec.
    ///
    /// # Errors
    /// Returns an error if the parameters are invalid or unsupported.
    /// `last_bit` must not be less than `first_bit`.
    pub fn new(
        padding_encoding: PackBitsPaddingEncoding,
        first_bit: Option<u64>,
        last_bit: Option<u64>,
    ) -> Result<Self, PluginCreateError> {
        if let (Some(first_bit), Some(last_bit)) = (first_bit, last_bit) {
            if last_bit < first_bit {
                return Err(PluginCreateError::from(
                    "packbits codec `last_bit` is less than `first_bit`",
                ));
            }
        }

        Ok(Self {
            padding_encoding,
            first_bit,
            last_bit,
        })
    }

    /// Create a new `packbits` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &PackBitsCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            PackBitsCodecConfiguration::V1(configuration) => Self::new(
                configuration.padding_encoding.unwrap_or_default(),
                configuration.first_bit,
                configuration.last_bit,
            ),
            _ => Err(PluginCreateError::Other(
                "this packbits codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for PackBitsCodec {
    fn identifier(&self) -> &str {
        PACKBITS
    }

    fn configuration_opt(
        &self,
        _name: &str,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = PackBitsCodecConfiguration::V1(PackBitsCodecConfigurationV1 {
            padding_encoding: Some(self.padding_encoding),
            first_bit: self.first_bit,
            last_bit: self.last_bit,
        });
        Some(configuration.into())
    }

    fn partial_decoder_should_cache_input(&self) -> bool {
        false
    }

    fn partial_decoder_decodes_all(&self) -> bool {
        false
    }
}

impl ArrayCodecTraits for PackBitsCodec {
    fn recommended_concurrency(
        &self,
        _decoded_representation: &ChunkRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for PackBitsCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError> {
        let DataTypeExtensionPackBitsCodecComponents {
            component_size_bits,
            num_components,
            sign_extension: _,
        } = pack_bits_components(decoded_representation.data_type())?;
        let first_bit = self.first_bit.unwrap_or(0);
        let last_bit = self.last_bit.unwrap_or(component_size_bits - 1);

        // Bytes codec fast path
        if component_size_bits % 8 == 0 && first_bit == 0 && last_bit == component_size_bits - 1 {
            // Data types are expected to support the bytes codec if their component size in bits is a multiple of 8.
            return BytesCodec::new(Some(Endianness::Little)).encode(
                bytes.clone(),
                decoded_representation,
                options,
            );
        }

        // Get the component and element size in bits
        let num_elements = decoded_representation.num_elements();
        let component_size_bits_extracted = last_bit - first_bit + 1;
        let element_size_bits = component_size_bits_extracted * num_components;
        let elements_size_bytes =
            usize::try_from((num_elements * element_size_bits).div_ceil(8)).unwrap();

        // Input checks
        let bytes = bytes.into_fixed()?;
        let data_type_size_dec =
            decoded_representation
                .data_type()
                .fixed_size()
                .ok_or_else(|| {
                    CodecError::Other(
                        "data type must have a fixed size for the packbits codec".to_string(),
                    )
                })?;
        if bytes.len() as u64 != num_elements * data_type_size_dec as u64 {
            return Err(InvalidBytesLengthError::new(
                bytes.len(),
                usize::try_from(num_elements * data_type_size_dec as u64).unwrap(),
            )
            .into());
        }

        // Allocate the output
        let padding_encoding_byte = match self.padding_encoding {
            PackBitsPaddingEncoding::None => 0,
            PackBitsPaddingEncoding::FirstByte | PackBitsPaddingEncoding::LastByte => 1,
        };
        let mut bytes_enc = vec![0u8; elements_size_bytes + padding_encoding_byte];

        // Set the padding encoding byte and grab the element bytes
        let padding_bits = padding_bits(num_elements, element_size_bits);
        let packed_elements = match self.padding_encoding {
            PackBitsPaddingEncoding::None => &mut bytes_enc[..],
            PackBitsPaddingEncoding::FirstByte => {
                bytes_enc[0] = padding_bits;
                &mut bytes_enc[1..]
            }
            PackBitsPaddingEncoding::LastByte => {
                bytes_enc[elements_size_bytes] = padding_bits;
                &mut bytes_enc[..elements_size_bytes]
            }
        };

        // Encode the components
        for component_idx in 0..num_elements * num_components {
            let bit_dec0 = component_idx * component_size_bits + first_bit;
            let bit_enc0 = component_idx * component_size_bits_extracted;
            for bit in 0..component_size_bits_extracted {
                let (byte_enc, bit_enc) = (bit_enc0 + bit).div_rem(&8);
                let (byte_dec, bit_dec) = div_rem_8bit(bit_dec0 + bit, component_size_bits);
                packed_elements[usize::try_from(byte_enc).unwrap()] |=
                    ((bytes[usize::try_from(byte_dec).unwrap()] >> (bit_dec % 8)) & 0b1) << bit_enc;
            }
        }

        Ok(RawBytes::from(bytes_enc))
    }

    fn decode<'a>(
        &self,
        bytes: RawBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let DataTypeExtensionPackBitsCodecComponents {
            component_size_bits,
            num_components,
            sign_extension,
        } = pack_bits_components(decoded_representation.data_type())?;
        let first_bit = self.first_bit.unwrap_or(0);
        let last_bit = self.last_bit.unwrap_or(component_size_bits - 1);

        // Bytes codec fast path
        if component_size_bits % 8 == 0 && first_bit == 0 && last_bit == component_size_bits - 1 {
            // Data types are expected to support the bytes codec if their element size in bits is a multiple of 8.
            return BytesCodec::new(Some(Endianness::Little)).decode(
                bytes.clone(),
                decoded_representation,
                options,
            );
        }

        // Get the component and element size in bits
        let num_elements = decoded_representation.num_elements();
        let component_size_bits_extracted = last_bit - first_bit + 1;
        let element_size_bits = component_size_bits_extracted * num_components;
        let elements_size_bytes =
            usize::try_from((num_elements * element_size_bits).div_ceil(8)).unwrap();

        // Input checks
        let data_type_size_dec =
            decoded_representation
                .data_type()
                .fixed_size()
                .ok_or_else(|| {
                    CodecError::Other(
                        "data type must have a fixed size for packbits codec".to_string(),
                    )
                })?;
        let expected_length = elements_size_bytes
            + match self.padding_encoding {
                PackBitsPaddingEncoding::None => 0,
                PackBitsPaddingEncoding::FirstByte | PackBitsPaddingEncoding::LastByte => 1,
            };
        if bytes.len() != expected_length {
            return Err(InvalidBytesLengthError::new(bytes.len(), expected_length).into());
        }

        let padding_bits = padding_bits(num_elements, element_size_bits);
        let packed_elements = match self.padding_encoding {
            PackBitsPaddingEncoding::None => &bytes[..],
            PackBitsPaddingEncoding::FirstByte => {
                if bytes[0] != padding_bits {
                    return Err(CodecError::Other(
                        "the packbits padding encoding start byte is incorrect".to_string(),
                    ));
                }
                &bytes[1..]
            }
            PackBitsPaddingEncoding::LastByte => {
                if bytes[elements_size_bytes] != padding_bits {
                    return Err(CodecError::Other(
                        "the packbits padding encoding last byte is incorrect".to_string(),
                    ));
                }
                &bytes[..elements_size_bytes]
            }
        };

        // Allocate the output
        let mut bytes_dec =
            vec![0u8; usize::try_from(num_elements * data_type_size_dec as u64).unwrap()];

        // Decode the components
        for component_idx in 0..num_elements * num_components {
            let bit_dec0 = component_idx * component_size_bits + first_bit;
            let bit_enc0 = component_idx * component_size_bits_extracted;
            for bit in 0..component_size_bits_extracted {
                let (byte_enc, bit_enc) = (bit_enc0 + bit).div_rem(&8);
                let (byte_dec, bit_dec) = div_rem_8bit(bit_dec0 + bit, component_size_bits);
                bytes_dec[usize::try_from(byte_dec).unwrap()] |=
                    ((packed_elements[usize::try_from(byte_enc).unwrap()] >> bit_enc) & 0b1)
                        << bit_dec;
            }
            if sign_extension {
                let signed: bool = {
                    let bit_enc0 = component_idx * component_size_bits_extracted;
                    let (byte_enc, bit_enc) =
                        (bit_enc0 + component_size_bits_extracted.saturating_sub(1)).div_rem(&8);
                    ((packed_elements[usize::try_from(byte_enc).unwrap()] >> bit_enc) & 0b1) == 1
                };
                if signed {
                    let (byte_dec, bit_dec) = div_rem_8bit(
                        bit_dec0 + component_size_bits_extracted.saturating_sub(1),
                        component_size_bits,
                    );
                    // Sign-extend to all remaining bits in the byte
                    // This differs from the spec which says sign extend to N (component_size_bits) bits
                    // This makes it just work with int4 / int2 -> int8
                    for bit_dec in bit_dec + 1..8 {
                        bytes_dec[usize::try_from(byte_dec).unwrap()] |= 1 << bit_dec;
                    }
                }
            }
        }

        Ok(ArrayBytes::Fixed(Cow::Owned(bytes_dec)))
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        let DataTypeExtensionPackBitsCodecComponents {
            component_size_bits,
            num_components: _,
            sign_extension: _,
        } = pack_bits_components(decoded_representation.data_type())?;
        let first_bit = self.first_bit.unwrap_or(0);
        let last_bit = self.last_bit.unwrap_or(component_size_bits - 1);

        // Bytes codec fast path
        if component_size_bits % 8 == 0 && first_bit == 0 && last_bit == component_size_bits - 1 {
            // Data types are expected to support the bytes codec if their element size in bits is a multiple of 8.
            Ok(Arc::new(BytesPartialDecoder::new(
                input_handle,
                decoded_representation.clone(),
                Some(Endianness::Little),
            )))
        } else {
            Ok(Arc::new(PackBitsPartialDecoder::new(
                input_handle,
                decoded_representation.clone(),
                self.padding_encoding,
                self.first_bit,
                self.last_bit,
            )))
        }
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        let DataTypeExtensionPackBitsCodecComponents {
            component_size_bits,
            num_components: _,
            sign_extension: _,
        } = pack_bits_components(decoded_representation.data_type())?;
        let first_bit = self.first_bit.unwrap_or(0);
        let last_bit = self.last_bit.unwrap_or(component_size_bits - 1);

        // Bytes codec fast path
        if component_size_bits % 8 == 0 && first_bit == 0 && last_bit == component_size_bits - 1 {
            // Data types are expected to support the bytes codec if their element size in bits is a multiple of 8.
            Ok(Arc::new(AsyncBytesPartialDecoder::new(
                input_handle,
                decoded_representation.clone(),
                Some(Endianness::Little),
            )))
        } else {
            Ok(Arc::new(AsyncPackBitsPartialDecoder::new(
                input_handle,
                decoded_representation.clone(),
                self.padding_encoding,
                self.first_bit,
                self.last_bit,
            )))
        }
    }

    fn encoded_representation(
        &self,
        decoded_representation: &ChunkRepresentation,
    ) -> Result<BytesRepresentation, CodecError> {
        let DataTypeExtensionPackBitsCodecComponents {
            component_size_bits,
            num_components,
            sign_extension: _,
        } = pack_bits_components(decoded_representation.data_type())?;
        let first_bit = self.first_bit.unwrap_or(0);
        let last_bit = self.last_bit.unwrap_or(component_size_bits - 1);

        let num_elements = decoded_representation.num_elements();
        let component_size_bits_extracted = last_bit - first_bit + 1;
        let element_size_bits = component_size_bits_extracted * num_components;
        let elements_size_bytes = (num_elements * element_size_bits).div_ceil(8);

        let padding_encoding_byte = match self.padding_encoding {
            PackBitsPaddingEncoding::None => 0,
            PackBitsPaddingEncoding::FirstByte | PackBitsPaddingEncoding::LastByte => 1,
        };
        Ok(BytesRepresentation::FixedSize(
            elements_size_bytes + padding_encoding_byte,
        ))
    }
}
