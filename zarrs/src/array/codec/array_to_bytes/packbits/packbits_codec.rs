#![allow(clippy::similar_names)]

// FIXME: This codec was really hacked together.
// It can probably be written much cleaner and with simpler logic.

use std::sync::Arc;

use num::Integer;
use zarrs_plugin::{PluginCreateError, ZarrVersion};

#[cfg(feature = "async")]
use super::packbits_partial_decoder::AsyncPackBitsPartialDecoder;
use super::packbits_partial_decoder::PackBitsPartialDecoder;
use super::{
    PackBitsCodecComponents, PackBitsCodecConfiguration, PackBitsCodecConfigurationV1,
    pack_bits_components,
};
use crate::array::codec::BytesCodec;
use crate::array::codec::array_to_bytes::bytes::BytesCodecPartial;
use crate::array::codec::array_to_bytes::packbits::div_rem_8bit;
use crate::array::{ArrayBytes, BytesRepresentation, CowBytes, DataType, FillValue};
use std::num::NonZeroU64;
use zarrs_codec::{
    ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits,
    BytesPartialDecoderTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecSpecificOptions, CodecTraits, InvalidBytesLengthError, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency, UnboundArrayToBytesCodecTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};
use zarrs_metadata::{Configuration, Endianness};
use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;

/// A `packbits` codec implementation.
#[derive(Debug, Clone)]
pub struct PackBitsCodec {
    padding_encoding: PackBitsPaddingEncoding,
    first_bit: Option<u64>,
    last_bit: Option<u64>,
}

/// A `packbits` codec implementation bound to a data type and fill value.
#[derive(Debug, Clone)]
struct PackBitsCodecBound {
    padding_encoding: PackBitsPaddingEncoding,
    components: PackBitsCodecComponents,
    first_bit: u64,
    last_bit: u64,
    data_type: DataType,
    fill_value: FillValue,
}

impl Default for PackBitsCodec {
    fn default() -> Self {
        Self::new(PackBitsPaddingEncoding::default(), None, None)
            .expect("this configuration is supported")
    }
}

fn padding_bits(elements_size_bits: u64) -> u8 {
    let rem = (elements_size_bits % 8) as u8;
    if rem == 0 { 0 } else { 8 - rem }
}

/// The sizes of the packed elements of an array, which are checked for overflow.
struct PackBitsSizes {
    num_elements: u64,
    /// The size of the decoded elements in bytes.
    elements_size_dec_bytes: u64,
    /// The size of the packed elements in bits, excluding padding.
    elements_size_bits: u64,
    /// The size of the packed elements in bytes, excluding the padding encoding byte.
    elements_size_bytes: usize,
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
        if let (Some(first_bit), Some(last_bit)) = (first_bit, last_bit)
            && last_bit < first_bit
        {
            return Err(PluginCreateError::from(
                "packbits codec `last_bit` is less than `first_bit`",
            ));
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
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = PackBitsCodecConfiguration::V1(PackBitsCodecConfigurationV1 {
            padding_encoding: Some(self.padding_encoding),
            first_bit: self.first_bit,
            last_bit: self.last_bit,
        });
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl UnboundArrayToBytesCodecTraits for PackBitsCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self as Arc<dyn UnboundArrayToBytesCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        let components = pack_bits_components(&data_type)?;
        let first_bit = self.first_bit.unwrap_or(0);
        let last_bit = self.last_bit.unwrap_or(components.component_size_bits - 1);

        if last_bit < first_bit {
            return Err(CodecCreateError::Other(
                "packbits codec `last_bit` is less than `first_bit`".to_string(),
            ));
        }
        if last_bit >= components.component_size_bits {
            return Err(CodecCreateError::Other(
                "packbits codec `last_bit` is outside the data type component".to_string(),
            ));
        }

        Ok(Arc::new(PackBitsCodecBound {
            padding_encoding: self.padding_encoding,
            components,
            first_bit,
            last_bit,
            data_type,
            fill_value,
        }))
    }
}

impl PackBitsCodecBound {
    /// Returns the sizes of an array with `shape`.
    ///
    /// # Errors
    /// Returns an error if the data type does not have a fixed size, or a size overflows.
    fn sizes(&self, shape: &[NonZeroU64]) -> Result<PackBitsSizes, CodecError> {
        let data_type_size_dec = self.data_type.fixed_size().ok_or_else(|| {
            CodecError::Other("data type must have a fixed size for the packbits codec".to_string())
        })?;
        let num_elements = shape
            .iter()
            .try_fold(1u64, |count, d| count.checked_mul(d.get()))
            .ok_or("the decoded chunk element count overflows u64")?;
        let elements_size_dec_bytes = num_elements
            .checked_mul(data_type_size_dec as u64)
            .ok_or("the decoded chunk size in bytes overflows u64")?;
        let element_size_bits = (self.last_bit - self.first_bit + 1)
            .checked_mul(self.components.num_components)
            .ok_or("the packed element size in bits overflows u64")?;
        let elements_size_bits = num_elements
            .checked_mul(element_size_bits)
            .ok_or("the packed chunk size in bits overflows u64")?;
        let elements_size_bytes = usize::try_from(elements_size_bits.div_ceil(8))
            .map_err(|_| CodecError::from("the packed chunk size in bytes overflows usize"))?;
        Ok(PackBitsSizes {
            num_elements,
            elements_size_dec_bytes,
            elements_size_bits,
            elements_size_bytes,
        })
    }
}

impl ArrayCodecTraits for PackBitsCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl zarrs_codec::ArrayToBytesCodecNoSubchunkingTraits for PackBitsCodecBound {}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for PackBitsCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        let PackBitsCodecComponents {
            component_size_bits,
            num_components,
            sign_extension: _,
        } = self.components;
        let first_bit = self.first_bit;
        let last_bit = self.last_bit;

        // Bytes codec fast path
        if component_size_bits.is_multiple_of(8)
            && first_bit == 0
            && last_bit == component_size_bits - 1
        {
            // Data types are expected to support the bytes codec if their component size in bits is a multiple of 8.
            return Arc::new(BytesCodec::new(Some(Endianness::Little)))
                .with_context(
                    self.data_type.clone(),
                    self.fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )
                .map_err(|err| CodecError::Other(err.to_string()))?
                .encode(bytes.clone(), shape, options);
        }

        // Get the component and element size in bits
        let PackBitsSizes {
            num_elements,
            elements_size_dec_bytes,
            elements_size_bits,
            elements_size_bytes,
        } = self.sizes(shape)?;
        let component_size_bits_extracted = last_bit - first_bit + 1;

        // Input checks
        let bytes = bytes.into_fixed()?;
        if bytes.len() as u64 != elements_size_dec_bytes {
            return Err(InvalidBytesLengthError::new(
                bytes.len(),
                usize::try_from(elements_size_dec_bytes).map_err(|_| {
                    CodecError::from("the decoded chunk size in bytes overflows usize")
                })?,
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
        let padding_bits = padding_bits(elements_size_bits);
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

        Ok(CowBytes::from(bytes_enc))
    }

    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let PackBitsCodecComponents {
            component_size_bits,
            num_components,
            sign_extension,
        } = self.components;
        let first_bit = self.first_bit;
        let last_bit = self.last_bit;

        // Bytes codec fast path
        if component_size_bits % 8 == 0 && first_bit == 0 && last_bit == component_size_bits - 1 {
            // Data types are expected to support the bytes codec if their element size in bits is a multiple of 8.
            return Arc::new(BytesCodec::new(Some(Endianness::Little)))
                .with_context(
                    self.data_type.clone(),
                    self.fill_value.clone(),
                    &CodecSpecificOptions::default(),
                )
                .map_err(|err| CodecError::Other(err.to_string()))?
                .decode(bytes.clone(), shape, options);
        }

        // Get the component and element size in bits
        let PackBitsSizes {
            num_elements,
            elements_size_dec_bytes,
            elements_size_bits,
            elements_size_bytes,
        } = self.sizes(shape)?;
        let component_size_bits_extracted = last_bit - first_bit + 1;

        // Input checks
        let expected_length = elements_size_bytes
            + match self.padding_encoding {
                PackBitsPaddingEncoding::None => 0,
                PackBitsPaddingEncoding::FirstByte | PackBitsPaddingEncoding::LastByte => 1,
            };
        if bytes.len() != expected_length {
            return Err(InvalidBytesLengthError::new(bytes.len(), expected_length).into());
        }

        let padding_bits = padding_bits(elements_size_bits);
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
        let mut bytes_dec = vec![
            0u8;
            usize::try_from(elements_size_dec_bytes).map_err(|_| {
                CodecError::from("the decoded chunk size in bytes overflows usize")
            })?
        ];

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

        Ok(ArrayBytes::Fixed(CowBytes::from(bytes_dec)))
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        let component_size_bits = self.components.component_size_bits;
        let first_bit = self.first_bit;
        let last_bit = self.last_bit;

        // Bytes codec fast path
        if component_size_bits.is_multiple_of(8)
            && first_bit == 0
            && last_bit == component_size_bits - 1
        {
            // Data types are expected to support the bytes codec if their element size in bits is a multiple of 8.
            Ok(Arc::new(BytesCodecPartial::new(
                input_handle,
                shape,
                &self.data_type,
                &self.fill_value,
                Some(Endianness::Little),
            )))
        } else {
            Ok(Arc::new(PackBitsPartialDecoder::new(
                input_handle,
                shape.to_vec(),
                self.data_type.clone(),
                self.fill_value.clone(),
                self.padding_encoding,
                self.components,
                self.first_bit,
                self.last_bit,
            )))
        }
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        let component_size_bits = self.components.component_size_bits;
        let first_bit = self.first_bit;
        let last_bit = self.last_bit;

        // Bytes codec fast path
        if component_size_bits.is_multiple_of(8)
            && first_bit == 0
            && last_bit == component_size_bits - 1
        {
            // Data types are expected to support the bytes codec if their element size in bits is a multiple of 8.
            Ok(Arc::new(BytesCodecPartial::new(
                input_handle,
                shape,
                &self.data_type,
                &self.fill_value,
                Some(Endianness::Little),
            )))
        } else {
            Ok(Arc::new(AsyncPackBitsPartialDecoder::new(
                input_handle,
                shape.to_vec(),
                self.data_type.clone(),
                self.fill_value.clone(),
                self.padding_encoding,
                self.components,
                self.first_bit,
                self.last_bit,
            )))
        }
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        let elements_size_bytes = self.sizes(shape)?.elements_size_bytes as u64;

        let padding_encoding_byte = match self.padding_encoding {
            PackBitsPaddingEncoding::None => 0,
            PackBitsPaddingEncoding::FirstByte | PackBitsPaddingEncoding::LastByte => 1,
        };
        Ok(BytesRepresentation::FixedSize(
            elements_size_bytes + padding_encoding_byte,
        ))
    }
}
