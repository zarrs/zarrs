#![allow(clippy::similar_names)]

// FIXME: This codec was really hacked together.
// It can probably be written much cleaner and with simpler logic.

use std::sync::Arc;

use num::Integer;
use zarrs_plugin::{PluginCreateError, ZarrVersion};

use super::{
    PackBitsCodecComponents, PackBitsCodecConfiguration, PackBitsCodecConfigurationV1,
    pack_bits_components, packbits_ranges,
};
use crate::array::codec::BytesCodec;
use crate::array::codec::array_to_bytes::packbits::{div_rem_8bit, sign_extend_component};
use crate::array::{ArrayBytes, BytesRepresentation, CowBytes, DataType, FillValue};
use std::num::NonZeroU64;
use zarrs_codec::plan::{
    ByteSource, DecodeRequest, PrepareDecode, SourceDecoder, Task, TaskHint, decode_whole,
    place_decoded, plan_decode_whole,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoInput, ArrayBytesDecodeIntoTarget, ArrayCodecTraits,
    ArrayToBytesCodecTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecSpecificOptions, CodecTraits, ComputeResources, InvalidBytesLengthError,
    UnboundArrayToBytesCodecTraits, decode_into_array_bytes_target,
};
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

impl PackBitsSizes {
    /// Returns the length of the decoded bytes.
    fn decoded_len(&self) -> Result<usize, CodecError> {
        usize::try_from(self.elements_size_dec_bytes)
            .map_err(|_| CodecError::from("the decoded chunk size in bytes overflows usize"))
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
}

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

        if self.padding_encoding == PackBitsPaddingEncoding::None
            && components.component_size_bits.is_multiple_of(8)
            && first_bit == 0
            && last_bit == components.component_size_bits - 1
        {
            // The components are whole bytes that are all packed without a padding encoding byte, so the codec is the
            // little endian `bytes` codec.
            // Data types are expected to support the bytes codec if their component size in bits is a multiple of 8.
            return Arc::new(BytesCodec::new(Some(Endianness::Little)))
                .with_context(data_type, fill_value, &CodecSpecificOptions::default())
                .map_err(|err| CodecCreateError::Other(err.to_string()));
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

    /// Unpack the components of `bytes` into `bytes_dec`.
    ///
    /// `bytes_dec` must be the length of the decoded bytes.
    /// It is zeroed once the input has been checked, since only the bits of the components are set, so it is not written to if the input is invalid.
    fn unpack(
        &self,
        bytes: &[u8],
        sizes: &PackBitsSizes,
        bytes_dec: &mut [u8],
    ) -> Result<(), CodecError> {
        let PackBitsCodecComponents {
            component_size_bits,
            num_components,
            sign_extension,
        } = self.components;
        let first_bit = self.first_bit;
        let last_bit = self.last_bit;

        // Get the component size in bits
        let PackBitsSizes {
            num_elements,
            elements_size_bits,
            elements_size_bytes,
            ..
        } = *sizes;
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
            PackBitsPaddingEncoding::None => bytes,
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

        // Decode the components
        bytes_dec.fill(0);
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
                sign_extend_component(bytes_dec, component_idx, last_bit, component_size_bits);
            }
        }
        Ok(())
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
}

impl zarrs_codec::ArrayToBytesCodecNoSubchunkingTraits for PackBitsCodecBound {}

impl ArrayToBytesCodecTraits for PackBitsCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let PackBitsCodecComponents {
            component_size_bits,
            num_components,
            sign_extension: _,
        } = self.components;
        let first_bit = self.first_bit;
        let last_bit = self.last_bit;

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
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let sizes = self.sizes(shape)?;
        let mut bytes_dec = vec![0u8; sizes.decoded_len()?];
        self.unpack(&bytes, &sizes, &mut bytes_dec)?;
        Ok(ArrayBytes::Fixed(CowBytes::from(bytes_dec)))
    }

    fn decode_into(
        &self,
        input: ArrayBytesDecodeIntoInput<'_>,
        shape: &[NonZeroU64],
        mut output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        let bytes = input.into_bytes(options, resources)?;

        // Unpack directly into an output that is one contiguous region
        let sizes = self.sizes(shape)?;
        if let ArrayBytesDecodeIntoTarget::Fixed(output) = &mut output_target
            && let Some(output) = output.as_mut_slice()
            && output.len() == sizes.decoded_len()?
        {
            return self.unpack(&bytes, &sizes, output);
        }

        let bytes = self.decode(bytes, shape, options, resources)?;
        decode_into_array_bytes_target(&bytes, output_target)
    }

    fn prepare_decode(
        self: Arc<Self>,
        source: ByteSource<'static>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<PrepareDecode, CodecError> {
        // Requests are planned from the source, as only their bytes are read and decoded
        Ok(PrepareDecode::Ready(Arc::new(SourceDecoder::new(
            self, source, shape,
        ))))
    }

    fn plan_decode<'a>(
        self: Arc<Self>,
        source: ByteSource<'a>,
        shape: &[NonZeroU64],
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        if request.indexer.is_whole(shape) {
            return plan_decode_whole(self, source, shape, request, options);
        }
        let indexer = request.indexer.as_indexer();
        indexer.validate(bytemuck::must_cast_slice(shape))?;

        // Read only the bytes holding the requested elements
        let (bit_ranges, byte_ranges) = packbits_ranges::partial_ranges(
            shape,
            self.padding_encoding,
            self.components,
            self.first_bit,
            self.last_bit,
            indexer,
        )?;
        let read_bytes = byte_ranges.iter().map(|range| range.length(u64::MAX)).sum();
        let shape = shape.to_vec();
        let options = *options;
        Ok(Task::read(source, byte_ranges, move |reads, ctx| {
            if reads.iter().any(Option::is_none) {
                return decode_whole(&*self, None, &shape, request, &options, ctx.resources());
            }
            let indexer = request.indexer.as_indexer();
            let bytes = packbits_ranges::partial_unpack(
                reads.iter().flatten().map(|read| read.as_slice()),
                &bit_ranges,
                &self.data_type,
                self.components,
                self.first_bit,
                self.last_bit,
                indexer.len(),
            )?;
            place_decoded(bytes, request.target)
        })
        .with_hint(TaskHint {
            read_bytes: Some(read_bytes),
            compute_bytes: read_bytes,
        }))
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
