use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};

use super::{
    PackBitsBitOrder, PackBitsCodecConfiguration, PackBitsCodecConfigurationV1, PackBitsLayout,
    pack, pack_bits_components, unpack_to_vec,
};
use crate::array::{ArrayBytes, BytesRepresentation, CowBytes, DataType, FillValue};
use std::num::NonZeroU64;
use std::ops::Range;
use zarrs_codec::{
    ArrayCodecTraits, ArrayToBytesCodecTraits, CodecCreateError, CodecError, CodecMetadataOptions,
    CodecOptions, CodecSpecificOptions, CodecTraits, InvalidBytesLengthError,
    UnboundArrayToBytesCodecTraits,
};

use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;

use super::PackBitsDataTypeExt;

/// A `packbits` codec implementation.
#[derive(Debug, Clone)]
pub struct PackBitsCodec {
    padding_encoding: PackBitsPaddingEncoding,
    first_bit: Option<u64>,
    last_bit: Option<u64>,
}

/// A `packbits` codec implementation bound to a data type and fill value.
#[derive(Debug, Clone)]
pub(crate) struct PackBitsCodecBound {
    pub(super) padding_encoding: PackBitsPaddingEncoding,
    pub(super) layout: PackBitsLayout,
    /// Whether decoded bytes are converted to and from the byte order of `layout` with the `bytes` codec.
    pub(super) convert_bytes: bool,
    /// The number of packed bits of an element.
    pub(super) element_size_bits: u64,
    pub(super) data_type: DataType,
    pub(super) fill_value: Option<FillValue>,
}

impl Default for PackBitsCodec {
    fn default() -> Self {
        Self::new(PackBitsPaddingEncoding::default(), None, None)
            .expect("this configuration is supported")
    }
}

fn padding_bits(packed_bits: u64) -> u8 {
    let rem = (packed_bits % 8) as u8;
    if rem == 0 { 0 } else { 8 - rem }
}

/// The sizes of the packed elements of an array, which are checked for overflow.
pub(super) struct PackBitsSizes {
    /// The size of the decoded elements in bytes.
    pub(super) decoded_bytes: u64,
    /// The size of the encoded bytes, including the padding encoding byte.
    pub(super) encoded_bytes: usize,
    /// The range of the packed elements in the encoded bytes.
    pub(super) packed: Range<usize>,
    /// The index and value of the padding encoding byte in the encoded bytes, if any.
    pub(super) padding_byte: Option<(usize, u8)>,
}

impl PackBitsSizes {
    /// Check the padding encoding byte of the encoded bytes.
    ///
    /// # Errors
    /// Returns an error if `byte` is not the expected padding encoding byte.
    pub(super) fn check_padding_byte(&self, byte: u8) -> Result<(), CodecError> {
        match self.padding_byte {
            Some((_, expected)) if byte != expected => Err(CodecError::Other(format!(
                "the packbits padding encoding byte is {byte}, expected {expected}"
            ))),
            _ => Ok(()),
        }
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
        version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        if version == ZarrVersion::V2 {
            return None;
        }
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
        fill_value: Option<FillValue>,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        let bound = self.bind(
            data_type.clone(),
            fill_value.clone(),
            PackBitsBitOrder::LsbFirst,
        )?;
        if self.padding_encoding == PackBitsPaddingEncoding::None
            && bound.layout.first_bit == 0
            && Some(bound.element_size_bits) == data_type.fixed_size().map(|size| size as u64 * 8)
        {
            return Arc::new(crate::array::codec::BytesCodec::new(Some(
                zarrs_metadata::Endianness::Little,
            )))
            .with_context(data_type, fill_value, _codec_specific_options);
        }
        Ok(Arc::new(bound))
    }
}

impl PackBitsCodec {
    /// Bind the codec to a data type and fill value.
    ///
    /// `packbits` fills bytes in [`PackBitsBitOrder::LsbFirst`] order, and `numcodecs.packbits` in [`PackBitsBitOrder::MsbFirst`] order.
    ///
    /// # Errors
    /// Returns an error if the data type is unsupported or incompatible with the configuration.
    pub(crate) fn bind(
        &self,
        data_type: DataType,
        fill_value: Option<FillValue>,
        bit_order: PackBitsBitOrder,
    ) -> Result<PackBitsCodecBound, CodecCreateError> {
        let components = pack_bits_components(&data_type)?;
        if components.component_size_bits > 64 {
            return Err(CodecCreateError::Other(
                "packbits codec data type components must not exceed 64 bits".to_string(),
            ));
        }
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

        // Components are packed contiguously from the decoded bytes
        if let Some(data_type_size) = data_type.fixed_size()
            && Some(data_type_size as u64)
                != components
                    .num_components
                    .checked_mul(components.component_size_bits.div_ceil(8))
        {
            return Err(CodecCreateError::Other(
                "packbits codec data type size does not match its components".to_string(),
            ));
        }
        let element_size_bits = (last_bit - first_bit + 1)
            .checked_mul(components.num_components)
            .ok_or_else(|| {
                CodecCreateError::Other("the packed element size in bits overflows u64".to_string())
            })?;

        Ok(PackBitsCodecBound {
            padding_encoding: self.padding_encoding,
            layout: PackBitsLayout::new(components, first_bit, last_bit, bit_order),
            convert_bytes: components.convert_bytes,
            element_size_bits,
            data_type,
            fill_value,
        })
    }
}

impl PackBitsCodecBound {
    /// Returns the sizes of an array with `shape`.
    ///
    /// # Errors
    /// Returns an error if the data type does not have a fixed size, or a size overflows.
    pub(super) fn sizes(&self, shape: &[NonZeroU64]) -> Result<PackBitsSizes, CodecError> {
        let data_type_size_dec = self.data_type.fixed_size().ok_or_else(|| {
            CodecError::Other("data type must have a fixed size for the packbits codec".to_string())
        })?;
        let num_elements = shape
            .iter()
            .try_fold(1u64, |count, d| count.checked_mul(d.get()))
            .ok_or("the decoded chunk element count overflows u64")?;
        let decoded_bytes = num_elements
            .checked_mul(data_type_size_dec as u64)
            .ok_or("the decoded chunk size in bytes overflows u64")?;
        let packed_bits = num_elements
            .checked_mul(self.element_size_bits)
            .ok_or("the packed chunk size in bits overflows u64")?;
        let packed_bytes = usize::try_from(packed_bits.div_ceil(8))
            .map_err(|_| CodecError::from("the packed chunk size in bytes overflows usize"))?;
        let padding_bits = padding_bits(packed_bits);
        let (packed_start, padding_byte) = match self.padding_encoding {
            PackBitsPaddingEncoding::None => (0, None),
            PackBitsPaddingEncoding::FirstByte => (1, Some((0, padding_bits))),
            PackBitsPaddingEncoding::LastByte => (0, Some((packed_bytes, padding_bits))),
        };
        let encoded_bytes = packed_bytes
            .checked_add(usize::from(padding_byte.is_some()))
            .ok_or("the encoded chunk size in bytes overflows usize")?;
        Ok(PackBitsSizes {
            decoded_bytes,
            encoded_bytes,
            packed: packed_start..packed_start + packed_bytes,
            padding_byte,
        })
    }

    /// Convert decoded bytes to the byte order of the packed components, if they differ.
    ///
    /// # Errors
    /// Returns an error if the data type does not support the `packbits` codec.
    pub(super) fn bytes_to_components<'a>(
        &self,
        bytes: CowBytes<'a>,
    ) -> Result<CowBytes<'a>, CodecError> {
        if !self.convert_bytes {
            return Ok(bytes);
        }
        let packbits = self.data_type.codec_packbits()?;
        Ok(packbits.encode(bytes, Some(self.layout.endianness))?)
    }

    /// Convert unpacked components to decoded bytes, if they differ.
    ///
    /// # Errors
    /// Returns an error if the data type does not support the `packbits` codec.
    pub(super) fn components_to_bytes(
        &self,
        components: Vec<u8>,
    ) -> Result<CowBytes<'static>, CodecError> {
        let components = CowBytes::from(components);
        if !self.convert_bytes {
            return Ok(components);
        }
        let packbits = self.data_type.codec_packbits()?;
        Ok(packbits.decode(components, Some(self.layout.endianness))?)
    }

    /// Returns the size in bytes of an encoded array with `shape`.
    ///
    /// # Errors
    /// Returns an error if the data type does not have a fixed size, or a size overflows.
    pub(crate) fn encoded_size(&self, shape: &[NonZeroU64]) -> Result<u64, CodecError> {
        Ok(self.sizes(shape)?.encoded_bytes as u64)
    }
}

impl ArrayCodecTraits for PackBitsCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> Option<&FillValue> {
        self.fill_value.as_ref()
    }
}

impl zarrs_codec::ArrayToBytesCodecNoSubchunkingTraits for PackBitsCodecBound {}

impl ArrayToBytesCodecTraits for PackBitsCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn prepare_decode(
        self: Arc<Self>,
        source: zarrs_codec::plan::ByteSource<'static>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<zarrs_codec::plan::PrepareDecode, CodecError> {
        Ok(zarrs_codec::plan::PrepareDecode::Ready(Arc::new(
            zarrs_codec::plan::SourceDecoder::new(self, source, shape),
        )))
    }
    fn plan_decode<'a>(
        self: Arc<Self>,
        source: zarrs_codec::plan::ByteSource<'a>,
        shape: &[NonZeroU64],
        request: zarrs_codec::plan::DecodeRequest<'a>,
        _options: &CodecOptions,
    ) -> Result<zarrs_codec::plan::Task<'a>, CodecError> {
        if request.indexer.is_whole(shape) {
            return zarrs_codec::plan::plan_decode_whole(self, source, shape, request, _options);
        }
        use zarrs_codec::plan::{Task, place_decoded, place_fill_value};
        request.indexer.validate(shape)?;
        let sizes = self.sizes(shape)?;
        let bit_ranges = request
            .indexer
            .as_indexer()
            .iter_contiguous_byte_ranges(
                bytemuck::must_cast_slice(shape),
                usize::try_from(self.element_size_bits)
                    .map_err(|_| CodecError::Other("packed element size exceeds usize".into()))?,
            )?
            .collect::<Vec<_>>();
        let ranges = bit_ranges
            .iter()
            .map(|range| {
                zarrs_storage::byte_range::ByteRange::new(
                    sizes.packed.start as u64 + range.start / 8
                        ..sizes.packed.start as u64 + range.end.div_ceil(8),
                )
            })
            .collect::<Vec<_>>();
        Ok(Task::read(source, ranges, move |reads, _ctx| {
            if reads.iter().any(Option::is_none) {
                return place_fill_value(
                    &self.data_type,
                    self.fill_value
                        .as_ref()
                        .ok_or(CodecError::UndefinedFillValue)?,
                    request,
                );
            }
            if reads.len() != bit_ranges.len() {
                return Err("unexpected packbits range count".into());
            }
            let decoded_len = usize::try_from(request.indexer.as_indexer().len())
                .ok()
                .and_then(|len| len.checked_mul(self.data_type.fixed_size().unwrap()))
                .ok_or("packed output length overflows")?;
            let mut bytes = Vec::with_capacity(decoded_len);
            let mut remaining = &mut bytes.spare_capacity_mut()[..decoded_len];
            for (packed, bits) in reads.iter().zip(&bit_ranges) {
                let packed = packed.as_ref().expect("checked present");
                let expected = usize::try_from(bits.end.div_ceil(8) - bits.start / 8).unwrap();
                if packed.len() != expected {
                    return Err(InvalidBytesLengthError::new(packed.len(), expected).into());
                }
                let len = usize::try_from((bits.end - bits.start) / self.element_size_bits)
                    .unwrap()
                    * self.data_type.fixed_size().unwrap();
                let (output, rest) = std::mem::take(&mut remaining)
                    .split_at_mut_checked(len)
                    .ok_or("packbits ranges exceed decoded elements")?;
                remaining = rest;
                super::unpack(self.layout, packed, (bits.start % 8) as u8, output);
            }
            if !remaining.is_empty() {
                return Err("packbits ranges do not cover decoded elements".into());
            }
            // SAFETY: unpack initialized every byte in the checked output ranges.
            unsafe {
                bytes.set_len(decoded_len);
            }
            place_decoded(
                ArrayBytes::Fixed(self.components_to_bytes(bytes)?),
                request.target,
            )
        }))
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &zarrs_codec::ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let sizes = self.sizes(shape)?;

        // Input checks
        let bytes = bytes.into_fixed()?;
        if bytes.len() as u64 != sizes.decoded_bytes {
            return Err(InvalidBytesLengthError::new(
                bytes.len(),
                usize::try_from(sizes.decoded_bytes).map_err(|_| {
                    CodecError::from("the decoded chunk size in bytes overflows usize")
                })?,
            )
            .into());
        }

        let bytes = self.bytes_to_components(bytes)?;
        let mut bytes_enc = Vec::with_capacity(sizes.encoded_bytes);
        let bytes_enc_uninit = &mut bytes_enc.spare_capacity_mut()[..sizes.encoded_bytes];
        if let Some((index, padding_bits)) = sizes.padding_byte {
            bytes_enc_uninit[index].write(padding_bits);
        }
        pack(
            self.layout,
            &bytes,
            &mut bytes_enc_uninit[sizes.packed.clone()],
        );
        // SAFETY: `pack` initialised the packed elements, and the padding encoding byte (if any) was written.
        // These are all of the first `sizes.encoded_bytes <= capacity` elements.
        unsafe { bytes_enc.set_len(sizes.encoded_bytes) };

        Ok(CowBytes::from(bytes_enc))
    }

    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &zarrs_codec::ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let sizes = self.sizes(shape)?;

        // Input checks
        if bytes.len() != sizes.encoded_bytes {
            return Err(InvalidBytesLengthError::new(bytes.len(), sizes.encoded_bytes).into());
        }
        if let Some((index, _)) = sizes.padding_byte {
            sizes.check_padding_byte(bytes[index])?;
        }
        let packed_elements = &bytes[sizes.packed];

        let decoded_bytes = usize::try_from(sizes.decoded_bytes)
            .map_err(|_| CodecError::from("the decoded chunk size in bytes overflows usize"))?;
        let bytes_dec = unpack_to_vec(self.layout, packed_elements, 0, decoded_bytes);

        Ok(ArrayBytes::Fixed(self.components_to_bytes(bytes_dec)?))
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        Ok(BytesRepresentation::FixedSize(self.encoded_size(shape)?))
    }
}
