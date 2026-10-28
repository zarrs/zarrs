use std::sync::Arc;

use zarrs_plugin::{PluginCreateError, ZarrVersion};

#[cfg(feature = "async")]
use super::packbits_partial_decoder::AsyncPackBitsPartialDecoder;
use super::packbits_partial_decoder::PackBitsPartialDecoder;
use super::{
    PackBitsBitOrder, PackBitsCodecConfiguration, PackBitsCodecConfigurationV1, PackBitsLayout,
    pack, pack_bits_components, unpack_to_vec,
};
use crate::array::{ArrayBytes, BytesRepresentation, CowBytes, DataType, FillValue};
use std::num::NonZeroU64;
use std::ops::Range;
use zarrs_codec::{
    ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits,
    BytesPartialDecoderTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecSpecificOptions, CodecTraits, InvalidBytesLengthError, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency, UnboundArrayToBytesCodecTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};
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
    pub(super) fill_value: FillValue,
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
        Ok(Arc::new(self.bind(
            data_type,
            fill_value,
            PackBitsBitOrder::LsbFirst,
        )?))
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
        fill_value: FillValue,
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

    /// Returns the range of the packed elements in chunks written by zarrs 0.20-0.23, if it differs from `sizes`.
    ///
    /// These omitted the padding encoding byte if all bits of whole-byte components are packed,
    /// so there are no padding bits.
    pub(super) fn legacy_packed(&self, sizes: &PackBitsSizes) -> Option<Range<usize>> {
        (sizes.padding_byte.is_some() && self.layout.packs_whole_bytes())
            .then(|| 0..sizes.packed.len())
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
        _options: &CodecOptions,
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
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let sizes = self.sizes(shape)?;

        // Input checks
        let packed = if bytes.len() == sizes.encoded_bytes {
            if let Some((index, _)) = sizes.padding_byte {
                sizes.check_padding_byte(bytes[index])?;
            }
            sizes.packed
        } else if let Some(legacy) = self.legacy_packed(&sizes)
            && bytes.len() == legacy.end
        {
            legacy
        } else {
            return Err(InvalidBytesLengthError::new(bytes.len(), sizes.encoded_bytes).into());
        };
        let packed_elements = &bytes[packed];

        let decoded_bytes = usize::try_from(sizes.decoded_bytes)
            .map_err(|_| CodecError::from("the decoded chunk size in bytes overflows usize"))?;
        let bytes_dec = unpack_to_vec(self.layout, packed_elements, 0, decoded_bytes);

        Ok(ArrayBytes::Fixed(self.components_to_bytes(bytes_dec)?))
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(PackBitsPartialDecoder::new(
            input_handle,
            shape.to_vec(),
            self,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(AsyncPackBitsPartialDecoder::new(
            input_handle,
            shape.to_vec(),
            self,
        )))
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        Ok(BytesRepresentation::FixedSize(self.encoded_size(shape)?))
    }
}
