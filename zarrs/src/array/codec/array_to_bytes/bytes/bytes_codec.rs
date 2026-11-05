// Note: No validation that this codec is created *without* a specified endianness for multi-byte data types.

use std::sync::Arc;

use zarrs_codec::{
    ArrayBytesDecodeIntoInput, ArrayBytesDecodeIntoTarget, ArrayCodecTraits,
    ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToBytesCodecTraits,
    BytesPartialDecoderTraits, BytesPartialEncoderTraits, CodecCreateError, CodecError,
    CodecMetadataOptions, CodecOptions, CodecSpecificOptions, CodecTraits, ComputeResources,
    InvalidBytesLengthError, PartialDecoderCapability, PartialEncoderCapability,
    RecommendedConcurrency, Resources, UnboundArrayToBytesCodecTraits,
    decode_into_array_bytes_target,
};
use zarrs_plugin::{ExtensionAliasesV3, PluginCreateError, ZarrVersion};

use super::{
    BytesCodecConfiguration, BytesCodecConfigurationV1, BytesDataTypeExt, BytesDataTypeTraits,
    BytesDataTypeTraitsArc, Endianness, bytes_codec_partial,
};
use crate::array::{ArrayBytes, BytesRepresentation, CowBytes, DataType, FillValue};
use std::num::NonZeroU64;

#[cfg(feature = "async")]
use zarrs_codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};
use zarrs_metadata::Configuration;

/// A `bytes` codec implementation.
#[derive(Debug, Clone)]
pub struct BytesCodec {
    endian: Option<Endianness>,
}

/// A `bytes` codec implementation bound to a data type and fill value.
#[derive(Clone, Debug)]
struct BytesCodecBound {
    endian: Option<Endianness>,
    passthrough: bool,
    decode_in_place: bool,
    data_type: DataType,
    /// The `bytes` codec support of `data_type`.
    bytes: BytesDataTypeTraitsArc,
    fill_value: Option<FillValue>,
}

impl Default for BytesCodec {
    fn default() -> Self {
        Self::new(Some(Endianness::native()))
    }
}

impl BytesCodec {
    /// Create a new `bytes` codec.
    ///
    /// `endian` is optional because an 8-bit type has no endianness.
    #[must_use]
    pub const fn new(endian: Option<Endianness>) -> Self {
        Self { endian }
    }

    /// Create a new `bytes` codec for little endian data.
    #[must_use]
    pub const fn little() -> Self {
        Self::new(Some(Endianness::Little))
    }

    /// Create a new `bytes` codec for big endian data.
    #[must_use]
    pub const fn big() -> Self {
        Self::new(Some(Endianness::Big))
    }

    /// Create a new `bytes` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &BytesCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            BytesCodecConfiguration::V1(configuration) => Ok(Self::new(configuration.endian)),
            _ => Err(PluginCreateError::Other(
                "this bytes codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for BytesCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = BytesCodecConfiguration::V1(BytesCodecConfigurationV1 {
            endian: self.endian,
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
            partial_encode: true,
        }
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl UnboundArrayToBytesCodecTraits for BytesCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self as Arc<dyn UnboundArrayToBytesCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: Option<FillValue>,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        if data_type.is_optional() {
            return Err(CodecCreateError::UnsupportedDataType(
                data_type,
                Self::aliases_v3().default_name.to_string(),
            ));
        }
        let bytes = data_type.codec_bytes_arc()?;
        let codec = &*bytes;
        let passthrough = codec.is_decode_passthrough(self.endian);
        // Decode one element to check the endianness before a decoder writes to an output.
        let decode_in_place = !passthrough
            && codec.is_decode_in_place_efficient()
            && data_type.fixed_size().is_some_and(|size| {
                codec
                    .decode_in_place(&mut vec![0; size], self.endian)
                    .is_ok()
            });
        Ok(Arc::new(BytesCodecBound {
            endian: self.endian,
            passthrough,
            decode_in_place,
            data_type,
            bytes,
            fill_value,
        }))
    }
}

/// Return the size of `data_type`, which must be fixed-size for the `bytes` codec.
pub(super) fn fixed_size(data_type: &DataType) -> Result<usize, CodecError> {
    data_type.fixed_size().ok_or_else(|| {
        CodecError::UnsupportedDataType(
            data_type.clone(),
            BytesCodec::aliases_v3().default_name.to_string(),
        )
    })
}

/// Return the decoded element count and size in bytes of a chunk with `shape`.
fn decoded_size(data_type: &DataType, shape: &[NonZeroU64]) -> Result<(u64, u64), CodecError> {
    let data_type_size = fixed_size(data_type)?;
    let num_elements = shape
        .iter()
        .try_fold(1u64, |count, d| count.checked_mul(d.get()))
        .ok_or("the decoded chunk element count overflows u64")?;
    let num_bytes = num_elements
        .checked_mul(data_type_size as u64)
        .ok_or("the decoded chunk size in bytes overflows u64")?;
    Ok((num_elements, num_bytes))
}

/// Return the decoded element count and size in bytes of a chunk with `shape`, with the size checked to fit a `usize`.
pub(super) fn decoded_len(
    data_type: &DataType,
    shape: &[NonZeroU64],
) -> Result<(u64, usize), CodecError> {
    let (num_elements, num_bytes) = decoded_size(data_type, shape)?;
    let len = usize::try_from(num_bytes)
        .map_err(|_| CodecError::from("the decoded chunk size in bytes overflows usize"))?;
    Ok((num_elements, len))
}

impl BytesCodecBound {
    /// How encoded elements can be decoded as they are copied to the output.
    fn in_place(&self) -> Option<InPlace<'_>> {
        if self.passthrough {
            Some(InPlace::Copy)
        } else if self.decode_in_place {
            Some(InPlace::Decode(&*self.bytes, self.endian))
        } else {
            None
        }
    }
}

/// How encoded elements are decoded as they are copied into an output.
#[derive(Clone, Copy)]
enum InPlace<'a> {
    /// The encoded elements are the decoded elements.
    Copy,
    /// The elements are decoded in place after they are copied.
    Decode(&'a dyn BytesDataTypeTraits, Option<Endianness>),
}

/// Copy a contiguous region of encoded elements into an output and decode them in place.
fn decode_copy(
    codec: &dyn BytesDataTypeTraits,
    endian: Option<Endianness>,
) -> impl Fn(&[u8], &mut [u8]) -> Result<(), CodecError> {
    move |source, destination| {
        destination.copy_from_slice(source);
        Ok(codec.decode_in_place(destination, endian)?)
    }
}

impl ArrayCodecTraits for BytesCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> Option<&FillValue> {
        self.fill_value.as_ref()
    }

    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        // TODO: Recomment > 1 if endianness needs changing and input is sufficiently large
        // if let Some(endian) = &self.endian {
        //     if !endian.is_native() {
        //         FIXME: Support parallel
        //         let min_elements_per_thread = 32768; // 32^3
        //         let num_elements = shape.iter().map(|d| d.get()).product::<u64>();
        //         unsafe {
        //             NonZeroU64::new_unchecked(
        //                 num_elements.div_ceil(min_elements_per_thread),
        //             )
        //         }
        //     }
        // }
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl zarrs_codec::ArrayToBytesCodecNoSubchunkingTraits for BytesCodecBound {}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for BytesCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn is_decode_passthrough(&self, _shape: &[NonZeroU64]) -> Result<bool, CodecError> {
        if !self.data_type.is_fixed() || self.data_type.is_optional() {
            return Ok(false);
        }
        Ok(self.passthrough)
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let (num_elements, _) = decoded_len(&self.data_type, shape)?;
        bytes.validate(num_elements, &self.data_type)?;
        let bytes = bytes.into_fixed()?;

        Ok(self.bytes.encode(bytes, self.endian)?)
    }

    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let (num_elements, _) = decoded_len(&self.data_type, shape)?;
        let bytes = if self.passthrough {
            bytes
        } else {
            self.bytes.decode(bytes, self.endian)?
        };
        let bytes_decoded = ArrayBytes::Fixed(bytes);

        bytes_decoded.validate(num_elements, &self.data_type)?;

        Ok(bytes_decoded)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_handle,
            shape,
            &self.data_type,
            self.bytes.clone(),
            self.fill_value.as_ref(),
            self.endian,
        )))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_output_handle,
            shape,
            &self.data_type,
            self.bytes.clone(),
            self.fill_value.as_ref(),
            self.endian,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_handle,
            shape,
            &self.data_type,
            self.bytes.clone(),
            self.fill_value.as_ref(),
            self.endian,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_output_handle,
            shape,
            &self.data_type,
            self.bytes.clone(),
            self.fill_value.as_ref(),
            self.endian,
        )))
    }

    fn decode_into(
        &self,
        input: ArrayBytesDecodeIntoInput<'_>,
        shape: &[NonZeroU64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        match output_target {
            ArrayBytesDecodeIntoTarget::Fixed(output) if self.data_type.is_fixed() => {
                let (num_elements, expected_len) = decoded_len(&self.data_type, shape)?;
                if output.num_elements() != num_elements {
                    return Err("the decoded element count does not match the output target".into());
                }
                let in_place = self.in_place();
                let input = match input {
                    ArrayBytesDecodeIntoInput::Deferred(source)
                        if source.is_decode_into_efficient()
                            && in_place.is_some()
                            && source.decoded_representation()
                                == &BytesRepresentation::FixedSize(expected_len as u64) =>
                    {
                        if let Some(bytes) = output.as_mut_slice()
                            && bytes.len() == expected_len
                        {
                            source.decode_into(bytes, options, resources)?;
                            return match in_place {
                                Some(InPlace::Decode(codec, endian)) => {
                                    Ok(codec.decode_in_place(bytes, endian)?)
                                }
                                _ => Ok(()),
                            };
                        }
                        ArrayBytesDecodeIntoInput::Deferred(source)
                    }
                    input @ (ArrayBytesDecodeIntoInput::Bytes(_)
                    | ArrayBytesDecodeIntoInput::Deferred(_)) => input,
                };

                // Keep owned intermediates intact when direct output is unsuitable.
                let bytes = input.into_bytes(options, resources)?;
                if bytes.len() != expected_len {
                    return Err(InvalidBytesLengthError::new(bytes.len(), expected_len).into());
                }
                match in_place {
                    Some(InPlace::Copy) => Ok(output.copy_from_slice(&bytes)?),
                    Some(InPlace::Decode(codec, endian)) => {
                        if output.contiguous_bytes_len() == expected_len {
                            output.try_copy_from_slice_with(&bytes, decode_copy(codec, endian))
                        } else {
                            let decoded = codec.decode(bytes, endian)?;
                            Ok(output.copy_from_slice(&decoded)?)
                        }
                    }
                    None => {
                        let decoded = self.bytes.decode(bytes, self.endian)?;
                        Ok(output.copy_from_slice(&decoded)?)
                    }
                }
            }
            target @ (ArrayBytesDecodeIntoTarget::Fixed(_)
            | ArrayBytesDecodeIntoTarget::Optional(..)) => {
                let bytes = input.into_bytes(options, resources)?;
                let decoded = self.decode(bytes, shape, options, resources)?;
                decode_into_array_bytes_target(&decoded, target)
            }
        }
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        let (_, size) = decoded_size(&self.data_type, shape)?;
        Ok(BytesRepresentation::FixedSize(size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{ArrayBytesFixedDisjointView, ArraySubset, data_type};
    use unsafe_cell_slice::UnsafeCellSlice;
    fn decode_into_view(
        codec: &Arc<dyn ArrayToBytesCodecTraits>,
        input: ArrayBytesDecodeIntoInput<'_>,
        chunk_shape: &[NonZeroU64],
        array_shape: &[u64],
    ) -> (Vec<u16>, Result<(), CodecError>) {
        let num_elements = usize::try_from(array_shape.iter().product::<u64>()).unwrap();
        let mut output = vec![0u16; num_elements];
        let result = {
            let mut view = unsafe {
                // SAFETY: the only view of output, used synchronously.
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(bytemuck::cast_slice_mut(&mut output)),
                    size_of::<u16>(),
                    array_shape,
                    ArraySubset::new_with_shape(bytemuck::must_cast_slice::<_, u64>(chunk_shape)),
                )
            }
            .unwrap();
            codec.decode_into(
                input,
                chunk_shape,
                (&mut view).into(),
                &CodecOptions::default(),
                &ComputeResources::default(),
            )
        };
        (output, result)
    }

    #[test]
    fn decode_into_endianness_and_noncontiguous_views() {
        let values = [1u16, 2, 3, 4];
        let shape = [NonZeroU64::new(2).unwrap(); 2];
        for (endian, encoded) in [
            (Endianness::Little, values.map(u16::to_le_bytes).concat()),
            (Endianness::Big, values.map(u16::to_be_bytes).concat()),
        ] {
            let codec = BytesCodec::new(Some(endian))
                .with_context(
                    data_type::uint16(),
                    Some(FillValue::from(0u16)),
                    &CodecSpecificOptions::default(),
                )
                .unwrap();
            for (array_shape, expected) in [
                ([2, 2], values.to_vec()),
                ([2, 4], vec![1, 2, 0, 0, 3, 4, 0, 0]),
            ] {
                let (output, result) = decode_into_view(
                    &codec,
                    CowBytes::from(encoded.as_slice()).into(),
                    &shape,
                    &array_shape,
                );
                result.unwrap();
                assert_eq!(output, expected);

                let (output, result) = decode_into_view(
                    &codec,
                    CowBytes::from(&encoded[1..]).into(),
                    &shape,
                    &array_shape,
                );
                assert!(result.is_err());
                assert!(output.iter().all(|&value| value == 0));
            }
        }
        let codec = BytesCodec::new(None)
            .with_context(
                data_type::uint16(),
                Some(FillValue::from(0u16)),
                &CodecSpecificOptions::default(),
            )
            .unwrap();
        for array_shape in [[2, 2], [2, 4]] {
            let (output, result) = decode_into_view(
                &codec,
                CowBytes::from(&[1; 8][..]).into(),
                &shape,
                &array_shape,
            );
            assert!(result.is_err());
            assert!(output.iter().all(|&value| value == 0));
        }
    }

    #[test]
    fn decoded_size_overflow() {
        let codec = BytesCodec::default()
            .with_context(
                data_type::uint16(),
                Some(FillValue::from(0u16)),
                &CodecSpecificOptions::default(),
            )
            .unwrap();
        let options = CodecOptions::default();
        for shape in [
            vec![NonZeroU64::new(u64::MAX).unwrap()],
            vec![NonZeroU64::new(1 << 32).unwrap(); 2],
        ] {
            assert!(codec.encoded_representation(&shape).is_err());
            assert!(
                codec
                    .encode(
                        ArrayBytes::new_flen(vec![0; 2]),
                        &shape,
                        &options,
                        &ComputeResources::default()
                    )
                    .is_err()
            );
            assert!(
                codec
                    .decode(
                        CowBytes::from(&[0; 2][..]),
                        &shape,
                        &options,
                        &ComputeResources::default()
                    )
                    .is_err()
            );
        }
    }

    #[test]
    fn decode_passthrough() {
        let shape = [NonZeroU64::new(1).unwrap()];
        let options = CodecSpecificOptions::default();
        assert!(
            BytesCodec::new(None)
                .with_context(data_type::uint8(), Some(FillValue::from(0u8)), &options)
                .unwrap()
                .is_decode_passthrough(&shape)
                .unwrap()
        );
        assert!(
            BytesCodec::new(Some(Endianness::native()))
                .with_context(data_type::uint16(), Some(FillValue::from(0u16)), &options)
                .unwrap()
                .is_decode_passthrough(&shape)
                .unwrap()
        );
        let non_native = if Endianness::native() == Endianness::Little {
            Endianness::Big
        } else {
            Endianness::Little
        };
        assert!(
            !BytesCodec::new(Some(non_native))
                .with_context(data_type::uint16(), Some(FillValue::from(0u16)), &options)
                .unwrap()
                .is_decode_passthrough(&shape)
                .unwrap()
        );
        assert!(
            BytesCodec::new(None)
                .with_context(
                    data_type::uint8().to_optional(),
                    Some(FillValue::from(0u8).into_optional()),
                    &options
                )
                .is_err()
        );
        assert!(
            BytesCodec::new(None)
                .with_context(
                    data_type::bytes(),
                    Some(FillValue::from(Vec::<u8>::new())),
                    &options
                )
                .is_err()
        );
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn decode_deferred_zstd_input() {
        use zarrs_codec::{BytesDecodeSource, BytesToBytesCodecTraits};

        let values = [1u16, 2, 3, 4];
        let shape = [NonZeroU64::new(2).unwrap(); 2];
        let producer = crate::array::codec::ZstdCodec::new(1, false);
        let options = CodecOptions::default();
        for endian in [Endianness::Little, Endianness::Big] {
            let codec = BytesCodec::new(Some(endian))
                .with_context(
                    data_type::uint16(),
                    Some(FillValue::from(0u16)),
                    &CodecSpecificOptions::default(),
                )
                .unwrap();
            let encoded = if endian == Endianness::Little {
                values.map(u16::to_le_bytes).concat()
            } else {
                values.map(u16::to_be_bytes).concat()
            };
            let compressed = producer
                .encode(encoded.into(), &options, &ComputeResources::default())
                .unwrap();
            for representation in [
                BytesRepresentation::FixedSize(8),
                BytesRepresentation::BoundedSize(8),
            ] {
                let input = ArrayBytesDecodeIntoInput::Deferred(BytesDecodeSource::new(
                    &producer,
                    compressed.clone(),
                    representation,
                ));
                let (output, result) = decode_into_view(&codec, input, &shape, &[2, 2]);
                result.unwrap();
                assert_eq!(output, values);
            }
        }
    }
}
