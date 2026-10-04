// Note: No validation that this codec is created *without* a specified endianness for multi-byte data types.

use std::sync::Arc;

use zarrs_plugin::{ExtensionAliasesV3, PluginCreateError, ZarrVersion};

use super::{
    BytesCodecConfiguration, BytesCodecConfigurationV1, BytesDataTypeExt, Endianness,
    bytes_codec_partial,
};
use crate::array::{
    ArrayBytes, BytesRepresentation, ChunkShapeTraits, CowBytes, DataType, DataTypeSize, FillValue,
};
use std::num::NonZeroU64;
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderTraits,
    ArrayPartialEncoderTraits, ArrayToBytesCodecTraits, BytesPartialDecoderTraits,
    BytesPartialEncoderTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecSpecificOptions, CodecTraits, InvalidBytesLengthError, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency, UnboundArrayToBytesCodecTraits,
    decode_into_array_bytes_target,
};
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
#[derive(Debug, Clone)]
struct BytesCodecBound {
    endian: Option<Endianness>,
    data_type: DataType,
    fill_value: FillValue,
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
        fill_value: FillValue,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        if data_type.is_optional() {
            return Err(CodecCreateError::UnsupportedDataType(
                data_type,
                Self::aliases_v3().default_name.to_string(),
            ));
        }
        data_type.codec_bytes()?;
        Ok(Arc::new(BytesCodecBound {
            endian: self.endian,
            data_type,
            fill_value,
        }))
    }
}

impl ArrayCodecTraits for BytesCodecBound {
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

    fn supports_decode_in_place(&self) -> bool {
        // An empty slice checks that the endianness is specified where it is required, so that
        // callers do not write to an output before finding that decoding in place fails
        self.data_type.is_fixed()
            && !self.data_type.is_optional()
            && self.data_type.codec_bytes().is_ok_and(|codec| {
                codec.is_decode_in_place_efficient()
                    && codec.decode_in_place(&mut [], self.endian).is_ok()
            })
    }

    fn decode_in_place(&self, bytes: &mut [u8]) -> Result<(), CodecError> {
        Ok(self
            .data_type
            .codec_bytes()?
            .decode_in_place(bytes, self.endian)?)
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<CowBytes<'a>, CodecError> {
        let num_elements = shape.iter().map(|d| d.get()).product::<u64>();
        bytes.validate(num_elements, &self.data_type)?;
        let bytes = bytes.into_fixed()?;

        Ok(self.data_type.codec_bytes()?.encode(bytes, self.endian)?)
    }

    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let bytes = self.data_type.codec_bytes()?.decode(bytes, self.endian)?;
        let bytes_decoded = ArrayBytes::Fixed(bytes);

        let num_elements = shape.iter().map(|d| d.get()).product::<u64>();
        bytes_decoded.validate(num_elements, &self.data_type)?;

        Ok(bytes_decoded)
    }

    fn decode_into(
        &self,
        bytes: CowBytes<'_>,
        shape: &[NonZeroU64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        match (output_target, self.data_type.size()) {
            (ArrayBytesDecodeIntoTarget::Fixed(output), DataTypeSize::Fixed(data_type_size)) => {
                let expected_len = shape
                    .iter()
                    .try_fold(data_type_size as u64, |len, d| len.checked_mul(d.get()))
                    .and_then(|len| usize::try_from(len).ok())
                    .ok_or("the decoded chunk size in bytes overflows usize")?;
                if bytes.len() != expected_len {
                    return Err(InvalidBytesLengthError::new(bytes.len(), expected_len).into());
                }
                // Change the endianness as the bytes are copied into the output, rather than
                // decoding to an intermediate allocation that is then copied
                let codec = self.data_type.codec_bytes()?;
                if codec.is_decode_passthrough(self.endian) {
                    Ok(output.copy_from_slice(&bytes)?)
                } else if codec.is_decode_in_place_efficient() {
                    // Check that the endianness is specified before writing to the output
                    codec.decode_in_place(&mut [], self.endian)?;
                    output.try_copy_from_slice_with(&bytes, |source, destination| {
                        destination.copy_from_slice(source);
                        Ok::<_, CodecError>(codec.decode_in_place(destination, self.endian)?)
                    })
                } else {
                    // Decoding in place would allocate for each contiguous region
                    let decoded = codec.decode(bytes, self.endian)?;
                    Ok(output.copy_from_slice(&decoded)?)
                }
            }
            (output_target, _) => {
                let bytes = self.decode(bytes, shape, options)?;
                decode_into_array_bytes_target(&bytes, output_target)
            }
        }
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_handle,
            shape,
            &self.data_type,
            &self.fill_value,
            self.endian,
        )))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_output_handle,
            shape,
            &self.data_type,
            &self.fill_value,
            self.endian,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_handle,
            shape,
            &self.data_type,
            &self.fill_value,
            self.endian,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(bytes_codec_partial::BytesCodecPartial::new(
            input_output_handle,
            shape,
            &self.data_type,
            &self.fill_value,
            self.endian,
        )))
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        match self.data_type.size() {
            DataTypeSize::Variable => Err(CodecError::UnsupportedDataType(
                self.data_type.clone(),
                BytesCodec::aliases_v3().default_name.to_string(),
            )),
            DataTypeSize::Fixed(data_type_size) => Ok(BytesRepresentation::FixedSize(
                shape.num_elements_u64() * data_type_size as u64,
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::codec::array_to_bytes::bytes::non_native_endianness;
    use crate::array::{ArrayBytesFixedDisjointView, ArraySubset, data_type};
    use unsafe_cell_slice::UnsafeCellSlice;

    /// Decode `encoded` (`u16` elements) with the `bytes` codec into the subset of an array of
    /// `array_shape` at the origin that has the shape of the chunk.
    ///
    /// Returns the array, which is written to partially if decoding fails, and the result of decoding.
    fn decode_into_view_array(
        endianness: Option<Endianness>,
        chunk_shape: &[u64],
        array_shape: &[u64],
        encoded: &[u8],
    ) -> (Vec<u16>, Result<(), CodecError>) {
        let codec = BytesCodec::new(endianness)
            .with_context(
                data_type::uint16(),
                FillValue::from(0u16),
                &CodecSpecificOptions::default(),
            )
            .unwrap();
        let chunk_shape_nz: Vec<NonZeroU64> = chunk_shape
            .iter()
            .map(|&s| NonZeroU64::new(s).unwrap())
            .collect();
        let mut output = vec![0u16; usize::try_from(array_shape.iter().product::<u64>()).unwrap()];
        let result = {
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(bytemuck::cast_slice_mut(&mut output)),
                    size_of::<u16>(),
                    array_shape,
                    ArraySubset::new_with_shape(chunk_shape.to_vec()),
                )
            }
            .unwrap();
            codec.decode_into(
                CowBytes::from(encoded),
                &chunk_shape_nz,
                (&mut view).into(),
                &CodecOptions::default(),
            )
        };
        (output, result)
    }

    /// Decode `encoded` (`u16` elements) with the `bytes` codec into a view of an array.
    fn decode_into_view(
        endianness: Option<Endianness>,
        chunk_shape: &[u64],
        array_shape: &[u64],
        encoded: &[u8],
    ) -> Result<Vec<u16>, CodecError> {
        let (output, result) =
            decode_into_view_array(endianness, chunk_shape, array_shape, encoded);
        result.map(|()| output)
    }

    #[test]
    fn decode_in_place() {
        let options = CodecSpecificOptions::default();
        let codec = |endianness| {
            BytesCodec::new(endianness)
                .with_context(data_type::uint16(), FillValue::from(0u16), &options)
                .unwrap()
        };
        let mut bytes = [1, 2, 3, 4];
        let native = codec(Some(Endianness::native()));
        assert!(native.supports_decode_in_place());
        native.decode_in_place(&mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
        let non_native = codec(Some(non_native_endianness()));
        assert!(non_native.supports_decode_in_place());
        non_native.decode_in_place(&mut bytes).unwrap();
        assert_eq!(bytes, [2, 1, 4, 3]);
        // Endianness is required for multi-byte data types
        assert!(codec(None).decode_in_place(&mut bytes).is_err());
        assert!(!codec(None).supports_decode_in_place());
    }

    #[test]
    fn decode_into_endianness() {
        let values = [1u16, 2, 3, 4];
        for (endianness, encoded) in [
            (Endianness::Little, values.map(u16::to_le_bytes).concat()),
            (Endianness::Big, values.map(u16::to_be_bytes).concat()),
        ] {
            // A contiguous view
            assert_eq!(
                decode_into_view(Some(endianness), &[2, 2], &[2, 2], &encoded).unwrap(),
                values
            );
            // A view with two contiguous regions
            assert_eq!(
                decode_into_view(Some(endianness), &[2, 2], &[2, 4], &encoded).unwrap(),
                [1, 2, 0, 0, 3, 4, 0, 0]
            );
            // The encoded bytes are not the length of the chunk
            assert!(decode_into_view(Some(endianness), &[2, 2], &[2, 2], &encoded[1..]).is_err());
            assert!(decode_into_view(Some(endianness), &[2, 1], &[2, 2], &encoded).is_err());
        }
        // Endianness is required for multi-byte data types, and the output is not written to
        for array_shape in [[2, 2], [2, 4]] {
            let (output, result) = decode_into_view_array(None, &[2, 2], &array_shape, &[1; 8]);
            assert!(result.is_err());
            assert!(output.iter().all(|&value| value == 0));
        }
    }

    #[test]
    fn decode_into_size_overflow() {
        let codec = BytesCodec::default()
            .with_context(
                data_type::uint16(),
                FillValue::from(0u16),
                &CodecSpecificOptions::default(),
            )
            .unwrap();
        // The size of the decoded chunk in bytes overflows, which is an error rather than a panic
        let shape = [NonZeroU64::new(1 << 32).unwrap(); 2];
        let mut output = [0u16; 1];
        let mut view = unsafe {
            ArrayBytesFixedDisjointView::new(
                UnsafeCellSlice::new(bytemuck::cast_slice_mut(&mut output)),
                size_of::<u16>(),
                &[1],
                ArraySubset::new_with_shape(vec![1]),
            )
        }
        .unwrap();
        assert!(
            codec
                .decode_into(
                    CowBytes::from(&[0u8; 2][..]),
                    &shape,
                    (&mut view).into(),
                    &CodecOptions::default(),
                )
                .is_err()
        );
    }

    #[test]
    fn data_type_decode_passthrough() {
        let options = CodecSpecificOptions::default();
        let uint8 = data_type::uint8();
        let uint16 = data_type::uint16();
        assert!(uint8.codec_bytes().unwrap().is_decode_passthrough(None));
        assert!(
            uint16
                .codec_bytes()
                .unwrap()
                .is_decode_passthrough(Some(Endianness::native()))
        );
        assert!(
            !uint16
                .codec_bytes()
                .unwrap()
                .is_decode_passthrough(Some(non_native_endianness()))
        );
        assert!(!uint16.codec_bytes().unwrap().is_decode_passthrough(None));
        assert!(
            BytesCodec::new(None)
                .with_context(
                    data_type::uint8().to_optional(),
                    FillValue::from(0u8).into_optional(),
                    &options
                )
                .is_err()
        );
        assert!(
            BytesCodec::new(None)
                .with_context(
                    data_type::bytes(),
                    FillValue::from(Vec::<u8>::new()),
                    &options
                )
                .is_err()
        );
    }
}
