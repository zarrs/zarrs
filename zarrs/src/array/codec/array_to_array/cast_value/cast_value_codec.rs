use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_codec::{
    ArrayBytes, ArrayBytesTarget, ArrayCodecTraits, ArrayPartialEncoderTraits,
    ArrayToArrayCodecGeometryIdentityTraits, ArrayToArrayCodecTraits, CodecCreateError, CodecError,
    CodecMetadataOptions, CodecOptions, CodecSpecificOptions, CodecTraits, ComputeResources,
    InvalidBytesLengthError, PartialDecoderCapability, PartialEncoderCapability,
    RecommendedConcurrency, Resources, SyncArrayPartialDecoderTraits,
    UnboundArrayToArrayCodecTraits, decode_into_array_bytes_target,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_data_type::codec_traits::cast_value::{
    CastValueDataTypeExt, CastValueDataTypeTraits, CastValueDataTypeTraitsArc, CastValueKernel,
    CastValueOutOfRangeMode as DataTypeCastValueOutOfRangeMode,
    CastValueRoundingMode as DataTypeCastValueRoundingMode, select_cast_kernel,
};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::cast_value::{
    CastValueCodecConfiguration, CastValueCodecConfigurationV1, CastValueOutOfRangeMode,
    CastValueRoundingMode,
};
use zarrs_plugin::{PluginCreateError, ZarrVersion};

use crate::array::{DataType, FillValue};
use crate::write_copy_of_slice;

pub(super) type ScalarMap = HashMap<Vec<u8>, Vec<u8>>;

/// A `cast_value` codec implementation.
#[derive(Clone, Debug)]
pub struct CastValueCodec {
    target_data_type: DataType,
    configuration: CastValueCodecConfigurationV1,
}

/// A `cast_value` codec implementation bound to a data type and fill value.
#[derive(Clone, Debug)]
struct CastValueCodecBound {
    data_type: DataType,
    /// The `cast_value` codec support of `data_type`.
    cast_value: CastValueDataTypeTraitsArc,
    fill_value: Option<FillValue>,
    encoded_data_type: DataType,
    /// The `cast_value` codec support of `encoded_data_type`.
    encoded_cast_value: CastValueDataTypeTraitsArc,
    encoded_fill_value: Option<FillValue>,
    encode_scalar_map: Option<ScalarMap>,
    decode_scalar_map: Option<ScalarMap>,
    rounding: DataTypeCastValueRoundingMode,
    out_of_range: Option<DataTypeCastValueOutOfRangeMode>,
    encode_kernel: Option<CastValueKernel>,
    decode_kernel: Option<CastValueKernel>,
}

impl CastValueCodec {
    /// Create a new `cast_value` codec from configuration.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if the configuration is unsupported.
    pub fn new_with_configuration(
        configuration: &CastValueCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            CastValueCodecConfiguration::V1(configuration) => {
                let target_data_type = DataType::from_metadata(&configuration.data_type)?;
                if !matches!(&target_data_type, DataType::Fixed(dt) if dt.size() > 0) {
                    return Err(PluginCreateError::Other(format!(
                        "cast_value requires non-zero fixed-size data types, got {target_data_type}"
                    )));
                }
                let target = target_data_type
                    .codec_castvalue()
                    .map_err(|err| PluginCreateError::Other(err.to_string()))?;
                validate_wrap_mode(&target_data_type, target, configuration.out_of_range)
                    .map_err(|err| PluginCreateError::Other(err.to_string()))?;
                Ok(Self {
                    target_data_type,
                    configuration: configuration.clone(),
                })
            }
            _ => Err(PluginCreateError::Other(
                "this cast_value codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for CastValueCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(CastValueCodecConfiguration::V1(self.configuration.clone()).into())
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
impl UnboundArrayToArrayCodecTraits for CastValueCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self as Arc<dyn UnboundArrayToArrayCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: Option<FillValue>,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        if !matches!(&data_type, DataType::Fixed(dt) if dt.size() > 0) {
            return Err(CodecCreateError::UnsupportedDataType(
                data_type,
                "cast_value".to_string(),
            ));
        }
        if !matches!(&self.target_data_type, DataType::Fixed(dt) if dt.size() > 0) {
            return Err(CodecCreateError::UnsupportedDataType(
                self.target_data_type.clone(),
                "cast_value".to_string(),
            ));
        }
        let cast_value = data_type.codec_castvalue_arc()?;
        let encoded_cast_value = self.target_data_type.codec_castvalue_arc()?;
        let (source, target) = (&*cast_value, &*encoded_cast_value);
        validate_wrap_mode(
            &self.target_data_type,
            target,
            self.configuration.out_of_range,
        )
        .map_err(CodecCreateError::other)?;
        let encode_scalar_map = self
            .configuration
            .scalar_map
            .as_ref()
            .and_then(|map| map.encode.as_ref())
            .map(|map| build_scalar_map(map, &data_type, &self.target_data_type))
            .transpose()
            .map_err(CodecCreateError::other)?;
        let decode_scalar_map = self
            .configuration
            .scalar_map
            .as_ref()
            .and_then(|map| map.decode.as_ref())
            .map(|map| build_scalar_map(map, &self.target_data_type, &data_type))
            .transpose()
            .map_err(CodecCreateError::other)?;

        let rounding = map_rounding(self.configuration.rounding);
        let out_of_range = map_out_of_range(self.configuration.out_of_range);
        // Select monomorphised bulk kernels when both data types expose a
        // numeric representation; otherwise the generic scalar path is used.
        let source_repr = source.cast_value_repr();
        let target_repr = target.cast_value_repr();
        let encode_kernel = source_repr.zip(target_repr).and_then(|(source, target)| {
            select_cast_kernel(source, target, rounding, out_of_range)
        });
        let decode_kernel = source_repr.zip(target_repr).and_then(|(source, target)| {
            select_cast_kernel(target, source, rounding, out_of_range)
        });
        let encoded_fill_value = fill_value
            .as_ref()
            .map(|fill_value| {
                let encoded = cast_element(
                    fill_value.as_ne_bytes(),
                    source,
                    target,
                    encode_scalar_map.as_ref(),
                    encode_kernel.as_ref(),
                    rounding,
                    out_of_range,
                )
                .map_err(CodecCreateError::other)?;
                let decoded = cast_element(
                    &encoded,
                    target,
                    source,
                    decode_scalar_map.as_ref(),
                    decode_kernel.as_ref(),
                    rounding,
                    out_of_range,
                )
                .map_err(CodecCreateError::other)?;
                if decoded != fill_value.as_ne_bytes() {
                    return Err(CodecCreateError::Other(
                        "cast_value fill value does not survive a round-trip cast".to_string(),
                    ));
                }
                Ok::<_, CodecCreateError>(FillValue::new(encoded))
            })
            .transpose()?;
        Ok(Arc::new(CastValueCodecBound {
            data_type,
            cast_value,
            fill_value,
            encoded_data_type: self.target_data_type.clone(),
            encoded_cast_value,
            encoded_fill_value,
            encode_scalar_map,
            decode_scalar_map,
            rounding,
            out_of_range,
            encode_kernel,
            decode_kernel,
        }))
    }
}

impl ArrayCodecTraits for CastValueCodecBound {
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
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl ArrayToArrayCodecGeometryIdentityTraits for CastValueCodecBound {}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for CastValueCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.encoded_data_type
    }

    fn encoded_fill_value(&self) -> Option<&FillValue> {
        self.encoded_fill_value.as_ref()
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(ArrayBytes::from(cast_bytes(
            &bytes.into_fixed()?,
            &CastBytesContext {
                source_data_type: &self.data_type,
                target_data_type: &self.encoded_data_type,
                source: &*self.cast_value,
                target: &*self.encoded_cast_value,
                scalar_map: self.encode_scalar_map.as_ref(),
                rounding: self.rounding,
                out_of_range: self.out_of_range,
                kernel: self.encode_kernel,
            },
        )?))
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(ArrayBytes::from(cast_bytes(
            &bytes.into_fixed()?,
            &CastBytesContext {
                source_data_type: &self.encoded_data_type,
                target_data_type: &self.data_type,
                source: &*self.encoded_cast_value,
                target: &*self.cast_value,
                scalar_map: self.decode_scalar_map.as_ref(),
                rounding: self.rounding,
                out_of_range: self.out_of_range,
                kernel: self.decode_kernel,
            },
        )?))
    }

    fn decode_into(
        &self,
        bytes: ArrayBytes<'_>,
        shape: &[NonZeroU64],
        target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        /// The number of elements cast at a time, which bounds the buffer they are cast into.
        const BLOCK_ELEMENTS: usize = 4096;
        /// The fewest elements in a region of the target to copy cast blocks into, which is slower per region than casting all elements and copying them.
        const MIN_REGION_ELEMENTS: usize = 64;
        if let (DataType::Fixed(source), DataType::Fixed(target_dt), ArrayBytesTarget::Fixed(view)) =
            (&self.encoded_data_type, &self.data_type, &mut *target)
            && (view.num_contiguous_elements() >= MIN_REGION_ELEMENTS
                || view.num_contiguous_elements() as u64 == view.num_elements())
        {
            let source_size = source.size();
            let target_size = target_dt.size();
            if source_size == 0 || target_size == 0 {
                return Err(CodecError::Other(
                    "cast_value requires non-zero fixed-size source and target data types"
                        .to_string(),
                ));
            }
            let bytes = bytes.into_fixed()?;
            if view.data_type_size() != target_size {
                return Err(
                    InvalidBytesLengthError::new(view.data_type_size(), target_size).into(),
                );
            }
            let Some(num_elements) = usize::try_from(view.num_elements()).ok() else {
                return Err(CodecError::from("cast_value element count overflows usize"));
            };
            let Some(expected_source_len) = num_elements.checked_mul(source_size) else {
                return Err(CodecError::from(
                    "cast_value encoded byte length overflows usize",
                ));
            };
            if bytes.len() != expected_source_len {
                return Err(InvalidBytesLengthError::new(bytes.len(), expected_source_len).into());
            }
            let context = CastBytesContext {
                source_data_type: &self.encoded_data_type,
                target_data_type: &self.data_type,
                source: &*self.encoded_cast_value,
                target: &*self.cast_value,
                scalar_map: self.decode_scalar_map.as_ref(),
                rounding: self.rounding,
                out_of_range: self.out_of_range,
                kernel: self.decode_kernel,
            };

            // A bulk kernel casts straight into the regions of the view
            if let (Some(kernel), None) = (&context.kernel, context.scalar_map) {
                let mut source: &[u8] = &bytes;
                let Some(region_source_len) =
                    view.num_contiguous_elements().checked_mul(source_size)
                else {
                    return Err(CodecError::from(
                        "cast_value encoded byte length overflows usize",
                    ));
                };
                // SAFETY: each region is cast from the next source elements, and `cast_into` writes every byte of its output on success.
                // There is a source element for every element of the view, so every region is initialised.
                return unsafe {
                    view.for_each_region_mut(|region| {
                        let (region_source, rest) = source.split_at(region_source_len);
                        source = rest;
                        kernel
                            .cast_into(region_source, region)
                            .map_err(|err| CodecError::Other(err.to_string()))
                    })
                };
            }

            // Otherwise elements are cast a block at a time into a buffer that stays in cache, and copied into the regions of the view
            let block_elements = num_elements.min(BLOCK_ELEMENTS);
            let Some(block_source_len) = block_elements.checked_mul(source_size) else {
                return Err(CodecError::from(
                    "cast_value block encoded byte length overflows usize",
                ));
            };
            let Some(block_target_len) = block_elements.checked_mul(target_size) else {
                return Err(CodecError::from(
                    "cast_value block decoded byte length overflows usize",
                ));
            };
            let mut source: &[u8] = &bytes;
            let mut cast = Vec::with_capacity(block_target_len);
            let mut cast_position = 0;
            // SAFETY: every region is written from the cast elements in order.
            // There is a source element for every element of the view, and each block is checked to cast to one element per source element, so the cast elements cover every region.
            unsafe {
                view.for_each_region_mut(|mut region| {
                    while !region.is_empty() {
                        if cast_position == cast.len() {
                            let (block, rest) = source.split_at(source.len().min(block_source_len));
                            source = rest;
                            cast.clear();
                            cast_position = 0;
                            cast_bytes_into(block, &context, &mut cast)?;
                            if block.is_empty()
                                || (block.len() / source_size)
                                    .checked_mul(target_size)
                                    .is_none_or(|expected| cast.len() != expected)
                            {
                                return Err(CodecError::Other(
                                    "cast_value cast an unexpected number of bytes".to_string(),
                                ));
                            }
                        }
                        let len = region.len().min(cast.len() - cast_position);
                        let (head, tail) = std::mem::take(&mut region).split_at_mut(len);
                        write_copy_of_slice(head, &cast[cast_position..cast_position + len]);
                        cast_position += len;
                        region = tail;
                    }
                    Ok(())
                })
            }
        } else {
            let decoded = self.decode(bytes, shape, options, resources)?;
            decode_into_array_bytes_target(&decoded, target)
        }
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn SyncArrayPartialDecoderTraits>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<Arc<dyn SyncArrayPartialDecoderTraits>, CodecError> {
        self.validate_partial_handle_data_type(input_handle.data_type())?;
        Ok(Arc::new(self.partial(input_handle)?))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        self.validate_partial_handle_data_type(input_output_handle.data_type())?;
        Ok(Arc::new(self.partial(input_output_handle)?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        self.validate_partial_handle_data_type(input_handle.data_type())?;
        Ok(Arc::new(self.partial(input_handle)?))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        self.validate_partial_handle_data_type(input_output_handle.data_type())?;
        Ok(Arc::new(self.partial(input_output_handle)?))
    }
}

impl CastValueCodecBound {
    fn validate_partial_handle_data_type(&self, data_type: &DataType) -> Result<(), CodecError> {
        if data_type != &self.encoded_data_type {
            return Err(CodecError::Other(format!(
                "cast_value partial handle data type {data_type} does not match encoded data type {}",
                self.encoded_data_type
            )));
        }
        Ok(())
    }

    fn partial<T: ?Sized>(
        &self,
        input_output_handle: Arc<T>,
    ) -> Result<super::cast_value_codec_partial::CastValueCodecPartial<T>, CodecError> {
        super::cast_value_codec_partial::CastValueCodecPartial::new(
            input_output_handle,
            &self.data_type,
            self.cast_value.clone(),
            &self.encoded_data_type,
            self.encoded_cast_value.clone(),
            self.encode_scalar_map.clone(),
            self.decode_scalar_map.clone(),
            self.rounding,
            self.out_of_range,
            self.encode_kernel,
            self.decode_kernel,
        )
    }
}

fn build_scalar_map(
    map: &[[zarrs_metadata::FillValueMetadata; 2]],
    source_data_type: &DataType,
    target_data_type: &DataType,
) -> Result<ScalarMap, CodecError> {
    let mut scalar_map = HashMap::with_capacity(map.len());
    for [key, value] in map {
        let key = source_data_type
            .fill_value_v3(key)
            .map_err(|err| CodecError::Other(err.to_string()))?
            .as_ne_bytes()
            .to_vec();
        let value = target_data_type
            .fill_value_v3(value)
            .map_err(|err| CodecError::Other(err.to_string()))?
            .as_ne_bytes()
            .to_vec();
        scalar_map.entry(key).or_insert(value);
    }
    Ok(scalar_map)
}

pub(super) struct CastBytesContext<'a> {
    pub(super) source_data_type: &'a DataType,
    pub(super) target_data_type: &'a DataType,
    pub(super) source: &'a dyn CastValueDataTypeTraits,
    pub(super) target: &'a dyn CastValueDataTypeTraits,
    pub(super) scalar_map: Option<&'a ScalarMap>,
    pub(super) rounding: DataTypeCastValueRoundingMode,
    pub(super) out_of_range: Option<DataTypeCastValueOutOfRangeMode>,
    pub(super) kernel: Option<CastValueKernel>,
}

pub(super) fn cast_bytes(
    bytes: &[u8],
    context: &CastBytesContext<'_>,
) -> Result<Vec<u8>, CodecError> {
    let mut output = Vec::new();
    cast_bytes_into(bytes, context, &mut output)?;
    Ok(output)
}

/// Cast the elements of `bytes` and append them to `output`, as in [`cast_bytes`].
fn cast_bytes_into(
    bytes: &[u8],
    context: &CastBytesContext<'_>,
    output: &mut Vec<u8>,
) -> Result<(), CodecError> {
    let (DataType::Fixed(source), DataType::Fixed(target)) =
        (context.source_data_type, context.target_data_type)
    else {
        return Err(CodecError::Other(
            "cast_value requires fixed-size source and target data types".to_string(),
        ));
    };
    let source_size = source.size();
    let target_size = target.size();
    if source_size == 0 || target_size == 0 {
        return Err(CodecError::Other(
            "cast_value requires non-zero fixed-size source and target data types".to_string(),
        ));
    }
    if !bytes.len().is_multiple_of(source_size) {
        return Err(CodecError::Other(
            "cast_value source bytes are not element aligned".to_string(),
        ));
    }
    let num_elements = bytes.len() / source_size;
    let Some(output_len) = num_elements.checked_mul(target_size) else {
        return Err(CodecError::from(
            "cast_value decoded byte length overflows usize",
        ));
    };
    output.reserve(output_len);
    if let Some(scalar_map) = context.scalar_map {
        for source_bytes in bytes.chunks_exact(source_size) {
            if let Some(mapped) = scalar_map.get(source_bytes) {
                output.extend_from_slice(mapped);
            } else {
                cast_element_into(
                    source_bytes,
                    context.source,
                    context.target,
                    context.kernel.as_ref(),
                    context.rounding,
                    context.out_of_range,
                    output,
                )?;
            }
        }
    } else if let Some(kernel) = &context.kernel {
        kernel
            .cast(bytes, output)
            .map_err(|err| CodecError::Other(err.to_string()))?;
    } else {
        context
            .source
            .cast_value_cast_slice(
                bytes,
                source_size,
                context.target,
                context.rounding,
                context.out_of_range,
                output,
            )
            .map_err(|err| CodecError::Other(err.to_string()))?;
    }
    Ok(())
}

/// Cast one element via the kernel if selected, else the generic scalar path.
fn cast_element_into(
    source_bytes: &[u8],
    source: &dyn CastValueDataTypeTraits,
    target: &dyn CastValueDataTypeTraits,
    kernel: Option<&CastValueKernel>,
    rounding: DataTypeCastValueRoundingMode,
    out_of_range: Option<DataTypeCastValueOutOfRangeMode>,
    output: &mut Vec<u8>,
) -> Result<(), CodecError> {
    if let Some(kernel) = kernel {
        kernel.cast(source_bytes, output)
    } else {
        source.cast_value_cast(source_bytes, target, rounding, out_of_range, output)
    }
    .map_err(|err| CodecError::Other(err.to_string()))
}

fn cast_element(
    source_bytes: &[u8],
    source: &dyn CastValueDataTypeTraits,
    target: &dyn CastValueDataTypeTraits,
    scalar_map: Option<&ScalarMap>,
    kernel: Option<&CastValueKernel>,
    rounding: DataTypeCastValueRoundingMode,
    out_of_range: Option<DataTypeCastValueOutOfRangeMode>,
) -> Result<Vec<u8>, CodecError> {
    if let Some(mapped) = scalar_map.and_then(|map| map.get(source_bytes)) {
        return Ok(mapped.clone());
    }
    let mut output = Vec::new();
    cast_element_into(
        source_bytes,
        source,
        target,
        kernel,
        rounding,
        out_of_range,
        &mut output,
    )?;
    Ok(output)
}

fn map_rounding(rounding: Option<CastValueRoundingMode>) -> DataTypeCastValueRoundingMode {
    match rounding.unwrap_or_default() {
        CastValueRoundingMode::NearestEven => DataTypeCastValueRoundingMode::NearestEven,
        CastValueRoundingMode::TowardsZero => DataTypeCastValueRoundingMode::TowardsZero,
        CastValueRoundingMode::TowardsPositive => DataTypeCastValueRoundingMode::TowardsPositive,
        CastValueRoundingMode::TowardsNegative => DataTypeCastValueRoundingMode::TowardsNegative,
        CastValueRoundingMode::NearestAway => DataTypeCastValueRoundingMode::NearestAway,
    }
}

fn map_out_of_range(
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Option<DataTypeCastValueOutOfRangeMode> {
    match out_of_range {
        Some(CastValueOutOfRangeMode::Clamp) => Some(DataTypeCastValueOutOfRangeMode::Clamp),
        Some(CastValueOutOfRangeMode::Wrap) => Some(DataTypeCastValueOutOfRangeMode::Wrap),
        None => None,
    }
}

fn validate_wrap_mode(
    data_type: &DataType,
    target: &dyn CastValueDataTypeTraits,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<(), CodecError> {
    if matches!(out_of_range, Some(CastValueOutOfRangeMode::Wrap))
        && !target.cast_value_is_integral()
    {
        return Err(CodecError::UnsupportedDataType(
            data_type.clone(),
            "cast_value out_of_range=wrap".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::borrow::Cow;

    use zarrs_data_type::codec_traits::cast_value::{CastValueError, CastValueScalar};
    use zarrs_data_type::{
        DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypeTraits, FixedDataTypeTraits,
    };
    use zarrs_metadata::FillValueMetadata;
    use zarrs_plugin::ExtensionName;

    use super::*;
    use crate::array::data_type;

    #[derive(Debug)]
    struct ZeroSizedCastValue;

    impl ExtensionName for ZeroSizedCastValue {
        fn name(&self, _version: ZarrVersion) -> Option<Cow<'static, str>> {
            Some(Cow::Borrowed("test.zero_sized_cast_value"))
        }
    }

    impl FixedDataTypeTraits for ZeroSizedCastValue {
        fn size(&self) -> usize {
            0
        }
    }

    impl DataTypeTraits for ZeroSizedCastValue {
        fn into_data_type(self: Arc<Self>) -> DataType {
            DataType::Fixed(self)
        }

        fn configuration(&self, _version: ZarrVersion) -> Configuration {
            Configuration::default()
        }

        fn fill_value(
            &self,
            _metadata: &FillValueMetadata,
            _version: ZarrVersion,
        ) -> Result<FillValue, DataTypeFillValueMetadataError> {
            Ok(FillValue::new(Vec::new()))
        }

        fn metadata_fill_value(
            &self,
            _fill_value: &FillValue,
        ) -> Result<FillValueMetadata, DataTypeFillValueError> {
            Ok(FillValueMetadata::Null)
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl CastValueDataTypeTraits for ZeroSizedCastValue {
        fn cast_value_is_integral(&self) -> bool {
            true
        }

        fn cast_value_read(&self, _bytes: &[u8]) -> Result<CastValueScalar, CastValueError> {
            Ok(CastValueScalar::Unsigned(0))
        }

        fn cast_value_write(
            &self,
            _value: CastValueScalar,
            _rounding: DataTypeCastValueRoundingMode,
            _out_of_range: Option<DataTypeCastValueOutOfRangeMode>,
            _output: &mut Vec<u8>,
        ) -> Result<(), CastValueError> {
            Ok(())
        }
    }

    #[test]
    fn cast_value_rejects_zero_sized_fixed_data_type() {
        let zero_dt = DataType::Fixed(Arc::new(ZeroSizedCastValue));
        let u8_dt = data_type::uint8();
        let u8_trait = u8_dt.codec_castvalue().unwrap();

        let configuration: CastValueCodecConfiguration =
            serde_json::from_str(r#"{"data_type": "uint8"}"#).unwrap();
        let codec = CastValueCodec::new_with_configuration(&configuration).unwrap();
        assert!(matches!(
            codec.with_context(zero_dt.clone(), None, &CodecSpecificOptions::default()),
            Err(CodecCreateError::UnsupportedDataType(..))
        ));

        let mut out = Vec::new();
        let ctx_zero_src = CastBytesContext {
            source_data_type: &zero_dt,
            target_data_type: &u8_dt,
            source: &ZeroSizedCastValue,
            target: u8_trait,
            scalar_map: None,
            rounding: DataTypeCastValueRoundingMode::NearestEven,
            out_of_range: None,
            kernel: None,
        };
        assert!(cast_bytes_into(&[], &ctx_zero_src, &mut out).is_err());

        let ctx_zero_dst = CastBytesContext {
            source_data_type: &u8_dt,
            target_data_type: &zero_dt,
            source: u8_trait,
            target: &ZeroSizedCastValue,
            scalar_map: None,
            rounding: DataTypeCastValueRoundingMode::NearestEven,
            out_of_range: None,
            kernel: None,
        };
        assert!(cast_bytes_into(&[1u8], &ctx_zero_dst, &mut out).is_err());
    }
}
