use std::mem::MaybeUninit;
use std::sync::Arc;

use zarrs_plugin::{ExtensionAliasesV3, PluginCreateError, ZarrVersion};

use super::{
    FixedScaleOffsetCodecConfiguration, FixedScaleOffsetCodecConfigurationNumcodecsF64,
    FixedScaleOffsetDataTypeExt, FixedScaleOffsetElementType,
};
use crate::array::{DataType, FillValue};
use crate::convert::data_type_metadata_v2_to_v3;
use std::num::NonZeroU64;
use zarrs_codec::{
    ArrayBytes, ArrayBytesFixedDisjointView, ArrayBytesTarget, ArrayCodecTraits,
    ArrayToArrayCodecGeometryIdentityTraits, ArrayToArrayCodecTraits, CodecCreateError, CodecError,
    CodecMetadataOptions, CodecOptions, CodecSpecificOptions, CodecTraits, ComputeResources,
    ExpectedOptionalBytesError, InvalidBytesLengthError, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency, UnboundArrayToArrayCodecTraits,
    decode_into_array_bytes_target,
};
use zarrs_metadata::Configuration;
use zarrs_metadata::v2::DataTypeMetadataV2;

/// A `fixedscaleoffset` codec implementation.
#[derive(Clone, Debug)]
pub struct FixedScaleOffsetCodec {
    offset: f64,
    scale: f64,
    dtype_str: String,
    astype_str: Option<String>,
    dtype: DataType,
    astype: Option<DataType>,
}

/// A `fixedscaleoffset` codec implementation bound to a data type and fill value.
#[derive(Clone, Debug)]
struct FixedScaleOffsetCodecBound {
    offset: f64,
    scale: f64,
    element_type: FixedScaleOffsetElementType,
    encoded_element_type: FixedScaleOffsetElementType,
    data_type: DataType,
    fill_value: Option<FillValue>,
    encoded_data_type: DataType,
    encoded_fill_value: Option<FillValue>,
}

/// Add a byteorder to a numpy data type string (e.g. `f8` -> `<f8`, `i1` -> `|i1`) if it does not have one.
fn add_byteorder_to_dtype(dtype: &str) -> String {
    if dtype.starts_with(['<', '>', '|']) {
        dtype.to_string()
    } else if matches!(dtype, "b1" | "i1" | "u1") {
        // Single byte data types are not applicable to byte ordering
        format!("|{dtype}")
    } else {
        format!("<{dtype}")
    }
}

impl FixedScaleOffsetCodec {
    fn new(
        offset: f64,
        scale: f64,
        dtype_str: &str,
        astype_str: Option<&str>,
    ) -> Result<Self, PluginCreateError> {
        // Add a byteorder to the data type name, byteorder may be omitted
        // FixedScaleOffsets permits `dtype` / `astype` with and without a byteoder character, but it is irrelevant
        let dtype = add_byteorder_to_dtype(dtype_str);
        let astype = astype_str.map(add_byteorder_to_dtype);

        // Get the data type metadata
        let dtype = DataTypeMetadataV2::Simple(dtype);
        let astype = astype
            .as_ref()
            .map(|dtype| DataTypeMetadataV2::Simple(dtype.clone()));

        // Convert to a V3 data type
        let dtype_err = |_| {
            PluginCreateError::Other(
                "fixedscaleoffset cannot interpret Zarr V2 data type as V3 equivalent".to_string(),
            )
        };
        let dtype =
            DataType::from_metadata(&data_type_metadata_v2_to_v3(&dtype).map_err(dtype_err)?)?;
        let astype = if let Some(astype) = astype {
            Some(DataType::from_metadata(
                &data_type_metadata_v2_to_v3(&astype).map_err(dtype_err)?,
            )?)
        } else {
            None
        };

        Ok(Self {
            offset,
            scale,
            dtype,
            astype,
            dtype_str: dtype_str.to_string(),
            astype_str: astype_str.map(str::to_string),
        })
    }

    /// Create a new `fixedscaleoffset` codec from a configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &FixedScaleOffsetCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            FixedScaleOffsetCodecConfiguration::NumcodecsF64(configuration) => Self::new(
                configuration.offset,
                configuration.scale,
                &configuration.dtype,
                configuration.astype.as_deref(),
            ),
            #[allow(deprecated)]
            FixedScaleOffsetCodecConfiguration::Numcodecs(configuration) => Self::new(
                f64::from(configuration.offset),
                f64::from(configuration.scale),
                &configuration.dtype,
                configuration.astype.as_deref(),
            ),
            _ => Err(PluginCreateError::Other(
                "this fixedscaleoffset codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for FixedScaleOffsetCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = FixedScaleOffsetCodecConfiguration::NumcodecsF64(
            FixedScaleOffsetCodecConfigurationNumcodecsF64 {
                offset: self.offset,
                scale: self.scale,
                dtype: self.dtype_str.clone(),
                astype: self.astype_str.clone(),
            },
        );
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        // NOTE: the default array-to-array partial decoder supports partial read/decode
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false, // TODO
        }
    }
}

fn get_element_type(
    data_type: &DataType,
) -> Result<FixedScaleOffsetElementType, zarrs_data_type::DataTypeCodecError> {
    let fso = data_type.codec_fixedscaleoffset()?;
    Ok(fso.fixedscaleoffset_element_type())
}

/// Bind `$T` to the Rust type of the `$element_type` elements, and optionally `$F` to the float type that `numpy` computes in for them (`f32` for `float32`, otherwise `f64`) or `const $IS_INT` to whether `$T` is an integer type.
macro_rules! with_element_type {
    ($element_type:expr, $T:ident $(, float: $F:ident)? $(, is_int: $IS_INT:ident)?, $body:block) => {
        match $element_type {
            FixedScaleOffsetElementType::I8 => {
                type $T = i8;
                $(type $F = f64;)?
                $(const $IS_INT: bool = true;)?
                $body
            }
            FixedScaleOffsetElementType::I16 => {
                type $T = i16;
                $(type $F = f64;)?
                $(const $IS_INT: bool = true;)?
                $body
            }
            FixedScaleOffsetElementType::I32 => {
                type $T = i32;
                $(type $F = f64;)?
                $(const $IS_INT: bool = true;)?
                $body
            }
            FixedScaleOffsetElementType::I64 => {
                type $T = i64;
                $(type $F = f64;)?
                $(const $IS_INT: bool = true;)?
                $body
            }
            FixedScaleOffsetElementType::U8 => {
                type $T = u8;
                $(type $F = f64;)?
                $(const $IS_INT: bool = true;)?
                $body
            }
            FixedScaleOffsetElementType::U16 => {
                type $T = u16;
                $(type $F = f64;)?
                $(const $IS_INT: bool = true;)?
                $body
            }
            FixedScaleOffsetElementType::U32 => {
                type $T = u32;
                $(type $F = f64;)?
                $(const $IS_INT: bool = true;)?
                $body
            }
            FixedScaleOffsetElementType::U64 => {
                type $T = u64;
                $(type $F = f64;)?
                $(const $IS_INT: bool = true;)?
                $body
            }
            FixedScaleOffsetElementType::F32 => {
                type $T = f32;
                $(type $F = f32;)?
                $(const $IS_INT: bool = false;)?
                $body
            }
            FixedScaleOffsetElementType::F64 => {
                type $T = f64;
                $(type $F = f64;)?
                $(const $IS_INT: bool = false;)?
                $body
            }
        }
    };
}

/// Transform a slice of encoded/decoded element bytes into a destination slice of the same element count.
type TransformSliceFn = fn(&[u8], &mut [MaybeUninit<u8>], f64, f64);

/// Select a monomorphised slice transform function from `$from` to `$to` using `$transform`, a function of an element `x` and `offset` and `scale` (`$offset` and `$scale` as the float type `x` is computed in).
///
/// As in `numpy`, elements are computed in [`f32`] if `from` is `float32`, otherwise [`f64`].
/// Integer elements of `to` are rounded to the nearest integer (ties to even) and saturated to the range of the type.
macro_rules! transform_slice_fn {
    ($from:expr, $to:expr, |$x:ident, $o:ident, $s:ident| $transform:expr) => {{
        with_element_type!($from, T, float: F, {
            with_element_type!($to, U, is_int: TO_INTEGER, {
                fn transform_elements(
                    input: &[u8],
                    output: &mut [MaybeUninit<u8>],
                    offset: f64,
                    scale: f64,
                ) {
                    let ($o, $s) = (offset as F, scale as F);
                    let input = input.as_chunks::<{ size_of::<T>() }>().0;
                    let output = output.as_chunks_mut::<{ size_of::<U>() }>().0;
                    // Writing through a pre-sized slice (rather than pushing each element) lets the loop vectorise
                    for (transformed, element) in std::iter::zip(output, input) {
                        let $x = <T>::from_ne_bytes(*element) as F;
                        let value = $transform;
                        let value = if TO_INTEGER {
                            value.round_ties_even()
                        } else {
                            value
                        };
                        *transformed = (value as U).to_ne_bytes().map(MaybeUninit::new);
                    }
                }
                transform_elements as TransformSliceFn
            })
        })
    }};
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::unnecessary_cast
)]
fn encode_slice_fn(
    element_type: FixedScaleOffsetElementType,
    encoded_element_type: FixedScaleOffsetElementType,
) -> TransformSliceFn {
    transform_slice_fn!(element_type, encoded_element_type, |x, offset, scale| ((x
        - offset)
        * scale)
        .round_ties_even())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss,
    clippy::unnecessary_cast
)]
fn decode_slice_fn(
    encoded_element_type: FixedScaleOffsetElementType,
    element_type: FixedScaleOffsetElementType,
) -> TransformSliceFn {
    transform_slice_fn!(encoded_element_type, element_type, |x, offset, scale| x
        / scale
        + offset)
}

fn transform_to_vec(
    bytes: &[u8],
    from: FixedScaleOffsetElementType,
    to: FixedScaleOffsetElementType,
    offset: f64,
    scale: f64,
    transform_fn: TransformSliceFn,
) -> Result<Vec<u8>, CodecError> {
    if !bytes.len().is_multiple_of(from.size()) {
        return Err(CodecError::Other(format!(
            "fixedscaleoffset input length {} is not a multiple of the element size {}",
            bytes.len(),
            from.size(),
        )));
    }
    let num_elements = bytes.len() / from.size();
    let Some(len) = num_elements.checked_mul(to.size()) else {
        return Err(CodecError::from(
            "fixedscaleoffset output byte length overflows usize",
        ));
    };
    let mut out: Vec<u8> = Vec::with_capacity(len);
    transform_fn(bytes, &mut out.spare_capacity_mut()[..len], offset, scale);
    // SAFETY: `bytes` holds at least `num_elements` elements of `from`, and `len == num_elements * to.size()`, so `transform_fn` initialises all `len <= capacity` bytes.
    unsafe { out.set_len(len) };
    Ok(out)
}

/// Encode elements of `element_type` as elements of `encoded_element_type`: `round((x - offset) * scale)`.
///
/// As in `numcodecs`, the transform is computed in [`f32`] for `float32` elements and otherwise [`f64`], and rounded to the nearest integer with ties to even (`numpy.around`).
/// Values out of the range of `encoded_element_type` are saturated (unspecified in `numcodecs`).
fn do_encode(
    bytes: ArrayBytes<'_>,
    element_type: FixedScaleOffsetElementType,
    offset: f64,
    scale: f64,
    encoded_element_type: FixedScaleOffsetElementType,
) -> Result<ArrayBytes<'_>, CodecError> {
    let bytes = bytes.into_fixed()?;
    let encoded = transform_to_vec(
        &bytes,
        element_type,
        encoded_element_type,
        offset,
        scale,
        encode_slice_fn(element_type, encoded_element_type),
    )?;
    Ok(encoded.into())
}

/// Decode elements of `encoded_element_type` as elements of `element_type`: `x / scale + offset`.
///
/// As in `numcodecs`, the transform is computed in [`f32`] for `float32` encoded elements and otherwise [`f64`].
/// `numcodecs` truncates decoded integer elements, whereas they are rounded to the nearest integer (ties to even) here, so they may differ by one (the transform is lossy).
fn do_decode(
    bytes: ArrayBytes<'_>,
    element_type: FixedScaleOffsetElementType,
    offset: f64,
    scale: f64,
    encoded_element_type: FixedScaleOffsetElementType,
) -> Result<ArrayBytes<'_>, CodecError> {
    let bytes = bytes.into_fixed()?;
    let decoded = transform_to_vec(
        &bytes,
        encoded_element_type,
        element_type,
        offset,
        scale,
        decode_slice_fn(encoded_element_type, element_type),
    )?;
    Ok(decoded.into())
}

/// Decode the elements of `encoded_element_type` in `encoded` into the elements of `element_type` of `view` (in C order of the view), as in [`do_decode`].
///
/// # Errors
/// Returns a [`CodecError`] if `encoded` does not hold one element per element of `view`, or the elements of `view` are not of `element_type`.
fn decode_into_view(
    encoded: &[u8],
    view: &mut ArrayBytesFixedDisjointView<'_>,
    element_type: FixedScaleOffsetElementType,
    offset: f64,
    scale: f64,
    encoded_element_type: FixedScaleOffsetElementType,
) -> Result<(), CodecError> {
    if view.data_type_size() != element_type.size() {
        return Err(
            InvalidBytesLengthError::new(view.data_type_size(), element_type.size()).into(),
        );
    }
    let Some(expected_len) = usize::try_from(view.num_elements())
        .ok()
        .and_then(|num_elements| num_elements.checked_mul(encoded_element_type.size()))
    else {
        return Err(CodecError::from(
            "fixedscaleoffset encoded byte length overflows usize",
        ));
    };
    if encoded.len() != expected_len {
        return Err(InvalidBytesLengthError::new(encoded.len(), expected_len).into());
    }
    let Some(region_encoded_len) = view
        .num_contiguous_elements()
        .checked_mul(encoded_element_type.size())
    else {
        return Err(CodecError::from(
            "fixedscaleoffset encoded byte length overflows usize",
        ));
    };
    let decode_fn = decode_slice_fn(encoded_element_type, element_type);
    let mut encoded = encoded;
    // SAFETY: each region is a whole number of `element_type` elements (the element size of the view), and `region_encoded` has the same number of `encoded_element_type` elements, so `decode_fn` initialises every byte of each region.
    unsafe {
        view.for_each_region_mut(|region| {
            let (region_encoded, rest) = encoded.split_at(region_encoded_len);
            encoded = rest;
            decode_fn(region_encoded, region, offset, scale);
            Ok::<_, CodecError>(())
        })
    }
}

fn encode_fill_value(
    fill_value: &FillValue,
    data_type: &DataType,
    element_type: FixedScaleOffsetElementType,
    offset: f64,
    scale: f64,
    encoded_element_type: FixedScaleOffsetElementType,
) -> Result<FillValue, CodecCreateError> {
    let fill_value_bytes = ArrayBytes::new_fill_value(data_type, 1, fill_value)?;
    let encoded_fill_value = do_encode(
        fill_value_bytes,
        element_type,
        offset,
        scale,
        encoded_element_type,
    )
    .map_err(CodecCreateError::other)?;
    Ok(FillValue::new(
        encoded_fill_value
            .into_fixed()
            .map_err(CodecCreateError::other)?
            .into_vec(),
    ))
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl UnboundArrayToArrayCodecTraits for FixedScaleOffsetCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self as Arc<dyn UnboundArrayToArrayCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: Option<FillValue>,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        let element_type = get_element_type(&data_type)?;
        if self.dtype != data_type {
            return Err(CodecCreateError::UnsupportedDataType(
                data_type,
                FixedScaleOffsetCodec::aliases_v3().default_name.to_string(),
            ));
        }
        let encoded_data_type = self.astype.clone().unwrap_or_else(|| self.dtype.clone());
        let encoded_element_type = get_element_type(&encoded_data_type)?;
        let encoded_fill_value = fill_value
            .as_ref()
            .map(|fill_value| {
                encode_fill_value(
                    fill_value,
                    &data_type,
                    element_type,
                    self.offset,
                    self.scale,
                    encoded_element_type,
                )
            })
            .transpose()?;
        Ok(Arc::new(FixedScaleOffsetCodecBound {
            offset: self.offset,
            scale: self.scale,
            element_type,
            encoded_element_type,
            data_type,
            fill_value,
            encoded_data_type,
            encoded_fill_value,
        }))
    }
}

impl ArrayCodecTraits for FixedScaleOffsetCodecBound {
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

impl ArrayToArrayCodecGeometryIdentityTraits for FixedScaleOffsetCodecBound {}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for FixedScaleOffsetCodecBound {
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
        do_encode(
            bytes,
            self.element_type,
            self.offset,
            self.scale,
            self.encoded_element_type,
        )
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        do_decode(
            bytes,
            self.element_type,
            self.offset,
            self.scale,
            self.encoded_element_type,
        )
    }

    fn decode_into(
        &self,
        bytes: ArrayBytes<'_>,
        shape: &[NonZeroU64],
        target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        /// The fewest elements in a region of the target to decode into directly, which is slower per region than decoding all elements and copying them.
        const MIN_REGION_ELEMENTS: usize = 16;
        let ArrayBytesTarget::Fixed(view) = target else {
            return Err(ExpectedOptionalBytesError.into());
        };
        if view.num_contiguous_elements() < MIN_REGION_ELEMENTS
            && view.num_contiguous_elements() as u64 != view.num_elements()
        {
            let decoded = self.decode(bytes, shape, options, resources)?;
            return decode_into_array_bytes_target(&decoded, target);
        }
        decode_into_view(
            &bytes.into_fixed()?,
            view,
            self.element_type,
            self.offset,
            self.scale,
            self.encoded_element_type,
        )
    }
}
