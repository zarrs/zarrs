use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_codec::{
    ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecTraits, CodecError, CodecMetadataOptions,
    CodecOptions, CodecTraits, PartialDecoderCapability, PartialEncoderCapability,
    RecommendedConcurrency,
};
use zarrs_data_type::codec_traits::fixedscaleoffset::{
    FixedScaleOffsetDataTypeExt, FixedScaleOffsetElementType,
};
use zarrs_metadata::Configuration;
use zarrs_metadata::v2::DataTypeMetadataV2;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::{PluginCreateError, ZarrVersion};

use crate::array::{DataType, FillValue};
use crate::convert::data_type_metadata_v2_to_v3;

use super::{DeltaCodecConfiguration, DeltaCodecConfigurationNumcodecs, DeltaCodecConfigurationV1};

/// Convert a `DataTypeMetadataV2` to a `DataType`.
fn v2_dtype_to_data_type(dtype: &DataTypeMetadataV2) -> Result<DataType, PluginCreateError> {
    let err = |_| {
        PluginCreateError::Other(
            "delta codec cannot interpret Zarr V2 data type as V3 equivalent".to_string(),
        )
    };
    DataType::from_metadata(&data_type_metadata_v2_to_v3(dtype).map_err(err)?)
        .map_err(|e| PluginCreateError::Other(e.to_string()))
}

/// Convert a `MetadataV3` to a `DataType`.
fn v3_dtype_to_data_type(dtype: &MetadataV3) -> Result<DataType, PluginCreateError> {
    DataType::from_metadata(dtype).map_err(|e| PluginCreateError::Other(e.to_string()))
}

/// Get the `FixedScaleOffsetElementType` for a data type, or return a codec error.
fn get_element_type(
    data_type: &DataType,
    codec_name: &str,
) -> Result<FixedScaleOffsetElementType, CodecError> {
    data_type
        .codec_fixedscaleoffset()
        .map(|fso| fso.fixedscaleoffset_element_type())
        .map_err(|_| CodecError::UnsupportedDataType(data_type.clone(), codec_name.to_string()))
}

/// Compute delta encoding on a byte slice.
///
/// Encodes as: `out[0] = arr[0]`, `out[i] = arr[i] wrapping_sub arr[i-1]` (integers)
/// or `out[i] = arr[i] - arr[i-1]` (floats).
fn delta_encode_bytes(bytes: &[u8], element_type: FixedScaleOffsetElementType) -> Vec<u8> {
    macro_rules! encode_int {
        ($ty:ty, $size:expr) => {{
            let chunks = bytes.as_chunks::<$size>().0;
            if chunks.is_empty() {
                return vec![];
            }
            let mut out = Vec::with_capacity(bytes.len());
            let mut prev = <$ty>::from_ne_bytes(chunks[0]);
            out.extend_from_slice(&prev.to_ne_bytes());
            for chunk in &chunks[1..] {
                let val = <$ty>::from_ne_bytes(*chunk);
                let delta = val.wrapping_sub(prev);
                out.extend_from_slice(&delta.to_ne_bytes());
                prev = val;
            }
            out
        }};
    }
    macro_rules! encode_float {
        ($ty:ty, $size:expr) => {{
            let chunks = bytes.as_chunks::<$size>().0;
            if chunks.is_empty() {
                return vec![];
            }
            let mut out = Vec::with_capacity(bytes.len());
            let mut prev = <$ty>::from_ne_bytes(chunks[0]);
            out.extend_from_slice(&prev.to_ne_bytes());
            for chunk in &chunks[1..] {
                let val = <$ty>::from_ne_bytes(*chunk);
                let delta = val - prev;
                out.extend_from_slice(&delta.to_ne_bytes());
                prev = val;
            }
            out
        }};
    }
    match element_type {
        FixedScaleOffsetElementType::I8 => encode_int!(i8, 1),
        FixedScaleOffsetElementType::I16 => encode_int!(i16, 2),
        FixedScaleOffsetElementType::I32 => encode_int!(i32, 4),
        FixedScaleOffsetElementType::I64 => encode_int!(i64, 8),
        FixedScaleOffsetElementType::U8 => encode_int!(u8, 1),
        FixedScaleOffsetElementType::U16 => encode_int!(u16, 2),
        FixedScaleOffsetElementType::U32 => encode_int!(u32, 4),
        FixedScaleOffsetElementType::U64 => encode_int!(u64, 8),
        FixedScaleOffsetElementType::F32 => encode_float!(f32, 4),
        FixedScaleOffsetElementType::F64 => encode_float!(f64, 8),
    }
}

/// Compute delta decoding on a byte slice (cumulative sum).
///
/// Decodes as: `dec[0] = enc[0]`, `dec[i] = dec[i-1] wrapping_add enc[i]` (integers)
/// or `dec[i] = dec[i-1] + enc[i]` (floats).
fn delta_decode_bytes(bytes: &[u8], element_type: FixedScaleOffsetElementType) -> Vec<u8> {
    macro_rules! decode_int {
        ($ty:ty, $size:expr) => {{
            let chunks = bytes.as_chunks::<$size>().0;
            if chunks.is_empty() {
                return vec![];
            }
            let mut out = Vec::with_capacity(bytes.len());
            let mut acc = <$ty>::from_ne_bytes(chunks[0]);
            out.extend_from_slice(&acc.to_ne_bytes());
            for chunk in &chunks[1..] {
                let delta = <$ty>::from_ne_bytes(*chunk);
                acc = acc.wrapping_add(delta);
                out.extend_from_slice(&acc.to_ne_bytes());
            }
            out
        }};
    }
    macro_rules! decode_float {
        ($ty:ty, $size:expr) => {{
            let chunks = bytes.as_chunks::<$size>().0;
            if chunks.is_empty() {
                return vec![];
            }
            let mut out = Vec::with_capacity(bytes.len());
            let mut acc = <$ty>::from_ne_bytes(chunks[0]);
            out.extend_from_slice(&acc.to_ne_bytes());
            for chunk in &chunks[1..] {
                let delta = <$ty>::from_ne_bytes(*chunk);
                acc += delta;
                out.extend_from_slice(&acc.to_ne_bytes());
            }
            out
        }};
    }
    match element_type {
        FixedScaleOffsetElementType::I8 => decode_int!(i8, 1),
        FixedScaleOffsetElementType::I16 => decode_int!(i16, 2),
        FixedScaleOffsetElementType::I32 => decode_int!(i32, 4),
        FixedScaleOffsetElementType::I64 => decode_int!(i64, 8),
        FixedScaleOffsetElementType::U8 => decode_int!(u8, 1),
        FixedScaleOffsetElementType::U16 => decode_int!(u16, 2),
        FixedScaleOffsetElementType::U32 => decode_int!(u32, 4),
        FixedScaleOffsetElementType::U64 => decode_int!(u64, 8),
        FixedScaleOffsetElementType::F32 => decode_float!(f32, 4),
        FixedScaleOffsetElementType::F64 => decode_float!(f64, 8),
    }
}

/// Cast bytes from one numeric element type to another via `f64` intermediate.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    clippy::cast_sign_loss
)]
fn cast_bytes(
    bytes: &[u8],
    from: FixedScaleOffsetElementType,
    to: FixedScaleOffsetElementType,
) -> Vec<u8> {
    macro_rules! read_f64 {
        ($bytes:expr, $ty:ty, $size:expr) => {
            $bytes
                .as_chunks::<$size>()
                .0
                .iter()
                .map(|c| <$ty>::from_ne_bytes(*c) as f64)
                .collect::<Vec<f64>>()
        };
    }
    macro_rules! write_from_f64 {
        ($elems:expr, $ty:ty) => {
            $elems
                .into_iter()
                .flat_map(|e| (e as $ty).to_ne_bytes())
                .collect::<Vec<u8>>()
        };
    }

    let as_f64: Vec<f64> = match from {
        FixedScaleOffsetElementType::I8 => read_f64!(bytes, i8, 1),
        FixedScaleOffsetElementType::I16 => read_f64!(bytes, i16, 2),
        FixedScaleOffsetElementType::I32 => read_f64!(bytes, i32, 4),
        FixedScaleOffsetElementType::I64 => read_f64!(bytes, i64, 8),
        FixedScaleOffsetElementType::U8 => read_f64!(bytes, u8, 1),
        FixedScaleOffsetElementType::U16 => read_f64!(bytes, u16, 2),
        FixedScaleOffsetElementType::U32 => read_f64!(bytes, u32, 4),
        FixedScaleOffsetElementType::U64 => read_f64!(bytes, u64, 8),
        FixedScaleOffsetElementType::F32 => read_f64!(bytes, f32, 4),
        FixedScaleOffsetElementType::F64 => read_f64!(bytes, f64, 8),
    };

    match to {
        FixedScaleOffsetElementType::I8 => write_from_f64!(as_f64, i8),
        FixedScaleOffsetElementType::I16 => write_from_f64!(as_f64, i16),
        FixedScaleOffsetElementType::I32 => write_from_f64!(as_f64, i32),
        FixedScaleOffsetElementType::I64 => write_from_f64!(as_f64, i64),
        FixedScaleOffsetElementType::U8 => write_from_f64!(as_f64, u8),
        FixedScaleOffsetElementType::U16 => write_from_f64!(as_f64, u16),
        FixedScaleOffsetElementType::U32 => write_from_f64!(as_f64, u32),
        FixedScaleOffsetElementType::U64 => write_from_f64!(as_f64, u64),
        FixedScaleOffsetElementType::F32 => write_from_f64!(as_f64, f32),
        FixedScaleOffsetElementType::F64 => as_f64.into_iter().flat_map(f64::to_ne_bytes).collect(),
    }
}

/// Core delta codec logic, shared by [`DeltaCodec`] and [`NumcodecsDeltaCodec`].
#[derive(Clone, Debug)]
pub(super) struct DeltaCodecImpl {
    /// Expected decoded data type (validated on encode/decode when `Some`).
    dtype: Option<DataType>,
    /// Encoded (delta) data type. `None` means same as decoded type.
    astype: Option<DataType>,
}

impl DeltaCodecImpl {
    pub(super) fn new(dtype: Option<DataType>, astype: Option<DataType>) -> Self {
        Self { dtype, astype }
    }

    pub(super) fn dtype(&self) -> Option<&DataType> {
        self.dtype.as_ref()
    }

    pub(super) fn encoded_data_type_impl(
        &self,
        decoded_data_type: &DataType,
    ) -> Result<DataType, CodecError> {
        decoded_data_type.codec_fixedscaleoffset().map_err(|_| {
            CodecError::UnsupportedDataType(decoded_data_type.clone(), "delta".to_string())
        })?;
        Ok(self
            .astype
            .clone()
            .unwrap_or_else(|| decoded_data_type.clone()))
    }

    pub(super) fn encode_impl<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        data_type: &DataType,
        codec_name: &str,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let bytes = bytes.into_fixed()?.into_owned();
        let dtype_element_type = get_element_type(data_type, codec_name)?;

        let delta_bytes = delta_encode_bytes(&bytes, dtype_element_type);

        let out = if let Some(astype) = &self.astype {
            let astype_element_type = get_element_type(astype, codec_name)?;
            cast_bytes(&delta_bytes, dtype_element_type, astype_element_type)
        } else {
            delta_bytes
        };

        Ok(out.into())
    }

    pub(super) fn decode_impl<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        data_type: &DataType,
        codec_name: &str,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let bytes = bytes.into_fixed()?.into_owned();
        let dtype_element_type = get_element_type(data_type, codec_name)?;

        let delta_bytes = if let Some(astype) = &self.astype {
            let astype_element_type = get_element_type(astype, codec_name)?;
            cast_bytes(&bytes, astype_element_type, dtype_element_type)
        } else {
            bytes
        };

        let out = delta_decode_bytes(&delta_bytes, dtype_element_type);
        Ok(out.into())
    }
}

// ============================================================================
// DeltaCodec — `zarrs.delta` (no dtype check)
// ============================================================================

/// A `zarrs.delta` codec implementation.
///
/// Encodes numeric array data as differences between adjacent values (delta encoding).
/// Decoding reconstructs the original values via cumulative sum.
///
/// This variant does not require or validate a `dtype` configuration parameter.
#[derive(Clone, Debug)]
pub struct DeltaCodec {
    inner: DeltaCodecImpl,
    /// Original `astype` metadata for configuration round-trip.
    astype_metadata: Option<MetadataV3>,
}

impl DeltaCodec {
    /// Create a new `zarrs.delta` codec with no `astype` conversion.
    pub fn new() -> Self {
        Self {
            inner: DeltaCodecImpl::new(None, None),
            astype_metadata: None,
        }
    }

    /// Create a new `zarrs.delta` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the `astype` metadata cannot be converted to a data type.
    pub fn new_with_configuration(
        configuration: &DeltaCodecConfigurationV1,
    ) -> Result<Self, PluginCreateError> {
        let astype = configuration
            .astype
            .as_ref()
            .map(v3_dtype_to_data_type)
            .transpose()?;
        Ok(Self {
            inner: DeltaCodecImpl::new(None, astype),
            astype_metadata: configuration.astype.clone(),
        })
    }
}

impl Default for DeltaCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl CodecTraits for DeltaCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = DeltaCodecConfiguration::V1(DeltaCodecConfigurationV1 {
            astype: self.astype_metadata.clone(),
        });
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl ArrayCodecTraits for DeltaCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for DeltaCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.inner.encode_impl(bytes, data_type, "zarrs.delta")
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        self.inner.decode_impl(bytes, data_type, "zarrs.delta")
    }

    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        self.inner.encoded_data_type_impl(decoded_data_type)
    }
}

// ============================================================================
// NumcodecsDeltaCodec — `numcodecs.delta` / V2 `delta` (dtype check)
// ============================================================================

/// A `numcodecs.delta` / Zarr V2 `delta` codec implementation.
///
/// Encodes numeric array data as differences between adjacent values (delta encoding).
/// Decoding reconstructs the original values via cumulative sum.
///
/// This variant requires a `dtype` configuration parameter, which is validated against
/// the actual array data type at encode/decode time.
#[derive(Clone, Debug)]
pub struct NumcodecsDeltaCodec {
    inner: DeltaCodecImpl,
    /// Original `dtype` metadata for configuration round-trip.
    dtype_metadata: DataTypeMetadataV2,
    /// Original `astype` metadata for configuration round-trip.
    astype_metadata: Option<DataTypeMetadataV2>,
}

impl NumcodecsDeltaCodec {
    /// Create a new `numcodecs.delta` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the `dtype` or `astype` cannot be converted to data types.
    pub fn new_with_configuration(
        configuration: &DeltaCodecConfigurationNumcodecs,
    ) -> Result<Self, PluginCreateError> {
        let dtype = v2_dtype_to_data_type(&configuration.dtype)?;
        let astype = configuration
            .astype
            .as_ref()
            .map(v2_dtype_to_data_type)
            .transpose()?;
        Ok(Self {
            inner: DeltaCodecImpl::new(Some(dtype), astype),
            dtype_metadata: configuration.dtype.clone(),
            astype_metadata: configuration.astype.clone(),
        })
    }
}

impl CodecTraits for NumcodecsDeltaCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = DeltaCodecConfigurationNumcodecs {
            dtype: self.dtype_metadata.clone(),
            astype: self.astype_metadata.clone(),
        };
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl ArrayCodecTraits for NumcodecsDeltaCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for NumcodecsDeltaCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        if let Some(expected) = self.inner.dtype()
            && data_type != expected
        {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                "numcodecs.delta".to_string(),
            ));
        }
        self.inner.encode_impl(bytes, data_type, "numcodecs.delta")
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        if let Some(expected) = self.inner.dtype()
            && data_type != expected
        {
            return Err(CodecError::UnsupportedDataType(
                data_type.clone(),
                "numcodecs.delta".to_string(),
            ));
        }
        self.inner.decode_impl(bytes, data_type, "numcodecs.delta")
    }

    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        if let Some(expected) = self.inner.dtype()
            && decoded_data_type != expected
        {
            return Err(CodecError::UnsupportedDataType(
                decoded_data_type.clone(),
                "numcodecs.delta".to_string(),
            ));
        }
        self.inner.encoded_data_type_impl(decoded_data_type)
    }
}
