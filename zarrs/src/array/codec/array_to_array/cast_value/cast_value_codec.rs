use std::cmp::Ordering;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_plugin::ZarrVersion;

use super::{
    CastValueCodecConfiguration, CastValueCodecConfigurationV1, CastValueOutOfRangeMode,
    CastValueRoundingMode, CastValueScalarMap,
};
use crate::array::{ArrayBytes, DataType, FillValue};
use zarrs_codec::{
    ArrayCodecTraits, ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
    ArrayToArrayCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits};
use zarrs_metadata::Configuration;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::PluginCreateError;

/// Registered `cast_value` array-to-array codec.
#[derive(Clone, Debug)]
pub struct CastValueCodec {
    target_metadata: MetadataV3,
    target_data_type: DataType,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
    scalar_map: Option<CastValueScalarMap>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DataKind {
    Int { bits: u8, signed: bool },
    Float16,
    BFloat16,
    Float32,
    Float64,
    Float8E3M4,
    Float8E4M3,
    Float8E5M2,
    UnsupportedReal,
}

#[derive(Clone, Copy, Debug)]
enum Scalar {
    Signed(i64),
    Unsigned(u64),
    Finite { value: f64, negative_zero: bool },
    PosInf,
    NegInf,
    NaN,
}

#[derive(Clone, Debug)]
struct CastValueCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    shape: Vec<NonZeroU64>,
    data_type: DataType,
    fill_value: FillValue,
    codec: Arc<CastValueCodec>,
}

impl<T: ?Sized> CastValueCodecPartial<T> {
    fn new(
        input_output_handle: Arc<T>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        codec: Arc<CastValueCodec>,
    ) -> Self {
        Self {
            input_output_handle,
            shape: shape.to_vec(),
            data_type: data_type.clone(),
            fill_value: fill_value.clone(),
            codec,
        }
    }
}

impl CastValueCodec {
    /// Create a `cast_value` codec from its metadata configuration.
    pub fn new_with_configuration(
        configuration: &CastValueCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            CastValueCodecConfiguration::V1(configuration) => {
                let target_data_type = DataType::from_metadata(&configuration.data_type)?;
                match data_kind(&target_data_type) {
                    Some(DataKind::UnsupportedReal) | None => {
                        return Err(PluginCreateError::Other(
                            "cast_value target data_type is not a supported real-number type"
                                .to_string(),
                        ));
                    }
                    _ => {}
                }
                if matches!(configuration.out_of_range, Some(CastValueOutOfRangeMode::Wrap))
                    && !matches!(
                        data_kind(&target_data_type),
                        Some(DataKind::Int { .. })
                    )
                {
                    return Err(PluginCreateError::Other(
                        "cast_value out_of_range=\"wrap\" requires an integral target data_type"
                            .to_string(),
                    ));
                }
                Ok(Self {
                    target_metadata: configuration.data_type.clone(),
                    target_data_type,
                    rounding: configuration.rounding,
                    out_of_range: configuration.out_of_range,
                    scalar_map: configuration.scalar_map.clone(),
                })
            }
            _ => Err(PluginCreateError::Other(
                "this cast_value codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    fn element_size(data_type: &DataType) -> Result<usize, CodecError> {
        data_type
            .fixed_size()
            .ok_or_else(|| CodecError::UnsupportedDataType(data_type.clone(), "cast_value".to_string()))
    }

    fn parsed_scalar_map(
        &self,
        scalar_map: &[[zarrs_metadata::FillValueMetadata; 2]],
        source_data_type: &DataType,
        target_data_type: &DataType,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, CodecError> {
        scalar_map
            .iter()
            .map(|entry| {
                let input = source_data_type
                    .fill_value_v3(&entry[0])
                    .map_err(|err| CodecError::Other(err.to_string()))?;
                let output = target_data_type
                    .fill_value_v3(&entry[1])
                    .map_err(|err| CodecError::Other(err.to_string()))?;
                Ok((input.as_ne_bytes().to_vec(), output.as_ne_bytes().to_vec()))
            })
            .collect()
    }

    fn scalar_map_encode(
        &self,
        source_data_type: &DataType,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, CodecError> {
        self.scalar_map.as_ref().map_or(Ok(Vec::new()), |scalar_map| {
            self.parsed_scalar_map(&scalar_map.encode, source_data_type, &self.target_data_type)
        })
    }

    fn scalar_map_decode(
        &self,
        decoded_data_type: &DataType,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>, CodecError> {
        self.scalar_map.as_ref().map_or(Ok(Vec::new()), |scalar_map| {
            self.parsed_scalar_map(&scalar_map.decode, &self.target_data_type, decoded_data_type)
        })
    }
}

impl CodecTraits for CastValueCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = CastValueCodecConfiguration::V1(CastValueCodecConfigurationV1 {
            data_type: self.target_metadata.clone(),
            rounding: self.rounding,
            out_of_range: self.out_of_range,
            scalar_map: self.scalar_map.clone(),
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
impl ArrayToArrayCodecTraits for CastValueCodec {
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
        if data_type == &self.target_data_type && self.scalar_map.is_none() {
            return Ok(bytes);
        }
        convert_array(
            bytes,
            data_type,
            &self.target_data_type,
            self.rounding,
            self.out_of_range,
            &self.scalar_map_encode(data_type)?,
        )
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        if data_type == &self.target_data_type && self.scalar_map.is_none() {
            return Ok(bytes);
        }
        convert_array(
            bytes,
            &self.target_data_type,
            data_type,
            self.rounding,
            self.out_of_range,
            &self.scalar_map_decode(data_type)?,
        )
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(CastValueCodecPartial::new(
            input_handle, shape, data_type, fill_value, self,
        )))
    }

    fn partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(CastValueCodecPartial::new(
            input_output_handle,
            shape,
            data_type,
            fill_value,
            self,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(CastValueCodecPartial::new(
            input_handle, shape, data_type, fill_value, self,
        )))
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(CastValueCodecPartial::new(
            input_output_handle,
            shape,
            data_type,
            fill_value,
            self,
        )))
    }

    fn encoded_data_type(&self, _decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        Ok(self.target_data_type.clone())
    }

    fn encoded_fill_value(
        &self,
        decoded_data_type: &DataType,
        decoded_fill_value: &FillValue,
    ) -> Result<FillValue, CodecError> {
        let bytes: ArrayBytes<'_> = decoded_fill_value.as_ne_bytes().to_vec().into();
        let encoded = self.encode(
            bytes,
            &[NonZeroU64::new(1).unwrap()],
            decoded_data_type,
            decoded_fill_value,
            &CodecOptions::default(),
        )?;
        let encoded = FillValue::new(encoded.into_fixed()?.into_owned());
        let decoded_back = self.decode(
            encoded.as_ne_bytes().to_vec().into(),
            &[NonZeroU64::new(1).unwrap()],
            decoded_data_type,
            decoded_fill_value,
            &CodecOptions::default(),
        )?;
        let decoded_back = FillValue::new(decoded_back.into_fixed()?.into_owned());
        if decoded_back != *decoded_fill_value {
            return Err(CodecError::Other(
                "cast_value fill value does not survive a round-trip cast".to_string(),
            ));
        }
        Ok(encoded)
    }
}

impl ArrayCodecTraits for CastValueCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits for CastValueCodecPartial<T>
where
    T: ArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, zarrs_storage::StorageError> {
        self.input_output_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let encoded = self.input_output_handle.partial_decode(indexer, options)?;
        self.codec.decode(
            encoded,
            &self.shape,
            &self.data_type,
            &self.fill_value,
            options,
        )
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits for CastValueCodecPartial<T>
where
    T: ArrayPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn ArrayPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &dyn crate::array::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let all = crate::array::ArraySubset::new_with_shape(
            self.shape.iter().map(|d| d.get()).collect(),
        );
        let encoded_value = self.input_output_handle.partial_decode(&all, options)?;
        let mut decoded_value = self.codec.decode(
            encoded_value,
            &self.shape,
            &self.data_type,
            &self.fill_value,
            options,
        )?;
        decoded_value = zarrs_codec::update_array_bytes(
            decoded_value,
            &self.shape.iter().map(|d| d.get()).collect::<Vec<_>>(),
            indexer,
            bytes,
            self.data_type.size(),
        )?;
        self.input_output_handle.erase()?;
        let encoded_value = self.codec.encode(
            decoded_value,
            &self.shape,
            &self.data_type,
            &self.fill_value,
            options,
        )?;
        self.input_output_handle.partial_encode(&all, &encoded_value, options)
    }

    fn supports_partial_encode(&self) -> bool {
        true
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits for CastValueCodecPartial<T>
where
    T: AsyncArrayPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    async fn exists(&self) -> Result<bool, zarrs_storage::StorageError> {
        self.input_output_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let encoded = self.input_output_handle.partial_decode(indexer, options).await?;
        self.codec.decode(
            encoded,
            &self.shape,
            &self.data_type,
            &self.fill_value,
            options,
        )
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits for CastValueCodecPartial<T>
where
    T: AsyncArrayPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncArrayPartialDecoderTraits> {
        self.clone()
    }

    async fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        indexer: &dyn crate::array::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let all = crate::array::ArraySubset::new_with_shape(
            self.shape.iter().map(|d| d.get()).collect(),
        );
        let encoded_value = self.input_output_handle.partial_decode(&all, options).await?;
        let mut decoded_value = self.codec.decode(
            encoded_value,
            &self.shape,
            &self.data_type,
            &self.fill_value,
            options,
        )?;
        decoded_value = zarrs_codec::update_array_bytes(
            decoded_value,
            &self.shape.iter().map(|d| d.get()).collect::<Vec<_>>(),
            indexer,
            bytes,
            self.data_type.size(),
        )?;
        self.input_output_handle.erase().await?;
        let encoded_value = self.codec.encode(
            decoded_value,
            &self.shape,
            &self.data_type,
            &self.fill_value,
            options,
        )?;
        self.input_output_handle
            .partial_encode(&all, &encoded_value, options)
            .await
    }

    fn supports_partial_encode(&self) -> bool {
        true
    }
}

fn data_kind(data_type: &DataType) -> Option<DataKind> {
    use crate::array::data_type;
    let type_id = data_type.as_any().type_id();
    Some(if type_id == std::any::TypeId::of::<data_type::Int2DataType>() {
        DataKind::Int { bits: 2, signed: true }
    } else if type_id == std::any::TypeId::of::<data_type::Int4DataType>() {
        DataKind::Int { bits: 4, signed: true }
    } else if type_id == std::any::TypeId::of::<data_type::Int8DataType>() {
        DataKind::Int { bits: 8, signed: true }
    } else if type_id == std::any::TypeId::of::<data_type::Int16DataType>() {
        DataKind::Int { bits: 16, signed: true }
    } else if type_id == std::any::TypeId::of::<data_type::Int32DataType>() {
        DataKind::Int { bits: 32, signed: true }
    } else if type_id == std::any::TypeId::of::<data_type::Int64DataType>() {
        DataKind::Int { bits: 64, signed: true }
    } else if type_id == std::any::TypeId::of::<data_type::UInt2DataType>() {
        DataKind::Int { bits: 2, signed: false }
    } else if type_id == std::any::TypeId::of::<data_type::UInt4DataType>() {
        DataKind::Int { bits: 4, signed: false }
    } else if type_id == std::any::TypeId::of::<data_type::UInt8DataType>() {
        DataKind::Int { bits: 8, signed: false }
    } else if type_id == std::any::TypeId::of::<data_type::UInt16DataType>() {
        DataKind::Int { bits: 16, signed: false }
    } else if type_id == std::any::TypeId::of::<data_type::UInt32DataType>() {
        DataKind::Int { bits: 32, signed: false }
    } else if type_id == std::any::TypeId::of::<data_type::UInt64DataType>() {
        DataKind::Int { bits: 64, signed: false }
    } else if type_id == std::any::TypeId::of::<data_type::Float16DataType>() {
        DataKind::Float16
    } else if type_id == std::any::TypeId::of::<data_type::BFloat16DataType>() {
        DataKind::BFloat16
    } else if type_id == std::any::TypeId::of::<data_type::Float32DataType>() {
        DataKind::Float32
    } else if type_id == std::any::TypeId::of::<data_type::Float64DataType>() {
        DataKind::Float64
    } else if type_id == std::any::TypeId::of::<data_type::Float8E3M4DataType>() {
        DataKind::Float8E3M4
    } else if type_id == std::any::TypeId::of::<data_type::Float8E4M3DataType>() {
        DataKind::Float8E4M3
    } else if type_id == std::any::TypeId::of::<data_type::Float8E5M2DataType>() {
        DataKind::Float8E5M2
    } else if type_id == std::any::TypeId::of::<data_type::Float4E2M1FNDataType>()
        || type_id == std::any::TypeId::of::<data_type::Float6E2M3FNDataType>()
        || type_id == std::any::TypeId::of::<data_type::Float6E3M2FNDataType>()
        || type_id == std::any::TypeId::of::<data_type::Float8E4M3B11FNUZDataType>()
        || type_id == std::any::TypeId::of::<data_type::Float8E4M3FNUZDataType>()
        || type_id == std::any::TypeId::of::<data_type::Float8E5M2FNUZDataType>()
        || type_id == std::any::TypeId::of::<data_type::Float8E8M0FNUDataType>()
    {
        DataKind::UnsupportedReal
    } else {
        return None;
    })
}

fn convert_array<'a>(
    bytes: ArrayBytes<'a>,
    source_data_type: &DataType,
    target_data_type: &DataType,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
    scalar_map: &[(Vec<u8>, Vec<u8>)],
) -> Result<ArrayBytes<'a>, CodecError> {
    let source_kind = data_kind(source_data_type)
        .ok_or_else(|| CodecError::UnsupportedDataType(source_data_type.clone(), "cast_value".to_string()))?;
    let target_kind = data_kind(target_data_type)
        .ok_or_else(|| CodecError::UnsupportedDataType(target_data_type.clone(), "cast_value".to_string()))?;
    if matches!(source_kind, DataKind::UnsupportedReal) || matches!(target_kind, DataKind::UnsupportedReal) {
        if source_kind == target_kind {
            return Ok(bytes);
        }
        return Err(CodecError::Other(
            "cast_value currently only supports cross-type casting for standard numeric formats".to_string(),
        ));
    }

    let source_size = CastValueCodec::element_size(source_data_type)?;
    let target_size = CastValueCodec::element_size(target_data_type)?;
    let bytes = bytes.into_fixed()?;
    let mut out = Vec::with_capacity(bytes.len() / source_size * target_size);
    for element in bytes.chunks_exact(source_size) {
        if let Some((_, mapped)) = scalar_map.iter().find(|(input, _)| input.as_slice() == element) {
            out.extend_from_slice(mapped);
            continue;
        }
        let scalar = bytes_to_scalar(element, source_kind)?;
        out.extend_from_slice(&scalar_to_bytes(
            scalar,
            target_kind,
            rounding,
            out_of_range,
        )?);
    }
    Ok(out.into())
}

fn bytes_to_scalar(bytes: &[u8], kind: DataKind) -> Result<Scalar, CodecError> {
    Ok(match kind {
        DataKind::Int { bits: 2, signed: true } | DataKind::Int { bits: 4, signed: true } => {
            Scalar::Signed(i8::from_ne_bytes([bytes[0]]) as i64)
        }
        DataKind::Int { signed: true, bits: 8 } => Scalar::Signed(i8::from_ne_bytes([bytes[0]]) as i64),
        DataKind::Int { signed: true, bits: 16 } => Scalar::Signed(i16::from_ne_bytes(bytes.try_into().unwrap()) as i64),
        DataKind::Int { signed: true, bits: 32 } => Scalar::Signed(i32::from_ne_bytes(bytes.try_into().unwrap()) as i64),
        DataKind::Int { signed: true, bits: 64 } => Scalar::Signed(i64::from_ne_bytes(bytes.try_into().unwrap())),
        DataKind::Int { signed: false, bits: 2 } | DataKind::Int { signed: false, bits: 4 } => {
            Scalar::Unsigned(u8::from_ne_bytes([bytes[0]]) as u64)
        }
        DataKind::Int { signed: false, bits: 8 } => Scalar::Unsigned(u8::from_ne_bytes([bytes[0]]) as u64),
        DataKind::Int { signed: false, bits: 16 } => Scalar::Unsigned(u16::from_ne_bytes(bytes.try_into().unwrap()) as u64),
        DataKind::Int { signed: false, bits: 32 } => Scalar::Unsigned(u32::from_ne_bytes(bytes.try_into().unwrap()) as u64),
        DataKind::Int { signed: false, bits: 64 } => Scalar::Unsigned(u64::from_ne_bytes(bytes.try_into().unwrap())),
        DataKind::Float16 => scalar_from_f64(half::f16::from_ne_bytes(bytes.try_into().unwrap()).to_f64()),
        DataKind::BFloat16 => scalar_from_f64(half::bf16::from_ne_bytes(bytes.try_into().unwrap()).to_f64()),
        DataKind::Float32 => {
            let value = f32::from_ne_bytes(bytes.try_into().unwrap());
            Scalar::Finite {
                value: value as f64,
                negative_zero: value == 0.0 && value.is_sign_negative(),
            }
        }
        DataKind::Float64 => {
            let value = f64::from_ne_bytes(bytes.try_into().unwrap());
            scalar_from_f64(value)
        }
        DataKind::Float8E3M4 => scalar_from_f64(float8_e3m4_to_f64(bytes[0])),
        DataKind::Float8E4M3 => scalar_from_f64(float8_e4m3_to_f64(bytes[0])),
        DataKind::Float8E5M2 => scalar_from_f64(float8_e5m2_to_f64(bytes[0])),
        DataKind::UnsupportedReal => unreachable!(),
        DataKind::Int { .. } => unreachable!(),
    })
}

fn scalar_from_f64(value: f64) -> Scalar {
    if value.is_nan() {
        Scalar::NaN
    } else if value == f64::INFINITY {
        Scalar::PosInf
    } else if value == f64::NEG_INFINITY {
        Scalar::NegInf
    } else {
        Scalar::Finite {
            value,
            negative_zero: value == 0.0 && value.is_sign_negative(),
        }
    }
}

fn scalar_to_bytes(
    scalar: Scalar,
    target_kind: DataKind,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<Vec<u8>, CodecError> {
    match target_kind {
        DataKind::Int { bits, signed } => {
            let value = scalar_to_int(scalar, bits, signed, rounding, out_of_range)?;
            Ok(match (bits, signed) {
                (2 | 4 | 8, true) => (value as i8).to_ne_bytes().to_vec(),
                (16, true) => (value as i16).to_ne_bytes().to_vec(),
                (32, true) => (value as i32).to_ne_bytes().to_vec(),
                (64, true) => value.to_ne_bytes().to_vec(),
                (2 | 4 | 8, false) => (value as u8).to_ne_bytes().to_vec(),
                (16, false) => (value as u16).to_ne_bytes().to_vec(),
                (32, false) => (value as u32).to_ne_bytes().to_vec(),
                (64, false) => (value as u64).to_ne_bytes().to_vec(),
                _ => unreachable!(),
            })
        }
        DataKind::Float16 => Ok(encode_small_float(scalar, rounding, out_of_range, encode_f16_candidates)?.to_vec()),
        DataKind::BFloat16 => Ok(encode_small_float(scalar, rounding, out_of_range, encode_bf16_candidates)?.to_vec()),
        DataKind::Float32 => Ok(encode_f32(scalar, rounding, out_of_range)?.to_ne_bytes().to_vec()),
        DataKind::Float64 => Ok(encode_f64(scalar, rounding, out_of_range)?.to_ne_bytes().to_vec()),
        DataKind::Float8E3M4 => Ok(encode_small_float(scalar, rounding, out_of_range, encode_f8e3m4_candidates)?.to_vec()),
        DataKind::Float8E4M3 => Ok(encode_small_float(scalar, rounding, out_of_range, encode_f8e4m3_candidates)?.to_vec()),
        DataKind::Float8E5M2 => Ok(encode_small_float(scalar, rounding, out_of_range, encode_f8e5m2_candidates)?.to_vec()),
        DataKind::UnsupportedReal => Err(CodecError::Other(
            "cast_value currently only supports identity casts for some custom float formats".to_string(),
        )),
    }
}

fn scalar_to_int(
    scalar: Scalar,
    bits: u8,
    signed: bool,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<i64, CodecError> {
    let (min, max) = int_bounds(bits, signed);
    match scalar {
        Scalar::Signed(value) => clamp_or_wrap_int(value as i128, min, max, bits, signed, out_of_range),
        Scalar::Unsigned(value) => clamp_or_wrap_int(value as i128, min, max, bits, signed, out_of_range),
        Scalar::Finite { value, .. } => {
            let rounded = round_float_to_int(value, rounding)?;
            clamp_or_wrap_int(rounded, min, max, bits, signed, out_of_range)
        }
        Scalar::NaN | Scalar::PosInf | Scalar::NegInf => Err(CodecError::Other(
            "cast_value cannot cast NaN or infinite values to an integral type without scalar_map".to_string(),
        )),
    }
}

fn int_bounds(bits: u8, signed: bool) -> (i128, i128) {
    if signed {
        let max = (1_i128 << (bits - 1)) - 1;
        let min = -(1_i128 << (bits - 1));
        (min, max)
    } else {
        (0, (1_i128 << bits) - 1)
    }
}

fn clamp_or_wrap_int(
    value: i128,
    min: i128,
    max: i128,
    bits: u8,
    signed: bool,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<i64, CodecError> {
    if (min..=max).contains(&value) {
        return Ok(value as i64);
    }
    match out_of_range {
        Some(CastValueOutOfRangeMode::Clamp) => Ok(value.clamp(min, max) as i64),
        Some(CastValueOutOfRangeMode::Wrap) => {
            let modulo = 1_i128 << bits;
            let wrapped = value.rem_euclid(modulo);
            if signed {
                let sign_bit = 1_i128 << (bits - 1);
                let signed_value = if wrapped >= sign_bit {
                    wrapped - modulo
                } else {
                    wrapped
                };
                Ok(signed_value as i64)
            } else {
                Ok(wrapped as i64)
            }
        }
        None => Err(CodecError::Other(
            "cast_value encountered an out-of-range scalar".to_string(),
        )),
    }
}

fn round_float_to_int(value: f64, rounding: CastValueRoundingMode) -> Result<i128, CodecError> {
    if !value.is_finite() {
        return Err(CodecError::Other(
            "cast_value cannot round a non-finite scalar to an integer".to_string(),
        ));
    }
    let rounded = match rounding {
        CastValueRoundingMode::NearestEven => value.round_ties_even(),
        CastValueRoundingMode::TowardsZero => value.trunc(),
        CastValueRoundingMode::TowardsPositive => value.ceil(),
        CastValueRoundingMode::TowardsNegative => value.floor(),
        CastValueRoundingMode::NearestAway => {
            let floor = value.floor();
            let ceil = value.ceil();
            let d_floor = (value - floor).abs();
            let d_ceil = (ceil - value).abs();
            match d_floor.partial_cmp(&d_ceil).unwrap_or(Ordering::Equal) {
                Ordering::Less => floor,
                Ordering::Greater => ceil,
                Ordering::Equal => {
                    if value.is_sign_negative() {
                        floor
                    } else {
                        ceil
                    }
                }
            }
        }
    };
    Ok(rounded as i128)
}

fn encode_f64(
    scalar: Scalar,
    _rounding: CastValueRoundingMode,
    _out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<f64, CodecError> {
    Ok(match scalar {
        Scalar::Signed(v) => v as f64,
        Scalar::Unsigned(v) => v as f64,
        Scalar::Finite { value, negative_zero } => if negative_zero { -0.0 } else { value },
        Scalar::PosInf => f64::INFINITY,
        Scalar::NegInf => f64::NEG_INFINITY,
        Scalar::NaN => f64::NAN,
    })
}

fn encode_f32(
    scalar: Scalar,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<f32, CodecError> {
    match scalar {
        Scalar::NaN => Ok(f32::NAN),
        Scalar::PosInf => Ok(f32::INFINITY),
        Scalar::NegInf => Ok(f32::NEG_INFINITY),
        Scalar::Finite { value, negative_zero } => {
            if negative_zero {
                return Ok(-0.0);
            }
            encode_f32_from_f64(value, rounding, out_of_range)
        }
        Scalar::Signed(value) => encode_f32_from_f64(value as f64, rounding, out_of_range),
        Scalar::Unsigned(value) => encode_f32_from_f64(value as f64, rounding, out_of_range),
    }
}

fn encode_f32_from_f64(
    value: f64,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
) -> Result<f32, CodecError> {
    if !value.is_finite() {
        return Err(CodecError::Other("invalid finite float cast".to_string()));
    }
    let rounded_val = if value > f32::MAX as f64 {
        match rounding {
            CastValueRoundingMode::TowardsNegative | CastValueRoundingMode::TowardsZero => f32::MAX as f64,
            _ => f64::INFINITY,
        }
    } else if value < -(f32::MAX as f64) {
        match rounding {
            CastValueRoundingMode::TowardsPositive | CastValueRoundingMode::TowardsZero => -(f32::MAX as f64),
            _ => f64::NEG_INFINITY,
        }
    } else {
        let nearest = value as f32;
        let nearest_value = nearest as f64;
        if nearest_value == value {
            return Ok(nearest);
        }
        let down = if nearest_value > value {
            next_down_f32(nearest)
        } else {
            nearest
        };
        let up = if nearest_value < value {
            next_up_f32(nearest)
        } else {
            nearest
        };
        (match rounding {
            CastValueRoundingMode::NearestEven => nearest,
            CastValueRoundingMode::TowardsZero => {
                if value.is_sign_negative() { up } else { down }
            }
            CastValueRoundingMode::TowardsPositive => up,
            CastValueRoundingMode::TowardsNegative => down,
            CastValueRoundingMode::NearestAway => {
                let dist_down = (value - down as f64).abs();
                let dist_up = (up as f64 - value).abs();
                match dist_down.partial_cmp(&dist_up).unwrap_or(Ordering::Equal) {
                    Ordering::Less => down,
                    Ordering::Greater => up,
                    Ordering::Equal => {
                        if value.is_sign_negative() { down } else { up }
                    }
                }
            }
        }) as f64
    };
    if rounded_val.is_finite() {
        Ok(rounded_val as f32)
    } else {
        match out_of_range {
            Some(CastValueOutOfRangeMode::Clamp) => {
                if rounded_val.is_sign_positive() { Ok(f32::INFINITY) } else { Ok(f32::NEG_INFINITY) }
            }
            _ => Err(CodecError::Other("cast_value float cast overflow".to_string())),
        }
    }
}

fn next_up_f32(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        value
    } else if value == -0.0 {
        f32::from_bits(1)
    } else if value.is_sign_negative() {
        f32::from_bits(value.to_bits() - 1)
    } else {
        f32::from_bits(value.to_bits() + 1)
    }
}

fn next_down_f32(value: f32) -> f32 {
    if value.is_nan() || value == f32::NEG_INFINITY {
        value
    } else if value == 0.0 {
        f32::from_bits(0x8000_0001)
    } else if value.is_sign_negative() {
        f32::from_bits(value.to_bits() + 1)
    } else {
        f32::from_bits(value.to_bits() - 1)
    }
}

fn encode_small_float<const N: usize>(
    scalar: Scalar,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
    candidates_fn: fn() -> Vec<([u8; N], Scalar)>,
) -> Result<[u8; N], CodecError> {
    let candidates = candidates_fn();
    match scalar {
        Scalar::NaN => {
            if let Some((bits, _)) = candidates.iter().find(|(_, scalar)| matches!(scalar, Scalar::NaN)) {
                return Ok(*bits);
            }
            return Err(CodecError::Other(
                "cast_value target format does not support NaN".to_string(),
            ));
        }
        Scalar::PosInf => {
            if let Some((bits, _)) = candidates.iter().find(|(_, scalar)| matches!(scalar, Scalar::PosInf)) {
                return Ok(*bits);
            }
            return Err(CodecError::Other(
                "cast_value target format does not support infinity".to_string(),
            ));
        }
        Scalar::NegInf => {
            if let Some((bits, _)) = candidates.iter().find(|(_, scalar)| matches!(scalar, Scalar::NegInf)) {
                return Ok(*bits);
            }
            return Err(CodecError::Other(
                "cast_value target format does not support infinity".to_string(),
            ));
        }
        Scalar::Finite { value, negative_zero } => {
            if negative_zero {
                if let Some((bits, _)) = candidates.iter().find(|(_, scalar)| {
                    matches!(scalar, Scalar::Finite { value, negative_zero: true } if *value == 0.0)
                }) {
                    return Ok(*bits);
                }
            }
            choose_small_float_candidate(value, rounding, out_of_range, &candidates)
        }
        Scalar::Signed(v) => choose_small_float_candidate(v as f64, rounding, out_of_range, &candidates),
        Scalar::Unsigned(v) => choose_small_float_candidate(v as f64, rounding, out_of_range, &candidates),
    }
}

fn choose_small_float_candidate<const N: usize>(
    value: f64,
    rounding: CastValueRoundingMode,
    out_of_range: Option<CastValueOutOfRangeMode>,
    candidates: &[([u8; N], Scalar)],
) -> Result<[u8; N], CodecError> {
    let finite_candidates: Vec<_> = candidates
        .iter()
        .filter_map(|(bits, scalar)| match scalar {
            Scalar::Finite { value, .. } => Some((*bits, *value)),
            Scalar::PosInf => Some((*bits, f64::INFINITY)),
            Scalar::NegInf => Some((*bits, f64::NEG_INFINITY)),
            Scalar::NaN | Scalar::Signed(_) | Scalar::Unsigned(_) => None,
        })
        .collect();
    let min = finite_candidates
        .iter()
        .filter(|(_, v)| v.is_finite())
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(_, v)| *v)
        .unwrap_or(0.0);
    let max = finite_candidates
        .iter()
        .filter(|(_, v)| v.is_finite())
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .map(|(_, v)| *v)
        .unwrap_or(0.0);
    if value < min || value > max {
        let rounded_val = if value > max {
            match rounding {
                CastValueRoundingMode::TowardsNegative | CastValueRoundingMode::TowardsZero => max,
                _ => f64::INFINITY,
            }
        } else if value < min {
            match rounding {
                CastValueRoundingMode::TowardsPositive | CastValueRoundingMode::TowardsZero => min,
                _ => f64::NEG_INFINITY,
            }
        } else {
            unreachable!();
        };
        if rounded_val.is_finite() {
            return Ok(finite_candidates
                .iter()
                .find(|(_, v)| *v == rounded_val)
                .map(|(bits, _)| *bits)
                .unwrap());
        } else if matches!(out_of_range, Some(CastValueOutOfRangeMode::Clamp)) {
            let target = if rounded_val.is_sign_positive() { max } else { min };
            return Ok(finite_candidates
                .iter()
                .find(|(_, v)| *v == target)
                .map(|(bits, _)| *bits)
                .unwrap());
        } else {
            return Err(CodecError::Other(
                "cast_value float cast overflow".to_string(),
            ));
        }
    }

    let mut lower = None;
    let mut upper = None;
    for candidate in &finite_candidates {
        if candidate.1 <= value {
            lower = Some(*candidate);
        }
        if candidate.1 >= value {
            upper = Some(*candidate);
            break;
        }
    }
    let lower = lower.unwrap_or(finite_candidates[0]);
    let upper = upper.unwrap_or(*finite_candidates.last().unwrap());
    if lower.1 == value {
        return Ok(lower.0);
    }
    if upper.1 == value {
        return Ok(upper.0);
    }
    Ok(match rounding {
        CastValueRoundingMode::TowardsZero => {
            if value.is_sign_negative() { upper.0 } else { lower.0 }
        }
        CastValueRoundingMode::TowardsPositive => upper.0,
        CastValueRoundingMode::TowardsNegative => lower.0,
        CastValueRoundingMode::NearestEven | CastValueRoundingMode::NearestAway => {
            let lower_dist = (value - lower.1).abs();
            let upper_dist = (upper.1 - value).abs();
            match lower_dist.partial_cmp(&upper_dist).unwrap_or(Ordering::Equal) {
                Ordering::Less => lower.0,
                Ordering::Greater => upper.0,
                Ordering::Equal => {
                    if matches!(rounding, CastValueRoundingMode::NearestAway) {
                        if value.is_sign_negative() { lower.0 } else { upper.0 }
                    } else if (lower.0[N - 1] & 1) == 0 {
                        lower.0
                    } else {
                        upper.0
                    }
                }
            }
        }
    })
}

fn encode_f16_candidates() -> Vec<([u8; 2], Scalar)> {
    (0u16..=u16::MAX)
        .map(|bits| {
            let value = half::f16::from_bits(bits).to_f64();
            (bits.to_ne_bytes(), scalar_from_f64(value))
        })
        .collect()
}

fn encode_bf16_candidates() -> Vec<([u8; 2], Scalar)> {
    (0u16..=u16::MAX)
        .map(|bits| {
            let value = half::bf16::from_bits(bits).to_f64();
            (bits.to_ne_bytes(), scalar_from_f64(value))
        })
        .collect()
}

fn encode_f8e3m4_candidates() -> Vec<([u8; 1], Scalar)> {
    (0u8..=u8::MAX)
        .map(|bits| ([bits], scalar_from_f64(float8_e3m4_to_f64(bits))))
        .collect()
}

fn encode_f8e4m3_candidates() -> Vec<([u8; 1], Scalar)> {
    (0u8..=u8::MAX)
        .map(|bits| ([bits], scalar_from_f64(float8_e4m3_to_f64(bits))))
        .collect()
}

fn encode_f8e5m2_candidates() -> Vec<([u8; 1], Scalar)> {
    (0u8..=u8::MAX)
        .map(|bits| ([bits], scalar_from_f64(float8_e5m2_to_f64(bits))))
        .collect()
}

fn float8_e3m4_to_f64(val: u8) -> f64 {
    let sign = (val & 0b1000_0000) as u32;
    let exponent = ((val >> 4) & 0b111) as i16;
    let mantissa = (val & 0b1111) as u32;
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits(sign << 24) as f64;
        }
        let scale = 2f64.powi(-2);
        let value = (mantissa as f64 / 16.0) * scale;
        return if sign != 0 { -value } else { value };
    }
    if exponent == 7 {
        return if mantissa == 0 {
            if sign != 0 { f64::NEG_INFINITY } else { f64::INFINITY }
        } else {
            f64::NAN
        };
    }
    let unbiased = exponent - 3;
    let value = (1.0 + mantissa as f64 / 16.0) * 2f64.powi(unbiased.into());
    if sign != 0 { -value } else { value }
}

fn decode_ieee_like_8bit(bits: u8, exponent_bits: u8, mantissa_bits: u8, bias: i16) -> f64 {
    let sign = (bits & 0x80) != 0;
    let exponent = ((bits >> mantissa_bits) & ((1 << exponent_bits) - 1)) as i16;
    let mantissa = (bits & ((1 << mantissa_bits) - 1)) as u32;
    let max_exponent = (1 << exponent_bits) - 1;
    if exponent == 0 {
        if mantissa == 0 {
            return if sign { -0.0 } else { 0.0 };
        }
        let value =
            (mantissa as f64 / f64::from(1u32 << mantissa_bits)) * 2f64.powi(1 - bias as i32);
        return if sign { -value } else { value };
    }
    if exponent == max_exponent as i16 {
        return if mantissa == 0 {
            if sign { f64::NEG_INFINITY } else { f64::INFINITY }
        } else {
            f64::NAN
        };
    }
    let value = (1.0 + mantissa as f64 / f64::from(1u32 << mantissa_bits))
        * 2f64.powi((exponent - bias) as i32);
    if sign { -value } else { value }
}

fn float8_e4m3_to_f64(bits: u8) -> f64 {
    decode_ieee_like_8bit(bits, 4, 3, 7)
}

fn float8_e5m2_to_f64(bits: u8) -> f64 {
    decode_ieee_like_8bit(bits, 5, 2, 15)
}
