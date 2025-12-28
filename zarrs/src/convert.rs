//! Zarr V2 to V3 conversion.

use std::{borrow::Cow, sync::Arc};

use thiserror::Error;
#[cfg(feature = "blosc")]
use zarrs_metadata::DataTypeSize;
use zarrs_metadata::{
    Endianness,
    v2::{
        ArrayMetadataV2, ArrayMetadataV2Order, DataTypeMetadataV2,
        DataTypeMetadataV2EndiannessError, FillValueMetadataV2, GroupMetadataV2, MetadataV2,
        data_type_metadata_v2_to_endianness,
    },
    v3::{ArrayMetadataV3, FillValueMetadataV3, GroupMetadataV3, MetadataV3},
};
use zarrs_plugin::{ExtensionIdentifier, ZarrVersions};

use crate::{
    array::{
        chunk_grid::RegularChunkGrid,
        chunk_key_encoding::V2ChunkKeyEncoding,
        codec::{BytesCodec, CodecPlugin, VlenArrayCodec, VlenBytesCodec, VlenUtf8Codec},
        data_type::{BoolDataType, DataTypePlugin, RawBitsDataType, StringDataType},
    },
    metadata_ext::{
        chunk_grid::regular::RegularChunkGridConfiguration,
        chunk_key_encoding::v2::V2ChunkKeyEncodingConfiguration,
        codec::bytes::BytesCodecConfigurationV1,
    },
};

#[cfg(feature = "blosc")]
use crate::{
    array::codec::BloscCodec,
    metadata_ext::codec::blosc::{
        BloscCodecConfigurationNumcodecs, BloscShuffleModeNumcodecs, codec_blosc_v2_numcodecs_to_v3,
    },
};

#[cfg(feature = "pcodec")]
use crate::array::codec::PcodecCodec;

#[cfg(feature = "transpose")]
use crate::{
    array::codec::TransposeCodec,
    metadata_ext::codec::transpose::{TransposeCodecConfigurationV1, TransposeOrder},
};

#[cfg(feature = "zfp")]
use crate::array::codec::{ZfpCodec, ZfpyCodec};

#[cfg(feature = "zstd")]
use crate::{
    array::codec::ZstdCodec,
    metadata_ext::codec::zstd::{ZstdCodecConfiguration, codec_zstd_v2_numcodecs_to_v3},
};

/// Try to find a V3 default name for a V2 data type name by iterating over registered data type plugins.
///
/// Returns `Some(default_v3_name)` if a match is found, `None` otherwise.
#[must_use]
fn data_type_v2_to_v3_name(v2_name: &str) -> Option<Cow<'static, str>> {
    // Special handling for RawBits V2 format (|V8 -> r64)
    // Must be checked before plugin iteration since the plugin won't know the size
    if RawBitsDataType::matches_name(v2_name, ZarrVersions::V2) {
        if let Some(size_str) = v2_name.strip_prefix("|V") {
            if let Ok(size_bytes) = size_str.parse::<usize>() {
                return Some(Cow::Owned(format!("r{}", size_bytes * 8)));
            }
        }
        // If it's already in r* format, return as-is
        return Some(Cow::Owned(v2_name.to_string()));
    }

    // Check plugins (all registered data types including bool, string, bytes, numeric, etc.)
    for plugin in inventory::iter::<DataTypePlugin> {
        if plugin.match_name(v2_name, ZarrVersions::V2) {
            return Some(plugin.default_name(ZarrVersions::V3));
        }
    }

    None
}

/// Try to find a V3 default name for a V2 codec name by iterating over registered plugins.
///
/// Returns `Some(default_v3_name)` if a match is found, `None` otherwise.
#[must_use]
fn codec_v2_to_v3_name(v2_name: &str) -> Option<Cow<'static, str>> {
    for plugin in inventory::iter::<CodecPlugin> {
        if plugin.match_name(v2_name, ZarrVersions::V2) {
            return Some(plugin.default_name(ZarrVersions::V3));
        }
    }
    None
}

/// Convert Zarr V2 group metadata to Zarr V3.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn group_metadata_v2_to_v3(group_metadata_v2: &GroupMetadataV2) -> GroupMetadataV3 {
    GroupMetadataV3::new().with_attributes(group_metadata_v2.attributes.clone())
}

/// An error converting Zarr V2 array metadata to Zarr V3.
#[derive(Clone, Debug, Error)]
pub enum ArrayMetadataV2ToV3Error {
    /// Unsupported data type.
    #[error("unsupported data type {_0:?}")]
    UnsupportedDataType(DataTypeMetadataV2),
    /// Invalid data type endianness.
    #[error(transparent)]
    InvalidEndianness(DataTypeMetadataV2EndiannessError),
    /// An unsupported codec.
    #[error("unsupported codec {_0} with configuration {_1:?}")]
    UnsupportedCodec(String, serde_json::Map<String, serde_json::Value>),
    /// An unsupported fill value.
    #[error("unsupported fill value {_1:?} for data type {_0}")]
    UnsupportedFillValue(String, FillValueMetadataV2),
    /// Serialization/deserialization error.
    #[error("JSON serialization or deserialization error: {_0}")]
    SerdeError(#[from] Arc<serde_json::Error>),
    /// Multiple array to bytes codecs.
    #[error("multiple array to bytes codecs")]
    MultipleArrayToBytesCodecs,
    /// Other.
    #[error("{_0}")]
    Other(String),
}

impl From<serde_json::Error> for ArrayMetadataV2ToV3Error {
    fn from(value: serde_json::Error) -> Self {
        Self::SerdeError(Arc::new(value))
    }
}

/// Convert Zarr V2 codec metadata to Zarr V3.
///
/// # Errors
/// Returns a [`ArrayMetadataV2ToV3Error`] if the metadata is invalid or is not compatible with Zarr V3 metadata.
#[allow(clippy::too_many_lines)]
pub fn codec_metadata_v2_to_v3(
    order: ArrayMetadataV2Order,
    #[cfg_attr(not(feature = "transpose"), allow(unused_variables))] dimensionality: usize,
    #[cfg_attr(not(feature = "blosc"), allow(unused_variables))] data_type: &MetadataV3,
    endianness: Option<Endianness>,
    filters: &Option<Vec<MetadataV2>>,
    compressor: &Option<MetadataV2>,
) -> Result<Vec<MetadataV3>, ArrayMetadataV2ToV3Error> {
    let mut codecs: Vec<MetadataV3> = vec![];

    // Array-to-array codecs
    #[cfg(feature = "transpose")]
    if order == ArrayMetadataV2Order::F {
        let transpose_metadata = MetadataV3::new_with_serializable_configuration(
            TransposeCodec::default_name(ZarrVersions::V3).to_string(),
            &TransposeCodecConfigurationV1 {
                order: {
                    let f_order: Vec<usize> = (0..dimensionality).rev().collect();
                    unsafe {
                        // SAFETY: f_order is valid
                        TransposeOrder::new(&f_order).unwrap_unchecked()
                    }
                },
            },
        )?;
        codecs.push(transpose_metadata);
    }
    #[cfg(not(feature = "transpose"))]
    if order == ArrayMetadataV2Order::F {
        return Err(ArrayMetadataV2ToV3Error::Other(
            "transpose feature is required for F-order arrays".to_string(),
        ));
    }

    // Filters (array to array or array to bytes codecs)
    let mut array_to_bytes_count = 0usize;
    if let Some(filters) = filters {
        for filter in filters {
            let id = filter.id();

            // Check for vlen codecs (array to bytes)
            if VlenArrayCodec::matches_name(id, ZarrVersions::V2)
                || VlenBytesCodec::matches_name(id, ZarrVersions::V2)
                || VlenUtf8Codec::matches_name(id, ZarrVersions::V2)
            {
                array_to_bytes_count += 1;
                let name = if VlenArrayCodec::matches_name(id, ZarrVersions::V2) {
                    VlenArrayCodec::default_name(ZarrVersions::V3)
                } else if VlenBytesCodec::matches_name(id, ZarrVersions::V2) {
                    VlenBytesCodec::default_name(ZarrVersions::V3)
                } else {
                    VlenUtf8Codec::default_name(ZarrVersions::V3)
                };
                let vlen_v2_metadata = MetadataV3::new_with_configuration(
                    name.to_string(),
                    serde_json::Map::default(),
                );
                codecs.push(vlen_v2_metadata);
            } else {
                // Generic filter - pass through with V2 id as name
                codecs.push(MetadataV3::new_with_configuration(
                    codec_v2_to_v3_name(id).unwrap_or(id.into()),
                    filter.configuration().clone(),
                ));
            }
        }
    }

    // Compressor (array to bytes codec)
    if let Some(compressor) = compressor {
        let id = compressor.id();

        #[cfg(feature = "zfp")]
        if ZfpyCodec::matches_name(id, ZarrVersions::V2) {
            array_to_bytes_count += 1;
            codecs.push(MetadataV3::new_with_configuration(
                ZfpyCodec::default_name(ZarrVersions::V3).to_string(),
                compressor.configuration().clone(),
            ));
        }
        #[cfg(feature = "zfp")]
        if ZfpCodec::matches_name(id, ZarrVersions::V2) {
            array_to_bytes_count += 1;
            codecs.push(MetadataV3::new_with_configuration(
                ZfpCodec::default_name(ZarrVersions::V3).to_string(),
                compressor.configuration().clone(),
            ));
        }
        #[cfg(feature = "pcodec")]
        if PcodecCodec::matches_name(id, ZarrVersions::V2) {
            // pcodec is v2/v3 compatible
            array_to_bytes_count += 1;
            codecs.push(MetadataV3::new_with_configuration(
                PcodecCodec::default_name(ZarrVersions::V3).to_string(),
                compressor.configuration().clone(),
            ));
        }
    }

    if array_to_bytes_count > 1 {
        return Err(ArrayMetadataV2ToV3Error::MultipleArrayToBytesCodecs);
    }

    if array_to_bytes_count == 0 {
        let bytes_metadata = MetadataV3::new_with_serializable_configuration(
            BytesCodec::default_name(ZarrVersions::V3).to_string(),
            &BytesCodecConfigurationV1 {
                endian: Some(endianness.unwrap_or(Endianness::native())),
            },
        )?;
        codecs.push(bytes_metadata);
    }

    // Compressor (bytes to bytes codec)
    if let Some(compressor) = compressor {
        let id = compressor.id();

        // Check if already handled above as array to bytes
        let is_array_to_bytes_codec = array_to_bytes_count > 0;

        #[allow(unused_mut)]
        let mut handled = is_array_to_bytes_codec;

        #[cfg(feature = "blosc")]
        if !handled && BloscCodec::matches_name(id, ZarrVersions::V2) {
            let blosc = serde_json::from_value::<BloscCodecConfigurationNumcodecs>(
                serde_json::to_value(compressor.configuration())?,
            )?;

            let data_type_size = if blosc.shuffle == BloscShuffleModeNumcodecs::NoShuffle {
                // The data type size does not matter
                None
            } else {
                // Special case for known Zarr V2 / Zarr V3 compatible data types
                // If the data type has an unknown size
                //  - the metadata will not match how the data is encoded, but it can still be decoded just fine
                //  - resaving the array metadata as v3 will not have optimal blosc encoding parameters
                get_data_type_size_for_blosc(data_type.name())?
            };

            let configuration = codec_blosc_v2_numcodecs_to_v3(&blosc, data_type_size);
            codecs.push(MetadataV3::new_with_serializable_configuration(
                BloscCodec::default_name(ZarrVersions::V3).to_string(),
                &configuration,
            )?);
            handled = true;
        }

        #[cfg(feature = "zstd")]
        if !handled && ZstdCodec::matches_name(id, ZarrVersions::V2) {
            let zstd = serde_json::from_value::<ZstdCodecConfiguration>(serde_json::to_value(
                compressor.configuration(),
            )?)?;
            let configuration = codec_zstd_v2_numcodecs_to_v3(&zstd);
            codecs.push(MetadataV3::new_with_serializable_configuration(
                ZstdCodec::default_name(ZarrVersions::V3).to_string(),
                &configuration,
            )?);
            handled = true;
        }

        if !handled {
            // Generic compressor - pass through
            codecs.push(MetadataV3::new_with_configuration(
                codec_v2_to_v3_name(id).unwrap_or(id.into()),
                compressor.configuration().clone(),
            ));
        }
    }

    Ok(codecs)
}

/// Get the data type size for blosc shuffle mode.
#[cfg(feature = "blosc")]
fn get_data_type_size_for_blosc(
    name: &str,
) -> Result<Option<DataTypeSize>, ArrayMetadataV2ToV3Error> {
    use crate::array::data_type::{
        BFloat16DataType, BytesDataType, Complex64DataType, Complex128DataType, Float16DataType,
        Float32DataType, Float64DataType, Int8DataType, Int16DataType, Int32DataType,
        Int64DataType, RawBitsDataType, UInt8DataType, UInt16DataType, UInt32DataType,
        UInt64DataType,
    };
    use zarrs_plugin::ExtensionIdentifier;

    // Check using ExtensionIdentifier matches for known data types
    if BoolDataType::matches_name(name, ZarrVersions::V3)
        || Int8DataType::matches_name(name, ZarrVersions::V3)
        || UInt8DataType::matches_name(name, ZarrVersions::V3)
    {
        return Ok(Some(DataTypeSize::Fixed(1)));
    }

    if Int16DataType::matches_name(name, ZarrVersions::V3)
        || UInt16DataType::matches_name(name, ZarrVersions::V3)
        || Float16DataType::matches_name(name, ZarrVersions::V3)
        || BFloat16DataType::matches_name(name, ZarrVersions::V3)
    {
        return Ok(Some(DataTypeSize::Fixed(2)));
    }

    if Int32DataType::matches_name(name, ZarrVersions::V3)
        || UInt32DataType::matches_name(name, ZarrVersions::V3)
        || Float32DataType::matches_name(name, ZarrVersions::V3)
    {
        return Ok(Some(DataTypeSize::Fixed(4)));
    }

    if Int64DataType::matches_name(name, ZarrVersions::V3)
        || UInt64DataType::matches_name(name, ZarrVersions::V3)
        || Float64DataType::matches_name(name, ZarrVersions::V3)
        || Complex64DataType::matches_name(name, ZarrVersions::V3)
    {
        return Ok(Some(DataTypeSize::Fixed(8)));
    }

    if Complex128DataType::matches_name(name, ZarrVersions::V3) {
        return Ok(Some(DataTypeSize::Fixed(16)));
    }

    if StringDataType::matches_name(name, ZarrVersions::V3)
        || BytesDataType::matches_name(name, ZarrVersions::V3)
    {
        return Ok(Some(DataTypeSize::Variable));
    }

    // Raw bits data types (r8, r16, r24, etc.)
    if RawBitsDataType::matches_name(name, ZarrVersions::V3) {
        if let Ok(size_bits) = name[1..].parse::<usize>() {
            if size_bits % 8 == 0 {
                let size_bytes = size_bits / 8;
                return Ok(Some(DataTypeSize::Fixed(size_bytes)));
            }
            return Err(ArrayMetadataV2ToV3Error::UnsupportedDataType(
                DataTypeMetadataV2::Simple(name.to_string()),
            ));
        }
    }

    // Unknown data type size
    Ok(None)
}

/// Convert Zarr V2 array metadata to Zarr V3.
///
/// # Errors
/// Returns a [`ArrayMetadataV2ToV3Error`] if the metadata is invalid or is not compatible with Zarr V3 metadata.
#[allow(clippy::too_many_lines)]
pub fn array_metadata_v2_to_v3(
    array_metadata_v2: &ArrayMetadataV2,
) -> Result<ArrayMetadataV3, ArrayMetadataV2ToV3Error> {
    let shape = array_metadata_v2.shape.clone();
    let chunk_grid = MetadataV3::new_with_serializable_configuration(
        RegularChunkGrid::default_name(ZarrVersions::V3).to_string(),
        &RegularChunkGridConfiguration {
            chunk_shape: array_metadata_v2.chunks.clone(),
        },
    )?;

    let endianness = data_type_metadata_v2_to_endianness(&array_metadata_v2.dtype)
        .map_err(ArrayMetadataV2ToV3Error::InvalidEndianness)?;
    let data_type = data_type_metadata_v2_to_v3(&array_metadata_v2.dtype)?;
    let fill_value = fill_value_metadata_v2_to_v3(&array_metadata_v2.fill_value, &data_type)?;

    let codecs = codec_metadata_v2_to_v3(
        array_metadata_v2.order,
        array_metadata_v2.shape.len(),
        &data_type,
        endianness,
        &array_metadata_v2.filters,
        &array_metadata_v2.compressor,
    )?;

    let chunk_key_encoding = MetadataV3::new_with_serializable_configuration(
        V2ChunkKeyEncoding::default_name(ZarrVersions::V3).to_string(),
        &V2ChunkKeyEncodingConfiguration {
            separator: array_metadata_v2.dimension_separator,
        },
    )?;

    let attributes = array_metadata_v2.attributes.clone();

    Ok(
        ArrayMetadataV3::new(shape, chunk_grid, data_type, fill_value, codecs)
            .with_attributes(attributes)
            .with_chunk_key_encoding(chunk_key_encoding),
    )
}

/// Convert Zarr V2 data type metadata to Zarr V3.
///
/// # Errors
/// Returns a [`ArrayMetadataV2ToV3Error`] if the data type is not supported.
pub fn data_type_metadata_v2_to_v3(
    data_type: &DataTypeMetadataV2,
) -> Result<MetadataV3, ArrayMetadataV2ToV3Error> {
    match data_type {
        DataTypeMetadataV2::Simple(name) => {
            // Look up the V3 name using the built-in data type registry
            if let Some(v3_name) = data_type_v2_to_v3_name(name) {
                Ok(MetadataV3::new(v3_name.to_string()))
            } else {
                // Unknown data type
                Err(ArrayMetadataV2ToV3Error::UnsupportedDataType(
                    data_type.clone(),
                ))
            }
        }
        DataTypeMetadataV2::Structured(_) => Err(ArrayMetadataV2ToV3Error::UnsupportedDataType(
            data_type.clone(),
        )),
    }
}

/// Convert Zarr V2 fill value metadata to Zarr V3.
///
/// # Errors
/// Returns a [`ArrayMetadataV2ToV3Error`] if the fill value is not supported for the given data type.
pub fn fill_value_metadata_v2_to_v3(
    fill_value: &FillValueMetadataV2,
    data_type: &MetadataV3,
) -> Result<FillValueMetadataV3, ArrayMetadataV2ToV3Error> {
    let converted_value = match fill_value {
        FillValueMetadataV2::Null => None,
        FillValueMetadataV2::Bool(_)
        | FillValueMetadataV2::Number(_)
        | FillValueMetadataV2::String(_)
        | FillValueMetadataV2::Array(_)
        | FillValueMetadataV2::Object(_) => Some(fill_value),
    };

    let data_type_name = data_type.name();

    let is_string = StringDataType::matches_name(data_type_name, ZarrVersions::V3);
    let is_bool = BoolDataType::matches_name(data_type_name, ZarrVersions::V3);

    // We add some special cases which are supported in v2 but not v3
    let converted_value = match converted_value {
        // A missing fill value is "undefined", so we choose something reasonable
        None => {
            log::warn!(
                "Fill value of `null` specified for data type {data_type_name}. This is unsupported in Zarr V3; mapping to a default value."
            );
            if is_string {
                // Support zarr-python encoded string arrays with a `null` fill value
                FillValueMetadataV3::from("")
            } else if is_bool {
                // Any other null fill value is "undefined"; we pick false for bools
                FillValueMetadataV3::from(false)
            } else {
                // And zero for other data types
                FillValueMetadataV3::from(0)
            }
        }
        Some(value) => {
            // Add a special case for `zarr-python` string data with a 0 fill value -> empty string
            if is_string {
                if let FillValueMetadataV3::Number(n) = value {
                    if n.as_u64() == Some(0) {
                        log::warn!(
                            "Permitting non-conformant `0` fill value for `string` data type (zarr-python compatibility)."
                        );
                        return Ok(FillValueMetadataV3::from(""));
                    }
                }
            }

            // Map a 0/1 scalar fill value to a bool
            if is_bool {
                if let FillValueMetadataV3::Number(n) = value {
                    if n.as_u64() == Some(0) {
                        return Ok(FillValueMetadataV3::from(false));
                    }
                    if n.as_u64() == Some(1) {
                        return Ok(FillValueMetadataV3::from(true));
                    }
                }
            }

            // NB this passed-through fill value may be incompatible; we will get errors when creating DataType
            value.clone()
        }
    };

    Ok(converted_value)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use zarrs_metadata::{ChunkKeySeparator, Endianness};

    use super::*;
    use crate::array::ChunkShape;

    #[cfg(all(feature = "blosc", feature = "transpose"))]
    use crate::metadata_ext::codec::{
        blosc::BloscCodecConfigurationV1, transpose::TransposeCodecConfigurationV1,
    };

    #[test]
    #[cfg(all(feature = "blosc", feature = "transpose"))]
    fn array_v2_config() -> Result<(), Box<dyn std::error::Error>> {
        let json = r#"
            {
                "chunks": [
                    1000,
                    1000
                ],
                "compressor": {
                    "id": "blosc",
                    "cname": "lz4",
                    "clevel": 5,
                    "shuffle": 1
                },
                "dtype": "<f8",
                "fill_value": "NaN",
                "filters": [
                    {"id": "delta", "dtype": "<f8", "astype": "<f4"}
                ],
                "order": "F",
                "shape": [
                    10000,
                    10000
                ],
                "zarr_format": 2
            }"#;
        let array_metadata_v2: zarrs_metadata::v2::ArrayMetadataV2 =
            serde_json::from_str(&json).unwrap();
        assert_eq!(
            array_metadata_v2.chunks,
            ChunkShape::try_from(vec![NonZeroU64::new(1000).unwrap(); 2]).unwrap()
        );
        assert_eq!(array_metadata_v2.shape, vec![10000, 10000]);
        assert_eq!(
            array_metadata_v2.dimension_separator,
            ChunkKeySeparator::Dot
        );
        assert_eq!(
            data_type_metadata_v2_to_v3(&array_metadata_v2.dtype)?.name(),
            "float64"
        );
        assert_eq!(
            data_type_metadata_v2_to_endianness(&array_metadata_v2.dtype)?,
            Some(Endianness::Little),
        );
        println!("{array_metadata_v2:?}");

        let array_metadata_v3 = array_metadata_v2_to_v3(&array_metadata_v2)?;
        println!("{array_metadata_v3:?}");

        let first_codec = array_metadata_v3.codecs.first().unwrap();
        assert_eq!(
            first_codec.name(),
            TransposeCodec::default_name(ZarrVersions::V3)
        );
        let configuration = first_codec
            .to_configuration::<TransposeCodecConfigurationV1>()
            .unwrap();
        assert_eq!(configuration.order.0, vec![1, 0]);

        let last_codec = array_metadata_v3.codecs.last().unwrap();
        assert_eq!(
            last_codec.name(),
            BloscCodec::default_name(ZarrVersions::V3)
        );
        let configuration = last_codec
            .to_configuration::<BloscCodecConfigurationV1>()
            .unwrap();
        assert_eq!(configuration.typesize, Some(8));

        Ok(())
    }
}
