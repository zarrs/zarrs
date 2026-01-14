//! Zarr V2 to V3 conversion.

use std::borrow::Cow;
use std::sync::Arc;

use thiserror::Error;
use zarrs_metadata::v2::{
    ArrayMetadataV2, ArrayMetadataV2Order, DataTypeMetadataV2, DataTypeMetadataV2EndiannessError,
    GroupMetadataV2, MetadataV2, data_type_metadata_v2_to_endianness,
};
use zarrs_metadata::v3::{ArrayMetadataV3, GroupMetadataV3, MetadataV3};
use zarrs_metadata::{Endianness, FillValueMetadata};
use zarrs_plugin::{ExtensionAliasesV2, ExtensionAliasesV3, ExtensionName};

use crate::array::chunk_grid::RegularChunkGrid;
use crate::array::chunk_key_encoding::V2ChunkKeyEncoding;
use crate::array::codec::{BytesCodec, VlenArrayCodec, VlenBytesCodec, VlenUtf8Codec};
use crate::array::data_type;
use zarrs_codec::CodecMetadataOptions;
use zarrs_metadata_ext::chunk_grid::regular::RegularChunkGridConfiguration;
use zarrs_metadata_ext::chunk_key_encoding::v2::V2ChunkKeyEncodingConfiguration;
use zarrs_metadata_ext::codec::bytes::BytesCodecConfigurationV1;

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

/// Try to find a V3 default name for a V2 data type name by creating an instance.
///
/// Returns `Some(default_v3_name)` if a match is found, `None` otherwise.
#[must_use]
fn data_type_v2_to_v3_name(v2_name: &str) -> Option<Cow<'static, str>> {
    use zarrs_data_type::DataTypePluginV2;

    // Special handling for RawBits V2 format (|V8 -> r64)
    // Must be checked before plugin iteration since the plugin won't know the size
    if data_type::RawBitsDataType::matches_name_v2(v2_name) {
        if let Some(size_str) = v2_name.strip_prefix("|V")
            && let Ok(size_bytes) = size_str.parse::<usize>()
        {
            return Some(Cow::Owned(format!("r{}", size_bytes * 8)));
        }
        // If it's already in r* format, return as-is
        return Some(Cow::Owned(v2_name.to_string()));
    }

    // Check if any V2 plugin matches the name and create an instance to get the V3 name
    let metadata = DataTypeMetadataV2::Simple(v2_name.to_string());
    for plugin in inventory::iter::<DataTypePluginV2> {
        if plugin.match_name(v2_name)
            && let Ok(data_type) = plugin.create(&metadata)
        {
            return data_type.name_v3();
        }
    }

    None
}

/// Try to convert V2 codec metadata to V3 metadata.
///
/// # Errors
/// Returns [`ArrayMetadataV2ToV3Error::UnsupportedCodec`] if the codec is not supported.
fn codec_v2_to_v3(v2_metadata: &MetadataV2) -> Result<MetadataV3, ArrayMetadataV2ToV3Error> {
    use zarrs_codec::CodecPluginV2;

    let v2_name = v2_metadata.id();

    // Try to instantiate the codec via V2 plugin to get the V3 name and configuration
    for plugin in inventory::iter::<CodecPluginV2> {
        if plugin.match_name(v2_name)
            && let Ok(codec) = plugin.create(v2_metadata)
            && let Some(v3_name) = codec.name_v3()
        {
            let configuration = codec.configuration_v3(&CodecMetadataOptions::default());
            if let Some(configuration) = configuration {
                return Ok(MetadataV3::new_with_configuration(
                    v3_name.into_owned(),
                    configuration,
                ));
            }
            return Ok(MetadataV3::new(v3_name.into_owned()));
        }
    }
    Err(ArrayMetadataV2ToV3Error::UnsupportedCodec(
        v2_name.to_string(),
        v2_metadata.configuration().clone().into(),
    ))
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
    UnsupportedFillValue(String, FillValueMetadata),
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
            TransposeCodec::aliases_v3()
                .default_name
                .clone()
                .to_string(),
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
            if VlenArrayCodec::matches_name_v2(id)
                || VlenBytesCodec::matches_name_v2(id)
                || VlenUtf8Codec::matches_name_v2(id)
            {
                array_to_bytes_count += 1;
                let name = if VlenArrayCodec::matches_name_v2(id) {
                    VlenArrayCodec::aliases_v3().default_name.clone()
                } else if VlenBytesCodec::matches_name_v2(id) {
                    VlenBytesCodec::aliases_v3().default_name.clone()
                } else {
                    VlenUtf8Codec::aliases_v3().default_name.clone()
                };
                let vlen_v2_metadata = MetadataV3::new_with_configuration(
                    name.to_string(),
                    serde_json::Map::default(),
                );
                codecs.push(vlen_v2_metadata);
            } else {
                // Generic filter - convert to V3
                codecs.push(codec_v2_to_v3(filter)?);
            }
        }
    }

    // Compressor (array to bytes codec)
    if let Some(compressor) = compressor {
        #[allow(unused_variables)]
        let id = compressor.id();

        #[cfg(feature = "zfp")]
        if ZfpyCodec::matches_name_v2(id) {
            array_to_bytes_count += 1;
            codecs.push(MetadataV3::new_with_configuration(
                ZfpyCodec::aliases_v3().default_name.clone().to_string(),
                compressor.configuration().clone(),
            ));
        }
        #[cfg(feature = "zfp")]
        if ZfpCodec::matches_name_v2(id) {
            array_to_bytes_count += 1;
            codecs.push(MetadataV3::new_with_configuration(
                ZfpCodec::aliases_v3().default_name.clone().to_string(),
                compressor.configuration().clone(),
            ));
        }
        #[cfg(feature = "pcodec")]
        if PcodecCodec::matches_name_v2(id) {
            // pcodec is v2/v3 compatible
            array_to_bytes_count += 1;
            codecs.push(MetadataV3::new_with_configuration(
                PcodecCodec::aliases_v3().default_name.clone().to_string(),
                compressor.configuration().clone(),
            ));
        }
    }

    if array_to_bytes_count > 1 {
        return Err(ArrayMetadataV2ToV3Error::MultipleArrayToBytesCodecs);
    }

    if array_to_bytes_count == 0 {
        let bytes_metadata = MetadataV3::new_with_serializable_configuration(
            BytesCodec::aliases_v3().default_name.clone().to_string(),
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
        if !handled && BloscCodec::matches_name_v2(id) {
            let blosc = serde_json::from_value::<BloscCodecConfigurationNumcodecs>(
                serde_json::to_value(compressor.configuration())?,
            )?;

            let data_type_size = if blosc.shuffle == BloscShuffleModeNumcodecs::NoShuffle {
                // The data type size does not matter
                None
            } else {
                // If the data type has an unknown size
                //  - the metadata will not match how the data is encoded, but it can still be decoded just fine
                //  - resaving the array metadata as v3 will not have optimal blosc encoding parameters
                zarrs_data_type::DataType::from_metadata(data_type)
                    .ok()
                    .map(|dt| dt.size())
            };

            let configuration = codec_blosc_v2_numcodecs_to_v3(&blosc, data_type_size);
            codecs.push(MetadataV3::new_with_serializable_configuration(
                BloscCodec::aliases_v3().default_name.clone().to_string(),
                &configuration,
            )?);
            handled = true;
        }

        if !handled {
            // Generic compressor - convert to V3
            codecs.push(codec_v2_to_v3(compressor)?);
        }
    }

    Ok(codecs)
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
        RegularChunkGrid::aliases_v3()
            .default_name
            .clone()
            .to_string(),
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
        V2ChunkKeyEncoding::aliases_v3()
            .default_name
            .clone()
            .to_string(),
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
    fill_value: &FillValueMetadata,
    data_type: &MetadataV3,
) -> Result<FillValueMetadata, ArrayMetadataV2ToV3Error> {
    let converted_value = match fill_value {
        FillValueMetadata::Null => None,
        FillValueMetadata::Bool(_)
        | FillValueMetadata::Number(_)
        | FillValueMetadata::String(_)
        | FillValueMetadata::Array(_)
        | FillValueMetadata::Object(_) => Some(fill_value),
    };

    let data_type_name = data_type.name();

    let is_string = data_type::StringDataType::matches_name_v3(data_type_name);
    let is_bool = data_type::BoolDataType::matches_name_v3(data_type_name);

    // Add some special cases which are supported in v2 but not v3
    let converted_value = match converted_value {
        // A missing fill value is "undefined", so we choose something reasonable
        None => {
            log::warn!(
                "Fill value of `null` specified for data type {data_type_name}. This is unsupported in Zarr V3; mapping to a default value."
            );
            if is_string {
                // Support zarr-python encoded string arrays with a `null` fill value
                FillValueMetadata::from("")
            } else if is_bool {
                // Any other null fill value is "undefined"; we pick false for bools
                FillValueMetadata::from(false)
            } else {
                // And zero for other data types
                FillValueMetadata::from(0)
            }
        }
        Some(value) => {
            // Add a special case for `zarr-python` string data with a 0 fill value -> empty string
            if is_string
                && let FillValueMetadata::Number(n) = value
                && n.as_u64() == Some(0)
            {
                log::warn!(
                    "Permitting non-conformant `0` fill value for `string` data type (zarr-python compatibility)."
                );
                return Ok(FillValueMetadata::from(""));
            }

            // Map a 0/1 scalar fill value to a bool
            if is_bool && let FillValueMetadata::Number(n) = value {
                if n.as_u64() == Some(0) {
                    return Ok(FillValueMetadata::from(false));
                }
                if n.as_u64() == Some(1) {
                    return Ok(FillValueMetadata::from(true));
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
    use zarrs_metadata_ext::codec::{
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
                "filters": null,
                "order": "F",
                "shape": [
                    10000,
                    10000
                ],
                "zarr_format": 2
            }"#;
        let array_metadata_v2: zarrs_metadata::v2::ArrayMetadataV2 =
            serde_json::from_str(json).unwrap();
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
            TransposeCodec::aliases_v3().default_name.clone()
        );
        let configuration = first_codec
            .to_typed_configuration::<TransposeCodecConfigurationV1>()
            .unwrap();
        assert_eq!(configuration.order.0, vec![1, 0]);

        let last_codec = array_metadata_v3.codecs.last().unwrap();
        assert_eq!(
            last_codec.name(),
            BloscCodec::aliases_v3().default_name.clone()
        );
        let configuration = last_codec
            .to_typed_configuration::<BloscCodecConfigurationV1>()
            .unwrap();
        assert_eq!(configuration.typesize, Some(8));

        Ok(())
    }
}
