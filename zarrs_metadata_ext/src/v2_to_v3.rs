//! Zarr V2 to V3 conversion.

use std::sync::Arc;

use thiserror::Error;

use crate::{
    chunk_grid::regular::RegularChunkGridConfiguration,
    chunk_key_encoding::v2::V2ChunkKeyEncodingConfiguration,
    codec::{
        blosc::{
            codec_blosc_v2_numcodecs_to_v3, BloscCodecConfigurationNumcodecs,
            BloscShuffleModeNumcodecs,
        },
        bytes::BytesCodecConfigurationV1,
        transpose::{TransposeCodecConfigurationV1, TransposeOrder},
        zstd::{codec_zstd_v2_numcodecs_to_v3, ZstdCodecConfiguration},
    },
};

use zarrs_metadata::{
    v2::{
        data_type_metadata_v2_to_endianness, ArrayMetadataV2, ArrayMetadataV2Order,
        DataTypeMetadataV2, DataTypeMetadataV2EndiannessError, FillValueMetadataV2,
        GroupMetadataV2, MetadataV2,
    },
    v3::{ArrayMetadataV3, FillValueMetadataV3, GroupMetadataV3, MetadataV3},
    DataTypeSize, Endianness,
};
use zarrs_registry::{
    ExtensionAliasesCodecV2, ExtensionAliasesCodecV3, ExtensionAliasesDataTypeV2,
    ExtensionAliasesDataTypeV3,
};

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
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub fn codec_metadata_v2_to_v3(
    order: ArrayMetadataV2Order,
    dimensionality: usize,
    data_type: &MetadataV3,
    endianness: Option<Endianness>,
    filters: &Option<Vec<MetadataV2>>,
    compressor: &Option<MetadataV2>,
    codec_aliases_v2: &ExtensionAliasesCodecV2,
    codec_aliases_v3: &ExtensionAliasesCodecV3,
) -> Result<Vec<MetadataV3>, ArrayMetadataV2ToV3Error> {
    let mut codecs: Vec<MetadataV3> = vec![];

    // Array-to-array codecs
    if order == ArrayMetadataV2Order::F {
        let transpose_metadata = MetadataV3::new_with_serializable_configuration(
            zarrs_registry::codec::TRANSPOSE.to_string(),
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

    // Filters (array to array or array to bytes codecs)
    let mut has_array_to_bytes = false;
    if let Some(filters) = filters {
        for filter in filters {
            let identifier = codec_aliases_v2.identifier(filter.id());
            let name = codec_aliases_v3.default_name(identifier).to_string();
            match identifier {
                zarrs_registry::codec::VLEN_ARRAY
                | zarrs_registry::codec::VLEN_BYTES
                | zarrs_registry::codec::VLEN_UTF8 => {
                    has_array_to_bytes = true;
                    let vlen_v2_metadata =
                        MetadataV3::new_with_configuration(name, serde_json::Map::default());
                    codecs.push(vlen_v2_metadata);
                }
                _ => {
                    codecs.push(MetadataV3::new_with_configuration(
                        name,
                        filter.configuration().clone(),
                    ));
                }
            }
        }
    }

    // Compressor (array to bytes codec)
    if let Some(compressor) = compressor {
        let identifier = codec_aliases_v2.identifier(compressor.id());
        let name = codec_aliases_v3.default_name(identifier).to_string();
        match identifier {
            zarrs_registry::codec::ZFPY | zarrs_registry::codec::PCODEC => {
                // zfpy / pcodec are v2/v3 compatible
                has_array_to_bytes = true;
                codecs.push(MetadataV3::new_with_configuration(
                    name,
                    compressor.configuration().clone(),
                ));
            }
            _ => {}
        }
    }

    if !has_array_to_bytes {
        let bytes_metadata = MetadataV3::new_with_serializable_configuration(
            zarrs_registry::codec::BYTES.to_string(),
            &BytesCodecConfigurationV1 {
                endian: Some(endianness.unwrap_or(Endianness::native())),
            },
        )?;
        codecs.push(bytes_metadata);
    }

    // Compressor (bytes to bytes codec)
    if let Some(compressor) = compressor {
        let identifier = codec_aliases_v2.identifier(compressor.id());
        let name = codec_aliases_v3.default_name(identifier).to_string();
        match identifier {
            zarrs_registry::codec::ZFPY | zarrs_registry::codec::PCODEC => {
                // already handled above
            }
            zarrs_registry::codec::BLOSC => {
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
                    match data_type.name() {
                        zarrs_registry::data_type::BOOL
                        | zarrs_registry::data_type::INT8
                        | zarrs_registry::data_type::UINT8 => Some(DataTypeSize::Fixed(1)),
                        zarrs_registry::data_type::INT16
                        | zarrs_registry::data_type::UINT16
                        | zarrs_registry::data_type::FLOAT16
                        | zarrs_registry::data_type::BFLOAT16 => Some(DataTypeSize::Fixed(2)),
                        zarrs_registry::data_type::INT32
                        | zarrs_registry::data_type::UINT32
                        | zarrs_registry::data_type::FLOAT32 => Some(DataTypeSize::Fixed(4)),
                        zarrs_registry::data_type::INT64
                        | zarrs_registry::data_type::UINT64
                        | zarrs_registry::data_type::FLOAT64
                        | zarrs_registry::data_type::COMPLEX64 => Some(DataTypeSize::Fixed(8)),
                        zarrs_registry::data_type::COMPLEX128 => Some(DataTypeSize::Fixed(16)),
                        zarrs_registry::data_type::STRING | zarrs_registry::data_type::BYTES => {
                            Some(DataTypeSize::Variable)
                        }
                        name => {
                            // Special case for raw bits data types
                            if name.starts_with('r') && name.len() > 1 {
                                if let Ok(size_bits) = name[1..].parse::<usize>() {
                                    if size_bits % 8 == 0 {
                                        let size_bytes = size_bits / 8;
                                        Some(DataTypeSize::Fixed(size_bytes))
                                    } else {
                                        return Err(ArrayMetadataV2ToV3Error::UnsupportedDataType(
                                            DataTypeMetadataV2::Simple(name.to_string()),
                                        ));
                                    }
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        }
                    }
                };

                let configuration = codec_blosc_v2_numcodecs_to_v3(&blosc, data_type_size);
                codecs.push(MetadataV3::new_with_serializable_configuration(
                    name,
                    &configuration,
                )?);
            }
            zarrs_registry::codec::ZSTD => {
                let zstd = serde_json::from_value::<ZstdCodecConfiguration>(serde_json::to_value(
                    compressor.configuration(),
                )?)?;
                let configuration = codec_zstd_v2_numcodecs_to_v3(&zstd);
                codecs.push(MetadataV3::new_with_serializable_configuration(
                    name,
                    &configuration,
                )?);
            }
            _ => codecs.push(MetadataV3::new_with_configuration(
                name,
                compressor.configuration().clone(),
            )),
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
    codec_aliases_v2: &ExtensionAliasesCodecV2,
    codec_aliases_v3: &ExtensionAliasesCodecV3,
    data_type_aliases_v2: &ExtensionAliasesDataTypeV2,
    data_type_aliases_v3: &ExtensionAliasesDataTypeV3,
) -> Result<ArrayMetadataV3, ArrayMetadataV2ToV3Error> {
    let shape = array_metadata_v2.shape.clone();
    let chunk_grid = MetadataV3::new_with_serializable_configuration(
        zarrs_registry::chunk_grid::REGULAR.to_string(),
        &RegularChunkGridConfiguration {
            chunk_shape: array_metadata_v2.chunks.clone(),
        },
    )?;

    let (Ok(data_type), endianness) = (
        data_type_metadata_v2_to_v3(
            &array_metadata_v2.dtype,
            data_type_aliases_v2,
            data_type_aliases_v3,
        ),
        data_type_metadata_v2_to_endianness(&array_metadata_v2.dtype)
            .map_err(ArrayMetadataV2ToV3Error::InvalidEndianness)?,
    ) else {
        return Err(ArrayMetadataV2ToV3Error::UnsupportedDataType(
            array_metadata_v2.dtype.clone(),
        ));
    };

    let fill_value = fill_value_metadata_v2_to_v3(&array_metadata_v2.fill_value, &data_type)?;

    let codecs = codec_metadata_v2_to_v3(
        array_metadata_v2.order,
        array_metadata_v2.shape.len(),
        &data_type,
        endianness,
        &array_metadata_v2.filters,
        &array_metadata_v2.compressor,
        codec_aliases_v2,
        codec_aliases_v3,
    )?;

    let chunk_key_encoding = MetadataV3::new_with_serializable_configuration(
        zarrs_registry::chunk_key_encoding::V2.to_string(),
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
    data_type_aliases_v2: &ExtensionAliasesDataTypeV2,
    data_type_aliases_v3: &ExtensionAliasesDataTypeV3,
) -> Result<MetadataV3, ArrayMetadataV2ToV3Error> {
    match data_type {
        DataTypeMetadataV2::Simple(name) => {
            let identifier = data_type_aliases_v2.identifier(name);
            let name = data_type_aliases_v3.default_name(identifier).to_string();
            Ok(MetadataV3::new(name))
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
        FillValueMetadataV2::NaN => Some(f32::NAN.into()),
        FillValueMetadataV2::Infinity => Some(f32::INFINITY.into()),
        FillValueMetadataV2::NegInfinity => Some(f32::NEG_INFINITY.into()),
        FillValueMetadataV2::Number(number) => Some(number.clone().into()),
        FillValueMetadataV2::String(string) => Some(string.clone().into()),
    };

    // We add some special cases which are supported in v2 but not v3
    let converted_value = match (data_type.name(), converted_value) {
        // A missing fill value is "undefined", so we choose something reasonable
        (name, None) => match name {
            // Support zarr-python encoded string arrays with a `null` fill value
            zarrs_registry::data_type::STRING => FillValueMetadataV3::from(""),
            // Any other null fill value is "undefined"; we pick false for bools
            zarrs_registry::data_type::BOOL => FillValueMetadataV3::from(false),
            // And zero for other data types
            _ => FillValueMetadataV3::from(0),
        },
        // Add a special case for `zarr-python` string data with a 0 fill value -> empty string
        (zarrs_registry::data_type::STRING, Some(FillValueMetadataV3::Number(n)))
            if n.as_u64() == Some(0) =>
        {
            FillValueMetadataV3::from("")
        }
        // Map a 0/1 scalar fill value to a bool
        (zarrs_registry::data_type::BOOL, Some(FillValueMetadataV3::Number(n)))
            if n.as_u64() == Some(0) =>
        {
            FillValueMetadataV3::from(false)
        }
        (zarrs_registry::data_type::BOOL, Some(FillValueMetadataV3::Number(n)))
            if n.as_u64() == Some(1) =>
        {
            FillValueMetadataV3::from(true)
        }
        // NB this passed-through fill value may be incompatible; we will get errors when creating DataType
        (_, Some(value)) => value,
    };

    Ok(converted_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::codec::{
        blosc::BloscCodecConfigurationV1, transpose::TransposeCodecConfigurationV1,
    };
    use zarrs_metadata::{ChunkKeySeparator, ChunkShape, Endianness};
    use zarrs_registry::{
        ExtensionAliasesCodecV2, ExtensionAliasesCodecV3, ExtensionAliasesDataTypeV2,
        ExtensionAliasesDataTypeV3,
    };

    #[test]
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
            ChunkShape::try_from(vec![1000, 1000]).unwrap()
        );
        assert_eq!(array_metadata_v2.shape, vec![10000, 10000]);
        assert_eq!(
            array_metadata_v2.dimension_separator,
            ChunkKeySeparator::Dot
        );
        let codec_aliases_v2 = ExtensionAliasesCodecV2::default();
        let codec_aliases_v3 = ExtensionAliasesCodecV3::default();
        let data_type_aliases_v2 = ExtensionAliasesDataTypeV2::default();
        let data_type_aliases_v3 = ExtensionAliasesDataTypeV3::default();
        assert_eq!(
            data_type_metadata_v2_to_v3(
                &array_metadata_v2.dtype,
                &data_type_aliases_v2,
                &data_type_aliases_v3
            )?
            .name(),
            "float64"
        );
        assert_eq!(
            data_type_metadata_v2_to_endianness(&array_metadata_v2.dtype)?,
            Some(Endianness::Little),
        );
        println!("{array_metadata_v2:?}");

        let array_metadata_v3 = array_metadata_v2_to_v3(
            &array_metadata_v2,
            &codec_aliases_v2,
            &codec_aliases_v3,
            &data_type_aliases_v2,
            &data_type_aliases_v3,
        )?;
        println!("{array_metadata_v3:?}");

        let first_codec = array_metadata_v3.codecs.first().unwrap();
        assert_eq!(first_codec.name(), zarrs_registry::codec::TRANSPOSE);
        let configuration = first_codec
            .to_configuration::<TransposeCodecConfigurationV1>()
            .unwrap();
        assert_eq!(configuration.order.0, vec![1, 0]);

        let last_codec = array_metadata_v3.codecs.last().unwrap();
        assert_eq!(last_codec.name(), zarrs_registry::codec::BLOSC);
        let configuration = last_codec
            .to_configuration::<BloscCodecConfigurationV1>()
            .unwrap();
        assert_eq!(configuration.typesize, Some(8));

        Ok(())
    }
}
