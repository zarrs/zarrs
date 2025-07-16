//! The `sharding` array to bytes codec (Core).
//!
//! Sharding logically splits chunks (shards) into sub-chunks (inner chunks) that can be individually compressed and accessed.
//! This allows to colocate multiple chunks within one storage object, bundling them in shards.
//!
//! This codec requires the `sharding` feature, which is enabled by default.
//!
//! The [`ShardingCodecBuilder`] can help with creating a [`ShardingCodec`].
//!
//! ### Compatible Implementations
//! This is a core codec and should be compatible with all Zarr V3 implementations that support it.
//!
//! ### Specification
//!
//! - <https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html>
//! - <https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/sharding_indexed>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `sharding_indexed`
//!
//! ### Codec `configuration` Example - [`ShardingCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "chunk_shape": [32, 32, 32],
//!     "codecs": [
//!         {
//!             "name": "endian",
//!             "configuration": {
//!                 "endian": "little"
//!             }
//!         },
//!         {
//!             "name": "gzip",
//!             "configuration": {
//!                 "level": 1
//!             }
//!         }
//!     ],
//!     "index_codecs": [
//!         {
//!             "name": "endian",
//!             "configuration": {
//!                 "endian": "little"
//!             }
//!         },
//!         { "name": "crc32c" }
//!     ]
//! }
//! # "#;
//! # use zarrs_metadata_ext::codec::sharding::ShardingCodecConfigurationV1;
//! # serde_json::from_str::<ShardingCodecConfigurationV1>(JSON).unwrap();
//!

mod sharding_codec;
mod sharding_codec_builder;
mod sharding_partial_decoder;
mod sharding_partial_encoder;

use std::{borrow::Cow, num::NonZeroU64, sync::Arc};

pub use zarrs_metadata_ext::codec::sharding::{
    ShardingCodecConfiguration, ShardingCodecConfigurationV1, ShardingIndexLocation,
};

pub use sharding_codec::ShardingCodec;
pub use sharding_codec_builder::ShardingCodecBuilder;
pub(crate) use sharding_partial_decoder::ShardingPartialDecoder;

#[cfg(feature = "async")]
pub(crate) use sharding_partial_decoder::AsyncShardingPartialDecoder;

use crate::{
    array::{
        codec::{
            ArrayToBytesCodecTraits, BytesPartialDecoderTraits, Codec, CodecError, CodecOptions,
            CodecPlugin,
        },
        BytesRepresentation, ChunkRepresentation, ChunkShape, CodecChain, DataType,
    },
    byte_range::ByteRange,
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};
use zarrs_registry::codec::SHARDING;

// Register the codec.
inventory::submit! {
    CodecPlugin::new(SHARDING, is_identifier_sharding, create_codec_sharding)
}

fn is_identifier_sharding(identifier: &str) -> bool {
    identifier == SHARDING
}

pub(crate) fn create_codec_sharding(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: ShardingCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(SHARDING, "codec", metadata.to_string()))?;
    let codec = Arc::new(ShardingCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToBytes(codec))
}

fn calculate_chunks_per_shard(
    shard_shape: &[NonZeroU64],
    chunk_shape: &[NonZeroU64],
) -> Result<ChunkShape, CodecError> {
    Ok(std::iter::zip(shard_shape, chunk_shape)
        .map(|(s, c)| {
            let s = s.get();
            let c = c.get();
            unsafe { 
                // SAFETY: s and c are nonzero
                NonZeroU64::new_unchecked(s.div_ceil(c))
            }
            // if num::Integer::is_multiple_of(&s, &c) {
            //     Ok(unsafe { NonZeroU64::new_unchecked(s / c) })
            // } else {
            //     Err(CodecError::Other(
            //         format!("invalid inner chunk shape {chunk_shape:?}, it must evenly divide {shard_shape:?}")
            //     ))
            // }
        })
        .collect::<Vec<_>>()
        // .collect::<Result<Vec<_>, _>>()?
        .into()
    )
}

fn sharding_index_decoded_representation(chunks_per_shard: &[NonZeroU64]) -> ChunkRepresentation {
    let mut index_shape = Vec::with_capacity(chunks_per_shard.len() + 1);
    index_shape.extend(chunks_per_shard);
    index_shape.push(unsafe { NonZeroU64::new_unchecked(2) });
    ChunkRepresentation::new(index_shape, DataType::UInt64, u64::MAX).unwrap()
}

fn compute_index_encoded_size(
    index_codecs: &dyn ArrayToBytesCodecTraits,
    index_array_representation: &ChunkRepresentation,
) -> Result<u64, CodecError> {
    let bytes_representation = index_codecs.encoded_representation(index_array_representation)?;
    match bytes_representation {
        BytesRepresentation::FixedSize(size) => Ok(size),
        BytesRepresentation::BoundedSize(_) | BytesRepresentation::UnboundedSize => {
            Err(CodecError::Other(
                "the array index cannot include a variable size output codec".to_string(),
            ))
        }
    }
}

fn decode_shard_index(
    encoded_shard_index: &[u8],
    index_array_representation: &ChunkRepresentation,
    index_codecs: &dyn ArrayToBytesCodecTraits,
    options: &CodecOptions,
) -> Result<Vec<u64>, CodecError> {
    // Decode the shard index
    let decoded_shard_index = index_codecs.decode(
        Cow::Borrowed(encoded_shard_index),
        index_array_representation,
        options,
    )?;
    let decoded_shard_index = decoded_shard_index.into_fixed()?;
    Ok(decoded_shard_index
        .chunks_exact(size_of::<u64>())
        .map(|v| u64::from_ne_bytes(v.try_into().unwrap() /* safe */))
        .collect())
}

fn get_index_array_representation(
    chunk_shape: &[NonZeroU64],
    decoded_representation: &ChunkRepresentation,
) -> Result<ChunkRepresentation, CodecError> {
    let shard_shape = decoded_representation.shape();
    let chunk_representation = unsafe {
        ChunkRepresentation::new_unchecked(
            chunk_shape.to_vec(),
            decoded_representation.data_type().clone(),
            decoded_representation.fill_value().clone(),
        )
    };

    // Calculate chunks per shard
    let chunks_per_shard = calculate_chunks_per_shard(shard_shape, chunk_representation.shape())?;

    // Get index array representation and encoded size
    Ok(sharding_index_decoded_representation(
        chunks_per_shard.as_slice(),
    ))
}

fn get_index_byte_range(
    index_array_representation: &ChunkRepresentation,
    index_codecs: &CodecChain,
    index_location: ShardingIndexLocation,
) -> Result<ByteRange, CodecError> {
    let index_encoded_size = compute_index_encoded_size(index_codecs, index_array_representation)
        .map_err(|e| CodecError::Other(e.to_string()))?;
    Ok(match index_location {
        ShardingIndexLocation::Start => ByteRange::FromStart(0, Some(index_encoded_size)),
        ShardingIndexLocation::End => ByteRange::Suffix(index_encoded_size),
    })
}

/// Returns `None` if there is no shard.
fn decode_shard_index_partial_decoder(
    input_handle: &dyn BytesPartialDecoderTraits,
    index_codecs: &CodecChain,
    index_location: ShardingIndexLocation,
    chunk_shape: &[NonZeroU64],
    decoded_representation: &ChunkRepresentation,
    options: &CodecOptions,
) -> Result<Option<Vec<u64>>, CodecError> {
    let index_array_representation =
        get_index_array_representation(chunk_shape, decoded_representation)?;
    let index_byte_range =
        get_index_byte_range(&index_array_representation, index_codecs, index_location)?;
    let encoded_shard_index = input_handle
        .partial_decode(&[index_byte_range], options)?
        .map(|mut v| v.remove(0));
    Ok(match encoded_shard_index {
        Some(encoded_shard_index) => Some(decode_shard_index(
            &encoded_shard_index,
            &index_array_representation,
            index_codecs,
            options,
        )?),
        None => None,
    })
}

#[cfg(feature = "async")]
/// Returns `None` if there is no shard.
async fn decode_shard_index_async_partial_decoder(
    input_handle: &dyn crate::array::codec::AsyncBytesPartialDecoderTraits,
    index_codecs: &CodecChain,
    index_location: ShardingIndexLocation,
    chunk_shape: &[NonZeroU64],
    decoded_representation: &ChunkRepresentation,
    options: &CodecOptions,
) -> Result<Option<Vec<u64>>, CodecError> {
    let index_array_representation =
        get_index_array_representation(chunk_shape, decoded_representation)?;
    let index_byte_range =
        get_index_byte_range(&index_array_representation, index_codecs, index_location)?;
    let encoded_shard_index = input_handle
        .partial_decode(&[index_byte_range], options)
        .await?
        .map(|mut v| v.remove(0));
    Ok(match encoded_shard_index {
        Some(encoded_shard_index) => Some(decode_shard_index(
            &encoded_shard_index,
            &index_array_representation,
            index_codecs,
            options,
        )?),
        None => None,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{
        array::{
            codec::{
                bytes_to_bytes::test_unbounded::TestUnboundedCodec, BytesToBytesCodecTraits,
                CodecOptionsBuilder,
            },
            ArrayBytes,
        },
        array_subset::ArraySubset,
        config::global_config,
    };

    use super::*;

    fn get_concurrent_target(parallel: bool) -> usize {
        if parallel {
            global_config().codec_concurrent_target()
        } else {
            1
        }
    }

    const JSON_VALID2: &str = r#"{
    "chunk_shape": [1, 2, 2],
    "codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        },
        {
            "name": "gzip",
            "configuration": {
                "level": 1
            }
        }
    ],
    "index_codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        },
        { "name": "crc32c" }
    ]
}"#;

    const JSON_VALID3: &str = r#"{
    "chunk_shape": [2, 2],
    "codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        }
    ],
    "index_codecs": [
        {
            "name": "bytes",
            "configuration": {
                "endian": "little"
            }
        }
    ],
    "index_location": "start"
}"#;

    fn codec_sharding_round_trip_impl(
        options: &CodecOptions,
        unbounded: bool,
        index_at_end: bool,
        all_fill_value: bool,
        mut bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    ) {
        let chunk_representation = ChunkRepresentation::new(
            ChunkShape::try_from(vec![4, 4]).unwrap().into(),
            DataType::UInt16,
            0u16,
        )
        .unwrap();
        let elements: Vec<u16> = if all_fill_value {
            vec![0; chunk_representation.num_elements() as usize]
        } else {
            (0..chunk_representation.num_elements() as u16).collect()
        };
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        if unbounded {
            bytes_to_bytes_codecs.push(Arc::new(TestUnboundedCodec::new()))
        }
        let codec = ShardingCodecBuilder::new(vec![2, 2].try_into().unwrap())
            .index_location(if index_at_end {
                ShardingIndexLocation::End
            } else {
                ShardingIndexLocation::Start
            })
            .bytes_to_bytes_codecs(bytes_to_bytes_codecs)
            .build();

        let encoded = codec
            .encode(bytes.clone(), &chunk_representation, options)
            .unwrap();
        let decoded = codec
            .decode(encoded.clone(), &chunk_representation, options)
            .unwrap();
        assert_eq!(bytes, decoded);
        assert_ne!(encoded, decoded.into_fixed().unwrap());
    }

    #[test]
    fn codec_sharding_round_trip1() {
        for index_at_end in [true, false] {
            for all_fill_value in [true, false] {
                for unbounded in [true, false] {
                    for parallel in [true, false] {
                        let concurrent_target = get_concurrent_target(parallel);
                        let options =
                            CodecOptionsBuilder::new().concurrent_target(concurrent_target);
                        codec_sharding_round_trip_impl(
                            &options.build(),
                            unbounded,
                            all_fill_value,
                            index_at_end,
                            vec![],
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "gzip")]
    #[cfg(feature = "crc32c")]
    #[test]
    fn codec_sharding_round_trip2() {
        use crate::array::codec::{Crc32cCodec, GzipCodec};

        for index_at_end in [true, false] {
            for all_fill_value in [true, false] {
                for unbounded in [true, false] {
                    for parallel in [true, false] {
                        let concurrent_target = get_concurrent_target(parallel);
                        let options =
                            CodecOptionsBuilder::new().concurrent_target(concurrent_target);
                        codec_sharding_round_trip_impl(
                            &options.build(),
                            unbounded,
                            all_fill_value,
                            index_at_end,
                            vec![
                                Arc::new(GzipCodec::new(5).unwrap()),
                                Arc::new(Crc32cCodec::new()),
                            ],
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "async")]
    async fn codec_sharding_async_round_trip_impl(
        options: &CodecOptions,
        unbounded: bool,
        index_at_end: bool,
        all_fill_value: bool,
        mut bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    ) {
        let chunk_representation = ChunkRepresentation::new(
            ChunkShape::try_from(vec![4, 4]).unwrap().into(),
            DataType::UInt16,
            0u16,
        )
        .unwrap();
        let elements: Vec<u16> = if all_fill_value {
            vec![0; chunk_representation.num_elements() as usize]
        } else {
            (0..chunk_representation.num_elements() as u16).collect()
        };
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        if unbounded {
            bytes_to_bytes_codecs.push(Arc::new(TestUnboundedCodec::new()))
        }
        let codec = ShardingCodecBuilder::new(vec![2, 2].try_into().unwrap())
            .index_location(if index_at_end {
                ShardingIndexLocation::End
            } else {
                ShardingIndexLocation::Start
            })
            .bytes_to_bytes_codecs(bytes_to_bytes_codecs)
            .build();

        let encoded = codec
            .encode(bytes.clone(), &chunk_representation, options)
            .unwrap();
        let decoded = codec
            .decode(encoded.clone(), &chunk_representation, options)
            .unwrap();
        assert_eq!(bytes, decoded);
        assert_ne!(encoded, decoded.into_fixed().unwrap());
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_sharding_async_round_trip() {
        for index_at_end in [true, false] {
            for all_fill_value in [true, false] {
                for unbounded in [true, false] {
                    for parallel in [true, false] {
                        let concurrent_target = get_concurrent_target(parallel);
                        let options =
                            CodecOptionsBuilder::new().concurrent_target(concurrent_target);
                        codec_sharding_async_round_trip_impl(
                            &options.build(),
                            unbounded,
                            all_fill_value,
                            index_at_end,
                            vec![],
                        )
                        .await;
                    }
                }
            }
        }
    }

    fn codec_sharding_partial_decode(
        options: &CodecOptions,
        unbounded: bool,
        index_at_end: bool,
        all_fill_value: bool,
    ) {
        let chunk_shape: ChunkShape = vec![4, 4].try_into().unwrap();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape.to_vec(), DataType::UInt8, 0u8).unwrap();
        let elements: Vec<u8> = if all_fill_value {
            vec![0; chunk_representation.num_elements() as usize]
        } else {
            (0..chunk_representation.num_elements() as u8).collect()
        };
        let answer: Vec<u8> = if all_fill_value {
            vec![0, 0]
        } else {
            vec![4, 8]
        };

        let bytes: ArrayBytes = elements.into();

        let bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>> = if unbounded {
            vec![Arc::new(TestUnboundedCodec::new())]
        } else {
            vec![]
        };
        let codec = Arc::new(
            ShardingCodecBuilder::new(vec![2, 2].try_into().unwrap())
                .index_location(if index_at_end {
                    ShardingIndexLocation::End
                } else {
                    ShardingIndexLocation::Start
                })
                .bytes_to_bytes_codecs(bytes_to_bytes_codecs)
                .build(),
        );

        let encoded = codec
            .encode(bytes.clone(), &chunk_representation, options)
            .unwrap();
        let decoded_regions = [ArraySubset::new_with_ranges(&[1..3, 0..1])];
        let input_handle = Arc::new(std::io::Cursor::new(encoded));
        let partial_decoder = codec
            .partial_decoder(input_handle, &chunk_representation, options)
            .unwrap();
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_regions, options)
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_iter()
            .map(|bytes| bytes.into_fixed().unwrap().to_vec())
            .flatten()
            .collect::<Vec<_>>()
            .chunks(size_of::<u8>())
            .map(|b| u8::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[test]
    fn codec_sharding_partial_decode_all() {
        for index_at_end in [true, false] {
            for all_fill_value in [true, false] {
                for unbounded in [true, false] {
                    for parallel in [true, false] {
                        let concurrent_target = get_concurrent_target(parallel);
                        let options =
                            CodecOptionsBuilder::new().concurrent_target(concurrent_target);
                        codec_sharding_partial_decode(
                            &options.build(),
                            unbounded,
                            all_fill_value,
                            index_at_end,
                        );
                    }
                }
            }
        }
    }

    #[cfg(feature = "async")]
    async fn codec_sharding_async_partial_decode(
        options: &CodecOptions,
        unbounded: bool,
        index_at_end: bool,
        all_fill_value: bool,
    ) {
        let chunk_shape: ChunkShape = vec![4, 4].try_into().unwrap();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape.to_vec(), DataType::UInt8, 0u8).unwrap();
        let elements: Vec<u8> = if all_fill_value {
            vec![0; chunk_representation.num_elements() as usize]
        } else {
            (0..chunk_representation.num_elements() as u8).collect()
        };
        let answer: Vec<u8> = if all_fill_value {
            vec![0, 0]
        } else {
            vec![4, 8]
        };
        let bytes: ArrayBytes = elements.into();

        let bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>> = if unbounded {
            vec![Arc::new(TestUnboundedCodec::new())]
        } else {
            vec![]
        };
        let codec = Arc::new(
            ShardingCodecBuilder::new(vec![2, 2].try_into().unwrap())
                .index_location(if index_at_end {
                    ShardingIndexLocation::End
                } else {
                    ShardingIndexLocation::Start
                })
                .bytes_to_bytes_codecs(bytes_to_bytes_codecs)
                .build(),
        );

        let encoded = codec
            .encode(bytes.clone(), &chunk_representation, options)
            .unwrap();
        let decoded_regions = [ArraySubset::new_with_ranges(&[1..3, 0..1])];
        let input_handle = Arc::new(std::io::Cursor::new(encoded));
        let partial_decoder = codec
            .async_partial_decoder(input_handle, &chunk_representation, options)
            .await
            .unwrap();
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_regions, options)
            .await
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_iter()
            .map(|bytes| bytes.into_fixed().unwrap().to_vec())
            .flatten()
            .collect::<Vec<_>>()
            .chunks(size_of::<u8>())
            .map(|b| u8::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn codec_sharding_async_partial_decode_all() {
        for index_at_end in [true, false] {
            for all_fill_value in [true, false] {
                for unbounded in [true, false] {
                    for parallel in [true, false] {
                        let concurrent_target = get_concurrent_target(parallel);
                        let options =
                            CodecOptionsBuilder::new().concurrent_target(concurrent_target);
                        codec_sharding_async_partial_decode(
                            &options.build(),
                            unbounded,
                            all_fill_value,
                            index_at_end,
                        )
                        .await;
                    }
                }
            }
        }
    }

    #[cfg(feature = "gzip")]
    #[cfg(feature = "crc32c")]
    #[test]
    fn codec_sharding_partial_decode2() {
        let chunk_shape: ChunkShape = vec![2, 4, 4].try_into().unwrap();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape.to_vec(), DataType::UInt16, 0u16).unwrap();
        let elements: Vec<u16> = (0..chunk_representation.num_elements() as u16).collect();
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();

        let codec_configuration: ShardingCodecConfiguration =
            serde_json::from_str(JSON_VALID2).unwrap();
        let codec = Arc::new(ShardingCodec::new_with_configuration(&codec_configuration).unwrap());

        let encoded = codec
            .encode(bytes, &chunk_representation, &CodecOptions::default())
            .unwrap();
        let decoded_regions = [ArraySubset::new_with_ranges(&[1..2, 0..2, 0..3])];
        let input_handle = Arc::new(std::io::Cursor::new(encoded));
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(
            partial_decoder.size(),
            input_handle.size() + size_of::<u64>() * 2 * 2 * 2 * 2
        ); // sharding partial decoder holds the shard index
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_regions, &CodecOptions::default())
            .unwrap();
        println!("decoded_partial_chunk {decoded_partial_chunk:?}");
        let decoded_partial_chunk: Vec<u16> = decoded_partial_chunk
            .into_iter()
            .map(|bytes| bytes.into_fixed().unwrap().to_vec())
            .flatten()
            .collect::<Vec<_>>()
            .chunks(size_of::<u16>())
            .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
            .collect();

        let answer: Vec<u16> = vec![16, 17, 18, 20, 21, 22];
        assert_eq!(answer, decoded_partial_chunk);
    }

    #[test]
    fn codec_sharding_partial_decode3() {
        let chunk_shape: ChunkShape = vec![4, 4].try_into().unwrap();
        let chunk_representation =
            ChunkRepresentation::new(chunk_shape.to_vec(), DataType::UInt8, 0u8).unwrap();
        let elements: Vec<u8> = (0..chunk_representation.num_elements() as u8).collect();
        let bytes: ArrayBytes = elements.into();

        let codec_configuration: ShardingCodecConfiguration =
            serde_json::from_str(JSON_VALID3).unwrap();
        let codec = Arc::new(ShardingCodec::new_with_configuration(&codec_configuration).unwrap());

        let encoded = codec
            .encode(bytes, &chunk_representation, &CodecOptions::default())
            .unwrap();
        let decoded_regions = [ArraySubset::new_with_ranges(&[1..3, 0..1])];
        let input_handle = Arc::new(std::io::Cursor::new(encoded));
        let partial_decoder = codec
            .partial_decoder(
                input_handle.clone(),
                &chunk_representation,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(
            partial_decoder.size(),
            input_handle.size() + size_of::<u64>() * 2 * 2 * 2
        ); // sharding partial decoder holds the shard index
        let decoded_partial_chunk = partial_decoder
            .partial_decode(&decoded_regions, &CodecOptions::default())
            .unwrap();

        let decoded_partial_chunk: Vec<u8> = decoded_partial_chunk
            .into_iter()
            .map(|bytes| bytes.into_fixed().unwrap().to_vec())
            .flatten()
            .collect::<Vec<_>>()
            .chunks(size_of::<u8>())
            .map(|b| u8::from_ne_bytes(b.try_into().unwrap()))
            .collect();
        let answer: Vec<u8> = vec![4, 8];
        assert_eq!(answer, decoded_partial_chunk);
    }
}
