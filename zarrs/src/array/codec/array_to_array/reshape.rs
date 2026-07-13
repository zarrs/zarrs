//! The `reshape` array to array codec (Experimental).
//!
//! Performs a reshaping operation.
//!
//! ### Compatible Implementations
//! None
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/reshape>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `reshape`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! None
//!
//! ### Codec `configuration` Example - [`ReshapeCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "shape": [[0, 1], -1, [3], 10]
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::reshape::ReshapeCodecConfiguration;
//! # let configuration: ReshapeCodecConfiguration = serde_json::from_str(JSON).unwrap();
//! ```
//!
//! ### Subchunking
//!
//! A codec chain maps the decoded chunk grid through `reshape` before asking a
//! downstream codec, such as `sharding`, for its subchunks. It then maps those
//! subchunks back to the decoded representation. For example:
//!
//! ```text
//! decoded grid [2, 8], chunks [1, 4]   encoded grid [16], chunks [4]
//! +----+----+                          +----+----+----+----+
//! | A  | B  |  -- reshape [[0, 1]] --> | A  | B  | C  | D  |
//! +----+----+     (encode)             +----+----+----+----+
//! | C  | D  |                                    | sharding codec
//! +----+----+                                    | chunk_shape: [2]
//!                                                | (encode)
//!                                                v
//! decoded subchunks [1, 2]             downstream encoded subchunks [2]
//! +--+--+--+--+                        +--+--+--+--+--+--+--+--+
//! |A0|A1|B0|B1| <- reshape [[0, 1]] -- |A0|A1|B0|B1|C0|C1|D0|D1|
//! +--+--+--+--+    (decode)            +--+--+--+--+--+--+--+--+
//! |C0|C1|D0|D1|
//! +--+--+--+--+
//! ```
//!
//! The boxes are chunk or subchunk boundaries.
//! A subchunk can be resolved when its linear span is an n-dimensional box in both representations.
//! With some chunk grids, the global subchunk grid may not be fully resolvable.
//! However, the local subchunk grid of an individual chunk can still be recovered via [`ArrayPartialDecoderTraits::local_subchunk_grid`](zarrs_codec::ArrayPartialDecoderTraits::local_subchunk_grid).

mod reshape_codec;
mod reshape_codec_grid_mapping;
mod reshape_codec_partial;

use std::num::NonZeroU64;
use std::sync::Arc;

use num::Integer;
pub use reshape_codec::ReshapeCodec;
use zarrs_metadata::v3::MetadataV3;

// use itertools::Itertools;
use crate::array::{ArrayIndices, ChunkShape, Indexer, IndexerError, unravel_index};
use zarrs_codec::{Codec, CodecError, CodecPluginV3, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::reshape::{
    ReshapeCodecConfiguration, ReshapeCodecConfigurationV1, ReshapeDim, ReshapeShape,
};

fn get_encoded_shape(
    reshape_shape: &ReshapeShape,
    decoded_shape: &[NonZeroU64],
) -> Result<ChunkShape, CodecError> {
    let mut encoded_shape = Vec::with_capacity(reshape_shape.0.len());
    let mut fill_index = None;
    for output_dim in &reshape_shape.0 {
        match output_dim {
            ReshapeDim::Size(size) => encoded_shape.push(*size),
            ReshapeDim::InputDims(input_dims) => {
                let mut product = NonZeroU64::new(1).unwrap();
                for input_dim in input_dims {
                    let input_shape = usize::try_from(*input_dim)
                        .ok()
                        .and_then(|input_dim| decoded_shape.get(input_dim))
                        .copied()
                        .ok_or_else(|| {
                            CodecError::Other(
                                format!("reshape codec shape references a dimension ({input_dim}) larger than the chunk dimensionality ({})", decoded_shape.len()),
                            )
                        })?;
                    product = product.checked_mul(input_shape).ok_or_else(|| {
                        CodecError::Other(format!(
                            "reshape codec encoded dimension overflows u64 for decoded shape {decoded_shape:?}"
                        ))
                    })?;
                }
                encoded_shape.push(product);
            }
            ReshapeDim::Auto(_) => {
                fill_index = Some(encoded_shape.len());
                encoded_shape.push(NonZeroU64::new(1).unwrap());
            }
        }
    }

    let num_elements = |shape: &[NonZeroU64]| {
        shape.iter().try_fold(1u64, |product, dim| {
            product.checked_mul(dim.get()).ok_or_else(|| {
                CodecError::Other(format!(
                    "reshape codec shape element count overflows u64: {shape:?}"
                ))
            })
        })
    };
    let num_elements_input = num_elements(decoded_shape)?;
    let num_elements_output = num_elements(&encoded_shape)?;
    if let Some(fill_index) = fill_index {
        let (quot, rem) = num_elements_input.div_rem(&num_elements_output);
        if rem == 0 {
            encoded_shape[fill_index] = NonZeroU64::new(quot).unwrap();
        } else {
            return Err(CodecError::Other(format!(
                "reshape codec no substitution for dim {fill_index} can satisfy decoded_shape {decoded_shape:?} == encoded_shape {encoded_shape:?}."
            )));
        }
    } else if num_elements_input != num_elements_output {
        return Err(CodecError::Other(format!(
            "reshape codec encoded/decoded number of elements differ: decoded_shape {decoded_shape:?} ({num_elements_input}) encoded_shape {encoded_shape:?} ({num_elements_output})."
        )));
    }

    Ok(encoded_shape)
}

fn get_reshaped_indexer(
    indexer: &dyn Indexer,
    decoded_shape: &[NonZeroU64],
    encoded_shape: &[NonZeroU64],
) -> Result<impl Indexer, CodecError> {
    if indexer.dimensionality() != decoded_shape.len() {
        return Err(IndexerError::new_incompatible_dimensionality(
            indexer.dimensionality(),
            decoded_shape.len(),
        )
        .into());
    }

    let decoded_shape = bytemuck::must_cast_slice(decoded_shape);
    let encoded_shape = bytemuck::must_cast_slice(encoded_shape);
    let indices = indexer
        .iter_linearised_indices(decoded_shape)?
        .map(|linear_index| {
            unravel_index(linear_index, encoded_shape).ok_or_else(|| {
                CodecError::Other(
                    "reshape codec encoded/decoded number of elements differ".to_string(),
                )
            })
        })
        .collect::<Result<Vec<ArrayIndices>, _>>()?;

    Ok(indices)
}

zarrs_plugin::impl_extension_aliases!(ReshapeCodec, v3: "reshape");

// Register the V3 codec.
inventory::submit! {
    CodecPluginV3::new::<ReshapeCodec>()
}

impl CodecTraitsV3 for ReshapeCodec {
    fn create(metadata: &MetadataV3) -> Result<Codec, zarrs_codec::CodecCreateError> {
        let configuration: ReshapeCodecConfiguration = metadata.to_typed_configuration()?;
        let codec = Arc::new(ReshapeCodec::new_with_configuration(&configuration)?);
        Ok(Codec::ArrayToArray(codec))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Mutex;

    use super::*;
    use crate::array::chunk_grid::{ChunkEdgeLengths, RectilinearChunkGrid, RegularChunkGrid};
    use crate::array::codec::BytesCodec;
    use crate::array::{ArrayBytes, ArraySubset, ChunkShapeTraits, DataType, FillValue, data_type};
    use zarrs_chunk_grid::ChunkGrid;
    use zarrs_codec::{
        ArrayPartialDecoderTraits, ChunkGridDecoded, ChunkGridEncoded, CodecOptions,
        UnboundArrayToArrayCodecTraits, UnboundArrayToBytesCodecTraits,
    };

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    fn codec_reshape_round_trip_impl(
        json: &str,
        data_type: DataType,
        fill_value: FillValue,
        output_shape: Vec<NonZeroU64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let shape = vec![
            NonZeroU64::new(5).unwrap(),
            NonZeroU64::new(4).unwrap(),
            NonZeroU64::new(4).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        let size = shape.num_elements_usize() * data_type.fixed_size().unwrap();
        let bytes: Vec<u8> = (0..size).map(|s| s as u8).collect();
        let bytes: ArrayBytes = bytes.into();

        let configuration: ReshapeCodecConfiguration = serde_json::from_str(json)?;
        let codec = Arc::new(ReshapeCodec::new_with_configuration(&configuration)?)
            .with_context(data_type.clone(), fill_value.clone())?;
        assert_eq!(codec.encoded_shape(&shape)?, output_shape);

        let encoded = codec.encode(bytes.clone(), &shape, &CodecOptions::default())?;
        let decoded = codec.decode(encoded, &shape, &CodecOptions::default())?;
        assert_eq!(bytes, decoded);
        Ok(())
    }

    #[test]
    fn codec_reshape_round_trip_array1() {
        const JSON: &str = r#"{
            "shape": [[0, 1], [2], 3]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(20).unwrap(),
            NonZeroU64::new(4).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                data_type::uint32(),
                FillValue::from(0u32),
                output_shape
            )
            .is_ok()
        );
    }

    #[test]
    fn codec_reshape_round_trip_array2() {
        const JSON: &str = r#"{
            "shape": [[0, 1], [2], -1]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(20).unwrap(),
            NonZeroU64::new(4).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                data_type::uint32(),
                FillValue::from(0u32),
                output_shape
            )
            .is_ok()
        );
    }

    #[test]
    fn codec_reshape_round_trip_array3() {
        const JSON: &str = r#"{
            "shape": [[0, 1, 2], 3]
        }"#;
        let output_shape = vec![NonZeroU64::new(80).unwrap(), NonZeroU64::new(3).unwrap()];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                data_type::uint32(),
                FillValue::from(0u32),
                output_shape
            )
            .is_ok()
        );
    }

    #[test]
    fn codec_reshape_round_trip_array4() {
        const JSON: &str = r#"{
            "shape": [[0], -1, [2, 3]]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(5).unwrap(),
            NonZeroU64::new(4).unwrap(),
            NonZeroU64::new(12).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                data_type::uint32(),
                FillValue::from(0u32),
                output_shape
            )
            .is_ok()
        );
    }

    #[test]
    fn codec_reshape_round_trip_array5() {
        const JSON: &str = r#"{
            "shape": [[0], -1, [3]]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(5).unwrap(),
            NonZeroU64::new(16).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                data_type::uint32(),
                FillValue::from(0u32),
                output_shape
            )
            .is_ok()
        );
    }

    #[test]
    fn codec_reshape_round_trip_array6() {
        const JSON: &str = r#"{
            "shape": [-1, 2, 2, [3]]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(20).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                data_type::uint32(),
                FillValue::from(0u32),
                output_shape
            )
            .is_ok()
        );
    }

    #[test]
    fn codec_reshape_invalid1() {
        const JSON: &str = r#"{
            "shape": [-1, 2, 2, [4]]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(20).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                data_type::uint32(),
                FillValue::from(0u32),
                output_shape
            )
            .is_err()
        );
    }

    #[test]
    fn codec_reshape_invalid2() {
        const JSON: &str = r#"{
            "shape": [2, 2, 2]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(20).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                data_type::uint32(),
                FillValue::from(0u32),
                output_shape
            )
            .is_err()
        );
    }

    #[test]
    fn codec_reshape_shape_overflow() {
        let decoded_shape = [nz(u64::MAX), nz(2)];
        let grouped_shape = ReshapeShape(vec![ReshapeDim::InputDims(vec![0, 1])]);
        assert!(get_encoded_shape(&grouped_shape, &decoded_shape).is_err());

        let fixed_shape = ReshapeShape(vec![
            ReshapeDim::Size(nz(u64::MAX)),
            ReshapeDim::Size(nz(2)),
        ]);
        assert!(get_encoded_shape(&fixed_shape, &[nz(1)]).is_err());
    }

    fn test_reshape_partial_decode_granularity(
        reshape_shape: ReshapeShape,
        decoded_shape: Vec<u64>,
        encoded_subchunk_shape: Vec<NonZeroU64>,
        expected_subchunk_grid_edge_lengths: Vec<Vec<NonZeroU64>>,
    ) {
        let decoded_shape_nonzero = decoded_shape
            .iter()
            .copied()
            .map(NonZeroU64::new)
            .collect::<Option<Vec<_>>>()
            .unwrap();
        let codec = Arc::new(ReshapeCodec::new(reshape_shape))
            .with_context(data_type::uint8(), FillValue::from(0u8))
            .unwrap();
        let encoded_shape = codec
            .encoded_shape(&decoded_shape_nonzero)
            .unwrap()
            .into_iter()
            .map(NonZeroU64::get)
            .collect();
        let chunk_grid = ChunkGrid::new(
            RegularChunkGrid::new(decoded_shape.clone(), decoded_shape_nonzero).unwrap(),
        );
        let encoded_subchunk_grid =
            ChunkGrid::new(RegularChunkGrid::new(encoded_shape, encoded_subchunk_shape).unwrap());
        let subchunk_grid = codec
            .decoded_subchunk_grid((&chunk_grid).into(), (&encoded_subchunk_grid).into())
            .unwrap();
        let ChunkGridDecoded::Array(subchunk_grid) = subchunk_grid else {
            panic!("expected array subchunk grid");
        };
        for (axis, expected_subchunk_grid_edge_lengths_axis) in
            expected_subchunk_grid_edge_lengths.into_iter().enumerate()
        {
            assert_eq!(
                subchunk_grid.chunk_edge_lengths(axis).unwrap(),
                expected_subchunk_grid_edge_lengths_axis
            );
        }
    }

    #[test]
    fn codec_reshape_partial_decode_granularity() {
        test_reshape_partial_decode_granularity(
            ReshapeShape(vec![
                ReshapeDim::InputDims(vec![0]),
                ReshapeDim::InputDims(vec![1]),
            ]),
            vec![4, 6],
            vec![nz(2), nz(3)],
            vec![vec![nz(2), nz(2)], vec![nz(3), nz(3)]],
        );

        test_reshape_partial_decode_granularity(
            ReshapeShape(vec![ReshapeDim::InputDims(vec![0, 1])]),
            vec![2, 20],
            vec![nz(5)],
            vec![vec![nz(1), nz(1)], vec![nz(5), nz(5), nz(5), nz(5)]],
        );
        test_reshape_partial_decode_granularity(
            ReshapeShape(vec![ReshapeDim::InputDims(vec![0, 1])]),
            vec![2, 20],
            vec![nz(20)],
            vec![vec![nz(1), nz(1)], vec![nz(20)]],
        );
        test_reshape_partial_decode_granularity(
            ReshapeShape(vec![ReshapeDim::InputDims(vec![0, 1])]),
            vec![2, 20],
            vec![nz(40)],
            vec![vec![nz(2)], vec![nz(20)]],
        );

        test_reshape_partial_decode_granularity(
            ReshapeShape(vec![
                ReshapeDim::Size(nz(2)),
                ReshapeDim::Size(nz(3)),
                ReshapeDim::Size(nz(2)),
            ]),
            vec![12],
            vec![nz(1), nz(1), nz(2)],
            vec![vec![nz(2), nz(2), nz(2), nz(2), nz(2), nz(2)]],
        );
        test_reshape_partial_decode_granularity(
            ReshapeShape(vec![
                ReshapeDim::Size(nz(2)),
                ReshapeDim::Size(nz(3)),
                ReshapeDim::Size(nz(2)),
            ]),
            vec![12],
            vec![nz(1), nz(3), nz(2)],
            vec![vec![nz(6), nz(6)]],
        );

        test_reshape_partial_decode_granularity(
            ReshapeShape(vec![
                ReshapeDim::InputDims(vec![2]),
                ReshapeDim::InputDims(vec![0, 1]),
            ]),
            vec![2, 3, 4],
            vec![nz(1), nz(6)],
            vec![vec![nz(2)], vec![nz(3)], vec![nz(1); 4]],
        );
        test_reshape_partial_decode_granularity(
            ReshapeShape(vec![
                ReshapeDim::InputDims(vec![2]),
                ReshapeDim::InputDims(vec![0, 1]),
            ]),
            vec![2, 3, 4],
            vec![nz(2), nz(6)],
            vec![vec![nz(2)], vec![nz(3)], vec![nz(2); 2]],
        );

        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![
            ReshapeDim::InputDims(vec![2]),
            ReshapeDim::InputDims(vec![0, 1]),
        ])))
        .with_context(data_type::uint8(), FillValue::from(0u8))
        .unwrap();
        let chunk_grid = ChunkGrid::new(
            RegularChunkGrid::new(vec![2, 3, 4], vec![nz(2), nz(3), nz(4)]).unwrap(),
        );
        let encoded_subchunk_grid =
            ChunkGrid::new(RegularChunkGrid::new(vec![4], vec![nz(1)]).unwrap());
        assert!(
            codec
                .decoded_subchunk_grid((&chunk_grid).into(), (&encoded_subchunk_grid).into(),)
                .is_err()
        );
    }

    #[test]
    fn codec_reshape_partial_decode_varying_grouped_edges() {
        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![
            ReshapeDim::InputDims(vec![0]),
            ReshapeDim::InputDims(vec![1, 2]),
        ])))
        .with_context(data_type::uint8(), FillValue::from(0u8))
        .unwrap();
        let decoded_chunk_grid = ChunkGrid::new(
            RectilinearChunkGrid::new(
                vec![4, 7, 2],
                &[
                    ChunkEdgeLengths::Scalar(nz(2)),
                    ChunkEdgeLengths::Varying(vec![nz(3).into(), nz(4).into()]),
                    ChunkEdgeLengths::Scalar(nz(2)),
                ],
            )
            .unwrap(),
        );
        let encoded_subchunk_grid =
            ChunkGrid::new(RegularChunkGrid::new(vec![4, 14], vec![nz(2), nz(2)]).unwrap());

        let ChunkGridDecoded::Array(decoded_subchunk_grid) = codec
            .decoded_subchunk_grid(
                (&decoded_chunk_grid).into(),
                (&encoded_subchunk_grid).into(),
            )
            .unwrap()
        else {
            panic!("expected decoded subchunk grid");
        };
        assert_eq!(
            decoded_subchunk_grid.chunk_edge_lengths(0).unwrap(),
            vec![nz(2); 2]
        );
        assert_eq!(
            decoded_subchunk_grid.chunk_edge_lengths(1).unwrap(),
            vec![nz(1); 7]
        );
        assert_eq!(
            decoded_subchunk_grid.chunk_edge_lengths(2).unwrap(),
            vec![nz(2)]
        );
    }

    #[test]
    fn codec_reshape_partial_decode_declines_non_rectilinear_spans() {
        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![
            ReshapeDim::InputDims(vec![0, 1]),
        ])))
        .with_context(data_type::uint8(), FillValue::from(0u8))
        .unwrap();
        let decoded_chunk_grid =
            ChunkGrid::new(RegularChunkGrid::new(vec![2, 4], vec![nz(2), nz(4)]).unwrap());
        let encoded_subchunk_grid = ChunkGrid::new(
            RectilinearChunkGrid::new(
                vec![8],
                &[ChunkEdgeLengths::Varying(vec![nz(3).into(), nz(5).into()])],
            )
            .unwrap(),
        );

        assert!(matches!(
            codec
                .decoded_subchunk_grid(
                    (&decoded_chunk_grid).into(),
                    (&encoded_subchunk_grid).into(),
                )
                .unwrap(),
            ChunkGridDecoded::None
        ));
    }

    #[test]
    fn codec_reshape_encoded_chunk_grid() {
        struct TestCase {
            name: &'static str,
            reshape_shape: ReshapeShape,
            decoded_chunk_grid: ChunkGrid,
            expected_array_shape: Vec<u64>,
            expected_chunk_edge_lengths: Vec<Vec<NonZeroU64>>,
        }

        let cases = [
            TestCase {
                name: "no-op reshape clones the decoded grid",
                reshape_shape: ReshapeShape(vec![
                    ReshapeDim::InputDims(vec![0]),
                    ReshapeDim::InputDims(vec![1]),
                ]),
                decoded_chunk_grid: ChunkGrid::new(
                    RectilinearChunkGrid::new(
                        vec![7, 6],
                        &[
                            ChunkEdgeLengths::Varying(vec![nz(3).into(), nz(4).into()]),
                            ChunkEdgeLengths::Scalar(nz(2)),
                        ],
                    )
                    .unwrap(),
                ),
                expected_array_shape: vec![7, 6],
                expected_chunk_edge_lengths: vec![vec![nz(3), nz(4)], vec![nz(2); 3]],
            },
            TestCase {
                name: "2D-to-1D flatten with contiguous chunks",
                reshape_shape: ReshapeShape(vec![ReshapeDim::InputDims(vec![0, 1])]),
                decoded_chunk_grid: ChunkGrid::new(
                    RegularChunkGrid::new(vec![2, 20], vec![nz(1), nz(5)]).unwrap(),
                ),
                expected_array_shape: vec![40],
                expected_chunk_edge_lengths: vec![vec![nz(5); 8]],
            },
            TestCase {
                name: "fixed 1D-to-ND reshape evaluated per chunk",
                reshape_shape: ReshapeShape(vec![ReshapeDim::Size(nz(2)), ReshapeDim::Size(nz(3))]),
                decoded_chunk_grid: ChunkGrid::new(
                    RegularChunkGrid::new(vec![12], vec![nz(6)]).unwrap(),
                ),
                expected_array_shape: vec![2, 6],
                expected_chunk_edge_lengths: vec![vec![nz(2)], vec![nz(3); 2]],
            },
            TestCase {
                name: "auto dimension with regular chunks",
                reshape_shape: ReshapeShape(vec![ReshapeDim::Size(nz(2)), ReshapeDim::auto()]),
                decoded_chunk_grid: ChunkGrid::new(
                    RegularChunkGrid::new(vec![12], vec![nz(4)]).unwrap(),
                ),
                expected_array_shape: vec![2, 6],
                expected_chunk_edge_lengths: vec![vec![nz(2)], vec![nz(2); 3]],
            },
            TestCase {
                name: "rectilinear identity prefix with reshaped suffix",
                reshape_shape: ReshapeShape(vec![
                    ReshapeDim::InputDims(vec![0]),
                    ReshapeDim::InputDims(vec![1, 2]),
                ]),
                decoded_chunk_grid: ChunkGrid::new(
                    RectilinearChunkGrid::new(
                        vec![7, 2, 20],
                        &[
                            ChunkEdgeLengths::Varying(vec![nz(3).into(), nz(4).into()]),
                            ChunkEdgeLengths::Scalar(nz(1)),
                            ChunkEdgeLengths::Scalar(nz(5)),
                        ],
                    )
                    .unwrap(),
                ),
                expected_array_shape: vec![7, 40],
                expected_chunk_edge_lengths: vec![vec![nz(3), nz(4)], vec![nz(5); 8]],
            },
            TestCase {
                name: "rectilinear identity suffix with reshaped prefix",
                reshape_shape: ReshapeShape(vec![
                    ReshapeDim::InputDims(vec![0, 1]),
                    ReshapeDim::InputDims(vec![2]),
                ]),
                decoded_chunk_grid: ChunkGrid::new(
                    RectilinearChunkGrid::new(
                        vec![2, 20, 7],
                        &[
                            ChunkEdgeLengths::Scalar(nz(1)),
                            ChunkEdgeLengths::Scalar(nz(5)),
                            ChunkEdgeLengths::Varying(vec![nz(3).into(), nz(4).into()]),
                        ],
                    )
                    .unwrap(),
                ),
                expected_array_shape: vec![40, 7],
                expected_chunk_edge_lengths: vec![vec![nz(5); 8], vec![nz(3), nz(4)]],
            },
            TestCase {
                name: "multiple independent grouped partitions",
                reshape_shape: ReshapeShape(vec![
                    ReshapeDim::InputDims(vec![0, 1]),
                    ReshapeDim::InputDims(vec![2, 3]),
                ]),
                decoded_chunk_grid: ChunkGrid::new(
                    RegularChunkGrid::new(vec![2, 3, 4, 5], vec![nz(1), nz(3), nz(2), nz(5)])
                        .unwrap(),
                ),
                expected_array_shape: vec![6, 20],
                expected_chunk_edge_lengths: vec![vec![nz(3); 2], vec![nz(10); 2]],
            },
            TestCase {
                name: "reordered input dimension partitions",
                reshape_shape: ReshapeShape(vec![
                    ReshapeDim::InputDims(vec![1]),
                    ReshapeDim::InputDims(vec![0]),
                ]),
                decoded_chunk_grid: ChunkGrid::new(
                    RegularChunkGrid::new(vec![2, 8], vec![nz(1), nz(4)]).unwrap(),
                ),
                expected_array_shape: vec![8, 2],
                expected_chunk_edge_lengths: vec![vec![nz(4); 2], vec![nz(1); 2]],
            },
            TestCase {
                name: "flatten with non-contiguous decoded chunks",
                reshape_shape: ReshapeShape(vec![ReshapeDim::InputDims(vec![0, 1])]),
                decoded_chunk_grid: ChunkGrid::new(
                    RegularChunkGrid::new(vec![2, 20], vec![nz(2), nz(5)]).unwrap(),
                ),
                expected_array_shape: vec![40],
                expected_chunk_edge_lengths: vec![vec![nz(10); 4]],
            },
            TestCase {
                name: "grouped partition with varying chunking",
                reshape_shape: ReshapeShape(vec![
                    ReshapeDim::InputDims(vec![0]),
                    ReshapeDim::InputDims(vec![1, 2]),
                ]),
                decoded_chunk_grid: ChunkGrid::new(
                    RectilinearChunkGrid::new(
                        vec![4, 7, 2],
                        &[
                            ChunkEdgeLengths::Scalar(nz(2)),
                            ChunkEdgeLengths::Varying(vec![nz(3).into(), nz(4).into()]),
                            ChunkEdgeLengths::Scalar(nz(2)),
                        ],
                    )
                    .unwrap(),
                ),
                expected_array_shape: vec![4, 14],
                expected_chunk_edge_lengths: vec![vec![nz(2); 2], vec![nz(6), nz(8)]],
            },
            TestCase {
                name: "grouped partition with non-contiguous chunks",
                reshape_shape: ReshapeShape(vec![
                    ReshapeDim::InputDims(vec![0]),
                    ReshapeDim::InputDims(vec![1, 2]),
                ]),
                decoded_chunk_grid: ChunkGrid::new(
                    RegularChunkGrid::new(vec![4, 2, 20], vec![nz(2), nz(2), nz(5)]).unwrap(),
                ),
                expected_array_shape: vec![4, 40],
                expected_chunk_edge_lengths: vec![vec![nz(2); 2], vec![nz(10); 4]],
            },
        ];

        for case in cases {
            let codec = Arc::new(ReshapeCodec::new(case.reshape_shape))
                .with_context(data_type::uint8(), FillValue::from(0u8))
                .unwrap();
            let ChunkGridEncoded::Array(encoded_chunk_grid) = codec
                .encoded_chunk_grid((&case.decoded_chunk_grid).into())
                .unwrap()
            else {
                panic!("{}: expected encoded chunk grid", case.name);
            };

            assert_eq!(
                encoded_chunk_grid.array_shape(),
                case.expected_array_shape,
                "{}",
                case.name
            );
            for (axis, expected_edge_lengths) in
                case.expected_chunk_edge_lengths.into_iter().enumerate()
            {
                assert_eq!(
                    encoded_chunk_grid.chunk_edge_lengths(axis).unwrap(),
                    expected_edge_lengths,
                    "{} axis {axis}",
                    case.name
                );
            }
        }
    }

    #[test]
    fn codec_reshape_encoded_chunk_grid_rejects_invalid_per_chunk_reshape() {
        struct TestCase {
            name: &'static str,
            reshape_shape: ReshapeShape,
            decoded_chunk_grid: ChunkGrid,
        }

        let cases = [
            TestCase {
                name: "fixed shape has the wrong element count for each chunk",
                reshape_shape: ReshapeShape(vec![
                    ReshapeDim::Size(nz(2)),
                    ReshapeDim::Size(nz(3)),
                    ReshapeDim::Size(nz(2)),
                ]),
                decoded_chunk_grid: ChunkGrid::new(
                    RegularChunkGrid::new(vec![12], vec![nz(6)]).unwrap(),
                ),
            },
            TestCase {
                name: "auto dimension cannot be resolved for each chunk",
                reshape_shape: ReshapeShape(vec![ReshapeDim::Size(nz(3)), ReshapeDim::auto()]),
                decoded_chunk_grid: ChunkGrid::new(
                    RegularChunkGrid::new(vec![12], vec![nz(4)]).unwrap(),
                ),
            },
        ];

        for case in cases {
            let codec = Arc::new(ReshapeCodec::new(case.reshape_shape))
                .with_context(data_type::uint8(), FillValue::from(0u8))
                .unwrap();

            assert!(
                matches!(
                    codec
                        .encoded_chunk_grid((&case.decoded_chunk_grid).into())
                        .unwrap(),
                    ChunkGridEncoded::None
                ),
                "{}: expected no encoded chunk grid",
                case.name
            );
        }
    }

    #[test]
    fn codec_reshape_encoded_chunk_grid_supports_array_shape_incompatible_reshape() {
        // Snapshot tests use this codec to reshape each decoded [4, 6] chunk
        // into an encoded [24] chunk.
        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![ReshapeDim::Size(nz(
            24,
        ))])))
        .with_context(data_type::uint8(), FillValue::from(0u8))
        .unwrap();

        let decoded_chunk_grid =
            ChunkGrid::new(RegularChunkGrid::new(vec![8, 24], vec![nz(4), nz(6)]).unwrap());

        let ChunkGridEncoded::Array(encoded_chunk_grid) = codec
            .encoded_chunk_grid((&decoded_chunk_grid).into())
            .unwrap()
        else {
            panic!("fixed per-chunk reshape is supported");
        };
        assert_eq!(encoded_chunk_grid.array_shape(), &[192]);
        assert_eq!(
            encoded_chunk_grid.chunk_edge_lengths(0).unwrap(),
            vec![nz(24); 8]
        );
    }

    fn partial_decoder_u16(
        codec: Arc<ReshapeCodec>,
        shape: &[NonZeroU64],
        elements: Vec<u16>,
    ) -> Arc<dyn ArrayPartialDecoderTraits> {
        let data_type = data_type::uint16();
        let fill_value = FillValue::from(0u16);
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();
        let codec = codec
            .with_context(data_type.clone(), fill_value.clone())
            .unwrap();
        let encoded = codec
            .encode(bytes, shape, &CodecOptions::default())
            .unwrap();
        let input_handle = Arc::new(encoded.into_fixed().unwrap());
        let bytes_codec = Arc::new(BytesCodec::default());
        let encoded_shape = codec.encoded_shape(shape).unwrap();
        let encoded_data_type = codec.encoded_data_type().clone();
        let encoded_fill_value = codec.encoded_fill_value().clone();
        let bytes_codec = bytes_codec
            .with_context(encoded_data_type.clone(), encoded_fill_value.clone())
            .unwrap();
        let input_handle = bytes_codec
            .partial_decoder(input_handle, &encoded_shape, &CodecOptions::default())
            .unwrap();
        codec
            .partial_decoder(input_handle, shape, &CodecOptions::default())
            .unwrap()
    }

    fn partial_decode_u16(
        partial_decoder: &dyn ArrayPartialDecoderTraits,
        indexer: &dyn Indexer,
    ) -> Vec<u16> {
        let decoded_partial_chunk = partial_decoder
            .partial_decode(indexer, &CodecOptions::default())
            .unwrap();
        crate::array::convert_from_bytes_slice::<u16>(&decoded_partial_chunk.into_fixed().unwrap())
            .to_vec()
    }

    fn partial_encode_u16(
        codec: Arc<ReshapeCodec>,
        shape: &[NonZeroU64],
        elements: Vec<u16>,
        indexer: &dyn Indexer,
        elements_partial_encode: Vec<u16>,
    ) -> Vec<u16> {
        let data_type = data_type::uint16();
        let fill_value = FillValue::from(0u16);
        let bytes = crate::array::transmute_to_bytes_vec(elements);
        let bytes: ArrayBytes = bytes.into();
        let codec = codec
            .with_context(data_type.clone(), fill_value.clone())
            .unwrap();
        let encoded = codec
            .encode(bytes, shape, &CodecOptions::default())
            .unwrap();

        let bytes_codec = Arc::new(BytesCodec::default());
        let encoded_shape = codec.encoded_shape(shape).unwrap();
        let encoded_data_type = codec.encoded_data_type().clone();
        let encoded_fill_value = codec.encoded_fill_value().clone();
        let bytes_codec = bytes_codec
            .with_context(encoded_data_type.clone(), encoded_fill_value.clone())
            .unwrap();
        let encoded_chunk = bytes_codec
            .encode(encoded, &encoded_shape, &CodecOptions::default())
            .unwrap()
            .into_owned();
        let output = Arc::new(Mutex::new(Some(encoded_chunk)));
        let input_output_handle = bytes_codec
            .clone()
            .partial_encoder(output.clone(), &encoded_shape, &CodecOptions::default())
            .unwrap();
        let partial_encoder = codec
            .clone()
            .partial_encoder(input_output_handle, shape, &CodecOptions::default())
            .unwrap();
        assert!(partial_encoder.supports_partial_encode());

        let bytes = crate::array::transmute_to_bytes_vec(elements_partial_encode);
        partial_encoder
            .partial_encode(indexer, &ArrayBytes::from(bytes), &CodecOptions::default())
            .unwrap();

        let output = output.lock().unwrap().clone().unwrap();
        let decoded_encoded = bytes_codec
            .decode(output.into(), &encoded_shape, &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(decoded_encoded, shape, &CodecOptions::default())
            .unwrap();
        crate::array::convert_from_bytes_slice::<u16>(&decoded.into_fixed().unwrap()).to_vec()
    }

    #[test]
    fn codec_reshape_partial_decode_array_subset() {
        // Decoded shape [2, 3, 4]:
        //
        //   decoded[0, :, :]          decoded[1, :, :]
        //   00 01 02 03              12 13 14 15
        //   04 05 06 07              16 17 18 19  <- select cols 1..4
        //   08 09 10 11              20 21 22 23  <- select cols 1..4
        //
        // Encoded shape [4, 6] after [[2], [0, 1]]:
        //
        //   00 01 02 03 04 05
        //   06 07 08 09 10 11
        //   12 13 14 15 16 17
        //   18 19 20 21 22 23
        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![
            ReshapeDim::InputDims(vec![2]),
            ReshapeDim::InputDims(vec![0, 1]),
        ])));
        let shape = vec![
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(4).unwrap(),
        ];
        let partial_decoder = partial_decoder_u16(codec, &shape, (0..24).collect());

        let decoded_region = ArraySubset::new_with_ranges(&[1..2, 1..3, 1..4]);
        assert_eq!(
            partial_decode_u16(partial_decoder.as_ref(), &decoded_region),
            [17, 18, 19, 21, 22, 23]
        );
    }

    #[test]
    fn codec_reshape_partial_decode_indexer() {
        // Decoded shape [2, 3, 4]:
        //
        //   decoded[0, :, :]          decoded[1, :, :]
        //   00 01 02 03              12 13 14 15
        //   04 05 06 07              16 17 18 19
        //   08 09 10 11              20 21 22 23
        //
        // Encoded shape [4, 6] after [[2], [0, 1]]:
        //
        //   00 01 02 03 04 05
        //   06 07 08 09 10 11
        //   12 13 14 15 16 17
        //   18 19 20 21 22 23
        //
        // Points:
        //   decoded[1, 2, 3] -> encoded[3, 5] -> 23
        //   decoded[0, 0, 1] -> encoded[0, 1] -> 01
        //   decoded[1, 0, 2] -> encoded[2, 2] -> 14
        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![
            ReshapeDim::InputDims(vec![2]),
            ReshapeDim::InputDims(vec![0, 1]),
        ])));
        let shape = vec![
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(4).unwrap(),
        ];
        let partial_decoder = partial_decoder_u16(codec, &shape, (0..24).collect());

        let indexer = vec![vec![1, 2, 3], vec![0, 0, 1], vec![1, 0, 2]];
        assert_eq!(
            partial_decode_u16(partial_decoder.as_ref(), &indexer),
            [23, 1, 14]
        );
    }

    #[test]
    fn codec_reshape_partial_decode_flatten_array_subset() {
        // Decoded shape [2, 3, 4] flattened to encoded shape [24]:
        //
        //   decoded[0, :, :]          decoded[1, :, :]
        //   00 01 02 03              12 13 14 15
        //   04 05 06 07  <- select   16 17 18 19  <- select
        //   08 09 10 11  <- select   20 21 22 23  <- select
        //
        //   encoded:
        //   00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![
            ReshapeDim::InputDims(vec![0, 1, 2]),
        ])));
        let shape = vec![
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(4).unwrap(),
        ];
        let partial_decoder = partial_decoder_u16(codec, &shape, (0..24).collect());

        let decoded_region = ArraySubset::new_with_ranges(&[0..2, 1..3, 2..4]);
        assert_eq!(
            partial_decode_u16(partial_decoder.as_ref(), &decoded_region),
            [6, 7, 10, 11, 18, 19, 22, 23]
        );
    }

    #[test]
    fn codec_reshape_partial_decode_auto_dimension() {
        // Decoded shape [2, 3, 4]:
        //
        //   decoded[0, :, :]          decoded[1, :, :]
        //   00 01 02 03              12 13 14 15
        //   04 05 06 07              16 17 18 19
        //   08 09 10 11              20 21 22 23
        //
        // Encoded shape [4, 6] after [4, -1]:
        //
        //   00 01 02 03 04 05
        //   06 07 08 09 10 11
        //   12 13 14 15 16 17
        //   18 19 20 21 22 23
        //
        // Points:
        //   decoded[0, 2, 3] -> encoded[1, 5] -> 11
        //   decoded[1, 0, 0] -> encoded[2, 0] -> 12
        //   decoded[1, 2, 2] -> encoded[3, 4] -> 22
        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![
            ReshapeDim::Size(NonZeroU64::new(4).unwrap()),
            ReshapeDim::auto(),
        ])));
        let shape = vec![
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(4).unwrap(),
        ];
        let partial_decoder = partial_decoder_u16(codec, &shape, (0..24).collect());

        let indexer = vec![vec![0, 2, 3], vec![1, 0, 0], vec![1, 2, 2]];
        assert_eq!(
            partial_decode_u16(partial_decoder.as_ref(), &indexer),
            [11, 12, 22]
        );
    }

    #[test]
    fn codec_reshape_partial_decode_1d_to_nd() {
        // Decoded shape [12]:
        //
        //   00 01 02 03 04 05 06 07 08 09 10 11
        //
        // Encoded shape [2, 3, 2]:
        //
        //   encoded[0, :, :]          encoded[1, :, :]
        //   00 01                    06 07
        //   02 03                    08 09
        //   04 05                    10 11
        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![
            ReshapeDim::Size(NonZeroU64::new(2).unwrap()),
            ReshapeDim::Size(NonZeroU64::new(3).unwrap()),
            ReshapeDim::Size(NonZeroU64::new(2).unwrap()),
        ])));
        let shape = vec![NonZeroU64::new(12).unwrap()];
        let partial_decoder = partial_decoder_u16(codec, &shape, (0..12).collect());

        #[expect(clippy::single_range_in_vec_init)]
        let decoded_region = ArraySubset::new_with_ranges(&[3..10]);
        assert_eq!(
            partial_decode_u16(partial_decoder.as_ref(), &decoded_region),
            [3, 4, 5, 6, 7, 8, 9]
        );
    }

    #[test]
    fn codec_reshape_partial_decode_composite_indexer() {
        // Decoded shape [2, 3, 4]:
        //
        //   decoded[0, :, :]          decoded[1, :, :]
        //   00 01 02 03  <- select   12 13 14 15
        //   04 05 06 07              16 17 18 19
        //   08 09 10 11              20 21 22 23  <- select
        //
        // The composite indexer requests two disjoint decoded regions:
        //   [0..1, 0..1, 2..4] -> 02 03
        //   [1..2, 2..3, 0..2] -> 20 21
        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![
            ReshapeDim::InputDims(vec![2]),
            ReshapeDim::InputDims(vec![0, 1]),
        ])));
        let shape = vec![
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(4).unwrap(),
        ];
        let partial_decoder = partial_decoder_u16(codec, &shape, (0..24).collect());

        let decoded_regions = [
            ArraySubset::new_with_ranges(&[0..1, 0..1, 2..4]),
            ArraySubset::new_with_ranges(&[1..2, 2..3, 0..2]),
        ];
        assert_eq!(
            partial_decode_u16(partial_decoder.as_ref(), &decoded_regions),
            [2, 3, 20, 21]
        );
    }

    #[test]
    fn codec_reshape_partial_decode_invalid_indexers() {
        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![
            ReshapeDim::InputDims(vec![2]),
            ReshapeDim::InputDims(vec![0, 1]),
        ])));
        let shape = vec![
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(4).unwrap(),
        ];
        let partial_decoder = partial_decoder_u16(codec, &shape, (0..24).collect());

        let wrong_dimensionality = ArraySubset::new_with_ranges(&[0..1, 0..1]);
        assert!(
            partial_decoder
                .partial_decode(&wrong_dimensionality, &CodecOptions::default())
                .is_err()
        );

        let out_of_bounds = vec![vec![2, 0, 0]];
        assert!(
            partial_decoder
                .partial_decode(&out_of_bounds, &CodecOptions::default())
                .is_err()
        );
    }

    #[test]
    fn codec_reshape_partial_encode_array_subset() {
        // Decoded shape [2, 3, 4]:
        //
        //   decoded[0, :, :]          decoded[1, :, :]
        //   00 01 02 03              12 13 14 15
        //   04 05 06 07              16 17 18 19  <- write 100 101 102
        //   08 09 10 11              20 21 22 23  <- write 103 104 105
        //
        // Encoded shape [4, 6] after [[2], [0, 1]]:
        //
        //   00 01 02 03 04 05
        //   06 07 08 09 10 11
        //   12 13 14 15 16 17
        //   18 19 20 21 22 23
        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![
            ReshapeDim::InputDims(vec![2]),
            ReshapeDim::InputDims(vec![0, 1]),
        ])));
        let shape = vec![
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(4).unwrap(),
        ];
        let decoded_region = ArraySubset::new_with_ranges(&[1..2, 1..3, 1..4]);

        assert_eq!(
            partial_encode_u16(
                codec,
                &shape,
                (0..24).collect(),
                &decoded_region,
                vec![100, 101, 102, 103, 104, 105],
            ),
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 100, 101, 102, 20, 103,
                104, 105,
            ]
        );
    }

    #[test]
    fn codec_reshape_partial_encode_indexer() {
        // Decoded shape [2, 3, 4]:
        //
        //   decoded[0, :, :]          decoded[1, :, :]
        //   00 01 02 03              12 13 14 15
        //   04 05 06 07              16 17 18 19
        //   08 09 10 11              20 21 22 23
        //
        // Encoded shape [4, 6] after [[2], [0, 1]]:
        //
        //   00 01 02 03 04 05
        //   06 07 08 09 10 11
        //   12 13 14 15 16 17
        //   18 19 20 21 22 23
        //
        // Writes:
        //   decoded[1, 2, 3] -> encoded[3, 5] <- 100
        //   decoded[0, 0, 1] -> encoded[0, 1] <- 101
        //   decoded[1, 0, 2] -> encoded[2, 2] <- 102
        let codec = Arc::new(ReshapeCodec::new(ReshapeShape(vec![
            ReshapeDim::InputDims(vec![2]),
            ReshapeDim::InputDims(vec![0, 1]),
        ])));
        let shape = vec![
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
            NonZeroU64::new(4).unwrap(),
        ];
        let indexer = vec![vec![1, 2, 3], vec![0, 0, 1], vec![1, 0, 2]];

        assert_eq!(
            partial_encode_u16(
                codec,
                &shape,
                (0..24).collect(),
                &indexer,
                vec![100, 101, 102],
            ),
            [
                0, 101, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 102, 15, 16, 17, 18, 19, 20, 21,
                22, 100,
            ]
        );
    }
}
