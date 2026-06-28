//! The `reshape` array to array codec (Experimental).
//!
//! Performs a reshaping operation.
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ### Compatible Implementations
//! None
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/blob/7295bf1ec15c978f1a63b90d55891712b950c797/codecs/reshape/README.md>
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

mod reshape_codec;
mod reshape_codec_partial;

use std::sync::Arc;

use num::Integer;
pub use reshape_codec::ReshapeCodec;
use zarrs_chunk_grid::ChunkShapeTraits;
use zarrs_metadata::v3::MetadataV3;

// use itertools::Itertools;
use crate::array::{ArrayIndices, ChunkShape, Indexer, IndexerError, unravel_index};
use zarrs_codec::{Codec, CodecError, CodecPluginV3, CodecTraitsV3};
pub use zarrs_metadata_ext::codec::reshape::{
    ReshapeCodecConfiguration, ReshapeCodecConfigurationV1, ReshapeDim, ReshapeShape,
};

fn get_encoded_shape(
    reshape_shape: &ReshapeShape,
    decoded_shape: &[u64],
) -> Result<ChunkShape, CodecError> {
    let mut encoded_shape = Vec::with_capacity(reshape_shape.0.len());
    let mut fill_index = None;
    for output_dim in &reshape_shape.0 {
        match output_dim {
            ReshapeDim::Size(size) => encoded_shape.push(size.get()),
            ReshapeDim::InputDims(input_dims) => {
                let mut product: u64 = 1;
                for input_dim in input_dims {
                    let input_shape = *decoded_shape
                        .get(usize::try_from(*input_dim).unwrap())
                        .ok_or_else(|| {
                            CodecError::Other(
                                format!("reshape codec shape references a dimension ({input_dim}) larger than the chunk dimensionality ({})", decoded_shape.len()),
                            )
                        })?;
                    product = product.checked_mul(input_shape).unwrap();
                }
                encoded_shape.push(product);
            }
            ReshapeDim::Auto(_) => {
                fill_index = Some(encoded_shape.len());
                encoded_shape.push(1);
            }
        }
    }

    let num_elements_input = decoded_shape.num_elements();
    let num_elements_output = encoded_shape.num_elements();
    if let Some(fill_index) = fill_index {
        let (quot, rem) = num_elements_input.div_rem(&num_elements_output);
        if rem == 0 {
            encoded_shape[fill_index] = quot;
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
    decoded_shape: &[u64],
    encoded_shape: &[u64],
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
    use crate::array::chunk_grid::RegularChunkGrid;
    use crate::array::codec::BytesCodec;
    use crate::array::{ArrayBytes, ArraySubset, ChunkShapeTraits, DataType, FillValue, data_type};
    use zarrs_chunk_grid::ChunkGrid;
    use zarrs_codec::{
        ArrayPartialDecoderTraits, CodecOptions, SubchunkGrid, UnboundArrayToArrayCodecTraits,
        UnboundArrayToBytesCodecTraits,
    };
    use zarrs_metadata::ChunkShapeNonEmpty;

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).unwrap()
    }

    fn codec_reshape_round_trip_impl(
        json: &str,
        data_type: DataType,
        fill_value: FillValue,
        output_shape: ChunkShape,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let shape = vec![5, 4, 4, 3];
        let size = shape.num_elements() as usize * data_type.fixed_size().unwrap();
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
        let output_shape = vec![20, 4, 3];
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
        let output_shape = vec![20, 4, 3];
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
        let output_shape = vec![80, 3];
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
        let output_shape = vec![5, 4, 12];
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
        let output_shape = vec![5, 16, 3];
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
        let output_shape = vec![20, 2, 2, 3];
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
        let output_shape = vec![20, 2, 2, 3];
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
        let output_shape = vec![20, 2, 2, 3];
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

    fn test_reshape_partial_decode_granularity(
        reshape_shape: ReshapeShape,
        decoded_shape: ChunkShape,
        encoded_subchunk_shape: ChunkShapeNonEmpty,
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
        let encoded_shape = codec.encoded_shape(&decoded_shape).unwrap();
        let chunk_grid = ChunkGrid::new(
            RegularChunkGrid::new(decoded_shape.clone(), decoded_shape_nonzero).unwrap(),
        );
        let encoded_subchunk_grid =
            ChunkGrid::new(RegularChunkGrid::new(encoded_shape, encoded_subchunk_shape).unwrap());
        let subchunk_grid = codec
            .decoded_subchunk_grid(&chunk_grid, &encoded_subchunk_grid)
            .unwrap();
        let SubchunkGrid::Array(subchunk_grid) = subchunk_grid else {
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
            vec![vec![nz(2)], vec![nz(3)], vec![nz(4)]],
        );
        test_reshape_partial_decode_granularity(
            ReshapeShape(vec![
                ReshapeDim::InputDims(vec![2]),
                ReshapeDim::InputDims(vec![0, 1]),
            ]),
            vec![2, 3, 4],
            vec![nz(2), nz(6)],
            vec![vec![nz(1), nz(1)], vec![nz(3)], vec![nz(4)]],
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
                .decoded_subchunk_grid(&chunk_grid, &encoded_subchunk_grid)
                .is_err()
        );
    }

    fn partial_decoder_u16(
        codec: Arc<ReshapeCodec>,
        shape: &[u64],
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
        shape: &[u64],
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
        let shape = vec![2, 3, 4];
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
        let shape = vec![2, 3, 4];
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
        let shape = vec![2, 3, 4];
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
        let shape = vec![2, 3, 4];
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
        let shape = vec![12];
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
        let shape = vec![2, 3, 4];
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
        let shape = vec![2, 3, 4];
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
        let shape = vec![2, 3, 4];
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
        let shape = vec![2, 3, 4];
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
