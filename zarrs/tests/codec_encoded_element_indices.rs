#![allow(missing_docs)]
use zarrs::array::chunk_grid::RegularChunkGrid;

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::{ReshapeCodec, ReshapeDim, SqueezeCodec, TransposeCodec, TransposeOrder};
use zarrs::array::{
    ArrayBytes, ArraySubset, ChunkGrid, ChunkShapeTraits, FillValue, data_type, ravel_indices,
};
use zarrs::metadata_ext::codec::reshape::ReshapeShape;
use zarrs_codec::{
    ArrayToArrayCodecTraits, CodecOptions, CodecSpecificOptions, UnboundArrayToArrayCodecTraits,
};

fn shape(shape: &[u64]) -> Vec<NonZeroU64> {
    shape.iter().map(|&s| NonZeroU64::new(s).unwrap()).collect()
}

fn bind(codec: Arc<dyn UnboundArrayToArrayCodecTraits>) -> Arc<dyn ArrayToArrayCodecTraits> {
    codec
        .with_context(
            data_type::uint32(),
            Some(FillValue::from(0u32)),
            &CodecSpecificOptions::default(),
        )
        .unwrap()
}

/// Check that every element of a decoded chunk is encoded at its `encoded_element_indices`.
fn check_encoded_element_indices(codec: &Arc<dyn ArrayToArrayCodecTraits>, decoded_shape: &[u64]) {
    let decoded_shape = shape(decoded_shape);
    let decoded_shape_u64: &[u64] = bytemuck::must_cast_slice(&decoded_shape);
    let num_elements = decoded_shape.num_elements_u64();
    let decoded: Vec<u32> = (0..u32::try_from(num_elements).unwrap()).collect();
    let encoded = codec
        .encode(
            ArrayBytes::new_flen(bytemuck::cast_slice::<u32, u8>(&decoded).to_vec()),
            &decoded_shape,
            &CodecOptions::default(),
        )
        .unwrap();
    let encoded: Vec<u32> = bytemuck::pod_collect_to_vec(&encoded.into_fixed().unwrap());
    let encoded_shape = codec.encoded_shape(&decoded_shape).unwrap();
    let encoded_shape_u64: &[u64] = bytemuck::must_cast_slice(&encoded_shape);

    for decoded_element_indices in ArraySubset::new_with_shape(decoded_shape_u64).indices() {
        let decoded_index = ravel_indices(&decoded_element_indices, decoded_shape_u64).unwrap();
        let encoded_element_indices = codec
            .encoded_element_indices(&decoded_shape, &decoded_element_indices)
            .unwrap()
            .unwrap();
        let encoded_index = ravel_indices(&encoded_element_indices, encoded_shape_u64).unwrap();
        assert_eq!(
            encoded[usize::try_from(encoded_index).unwrap()],
            decoded[usize::try_from(decoded_index).unwrap()],
            "{decoded_element_indices:?} -> {encoded_element_indices:?}"
        );
    }

    // Out of bounds element indices are rejected
    assert!(
        codec
            .encoded_element_indices(&decoded_shape, decoded_shape_u64)
            .is_err()
    );
    for element_indices in [
        vec![],
        vec![0; decoded_shape.len() - 1],
        vec![0; decoded_shape.len() + 1],
    ] {
        assert!(
            codec
                .encoded_element_indices(&decoded_shape, &element_indices)
                .is_err()
        );
    }
    for axis in 0..decoded_shape.len() {
        let mut outside = vec![0; decoded_shape.len()];
        outside[axis] = decoded_shape[axis].get();
        assert!(codec.encoded_element_indices(&decoded_shape, &outside).is_err());
    }
}

#[test]
fn encoded_element_indices_transpose() {
    let codec = bind(Arc::new(TransposeCodec::new(
        TransposeOrder::new(&[2, 0, 1]).unwrap(),
    )));
    check_encoded_element_indices(&codec, &[2, 3, 4]);
}

#[test]
fn encoded_element_indices_reshape() {
    let codec = bind(Arc::new(ReshapeCodec::new(
        ReshapeShape::new(vec![ReshapeDim::from(shape(&[3])[0]), ReshapeDim::auto()]).unwrap(),
    )));
    check_encoded_element_indices(&codec, &[2, 3, 4]);
    // [2, 3, 4] is encoded as [3, 8]
}

#[test]
fn encoded_element_indices_squeeze() {
    let codec = bind(Arc::new(SqueezeCodec::new()));
    check_encoded_element_indices(&codec, &[1, 3, 1, 4]);
    check_encoded_element_indices(&codec, &[1, 1]);
}

#[test]
fn identity_element_indices_are_independent_of_value_changes() {
    #[cfg(feature = "bitround")]
    use zarrs::array::codec::BitroundCodec;
    use zarrs::array::codec::{CastValueCodec, FixedScaleOffsetCodec};
    let cast = CastValueCodec::new_with_configuration(
        &serde_json::from_value(serde_json::json!({"data_type": "float64"})).unwrap(),
    )
    .unwrap();
    let scale = FixedScaleOffsetCodec::new_with_configuration(
        &serde_json::from_value(serde_json::json!({"offset": 1, "scale": 2, "dtype": "f4"}))
            .unwrap(),
    )
    .unwrap();
    let codecs: Vec<Arc<dyn UnboundArrayToArrayCodecTraits>> = vec![
        #[cfg(feature = "bitround")]
        Arc::new(BitroundCodec::new(3)),
        Arc::new(cast),
        Arc::new(scale),
    ];
    for codec in codecs {
        let codec = codec
            .with_context(
                data_type::float32(),
                Some(0f32.into()),
                &CodecSpecificOptions::default(),
            )
            .unwrap();
        for element_indices in ArraySubset::new_with_shape(vec![2, 3]).indices() {
            assert_eq!(
                codec
                    .encoded_element_indices(&shape(&[2, 3]), &element_indices)
                    .unwrap()
                    .unwrap(),
                element_indices
            );
        }
        for invalid in [vec![], vec![0], vec![0, 0, 0], vec![2, 0], vec![0, 3]] {
            assert!(
                codec
                    .encoded_element_indices(&shape(&[2, 3]), &invalid)
                    .is_err()
            );
        }
        let input = bytemuck::cast_slice::<f32, u8>(&[
            1.12345, 2.23456, 3.34567, 4.45678, 5.56789, 6.67891,
        ])
        .to_vec();
        let output = codec
            .encode(
                ArrayBytes::new_flen(input.clone()),
                &shape(&[2, 3]),
                &CodecOptions::default(),
            )
            .unwrap()
            .into_fixed()
            .unwrap();
        assert_ne!(output.as_ref(), input);
    }
}

#[test]
fn packbits_has_no_element_mapping() {
    use zarrs::array::codec::NumcodecsPackBitsCodec;
    let codec = NumcodecsPackBitsCodec::new()
        .with_context(
            data_type::bool(),
            Some(false.into()),
            &CodecSpecificOptions::default(),
        )
        .unwrap();
    assert!(
        codec
            .encoded_element_indices(&shape(&[2, 3]), &[0, 0])
            .unwrap()
            .is_none()
    );
    for element_indices in [vec![], vec![0], vec![0, 0, 0], vec![2, 0]] {
        assert!(
            codec
                .encoded_element_indices(&shape(&[2, 3]), &element_indices)
                .is_err()
        );
    }
}

/// Check that the decoded local subchunk grid addresses the same subchunks as `encoded_subchunk_shape` in the encoded
/// chunk, with elements mapped by `encoded_element_indices`.
fn check_decoded_local_subchunk_grid(
    codec: &Arc<dyn ArrayToArrayCodecTraits>,
    decoded_shape: &[u64],
    encoded_subchunk_shape: &[u64],
) {
    let decoded_shape = shape(decoded_shape);
    let encoded_shape = codec.encoded_shape(&decoded_shape).unwrap();
    let encoded_grid = ChunkGrid::new(
        RegularChunkGrid::new(
            bytemuck::must_cast_slice(&encoded_shape).to_vec(),
            shape(encoded_subchunk_shape),
        )
        .unwrap(),
    );
    check_grid(codec, &decoded_shape, &encoded_grid);
}

fn check_grid(
    codec: &Arc<dyn ArrayToArrayCodecTraits>,
    decoded_shape: &[NonZeroU64],
    encoded_grid: &ChunkGrid,
) {
    let decoded_shape_u64: &[u64] = bytemuck::must_cast_slice(decoded_shape);
    let decoded_grid = codec
        .decoded_local_subchunk_grid(decoded_shape, encoded_grid)
        .unwrap()
        .unwrap();
    assert_eq!(decoded_grid.array_shape(), decoded_shape_u64);

    // Each decoded subchunk maps to exactly one encoded subchunk, and vice versa
    let mut decoded_to_encoded = std::collections::HashMap::new();
    let mut encoded_to_decoded = std::collections::HashMap::new();
    for decoded_element_indices in ArraySubset::new_with_shape(decoded_shape_u64).indices() {
        let decoded_subchunk = decoded_grid
            .chunk_indices(&decoded_element_indices)
            .unwrap()
            .unwrap()
            .to_vec();
        let encoded_element_indices = codec
            .encoded_element_indices(decoded_shape, &decoded_element_indices)
            .unwrap()
            .unwrap();
        let encoded_subchunk = encoded_grid
            .chunk_indices(&encoded_element_indices)
            .unwrap()
            .unwrap()
            .to_vec();
        assert_eq!(
            decoded_to_encoded
                .entry(decoded_subchunk.clone())
                .or_insert_with(|| encoded_subchunk.clone()),
            &encoded_subchunk
        );
        assert_eq!(
            encoded_to_decoded
                .entry(encoded_subchunk)
                .or_insert_with(|| decoded_subchunk.clone()),
            &decoded_subchunk
        );
    }
}

#[test]
fn local_grid_mappings() {
    check_decoded_local_subchunk_grid(
        &bind(Arc::new(TransposeCodec::new(
            TransposeOrder::new(&[2, 0, 1]).unwrap(),
        ))),
        &[2, 3, 4],
        &[3, 1, 2],
    );
    check_decoded_local_subchunk_grid(&bind(Arc::new(SqueezeCodec::new())), &[1, 3, 1, 4], &[2, 3]);
    check_decoded_local_subchunk_grid(&bind(Arc::new(SqueezeCodec::new())), &[1, 1], &[1]);
    let reshape = bind(Arc::new(ReshapeCodec::new(
        ReshapeShape::new(vec![ReshapeDim::from(shape(&[3])[0]), ReshapeDim::auto()]).unwrap(),
    )));
    check_decoded_local_subchunk_grid(&reshape, &[2, 3, 4], &[1, 4]);
}

#[test]
fn local_grid_rectilinear_negative_and_identity_contracts() {
    use zarrs::array::chunk_grid::{ChunkEdgeLengths, RectilinearChunkGrid};
    use zarrs::array::codec::{CastValueCodec, NumcodecsPackBitsCodec};
    let grid = ChunkGrid::new(
        RectilinearChunkGrid::new(
            vec![4, 3],
            &[
                ChunkEdgeLengths::encode(&shape(&[1, 3])),
                ChunkEdgeLengths::encode(&shape(&[2, 1])),
            ],
        )
        .unwrap(),
    );
    let transpose = bind(Arc::new(TransposeCodec::new(
        TransposeOrder::new(&[1, 0]).unwrap(),
    )));
    check_grid(&transpose, &shape(&[3, 4]), &grid);
    let squeeze = bind(Arc::new(SqueezeCodec::new()));
    check_grid(&squeeze, &shape(&[1, 4, 1, 3]), &grid);
    let identity = bind(Arc::new(
        CastValueCodec::new_with_configuration(
            &serde_json::from_value(serde_json::json!({"data_type": "uint32"})).unwrap(),
        )
        .unwrap(),
    ));
    check_grid(&identity, &shape(&[4, 3]), &grid);
    let wrong = ChunkGrid::new(RegularChunkGrid::new(vec![12], shape(&[2])).unwrap());
    let wrong_shape = ChunkGrid::new(RegularChunkGrid::new(vec![2, 2], shape(&[2, 2])).unwrap());
    for invalid_grid in [&wrong, &wrong_shape] {
        assert!(
            identity
                .decoded_local_subchunk_grid(&shape(&[4, 3]), invalid_grid)
                .is_err()
        );
        assert!(
            transpose
                .decoded_local_subchunk_grid(&shape(&[3, 4]), invalid_grid)
                .is_err()
        );
        assert!(
            squeeze
                .decoded_local_subchunk_grid(&shape(&[1, 4, 1, 3]), invalid_grid)
                .is_err()
        );
    }
    let reshape = bind(Arc::new(ReshapeCodec::new(
        ReshapeShape::new(vec![ReshapeDim::from(shape(&[3])[0]), ReshapeDim::auto()]).unwrap(),
    )));
    // Encoded columns do not become rectangles in [2,6].
    let columns = ChunkGrid::new(RegularChunkGrid::new(vec![3, 4], shape(&[3, 1])).unwrap());
    assert!(
        reshape
            .decoded_local_subchunk_grid(&shape(&[2, 6]), &columns)
            .unwrap()
            .is_none()
    );
    for invalid_grid in [&wrong, &wrong_shape] {
        assert!(
            reshape
                .decoded_local_subchunk_grid(&shape(&[2, 6]), invalid_grid)
                .is_err()
        );
    }
    let packbits = NumcodecsPackBitsCodec::new()
        .with_context(
            data_type::bool(),
            Some(false.into()),
            &CodecSpecificOptions::default(),
        )
        .unwrap();
    // [4, 3] booleans pack into 1 + ceil(12 / 8) = 3 bytes
    let packed_grid = ChunkGrid::new(RegularChunkGrid::new(vec![3], shape(&[1])).unwrap());
    assert!(
        packbits
            .decoded_local_subchunk_grid(&shape(&[4, 3]), &packed_grid)
            .unwrap()
            .is_none()
    );
    for invalid_grid in [&grid, &wrong] {
        assert!(
            packbits
                .decoded_local_subchunk_grid(&shape(&[4, 3]), invalid_grid)
                .is_err()
        );
    }
}

#[test]
fn local_grid_hierarchy_preserves_order_none_and_errors() {
    let codec = bind(Arc::new(TransposeCodec::new(
        TransposeOrder::new(&[1, 0]).unwrap(),
    )));
    let outer = ChunkGrid::new(RegularChunkGrid::new(vec![4, 6], shape(&[3, 4])).unwrap());
    let inner = ChunkGrid::new(RegularChunkGrid::new(vec![4, 6], shape(&[1, 2])).unwrap());
    let mapped = codec
        .decoded_local_subchunk_grids(
            &shape(&[6, 4]),
            vec![Some(outer.clone()), None, Some(inner.clone())],
        )
        .unwrap();
    assert_eq!(mapped.len(), 3);
    assert!(mapped[1].is_none());
    for (level, grid) in [(0, &outer), (2, &inner)] {
        let expected = codec
            .decoded_local_subchunk_grid(&shape(&[6, 4]), grid)
            .unwrap()
            .unwrap();
        assert_eq!(
            mapped[level]
                .as_ref()
                .unwrap()
                .chunk_shape(&[0, 0])
                .unwrap(),
            expected.chunk_shape(&[0, 0]).unwrap()
        );
    }
    assert!(
        codec
            .decoded_local_subchunk_grids(&shape(&[6, 4]), vec![])
            .unwrap()
            .is_empty()
    );
    let wrong = ChunkGrid::new(RegularChunkGrid::new(vec![24], shape(&[2])).unwrap());
    assert!(
        codec
            .decoded_local_subchunk_grids(&shape(&[6, 4]), vec![None, Some(wrong)])
            .is_err()
    );
}
