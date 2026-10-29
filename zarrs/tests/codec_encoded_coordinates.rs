#![allow(missing_docs)]

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::chunk_grid::RegularChunkGrid;
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

/// Check that every element of a decoded chunk is encoded at its `encoded_coordinates`.
fn check_encoded_coordinates(codec: &Arc<dyn ArrayToArrayCodecTraits>, decoded_shape: &[u64]) {
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

    for decoded_coordinates in ArraySubset::new_with_shape(decoded_shape_u64).indices() {
        let decoded_index = ravel_indices(&decoded_coordinates, decoded_shape_u64).unwrap();
        let encoded_coordinates = codec
            .encoded_coordinates(&decoded_shape, &decoded_coordinates)
            .unwrap();
        let encoded_index = ravel_indices(&encoded_coordinates, encoded_shape_u64).unwrap();
        assert_eq!(
            encoded[usize::try_from(encoded_index).unwrap()],
            decoded[usize::try_from(decoded_index).unwrap()],
            "{decoded_coordinates:?} -> {encoded_coordinates:?}"
        );
    }

    // Out of bounds coordinates are rejected
    assert!(
        codec
            .encoded_coordinates(&decoded_shape, decoded_shape_u64)
            .is_err()
    );
    assert!(codec.encoded_coordinates(&decoded_shape, &[]).is_err());
}

/// Check that the decoded local subchunk grid addresses the same subchunks as `encoded_subchunk_shape` in the encoded
/// chunk, with elements mapped by `encoded_coordinates`.
fn check_decoded_local_subchunk_grid(
    codec: &Arc<dyn ArrayToArrayCodecTraits>,
    decoded_shape: &[u64],
    encoded_subchunk_shape: &[u64],
) {
    let decoded_shape = shape(decoded_shape);
    let decoded_shape_u64: &[u64] = bytemuck::must_cast_slice(&decoded_shape);
    let encoded_shape = codec.encoded_shape(&decoded_shape).unwrap();
    let encoded_grid = ChunkGrid::new(
        RegularChunkGrid::new(
            bytemuck::must_cast_slice(&encoded_shape).to_vec(),
            shape(encoded_subchunk_shape),
        )
        .unwrap(),
    );
    let decoded_grid = codec
        .decoded_local_subchunk_grid(&decoded_shape, &encoded_grid)
        .unwrap()
        .unwrap();
    assert_eq!(decoded_grid.array_shape(), decoded_shape_u64);

    // Each decoded subchunk maps to exactly one encoded subchunk, and vice versa
    let mut decoded_to_encoded = std::collections::HashMap::new();
    let mut encoded_to_decoded = std::collections::HashMap::new();
    for decoded_coordinates in ArraySubset::new_with_shape(decoded_shape_u64).indices() {
        let decoded_subchunk = decoded_grid
            .chunk_indices(&decoded_coordinates)
            .unwrap()
            .unwrap()
            .to_vec();
        let encoded_coordinates = codec
            .encoded_coordinates(&decoded_shape, &decoded_coordinates)
            .unwrap();
        let encoded_subchunk = encoded_grid
            .chunk_indices(&encoded_coordinates)
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
fn encoded_coordinates_transpose() {
    let codec = bind(Arc::new(TransposeCodec::new(
        TransposeOrder::new(&[2, 0, 1]).unwrap(),
    )));
    check_encoded_coordinates(&codec, &[2, 3, 4]);
    check_decoded_local_subchunk_grid(&codec, &[2, 3, 4], &[2, 1, 3]);
}

#[test]
fn encoded_coordinates_reshape() {
    let codec = bind(Arc::new(ReshapeCodec::new(
        ReshapeShape::new(vec![ReshapeDim::from(shape(&[3])[0]), ReshapeDim::auto()]).unwrap(),
    )));
    check_encoded_coordinates(&codec, &[2, 3, 4]);
    // [2, 3, 4] is encoded as [3, 8]
    check_decoded_local_subchunk_grid(&codec, &[2, 3, 4], &[1, 4]);
}

#[test]
fn encoded_coordinates_squeeze() {
    let codec = bind(Arc::new(SqueezeCodec::new()));
    check_encoded_coordinates(&codec, &[1, 3, 1, 4]);
    check_encoded_coordinates(&codec, &[1, 1]);
    check_decoded_local_subchunk_grid(&codec, &[1, 3, 1, 4], &[3, 2]);
}
