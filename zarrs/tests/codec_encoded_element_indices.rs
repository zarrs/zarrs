#![allow(missing_docs)]

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::{ReshapeCodec, ReshapeDim, SqueezeCodec, TransposeCodec, TransposeOrder};
use zarrs::array::{
    ArrayBytes, ArraySubset, ChunkShapeTraits, FillValue, data_type, ravel_indices,
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
        assert!(
            codec
                .encoded_element_indices(&decoded_shape, &outside)
                .is_err()
        );
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
