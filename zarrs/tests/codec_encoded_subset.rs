#![allow(missing_docs)]

use std::collections::BTreeSet;
use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::codec::{
    CastValueCodec, NumcodecsPackBitsCodec, ReshapeCodec, ReshapeDim, ReshapeShape, SqueezeCodec,
    TransposeCodec, TransposeOrder,
};
use zarrs::array::{ArrayBytes, ArraySubset, data_type};
use zarrs_codec::{
    ArrayToArrayCodecTraits, CodecOptions, CodecSpecificOptions, UnboundArrayToArrayCodecTraits,
};

fn shape(s: &[u64]) -> Vec<NonZeroU64> {
    s.iter().map(|&n| NonZeroU64::new(n).unwrap()).collect()
}
fn bind(codec: impl UnboundArrayToArrayCodecTraits) -> Arc<dyn ArrayToArrayCodecTraits> {
    codec
        .with_context(
            data_type::uint32(),
            Some(0u32.into()),
            &CodecSpecificOptions::default(),
        )
        .unwrap()
}
fn reshape(s: &[u64]) -> Arc<dyn ArrayToArrayCodecTraits> {
    bind(ReshapeCodec::new(
        ReshapeShape::new(shape(s).into_iter().map(ReshapeDim::from)).unwrap(),
    ))
}
fn subsets(s: &[u64]) -> Vec<ArraySubset> {
    fn visit(s: &[u64], ranges: &mut Vec<std::ops::Range<u64>>, out: &mut Vec<ArraySubset>) {
        if ranges.len() == s.len() {
            out.push(ArraySubset::new_with_ranges(ranges));
            return;
        }
        for start in 0..s[ranges.len()] {
            for end in start + 1..=s[ranges.len()] {
                ranges.push(start..end);
                visit(s, ranges, out);
                ranges.pop();
            }
        }
    }
    let mut out = vec![];
    visit(s, &mut vec![], &mut out);
    out
}
fn linear_set(subset: &ArraySubset, s: &[u64]) -> BTreeSet<u64> {
    // Independent element enumeration, deliberately not the production contiguous-run comparison.
    subset
        .indices()
        .into_iter()
        .map(|coordinates| coordinates.iter().zip(s).fold(0, |i, (&c, &n)| i * n + c))
        .collect()
}
fn check_all(codec: Arc<dyn ArrayToArrayCodecTraits>, s: &[u64]) {
    let encoded_shape: Vec<_> = codec
        .encoded_shape(&shape(s))
        .unwrap()
        .iter()
        .map(|n| n.get())
        .collect();
    let values: Vec<u32> = (0..u32::try_from(s.iter().product::<u64>()).unwrap()).collect();
    let encoded = codec
        .encode(
            ArrayBytes::new_flen(bytemuck::cast_slice(&values).to_vec()),
            &shape(s),
            &CodecOptions::default(),
        )
        .unwrap()
        .into_fixed()
        .unwrap();
    let encoded_values: Vec<u32> = bytemuck::pod_collect_to_vec(&encoded);
    let rectangles: Vec<_> = subsets(&encoded_shape)
        .iter()
        .map(|subset| linear_set(subset, &encoded_shape))
        .collect();
    for subset in subsets(s) {
        let selected = linear_set(&subset, s);
        let positions: BTreeSet<u64> = encoded_values
            .iter()
            .enumerate()
            .filter(|(_, v)| selected.contains(&u64::from(**v)))
            .map(|(i, _)| u64::try_from(i).unwrap())
            .collect();
        let expected_representable = rectangles.contains(&positions);
        let result = codec.encoded_subset(&shape(s), &subset).unwrap();
        assert_eq!(
            result.is_some(),
            expected_representable,
            "{s:?} -> {encoded_shape:?}: {subset:?}: {result:?}"
        );
        if let Some(mapped) = result {
            assert_eq!(
                linear_set(&mapped, &encoded_shape),
                positions,
                "{subset:?} -> {mapped:?}"
            );
        }
    }
    for subset in [
        ArraySubset::new_empty(s.len()),
        ArraySubset::new_with_shape(vec![1; s.len() + 1]),
        ArraySubset::new_with_start_shape(s, vec![1; s.len()]).unwrap(),
    ] {
        assert!(codec.encoded_subset(&shape(s), &subset).is_err());
    }
}

#[test]
fn exhaustive_small_subset_contracts() {
    check_all(
        bind(TransposeCodec::new(TransposeOrder::new(&[1, 0]).unwrap())),
        &[3, 4],
    );
    check_all(bind(SqueezeCodec::new()), &[1, 3, 1, 4]);
    check_all(bind(SqueezeCodec::new()), &[1, 1]);
    check_all(
        bind(
            CastValueCodec::new_with_configuration(
                &serde_json::from_value(serde_json::json!({"data_type":"uint32"})).unwrap(),
            )
            .unwrap(),
        ),
        &[3, 4],
    );
    #[cfg(feature = "bitround")]
    check_all(bind(zarrs::array::codec::BitroundCodec::new(32)), &[3, 4]);
    check_all(reshape(&[5, 2]), &[2, 5]);
    check_all(reshape(&[3, 8]), &[2, 3, 4]);
    check_all(reshape(&[2, 3, 4]), &[3, 8]);
}

#[test]
fn reshape_corner_cardinality_counterexample() {
    let subset = ArraySubset::new_with_ranges(&[0..2, 0..2]);
    let wrong = ArraySubset::new_with_ranges(&[0..4, 0..1]);
    assert_eq!(linear_set(&subset, &[2, 5]), BTreeSet::from([0, 1, 5, 6]));
    assert_eq!(linear_set(&wrong, &[5, 2]), BTreeSet::from([0, 2, 4, 6]));
    assert_eq!(subset.num_elements(), wrong.num_elements());
    assert!(
        reshape(&[5, 2])
            .encoded_subset(&shape(&[2, 5]), &subset)
            .unwrap()
            .is_none()
    );
}

#[test]
fn reshape_large_contiguous_subset_uses_runs() {
    let codec = reshape(&[1, 1_000_000_000]);
    let subset = ArraySubset::new_with_ranges(&[0..1_000_000, 0..1_000]);
    assert_eq!(
        codec
            .encoded_subset(&shape(&[1_000_000, 1_000]), &subset)
            .unwrap()
            .unwrap(),
        ArraySubset::new_with_shape(vec![1, 1_000_000_000])
    );
}

#[test]
fn packbits_subsets_are_unavailable() {
    let codec = NumcodecsPackBitsCodec::new()
        .with_context(
            data_type::bool(),
            Some(false.into()),
            &CodecSpecificOptions::default(),
        )
        .unwrap();
    for subset in subsets(&[2, 4]) {
        assert!(
            codec
                .encoded_subset(&shape(&[2, 4]), &subset)
                .unwrap()
                .is_none()
        );
    }
    for subset in [
        ArraySubset::new_empty(2),
        ArraySubset::new_with_shape(vec![3, 4]),
    ] {
        assert!(codec.encoded_subset(&shape(&[2, 4]), &subset).is_err());
    }
}
