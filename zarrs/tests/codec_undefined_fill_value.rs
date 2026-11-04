#![allow(missing_docs)]

use std::num::NonZeroU64;
use zarrs::array::{ArrayBytes, ArraySubset, CodecChain, data_type};
use zarrs_codec::plan::{partial_decode_memory, partial_encode_memory};
use zarrs_codec::{CodecError, CodecOptions, CodecSpecificOptions, ComputeResources};

#[test]
fn planned_missing_input_requires_a_defined_fill_value() {
    let shape = [NonZeroU64::new(4).unwrap()];
    let subset = ArraySubset::new_with_ranges(&[1..3]);
    let whole = ArraySubset::new_with_ranges(&[0..4]);
    let options = CodecOptions::default();
    let resources = ComputeResources::default();
    for name in ["bytes", "packbits"] {
        let metadata = serde_json::from_str(&format!(r#"{{"name":"{name}"}}"#)).unwrap();
        let codec = CodecChain::from_metadata(&[metadata])
            .unwrap()
            .with_context(data_type::uint8(), None, &CodecSpecificOptions::default())
            .unwrap();
        assert!(matches!(
            partial_decode_memory(codec.clone(), None, &shape, &subset, &options, &resources),
            Err(CodecError::UndefinedFillValue)
        ));
        let mut encoded = None;
        assert!(matches!(
            partial_encode_memory(
                codec.clone(),
                &mut encoded,
                &shape,
                &subset,
                &ArrayBytes::new_flen(vec![1, 2]),
                &options,
                &resources
            ),
            Err(CodecError::UndefinedFillValue)
        ));
        assert!(encoded.is_none());
        let data = ArrayBytes::new_flen(vec![1, 2, 3, 4]);
        partial_encode_memory(
            codec.clone(),
            &mut encoded,
            &shape,
            &whole,
            &data,
            &options,
            &resources,
        )
        .unwrap();
        let decoded = partial_decode_memory(
            codec,
            encoded.map(Into::into),
            &shape,
            &whole,
            &options,
            &resources,
        )
        .unwrap();
        assert_eq!(decoded, data);
    }
}
