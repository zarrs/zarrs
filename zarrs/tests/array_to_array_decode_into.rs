#![allow(missing_docs)]

//! Decoding through array-to-array codecs into targets of various layouts matches decoding and then copying.

use std::num::NonZeroU64;

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::codec::CodecChain;
use zarrs::array::{DataType, FillValue, data_type};
use zarrs_codec::{
    ArrayBytes, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArrayToBytesCodecTraits,
    CodecOptions, CodecSpecificOptions, ComputeResources, decode_into_array_bytes_target,
};
use zarrs_metadata::v3::MetadataV3;

const SENTINEL: u8 = 0xa5;

/// A layout of a target in an output buffer: the number of elements in the buffer, and the offset, shape and strides of the target.
#[derive(Debug, Clone)]
struct Layout {
    name: &'static str,
    len: u64,
    offset: u64,
    shape: Vec<u64>,
    strides: Vec<u64>,
}

fn c_strides(shape: &[u64]) -> Vec<u64> {
    let mut strides = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
}

fn f_strides(shape: &[u64]) -> Vec<u64> {
    (0..shape.len())
        .map(|d| shape[..d].iter().product())
        .collect()
}

/// Target layouts for a decoded shape: contiguous, padded, column-major, flattened, and strided.
fn layouts(shape: &[u64]) -> Vec<Layout> {
    let num_elements: u64 = shape.iter().product();
    let padded: Vec<u64> = shape.iter().map(|size| size + 2).collect();
    let offset = |strides: &[u64]| strides.iter().sum();
    vec![
        Layout {
            name: "contiguous",
            len: num_elements,
            offset: 0,
            shape: shape.to_vec(),
            strides: c_strides(shape),
        },
        Layout {
            name: "padded",
            len: padded.iter().product(),
            offset: offset(&c_strides(&padded)),
            shape: shape.to_vec(),
            strides: c_strides(&padded),
        },
        Layout {
            name: "f_order",
            len: num_elements,
            offset: 0,
            shape: shape.to_vec(),
            strides: f_strides(shape),
        },
        Layout {
            name: "f_order_padded",
            len: padded.iter().product(),
            offset: offset(&f_strides(&padded)),
            shape: shape.to_vec(),
            strides: f_strides(&padded),
        },
        Layout {
            name: "flat",
            len: num_elements + 4,
            offset: 2,
            shape: vec![num_elements],
            strides: vec![1],
        },
        Layout {
            name: "flat_strided",
            len: 2 * num_elements + 1,
            offset: 1,
            shape: vec![num_elements],
            strides: vec![2],
        },
    ]
}

/// Output buffers (data and optional masks) for a target with a layout.
struct Buffers {
    data: Vec<u8>,
    masks: Vec<Vec<u8>>,
    size: usize,
}

impl Buffers {
    fn new(layout: &Layout, size: usize, depth: usize) -> Self {
        let len = usize::try_from(layout.len).unwrap();
        Self {
            data: vec![SENTINEL; len * size],
            masks: vec![vec![SENTINEL; len]; depth],
            size,
        }
    }

    fn target(&mut self, layout: &Layout) -> ArrayBytesDecodeIntoTarget<'_> {
        // SAFETY: each view is the only view of its buffer.
        let view = |bytes, size| unsafe {
            ArrayBytesFixedDisjointView::new_strided(
                UnsafeCellSlice::new(bytes),
                size,
                layout.offset,
                &layout.shape,
                &layout.strides,
            )
            .unwrap()
        };
        let masks: Vec<_> = self.masks.iter_mut().map(|mask| view(mask, 1)).collect();
        ArrayBytesDecodeIntoTarget::from(view(&mut self.data, self.size))
            .with_optional_masks(masks)
            .unwrap()
    }

    fn is_untouched(&self) -> bool {
        self.data
            .iter()
            .chain(self.masks.iter().flatten())
            .all(|&byte| byte == SENTINEL)
    }
}

struct Case {
    name: &'static str,
    codecs: &'static str,
    data_type: DataType,
    fill_value: FillValue,
    shape: Vec<u64>,
    elements: ArrayBytes<'static>,
}

fn float32_elements(shape: &[u64]) -> ArrayBytes<'static> {
    let num_elements: u64 = shape.iter().product();
    let values: Vec<u8> = (0..num_elements)
        .flat_map(|i| (i as f32 * 0.5 - 3.0).to_ne_bytes())
        .collect();
    ArrayBytes::new_flen(values)
}

fn float32_case(name: &'static str, codecs: &'static str, shape: &[u64]) -> Case {
    Case {
        name,
        codecs,
        data_type: data_type::float32(),
        fill_value: FillValue::from(0f32),
        shape: shape.to_vec(),
        elements: float32_elements(shape),
    }
}

fn optional_case(name: &'static str, codecs: &'static str, shape: &[u64]) -> Case {
    let num_elements: u64 = shape.iter().product();
    let data: Vec<u8> = (0..num_elements)
        .flat_map(|i| u16::try_from(i * 7).unwrap().to_ne_bytes())
        .collect();
    let mask: Vec<u8> = (0..num_elements).map(|i| u8::from(i % 3 != 0)).collect();
    Case {
        name,
        codecs,
        data_type: data_type::uint16().to_optional(),
        fill_value: FillValue::new_optional_null(),
        shape: shape.to_vec(),
        elements: ArrayBytes::new_flen(data).with_optional_mask(mask),
    }
}

const BYTES: &str = r#"{"name": "bytes", "configuration": {"endian": "little"}}"#;
const OPTIONAL: &str = r#"{"name": "zarrs.optional", "configuration": {"mask_codecs": [{"name": "packbits"}], "data_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}]}}"#;

fn cases() -> Vec<Case> {
    let mut cases = vec![
        float32_case(
            "reshape_merge",
            r#"[{"name": "reshape", "configuration": {"shape": [[0, 1]]}}, BYTES]"#,
            &[4, 6],
        ),
        float32_case(
            "reshape_split",
            r#"[{"name": "reshape", "configuration": {"shape": [[0], 2, -1]}}, BYTES]"#,
            &[4, 6],
        ),
        float32_case(
            "squeeze",
            r#"[{"name": "zarrs.squeeze"}, BYTES]"#,
            &[1, 4, 1, 6, 1],
        ),
        float32_case(
            "squeeze_trailing",
            r#"[{"name": "zarrs.squeeze"}, BYTES]"#,
            &[4, 6, 1],
        ),
        float32_case(
            "squeeze_bytes_non_native",
            r#"[{"name": "zarrs.squeeze"}, {"name": "bytes", "configuration": {"endian": "big"}}]"#,
            &[3, 1, 5],
        ),
        Case {
            name: "fixedscaleoffset",
            codecs: r#"[{"name": "numcodecs.fixedscaleoffset", "configuration": {"offset": 1000, "scale": 10, "dtype": "f8", "astype": "u1"}}, BYTES]"#,
            data_type: data_type::float64(),
            fill_value: FillValue::from(1000f64),
            shape: vec![4, 6],
            elements: ArrayBytes::new_flen(
                (0..24u32)
                    .flat_map(|i| (1000.0 + f64::from(i) * 0.7).to_ne_bytes())
                    .collect::<Vec<u8>>(),
            ),
        },
        Case {
            name: "numcodecs_packbits",
            codecs: r#"[{"name": "numcodecs.packbits"}, BYTES]"#,
            data_type: data_type::bool(),
            fill_value: FillValue::from(false),
            shape: vec![3, 5],
            elements: ArrayBytes::new_flen(
                (0..15u8).map(|i| u8::from(i % 3 == 0)).collect::<Vec<u8>>(),
            ),
        },
        Case {
            name: "cast_value",
            codecs: r#"[{"name": "cast_value", "configuration": {"data_type": "int16"}}, BYTES]"#,
            data_type: data_type::int32(),
            fill_value: FillValue::from(0i32),
            shape: vec![4, 6],
            elements: ArrayBytes::new_flen(
                (0..24i32)
                    .flat_map(|i| (i * 1000 - 12000).to_ne_bytes())
                    .collect::<Vec<u8>>(),
            ),
        },
        // More elements than are cast at a time
        Case {
            name: "cast_value_blocks",
            codecs: r#"[{"name": "cast_value", "configuration": {"data_type": "float32"}}, BYTES]"#,
            data_type: data_type::float64(),
            fill_value: FillValue::from(0f64),
            shape: vec![90, 100],
            elements: ArrayBytes::new_flen(
                (0..9000u32)
                    .flat_map(|i| (f64::from(i) * 0.5).to_ne_bytes())
                    .collect::<Vec<u8>>(),
            ),
        },
        Case {
            name: "fixedscaleoffset_regions",
            codecs: r#"[{"name": "numcodecs.fixedscaleoffset", "configuration": {"offset": 0, "scale": 100, "dtype": "<f4", "astype": "<i2"}}, BYTES]"#,
            data_type: data_type::float32(),
            fill_value: FillValue::from(0f32),
            shape: vec![4, 1100],
            elements: ArrayBytes::new_flen(
                (0..4400u16)
                    .flat_map(|i| (f32::from(i % 300) * 0.25).to_ne_bytes())
                    .collect::<Vec<u8>>(),
            ),
        },
        Case {
            name: "numcodecs_packbits_regions",
            codecs: r#"[{"name": "numcodecs.packbits"}, BYTES]"#,
            data_type: data_type::bool(),
            fill_value: FillValue::from(false),
            shape: vec![3, 1100],
            elements: ArrayBytes::new_flen(
                (0..3300u32)
                    .map(|i| u8::from(i % 7 < 3))
                    .collect::<Vec<u8>>(),
            ),
        },
        optional_case(
            "optional_reshape",
            r#"[{"name": "reshape", "configuration": {"shape": [-1]}}, OPTIONAL]"#,
            &[4, 6],
        ),
        optional_case(
            "optional_squeeze",
            r#"[{"name": "zarrs.squeeze"}, OPTIONAL]"#,
            &[4, 1, 6],
        ),
    ];
    #[cfg(feature = "transpose")]
    cases.extend([
        float32_case(
            "transpose_reshape",
            r#"[{"name": "transpose", "configuration": {"order": [1, 0]}}, {"name": "reshape", "configuration": {"shape": [-1]}}, BYTES]"#,
            &[4, 6],
        ),
        float32_case(
            "transpose",
            r#"[{"name": "transpose", "configuration": {"order": [1, 0]}}, BYTES]"#,
            &[4, 6],
        ),
        float32_case(
            "transpose_3d",
            r#"[{"name": "transpose", "configuration": {"order": [2, 0, 1]}}, BYTES]"#,
            &[2, 3, 4],
        ),
        float32_case(
            "squeeze_transpose",
            r#"[{"name": "zarrs.squeeze"}, {"name": "transpose", "configuration": {"order": [1, 0]}}, BYTES]"#,
            &[4, 1, 6],
        ),
        optional_case(
            "optional_transpose",
            r#"[{"name": "transpose", "configuration": {"order": [1, 0]}}, OPTIONAL]"#,
            &[4, 6],
        ),
    ]);
    #[cfg(feature = "zstd")]
    cases.extend([
        float32_case(
            "reshape_zstd",
            r#"[{"name": "reshape", "configuration": {"shape": [-1]}}, BYTES, {"name": "zstd", "configuration": {"level": 1, "checksum": false}}]"#,
            &[4, 6],
        ),
        float32_case(
            "squeeze_zstd",
            r#"[{"name": "zarrs.squeeze"}, BYTES, {"name": "zstd", "configuration": {"level": 1, "checksum": false}}]"#,
            &[4, 1, 6],
        ),
    ]);
    #[cfg(all(feature = "transpose", feature = "zstd"))]
    cases.push(float32_case(
        "transpose_zstd",
        r#"[{"name": "transpose", "configuration": {"order": [1, 0]}}, BYTES, {"name": "zstd", "configuration": {"level": 1, "checksum": false}}]"#,
        &[4, 6],
    ));
    #[cfg(feature = "bitround")]
    cases.push(float32_case(
        "bitround",
        r#"[{"name": "bitround", "configuration": {"keepbits": 6}}, BYTES]"#,
        &[4, 6],
    ));
    cases
}

fn bind(case: &Case) -> std::sync::Arc<dyn ArrayToBytesCodecTraits> {
    let codecs = case
        .codecs
        .replace("BYTES", BYTES)
        .replace("OPTIONAL", OPTIONAL);
    let metadatas: Vec<MetadataV3> = serde_json::from_str(&codecs).unwrap();
    CodecChain::from_metadata(&metadatas)
        .unwrap()
        .with_context(
            case.data_type.clone(),
            Some(case.fill_value.clone()),
            &CodecSpecificOptions::default(),
        )
        .unwrap()
}

#[test]
fn array_to_array_decode_into_matches_decode() {
    let options = CodecOptions::default();
    let resources = ComputeResources::default();
    for case in cases() {
        let chain = bind(&case);
        let shape: Vec<NonZeroU64> = case
            .shape
            .iter()
            .map(|&size| NonZeroU64::new(size).unwrap())
            .collect();
        let encoded = chain
            .encode(case.elements.clone(), &shape, &options, &resources)
            .unwrap();
        let decoded = chain
            .decode(encoded.clone(), &shape, &options, &resources)
            .unwrap();
        let size = case.data_type.fixed_size().unwrap_or_else(|| {
            case.data_type
                .as_optional()
                .unwrap()
                .data_type()
                .fixed_size()
                .unwrap()
        });
        let depth = usize::from(case.data_type.is_optional());
        for layout in layouts(&case.shape) {
            let mut expected = Buffers::new(&layout, size, depth);
            decode_into_array_bytes_target(&decoded, &mut expected.target(&layout)).unwrap();
            let mut actual = Buffers::new(&layout, size, depth);
            chain
                .decode_into(
                    encoded.clone().into(),
                    &shape,
                    &mut actual.target(&layout),
                    &options,
                    &resources,
                )
                .unwrap();
            assert_eq!(actual.data, expected.data, "{} {}", case.name, layout.name);
            assert_eq!(
                actual.masks, expected.masks,
                "{} {}",
                case.name, layout.name
            );

            // An error before placement leaves the target untouched
            let mut untouched = Buffers::new(&layout, size, depth);
            let truncated = encoded.clone().slice(..encoded.len() - 1);
            assert!(
                chain
                    .decode_into(
                        truncated.into(),
                        &shape,
                        &mut untouched.target(&layout),
                        &options,
                        &resources,
                    )
                    .is_err(),
                "{} {}",
                case.name,
                layout.name
            );
            assert!(untouched.is_untouched(), "{} {}", case.name, layout.name);
        }
    }
}
