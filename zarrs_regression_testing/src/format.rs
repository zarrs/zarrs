//! Formatting array data for people.

use std::fmt::Write;

use serde_json::Value;
use zarrs::array::FillValue;

use crate::cases::{DataTypeCase, Values};
use crate::data::Data;

/// The bytes of each element of `data`, with elements of `element_size` bytes if fixed length.
pub(crate) fn element_bytes(data: &Data, element_size: Option<usize>) -> Vec<&[u8]> {
    if let Some(offsets) = &data.offsets {
        offsets
            .windows(2)
            .map(|window| &data.bytes[window[0]..window[1]])
            .collect()
    } else if let Some(size) = element_size.filter(|&size| size > 0) {
        data.bytes.chunks(size).collect()
    } else {
        vec![&data.bytes]
    }
}

/// Format the elements of `data` as in fill value metadata (see [`format_element`]), with nulls as `null` (wrapped as in fill value metadata if nested).
pub(crate) fn elements(data: &Data, data_type: &DataTypeCase) -> Vec<String> {
    element_bytes(data, data_type.values.element_size())
        .iter()
        .enumerate()
        .map(|(index, bytes)| {
            // A null at depth `n` (outermost first) is wrapped `n` times, as in fill value metadata
            match data
                .masks
                .iter()
                .position(|mask| mask.get(index) == Some(&0))
            {
                Some(depth) => format!("{}null{}", "[".repeat(depth), "]".repeat(depth)),
                None => format_element(data_type, bytes),
            }
        })
        .collect()
}

/// Format the bytes of a (non-null) element as in fill value metadata (e.g. `-1`, `1.5`, `NaN`, `"text"`), with complex numbers as `1.5-2j`.
///
/// Raw bits and bytes, and elements that cannot be formatted, are formatted in hex.
pub(crate) fn format_element(data_type: &DataTypeCase, bytes: &[u8]) -> String {
    let mut inner = &data_type.data_type;
    while let Some(optional_inner) = inner.optional_inner() {
        inner = optional_inner;
    }
    let mut values = &data_type.values;
    while let Values::Optional(optional_values) = values {
        values = optional_values;
    }
    if matches!(values, Values::Raw(_) | Values::Bytes) {
        return hex(bytes);
    }
    let Some(value) = inner
        .metadata_fill_value(&FillValue::new(bytes.to_vec()))
        .ok()
        .and_then(|metadata| serde_json::to_value(metadata).ok())
    else {
        return hex(bytes);
    };
    // Strings are quoted with non-printable characters (e.g. bidirectional overrides) escaped, but not special values (e.g. `NaN`, `NaT`)
    let quote = matches!(values, Values::String | Values::Utf32(_));
    let scalar = |value: &Value| match value {
        Value::String(string) if quote => format!("{string:?}"),
        Value::String(string) => string.clone(),
        value => value.to_string(),
    };
    match value.as_array().map(Vec::as_slice) {
        Some([re, im]) => {
            let (re, im) = (scalar(re), scalar(im));
            match im.strip_prefix('-') {
                Some(im) => format!("{re}-{im}j"),
                None => format!("{re}+{im}j"),
            }
        }
        _ => scalar(&value),
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        return "∅".to_string();
    }
    bytes.iter().fold(String::new(), |mut hex, byte| {
        let _ = write!(hex, "{byte:02x}");
        hex
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases;

    #[test]
    fn format_elements() {
        let data_types = cases::data_types();
        let data_type = |label: &str| {
            data_types
                .iter()
                .find(|data_type| data_type.label == label)
                .unwrap()
        };
        let elements = |label: &str, elements: &[&[u8]], masks: Vec<Vec<u8>>| {
            let variable = data_type(label).values.element_size().is_none();
            let elements: Vec<Vec<u8>> = elements.iter().map(|element| element.to_vec()).collect();
            let data = Data::from_elements(&elements, variable, masks);
            super::elements(&data, data_type(label)).join(" ")
        };
        let int16 = |value: i16| value.to_ne_bytes();
        assert_eq!(
            elements("int16", &[&int16(-2), &int16(300)], vec![]),
            "-2 300"
        );
        assert_eq!(elements("int4", &[&[0xf9]], vec![]), "-7");
        assert_eq!(elements("bool", &[&[0], &[1]], vec![]), "false true");
        let float32 = |value: f32| value.to_ne_bytes();
        assert_eq!(
            elements("float32", &[&float32(1.5), &float32(f32::NAN)], vec![]),
            "1.5 NaN"
        );
        assert_eq!(
            elements("float16", &[&[0x00, 0x3c], &[0x00, 0xfc]], vec![]),
            "1.0 -Infinity"
        );
        assert_eq!(elements("float8_e4m3", &[&[0x38]], vec![]), "1.0");
        assert_eq!(
            elements(
                "complex64",
                &[&[float32(1.5), float32(-2.0)].concat()],
                vec![]
            ),
            "1.5-2.0j"
        );
        assert_eq!(
            elements(
                "numpy.datetime64",
                &[&i64::MIN.to_ne_bytes(), &7_i64.to_ne_bytes()],
                vec![]
            ),
            "NaT 7"
        );
        assert_eq!(elements("r24", &[&[1, 2, 255]], vec![]), "0102ff");
        assert_eq!(elements("bytes", &[b"", b"ab"], vec![]), "∅ 6162");
        assert_eq!(
            elements("string", &[b"a\"b", b""], vec![]),
            "\"a\\\"b\" \"\""
        );
        assert_eq!(
            elements("string", &["a\u{202e}".as_bytes()], vec![]),
            "\"a\\u{202e}\""
        );
        let utf32: Vec<u8> = "ab\0"
            .chars()
            .flat_map(|char| u32::from(char).to_ne_bytes())
            .collect();
        assert_eq!(elements("fixed_length_utf32", &[&utf32], vec![]), "\"ab\"");
        assert_eq!(
            elements(
                "optional<optional<float32>>",
                &[&float32(0.0), &float32(0.0), &float32(2.5)],
                vec![vec![0, 1, 1], vec![1, 0, 1]],
            ),
            "null [null] 2.5"
        );
    }
}
