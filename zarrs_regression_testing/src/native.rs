//! Array metadata in the form supported by previous releases.
//!
//! Cases are written by releases with the array metadata of the current `zarrs`.
//! Where a release supports a codec or data type with another name or configuration, the metadata is converted to the form it supports.
//! Otherwise, the release would be considered to not support the codec or data type, and data it writes natively would never be read by the current `zarrs`.

use serde_json::{Map, Value};
use zarrs::array::{DataType, FillValueMetadata};
use zarrs::metadata::v3::MetadataV3;

use crate::releases::Release;

/// Data types that only support hexadecimal fill value metadata (e.g. `"0x00"`) in zarrs 0.21 and 0.22.
const HEX_FILL_VALUE_DATA_TYPES: &[&str] = &[
    "float4_e2m1fn",
    "float6_e2m3fn",
    "float6_e3m2fn",
    "float8_e3m4",
    "float8_e4m3b11fnuz",
    "float8_e4m3fnuz",
    "float8_e5m2fnuz",
    "float8_e8m0fnu",
];

/// Convert array `metadata` of the current `zarrs` to the form supported by `release`.
#[must_use]
pub(crate) fn native_metadata(release: Release, metadata: &Value) -> Value {
    let minor = release.0;
    let mut metadata = metadata.clone();
    if let Some(codecs) = metadata.get_mut("codecs") {
        for_each_codec(codecs, &mut |codec| native_codec(minor, codec));
    }
    let data_type = metadata["data_type"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    if (16..=18).contains(&minor) && data_type == "bytes" {
        metadata["data_type"] = Value::from("binary");
    }
    if (21..=22).contains(&minor)
        && HEX_FILL_VALUE_DATA_TYPES.contains(&data_type.as_str())
        && let Some([byte]) = fill_value(&metadata).as_deref()
    {
        metadata["fill_value"] = Value::from(format!("0x{byte:02x}"));
    }
    metadata
}

/// The fill value (native-endian bytes) of array `metadata`, if valid.
fn fill_value(metadata: &Value) -> Option<Vec<u8>> {
    let data_type = serde_json::from_value::<MetadataV3>(metadata["data_type"].clone()).ok()?;
    let data_type = DataType::from_metadata(&data_type).ok()?;
    let fill_value =
        serde_json::from_value::<FillValueMetadata>(metadata["fill_value"].clone()).ok()?;
    let fill_value = data_type.fill_value_v3(&fill_value).ok()?;
    Some(fill_value.as_ne_bytes().to_vec())
}

/// Apply `f` to each codec in `codecs`, including the inner codecs of codecs such as `sharding_indexed` and `vlen`.
pub(crate) fn for_each_codec(codecs: &mut Value, f: &mut impl FnMut(&mut Map<String, Value>)) {
    let Some(codecs) = codecs.as_array_mut() else {
        return;
    };
    for codec in codecs {
        let Some(codec) = codec.as_object_mut() else {
            continue;
        };
        f(codec);
        if let Some(Value::Object(configuration)) = codec.get_mut("configuration") {
            for key in ["codecs", "index_codecs", "data_codecs"] {
                if let Some(inner) = configuration.get_mut(key) {
                    for_each_codec(inner, f);
                }
            }
        }
    }
}

/// Convert codec metadata of the current `zarrs` to the form supported by zarrs `0.{minor}`.
fn native_codec(minor: u32, codec: &mut Map<String, Value>) {
    let Some(Value::String(name)) = codec.get_mut("name") else {
        return;
    };
    // Codecs were identified without a `numcodecs.` or `zarrs.` prefix prior to 0.20
    if minor <= 19
        && let Some(identifier) = name
            .strip_prefix("numcodecs.")
            .or_else(|| name.strip_prefix("zarrs."))
    {
        *name = identifier.to_string();
    }
    let name = name.clone();
    let Some(Value::Object(configuration)) = codec.get_mut("configuration") else {
        return;
    };
    match name.as_str() {
        // The index was always at the start prior to 0.22
        "vlen" | "zarrs.vlen"
            if minor <= 21 && configuration.get("index_location") == Some(&"start".into()) =>
        {
            configuration.remove("index_location");
        }
        // zfp modes were not snake case prior to 0.16 (e.g. `fixedrate`)
        "zfp" if minor <= 15 => {
            if let Some(Value::String(mode)) = configuration.get_mut("mode") {
                *mode = mode.replace('_', "");
            }
        }
        // pcodec had no delta or paging spec prior to 0.19, with a maximum page size prior to 0.16 and mult specs rather than a mode spec prior to 0.15
        "pcodec" if minor <= 18 => {
            let delta_encoding_order = match configuration.get("delta_spec").and_then(Value::as_str)
            {
                Some("auto") => Value::Null,
                Some("none") => 0.into(),
                Some("try_consecutive") => configuration["delta_encoding_order"].clone(),
                _ => return,
            };
            let (Some(level), Some(mode_spec), Some(page_size)) = (
                configuration.get("level").cloned(),
                configuration.get("mode_spec").cloned(),
                configuration.get("equal_pages_up_to").cloned(),
            ) else {
                return;
            };
            let mut native = Map::from_iter([
                ("level".to_string(), level),
                ("delta_encoding_order".to_string(), delta_encoding_order),
            ]);
            if minor <= 14 {
                let multiply = match mode_spec.as_str() {
                    Some("auto") => true,
                    Some("classic") => false,
                    _ => return,
                };
                native.insert("int_mult_spec".to_string(), multiply.into());
                native.insert("float_mult_spec".to_string(), multiply.into());
            } else {
                native.insert("mode_spec".to_string(), mode_spec);
            }
            let page_size_key = if minor <= 15 {
                "max_page_n"
            } else {
                "equal_pages_up_to"
            };
            native.insert(page_size_key.to_string(), page_size);
            *configuration = native;
        }
        // zfpy modes were integers (as in numcodecs) in 0.20 to 0.22
        "numcodecs.zfpy" if (20..=22).contains(&minor) => {
            let mode = match configuration.get("mode").and_then(Value::as_str) {
                Some("fixed_rate") => 2,
                Some("fixed_precision") => 3,
                Some("fixed_accuracy") => 4,
                Some("reversible") => 5,
                _ => return,
            };
            configuration.insert("mode".to_string(), mode.into());
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn native_codecs() {
        let metadata = json!({
            "data_type": "bytes",
            "fill_value": [],
            "codecs": [{"name": "sharding_indexed", "configuration": {
                "codecs": [
                    {"name": "zarrs.vlen", "configuration": {"index_codecs": [], "data_codecs": [{"name": "numcodecs.bz2"}], "index_data_type": "uint64", "index_location": "start"}},
                ],
                "index_codecs": [{"name": "bytes"}],
            }}],
        });
        assert_eq!(
            native_metadata(Release(18), &metadata),
            json!({
                "data_type": "binary",
                "fill_value": [],
                "codecs": [{"name": "sharding_indexed", "configuration": {
                    "codecs": [
                        {"name": "vlen", "configuration": {"index_codecs": [], "data_codecs": [{"name": "bz2"}], "index_data_type": "uint64"}},
                    ],
                    "index_codecs": [{"name": "bytes"}],
                }}],
            })
        );
        assert_eq!(native_metadata(Release(23), &metadata), metadata);
    }

    #[test]
    fn native_pcodec() {
        let metadata = json!({"codecs": [{"name": "numcodecs.pcodec", "configuration": {
            "level": 8, "mode_spec": "auto", "delta_spec": "auto", "paging_spec": "equal_pages_up_to", "equal_pages_up_to": 262_144
        }}]});
        assert_eq!(
            native_metadata(Release(14), &metadata),
            json!({"codecs": [{"name": "pcodec", "configuration": {
                "level": 8, "delta_encoding_order": null, "int_mult_spec": true, "float_mult_spec": true, "max_page_n": 262_144
            }}]})
        );
        assert_eq!(
            native_metadata(Release(15), &metadata),
            json!({"codecs": [{"name": "pcodec", "configuration": {
                "level": 8, "delta_encoding_order": null, "mode_spec": "auto", "max_page_n": 262_144
            }}]})
        );
        assert_eq!(
            native_metadata(Release(18), &metadata),
            json!({"codecs": [{"name": "pcodec", "configuration": {
                "level": 8, "delta_encoding_order": null, "mode_spec": "auto", "equal_pages_up_to": 262_144
            }}]})
        );
        assert_eq!(
            native_metadata(Release(19), &metadata)["codecs"][0]["configuration"],
            metadata["codecs"][0]["configuration"]
        );
    }

    #[test]
    fn native_zfp_and_fill_values() {
        let zfpy = json!({"data_type": "float8_e3m4", "fill_value": 0.0, "codecs": [{"name": "numcodecs.zfpy", "configuration": {"mode": "fixed_rate", "rate": 4.0}}]});
        assert_eq!(
            native_metadata(Release(21), &zfpy),
            json!({"data_type": "float8_e3m4", "fill_value": "0x00", "codecs": [{"name": "numcodecs.zfpy", "configuration": {"mode": 2, "rate": 4.0}}]})
        );
        let zfp = json!({"codecs": [{"name": "zfp", "configuration": {"mode": "fixed_accuracy", "tolerance": 0.5}}]});
        assert_eq!(
            native_metadata(Release(15), &zfp),
            json!({"codecs": [{"name": "zfp", "configuration": {"mode": "fixedaccuracy", "tolerance": 0.5}}]})
        );
    }
}
