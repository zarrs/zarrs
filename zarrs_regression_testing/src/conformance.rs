//! Known non-conformant data and unregistered names written by previous releases.
//!
//! The current `zarrs` reads some non-conformant data and unregistered names for backwards compatibility (marked `NON-CONFORMANT` and `UNREGISTERED` in its source).
//! Such data is reported even if the current `zarrs` reads it.

use std::path::Path;

use serde_json::Value;
use zarrs_regression_testing::schema::FLETCHER32_ODD_LENGTH;

use crate::native::for_each_codec;
use crate::releases::Release;

/// Codec names written by previous releases that are not registered in [zarr-extensions](https://github.com/zarr-developers/zarr-extensions).
const UNREGISTERED_CODEC_NAMES: &[&str] = &["bz2", "fletcher32", "gdeflate", "pcodec", "vlen_v2"];

/// Data type names written by previous releases that are not registered in [zarr-extensions](https://github.com/zarr-developers/zarr-extensions).
const UNREGISTERED_DATA_TYPE_NAMES: &[&str] = &["binary"];

/// Describe the non-conformances of the array `metadata` (and its chunks in `array_dir`, if any) written by `release`.
#[must_use]
pub(crate) fn non_conformances(
    release: Release,
    metadata: &Value,
    array_dir: Option<&Path>,
) -> Vec<String> {
    let mut non_conformances = array_dir
        .map(|array_dir| chunk_non_conformances(metadata, array_dir))
        .unwrap_or_default();
    let data_type = metadata["data_type"]
        .as_str()
        .or_else(|| metadata["data_type"]["name"].as_str());
    if let Some(data_type) = data_type
        && UNREGISTERED_DATA_TYPE_NAMES.contains(&data_type)
    {
        non_conformances.push(format!("unregistered data type name `{data_type}`"));
    }
    // `binary` is an old name of the `bytes` data type, reported above
    let data_type = data_type.map(|data_type| match data_type {
        "binary" => "bytes",
        data_type => data_type,
    });
    let mut codecs = metadata["codecs"].clone();
    for_each_codec(&mut codecs, &mut |codec| {
        let name = codec
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let mode = codec
            .get("configuration")
            .and_then(|configuration| configuration.get("mode"));
        if UNREGISTERED_CODEC_NAMES.contains(&name) {
            non_conformances.push(format!("unregistered codec name `{name}`"));
        }
        match (name, mode) {
            ("zfp", Some(Value::String(mode)))
                if !mode.contains('_') && mode != "reversible" && mode != "expert" =>
            {
                non_conformances.push(format!(
                    "`zfp` mode `{mode}` (registered as `{}`)",
                    zfp_mode(mode)
                ));
            }
            ("numcodecs.zfpy", Some(Value::String(mode))) => {
                non_conformances.push(format!(
                    "`numcodecs.zfpy` mode `{mode}` (an integer in numcodecs)"
                ));
            }
            // `vlen-bytes` is only compatible with `bytes`, and `vlen-utf8` with `string`
            ("vlen-bytes", _) if data_type.is_some_and(|data_type| data_type != "bytes") => {
                non_conformances.push(format!(
                    "`vlen-bytes` codec with the `{}` data type (only compatible with `bytes`)",
                    data_type.unwrap_or_default()
                ));
            }
            ("vlen-utf8", _) if data_type.is_some_and(|data_type| data_type != "string") => {
                non_conformances.push(format!(
                    "`vlen-utf8` codec with the `{}` data type (only compatible with `string`)",
                    data_type.unwrap_or_default()
                ));
            }
            // The crc32c codec computed CRC32 rather than CRC32C checksums prior to 0.11.5
            ("crc32c", _) if release.0 <= 10 => {
                non_conformances
                    .push("`crc32c` checksums are CRC32 rather than CRC32C".to_string());
            }
            _ => {}
        }
    });
    non_conformances.sort();
    non_conformances.dedup();
    non_conformances
}

/// Describe the non-conformances of the chunks in `array_dir` of an array with `metadata`.
///
/// The checksums of a `numcodecs.fletcher32` codec at the end of the codec chain are checked: zarrs 0.19 to 0.23 omitted the last byte of data with an odd length.
fn chunk_non_conformances(metadata: &Value, array_dir: &Path) -> Vec<String> {
    let last_codec = metadata["codecs"]
        .as_array()
        .and_then(|codecs| codecs.last())
        .and_then(|codec| codec.get("name").or(Some(codec)))
        .and_then(Value::as_str);
    if !matches!(last_codec, Some("numcodecs.fletcher32" | "fletcher32")) {
        return vec![];
    }
    let mut non_conformances = vec![];
    for chunk in chunk_files(array_dir) {
        let Ok(bytes) = std::fs::read(&chunk) else {
            continue;
        };
        if let Some((data, checksum)) = bytes.split_last_chunk::<4>()
            && fletcher32(data).to_le_bytes() != *checksum
        {
            non_conformances.push(FLETCHER32_ODD_LENGTH.to_string());
        }
    }
    non_conformances
}

/// The chunk files in `array_dir` (every file except the metadata).
fn chunk_files(array_dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = vec![];
    let mut dirs = vec![array_dir.to_path_buf()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.file_name().is_some_and(|name| name != "zarr.json") {
                files.push(path);
            }
        }
    }
    files
}

/// The HDF5 Fletcher32 checksum of `data`, as in `numcodecs`.
fn fletcher32(data: &[u8]) -> u32 {
    let (mut sum1, mut sum2) = (0u32, 0u32);
    for block in data.chunks(720) {
        for word in block.chunks(2) {
            sum1 += u32::from(word[0]) << 8 | u32::from(word.get(1).copied().unwrap_or(0));
            sum2 += sum1;
        }
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    (sum2 << 16) | sum1
}

/// The registered name of a `zfp` mode without underscores (e.g. `fixedrate` is `fixed_rate`).
fn zfp_mode(mode: &str) -> String {
    mode.strip_prefix("fixed")
        .map_or_else(|| mode.to_string(), |mode| format!("fixed_{mode}"))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn non_conformances_detected() {
        let metadata = json!({
            "data_type": "binary",
            "codecs": [{"name": "sharding_indexed", "configuration": {
                "codecs": [{"name": "https://codec.zarrs.dev/array_to_bytes/vlen"}, {"name": "gdeflate"}],
                "index_codecs": [{"name": "bytes"}, {"name": "crc32c"}],
            }}],
        });
        assert_eq!(
            non_conformances(Release(18), &metadata, None),
            [
                "unregistered codec name `gdeflate`",
                "unregistered data type name `binary`"
            ]
        );
        assert_eq!(
            non_conformances(
                Release(10),
                &json!({"data_type": "uint8", "codecs": [
                    {"name": "zfp", "configuration": {"mode": "fixedrate", "rate": 8.0}},
                    {"name": "crc32c"},
                ]}),
                None
            ),
            [
                "`crc32c` checksums are CRC32 rather than CRC32C",
                "`zfp` mode `fixedrate` (registered as `fixed_rate`)",
            ]
        );
        assert_eq!(
            non_conformances(
                Release(23),
                &json!({"data_type": "string", "codecs": [{"name": "vlen-bytes"}]}),
                None
            ),
            ["`vlen-bytes` codec with the `string` data type (only compatible with `bytes`)"]
        );
        assert_eq!(
            non_conformances(
                Release(23),
                &json!({"data_type": "bytes", "codecs": [{"name": "vlen-utf8"}]}),
                None
            ),
            ["`vlen-utf8` codec with the `bytes` data type (only compatible with `string`)"]
        );
        assert_eq!(
            non_conformances(
                Release(18),
                &json!({"data_type": "binary", "codecs": [{"name": "vlen-bytes"}]}),
                None
            ),
            ["unregistered data type name `binary`"]
        );
        assert_eq!(
            non_conformances(
                Release(23),
                &json!({"data_type": {"name": "string"}, "codecs": [{"name": "vlen-bytes"}]}),
                None
            ),
            ["`vlen-bytes` codec with the `string` data type (only compatible with `bytes`)"]
        );
        for (data_type, codec) in [("bytes", "vlen-bytes"), ("string", "vlen-utf8")] {
            assert_eq!(
                non_conformances(
                    Release(23),
                    &json!({"data_type": data_type, "codecs": [{"name": codec}]}),
                    None
                ),
                Vec::<String>::new()
            );
        }
        let conformant = json!({"data_type": "uint8", "codecs": [
            {"name": "zfp", "configuration": {"mode": "fixed_rate", "rate": 8.0}},
            {"name": "numcodecs.zfpy", "configuration": {"mode": 2, "rate": 8.0}},
            {"name": "https://codec.zarrs.dev/array_to_bytes/zfp", "configuration": {"mode": "fixedrate", "rate": 8.0}},
            {"name": "crc32c"},
        ]});
        assert_eq!(
            non_conformances(Release(11), &conformant, None),
            Vec::<String>::new()
        );
    }

    #[test]
    fn fletcher32_checksums() {
        // Checksums of `numcodecs` 0.17.0
        assert_eq!(fletcher32(b"a").to_le_bytes(), [0, 97, 0, 97]);
        assert_eq!(fletcher32(b"abc").to_le_bytes(), [98, 196, 197, 37]);
        assert_eq!(fletcher32(b"abcd").to_le_bytes(), [198, 196, 41, 38]);
        let data: Vec<u8> = (0..255).collect();
        assert_eq!(fletcher32(&data).to_le_bytes(), [64, 191, 118, 84]);

        let dir = std::env::temp_dir().join(format!(
            "zarrs_regression_testing_fletcher32_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(dir.join("c/0")).unwrap();
        let metadata = json!({"codecs": [{"name": "bytes"}, {"name": "numcodecs.fletcher32"}]});
        std::fs::write(
            dir.join("c/0/0"),
            [b"abc".as_slice(), &[98, 196, 197, 37]].concat(),
        )
        .unwrap();
        assert_eq!(
            non_conformances(Release(23), &metadata, Some(&dir)),
            Vec::<String>::new()
        );
        // Without the last byte of the odd-length data
        let legacy = fletcher32(b"ab").to_le_bytes();
        std::fs::write(dir.join("c/0/0"), [b"abc".as_slice(), &legacy].concat()).unwrap();
        assert_eq!(
            non_conformances(Release(23), &metadata, Some(&dir)),
            ["`numcodecs.fletcher32` checksums omit the last byte of data with an odd length"]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
