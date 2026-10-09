//! The `zarrs` releases that introduced each tested codec and data type.
//!
//! This is the first release that supported a codec or data type under any name or configuration (per the changelog, checked against the source at release tags).
//! A release may not support it as tested (e.g. with its current name or configuration), which the report marks.

use crate::cases::{CodecKind, VlenKind};

/// The release that introduced a codec, or [`None`] if not yet released.
pub(crate) fn codec(kind: CodecKind) -> Option<&'static str> {
    match kind {
        CodecKind::CastValue { .. } => None,
        CodecKind::Bytes
        | CodecKind::Transpose
        | CodecKind::Blosc { .. }
        | CodecKind::Crc32c
        | CodecKind::Gzip
        // The `checksum` option is accepted, but only written from 0.6.0
        | CodecKind::Zstd { .. }
        // `index_location` (and so `index=end`) from 0.6.0
        | CodecKind::Sharding {
            index_at_start: false,
        } => Some("0.2"),
        CodecKind::Sharding {
            index_at_start: true,
        }
        | CodecKind::BitRound
        // Modes named `fixedrate` etc. until 0.16.0
        | CodecKind::Zfp(_) => Some("0.6"),
        // Named `pcodec` and `bz2` (rather than `numcodecs.*`) until 0.20.0
        CodecKind::Pcodec | CodecKind::Bz2 => Some("0.11.2"),
        // Named `vlen` and `vlen_v2` (rather than `zarrs.*`) until 0.20.0, and `vlen` has `index_location` from 0.22.0
        CodecKind::Vlen(VlenKind::Vlen | VlenKind::V2) => Some("0.16"),
        // Named `gdeflate` (rather than `zarrs.gdeflate`) until 0.20.0
        CodecKind::Gdeflate => Some("0.16.2"),
        CodecKind::Vlen(VlenKind::Array | VlenKind::Bytes | VlenKind::Utf8) => Some("0.18"),
        // Named `fletcher32` (rather than `numcodecs.fletcher32`) until 0.20.0
        CodecKind::Fletcher32 => Some("0.19"),
        CodecKind::FixedScaleOffset { .. }
        // `zarrs.squeeze` is read from `zarrs_registry` 0.1.5
        | CodecKind::Squeeze
        | CodecKind::PackBits
        | CodecKind::Shuffle
        | CodecKind::Zlib
        // As a Zarr V3 codec (Zarr V2 only from 0.16.0)
        | CodecKind::Zfpy(_) => Some("0.20"),
        CodecKind::Reshape | CodecKind::Adler32 => Some("0.22"),
        CodecKind::Optional => Some("0.23"),
    }
}

/// Why releases since a codec was introduced do not support it as tested (if known).
pub(crate) fn codec_note(kind: CodecKind) -> Option<&'static str> {
    match kind {
        CodecKind::Squeeze => Some(
            "0.20 and 0.21 write `zarrs.squeeze`, but read it from zarrs_registry 0.1.5 (published after them)",
        ),
        CodecKind::Zfpy(_) => Some("0.20 to 0.22 write string modes, which they cannot read"),
        _ => None,
    }
}

/// The release that introduced a data type (by its label in [`crate::cases::data_types`]).
///
/// # Panics
/// Panics if the data type is unknown.
pub(crate) fn data_type(label: &str) -> &'static str {
    match label {
        // Signed integer and complex fill values of `0` are read from 0.5.1
        "bool" | "int8" | "int16" | "int32" | "int64" | "uint8" | "uint16" | "uint32"
        | "uint64" | "float16" | "bfloat16" | "float32" | "float32(fill=NaN)" | "float64"
        | "float64(fill=NaN)" | "complex64" | "complex128" | "r24" => "0.2",
        // `bytes` was named `binary` until 0.19.0
        "string" | "bytes" => "0.16",
        "int2" | "int4" | "uint2" | "uint4" | "float4_e2m1fn" | "float6_e2m3fn"
        | "float6_e3m2fn" | "float8_e3m4" | "float8_e4m3" | "float8_e4m3b11fnuz"
        | "float8_e4m3fnuz" | "float8_e5m2" | "float8_e5m2fnuz" | "float8_e8m0fnu"
        | "complex_bfloat16" | "complex_float16" | "complex_float32" | "complex_float64" => "0.21",
        "numpy.datetime64" | "numpy.timedelta64" => "0.21.1",
        "complex_float4_e2m1fn"
        | "complex_float6_e2m3fn"
        | "complex_float6_e3m2fn"
        | "complex_float8_e3m4"
        | "complex_float8_e4m3"
        | "complex_float8_e4m3b11fnuz"
        | "complex_float8_e4m3fnuz"
        | "complex_float8_e5m2"
        | "complex_float8_e5m2fnuz"
        | "complex_float8_e8m0fnu" => "0.21.2",
        "optional<uint8>"
        | "optional<float32>"
        | "optional<optional<float32>>"
        | "optional<string>" => "0.23",
        "fixed_length_utf32" => "0.23.12",
        _ => panic!("unknown data type {label}"),
    }
}

/// Why releases since a data type was introduced do not support it as tested (if known).
pub(crate) fn data_type_note(label: &str) -> Option<&'static str> {
    // Complex subfloats
    (label.starts_with("complex_float")
        && !matches!(
            label,
            "complex_float16" | "complex_float32" | "complex_float64"
        ))
    .then_some("0.21.2 and 0.22 cannot read its name in array metadata")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cases;

    #[test]
    fn all_data_types() {
        for case in cases::data_types() {
            data_type(case.label);
        }
    }
}
