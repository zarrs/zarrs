//! Test case generation.
//!
//! Categorical properties (codec kind and mode, data type) are fully swept as *combinations*.
//! Everything else (codec levels and parameters, shapes, data) is sampled with proptest.

use std::num::{NonZeroU32, NonZeroU64};
use std::sync::Arc;

use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
use serde_json::{Value, json};
use zarrs::array::codec::{VlenArrayCodec, VlenBytesCodec, VlenCodec, VlenUtf8Codec, VlenV2Codec};
use zarrs::array::{
    ArrayBuilder, CodecMetadataOptions, DataType, FillValue, UnboundArrayToBytesCodecTraits,
    data_type,
};
use zarrs::metadata_ext::data_type::NumpyTimeUnit;

use zarrs_regression_testing::schema::Numeric;

use crate::data::{Data, to_array_bytes};

/// The values of a data type, used for generating data.
#[derive(Debug, Clone)]
pub(crate) enum Values {
    Bool,
    /// An unsigned integer with this many bits.
    UInt(u32),
    /// A signed integer with this many bits.
    Int(u32),
    /// A 16-bit float (any bit pattern).
    Half,
    Float32,
    Float64,
    /// A byte with only the low bits set (e.g. sub-byte floats).
    LowBits(u32),
    /// Any byte (e.g. 8-bit floats).
    Byte,
    /// Raw bits with this many bytes.
    Raw(usize),
    Complex(Box<Values>),
    /// UTF-32 code points (any valid character) with this many code points.
    Utf32(usize),
    String,
    Bytes,
    Optional(Box<Values>),
}

impl Values {
    /// The kind of the elements if numeric (for comparing lossy data).
    pub(crate) fn numeric(&self) -> Option<Numeric> {
        match self {
            Self::UInt(8 | 16 | 32 | 64) => Some(Numeric::UInt),
            Self::Int(8 | 16 | 32 | 64) => Some(Numeric::Int),
            Self::Float32 | Self::Float64 => Some(Numeric::Float),
            _ => None,
        }
    }

    /// The size of an innermost element in bytes, or [`None`] if variable length.
    pub(crate) fn element_size(&self) -> Option<usize> {
        match self {
            Self::Bool | Self::LowBits(_) | Self::Byte => Some(1),
            Self::UInt(bits) | Self::Int(bits) => Some((*bits as usize).div_ceil(8)),
            Self::Half => Some(2),
            Self::Float32 => Some(4),
            Self::Float64 => Some(8),
            Self::Raw(size) => Some(*size),
            Self::Utf32(length) => Some(4 * length),
            Self::Complex(values) => values.element_size().map(|size| size * 2),
            Self::String | Self::Bytes => None,
            Self::Optional(values) => values.element_size(),
        }
    }

    fn optional_depth(&self) -> usize {
        match self {
            Self::Optional(values) => 1 + values.optional_depth(),
            _ => 0,
        }
    }

    fn element(&self) -> BoxedStrategy<Vec<u8>> {
        match self {
            Self::Bool => (0_u8..=1).prop_map(|value| vec![value]).boxed(),
            Self::UInt(bits) | Self::LowBits(bits) if *bits < 8 => {
                (0_u8..(1 << bits)).prop_map(|value| vec![value]).boxed()
            }
            Self::Int(bits) if *bits < 8 => {
                let half = 1_i8 << (bits - 1);
                (-half..half)
                    .prop_map(|value| value.to_ne_bytes().to_vec())
                    .boxed()
            }
            Self::UInt(bits) if bits % 8 != 0 => {
                // e.g. 31 or 63 bits for unsigned integers restricted to the signed range
                let size = self.element_size().unwrap();
                (0..(1_u64 << bits))
                    .prop_map(move |value| match size {
                        4 => u32::try_from(value).unwrap().to_ne_bytes().to_vec(),
                        _ => value.to_ne_bytes().to_vec(),
                    })
                    .boxed()
            }
            Self::Float32 => (-1.0e6_f32..1.0e6)
                .prop_map(|value| value.to_ne_bytes().to_vec())
                .boxed(),
            Self::Float64 => (-1.0e12_f64..1.0e12)
                .prop_map(|value| value.to_ne_bytes().to_vec())
                .boxed(),
            Self::Complex(values) => (values.element(), values.element())
                .prop_map(|(re, im)| [re, im].concat())
                .boxed(),
            Self::Utf32(length) => prop::collection::vec(any::<char>(), *length)
                .prop_map(|chars| {
                    chars
                        .into_iter()
                        .flat_map(|char| u32::from(char).to_ne_bytes())
                        .collect()
                })
                .boxed(),
            Self::String => "\\PC{0,6}".prop_map(String::into_bytes).boxed(),
            Self::Bytes => prop::collection::vec(any::<u8>(), 0..8).boxed(),
            Self::Optional(values) => values.element(),
            Self::UInt(_)
            | Self::Int(_)
            | Self::LowBits(_)
            | Self::Half
            | Self::Byte
            | Self::Raw(_) => {
                prop::collection::vec(any::<u8>(), self.element_size().unwrap()).boxed()
            }
        }
    }

    fn data(&self, num_elements: usize) -> BoxedStrategy<Data> {
        let variable = self.element_size().is_none();
        (
            prop::collection::vec(self.element(), num_elements),
            prop::collection::vec(
                prop::collection::vec(0_u8..=1, num_elements),
                self.optional_depth(),
            ),
        )
            .prop_map(move |(elements, masks)| Data::from_elements(&elements, variable, masks))
            .boxed()
    }
}

/// A data type and fill value.
#[derive(Debug)]
pub(crate) struct DataTypeCase {
    pub(crate) label: &'static str,
    pub(crate) data_type: DataType,
    pub(crate) fill_value: FillValue,
    pub(crate) values: Values,
}

impl DataTypeCase {
    fn is_optional(&self) -> bool {
        matches!(self.values, Values::Optional(_))
    }

    fn is_variable(&self) -> bool {
        self.values.element_size().is_none()
    }

    fn is_fixed(&self) -> bool {
        !self.is_optional() && !self.is_variable()
    }

    /// Normalise data for comparison: null elements of optional data are zeroed.
    pub(crate) fn canonical(&self, data: &Data) -> Data {
        data.canonical(self.values.element_size())
    }

    /// The `numcodecs.fixedscaleoffset` dtype of this data type, if supported.
    fn numpy_dtype(&self) -> Option<&'static str> {
        Some(match self.label {
            "uint8" => "u1",
            "int8" => "|i1",
            "uint16" => "<u2",
            "int16" => "<i2",
            "uint32" => "<u4",
            "int32" => "<i4",
            "uint64" => "<u8",
            "int64" => "<i8",
            "float32" | "float32(fill=NaN)" => "<f4",
            "float64" | "float64(fill=NaN)" => "<f8",
            _ => return None,
        })
    }
}

/// All tested data types.
#[must_use]
#[allow(clippy::too_many_lines)]
pub(crate) fn data_types() -> Vec<DataTypeCase> {
    use Values::{
        Bool, Byte, Bytes, Complex, Float32, Float64, Half, Int, LowBits, Optional, Raw, String,
        UInt, Utf32,
    };
    fn dt(
        label: &'static str,
        data_type: DataType,
        fill_value: impl Into<FillValue>,
        values: Values,
    ) -> DataTypeCase {
        DataTypeCase {
            label,
            data_type,
            fill_value: fill_value.into(),
            values,
        }
    }
    let complex = |values: Values| Complex(Box::new(values));
    let optional = |values: Values| Optional(Box::new(values));
    let zeros = |size: usize| FillValue::new(vec![0; size]);
    let second = NonZeroU32::new(1).unwrap();
    vec![
        dt("bool", data_type::bool(), false, Bool),
        dt("int2", data_type::int2(), zeros(1), Int(2)),
        dt("int4", data_type::int4(), zeros(1), Int(4)),
        dt("int8", data_type::int8(), 0_i8, Int(8)),
        dt("int16", data_type::int16(), 0_i16, Int(16)),
        dt("int32", data_type::int32(), 0_i32, Int(32)),
        dt("int64", data_type::int64(), 0_i64, Int(64)),
        dt("uint2", data_type::uint2(), zeros(1), UInt(2)),
        dt("uint4", data_type::uint4(), zeros(1), UInt(4)),
        dt("uint8", data_type::uint8(), 0_u8, UInt(8)),
        dt("uint16", data_type::uint16(), 0_u16, UInt(16)),
        dt("uint32", data_type::uint32(), 0_u32, UInt(32)),
        dt("uint64", data_type::uint64(), 0_u64, UInt(64)),
        dt(
            "float4_e2m1fn",
            data_type::float4_e2m1fn(),
            zeros(1),
            LowBits(4),
        ),
        dt(
            "float6_e2m3fn",
            data_type::float6_e2m3fn(),
            zeros(1),
            LowBits(6),
        ),
        dt(
            "float6_e3m2fn",
            data_type::float6_e3m2fn(),
            zeros(1),
            LowBits(6),
        ),
        dt("float8_e3m4", data_type::float8_e3m4(), zeros(1), Byte),
        dt("float8_e4m3", data_type::float8_e4m3(), zeros(1), Byte),
        dt(
            "float8_e4m3b11fnuz",
            data_type::float8_e4m3b11fnuz(),
            zeros(1),
            Byte,
        ),
        dt(
            "float8_e4m3fnuz",
            data_type::float8_e4m3fnuz(),
            zeros(1),
            Byte,
        ),
        dt("float8_e5m2", data_type::float8_e5m2(), zeros(1), Byte),
        dt(
            "float8_e5m2fnuz",
            data_type::float8_e5m2fnuz(),
            zeros(1),
            Byte,
        ),
        dt(
            "float8_e8m0fnu",
            data_type::float8_e8m0fnu(),
            zeros(1),
            Byte,
        ),
        dt("bfloat16", data_type::bfloat16(), zeros(2), Half),
        dt("float16", data_type::float16(), zeros(2), Half),
        dt("float32", data_type::float32(), 0.0_f32, Float32),
        dt("float32(fill=NaN)", data_type::float32(), f32::NAN, Float32),
        dt("float64", data_type::float64(), 0.0_f64, Float64),
        dt("float64(fill=NaN)", data_type::float64(), f64::NAN, Float64),
        dt(
            "complex_float4_e2m1fn",
            data_type::complex_float4_e2m1fn(),
            zeros(2),
            complex(LowBits(4)),
        ),
        dt(
            "complex_float6_e2m3fn",
            data_type::complex_float6_e2m3fn(),
            zeros(2),
            complex(LowBits(6)),
        ),
        dt(
            "complex_float6_e3m2fn",
            data_type::complex_float6_e3m2fn(),
            zeros(2),
            complex(LowBits(6)),
        ),
        dt(
            "complex_float8_e3m4",
            data_type::complex_float8_e3m4(),
            zeros(2),
            complex(Byte),
        ),
        dt(
            "complex_float8_e4m3",
            data_type::complex_float8_e4m3(),
            zeros(2),
            complex(Byte),
        ),
        dt(
            "complex_float8_e4m3b11fnuz",
            data_type::complex_float8_e4m3b11fnuz(),
            zeros(2),
            complex(Byte),
        ),
        dt(
            "complex_float8_e4m3fnuz",
            data_type::complex_float8_e4m3fnuz(),
            zeros(2),
            complex(Byte),
        ),
        dt(
            "complex_float8_e5m2",
            data_type::complex_float8_e5m2(),
            zeros(2),
            complex(Byte),
        ),
        dt(
            "complex_float8_e5m2fnuz",
            data_type::complex_float8_e5m2fnuz(),
            zeros(2),
            complex(Byte),
        ),
        dt(
            "complex_float8_e8m0fnu",
            data_type::complex_float8_e8m0fnu(),
            zeros(2),
            complex(Byte),
        ),
        dt(
            "complex_bfloat16",
            data_type::complex_bfloat16(),
            zeros(4),
            complex(Half),
        ),
        dt(
            "complex_float16",
            data_type::complex_float16(),
            zeros(4),
            complex(Half),
        ),
        dt(
            "complex_float32",
            data_type::complex_float32(),
            zeros(8),
            complex(Float32),
        ),
        dt(
            "complex_float64",
            data_type::complex_float64(),
            zeros(16),
            complex(Float64),
        ),
        dt(
            "complex64",
            data_type::complex64(),
            zeros(8),
            complex(Float32),
        ),
        dt(
            "complex128",
            data_type::complex128(),
            zeros(16),
            complex(Float64),
        ),
        dt(
            "numpy.datetime64",
            data_type::numpy_datetime64(NumpyTimeUnit::Second, second),
            zeros(8),
            Int(64),
        ),
        dt(
            "numpy.timedelta64",
            data_type::numpy_timedelta64(NumpyTimeUnit::Second, second),
            zeros(8),
            Int(64),
        ),
        dt("r24", data_type::raw_bits(3), zeros(3), Raw(3)),
        dt(
            "fixed_length_utf32",
            data_type::fixed_length_utf32(NonZeroU64::new(12).unwrap()).unwrap(),
            zeros(12),
            Utf32(3),
        ),
        dt("string", data_type::string(), "", String),
        dt("bytes", data_type::bytes(), Vec::<u8>::new(), Bytes),
        dt(
            "optional<uint8>",
            data_type::uint8().to_optional(),
            FillValue::new_optional_null(),
            optional(UInt(8)),
        ),
        dt(
            "optional<float32>",
            data_type::float32().to_optional(),
            FillValue::new_optional_null(),
            optional(Float32),
        ),
        dt(
            "optional<optional<float32>>",
            data_type::float32().to_optional().to_optional(),
            FillValue::new_optional_null(),
            optional(optional(Float32)),
        ),
        dt(
            "optional<string>",
            data_type::string().to_optional(),
            FillValue::new_optional_null(),
            optional(String),
        ),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ZfpMode {
    FixedRate,
    FixedPrecision,
    FixedAccuracy,
    Reversible,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VlenKind {
    Vlen,
    V2,
    Array,
    Bytes,
    Utf8,
}

/// The fully swept (categorical) part of a codec configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CodecKind {
    // Array-to-array
    CastValue {
        data_type: &'static str,
    },
    FixedScaleOffset {
        astype: bool,
    },
    Reshape,
    Squeeze,
    Transpose,
    BitRound,
    // Array-to-bytes
    Bytes,
    PackBits,
    Pcodec,
    Sharding {
        index_at_start: bool,
    },
    Zfp(ZfpMode),
    Zfpy(ZfpMode),
    Vlen(VlenKind),
    Optional,
    // Bytes-to-bytes
    Adler32,
    Blosc {
        cname: &'static str,
        shuffle: &'static str,
    },
    Bz2,
    Crc32c,
    Fletcher32,
    Gdeflate,
    Gzip,
    Shuffle,
    Zlib,
    Zstd {
        checksum: bool,
    },
}

impl std::fmt::Display for CodecKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CastValue { data_type } => write!(f, "cast_value({data_type})"),
            Self::FixedScaleOffset { astype: false } => write!(f, "fixedscaleoffset"),
            Self::FixedScaleOffset { astype: true } => write!(f, "fixedscaleoffset(astype=u1)"),
            Self::Reshape => write!(f, "reshape"),
            Self::Squeeze => write!(f, "squeeze"),
            Self::Transpose => write!(f, "transpose"),
            Self::BitRound => write!(f, "bitround"),
            Self::Bytes => write!(f, "bytes"),
            Self::PackBits => write!(f, "packbits"),
            Self::Pcodec => write!(f, "pcodec"),
            Self::Sharding { index_at_start } => write!(
                f,
                "sharding(index={})",
                if *index_at_start { "start" } else { "end" }
            ),
            Self::Zfp(mode) => write!(f, "zfp({mode:?})"),
            Self::Zfpy(mode) => write!(f, "zfpy({mode:?})"),
            Self::Vlen(vlen) => write!(
                f,
                "{}",
                match vlen {
                    VlenKind::Vlen => "vlen",
                    VlenKind::V2 => "vlen-v2",
                    VlenKind::Array => "vlen-array",
                    VlenKind::Bytes => "vlen-bytes",
                    VlenKind::Utf8 => "vlen-utf8",
                }
            ),
            Self::Optional => write!(f, "optional"),
            Self::Adler32 => write!(f, "adler32"),
            Self::Blosc { cname, shuffle } => write!(f, "blosc({cname},{shuffle})"),
            Self::Bz2 => write!(f, "bz2"),
            Self::Crc32c => write!(f, "crc32c"),
            Self::Fletcher32 => write!(f, "fletcher32"),
            Self::Gdeflate => write!(f, "gdeflate"),
            Self::Gzip => write!(f, "gzip"),
            Self::Shuffle => write!(f, "shuffle"),
            Self::Zlib => write!(f, "zlib"),
            Self::Zstd { checksum } => write!(f, "zstd(checksum={checksum})"),
        }
    }
}

/// All tested codec kinds.
#[must_use]
pub(crate) fn codec_kinds() -> Vec<CodecKind> {
    use CodecKind as C;
    let zfp_modes = [
        ZfpMode::FixedRate,
        ZfpMode::FixedPrecision,
        ZfpMode::FixedAccuracy,
        ZfpMode::Reversible,
    ];
    let mut kinds: Vec<CodecKind> = ["uint8", "int16", "int32", "float32", "float64"]
        .into_iter()
        .map(|data_type| C::CastValue { data_type })
        .collect();
    kinds.extend([
        C::FixedScaleOffset { astype: false },
        C::FixedScaleOffset { astype: true },
        C::Reshape,
        C::Squeeze,
        C::Transpose,
        C::BitRound,
        C::Bytes,
        C::PackBits,
        C::Pcodec,
        C::Sharding {
            index_at_start: false,
        },
        C::Sharding {
            index_at_start: true,
        },
    ]);
    kinds.extend(zfp_modes.map(C::Zfp));
    kinds.extend(zfp_modes.map(C::Zfpy));
    kinds.extend(
        [
            VlenKind::Vlen,
            VlenKind::V2,
            VlenKind::Array,
            VlenKind::Bytes,
            VlenKind::Utf8,
        ]
        .map(C::Vlen),
    );
    kinds.extend([C::Optional, C::Adler32]);
    for cname in ["blosclz", "lz4", "lz4hc", "zlib", "zstd"] {
        for shuffle in ["noshuffle", "shuffle", "bitshuffle"] {
            kinds.push(C::Blosc { cname, shuffle });
        }
    }
    kinds.extend([
        C::Bz2,
        C::Crc32c,
        C::Fletcher32,
        C::Gdeflate,
        C::Gzip,
        C::Shuffle,
        C::Zlib,
        C::Zstd { checksum: false },
        C::Zstd { checksum: true },
    ]);
    kinds
}

fn vlen_codec(vlen: VlenKind) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
    match vlen {
        VlenKind::Vlen => Arc::new(VlenCodec::default()),
        VlenKind::V2 => Arc::new(VlenV2Codec::new()),
        VlenKind::Array => Arc::new(VlenArrayCodec::new()),
        VlenKind::Bytes => Arc::new(VlenBytesCodec::new()),
        VlenKind::Utf8 => Arc::new(VlenUtf8Codec::new()),
    }
}

impl CodecKind {
    /// Whether this codec kind is structurally applicable to a data type.
    ///
    /// Anything else is left to `zarrs` to decide.
    fn applies_to(self, data_type: &DataTypeCase) -> bool {
        match self {
            Self::FixedScaleOffset { .. } => data_type.numpy_dtype().is_some(),
            Self::BitRound
            | Self::CastValue { .. }
            | Self::Bytes
            | Self::PackBits
            | Self::Pcodec
            | Self::Zfp(_)
            | Self::Zfpy(_)
            | Self::Shuffle => data_type.is_fixed(),
            Self::Vlen(_) => data_type.is_variable() && !data_type.is_optional(),
            Self::Optional => data_type.is_optional(),
            _ => true,
        }
    }

    fn is_lossy(self) -> bool {
        match self {
            Self::CastValue { .. } | Self::FixedScaleOffset { .. } | Self::BitRound => true,
            Self::Zfp(mode) | Self::Zfpy(mode) => mode != ZfpMode::Reversible,
            _ => false,
        }
    }

    /// The values to generate for a data type with this codec.
    fn values(self, data_type: &DataTypeCase) -> Values {
        match (self, &data_type.values) {
            // zfp clamps unsigned 32/64-bit integers to the signed range
            (Self::Zfp(_) | Self::Zfpy(_), Values::UInt(bits @ (32 | 64))) => {
                Values::UInt(bits - 1)
            }
            (_, values) => values.clone(),
        }
    }
}

/// A fully swept combination of a codec kind and data type.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Combination {
    pub(crate) codec: CodecKind,
    pub(crate) data_type: usize,
}

/// All applicable combinations of `codecs` and `data_types`.
#[must_use]
pub(crate) fn combinations(codecs: &[CodecKind], data_types: &[DataTypeCase]) -> Vec<Combination> {
    codecs
        .iter()
        .flat_map(|&codec| {
            data_types
                .iter()
                .enumerate()
                .filter(move |(_, data_type)| codec.applies_to(data_type))
                .map(move |(data_type, _)| Combination { codec, data_type })
        })
        .collect()
}

/// Sampled (non-categorical) codec parameters.
///
/// Each parameter is only used by the codecs it is relevant to.
#[derive(Debug, Clone)]
pub(crate) struct CodecParams {
    level: u8,
    keepbits: u32,
    offset: f32,
    scale: f32,
    rate: f64,
    precision: u32,
    tolerance: f64,
    rounding: &'static str,
    inner_chunk_shape: Vec<u64>,
}

impl CodecParams {
    fn strategy(chunk_shape: &[u64]) -> impl Strategy<Value = Self> + use<> {
        let inner_chunk_shape = chunk_shape
            .iter()
            .map(|&size| {
                prop::sample::select((1..=size).filter(|d| size % d == 0).collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        (
            1_u8..=9,
            1_u32..=7,
            prop::sample::select(vec![0.0_f32, 1000.0]),
            prop::sample::select(vec![1.0_f32, 10.0, 0.1]),
            prop::sample::select(vec![4.0_f64, 8.0, 16.0]),
            prop::sample::select(vec![8_u32, 16, 24]),
            prop::sample::select(vec![1e-3_f64, 1e-2, 0.1]),
            prop::sample::select(vec![
                "nearest-even",
                "towards-zero",
                "towards-positive",
                "towards-negative",
                "nearest-away",
            ]),
            inner_chunk_shape,
        )
            .prop_map(
                |(
                    level,
                    keepbits,
                    offset,
                    scale,
                    rate,
                    precision,
                    tolerance,
                    rounding,
                    inner_chunk_shape,
                )| {
                    Self {
                        level,
                        keepbits,
                        offset,
                        scale,
                        rate,
                        precision,
                        tolerance,
                        rounding,
                        inner_chunk_shape,
                    }
                },
            )
    }
}

/// A test case: a sample of a [`Combination`].
#[derive(Debug, Clone)]
pub(crate) struct Case {
    pub(crate) combination: usize,
    pub(crate) shape: Vec<u64>,
    pub(crate) chunk_shape: Vec<u64>,
    /// The array metadata as written to the store.
    pub(crate) metadata: Value,
    pub(crate) data: Data,
    pub(crate) lossy: bool,
}

impl Case {
    /// A short description of the case: its codecs, shape and chunk shape.
    #[must_use]
    pub(crate) fn describe(&self) -> String {
        zarrs_regression_testing::schema::describe(&self.metadata, &self.shape, &self.chunk_shape)
    }
}

/// Sample `samples` cases of each combination with a seeded RNG.
///
/// # Errors
/// Returns an error if case generation fails.
pub(crate) fn sample_cases(
    combinations: &[Combination],
    data_types: &[DataTypeCase],
    samples: usize,
    seed: u64,
) -> Result<Vec<Case>, String> {
    let mut rng_seed = [0; 32];
    rng_seed[..8].copy_from_slice(&seed.to_le_bytes());
    let mut runner = TestRunner::new_with_rng(
        Config::default(),
        TestRng::from_seed(RngAlgorithm::ChaCha, &rng_seed),
    );
    let mut cases = Vec::with_capacity(combinations.len() * samples);
    for (index, combination) in combinations.iter().enumerate() {
        let data_type = &data_types[combination.data_type];
        let values = combination.codec.values(data_type);
        let strategy = (1_u64..=4, 1_u64..=4, 1_u64..=4, 1_u64..=4).prop_flat_map(
            |(shape0, shape1, chunk0, chunk1)| {
                let shape = vec![shape0, shape1];
                let chunk_shape = vec![chunk0.min(shape0), chunk1.min(shape1)];
                let num_elements = usize::try_from(shape0 * shape1).unwrap();
                (
                    Just(shape),
                    CodecParams::strategy(&chunk_shape),
                    Just(chunk_shape),
                    values.data(num_elements),
                )
            },
        );
        for _ in 0..samples {
            // Resample data that is entirely the fill value, as no chunks would be encoded
            let mut sample = || -> Result<_, String> {
                let (shape, mut params, chunk_shape, mut data) = strategy
                    .new_tree(&mut runner)
                    .map_err(|err| format!("generate case: {err}"))?
                    .current();
                if let CodecKind::FixedScaleOffset { astype } = combination.codec {
                    fixedscaleoffset_in_range(&mut data, &values, &mut params, astype);
                }
                Ok((shape, params, chunk_shape, data))
            };
            let (mut shape, mut params, mut chunk_shape, mut data) = sample()?;
            for _ in 0..100 {
                if !to_array_bytes(&data)?.is_fill_value(&data_type.fill_value) {
                    break;
                }
                (shape, params, chunk_shape, data) = sample()?;
            }
            let metadata =
                array_metadata(combination.codec, &params, data_type, &shape, &chunk_shape)?;
            cases.push(Case {
                combination: index,
                shape,
                chunk_shape,
                metadata,
                data,
                lossy: combination.codec.is_lossy(),
            });
        }
    }
    Ok(cases)
}

/// The range of an integer data type with `values`.
#[allow(clippy::cast_precision_loss)]
fn integer_range(values: &Values) -> Option<(f64, f64)> {
    match *values {
        Values::UInt(bits) => Some((0.0, (u128::pow(2, bits) - 1) as f64)),
        Values::Int(bits) => Some((
            -(u128::pow(2, bits - 1) as f64),
            (u128::pow(2, bits - 1) - 1) as f64,
        )),
        _ => None,
    }
}

/// Clamp the elements of `data` (with `values`) so that `numcodecs.fixedscaleoffset` with `params` encodes them in the range of the encoded data type (`u1` if `astype`, otherwise the data type).
///
/// Encoding values out of range is unspecified, as `numcodecs` casts them with `numpy`, which is platform dependent.
/// If no values are in range with the sampled offset, the offset is zero.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless
)]
fn fixedscaleoffset_in_range(
    data: &mut Data,
    values: &Values,
    params: &mut CodecParams,
    astype: bool,
) {
    let encoded_range = if astype {
        Some((0.0, 255.0))
    } else {
        integer_range(values)
    };
    // Float encoded values are not out of range
    let Some((encoded_min, encoded_max)) = encoded_range else {
        return;
    };
    let scale = f64::from(params.scale);
    let range = |offset: f64| {
        let (mut min, mut max) = (offset + encoded_min / scale, offset + encoded_max / scale);
        if let Some((data_min, data_max)) = integer_range(values) {
            min = min.max(data_min).ceil();
            max = max.min(data_max).floor();
        }
        (min, max)
    };
    let (mut min, mut max) = range(f64::from(params.offset));
    if min > max {
        params.offset = 0.0;
        (min, max) = range(0.0);
    }
    let clamp = |value: f64| {
        if value.is_nan() {
            min
        } else {
            value.clamp(min, max)
        }
    };
    macro_rules! clamp_impl {
        ($ty:ty) => {
            for chunk in data.bytes.as_chunks_mut::<{ size_of::<$ty>() }>().0 {
                let value = clamp(<$ty>::from_ne_bytes(*chunk) as f64);
                *chunk = (value as $ty).to_ne_bytes();
            }
        };
    }
    match values {
        Values::UInt(8) => clamp_impl!(u8),
        Values::UInt(16) => clamp_impl!(u16),
        Values::UInt(32) => clamp_impl!(u32),
        Values::UInt(64) => clamp_impl!(u64),
        Values::Int(8) => clamp_impl!(i8),
        Values::Int(16) => clamp_impl!(i16),
        Values::Int(32) => clamp_impl!(i32),
        Values::Int(64) => clamp_impl!(i64),
        Values::Float32 => clamp_impl!(f32),
        Values::Float64 => clamp_impl!(f64),
        _ => {}
    }
}

fn bytes_codec() -> Value {
    json!({"name": "bytes", "configuration": {"endian": "little"}})
}

/// Generate the array metadata for a case.
///
/// The data type, fill value, and default codecs are those `zarrs` produces for the data type.
#[allow(clippy::too_many_lines)]
fn array_metadata(
    kind: CodecKind,
    params: &CodecParams,
    data_type: &DataTypeCase,
    shape: &[u64],
    chunk_shape: &[u64],
) -> Result<Value, String> {
    let builder = ArrayBuilder::new(
        shape.to_vec(),
        chunk_shape.to_vec(),
        data_type.data_type.clone(),
        data_type.fill_value.clone(),
    );
    let metadata = builder
        .build_metadata()
        .map_err(|err| format!("build metadata for {kind} {}: {err}", data_type.label))?;
    let mut metadata = serde_json::to_value(metadata).map_err(|err| err.to_string())?;
    let default_codecs = metadata["codecs"].take();
    let default_codecs = default_codecs.as_array().cloned().unwrap_or_default();
    let element_size = data_type.data_type.fixed_size().unwrap_or(1);

    let p = params;
    let zfp = |name: &str, mode: ZfpMode| -> Value {
        // `zfp` uses string modes, `numcodecs.zfpy` uses integer modes
        let (mode_name, mode_int, key, value) = match mode {
            ZfpMode::FixedRate => ("fixed_rate", 2, "rate", json!(p.rate)),
            ZfpMode::FixedPrecision => ("fixed_precision", 3, "precision", json!(p.precision)),
            ZfpMode::FixedAccuracy => ("fixed_accuracy", 4, "tolerance", json!(p.tolerance)),
            ZfpMode::Reversible => ("reversible", 5, "", Value::Null),
        };
        let mut configuration = if name == "zfp" {
            json!({"mode": mode_name})
        } else {
            json!({"mode": mode_int})
        };
        if !key.is_empty() {
            configuration[key] = value;
        }
        json!({"name": name, "configuration": configuration})
    };
    // An array-to-array codec before, or a bytes-to-bytes codec after, the default codecs
    let before = |codec: Value| [vec![codec], default_codecs.clone()].concat();
    let after = |codec: Value| [default_codecs.clone(), vec![codec]].concat();

    let codecs: Vec<Value> = match kind {
        // Array-to-array
        CodecKind::CastValue {
            data_type: target_data_type,
        } => {
            let codec = json!({
                "name": "cast_value",
                "configuration": {
                    "data_type": target_data_type,
                    "rounding": p.rounding,
                    "out_of_range": "clamp"
                }
            });
            vec![codec, bytes_codec()]
        }
        CodecKind::FixedScaleOffset { astype } => {
            let mut configuration = json!({
                "offset": p.offset,
                "scale": p.scale,
                "dtype": data_type.numpy_dtype().unwrap(),
            });
            if astype {
                configuration["astype"] = json!("u1");
            }
            before(json!({"name": "numcodecs.fixedscaleoffset", "configuration": configuration}))
        }
        CodecKind::Reshape => {
            let num_elements: u64 = chunk_shape.iter().product();
            before(json!({"name": "reshape", "configuration": {"shape": [num_elements]}}))
        }
        CodecKind::Squeeze => before(json!({"name": "zarrs.squeeze", "configuration": {}})),
        CodecKind::Transpose => {
            before(json!({"name": "transpose", "configuration": {"order": [1, 0]}}))
        }
        CodecKind::BitRound => {
            before(json!({"name": "bitround", "configuration": {"keepbits": p.keepbits}}))
        }
        // Array-to-bytes
        CodecKind::Bytes => vec![bytes_codec()],
        CodecKind::PackBits => vec![json!({"name": "packbits", "configuration": {}})],
        CodecKind::Pcodec => vec![json!({
            "name": "numcodecs.pcodec",
            "configuration": {
                "level": p.level,
                "mode_spec": "auto",
                "delta_spec": "auto",
                "paging_spec": "equal_pages_up_to",
                "equal_pages_up_to": 262_144
            }
        })],
        CodecKind::Sharding { index_at_start } => vec![json!({
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": p.inner_chunk_shape,
                "codecs": default_codecs,
                "index_codecs": [bytes_codec(), {"name": "crc32c"}],
                "index_location": if index_at_start { "start" } else { "end" }
            }
        })],
        CodecKind::Zfp(mode) => vec![zfp("zfp", mode)],
        CodecKind::Zfpy(mode) => vec![zfp("numcodecs.zfpy", mode)],
        // Not set with the builder, which rejects non-conformant combinations (e.g. `vlen-bytes` with `string`) that are still read
        CodecKind::Vlen(vlen) => {
            let codec = vlen_codec(vlen);
            let name = codec.name_v3().ok_or("vlen codec has no V3 name")?;
            let configuration = codec
                .configuration_v3(&CodecMetadataOptions::default())
                .ok_or("vlen codec has no V3 configuration")?;
            vec![json!({"name": name, "configuration": configuration})]
        }
        CodecKind::Optional => default_codecs,
        // Bytes-to-bytes
        CodecKind::Adler32 => after(json!({"name": "numcodecs.adler32", "configuration": {}})),
        CodecKind::Blosc { cname, shuffle } => after(json!({
            "name": "blosc",
            "configuration": {
                "cname": cname,
                "clevel": p.level,
                "shuffle": shuffle,
                "typesize": element_size,
                "blocksize": 0
            }
        })),
        CodecKind::Bz2 => {
            after(json!({"name": "numcodecs.bz2", "configuration": {"level": p.level}}))
        }
        CodecKind::Crc32c => after(json!({"name": "crc32c"})),
        CodecKind::Fletcher32 => {
            after(json!({"name": "numcodecs.fletcher32", "configuration": {}}))
        }
        CodecKind::Gdeflate => {
            after(json!({"name": "zarrs.gdeflate", "configuration": {"level": p.level}}))
        }
        CodecKind::Gzip => after(json!({"name": "gzip", "configuration": {"level": p.level}})),
        CodecKind::Shuffle => after(
            json!({"name": "numcodecs.shuffle", "configuration": {"elementsize": element_size}}),
        ),
        CodecKind::Zlib => {
            after(json!({"name": "numcodecs.zlib", "configuration": {"level": p.level}}))
        }
        CodecKind::Zstd { checksum } => after(json!({
            "name": "zstd",
            "configuration": {"level": p.level, "checksum": checksum}
        })),
    };
    metadata["codecs"] = Value::Array(codecs);
    Ok(metadata)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// Collect the `name` of every object in `value` (recursively), or `value` itself if it is a string.
    fn collect_names(value: &Value, names: &mut BTreeSet<String>) {
        match value {
            Value::String(name) => {
                names.insert(name.clone());
            }
            Value::Object(object) => {
                if let Some(Value::String(name)) = object.get("name") {
                    names.insert(name.clone());
                }
                for value in object.values().filter(|value| !value.is_string()) {
                    collect_names(value, names);
                }
            }
            Value::Array(array) => {
                for value in array {
                    collect_names(value, names);
                }
            }
            _ => {}
        }
    }

    /// The names of all codecs and data types in the metadata of a sample of every combination.
    fn tested_names(key: &str) -> BTreeSet<String> {
        let data_types = data_types();
        let combinations = combinations(&codec_kinds(), &data_types);
        let mut names = BTreeSet::new();
        for case in sample_cases(&combinations, &data_types, 1, 0).unwrap() {
            collect_names(&case.metadata[key], &mut names);
        }
        names
    }

    #[test]
    fn all_codec_features_enabled() {
        let zarrs_manifest = include_str!("../../zarrs/Cargo.toml");
        let manifest = include_str!("../Cargo.toml");
        for line in zarrs_manifest.lines().filter(|line| {
            line.split_once('#').is_some_and(|(_, comment)| {
                comment.contains("Enable")
                    && comment.contains("codec")
                    && !comment.contains("DEPRECATED")
            })
        }) {
            let feature = line.split_whitespace().next().unwrap();
            assert!(
                manifest.contains(&format!("\"{feature}\"")),
                "the zarrs codec feature `{feature}` is not enabled"
            );
        }
    }

    #[test]
    fn all_registered_codecs_tested() {
        let names = tested_names("codecs");
        let untested = inventory::iter::<zarrs_codec::CodecPluginV3>()
            .filter(|plugin| !names.iter().any(|name| plugin.match_name(name)))
            .count();
        assert_eq!(
            untested, 0,
            "{untested} registered codec(s) are not tested (tested: {names:?})"
        );
    }

    #[test]
    fn all_registered_data_types_tested() {
        let names = tested_names("data_type");
        let untested = inventory::iter::<zarrs_data_type::DataTypePluginV3>()
            .filter(|plugin| !names.iter().any(|name| plugin.match_name(name)))
            .count();
        assert_eq!(
            untested, 0,
            "{untested} registered data type(s) are not tested (tested: {names:?})"
        );
    }
}
