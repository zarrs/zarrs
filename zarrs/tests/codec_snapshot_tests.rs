//! Snapshot tests for zarrs codecs and data types.
//!
//! These tests create real Zarr arrays on disk for each codec/data type combination,
//! enabling inspection with any Zarr-compatible tool.
//!
//! Run with: `cargo test --all-features -p zarrs codec_snapshot_tests`
//! Update snapshots: `UPDATE_SNAPSHOTS=1 cargo test --all-features -p zarrs codec_snapshot_tests`
//! Add newly supported: `ADD_SNAPSHOTS=1 cargo test --all-features -p zarrs codec_snapshot_tests`
//!
//! The `ADD_SNAPSHOTS` mode only permits adding new snapshots for combinations that were
//! previously marked as unsupported but now succeed. It will not update existing snapshots.

#![allow(missing_docs)]

use half::{bf16, f16};
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::borrow::Cow;
use std::fs;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use zarrs::array::codec::array_to_bytes::optional::OptionalCodec;
use zarrs::array::{
    ArrayBuilder, ArrayBytes, ArrayBytesOffsets, ArrayMetadataOptions, DataType, FillValue,
    data_type,
};
use zarrs::metadata_ext::data_type::NumpyTimeUnit;
use zarrs_codec::{ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesToBytesCodecTraits};
use zarrs_filesystem::FilesystemStore;
use zarrs_plugin::ExtensionAliasesV3;

/// Get a string identifier for a data type (for test matching purposes)
fn data_type_id(data_type: &DataType) -> &'static str {
    let type_id = data_type.as_any().type_id();
    if type_id == TypeId::of::<data_type::BoolDataType>() {
        "bool"
    } else if type_id == TypeId::of::<data_type::Int2DataType>() {
        "int2"
    } else if type_id == TypeId::of::<data_type::Int4DataType>() {
        "int4"
    } else if type_id == TypeId::of::<data_type::Int8DataType>() {
        "int8"
    } else if type_id == TypeId::of::<data_type::Int16DataType>() {
        "int16"
    } else if type_id == TypeId::of::<data_type::Int32DataType>() {
        "int32"
    } else if type_id == TypeId::of::<data_type::Int64DataType>() {
        "int64"
    } else if type_id == TypeId::of::<data_type::UInt2DataType>() {
        "uint2"
    } else if type_id == TypeId::of::<data_type::UInt4DataType>() {
        "uint4"
    } else if type_id == TypeId::of::<data_type::UInt8DataType>() {
        "uint8"
    } else if type_id == TypeId::of::<data_type::UInt16DataType>() {
        "uint16"
    } else if type_id == TypeId::of::<data_type::UInt32DataType>() {
        "uint32"
    } else if type_id == TypeId::of::<data_type::UInt64DataType>() {
        "uint64"
    } else if type_id == TypeId::of::<data_type::Float16DataType>() {
        "float16"
    } else if type_id == TypeId::of::<data_type::Float32DataType>() {
        "float32"
    } else if type_id == TypeId::of::<data_type::Float64DataType>() {
        "float64"
    } else if type_id == TypeId::of::<data_type::BFloat16DataType>() {
        "bfloat16"
    } else if type_id == TypeId::of::<data_type::Float8E4M3DataType>() {
        "float8_e4m3"
    } else if type_id == TypeId::of::<data_type::Float8E5M2DataType>() {
        "float8_e5m2"
    } else if type_id == TypeId::of::<data_type::StringDataType>() {
        "string"
    } else if type_id == TypeId::of::<data_type::BytesDataType>() {
        "bytes"
    } else if type_id == TypeId::of::<data_type::Complex64DataType>() {
        "complex64"
    } else if type_id == TypeId::of::<data_type::Complex128DataType>() {
        "complex128"
    } else if type_id == TypeId::of::<data_type::ComplexFloat16DataType>() {
        "complex_float16"
    } else if type_id == TypeId::of::<data_type::ComplexFloat32DataType>() {
        "complex_float32"
    } else if type_id == TypeId::of::<data_type::ComplexFloat64DataType>() {
        "complex_float64"
    } else if type_id == TypeId::of::<data_type::ComplexBFloat16DataType>() {
        "complex_bfloat16"
    } else if type_id == TypeId::of::<data_type::ComplexFloat8E4M3DataType>() {
        "complex_float8_e4m3"
    } else if type_id == TypeId::of::<data_type::ComplexFloat8E5M2DataType>() {
        "complex_float8_e5m2"
    } else if type_id == TypeId::of::<data_type::NumpyDateTime64DataType>() {
        "numpy.datetime64"
    } else if type_id == TypeId::of::<data_type::NumpyTimeDelta64DataType>() {
        "numpy.timedelta64"
    } else if type_id == TypeId::of::<data_type::RawBitsDataType>() {
        "raw_bits"
    } else if type_id == TypeId::of::<data_type::OptionalDataType>() {
        "optional"
    } else {
        "unknown"
    }
}

// =============================================================================
// Core Types
// =============================================================================

/// Result of a codec test
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "status")]
pub enum CodecTestResult {
    /// The codec combination is supported and produced results
    #[serde(rename = "success")]
    Success,
    /// The codec combination is not supported (codec creation or storage failed)
    #[serde(rename = "unsupported")]
    Unsupported { reason: String },
    /// The codec ran but produced incorrect results (round-trip failure)
    #[serde(rename = "failure")]
    Failure { reason: String },
}

/// Configuration for a single test case
#[derive(Clone, Debug)]
pub struct TestConfig {
    /// Data type to test
    pub data_type: DataType,
    /// Fill value for the array
    pub fill_value: FillValue,
    /// Array shape
    pub array_shape: Vec<u64>,
    /// Chunk shape (outer chunks for sharding)
    pub chunk_shape: Vec<u64>,
    /// Optional array-to-array codecs
    pub array_to_array_codecs: Vec<Arc<dyn ArrayToArrayCodecTraits>>,
    /// Optional array-to-bytes codec (None = use default for data type)
    pub array_to_bytes_codec: Option<Arc<dyn ArrayToBytesCodecTraits>>,
    /// Optional bytes-to-bytes codecs
    pub bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    /// Chunk grid name for snapshot naming (e.g., "regular", "rectangular")
    pub chunk_grid_name: String,
    /// If true, skip byte-by-byte chunk comparison (for non-deterministic codecs like sharding)
    pub non_deterministic: bool,
    /// If true, the codec is lossy and round-trip data won't match exactly
    pub lossy: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            data_type: data_type::uint8(),
            fill_value: 0u8.into(),
            array_shape: vec![8, 24],
            chunk_shape: vec![4, 6],
            array_to_array_codecs: vec![],
            array_to_bytes_codec: None,
            bytes_to_bytes_codecs: vec![],
            chunk_grid_name: "regular".to_string(),
            non_deterministic: false,
            lossy: false,
        }
    }
}

// =============================================================================
// Data Type Matrix
// =============================================================================

/// All data types to test with their fill values and description
#[must_use]
pub fn all_data_types() -> Vec<(DataType, FillValue, &'static str)> {
    vec![
        // Core integers
        (data_type::bool(), false.into(), "fill_false"),
        (data_type::int8(), 0i8.into(), "fill_0"),
        (data_type::int16(), 0i16.into(), "fill_0"),
        (data_type::int32(), 0i32.into(), "fill_0"),
        (data_type::int64(), 0i64.into(), "fill_0"),
        (data_type::uint8(), 0u8.into(), "fill_0"),
        (data_type::uint16(), 0u16.into(), "fill_0"),
        (data_type::uint32(), 0u32.into(), "fill_0"),
        (data_type::uint64(), 0u64.into(), "fill_0"),
        // Sub-byte integers (fill value should be 1 byte packed value)
        (data_type::int2(), FillValue::new(vec![0u8]), "fill_0"),
        (data_type::int4(), FillValue::new(vec![0u8]), "fill_0"),
        (data_type::uint2(), FillValue::new(vec![0u8]), "fill_0"),
        (data_type::uint4(), FillValue::new(vec![0u8]), "fill_0"),
        // Half-precision floats (2 bytes)
        (
            data_type::bfloat16(),
            FillValue::new(vec![0u8; 2]),
            "fill_0",
        ),
        (data_type::float16(), FillValue::new(vec![0u8; 2]), "fill_0"),
        // Float8 variants (1 byte each)
        (
            data_type::float8_e4m3(),
            FillValue::new(vec![0u8]),
            "fill_0",
        ),
        (
            data_type::float8_e5m2(),
            FillValue::new(vec![0u8]),
            "fill_0",
        ),
        // Standard floats
        (data_type::float32(), 0.0f32.into(), "fill_0"),
        (data_type::float32(), f32::NAN.into(), "fill_nan"),
        (data_type::float64(), 0.0f64.into(), "fill_0"),
        (data_type::float64(), f64::NAN.into(), "fill_nan"),
        // Complex half-precision (2 bytes each = 4 bytes total)
        (
            data_type::complex_bfloat16(),
            FillValue::new(vec![0u8; 4]),
            "fill_0",
        ),
        (
            data_type::complex_float16(),
            FillValue::new(vec![0u8; 4]),
            "fill_0",
        ),
        // Complex float8 (1 byte each = 2 bytes total)
        (
            data_type::complex_float8_e4m3(),
            FillValue::new(vec![0u8; 2]),
            "fill_0",
        ),
        (
            data_type::complex_float8_e5m2(),
            FillValue::new(vec![0u8; 2]),
            "fill_0",
        ),
        // Complex standard (8 and 16 bytes)
        (
            data_type::complex_float32(),
            FillValue::new(vec![0u8; 8]),
            "fill_0",
        ),
        (
            data_type::complex_float64(),
            FillValue::new(vec![0u8; 16]),
            "fill_0",
        ),
        (
            data_type::complex64(),
            FillValue::new(vec![0u8; 8]),
            "fill_0",
        ),
        (
            data_type::complex128(),
            FillValue::new(vec![0u8; 16]),
            "fill_0",
        ),
        // NumPy datetime/timedelta (8 bytes - stored as i64)
        (
            data_type::numpy_datetime64(
                NumpyTimeUnit::Second,
                std::num::NonZeroU32::new(1).unwrap(),
            ),
            FillValue::new(vec![0u8; 8]),
            "fill_0",
        ),
        (
            data_type::numpy_timedelta64(
                NumpyTimeUnit::Second,
                std::num::NonZeroU32::new(1).unwrap(),
            ),
            FillValue::new(vec![0u8; 8]),
            "fill_0",
        ),
        // Variable-length
        (data_type::string(), "".into(), "fill_empty"),
        (data_type::bytes(), FillValue::new(vec![]), "fill_empty"),
        // RawBits
        (
            data_type::raw_bits(3),
            FillValue::new(vec![0, 0, 0]),
            "fill_zeros",
        ),
        // Optional types
        (
            data_type::uint8().to_optional(),
            FillValue::new_optional_null(),
            "fill_null",
        ),
        (
            data_type::float32().to_optional(),
            FillValue::new_optional_null(),
            "fill_null",
        ),
        // Nested optional (Optional<Optional<Float32>>)
        (
            data_type::float32().to_optional().to_optional(),
            FillValue::new_optional_null(),
            "fill_null",
        ),
        // Optional string
        (
            data_type::string().to_optional(),
            FillValue::new_optional_null(),
            "fill_null",
        ),
    ]
}

// =============================================================================
// Test Data Generation
// =============================================================================

/// Generate deterministic fixed-length test data bytes for a given data type
fn generate_fixed_bytes(data_type: &DataType, num_elements: usize) -> Vec<u8> {
    match data_type_id(data_type) {
        "bool" => (0..num_elements).map(|i| (i % 2) as u8).collect(),

        // Sub-byte integer types use i8/u8 representation in memory (unpacked)
        // The packbits codec handles packing to sub-byte sizes
        "int2" => {
            // Int2 values range from -2 to 1 (stored as i8)
            (0..num_elements)
                .map(|i| {
                    let v = ((i % 4) as i8) - 2; // cycles -2, -1, 0, 1
                    v as u8
                })
                .collect()
        }
        "uint2" => {
            // UInt2 values range from 0 to 3 (stored as u8)
            (0..num_elements).map(|i| (i % 4) as u8).collect()
        }
        "int4" => {
            // Int4 values range from -8 to 7 (stored as i8)
            (0..num_elements)
                .map(|i| {
                    let v = ((i % 16) as i8) - 8; // cycles -8 to 7
                    v as u8
                })
                .collect()
        }
        "uint4" => {
            // UInt4 values range from 0 to 15 (stored as u8)
            (0..num_elements).map(|i| (i % 16) as u8).collect()
        }

        // Integer types
        "int8" => (0..num_elements)
            .map(|i| ((i % 256) as i8).to_ne_bytes()[0])
            .collect(),
        "uint8" => (0..num_elements).map(|i| (i % 256) as u8).collect(),
        "int16" => (0..num_elements)
            .flat_map(|i| ((i % 65536) as i16).to_ne_bytes())
            .collect(),
        "uint16" => (0..num_elements)
            .flat_map(|i| ((i % 65536) as u16).to_ne_bytes())
            .collect(),
        "int32" => (0..num_elements)
            .flat_map(|i| (i as i32).to_ne_bytes())
            .collect(),
        "uint32" => (0..num_elements)
            .flat_map(|i| (i as u32).to_ne_bytes())
            .collect(),
        "int64" => (0..num_elements)
            .flat_map(|i| (i as i64).to_ne_bytes())
            .collect(),
        "uint64" => (0..num_elements)
            .flat_map(|i| (i as u64).to_ne_bytes())
            .collect(),
        "bfloat16" => (0..num_elements)
            .flat_map(|i| bf16::from_f32((i as f32) * 0.5).to_ne_bytes())
            .collect(),
        "float16" => (0..num_elements)
            .flat_map(|i| f16::from_f32((i as f32) * 0.5).to_ne_bytes())
            .collect(),

        // Float8 variants (1 byte each)
        "float8_e4m3" | "float8_e5m2" => (0..num_elements).map(|i| (i % 128) as u8).collect(),

        "float32" => (0..num_elements)
            .flat_map(|i| ((i as f32) * 0.5).to_ne_bytes())
            .collect(),

        "float64" => (0..num_elements)
            .flat_map(|i| ((i as f64) * 0.5).to_ne_bytes())
            .collect(),

        // Complex half-precision (4 bytes total)
        "complex_bfloat16" => (0..num_elements)
            .flat_map(|i| {
                let real = bf16::from_f32((i as f32) * 0.5);
                let imag = bf16::from_f32((i as f32) * 0.25);
                let mut bytes = real.to_ne_bytes().to_vec();
                bytes.extend(imag.to_ne_bytes());
                bytes
            })
            .collect(),

        "complex_float16" => (0..num_elements)
            .flat_map(|i| {
                let real = f16::from_f32((i as f32) * 0.5);
                let imag = f16::from_f32((i as f32) * 0.25);
                let mut bytes = real.to_ne_bytes().to_vec();
                bytes.extend(imag.to_ne_bytes());
                bytes
            })
            .collect(),

        // Complex float8 (2 bytes total)
        "complex_float8_e4m3" | "complex_float8_e5m2" => (0..num_elements)
            .flat_map(|i| vec![(i % 128) as u8, ((i + 1) % 128) as u8])
            .collect(),

        "complex_float32" | "complex64" => (0..num_elements)
            .flat_map(|i| {
                let real = (i as f32) * 0.5;
                let imag = (i as f32) * 0.25;
                let mut bytes = real.to_ne_bytes().to_vec();
                bytes.extend(imag.to_ne_bytes());
                bytes
            })
            .collect(),

        "complex_float64" | "complex128" => (0..num_elements)
            .flat_map(|i| {
                let real = (i as f64) * 0.5;
                let imag = (i as f64) * 0.25;
                let mut bytes = real.to_ne_bytes().to_vec();
                bytes.extend(imag.to_ne_bytes());
                bytes
            })
            .collect(),

        // NumPy datetime/timedelta (8 bytes - stored as i64)
        "numpy.datetime64" | "numpy.timedelta64" => (0..num_elements)
            .flat_map(|i| (i as i64).to_ne_bytes())
            .collect(),

        "raw_bits" => {
            let size = data_type.fixed_size().unwrap();
            (0..num_elements)
                .flat_map(|i| vec![(i % 256) as u8; size])
                .collect()
        }

        // Handle remaining fixed-size types generically
        _ => {
            if let Some(size) = data_type.fixed_size() {
                (0..num_elements)
                    .flat_map(|i| vec![(i % 256) as u8; size])
                    .collect()
            } else {
                vec![]
            }
        }
    }
}

/// Generate deterministic test data for a given data type and element count
/// Returns `ArrayBytes` which can be Fixed, Variable, or Optional
pub fn generate_test_data(data_type: &DataType, num_elements: usize) -> ArrayBytes<'static> {
    match data_type_id(data_type) {
        // Variable-length String type
        "string" => {
            let strings: Vec<String> = (0..num_elements)
                .map(|i| format!("str_{:04}", i % 10000))
                .collect();

            // Build bytes and offsets for vlen encoding
            let mut bytes = Vec::new();
            let mut offsets = vec![0usize];
            for s in &strings {
                bytes.extend(s.as_bytes());
                offsets.push(bytes.len());
            }

            let offsets = unsafe { ArrayBytesOffsets::new_unchecked(offsets) };
            unsafe { ArrayBytes::new_vlen_unchecked(bytes, offsets) }
        }

        // Variable-length Bytes type
        "bytes" => {
            let byte_arrays: Vec<Vec<u8>> = (0..num_elements)
                .map(|i| vec![(i % 256) as u8; (i % 8) + 1])
                .collect();

            // Build bytes and offsets for vlen encoding
            let mut bytes = Vec::new();
            let mut offsets = vec![0usize];
            for b in &byte_arrays {
                bytes.extend(b);
                offsets.push(bytes.len());
            }

            let offsets = unsafe { ArrayBytesOffsets::new_unchecked(offsets) };
            unsafe { ArrayBytes::new_vlen_unchecked(bytes, offsets) }
        }

        // Optional types - wrap inner data with validity mask
        "optional" => {
            let opt = data_type.as_optional().unwrap();
            let inner_bytes = generate_test_data(opt.data_type(), num_elements);
            // Create validity mask - every 4th element is null
            let mask: Vec<u8> = (0..num_elements).map(|i| u8::from(i % 4 != 3)).collect();
            inner_bytes.with_optional_mask(mask)
        }

        // All fixed-size types
        _ => ArrayBytes::new_flen(generate_fixed_bytes(data_type, num_elements)),
    }
}

// =============================================================================
// Snapshot Paths
// =============================================================================

/// Get the path to the snapshots directory
#[must_use]
pub fn snapshots_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("snapshots")
}

/// Sanitize a data type name for use in file paths
pub fn sanitize_data_type_name(data_type: &DataType) -> String {
    // For optional types, use the default name and include the inner type name
    let name = if let Some(opt) = data_type.as_optional() {
        format!(
            "{}({})",
            OptionalCodec::aliases_v3().default_name.clone(),
            sanitize_data_type_name(opt.data_type())
        )
    } else {
        data_type
            .name_v3()
            .map_or_else(String::new, |n| n.to_string())
    };
    name.chars()
        .map(|c: char| {
            if c.is_alphanumeric() || c == '_' || c == '.' || c == '(' || c == ')' {
                c.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// Codec category for path organization
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CodecCategory {
    BytesToBytes,
    ArrayToArray,
    ArrayToBytes,
}

impl CodecCategory {
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            CodecCategory::BytesToBytes => "b2b",
            CodecCategory::ArrayToArray => "a2a",
            CodecCategory::ArrayToBytes => "a2b",
        }
    }
}

/// Compute a short checksum of array metadata for use as snapshot ID
fn compute_metadata_checksum(metadata_json: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    metadata_json.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Snapshot path configuration
/// Structure: <supported/unsupported>/<`chunk_grid`>/<`data_type`>/<category>/<codec>/<checksum>
#[derive(Clone)]
pub struct SnapshotPath {
    pub chunk_grid: String,
    pub data_type: String,
    pub category: CodecCategory,
    pub codec: String,
}

impl SnapshotPath {
    pub fn new(
        chunk_grid: &str,
        data_type: &DataType,
        category: CodecCategory,
        codec_name: &str,
        codec_suffix: Option<&str>,
    ) -> Self {
        let codec = match codec_suffix {
            Some(s) if !s.is_empty() => format!("{codec_name}({s})"),
            _ => codec_name.to_string(),
        };
        Self {
            chunk_grid: chunk_grid.to_string(),
            data_type: sanitize_data_type_name(data_type),
            category,
            codec,
        }
    }

    /// Get the relative path for this snapshot (without supported/unsupported prefix and without ID)
    #[must_use]
    pub fn relative_path(&self) -> PathBuf {
        PathBuf::from(&self.chunk_grid)
            .join(&self.data_type)
            .join(self.category.as_str())
            .join(&self.codec)
    }

    /// Get the full path for a supported snapshot with a checksum ID
    #[must_use]
    pub fn supported_path(&self, base_dir: &Path, checksum: &str) -> PathBuf {
        base_dir
            .join("supported")
            .join(self.relative_path())
            .join(checksum)
    }

    /// Get the full path for an unsupported marker with a checksum ID
    #[must_use]
    pub fn unsupported_path(&self, base_dir: &Path, checksum: &str) -> PathBuf {
        base_dir
            .join("unsupported")
            .join(self.relative_path())
            .join(format!("{checksum}.json"))
    }

    /// Get the full path for a failure marker with a checksum ID
    #[must_use]
    pub fn failure_path(&self, base_dir: &Path, checksum: &str) -> PathBuf {
        base_dir
            .join("failure")
            .join(self.relative_path())
            .join(format!("{checksum}.json"))
    }

    /// Find an existing snapshot by checksum
    /// Returns: Some(SnapshotStatus) if found, None if not found
    #[must_use]
    pub fn find_existing(&self, base_dir: &Path, checksum: &str) -> Option<SnapshotStatus> {
        // Check supported first
        let supported_path = self.supported_path(base_dir, checksum);
        if supported_path.exists() {
            return Some(SnapshotStatus::Supported);
        }
        // Check unsupported
        let unsupported_path = self.unsupported_path(base_dir, checksum);
        if unsupported_path.exists() {
            return Some(SnapshotStatus::Unsupported);
        }
        // Check failure
        let failure_path = self.failure_path(base_dir, checksum);
        if failure_path.exists() {
            return Some(SnapshotStatus::Failure);
        }
        None
    }
}

/// Status of an existing snapshot
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotStatus {
    Supported,
    Unsupported,
    Failure,
}

// =============================================================================
// Test Execution
// =============================================================================

/// Generate array metadata JSON from a `TestConfig` (for use in unsupported markers)
/// Returns None if the metadata cannot be generated (e.g., codec creation fails)
#[must_use]
pub fn generate_array_metadata(config: &TestConfig) -> Option<serde_json::Value> {
    use zarrs::storage::store::MemoryStore;

    let store = Arc::new(MemoryStore::default());

    let mut builder = ArrayBuilder::new(
        config.array_shape.clone(),
        config.chunk_shape.clone(),
        config.data_type.clone(),
        config.fill_value.clone(),
    );

    if !config.array_to_array_codecs.is_empty() {
        builder.array_to_array_codecs(config.array_to_array_codecs.clone());
    }

    if let Some(ref a2b) = config.array_to_bytes_codec {
        builder.array_to_bytes_codec(a2b.clone());
    }

    if !config.bytes_to_bytes_codecs.is_empty() {
        builder.bytes_to_bytes_codecs(config.bytes_to_bytes_codecs.clone());
    }

    let array = builder.build(store, "/").ok()?;
    let metadata_options = ArrayMetadataOptions::default().with_include_zarrs_metadata(false);
    let metadata = array.metadata_opt(&metadata_options);

    serde_json::to_value(metadata).ok()
}

/// Execute a single codec test
#[must_use]
pub fn run_codec_test(config: &TestConfig, output_dir: &Path) -> CodecTestResult {
    // Create filesystem store at output directory
    let store = match FilesystemStore::new(output_dir) {
        Ok(s) => Arc::new(s),
        Err(e) => {
            return CodecTestResult::Unsupported {
                reason: format!("Failed to create store: {e}"),
            };
        }
    };

    // Build array
    let mut builder = ArrayBuilder::new(
        config.array_shape.clone(),
        config.chunk_shape.clone(),
        config.data_type.clone(),
        config.fill_value.clone(),
    );

    // Set codecs (builder methods return &mut Self)
    if !config.array_to_array_codecs.is_empty() {
        builder.array_to_array_codecs(config.array_to_array_codecs.clone());
    }

    if let Some(ref a2b) = config.array_to_bytes_codec {
        builder.array_to_bytes_codec(a2b.clone());
    }

    if !config.bytes_to_bytes_codecs.is_empty() {
        builder.bytes_to_bytes_codecs(config.bytes_to_bytes_codecs.clone());
    }

    // Build the array
    let array = match builder.build(store.clone(), "/") {
        Ok(a) => a,
        Err(e) => {
            return CodecTestResult::Unsupported {
                reason: format!("Array creation failed: {e}"),
            };
        }
    };

    // Store metadata (without zarrs-specific metadata for cleaner snapshots)
    let metadata_options = ArrayMetadataOptions::default().with_include_zarrs_metadata(false);
    if let Err(e) = array.store_metadata_opt(&metadata_options) {
        return CodecTestResult::Unsupported {
            reason: format!("Metadata storage failed: {e}"),
        };
    }

    // Generate and store test data
    let num_elements: usize = config.array_shape.iter().map(|&x| x as usize).product();
    let test_data = generate_test_data(&config.data_type, num_elements);

    // Store data using ArrayBytes API (handles fixed, variable, and optional types)
    let subset = zarrs::array::ArraySubset::new_with_shape(config.array_shape.clone());
    if let Err(e) = array.store_array_subset(&subset, test_data.clone()) {
        return CodecTestResult::Unsupported {
            reason: format!("Data storage failed: {e}"),
        };
    }

    // Round-trip test: read back the data and verify it matches
    let read_data = match array.retrieve_array_subset::<ArrayBytes>(&subset) {
        Ok(data) => data,
        Err(e) => {
            return CodecTestResult::Unsupported {
                reason: format!("Round-trip read failed: {e}"),
            };
        }
    };

    // For lossless codecs, verify the data matches
    // If data doesn't match, this indicates a codec bug or incompatible data type
    if !config.lossy && read_data != test_data {
        return CodecTestResult::Failure {
            reason: format!(
                "Round-trip verification failed: read data does not match written data for {:?}",
                config.data_type
            ),
        };
    }

    CodecTestResult::Success
}

/// Check that a snapshot directory contains chunk files
fn has_chunks(dir: &Path) -> bool {
    let chunks_dir = dir.join("c");
    if !chunks_dir.exists() {
        return false;
    }
    // Recursively check for any files in the chunks directory
    fn has_files(dir: &Path) -> bool {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() || (path.is_dir() && has_files(&path)) {
                    return true;
                }
            }
        }
        false
    }
    has_files(&chunks_dir)
}

/// Compare a generated snapshot directory against a reference
pub fn verify_snapshot(
    generated_dir: &Path,
    reference_dir: &Path,
    skip_chunk_comparison: bool,
) -> Result<(), String> {
    // Check zarr.json
    let gen_meta_path = generated_dir.join("zarr.json");
    let ref_meta_path = reference_dir.join("zarr.json");

    if !ref_meta_path.exists() {
        return Err(format!(
            "Reference snapshot not found: {}",
            reference_dir.display()
        ));
    }

    let gen_meta = fs::read_to_string(&gen_meta_path)
        .map_err(|e| format!("Failed to read generated metadata: {e}"))?;
    let ref_meta = fs::read_to_string(&ref_meta_path)
        .map_err(|e| format!("Failed to read reference metadata: {e}"))?;

    // Parse and compare JSON for better error messages
    let gen_json: serde_json::Value = serde_json::from_str(&gen_meta)
        .map_err(|e| format!("Failed to parse generated metadata: {e}"))?;
    let ref_json: serde_json::Value = serde_json::from_str(&ref_meta)
        .map_err(|e| format!("Failed to parse reference metadata: {e}"))?;

    if gen_json != ref_json {
        return Err(format!(
            "Metadata mismatch:\nGenerated: {}\nReference: {}",
            serde_json::to_string_pretty(&gen_json).unwrap(),
            serde_json::to_string_pretty(&ref_json).unwrap()
        ));
    }

    // Compare chunk files (skip for non-deterministic codecs like sharding)
    if !skip_chunk_comparison {
        let chunks_dir = reference_dir.join("c");
        if chunks_dir.exists() {
            compare_directories(&generated_dir.join("c"), &chunks_dir)?;
        }
    }

    Ok(())
}

/// Recursively compare two directories
fn compare_directories(gen_dir: &Path, ref_dir: &Path) -> Result<(), String> {
    if !ref_dir.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(ref_dir).map_err(|e| format!("Failed to read dir: {e}"))? {
        let entry = entry.map_err(|e| format!("Failed to read entry: {e}"))?;
        let ref_path = entry.path();
        let rel_path = ref_path.strip_prefix(ref_dir).unwrap();
        let gen_path = gen_dir.join(rel_path);

        if ref_path.is_dir() {
            compare_directories(&gen_path, &ref_path)?;
        } else {
            let gen_bytes = fs::read(&gen_path).map_err(|e| {
                format!(
                    "Failed to read generated file {}: {}",
                    gen_path.display(),
                    e
                )
            })?;
            let ref_bytes = fs::read(&ref_path).map_err(|e| {
                format!(
                    "Failed to read reference file {}: {}",
                    ref_path.display(),
                    e
                )
            })?;

            if gen_bytes != ref_bytes {
                return Err(format!(
                    "Chunk mismatch at {}: generated {} bytes, reference {} bytes",
                    rel_path.display(),
                    gen_bytes.len(),
                    ref_bytes.len()
                ));
            }
        }
    }

    Ok(())
}

/// Check if we're in update mode (full updates allowed)
fn update_mode() -> bool {
    std::env::var("UPDATE_SNAPSHOTS").is_ok()
}

/// Check if we're in add mode (only add newly supported snapshots)
fn add_mode() -> bool {
    std::env::var("ADD_SNAPSHOTS").is_ok()
}

/// Run a test and verify/update snapshot using new nested directory structure
pub fn run_and_verify_snapshot_v2(config: &TestConfig, snapshot_path: &SnapshotPath) {
    let snapshots = snapshots_dir();
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let generated_dir = temp_dir.path();

    let result = run_codec_test(config, generated_dir);
    let display_path = snapshot_path.relative_path().display().to_string();

    // Read metadata JSON to compute checksum (if test produced output)
    let metadata_path = generated_dir.join("zarr.json");
    let checksum = if metadata_path.exists() {
        let metadata_json = fs::read_to_string(&metadata_path).expect("Failed to read metadata");
        compute_metadata_checksum(&metadata_json)
    } else {
        // For unsupported tests, compute checksum from config description
        compute_metadata_checksum(&format!(
            "{}:{}:{}",
            config
                .data_type
                .name_v3()
                .map_or_else(String::new, |n| n.to_string()),
            config.chunk_grid_name,
            snapshot_path.codec
        ))
    };

    // Helper to clean up old markers when status changes
    let cleanup_old_markers = |new_status: SnapshotStatus| {
        if new_status != SnapshotStatus::Supported {
            let old_snapshot = snapshot_path.supported_path(&snapshots, &checksum);
            if old_snapshot.exists() {
                fs::remove_dir_all(&old_snapshot).ok();
            }
        }
        if new_status != SnapshotStatus::Unsupported {
            let old_marker = snapshot_path.unsupported_path(&snapshots, &checksum);
            if old_marker.exists() {
                fs::remove_file(&old_marker).ok();
            }
        }
        if new_status != SnapshotStatus::Failure {
            let old_marker = snapshot_path.failure_path(&snapshots, &checksum);
            if old_marker.exists() {
                fs::remove_file(&old_marker).ok();
            }
        }
    };

    match result {
        CodecTestResult::Success => {
            // Verify that chunks were written (test data should never all equal the fill value)
            assert!(
                has_chunks(generated_dir),
                "Snapshot {display_path} has no chunks. This likely indicates a bug in test data generation \
                 where all values equal the fill value."
            );

            let reference_dir = snapshot_path.supported_path(&snapshots, &checksum);

            // Check if we should update snapshots
            if update_mode() {
                cleanup_old_markers(SnapshotStatus::Supported);

                // Update the reference snapshot
                if reference_dir.exists() {
                    fs::remove_dir_all(&reference_dir).expect("Failed to remove old snapshot");
                }
                fs::create_dir_all(reference_dir.parent().unwrap()).ok();
                copy_dir_all(generated_dir, &reference_dir).expect("Failed to copy snapshot");
                println!("Updated snapshot: {display_path}/{checksum}");
            } else {
                // Find existing reference
                match snapshot_path.find_existing(&snapshots, &checksum) {
                    Some(SnapshotStatus::Supported) => {
                        // Verify against reference (skip chunk comparison for non-deterministic codecs)
                        if let Err(e) =
                            verify_snapshot(generated_dir, &reference_dir, config.non_deterministic)
                        {
                            panic!(
                                "Snapshot verification failed for {display_path}/{checksum}/{config:?}: {e}"
                            );
                        }
                    }
                    Some(SnapshotStatus::Unsupported) => {
                        // In add mode, promote unsupported to supported
                        if add_mode() {
                            cleanup_old_markers(SnapshotStatus::Supported);
                            fs::create_dir_all(reference_dir.parent().unwrap()).ok();
                            copy_dir_all(generated_dir, &reference_dir)
                                .expect("Failed to copy snapshot");
                            println!("Added newly supported snapshot: {display_path}/{checksum}");
                        } else {
                            panic!(
                                "Test {} was previously unsupported but now succeeds. Run with ADD_SNAPSHOTS=1 to add or UPDATE_SNAPSHOTS=1 to update.",
                                display_path
                            );
                        }
                    }
                    Some(SnapshotStatus::Failure) => {
                        // In add mode, promote failure to supported
                        if add_mode() {
                            cleanup_old_markers(SnapshotStatus::Supported);
                            fs::create_dir_all(reference_dir.parent().unwrap()).ok();
                            copy_dir_all(generated_dir, &reference_dir)
                                .expect("Failed to copy snapshot");
                            println!(
                                "Added newly supported snapshot (was failure): {display_path}/{checksum}"
                            );
                        } else {
                            panic!(
                                "Test {display_path} was previously a failure but now succeeds. Run with ADD_SNAPSHOTS=1 to add or UPDATE_SNAPSHOTS=1 to update."
                            );
                        }
                    }
                    None => {
                        // No reference exists - this is a new test
                        // In add mode, we also allow adding completely new snapshots
                        if add_mode() {
                            fs::create_dir_all(reference_dir.parent().unwrap()).ok();
                            copy_dir_all(generated_dir, &reference_dir)
                                .expect("Failed to copy snapshot");
                            println!("Added new snapshot: {display_path}/{checksum}");
                        } else {
                            panic!(
                                "No reference snapshot found for {display_path}/{checksum}. Run with ADD_SNAPSHOTS=1 to add or UPDATE_SNAPSHOTS=1 to create it."
                            );
                        }
                    }
                }
            }
        }
        CodecTestResult::Unsupported { reason } => {
            let marker_path = snapshot_path.unsupported_path(&snapshots, &checksum);

            if update_mode() {
                cleanup_old_markers(SnapshotStatus::Unsupported);

                fs::create_dir_all(marker_path.parent().unwrap()).ok();

                // Build marker with full array metadata if possible
                let mut marker = serde_json::json!({
                    "status": "unsupported",
                    "reason": reason,
                });

                // Try to generate array metadata (may fail if codec creation itself failed)
                if let Some(array_metadata) = generate_array_metadata(config) {
                    marker["array_metadata"] = array_metadata;
                }

                fs::write(&marker_path, serde_json::to_string_pretty(&marker).unwrap())
                    .expect("Failed to write unsupported marker");
                println!("Marked as unsupported: {display_path}/{checksum}");
            } else {
                // Check if this was expected to be unsupported
                match snapshot_path.find_existing(&snapshots, &checksum) {
                    Some(SnapshotStatus::Unsupported) => {
                        // Expected to be unsupported, all good
                    }
                    Some(SnapshotStatus::Supported) => {
                        panic!(
                            "Test {display_path} was previously supported but now unsupported: {reason}. Run with UPDATE_SNAPSHOTS=1 to update."
                        );
                    }
                    Some(SnapshotStatus::Failure) => {
                        panic!(
                            "Test {display_path} was previously a failure but now unsupported: {reason}. Run with UPDATE_SNAPSHOTS=1 to update."
                        );
                    }
                    None => {
                        panic!(
                            "Test {display_path} is unsupported ({reason}). Run with UPDATE_SNAPSHOTS=1 to record this."
                        );
                    }
                }
            }
        }
        CodecTestResult::Failure { reason } => {
            let marker_path = snapshot_path.failure_path(&snapshots, &checksum);

            if update_mode() {
                cleanup_old_markers(SnapshotStatus::Failure);

                fs::create_dir_all(marker_path.parent().unwrap()).ok();
                let marker = serde_json::json!({
                    "status": "failure",
                    "reason": reason,
                    "data_type": config.data_type.name_v3().map_or_else(String::new, |n| n.to_string()),
                    "chunk_grid": config.chunk_grid_name,
                });
                fs::write(&marker_path, serde_json::to_string_pretty(&marker).unwrap())
                    .expect("Failed to write failure marker");
                println!("Marked as failure: {display_path}/{checksum}");
            } else {
                // Check if this was expected to be a failure
                match snapshot_path.find_existing(&snapshots, &checksum) {
                    Some(SnapshotStatus::Failure) => {
                        // Expected to be a failure, all good
                    }
                    Some(SnapshotStatus::Supported) => {
                        panic!(
                            "Test {display_path} was previously supported but now fails: {reason}. Run with UPDATE_SNAPSHOTS=1 to update."
                        );
                    }
                    Some(SnapshotStatus::Unsupported) => {
                        panic!(
                            "Test {display_path} was previously unsupported but now fails differently: {reason}. Run with UPDATE_SNAPSHOTS=1 to update."
                        );
                    }
                    None => {
                        panic!(
                            "Test {display_path} failed ({reason}). Run with UPDATE_SNAPSHOTS=1 to record this."
                        );
                    }
                }
            }
        }
    }
}

/// Recursively copy a directory
fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

// =============================================================================
// Codec Registry
// =============================================================================

/// A codec instance that can be any of the three codec categories
pub enum CodecInstance {
    ArrayToArray(Arc<dyn ArrayToArrayCodecTraits>),
    ArrayToBytes(Arc<dyn ArrayToBytesCodecTraits>),
    BytesToBytes(Arc<dyn BytesToBytesCodecTraits>),
}

/// Definition of a codec for testing
pub struct CodecDef {
    /// Codec name for snapshot paths
    pub name: Cow<'static, str>,
    /// Codec category
    pub category: CodecCategory,
    /// Optional name suffix (e.g., "level5", "keepbits10")
    pub name_suffix: Option<&'static str>,
    /// Factory function to create the codec instance for a given data type
    pub factory: fn(&DataType) -> CodecInstance,
    /// Whether the codec is lossy
    pub lossy: bool,
    /// Whether output is non-deterministic (e.g., sharding with parallelism)
    pub non_deterministic: bool,
    /// Optional predicate to skip certain data types (returns true to skip)
    pub skip: Option<fn(&DataType) -> bool>,
}

/// Build the codec registry with all available codecs
fn codec_registry() -> Vec<CodecDef> {
    use zarrs::array::codec::*;
    use zarrs::metadata_ext::codec::fixedscaleoffset::{
        FixedScaleOffsetCodecConfiguration, FixedScaleOffsetCodecConfigurationNumcodecs,
    };
    use zarrs::metadata_ext::codec::reshape::{ReshapeDim, ReshapeShape};

    let mut codecs = Vec::new();

    // =========================================================================
    // Bytes-to-Bytes Codecs
    // =========================================================================

    #[cfg(feature = "gzip")]
    codecs.push(CodecDef {
        name: GzipCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::BytesToBytes,
        name_suffix: Some("level5"),
        factory: |_dt| CodecInstance::BytesToBytes(Arc::new(GzipCodec::new(5).unwrap())),
        lossy: false,
        non_deterministic: false,
        skip: None,
    });

    #[cfg(feature = "zstd")]
    codecs.push(CodecDef {
        name: ZstdCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::BytesToBytes,
        name_suffix: Some("level5"),
        factory: |_dt| CodecInstance::BytesToBytes(Arc::new(ZstdCodec::new(5, false))),
        lossy: false,
        non_deterministic: false,
        skip: None,
    });

    #[cfg(feature = "blosc")]
    codecs.push(CodecDef {
        name: BloscCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::BytesToBytes,
        name_suffix: None,
        factory: |_dt| {
            CodecInstance::BytesToBytes(Arc::new(
                BloscCodec::new(
                    BloscCompressor::BloscLZ,
                    BloscCompressionLevel::try_from(5).unwrap(),
                    None,
                    BloscShuffleMode::NoShuffle,
                    None,
                )
                .unwrap(),
            ))
        },
        lossy: false,
        non_deterministic: false,
        skip: None,
    });

    #[cfg(feature = "crc32c")]
    codecs.push(CodecDef {
        name: Crc32cCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::BytesToBytes,
        name_suffix: None,
        factory: |_dt| CodecInstance::BytesToBytes(Arc::new(Crc32cCodec::new())),
        lossy: false,
        non_deterministic: false,
        skip: None,
    });

    #[cfg(feature = "zlib")]
    codecs.push(CodecDef {
        name: ZlibCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::BytesToBytes,
        name_suffix: Some("level5"),
        factory: |_dt| {
            use zarrs::metadata_ext::codec::zlib::ZlibCompressionLevel;
            CodecInstance::BytesToBytes(Arc::new(ZlibCodec::new(
                ZlibCompressionLevel::try_from(5u32).unwrap(),
            )))
        },
        lossy: false,
        non_deterministic: false,
        skip: None,
    });

    #[cfg(feature = "bz2")]
    codecs.push(CodecDef {
        name: Bz2Codec::aliases_v3().default_name.clone(),
        category: CodecCategory::BytesToBytes,
        name_suffix: Some("level5"),
        factory: |_dt| {
            use zarrs::metadata_ext::codec::bz2::Bz2CompressionLevel;
            CodecInstance::BytesToBytes(Arc::new(Bz2Codec::new(
                Bz2CompressionLevel::try_from(5u32).unwrap(),
            )))
        },
        lossy: false,
        non_deterministic: false,
        skip: None,
    });

    #[cfg(feature = "adler32")]
    codecs.push(CodecDef {
        name: Adler32Codec::aliases_v3().default_name.clone(),
        category: CodecCategory::BytesToBytes,
        name_suffix: None,
        factory: |_dt| CodecInstance::BytesToBytes(Arc::new(Adler32Codec::default())),
        lossy: false,
        non_deterministic: false,
        skip: None,
    });

    #[cfg(feature = "fletcher32")]
    codecs.push(CodecDef {
        name: Fletcher32Codec::aliases_v3().default_name.clone(),
        category: CodecCategory::BytesToBytes,
        name_suffix: None,
        factory: |_dt| CodecInstance::BytesToBytes(Arc::new(Fletcher32Codec::new())),
        lossy: false,
        non_deterministic: false,
        skip: None,
    });

    #[cfg(feature = "gdeflate")]
    codecs.push(CodecDef {
        name: GDeflateCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::BytesToBytes,
        name_suffix: Some("level5"),
        factory: |_dt| CodecInstance::BytesToBytes(Arc::new(GDeflateCodec::new(5).unwrap())),
        lossy: false,
        non_deterministic: false,
        skip: None,
    });

    codecs.push(CodecDef {
        name: ShuffleCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::BytesToBytes,
        name_suffix: None,
        factory: |dt| {
            // Shuffle requires element size - use the data type's fixed size, defaulting to 1
            let elementsize = dt.fixed_size().unwrap_or(1);
            CodecInstance::BytesToBytes(Arc::new(ShuffleCodec::new(elementsize)))
        },
        lossy: false,
        non_deterministic: false,
        // Skip variable-length / optional data types (shuffle requires fixed element size)
        skip: Some(|dt| dt.is_variable() || dt.is_optional()),
    });

    // =========================================================================
    // Array-to-Array Codecs
    // =========================================================================

    #[cfg(feature = "transpose")]
    codecs.push(CodecDef {
        name: TransposeCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToArray,
        name_suffix: None,
        factory: |_dt| {
            CodecInstance::ArrayToArray(Arc::new(TransposeCodec::new(
                TransposeOrder::new(&[1, 0]).unwrap(),
            )))
        },
        lossy: false,
        non_deterministic: false,
        skip: None,
    });

    // Helper: skip for variable-length and optional data types
    fn is_vlen_or_optional(dt: &DataType) -> bool {
        dt.fixed_size().is_none() || dt.as_optional().is_some()
    }

    #[cfg(feature = "bitround")]
    codecs.push(CodecDef {
        name: BitroundCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToArray,
        name_suffix: Some("keepbits10"),
        factory: |_dt| CodecInstance::ArrayToArray(Arc::new(BitroundCodec::new(10))),
        lossy: true,
        non_deterministic: false,
        skip: Some(is_vlen_or_optional),
    });

    codecs.push(CodecDef {
        name: SqueezeCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToArray,
        name_suffix: None,
        factory: |_dt| CodecInstance::ArrayToArray(Arc::new(SqueezeCodec::new())),
        lossy: false,
        non_deterministic: false,
        skip: None,
    });

    codecs.push(CodecDef {
        name: ReshapeCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToArray,
        name_suffix: Some("flatten"),
        factory: |_dt| {
            // Reshape from [4, 6] (24 elements) to [24] (flatten)
            let shape =
                ReshapeShape::new(vec![ReshapeDim::Size(NonZeroU64::new(24).unwrap())]).unwrap();
            CodecInstance::ArrayToArray(Arc::new(ReshapeCodec::new(shape)))
        },
        lossy: false,
        non_deterministic: false,
        skip: None,
    });

    codecs.push(CodecDef {
        name: FixedScaleOffsetCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToArray,
        name_suffix: None,
        factory: |dt| {
            // fixedscaleoffset requires a dtype configuration - use a sensible default
            // based on the data type, or fall back to f64 for unsupported types
            let dtype_str = match data_type_id(dt) {
                "int16" => "<i2",
                "int32" => "<i4",
                "int64" => "<i8",
                "uint16" => "<u2",
                "uint32" => "<u4",
                "uint64" => "<u8",
                "float32" => "<f4",
                "float64" => "<f8",
                // For unsupported types, use f64 as a fallback - the codec will fail at runtime
                _ => "<f8",
            };
            CodecInstance::ArrayToArray(Arc::new(
                FixedScaleOffsetCodec::new_with_configuration(
                    &FixedScaleOffsetCodecConfiguration::Numcodecs(
                        FixedScaleOffsetCodecConfigurationNumcodecs {
                            offset: 0.0,
                            scale: 1.0,
                            dtype: dtype_str.to_string(),
                            astype: None,
                        },
                    ),
                )
                .unwrap(),
            ))
        },
        // Float types may have precision loss
        lossy: true,
        non_deterministic: false,
        skip: Some(is_vlen_or_optional),
    });

    // =========================================================================
    // Array-to-Bytes Codecs
    // =========================================================================

    codecs.push(CodecDef {
        name: BytesCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToBytes,
        name_suffix: None,
        factory: |_dt| CodecInstance::ArrayToBytes(Arc::new(BytesCodec::default())),
        lossy: false,
        non_deterministic: false,
        skip: Some(is_vlen_or_optional),
    });

    // Helper: skip for optional data types
    fn is_optional(dt: &DataType) -> bool {
        dt.as_optional().is_some()
    }

    #[cfg(feature = "sharding")]
    codecs.push(CodecDef {
        name: ShardingCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToBytes,
        name_suffix: Some("inner2x2"),
        factory: |dt| {
            CodecInstance::ArrayToBytes(Arc::new(
                ShardingCodecBuilder::new(vec![NonZeroU64::new(2u64).unwrap(); 2], dt).build(),
            ))
        },
        lossy: false,
        non_deterministic: false, // Using single-threaded pool makes it deterministic
        skip: Some(is_optional),
    });

    #[cfg(feature = "pcodec")]
    codecs.push(CodecDef {
        name: PcodecCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToBytes,
        name_suffix: None,
        factory: |_dt| {
            use zarrs::metadata_ext::codec::pcodec::PcodecCodecConfiguration;
            CodecInstance::ArrayToBytes(Arc::new(
                PcodecCodec::new_with_configuration(&PcodecCodecConfiguration::default()).unwrap(),
            ))
        },
        lossy: false,
        non_deterministic: false,
        skip: Some(is_vlen_or_optional),
    });

    #[cfg(feature = "zfp")]
    codecs.push(CodecDef {
        name: ZfpCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToBytes,
        name_suffix: Some("reversible"),
        factory: |_dt| CodecInstance::ArrayToBytes(Arc::new(ZfpCodec::new_reversible())),
        lossy: false,
        non_deterministic: false,
        skip: Some(is_vlen_or_optional),
    });

    // Helper: skip vlen codecs for fixed-length or optional data types
    fn is_fixed_length_or_optional(dt: &DataType) -> bool {
        dt.fixed_size().is_some() || dt.as_optional().is_some()
    }

    codecs.push(CodecDef {
        name: VlenCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToBytes,
        name_suffix: None,
        factory: |_dt| CodecInstance::ArrayToBytes(Arc::new(VlenCodec::default())),
        lossy: false,
        non_deterministic: false,
        skip: Some(is_fixed_length_or_optional),
    });

    codecs.push(CodecDef {
        name: VlenV2Codec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToBytes,
        name_suffix: None,
        factory: |_dt| CodecInstance::ArrayToBytes(Arc::new(VlenV2Codec::new())),
        lossy: false,
        non_deterministic: false,
        skip: Some(is_fixed_length_or_optional),
    });

    codecs.push(CodecDef {
        name: VlenArrayCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToBytes,
        name_suffix: None,
        factory: |_dt| CodecInstance::ArrayToBytes(Arc::new(VlenArrayCodec::new())),
        lossy: false,
        non_deterministic: false,
        skip: Some(is_fixed_length_or_optional),
    });

    codecs.push(CodecDef {
        name: VlenBytesCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToBytes,
        name_suffix: None,
        factory: |_dt| CodecInstance::ArrayToBytes(Arc::new(VlenBytesCodec::new())),
        lossy: false,
        non_deterministic: false,
        skip: Some(is_fixed_length_or_optional),
    });

    codecs.push(CodecDef {
        name: VlenUtf8Codec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToBytes,
        name_suffix: None,
        factory: |_dt| CodecInstance::ArrayToBytes(Arc::new(VlenUtf8Codec::new())),
        lossy: false,
        non_deterministic: false,
        skip: Some(is_fixed_length_or_optional),
    });

    codecs.push(CodecDef {
        name: PackBitsCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToBytes,
        name_suffix: None,
        factory: |_dt| CodecInstance::ArrayToBytes(Arc::new(PackBitsCodec::default())),
        lossy: false,
        non_deterministic: false,
        skip: Some(is_vlen_or_optional),
    });

    // Helper: skip optional codec for non-optional data types
    fn is_not_optional(dt: &DataType) -> bool {
        dt.as_optional().is_none()
    }

    codecs.push(CodecDef {
        name: OptionalCodec::aliases_v3().default_name.clone(),
        category: CodecCategory::ArrayToBytes,
        name_suffix: None,
        factory: |dt| {
            // For optional types, default_array_to_bytes_codec returns an OptionalCodec
            CodecInstance::ArrayToBytes(default_array_to_bytes_codec(dt))
        },
        lossy: false,
        non_deterministic: false,
        skip: Some(is_not_optional),
    });

    codecs
}

/// Build a `TestConfig` from a codec definition and instance
fn build_test_config(
    codec: &CodecDef,
    instance: CodecInstance,
    data_type: &DataType,
    fill_value: &FillValue,
) -> TestConfig {
    let mut config = TestConfig {
        data_type: data_type.clone(),
        fill_value: fill_value.clone(),
        lossy: codec.lossy,
        non_deterministic: codec.non_deterministic,
        ..Default::default()
    };

    match instance {
        CodecInstance::ArrayToArray(a2a) => {
            config.array_to_array_codecs = vec![a2a];
            // Use default a2b for this data type (None lets ArrayBuilder choose)
            // Use empty b2b
        }
        CodecInstance::ArrayToBytes(a2b) => {
            config.array_to_bytes_codec = Some(a2b);
            // Use empty a2a and b2b
        }
        CodecInstance::BytesToBytes(b2b) => {
            config.bytes_to_bytes_codecs = vec![b2b];
            // Use empty a2a and default a2b
        }
    }

    config
}

// =============================================================================
// Main Test
// =============================================================================

fn run_all_codec_datatype_combinations() {
    // Use a single-threaded rayon pool for deterministic sharding output
    let pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("Failed to build single-threaded pool");

    pool.install(|| {
        let data_types = all_data_types();
        let codecs = codec_registry();

        for codec in &codecs {
            for (data_type, fill_value, _fill_desc) in &data_types {
                // Skip this codec/data type combination if the skip predicate returns true
                if let Some(skip_fn) = codec.skip
                    && skip_fn(data_type)
                {
                    continue;
                }

                let instance = (codec.factory)(data_type);
                let config = build_test_config(codec, instance, data_type, fill_value);

                let snapshot_path = SnapshotPath::new(
                    "regular",
                    data_type,
                    codec.category,
                    &codec.name,
                    codec.name_suffix,
                );

                run_and_verify_snapshot_v2(&config, &snapshot_path);
            }
        }
    });
}

// =============================================================================
// Compatibility Matrix Generation
// =============================================================================

mod compatibility_matrix {
    use std::borrow::Cow;
    use std::collections::{BTreeMap, BTreeSet};
    use std::fs;
    use std::path::{Path, PathBuf};

    use zarrs::array::codec;
    use zarrs_plugin::ExtensionAliasesV3;

    /// Get codecs for a specific category from `registered_codecs`
    fn get_codecs_for_category(category: &str) -> Vec<String> {
        registered_codecs()
            .iter()
            .filter(|(_, cat)| *cat == category)
            .map(|(id, _)| id.to_string())
            .collect()
    }

    /// Sanitize a data type name to match directory naming convention
    fn sanitize_data_type_name(name: &str) -> String {
        name.chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c == '.' {
                    c.to_ascii_lowercase()
                } else {
                    '_'
                }
            })
            .collect()
    }

    /// Get all registered data types as sanitized names for table rows
    /// Excludes parameterized types like `r*` and `optional` which are tested via
    /// specific instances (e.g., `r24`, `optional_float32`)
    fn get_registered_data_types_for_table() -> BTreeSet<String> {
        REGISTERED_DATA_TYPES
            .iter()
            .filter(|dt| {
                // Exclude parameterized types - these are tested via specific instances
                **dt != "raw_bits" && **dt != "optional"
            })
            .map(|dt| sanitize_data_type_name(dt))
            .collect()
    }

    /// Collect all data types that have been tested (appear in any snapshot directory)
    fn collect_tested_data_types(snapshots_dir: &Path) -> BTreeSet<String> {
        let mut tested = BTreeSet::new();

        for status in ["supported", "unsupported", "failure"] {
            let base_dir = snapshots_dir.join(status);
            if !base_dir.exists() {
                continue;
            }

            // Iterate chunk grid directories
            let Ok(chunk_grids) = fs::read_dir(&base_dir) else {
                continue;
            };

            for chunk_grid_entry in chunk_grids.flatten() {
                if !chunk_grid_entry.path().is_dir() {
                    continue;
                }

                // Iterate data type directories
                let Ok(data_types) = fs::read_dir(chunk_grid_entry.path()) else {
                    continue;
                };

                for dt_entry in data_types.flatten() {
                    if dt_entry.path().is_dir() {
                        tested.insert(dt_entry.file_name().to_string_lossy().to_string());
                    }
                }
            }
        }

        tested
    }

    /// Get all data types for the compatibility table:
    /// - All tested data types from snapshots
    /// - All registered data types (except parameterized ones like r* and optional)
    fn get_all_data_types_for_table(snapshots_dir: &Path) -> BTreeSet<String> {
        let mut all_types = collect_tested_data_types(snapshots_dir);
        all_types.extend(get_registered_data_types_for_table());
        all_types
    }

    /// Map a directory name to the canonical data type name for display
    /// Directory names are already in canonical form (e.g., "zarrs.optional<float32>")
    fn canonical_data_type_name(name: &str) -> String {
        name.to_string()
    }

    /// All registered codecs with their categories
    fn registered_codecs() -> Vec<(Cow<'static, str>, &'static str)> {
        vec![
            // Array-to-Array
            (
                codec::BitroundCodec::aliases_v3().default_name.clone(),
                "a2a",
            ),
            (
                codec::FixedScaleOffsetCodec::aliases_v3()
                    .default_name
                    .clone(),
                "a2a",
            ),
            (
                codec::ReshapeCodec::aliases_v3().default_name.clone(),
                "a2a",
            ),
            (
                codec::SqueezeCodec::aliases_v3().default_name.clone(),
                "a2a",
            ),
            (
                codec::TransposeCodec::aliases_v3().default_name.clone(),
                "a2a",
            ),
            // Array-to-Bytes
            (codec::BytesCodec::aliases_v3().default_name.clone(), "a2b"),
            (
                codec::OptionalCodec::aliases_v3().default_name.clone(),
                "a2b",
            ),
            (
                codec::PackBitsCodec::aliases_v3().default_name.clone(),
                "a2b",
            ),
            (codec::PcodecCodec::aliases_v3().default_name.clone(), "a2b"),
            (
                codec::ShardingCodec::aliases_v3().default_name.clone(),
                "a2b",
            ),
            (codec::VlenCodec::aliases_v3().default_name.clone(), "a2b"),
            (
                codec::VlenArrayCodec::aliases_v3().default_name.clone(),
                "a2b",
            ),
            (
                codec::VlenBytesCodec::aliases_v3().default_name.clone(),
                "a2b",
            ),
            (
                codec::VlenUtf8Codec::aliases_v3().default_name.clone(),
                "a2b",
            ),
            (codec::VlenV2Codec::aliases_v3().default_name.clone(), "a2b"),
            (codec::ZfpCodec::aliases_v3().default_name.clone(), "a2b"),
            // (ZfpyCodec::aliases_v3().default_name.clone(), "a2b"),
            // Bytes-to-Bytes
            (
                codec::Adler32Codec::aliases_v3().default_name.clone(),
                "b2b",
            ),
            (codec::BloscCodec::aliases_v3().default_name.clone(), "b2b"),
            (codec::Bz2Codec::aliases_v3().default_name.clone(), "b2b"),
            (codec::Crc32cCodec::aliases_v3().default_name.clone(), "b2b"),
            (
                codec::Fletcher32Codec::aliases_v3().default_name.clone(),
                "b2b",
            ),
            (
                codec::GDeflateCodec::aliases_v3().default_name.clone(),
                "b2b",
            ),
            (codec::GzipCodec::aliases_v3().default_name.clone(), "b2b"),
            (
                codec::ShuffleCodec::aliases_v3().default_name.clone(),
                "b2b",
            ),
            (codec::ZlibCodec::aliases_v3().default_name.clone(), "b2b"),
            (codec::ZstdCodec::aliases_v3().default_name.clone(), "b2b"),
        ]
    }

    /// All registered data types from `zarrs_registry::data_type`
    #[rustfmt::skip]
    const REGISTERED_DATA_TYPES: &[&str] = &[
        "bool",
        "int2",
        "int4",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint2",
        "uint4",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float4_e2m1fn",
        "float6_e2m3fn",
        "float6_e3m2fn",
        "float8_e3m4",
        "float8_e4m3",
        "float8_e4m3b11fnuz",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "raw_bits",
        "bfloat16",
        "complex_bfloat16",
        "complex_float16",
        "complex_float32",
        "complex_float64",
        "complex_float4_e2m1fn",
        "complex_float6_e2m3fn",
        "complex_float6_e3m2fn",
        "complex_float8_e3m4",
        "complex_float8_e4m3",
        "complex_float8_e4m3b11fnuz",
        "complex_float8_e4m3fnuz",
        "complex_float8_e5m2",
        "complex_float8_e5m2fnuz",
        "complex_float8_e8m0fnu",
        "string",
        "bytes",
        "numpy.datetime64",
        "numpy.timedelta64",
        "optional",
    ];

    /// Extract the base codec name from a directory name like "gzip(level5)" -> "gzip"
    fn extract_codec_name(dir_name: &str) -> String {
        // Strip any parenthesized suffix (e.g., "gzip(level5)" -> "gzip")
        dir_name
            .split_once('(')
            .map_or(dir_name, |(base, _)| base)
            .to_string()
    }

    /// Scan a specific `chunk_grid` directory: <`data_type`>/<category>/<codec>/<checksum>
    /// Returns: codec -> set of data_types
    fn scan_chunk_grid_dir(chunk_grid_dir: &Path) -> BTreeMap<String, BTreeSet<String>> {
        let mut results: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

        let Ok(data_types) = fs::read_dir(chunk_grid_dir) else {
            return results;
        };

        for data_type_entry in data_types.flatten() {
            if !data_type_entry.path().is_dir() {
                continue;
            }
            let data_type = data_type_entry.file_name().to_string_lossy().to_string();

            // Iterate category directories (a2a, a2b, b2b)
            let Ok(categories) = fs::read_dir(data_type_entry.path()) else {
                continue;
            };

            for category_entry in categories.flatten() {
                if !category_entry.path().is_dir() {
                    continue;
                }

                // Iterate codec directories
                let Ok(codecs) = fs::read_dir(category_entry.path()) else {
                    continue;
                };

                for codec_entry in codecs.flatten() {
                    if !codec_entry.path().is_dir() {
                        continue;
                    }
                    let codec_full = codec_entry.file_name().to_string_lossy().to_string();
                    // Extract base codec name, handling codecs with underscores in their names
                    let codec = extract_codec_name(&codec_full);

                    // Check if there are any checksum subdirectories/files
                    if let Ok(checksums) = fs::read_dir(codec_entry.path())
                        && checksums.flatten().next().is_some()
                    {
                        results.entry(codec).or_default().insert(data_type.clone());
                    }
                }
            }
        }

        results
    }

    /// Generate a markdown table for a specific codec category
    /// Shows all registered codecs and data types, with "-" for untested combinations
    fn generate_category_table(
        title: &str,
        codec_list: &[String],
        supported: &BTreeMap<String, BTreeSet<String>>,
        unsupported: &BTreeMap<String, BTreeSet<String>>,
        failure: &BTreeMap<String, BTreeSet<String>>,
        all_datatypes: &BTreeSet<String>,
    ) -> String {
        let mut output = String::new();

        if codec_list.is_empty() {
            return output;
        }

        output.push_str(&format!("## {title}\n\n"));

        // Header row - include all codecs in the category with default names
        output.push_str("| Data Type |");
        for codec in codec_list {
            output.push_str(&format!(" {codec} |"));
        }
        output.push('\n');

        // Separator row
        output.push_str("|-----------|");
        for _ in codec_list {
            output.push_str("---|");
        }
        output.push('\n');

        // Data rows - include all registered data types
        for datatype in all_datatypes {
            let display_name = canonical_data_type_name(datatype);
            output.push_str(&format!("| {display_name} |"));
            for codec in codec_list {
                // Look up using both the identifier and common variations
                let codec_underscore = codec.replace('-', "_");
                let codec_hyphen = codec.replace('_', "-");
                let codec_base = codec.trim_end_matches("_indexed");

                let lookup = |map: &BTreeMap<String, BTreeSet<String>>| -> bool {
                    map.get(codec)
                        .or_else(|| map.get(&codec_underscore))
                        .or_else(|| map.get(&codec_hyphen))
                        .or_else(|| map.get(codec_base))
                        .is_some_and(|dt| dt.contains(datatype))
                };

                let is_supported = lookup(supported);
                let is_unsupported = lookup(unsupported);
                let is_failure = lookup(failure);

                let symbol = if is_supported {
                    "✓"
                } else if is_failure {
                    "💥"
                } else if is_unsupported {
                    "✗"
                } else {
                    "-"
                };
                output.push_str(&format!(" {symbol} |"));
            }
            output.push('\n');
        }
        output.push('\n');

        output
    }

    /// Generate the full compatibility matrix markdown
    fn generate_matrix() -> String {
        let snapshots_dir = super::snapshots_dir();

        // Only scan "regular" chunk grid for standard codec tests
        let supported_regular = snapshots_dir.join("supported").join("regular");
        let unsupported_regular = snapshots_dir.join("unsupported").join("regular");
        let failure_regular = snapshots_dir.join("failure").join("regular");

        let supported = scan_chunk_grid_dir(&supported_regular);
        let unsupported = scan_chunk_grid_dir(&unsupported_regular);
        let failure = scan_chunk_grid_dir(&failure_regular);

        // Combine tested data types with registered data types (excluding parameterized types)
        let all_datatypes = get_all_data_types_for_table(&snapshots_dir);

        // Get all registered codecs by category
        let a2a_codecs = get_codecs_for_category("a2a");
        let a2b_codecs = get_codecs_for_category("a2b");
        let b2b_codecs = get_codecs_for_category("b2b");

        let mut output = String::new();
        output.push_str("# Codec & Data Type Compatibility Matrix\n\n");

        // Description of how compatibility is evaluated
        output.push_str("## How Compatibility is Evaluated\n\n");
        output.push_str("Each codec/data type combination is tested by:\n");
        output.push_str(
            "1. Creating a small test array with representative values for the data type\n",
        );
        output.push_str(
            "2. Encoding the array using the codec, using default codecs for the data type where needed\n",
        );
        output.push_str("3. Decoding the encoded data back to an array\n");
        output
            .push_str("4. Verifying the decoded array matches the original (round-trip test)\n\n");
        output.push_str("Results:\n");
        output.push_str(
            "- **✓ supported**: The codec successfully encodes and decodes the data type\n",
        );
        output
            .push_str("- **✗ unsupported**: The codec explicitly does not support the data type\n");
        output.push_str(
            "- **💥 failure**: The codec claims support but the round-trip test failed\n",
        );
        output.push_str("- **- not tested**: The combination has not been tested\n\n");

        // Table of Contents
        output.push_str("## Contents\n\n");
        output.push_str("- [Array-to-Array Codecs](#array-to-array-codecs)\n");
        output.push_str("- [Array-to-Bytes Codecs](#array-to-bytes-codecs)\n");
        output.push_str("- [Bytes-to-Bytes Codecs](#bytes-to-bytes-codecs)\n\n");

        output.push_str("---\n\n");

        // Generate compatibility tables by category (includes all registered codecs and data types)
        output.push_str(&generate_category_table(
            "Array-to-Array Codecs",
            &a2a_codecs,
            &supported,
            &unsupported,
            &failure,
            &all_datatypes,
        ));

        output.push_str(&generate_category_table(
            "Array-to-Bytes Codecs",
            &a2b_codecs,
            &supported,
            &unsupported,
            &failure,
            &all_datatypes,
        ));

        output.push_str(&generate_category_table(
            "Bytes-to-Bytes Codecs",
            &b2b_codecs,
            &supported,
            &unsupported,
            &failure,
            &all_datatypes,
        ));

        output
    }

    pub(crate) fn run_generate_compatibility_matrix() {
        let matrix = generate_matrix();
        // Write to doc/DATA_TYPE_AND_CODEC_COMPATIBILITY.md in the workspace root
        let output_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("doc")
            .join("DATA_TYPE_AND_CODEC_COMPATIBILITY.md");
        fs::write(&output_path, &matrix).expect("Failed to write matrix");
        println!("Generated: {}", output_path.display());
        println!("\n{matrix}");
    }
}

// =============================================================================
// Combined Test Entry Point
// =============================================================================

/// Check if snapshot data is available (submodule initialized)
fn snapshots_available() -> bool {
    let dir = snapshots_dir();
    // Check if the directory exists and contains at least one subdirectory
    // (empty directory from uninitialized submodule won't have content)
    if !dir.exists() {
        return false;
    }
    if let Ok(mut entries) = fs::read_dir(&dir) {
        return entries.next().is_some();
    }
    false
}

#[test]
fn codec_snapshot_tests() {
    if !snapshots_available() {
        eprintln!("WARNING: Snapshot directory not found or empty. Skipping snapshot tests.");
        eprintln!("To run snapshot tests, initialize the submodule:");
        eprintln!("  git submodule update --init zarrs/tests/data/snapshots");
        return;
    }

    // Run all standard codec/datatype combinations
    run_all_codec_datatype_combinations();

    // Generate the compatibility matrix
    compatibility_matrix::run_generate_compatibility_matrix();
}
