//! The machine-readable interface of `zarrs_regression_testing`: the helper protocol, the case manifest, and results.
//!
//! These types only depend on `serde`, so other tools (e.g. [`zarr_compatibility`](https://github.com/zarrs/zarr_compatibility)) can use them without building `zarrs`.
//! Breaking changes to them (or to their JSON) are breaking changes of this crate, and [`SCHEMA`] is incremented.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The version of the manifest and results JSON.
pub const SCHEMA: u32 = 1;

/// Array data: element bytes, element offsets if variable length, and validity masks (outermost first) if optional.
///
/// Multi-byte elements are little-endian.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Data {
    /// The bytes of the elements.
    pub bytes: Vec<u8>,
    /// The offsets of each element in `bytes` (and the end), if variable length.
    pub offsets: Option<Vec<usize>>,
    /// The validity mask of each optional level (`0` if null), outermost first.
    pub masks: Vec<Vec<u8>>,
}

impl Data {
    /// Create data from its elements, with offsets if `variable`.
    #[must_use]
    pub fn from_elements(elements: &[Vec<u8>], variable: bool, masks: Vec<Vec<u8>>) -> Self {
        let offsets = variable.then(|| {
            std::iter::once(0)
                .chain(elements.iter().scan(0, |offset, element| {
                    *offset += element.len();
                    Some(*offset)
                }))
                .collect()
        });
        Self {
            bytes: elements.concat(),
            offsets,
            masks,
        }
    }

    /// Zero the elements (and inner masks) of null elements, as their content is unspecified.
    ///
    /// `element_size` is the size of an element in bytes, or [`None`] if variable length.
    #[must_use]
    pub fn canonical(&self, element_size: Option<usize>) -> Self {
        let Some(num_elements) = self.masks.first().map(Vec::len) else {
            return self.clone();
        };
        let mut masks = self.masks.clone();
        let elements = (0..num_elements).map(|index| {
            let element = match (&self.offsets, element_size) {
                (Some(offsets), _) => self.bytes.get(offsets[index]..offsets[index + 1]),
                (None, Some(size)) => self.bytes.get(index * size..(index + 1) * size),
                (None, None) => None,
            };
            let element = element.unwrap_or_default().to_vec();
            if let Some(level) = masks.iter().position(|mask| mask[index] == 0) {
                for mask in &mut masks[level..] {
                    mask[index] = 0;
                }
                vec![0; element_size.unwrap_or(0)]
            } else {
                element
            }
        });
        let elements: Vec<_> = elements.collect();
        Self::from_elements(&elements, self.offsets.is_some(), masks)
    }

    /// Whether this data equals `other`, or approximately if `numeric` (with elements of `element_size` bytes).
    ///
    /// Lossy data decoded by different implementations may differ slightly, so integer elements may differ by one and float elements by a relative 1e-6.
    #[must_use]
    pub fn approx_eq(
        &self,
        other: &Self,
        numeric: Option<Numeric>,
        element_size: Option<usize>,
    ) -> bool {
        if self == other {
            return true;
        }
        let (Some(numeric), Some(size)) = (numeric, element_size) else {
            return false;
        };
        if self.offsets != other.offsets
            || self.masks != other.masks
            || self.bytes.len() != other.bytes.len()
        {
            return false;
        }
        self.bytes
            .chunks(size)
            .zip(other.bytes.chunks(size))
            .all(|(a, b)| numeric.approx_eq(a, b))
    }
}

/// The kind of the elements of a numeric data type (with little-endian elements of 1, 2, 4 or 8 bytes, 4 or 8 if float).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Numeric {
    /// A signed integer.
    Int,
    /// An unsigned integer.
    UInt,
    /// An IEEE 754 float.
    Float,
}

impl Numeric {
    /// Whether the elements `a` and `b` are approximately equal (see [`Data::approx_eq`]).
    #[must_use]
    pub fn approx_eq(self, a: &[u8], b: &[u8]) -> bool {
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
        let value = |bytes: &[u8]| -> Option<f64> {
            Some(match (self, bytes.len()) {
                (Self::Int, 1) => f64::from(bytes[0] as i8),
                (Self::Int, 2) => f64::from(i16::from_le_bytes(bytes.try_into().ok()?)),
                (Self::Int, 4) => f64::from(i32::from_le_bytes(bytes.try_into().ok()?)),
                (Self::Int, 8) => i64::from_le_bytes(bytes.try_into().ok()?) as f64,
                (Self::UInt, 1) => f64::from(bytes[0]),
                (Self::UInt, 2) => f64::from(u16::from_le_bytes(bytes.try_into().ok()?)),
                (Self::UInt, 4) => f64::from(u32::from_le_bytes(bytes.try_into().ok()?)),
                (Self::UInt, 8) => u64::from_le_bytes(bytes.try_into().ok()?) as f64,
                (Self::Float, 4) => f64::from(f32::from_le_bytes(bytes.try_into().ok()?)),
                (Self::Float, 8) => f64::from_le_bytes(bytes.try_into().ok()?),
                _ => return None,
            })
        };
        match (value(a), value(b)) {
            _ if a == b => true,
            (Some(a), Some(b)) if self == Self::Float => {
                (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-6 * a.abs().max(b.abs())
            }
            (Some(a), Some(b)) => (a - b).abs() <= 1.0,
            _ => false,
        }
    }
}

/// A request to an implementation adapter.
///
/// An adapter reads a JSON array of requests from stdin and writes a JSON array of [`Response`]s (one per request) to stdout.
/// Arrays are at `/array` in the filesystem store at `path`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum Request {
    /// Identify the implementation, responded to with [`Response::Info`].
    Info,
    /// Create an array with `metadata` and write `data` to the whole array.
    Write {
        /// The filesystem store.
        path: PathBuf,
        /// The array metadata (`zarr.json`).
        metadata: Value,
        /// The shape of the array.
        shape: Vec<u64>,
        /// The data of the whole array.
        data: Data,
    },
    /// Read the whole array.
    Read {
        /// The filesystem store.
        path: PathBuf,
        /// The shape of the array.
        shape: Vec<u64>,
    },
}

/// A response from an implementation adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Response {
    /// Success, with the data read for a read request.
    Ok(Option<Data>),
    /// The implementation failed (e.g. it does not support the array metadata).
    Err(String),
    /// The adapter cannot express the request with the implementation (e.g. a data type with no equivalent in the adapter's language), so the implementation is not tested.
    Unsupported(String),
    /// The response to [`Request::Info`].
    Info(Info),
}

/// The identity of an implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Info {
    /// The implementation, e.g. `zarrs` or `zarr-python`.
    pub implementation: String,
    /// The version of the implementation, e.g. `0.23` or `3.1.3`.
    pub version: String,
    /// Anything else about the adapter or implementation, e.g. how arrays are created.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

/// What generated a manifest or results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generator {
    /// The name of the generator, e.g. `zarrs_regression_testing`.
    pub name: String,
    /// The version of the generator.
    pub version: String,
    /// The repository of the generator, e.g. `https://github.com/zarrs/zarrs`.
    pub repository: String,
    /// The commit of the generator (if known).
    pub commit: Option<String>,
    /// Whether the generator has changes not committed.
    pub modified: bool,
}

/// The parameters of a manifest or results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    /// What generated the cases.
    pub generator: Generator,
    /// The date the cases were run (UTC), e.g. `2026-10-08`.
    pub date: String,
    /// The seed for sampling cases.
    pub seed: u64,
    /// The number of samples of each combination.
    pub samples: usize,
    /// Only combinations whose codec or data type contains this string.
    pub filter: Option<String>,
    /// A command that reproduces the results.
    pub reproduce: String,
}

/// A tested codec kind (e.g. `blosc(lz4,shuffle)`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Codec {
    /// The label of the codec kind.
    pub label: String,
    /// The `zarrs` release that introduced the codec (e.g. `0.20` or `0.20.1`), or [`None`] if not yet released.
    pub introduced: Option<String>,
    /// A note on when the codec was introduced.
    pub note: Option<String>,
}

/// A tested data type (with its fill value, e.g. `float32(fill=NaN)`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataType {
    /// The label of the data type.
    pub label: String,
    /// The `zarrs` release that introduced the data type (e.g. `0.20` or `0.20.1`), or [`None`] if not yet released.
    pub introduced: Option<String>,
    /// A note on when the data type was introduced.
    pub note: Option<String>,
    /// The size of an element in bytes, or [`None`] if variable length.
    pub element_size: Option<usize>,
    /// The kind of its elements if numeric (for comparing lossy data, see [`Data::approx_eq`]).
    #[serde(default)]
    pub numeric: Option<Numeric>,
}

/// A fully swept combination of a codec kind and data type (indices into [`Manifest::codecs`] and [`Manifest::data_types`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Combination {
    /// The index of the codec.
    pub codec: usize,
    /// The index of the data type.
    pub data_type: usize,
}

/// A test case: a sample of a [`Combination`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    /// The index of the combination in [`Manifest::combinations`].
    pub combination: usize,
    /// The shape of the array.
    pub shape: Vec<u64>,
    /// The (outer) chunk shape.
    pub chunk_shape: Vec<u64>,
    /// The array metadata.
    pub metadata: Value,
    /// The data of the whole array.
    pub data: Data,
    /// The elements of `data` formatted as in fill value metadata (e.g. `-1`, `1.5`, `NaN`, `"text"`, `[null]`), or in hex if raw.
    pub elements: Vec<String>,
    /// Whether the codecs are lossy, so data read is compared approximately against the writer's own decoding (see [`Data::approx_eq`]).
    pub lossy: bool,
}

impl Case {
    /// A short description of the case: its codecs, shape and chunk shape.
    #[must_use]
    pub fn describe(&self) -> String {
        describe(&self.metadata, &self.shape, &self.chunk_shape)
    }
}

/// A short description of a case with array `metadata`, `shape` and `chunk_shape` (see [`Case::describe`]).
#[must_use]
pub fn describe(metadata: &Value, shape: &[u64], chunk_shape: &[u64]) -> String {
    let codecs = metadata["codecs"]
        .as_array()
        .map(|codecs| {
            codecs
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    format!("shape={shape:?} chunks={chunk_shape:?} codecs=[{codecs}]")
}

/// The cases of a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// [`SCHEMA`].
    pub schema: u32,
    /// The parameters of the run.
    pub meta: Meta,
    /// The tested codec kinds.
    pub codecs: Vec<Codec>,
    /// The tested data types.
    pub data_types: Vec<DataType>,
    /// The tested combinations of codec kind and data type.
    pub combinations: Vec<Combination>,
    /// The cases: samples of each combination.
    pub cases: Vec<Case>,
}

/// An implementation at a version that cases were run with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Subject {
    /// The implementation, e.g. `zarrs`.
    pub implementation: String,
    /// The version of the implementation, e.g. `0.23`.
    pub version: String,
    /// A short label, e.g. `current` or `0.23`.
    pub label: String,
    /// Whether this is the reference subject that the others are compared against (e.g. the current `zarrs`).
    pub reference: bool,
}

/// Non-conformances (see [`Outcomes::non_conformances`]) of data written by a subject that also make it reject the conformant equivalent.
///
/// A subject failing to read conformant data of a case is a known issue rather than a regression if it wrote the case with one of these non-conformances.
pub const REJECTS_CONFORMANT: &[&str] = &[FLETCHER32_ODD_LENGTH];

/// The non-conformance of `numcodecs.fletcher32` checksums that omit the last byte of data with an odd length (zarrs 0.19 to 0.23), which also makes a subject reject conformant checksums (see [`REJECTS_CONFORMANT`]).
pub const FLETCHER32_ODD_LENGTH: &str =
    "`numcodecs.fletcher32` checksums omit the last byte of data with an odd length";

/// The outcome of a subject writing a case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Write {
    /// The case was written.
    Ok,
    /// The implementation could not write the case.
    Error(String),
    /// The adapter cannot express the case (see [`Response::Unsupported`]).
    Unsupported(String),
}

/// The outcome of a subject reading data written by a subject.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// The data was read and matched the expected data.
    Ok,
    /// The writer could not write the data.
    NotWritten,
    /// The data could not be read or did not match.
    Fail(String),
    /// The adapter of the reader cannot express the case (see [`Response::Unsupported`]).
    Unsupported(String),
}

/// A subject reading data written by a subject (indices into [`Results::subjects`]).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Read {
    /// The subject that wrote the data.
    pub writer: usize,
    /// The subject that read the data.
    pub reader: usize,
    /// The outcome.
    pub status: Status,
}

/// The outcomes of a case.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Outcomes {
    /// The outcome of each subject writing the case (in the order of [`Results::subjects`]).
    pub writes: Vec<Write>,
    /// The tested reads (not necessarily every pair of subjects).
    pub reads: Vec<Read>,
    /// The non-conformances of the array metadata written by each subject (in the order of [`Results::subjects`]).
    pub non_conformances: Vec<Vec<String>>,
}

impl Outcomes {
    /// The status of `reader` reading data written by `writer`, if tested.
    #[must_use]
    pub fn read(&self, writer: usize, reader: usize) -> Option<&Status> {
        self.reads
            .iter()
            .find(|read| read.writer == writer && read.reader == reader)
            .map(|read| &read.status)
    }
}

/// The results of running a [`Manifest`] with subjects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Results {
    /// The cases.
    #[serde(flatten)]
    pub manifest: Manifest,
    /// The subjects: the reference first, then the others (newest first if versions of the same implementation).
    pub subjects: Vec<Subject>,
    /// The outcomes of each case (in the order of [`Manifest::cases`]).
    pub outcomes: Vec<Outcomes>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_approx_eq() {
        let data = |bytes: Vec<u8>| Data {
            bytes,
            offsets: None,
            masks: vec![],
        };
        let int16 = |values: &[i16]| data(values.iter().flat_map(|v| v.to_le_bytes()).collect());
        let a = int16(&[-8220, 5]);
        assert!(a.approx_eq(&int16(&[-8219, 4]), Some(Numeric::Int), Some(2)));
        assert!(!a.approx_eq(&int16(&[-8218, 5]), Some(Numeric::Int), Some(2)));
        assert!(!a.approx_eq(&int16(&[-8219, 5]), None, Some(2)));
        let float32 = |values: &[f32]| data(values.iter().flat_map(|v| v.to_le_bytes()).collect());
        let a = float32(&[1.0, f32::NAN]);
        assert!(a.approx_eq(
            &float32(&[1.000_000_1, f32::NAN]),
            Some(Numeric::Float),
            Some(4)
        ));
        assert!(!a.approx_eq(&float32(&[1.001, f32::NAN]), Some(Numeric::Float), Some(4)));
        let uint8 = data(vec![0, 255]);
        assert!(uint8.approx_eq(&data(vec![1, 254]), Some(Numeric::UInt), Some(1)));
        assert!(!uint8.approx_eq(&data(vec![0, 0]), Some(Numeric::UInt), Some(1)));
    }
}
