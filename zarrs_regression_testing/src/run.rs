//! Running cases with the current `zarrs` and previous releases.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;
use serde_json::Value;

use zarrs::array::{Array, ArrayBytes, ArrayMetadata, ArrayMetadataOptions, ArraySubset};
use zarrs::filesystem::FilesystemStore;

use zarrs_regression_testing::schema::{Status, Write};

use crate::cases::{Case, DataTypeCase};
use crate::conformance::non_conformances;
use crate::data::{Data, from_array_bytes, to_array_bytes};
use crate::helper::{self, Request, Response};
use crate::native::native_metadata;
use crate::releases::Release;

const ARRAY_PATH: &str = "/array";

/// The results of a case with a release.
#[derive(Debug, Clone)]
pub(crate) struct ReleaseResult {
    /// Whether the release wrote the case.
    pub(crate) write: Write,
    /// Data written by current read by the release.
    pub(crate) forward: Status,
    /// Data written by the release read by current.
    pub(crate) backward: Status,
    /// Data written by the release read by the release.
    pub(crate) roundtrip: Status,
    /// The non-conformances of data written by the release (see [`crate::conformance`]).
    pub(crate) non_conformances: Vec<String>,
}

/// The results of a case.
#[derive(Debug, Clone)]
pub(crate) struct CaseResult {
    /// Whether current wrote the case.
    pub(crate) current_write: Write,
    /// Current reading its own data.
    pub(crate) current_roundtrip: Status,
    /// Results per release.
    pub(crate) releases: Vec<ReleaseResult>,
}

/// The work directory of a case written by `writer` (`current` or a release).
pub(crate) fn case_dir(work_dir: &Path, writer: &str, index: usize) -> PathBuf {
    work_dir.join(writer).join(index.to_string())
}

/// Run `cases` with the current `zarrs` and each of `releases` (in parallel).
///
/// # Errors
/// Returns an error if the work directory cannot be prepared or a helper fails to build.
pub(crate) fn run(
    cases: &[Case],
    combinations: &[crate::cases::Combination],
    data_types: &[DataTypeCase],
    releases: &[Release],
    work_dir: &Path,
) -> Result<Vec<CaseResult>, String> {
    if work_dir.exists() {
        std::fs::remove_dir_all(work_dir)
            .map_err(|err| format!("remove {}: {err}", work_dir.display()))?;
    }
    let case_data_types: Vec<&DataTypeCase> = cases
        .iter()
        .map(|case| &data_types[combinations[case.combination].data_type])
        .collect();

    let mut binaries = Vec::with_capacity(releases.len());
    for (index, release) in releases.iter().enumerate() {
        eprintln!(
            "building helper for zarrs {release} ({}/{})",
            index + 1,
            releases.len()
        );
        binaries.push(helper::build(*release)?);
    }

    eprintln!("writing {} cases with current zarrs", cases.len());
    let current: Vec<(Write, Result<Data, String>)> = cases
        .par_iter()
        .enumerate()
        .map(|(index, case)| {
            let path = case_dir(work_dir, "current", index);
            match write_current(&path, &case.metadata, &case.shape, &case.data) {
                Ok(()) => (Write::Ok, read_current(&path, &case.shape)),
                Err(err) => (Write::Error(err), Err("not written".to_string())),
            }
        })
        .collect();

    eprintln!("testing {} releases", releases.len());
    let release_results: Vec<Vec<ReleaseResult>> = std::thread::scope(|scope| {
        let handles: Vec<_> = releases
            .iter()
            .zip(&binaries)
            .map(|(release, binary)| {
                let (current, case_data_types) = (&current, &case_data_types);
                scope.spawn(move || {
                    run_release(*release, binary, cases, current, case_data_types, work_dir)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("release thread panicked"))
            .collect()
    });

    Ok(cases
        .iter()
        .enumerate()
        .map(|(index, case)| {
            let (current_write, current_read) = &current[index];
            let current_roundtrip = if matches!(current_write, Write::Error(_)) {
                Status::NotWritten
            } else {
                compare(
                    current_read,
                    &case.data,
                    Comparison::any_if(case.lossy),
                    case_data_types[index],
                )
            };
            CaseResult {
                current_write: current_write.clone(),
                current_roundtrip,
                releases: release_results
                    .iter()
                    .map(|results| results[index].clone())
                    .collect(),
            }
        })
        .collect())
}

fn run_release(
    release: Release,
    binary: &Path,
    cases: &[Case],
    current: &[(Write, Result<Data, String>)],
    data_types: &[&DataTypeCase],
    work_dir: &Path,
) -> Vec<ReleaseResult> {
    let release_name = release.to_string();
    let metadata: Vec<Value> = cases
        .iter()
        .map(|case| native_metadata(release, &case.metadata))
        .collect();
    let requests: Vec<Request> = cases
        .iter()
        .enumerate()
        .flat_map(|(index, case)| {
            let release_dir = case_dir(work_dir, &release_name, index);
            [
                Request::Read {
                    path: case_dir(work_dir, "current", index),
                    shape: &case.shape,
                },
                Request::Write {
                    path: release_dir.clone(),
                    metadata: &metadata[index],
                    shape: &case.shape,
                    data: &case.data,
                },
                Request::Read {
                    path: release_dir,
                    shape: &case.shape,
                },
            ]
        })
        .collect();
    // Split the requests over multiple helper processes
    let chunk_size = 3 * cases.len().div_ceil(rayon::current_num_threads()).max(64);
    let responses: Vec<Response> = requests
        .par_chunks(chunk_size)
        .flat_map_iter(|requests| helper::run(binary, requests))
        .collect();

    cases
        .par_iter()
        .zip(current)
        .zip(responses.par_chunks_exact(3))
        .enumerate()
        .map(
            |(index, ((case, (current_write, current_read)), responses))| {
                let data_type = data_types[index];
                let compare = |read: &Result<Data, String>, expected: &Data, comparison| {
                    self::compare(read, expected, comparison, data_type)
                };
                let [release_read, release_write, release_self_read] = responses else {
                    unreachable!()
                };
                let release_read = release_read.clone().map(Option::unwrap_or_default);
                let release_self_read = release_self_read.clone().map(Option::unwrap_or_default);

                // Lossy data is compared against the writer's own decoding
                let forward = if matches!(current_write, Write::Error(_)) {
                    Status::NotWritten
                } else if case.lossy {
                    match current_read {
                        Ok(expected) => compare(&release_read, expected, Comparison::Approximate),
                        Err(err) => {
                            Status::Fail(format!("current failed to read its own data: {err}"))
                        }
                    }
                } else {
                    compare(&release_read, &case.data, Comparison::Exact)
                };

                let (backward, roundtrip, non_conformances) = if release_write.is_err() {
                    (Status::NotWritten, Status::NotWritten, vec![])
                } else {
                    let release_dir = case_dir(work_dir, &release_name, index);
                    let read = read_current(&release_dir, &case.shape);
                    let backward = match (&release_self_read, case.lossy) {
                        (Ok(expected), true) => compare(&read, expected, Comparison::Approximate),
                        (Err(_), true) => compare(&read, &Data::default(), Comparison::Any),
                        (_, false) => compare(&read, &case.data, Comparison::Exact),
                    };
                    let roundtrip = compare(
                        &release_self_read,
                        &case.data,
                        Comparison::any_if(case.lossy),
                    );
                    let array_dir = release_dir.join(ARRAY_PATH.trim_start_matches('/'));
                    let non_conformances = written_metadata(&array_dir)
                        .map(|metadata| non_conformances(release, &metadata, Some(&array_dir)))
                        .unwrap_or_default();
                    (backward, roundtrip, non_conformances)
                };

                ReleaseResult {
                    write: release_write
                        .as_ref()
                        .map_or_else(|err| Write::Error(err.clone()), |_| Write::Ok),
                    forward,
                    backward,
                    roundtrip,
                    non_conformances,
                }
            },
        )
        .collect()
}

/// The array metadata written to `array_dir`, if any.
fn written_metadata(array_dir: &Path) -> Option<Value> {
    let metadata = std::fs::read(array_dir.join("zarr.json")).ok()?;
    serde_json::from_slice(&metadata).ok()
}

/// How read data is compared with the expected data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Comparison {
    Exact,
    /// Approximately, as lossy data decoded by different implementations may differ slightly (see [`Data::approx_eq`]).
    Approximate,
    /// Any successfully read data matches.
    Any,
}

impl Comparison {
    /// [`Comparison::Any`] if `lossy`, otherwise [`Comparison::Exact`].
    fn any_if(lossy: bool) -> Self {
        if lossy { Self::Any } else { Self::Exact }
    }
}

/// Compare read data with the expected data.
fn compare(
    read: &Result<Data, String>,
    expected: &Data,
    comparison: Comparison,
    data_type: &DataTypeCase,
) -> Status {
    match read {
        Err(err) => Status::Fail(err.clone()),
        Ok(_) if comparison == Comparison::Any => Status::Ok,
        Ok(read) => {
            let read = data_type.canonical(read);
            let expected = data_type.canonical(expected);
            let numeric = (comparison == Comparison::Approximate)
                .then(|| data_type.values.numeric())
                .flatten();
            if read.approx_eq(&expected, numeric, data_type.values.element_size()) {
                Status::Ok
            } else {
                Status::Fail(mismatch(&read, &expected))
            }
        }
    }
}

fn mismatch(read: &Data, expected: &Data) -> String {
    if read.masks != expected.masks {
        "decoded validity mask differs".to_string()
    } else if read.offsets != expected.offsets {
        "decoded element offsets differ".to_string()
    } else if read.bytes.len() != expected.bytes.len() {
        format!(
            "decoded length differs: {} bytes, expected {}",
            read.bytes.len(),
            expected.bytes.len()
        )
    } else {
        let offset = read
            .bytes
            .iter()
            .zip(&expected.bytes)
            .position(|(read, expected)| read != expected)
            .unwrap_or_default();
        format!(
            "decoded bytes differ at byte {offset}: 0x{:02x}, expected 0x{:02x}",
            read.bytes[offset], expected.bytes[offset]
        )
    }
}

/// Write `data` to a new array with `metadata` and `shape` in the filesystem store at `path` with the current `zarrs`.
pub(crate) fn write_current(
    path: &Path,
    metadata: &Value,
    shape: &[u64],
    data: &Data,
) -> Result<(), String> {
    let store = Arc::new(FilesystemStore::new(path).map_err(|err| format!("create store: {err}"))?);
    let metadata = serde_json::from_value::<ArrayMetadata>(metadata.clone())
        .map_err(|err| format!("parse metadata: {err}"))?;
    let array = Array::new_with_metadata(store, ARRAY_PATH, metadata)
        .map_err(|err| format!("create array: {err}"))?
        .with_metadata_options(ArrayMetadataOptions::default().with_include_zarrs_metadata(false));
    array
        .store_metadata()
        .map_err(|err| format!("store metadata: {err}"))?;
    let bytes = to_array_bytes(data)?;
    array
        .store_array_subset(&ArraySubset::new_with_shape(shape.to_vec()), bytes)
        .map_err(|err| format!("store array subset: {err}"))
}

/// Read the whole array with `shape` in the filesystem store at `path` with the current `zarrs`.
pub(crate) fn read_current(path: &Path, shape: &[u64]) -> Result<Data, String> {
    let store = Arc::new(FilesystemStore::new(path).map_err(|err| format!("create store: {err}"))?);
    let array = Array::open(store, ARRAY_PATH).map_err(|err| format!("open array: {err}"))?;
    let bytes = array
        .retrieve_array_subset::<ArrayBytes>(&ArraySubset::new_with_shape(shape.to_vec()))
        .map_err(|err| format!("retrieve array subset: {err}"))?;
    Ok(from_array_bytes(bytes))
}
