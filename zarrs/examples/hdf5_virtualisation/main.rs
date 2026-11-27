//! Read-only HDF5 virtualisation example: accessing multiple pre-existing chunked HDF5 files as
//! a single Zarr V3 array.
//!
//! Combines three general Zarr V3 building blocks:
//! - Standard Zarr V3 `rectilinear` chunk grid (supports unequal outer file shapes along any axis).
//! - [`ExplicitChunkKeyEncoding`] (`zarrs.experimental.explicit`), mapping explicit chunk grid
//!   coordinates to arbitrary relative store keys (e.g. `"part_a.h5"`, `"nested/part_b.h5"`).
//! - [`Hdf5Codec`] (`zarrs.experimental.hdf5`), a read-only subchunking `ArrayToBytes` codec that
//!   extracts HDF5 internal chunk byte offsets via `hdf5-reader` and decodes subchunks via
//!   `zarrs`'s own [`CodecChain`] (`bytes` + `numcodecs.shuffle` + `numcodecs.zlib` +
//!   `numcodecs.fletcher32`).
//!
//! See `spec/codecs/hdf5/README.md` and `spec/chunk-key-encodings/explicit/README.md` for draft
//! specifications in the style of `zarr-developers/zarr-extensions`.

#![allow(missing_docs, clippy::cast_possible_truncation)]

mod chunk_key_encoding;
mod hdf5_codec;

use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;

use hdf5_reader::messages::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SHUFFLE, FilterDescription,
};
use hdf5_writer::{AttributeBuilder, DatasetBuilder, Hdf5Builder, WriteOptions};
use ndarray::ArrayD;
use zarrs::array::cache::{SubchunkCache, SyncChunkCacheDecoded, SyncChunkCachePartialDecoder};
use zarrs::array::chunk_grid::{ChunkEdgeLengths, RectilinearChunkGridConfiguration};
use zarrs::array::codec::{BytesCodec, CodecChain, Fletcher32Codec, ShuffleCodec, ZlibCodec};
use zarrs::array::{Array, ArrayBuilder, ArrayCached, ArraySubset, Resources, data_type};
use zarrs::filesystem::SyncFilesystemStore;
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::SyncReadableWritableListableStorage;
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;

use chunk_key_encoding::ExplicitChunkKeyEncoding;
use hdf5_codec::Hdf5Codec;

fn synthetic_element(d0: usize, d1: usize, d2: usize) -> i16 {
    ((d0 * 1000 + d1 * 20 + d2) % 30_000) as i16
}

/// Create two pre-existing HDF5 files simulating external multi-object containers on disk.
///
/// `part_a.h5` includes a sibling `/aux` dataset and a file-level attribute to illustrate why
/// a per-array Zarr codec targeting `/data` should be read-only (overwriting the file would
/// destroy sibling datasets and container metadata).
fn write_external_hdf5_files(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let filters = [
        FilterDescription {
            id: FILTER_SHUFFLE,
            name: None,
            client_data: vec![2],
        },
        FilterDescription {
            id: FILTER_DEFLATE,
            name: None,
            client_data: vec![1],
        },
        FilterDescription {
            id: FILTER_FLETCHER32,
            name: None,
            client_data: vec![],
        },
    ];

    let build_dataset = |d0_len: usize, d0_offset: usize| -> Result<DatasetBuilder, _> {
        let mut values = Vec::with_capacity(d0_len * 10 * 10);
        for d0 in 0..d0_len {
            for d1 in 0..10 {
                for d2 in 0..10 {
                    values.push(synthetic_element(d0 + d0_offset, d1, d2));
                }
            }
        }
        let mut ds = DatasetBuilder::typed_data("data", vec![d0_len as u64, 10, 10], &values)?
            .chunked(vec![4, 6, 6])
            .fill_value(0i16.to_le_bytes().to_vec());
        for f in &filters {
            ds = ds.filter(f.clone());
        }
        Ok::<_, hdf5_writer::Error>(ds)
    };

    // File 0: part_a.h5 ([14, 10, 10]) + sibling dataset `/aux` + file attribute.
    let part_a_bytes = Hdf5Builder::new()
        .attribute(AttributeBuilder::fixed_string(
            "description",
            "Pre-existing multi-dataset HDF5 container",
        ))
        .dataset(build_dataset(14, 0)?)
        .dataset(DatasetBuilder::typed_data(
            "aux",
            vec![4],
            &[1.0f32, 2.0, 3.0, 4.0],
        )?)
        .into_plan()?
        .encode(WriteOptions::default())?;
    std::fs::write(dir.join("part_a.h5"), part_a_bytes)?;

    // File 1: nested/part_b.h5 ([8, 10, 10]).
    let part_b_path = dir.join("nested/part_b.h5");
    if let Some(parent) = part_b_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let part_b_bytes = Hdf5Builder::new()
        .dataset(build_dataset(8, 14)?)
        .into_plan()?
        .encode(WriteOptions::default())?;
    std::fs::write(part_b_path, part_b_bytes)?;

    Ok(())
}

fn run_hdf5_virtualisation(output_dir: Option<&Path>) -> Result<(), Box<dyn std::error::Error>> {
    let resources = Resources::default();
    let _tempdir;
    let dir_path = match output_dir {
        Some(path) => {
            std::fs::create_dir_all(path)?;
            path.to_path_buf()
        }
        None => {
            _tempdir = tempfile::TempDir::new()?;
            _tempdir.path().to_path_buf()
        }
    };

    // 1. Simulate pre-existing chunked HDF5 files already on disk.
    write_external_hdf5_files(&dir_path)?;
    let part_a_size = std::fs::metadata(dir_path.join("part_a.h5"))?.len();
    let part_b_size = std::fs::metadata(dir_path.join("nested/part_b.h5"))?.len();
    println!(
        "Pre-existing HDF5 files: part_a.h5 ({part_a_size} B), nested/part_b.h5 ({part_b_size} B)"
    );

    // 2. Virtualise them by writing ONLY `zarr.json` alongside the HDF5 files.
    let fs_store: SyncReadableWritableListableStorage =
        Arc::new(SyncFilesystemStore::new(&dir_path)?);
    let inner_chunk_shape: Vec<NonZeroU64> = [4, 6, 6]
        .into_iter()
        .map(|d| NonZeroU64::new(d).unwrap())
        .collect();
    let subchunk_codecs = Arc::new(CodecChain::new(
        vec![],
        Arc::new(BytesCodec::little()),
        vec![
            Arc::new(ShuffleCodec::new(2)),
            Arc::new(ZlibCodec::new(1u32.try_into().unwrap())),
            Arc::new(Fletcher32Codec::new()),
        ],
    ));

    let array = ArrayBuilder::new(
        vec![22, 10, 10],
        MetadataV3::new_with_configuration(
            "rectilinear",
            RectilinearChunkGridConfiguration::Inline {
                chunk_shapes: vec![
                    ChunkEdgeLengths::Varying(serde_json::from_str("[14, 8]")?),
                    ChunkEdgeLengths::Scalar(NonZeroU64::new(10).unwrap()),
                    ChunkEdgeLengths::Scalar(NonZeroU64::new(10).unwrap()),
                ],
            },
        ),
        data_type::int16(),
        0i16,
    )
    .chunk_key_encoding(ExplicitChunkKeyEncoding::new([
        (vec![0, 0, 0], "part_a.h5"),
        (vec![1, 0, 0], "nested/part_b.h5"),
    ])?)
    .array_to_bytes_codec(Arc::new(Hdf5Codec::new(
        "data",
        Some(inner_chunk_shape),
        Some(subchunk_codecs),
    )))
    .dimension_names(["dim_0", "dim_1", "dim_2"].into())
    .build(fs_store.clone(), "/")?;

    array.store_metadata()?;
    println!(
        "Virtual Zarr V3 metadata (zarr.json):\n{}\n",
        array.metadata().to_string_pretty()
    );

    // 3. Re-open purely from `zarr.json` with performance metrics tracking.
    let metrics = Arc::new(PerformanceMetricsStorageAdapter::new(fs_store.clone()));
    let reopened = Array::open(metrics.clone(), "/")?;

    // 4. Verify global and chunk-local subchunk grids (and that raw subchunk codecs are not exposed).
    assert!(reopened.subchunk_codec().is_none());
    let global_subchunk_grid = reopened
        .subchunk_grid()
        .as_chunk_grid()
        .expect("global subchunk grid should resolve when chunk_shape is configured");
    assert_eq!(global_subchunk_grid.grid_shape(), &[6, 2, 2]);
    let local0 = reopened
        .local_subchunk_grid(&[0, 0, 0], &resources)?
        .unwrap();
    let local1 = reopened
        .local_subchunk_grid(&[1, 0, 0], &resources)?
        .unwrap();
    assert_eq!(local0.grid_shape(), &[4, 2, 2]);
    assert_eq!(local1.grid_shape(), &[2, 2, 2]);
    println!(
        "Subchunk grids: global={:?}, chunk0_local={:?}, chunk1_local={:?}",
        global_subchunk_grid.grid_shape(),
        local0.grid_shape(),
        local1.grid_shape()
    );

    // 5. Cross-file partial read (decodes SHUFFLE + DEFLATE + FLETCHER32 via zarrs CodecChain).
    metrics.reset();
    let roi = ArraySubset::new_with_ranges(&[12..16, 0..4, 0..4]);
    let roi_data: ArrayD<i16> = reopened.retrieve_array_subset(&roi, &resources)?;
    let expected_roi = ArrayD::<i16>::from_shape_fn(vec![4, 4, 4], |idx| {
        synthetic_element(idx[0] + 12, idx[1], idx[2])
    });
    assert_eq!(roi_data, expected_roi);

    let bytes_read = metrics.bytes_read();
    let total_file_bytes = usize::try_from(part_a_size + part_b_size).unwrap();
    println!(
        "Cross-file ROI [12..16, 0..4, 0..4]: read {bytes_read} B vs {total_file_bytes} B total ({:.1}%)",
        (bytes_read as f64 / total_file_bytes as f64) * 100.0
    );
    assert!(
        bytes_read < total_file_bytes,
        "expected byte-range partial read ({bytes_read} B) < total file bytes ({total_file_bytes} B)"
    );

    // 6. Direct subchunk reads (`retrieve_subchunk` and `retrieve_subchunks`).
    let edge_subchunk: ArrayD<i16> = reopened.retrieve_subchunk(&[3, 1, 1], &resources)?;
    assert_eq!(edge_subchunk.shape(), &[2, 4, 4]);
    assert_eq!(
        edge_subchunk,
        ArrayD::<i16>::from_shape_fn(vec![2, 4, 4], |idx| {
            synthetic_element(idx[0] + 12, idx[1] + 6, idx[2] + 6)
        })
    );

    let spanning_subchunks: ArrayD<i16> =
        reopened.retrieve_subchunks(&[3..5, 0..1, 0..1], &resources)?;
    assert_eq!(spanning_subchunks.shape(), &[6, 6, 6]);
    assert_eq!(
        spanning_subchunks,
        ArrayD::<i16>::from_shape_fn(vec![6, 6, 6], |idx| {
            synthetic_element(idx[0] + 12, idx[1], idx[2])
        })
    );

    // 7. SubchunkCache integration: second read of overlapping region hits cache with 0 I/O reads.
    let cache = SubchunkCache::new(
        0,
        SyncChunkCacheDecoded::new_chunk_limit(32),
        SyncChunkCachePartialDecoder::new_chunk_limit(4),
    );
    let cached = ArrayCached::new(reopened.clone(), cache);
    let first: ArrayD<i16> = cached.retrieve_array_subset(&roi, &resources)?;
    assert_eq!(first, expected_roi);
    metrics.reset();
    let second: ArrayD<i16> = cached.retrieve_array_subset(&[12..15, 0..3, 0..3], &resources)?;
    assert_eq!(
        metrics.reads(),
        0,
        "expected SubchunkCache hit with 0 reads"
    );
    assert_eq!(
        second,
        ArrayD::<i16>::from_shape_fn(vec![3, 3, 3], |idx| {
            synthetic_element(idx[0] + 12, idx[1], idx[2])
        })
    );
    println!("SubchunkCache hit verified (0 storage reads on overlapping subset).");

    // 8. Full array read verifies all 2200 elements across both HDF5 files.
    let full: ArrayD<i16> = reopened.retrieve_array_subset(&reopened.subset_all(), &resources)?;
    for d0 in 0..22 {
        for d1 in 0..10 {
            for d2 in 0..10 {
                assert_eq!(full[[d0, d1, d2]], synthetic_element(d0, d1, d2));
            }
        }
    }
    println!("Full array read verified across all 2200 elements.");

    // 9. Verify the codec is strictly read-only (leaves pre-existing HDF5 files untouched).
    let write_err = array
        .store_chunk(&[0, 0, 0], vec![1i16; 14 * 10 * 10], &resources)
        .unwrap_err();
    println!("Read-only write guard: {write_err}");
    assert!(write_err.to_string().contains("read-only"));
    assert_eq!(
        std::fs::metadata(dir_path.join("part_a.h5"))?.len(),
        part_a_size
    );

    Ok(())
}

fn main() {
    let output_dir = std::env::args().nth(1).map(std::path::PathBuf::from);
    if let Err(err) = run_hdf5_virtualisation(output_dir.as_deref()) {
        eprintln!("Error: {err:?}");
        std::process::exit(1);
    }
}

#[test]
fn hdf5_virtualisation() {
    run_hdf5_virtualisation(None).unwrap();
}
