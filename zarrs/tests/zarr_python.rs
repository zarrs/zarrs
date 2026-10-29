#![allow(missing_docs)]

use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use zarrs::array::{Array, ArrayBytes, ArraySubset};
use zarrs_filesystem::FilesystemStore;

#[cfg(feature = "fletcher32")]
#[test]
fn zarr_python_compat_fletcher32_v2() -> Result<(), Box<dyn Error>> {
    // NOTE: could support numcodecs.zarr3.fletcher32, but would need to permit and ignore "id" field
    // zarrs::config::global_config_mut()
    //     .experimental_codec_names_mut()
    //     .entry("fletcher32".to_string())
    //     .and_modify(|e| *e = "numcodecs.fletcher32".to_string());

    let path = PathBuf::from("tests/data/zarr_python_compat/fletcher32.zarr");
    let store = Arc::new(FilesystemStore::new(&path)?);

    let array = Array::open(store, "/")?;
    assert_eq!(array.shape(), vec![100, 100]);
    let elements =
        array.retrieve_array_subset::<Vec<u16>>(&ArraySubset::new_with_shape(array.shape()))?;
    assert_eq!(elements, (0..100 * 100).collect::<Vec<u16>>());

    Ok(())
}

#[cfg(feature = "adler32")]
#[test]
fn zarr_python_compat_adler32_v2() -> Result<(), Box<dyn Error>> {
    // NOTE: could support numcodecs.zarr3.adler32, but would need to permit and ignore "id" field
    // zarrs::config::global_config_mut()
    //     .experimental_codec_names_mut()
    //     .entry("adler32".to_string())
    //     .and_modify(|e| *e = "numcodecs.adler32".to_string());

    let path = PathBuf::from("tests/data/zarr_python_compat/adler32.zarr");
    let store = Arc::new(FilesystemStore::new(&path)?);

    let array = Array::open(store, "/")?;
    assert_eq!(array.shape(), vec![100, 100]);
    let elements =
        array.retrieve_array_subset::<Vec<u16>>(&ArraySubset::new_with_shape(array.shape()))?;
    assert_eq!(elements, (0..100 * 100).collect::<Vec<u16>>());

    Ok(())
}

#[test]
fn zarr_python_compat_packbits_v2() -> Result<(), Box<dyn Error>> {
    check_zarr_python_packbits(
        "tests/data/zarr_python_compat/packbits.zarr",
        &["0.0", "0.1", "0.2", "1.0", "1.1", "1.2"],
    )
}

#[test]
fn zarr_python_compat_packbits_v3() -> Result<(), Box<dyn Error>> {
    check_zarr_python_packbits(
        "tests/data/zarr_python_compat/packbits_v3.zarr",
        &["c/0/0", "c/0/1", "c/0/2", "c/1/0", "c/1/1", "c/1/2"],
    )
}

fn check_zarr_python_packbits(path: &str, chunks: &[&str]) -> Result<(), Box<dyn Error>> {
    use zarrs::storage::store::MemoryStore;
    use zarrs::storage::{ReadableStorageTraits, StoreKey};

    let path = PathBuf::from(path);
    let store = Arc::new(FilesystemStore::new(path)?);
    let array = Array::open(store.clone(), "/")?;
    assert_eq!(array.shape(), vec![10, 9]);
    let expected = |i: u64, j: u64| (i * 9 + j) * 7 % 11 < 4;

    // Decoding
    let elements_all = array.retrieve_array_subset::<Vec<bool>>(&array.subset_all())?;
    let answer: Vec<bool> = (0..10)
        .flat_map(|i| (0..9).map(move |j| expected(i, j)))
        .collect();
    assert_eq!(elements_all, answer);

    // Partial decoding
    let elements =
        array.retrieve_array_subset::<Vec<bool>>(&ArraySubset::new_with_ranges(&[3..8, 2..7]))?;
    let answer: Vec<bool> = (3..8)
        .flat_map(|i| (2..7).map(move |j| expected(i, j)))
        .collect();
    assert_eq!(elements, answer);

    // Encoding matches numcodecs
    let store_out = Arc::new(MemoryStore::new());
    let array_out = Array::new_with_metadata(store_out.clone(), "/", array.metadata_opt())?;
    array_out.store_array_subset(&array.subset_all(), elements_all)?;
    for &chunk in chunks {
        let key = StoreKey::new(chunk)?;
        assert_eq!(store_out.get(&key)?, store.get(&key)?, "chunk {chunk}");
    }

    Ok(())
}

#[test]
fn zarr_python_packbits_rejects_non_bool() -> Result<(), Box<dyn Error>> {
    use zarrs::array::ArrayCreateError;
    use zarrs::storage::store::MemoryStore;
    use zarrs_codec::CodecCreateError;

    for (json, dtype_key, dtypes) in [
        (
            include_str!("data/zarr_python_compat/packbits.zarr/.zarray"),
            "dtype",
            ["|u1", "<i2", "<f4"],
        ),
        (
            include_str!("data/zarr_python_compat/packbits_v3.zarr/zarr.json"),
            "data_type",
            ["uint8", "int16", "float32"],
        ),
    ] {
        let metadata: serde_json::Value = serde_json::from_str(json)?;
        for dtype in dtypes {
            let mut metadata = metadata.clone();
            metadata[dtype_key] = dtype.into();
            metadata["fill_value"] = 0.into();
            let result = Array::new_with_metadata(
                Arc::new(MemoryStore::new()),
                "/",
                serde_json::from_value(metadata)?,
            );
            assert!(
                matches!(
                    result,
                    Err(ArrayCreateError::CodecsCreateError(
                        CodecCreateError::UnsupportedDataType(_, _)
                    ))
                ),
                "{dtype_key} {dtype}"
            );
        }
    }
    Ok(())
}

#[test]
fn zarr_python_packbits_v3_partial_encode() -> Result<(), Box<dyn Error>> {
    use zarrs::storage::store::MemoryStore;
    use zarrs_codec::CodecOptions;

    let mut metadata: serde_json::Value = serde_json::from_str(include_str!(
        "data/zarr_python_compat/packbits_v3.zarr/zarr.json"
    ))?;
    for fill in [false, true] {
        metadata["fill_value"] = fill.into();
        let array = Array::new_with_metadata(
            Arc::new(MemoryStore::new()),
            "/",
            serde_json::from_value(metadata.clone())?,
        )?;
        assert_eq!(array.retrieve_chunk::<Vec<bool>>(&[0, 0])?, vec![fill; 20]);
        // The encoded fill value of a missing chunk is partially decoded to the fill value
        assert_eq!(
            array
                .retrieve_array_subset::<Vec<bool>>(&ArraySubset::new_with_ranges(&[1..4, 1..3]))?,
            vec![fill; 6]
        );
        let encoder = array.partial_encoder(&[0, 0])?;
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..4]);
        let bytes = ArrayBytes::new_flen(vec![u8::from(!fill); 6]);
        encoder.partial_encode(&subset, &bytes, &CodecOptions::default())?;
        let mut expected = vec![fill; 20];
        expected[5..8].fill(!fill);
        expected[9..12].fill(!fill);
        assert_eq!(array.retrieve_chunk::<Vec<bool>>(&[0, 0])?, expected);

        encoder.partial_encode(
            &ArraySubset::new_with_ranges(&[0..1, 0..1]),
            &ArrayBytes::new_flen(vec![u8::from(!fill)]),
            &CodecOptions::default(),
        )?;
        expected[0] = !fill;
        assert_eq!(array.retrieve_chunk::<Vec<bool>>(&[0, 0])?, expected);
    }
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn zarr_python_packbits_v3_async() -> Result<(), Box<dyn Error>> {
    use zarrs::storage::storage_adapter::sync_to_async::{
        SyncToAsyncSpawnBlocking, SyncToAsyncStorageAdapter,
    };
    use zarrs::storage::store::MemoryStore;
    use zarrs::storage::{ReadableStorageTraits, StoreKey};
    use zarrs_codec::CodecOptions;

    struct TokioSpawnBlocking;

    impl SyncToAsyncSpawnBlocking for TokioSpawnBlocking {
        async fn spawn_blocking<F, R>(&self, f: F) -> R
        where
            F: FnOnce() -> R + Send + 'static,
            R: Send + 'static,
        {
            tokio::task::spawn_blocking(f).await.unwrap()
        }
    }

    let store = Arc::new(FilesystemStore::new(
        "tests/data/zarr_python_compat/packbits_v3.zarr",
    )?);
    let array = Array::async_open(
        Arc::new(SyncToAsyncStorageAdapter::new(
            store.clone(),
            TokioSpawnBlocking,
        )),
        "/",
    )
    .await?;
    let elements = array
        .async_retrieve_array_subset::<Vec<bool>>(&array.subset_all())
        .await?;
    let expected: Vec<bool> = (0..90).map(|i| i * 7 % 11 < 4).collect();
    assert_eq!(elements, expected);
    let subset = ArraySubset::new_with_ranges(&[3..8, 2..7]);
    let partial = array
        .async_retrieve_array_subset::<Vec<bool>>(&subset)
        .await?;
    let expected: Vec<bool> = (3..8)
        .flat_map(|i| (2..7).map(move |j| (i * 9 + j) * 7 % 11 < 4))
        .collect();
    assert_eq!(partial, expected);

    let output = Arc::new(MemoryStore::new());
    let array_out = Array::new_with_metadata(
        Arc::new(SyncToAsyncStorageAdapter::new(
            output.clone(),
            TokioSpawnBlocking,
        )),
        "/",
        array.metadata_opt(),
    )?;
    array_out
        .async_store_array_subset(&array.subset_all(), elements)
        .await?;
    for chunk in ["c/0/0", "c/0/1", "c/0/2", "c/1/0", "c/1/1", "c/1/2"] {
        let key = StoreKey::new(chunk)?;
        assert_eq!(output.get(&key)?, store.get(&key)?, "chunk {chunk}");
    }

    for fill in [false, true] {
        let mut metadata = serde_json::to_value(array.metadata_opt())?;
        metadata["fill_value"] = fill.into();
        let array = Array::new_with_metadata(
            Arc::new(SyncToAsyncStorageAdapter::new(
                Arc::new(MemoryStore::new()),
                TokioSpawnBlocking,
            )),
            "/",
            serde_json::from_value(metadata)?,
        )?;
        assert_eq!(
            array.async_retrieve_chunk::<Vec<bool>>(&[0, 0]).await?,
            vec![fill; 20]
        );
        let encoder = array.async_partial_encoder(&[0, 0]).await?;
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..4]);
        let bytes = ArrayBytes::new_flen(vec![u8::from(!fill); 6]);
        encoder
            .partial_encode(&subset, &bytes, &CodecOptions::default())
            .await?;
        let mut expected = vec![fill; 20];
        expected[5..8].fill(!fill);
        expected[9..12].fill(!fill);
        assert_eq!(
            array.async_retrieve_chunk::<Vec<bool>>(&[0, 0]).await?,
            expected
        );
        encoder
            .partial_encode(
                &ArraySubset::new_with_ranges(&[0..1, 0..1]),
                &ArrayBytes::new_flen(vec![u8::from(!fill)]),
                &CodecOptions::default(),
            )
            .await?;
        expected[0] = !fill;
        assert_eq!(
            array.async_retrieve_chunk::<Vec<bool>>(&[0, 0]).await?,
            expected
        );
    }
    Ok(())
}

#[test]
fn zarr_python_v2_compat_str_fv_0() -> Result<(), Box<dyn Error>> {
    let store = Arc::new(FilesystemStore::new(
        "tests/data/zarr_python_compat/str_v2_fv_0.zarr",
    )?);
    let array = zarrs::array::Array::open(store, "/")?;
    let subset_all = array.subset_all();
    let elements = array.retrieve_array_subset::<Vec<String>>(&subset_all)?;

    assert_eq!(elements, &["a", "bb", "", "", ""]);

    Ok(())
}

#[test]
fn zarr_python_v2_compat_str_fv_null() -> Result<(), Box<dyn Error>> {
    let store = Arc::new(FilesystemStore::new(
        "tests/data/zarr_python_compat/str_v2_fv_null.zarr",
    )?);
    let array = zarrs::array::Array::open(store, "/")?;
    let subset_all = array.subset_all();
    let elements = array.retrieve_array_subset::<Vec<String>>(&subset_all)?;

    assert_eq!(elements, &["a", "bb", "", "", ""]);

    Ok(())
}

#[test]
fn zarr_python_v2_compat_bool_fv_null() -> Result<(), Box<dyn Error>> {
    let store = Arc::new(FilesystemStore::new(
        "tests/data/zarr_python_compat/bool_v2_fv_null.zarr",
    )?);
    let array = Array::open(store, "/")?;
    let subset_all = array.subset_all();
    let elements = array.retrieve_array_subset::<Vec<bool>>(&subset_all)?;
    assert_eq!(elements, &[true, false, false, false, false]);

    Ok(())
}

#[test]
fn zarr_python_v2_compat_int_fv_null() -> Result<(), Box<dyn Error>> {
    let store = Arc::new(FilesystemStore::new(
        "tests/data/zarr_python_compat/int_v2_fv_null.zarr",
    )?);
    let array = Array::open(store, "/")?;
    let subset_all = array.subset_all();
    let elements = array.retrieve_array_subset::<Vec<i32>>(&subset_all)?;
    assert_eq!(elements, &[42, 123, 0, 0, 0]);

    Ok(())
}

#[allow(clippy::approx_constant)]
#[test]
fn zarr_python_v2_compat_float_fv_null() -> Result<(), Box<dyn Error>> {
    let store = Arc::new(FilesystemStore::new(
        "tests/data/zarr_python_compat/float_v2_fv_null.zarr",
    )?);
    let array = Array::open(store, "/")?;
    let subset_all = array.subset_all();
    let elements = array.retrieve_array_subset::<Vec<f32>>(&subset_all)?;
    assert_eq!(elements, &[3.14, 2.71, 0.0, 0.0, 0.0]);

    Ok(())
}

/// Test opening a group with `"consolidated_metadata": null`
/// This is non-conformant and is generated by zarr-python <=3.1.3
/// `zarrs` permits it on read only.
#[test]
fn zarr_python_v3_consolidated_metadata_null() -> Result<(), Box<dyn Error>> {
    use zarrs::group::Group;

    let store = Arc::new(FilesystemStore::new("tests/data/issue_287.zarr")?);
    let group = Group::open(store.clone(), "/")?;

    // Should be able to open the group without errors
    assert_eq!(group.path().as_str(), "/");

    // Should be able to open the meta subgroup
    let meta_group = Group::open(store, "/meta")?;
    assert_eq!(meta_group.path().as_str(), "/meta");

    Ok(())
}

#[derive(Deserialize)]
struct CastValueMatrixManifest {
    cases: Vec<CastValueMatrixCase>,
}

#[derive(Deserialize)]
struct CastValueMatrixCase {
    source_dtype: String,
    target_dtype: String,
    rounding: String,
    out_of_range: Option<String>,
    path: String,
    expected_decoded_bytes: Vec<u8>,
}

fn zarr_python_uses_nearest_even_for_directed_uint64_to_float(case: &CastValueMatrixCase) -> bool {
    // zarr-python 3.2.1 rounds the decoded `u64::MAX` to the nearest float
    // even when the codec requests a directed mode. The cast_value spec
    // requires the configured rounding mode for this integer-to-float cast.
    case.source_dtype.starts_with("float")
        && case.target_dtype == "uint64"
        && case.out_of_range.as_deref() == Some("wrap")
        && matches!(case.rounding.as_str(), "towards-zero" | "towards-negative")
}

#[ignore = "requires generated data from tests/data/zarr_python_cast_value_matrix.py"]
#[test]
fn zarr_python_v3_cast_value_matrix_read() -> Result<(), Box<dyn Error>> {
    let root = Path::new("tests/data/zarr_python_compat/cast_value_matrix");
    let manifest_path = root.join("manifest.json");
    if !manifest_path.exists() {
        return Err(format!(
            "missing {}; run `uv run zarrs/tests/data/zarr_python_cast_value_matrix.py` first",
            manifest_path.display()
        )
        .into());
    }

    let manifest: CastValueMatrixManifest =
        serde_json::from_slice(&std::fs::read(&manifest_path)?)?;

    let mut read = 0usize;
    for case in &manifest.cases {
        let path = root.join(&case.path);
        let store = Arc::new(FilesystemStore::new(&path)?);
        let array = Array::open(store, "/")
            .map_err(|err| format!("failed to open {}: {err}", case.path))?;
        let subset_all = array.subset_all();
        let bytes = array
            .retrieve_array_subset::<ArrayBytes<'static>>(&subset_all)
            .map_err(|err| format!("failed to decode {}: {err}", case.path))?;
        let bytes = bytes.into_fixed()?.into_vec();
        if zarr_python_uses_nearest_even_for_directed_uint64_to_float(case) {
            continue;
        }
        assert_eq!(
            bytes,
            case.expected_decoded_bytes.as_slice(),
            "decoded bytes differ for {}",
            case.path
        );
        read += 1;
    }

    assert!(read > 0, "no readable cast_value matrix cases were checked");

    Ok(())
}
