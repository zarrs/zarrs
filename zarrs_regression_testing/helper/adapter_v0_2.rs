// zarrs 0.2-0.3
/// The array at `path` with `metadata`, where `action` describes an error creating it.
fn array(
    path: &Path,
    metadata: serde_json::Value,
    action: &str,
) -> Result<zarrs_version::array::Array<Arc<zarrs_version::storage::store::FilesystemStore>>, String> {
    let store = Arc::new(
        zarrs_version::storage::store::FilesystemStore::new(path)
            .map_err(|err| format!("create store: {err}"))?,
    );
    let metadata = serde_json::from_value::<zarrs_version::array::ArrayMetadata>(metadata)
        .map_err(|err| format!("parse metadata: {err}"))?;
    zarrs_version::array::Array::new_with_metadata(store, ARRAY_PATH, metadata)
        .map_err(|err| format!("{action}: {err}"))
}

fn write_array(path: &Path, metadata: serde_json::Value, shape: &[u64], data: Data) -> Result<(), String> {
    let array = array(path, metadata, "create array")?;
    array.store_metadata().map_err(|err| format!("store metadata: {err}"))?;
    let subset = zarrs_version::array_subset::ArraySubset::new_with_shape(shape_usize(shape));
    array
        .store_array_subset(&subset, &fixed_bytes(data)?)
        .map_err(|err| format!("store array subset: {err}"))
}

fn read_array(path: &Path, shape: &[u64]) -> Result<Data, String> {
    let array = array(path, read_metadata(path)?, "open array")?;
    let subset = zarrs_version::array_subset::ArraySubset::new_with_shape(shape_usize(shape));
    let bytes = array
        .retrieve_array_subset(&subset)
        .map_err(|err| format!("retrieve array subset: {err}"))?;
    Ok(fixed_data(bytes.to_vec()))
}
