// zarrs 0.16+ (after the version-specific adapter)
fn open_store(path: &Path) -> Result<Arc<FilesystemStore>, String> {
    Ok(Arc::new(
        FilesystemStore::new(path).map_err(|err| format!("create store: {err}"))?,
    ))
}

fn write_array(path: &Path, metadata: serde_json::Value, shape: &[u64], data: Data) -> Result<(), String> {
    let metadata = serde_json::from_value::<zarrs_version::array::ArrayMetadata>(metadata)
        .map_err(|err| format!("parse metadata: {err}"))?;
    let array = zarrs_version::array::Array::new_with_metadata(open_store(path)?, ARRAY_PATH, metadata)
        .map_err(|err| format!("create array: {err}"))?;
    array.store_metadata().map_err(|err| format!("store metadata: {err}"))?;
    let subset = ArraySubset::new_with_shape(shape.to_vec());
    array
        .store_array_subset(&subset, to_array_bytes(data)?)
        .map_err(|err| format!("store array subset: {err}"))
}

fn read_array(path: &Path, shape: &[u64]) -> Result<Data, String> {
    let array = open_array(path)?;
    let subset = ArraySubset::new_with_shape(shape.to_vec());
    let bytes = array
        .retrieve_array_subset(&subset)
        .map_err(|err| format!("retrieve array subset: {err}"))?;
    from_array_bytes(bytes)
}
