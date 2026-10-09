// zarrs 0.16
use zarrs_version::array::ArrayBytes;
use zarrs_version::storage::store::FilesystemStore;
use zarrs_version::array_subset::ArraySubset;

fn open_array(path: &Path) -> Result<zarrs_version::array::Array<FilesystemStore>, String> {
    let metadata = serde_json::from_value::<zarrs_version::array::ArrayMetadata>(read_metadata(path)?)
        .map_err(|err| format!("parse metadata: {err}"))?;
    zarrs_version::array::Array::new_with_metadata(open_store(path)?, ARRAY_PATH, metadata)
        .map_err(|err| format!("open array: {err}"))
}

fn to_array_bytes(data: Data) -> Result<ArrayBytes<'static>, String> {
    if !data.masks.is_empty() {
        return Err("optional data is not supported by this zarrs version".to_string());
    }
    Ok(match data.offsets {
        None => ArrayBytes::new_flen(data.bytes),
        Some(offsets) => ArrayBytes::new_vlen(data.bytes, offsets),
    })
}

#[allow(unreachable_patterns)]
fn from_array_bytes(bytes: ArrayBytes<'_>) -> Result<Data, String> {
    match bytes {
        ArrayBytes::Fixed(bytes) => Ok(fixed_data(bytes.into_owned())),
        ArrayBytes::Variable(bytes, offsets) => Ok(Data {
            bytes: bytes.into_owned(),
            offsets: Some(offsets.to_vec()),
            masks: vec![],
        }),
        _ => Err("unsupported array bytes".to_string()),
    }
}
