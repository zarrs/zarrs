// zarrs 0.23
use zarrs_version::array::{ArrayBytes, ArrayBytesOffsets};
use zarrs_version::filesystem::FilesystemStore;
use zarrs_version::array::ArraySubset;

fn open_array(path: &Path) -> Result<zarrs_version::array::Array<FilesystemStore>, String> {
    zarrs_version::array::Array::open(open_store(path)?, ARRAY_PATH)
        .map_err(|err| format!("open array: {err}"))
}

fn to_array_bytes(data: Data) -> Result<ArrayBytes<'static>, String> {
    let mut bytes = match data.offsets {
        None => ArrayBytes::new_flen(data.bytes),
        Some(offsets) => {
            let offsets = ArrayBytesOffsets::new(offsets).map_err(|err| format!("offsets: {err}"))?;
            ArrayBytes::new_vlen(data.bytes, offsets).map_err(|err| format!("offsets: {err}"))?
        }
    };
    for mask in data.masks.into_iter().rev() {
        bytes = bytes.with_optional_mask(mask);
    }
    Ok(bytes)
}

#[allow(unreachable_patterns)]
fn from_array_bytes(bytes: ArrayBytes<'_>) -> Result<Data, String> {
    match bytes {
        ArrayBytes::Fixed(bytes) => Ok(fixed_data(bytes.into_owned())),
        ArrayBytes::Variable(bytes) => {
            let (bytes, offsets) = bytes.into_parts();
            Ok(Data {
                bytes: bytes.into_owned(),
                offsets: Some(offsets.to_vec()),
                masks: vec![],
            })
        }
        ArrayBytes::Optional(bytes) => {
            let (data, mask) = bytes.into_parts();
            let mut data = from_array_bytes(*data)?;
            data.masks.insert(0, mask.into_owned());
            Ok(data)
        }
        _ => Err("unsupported array bytes".to_string()),
    }
}
