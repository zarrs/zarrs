//! Conversion of array data (see [`Data`]) to and from the current `zarrs`.

use zarrs::array::{ArrayBytes, ArrayBytesOffsets};
pub(crate) use zarrs_regression_testing::schema::Data;

/// Convert to current `zarrs` array bytes.
pub(crate) fn to_array_bytes(data: &Data) -> Result<ArrayBytes<'static>, String> {
    let mut bytes = match &data.offsets {
        None => ArrayBytes::new_flen(data.bytes.clone()),
        Some(offsets) => {
            let offsets = offsets
                .iter()
                .map(|&offset| offset as u64)
                .collect::<Vec<_>>();
            let offsets =
                ArrayBytesOffsets::new(offsets).map_err(|err| format!("offsets: {err}"))?;
            ArrayBytes::new_vlen(data.bytes.clone(), offsets)
                .map_err(|err| format!("offsets: {err}"))?
        }
    };
    for mask in data.masks.iter().rev() {
        bytes = bytes.with_optional_mask(mask.clone());
    }
    Ok(bytes)
}

/// Convert from current `zarrs` array bytes.
pub(crate) fn from_array_bytes(bytes: ArrayBytes<'_>) -> Data {
    match bytes {
        ArrayBytes::Fixed(bytes) => Data {
            bytes: bytes.into_vec(),
            ..Data::default()
        },
        ArrayBytes::Variable(bytes) => {
            let (bytes, offsets) = bytes.into_parts();
            Data {
                bytes: bytes.into_vec(),
                offsets: Some(offsets.iter().collect()),
                masks: vec![],
            }
        }
        ArrayBytes::Optional(bytes) => {
            let (data, mask) = bytes.into_parts();
            let mut data = from_array_bytes(*data);
            data.masks.insert(0, mask.into_vec());
            data
        }
    }
}
