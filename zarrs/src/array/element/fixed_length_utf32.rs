//! Element implementations for `fixed_length_utf32` data type.

use std::any::TypeId;

use crate::array::{ArrayBytes, DataType, data_type};

use super::{Element, ElementError, ElementOwned};

use ElementError::IncompatibleElementType as IET;

/// Convert UTF-32 code units (native-endian) to chars, trimming trailing U+0000.
/// Interior U+0000 characters are preserved; only trailing padding nulls are removed.
fn utf32_ne_bytes_to_trimmed_chars(code_units: &[[u8; 4]]) -> Vec<char> {
    let mut chars = Vec::with_capacity(code_units.len());

    for code_unit in code_units {
        let code_unit_scalar = u32::from_ne_bytes(*code_unit);
        if let Some(ch) = char::from_u32(code_unit_scalar) {
            chars.push(ch);
        }
    }

    // Trim only trailing U+0000 padding
    while chars.last() == Some(&'\0') {
        chars.pop();
    }

    chars
}

/// Convert UTF-32 code units (native-endian) to exactly N chars (preserving U+0000).
fn utf32_ne_bytes_to_exact_chars<const N: usize>(code_units: &[[u8; 4]]) -> [char; N] {
    let mut result = ['\0'; N];
    for (i, code_unit) in code_units.iter().take(N).enumerate() {
        let code_unit_scalar = u32::from_ne_bytes(*code_unit);
        if let Some(ch) = char::from_u32(code_unit_scalar) {
            result[i] = ch;
        }
    }
    result
}

/// Encode variable-length char slices into fixed-length UTF-32 bytes, zero-padded.
fn encode_variable_length<F>(
    elements: impl Iterator<Item = F>,
    length_bytes: usize,
    capacity: usize,
) -> Result<Vec<u8>, ElementError>
where
    F: AsRef<[char]>,
{
    let count = elements.size_hint().0;
    let mut bytes = Vec::with_capacity(count * length_bytes);
    for element in elements {
        let slice = element.as_ref();
        if slice.len() > capacity {
            return Err(ElementError::InvalidElementValue);
        }
        let encoded = bytemuck::cast_slice::<char, u8>(slice);
        bytes.extend_from_slice(encoded);
        let padding_bytes = length_bytes - encoded.len();
        if padding_bytes > 0 {
            let start = bytes.len();
            bytes.resize(start + padding_bytes, 0);
        }
    }
    Ok(bytes)
}

/// Encode fixed-size char arrays (or slices) into UTF-32 bytes, no padding.
fn encode_fixed_length<F>(elements: impl Iterator<Item = F>, length_bytes: usize) -> Vec<u8>
where
    F: AsRef<[char]>,
{
    let count = elements.size_hint().0;
    let mut bytes = Vec::with_capacity(count * length_bytes);
    for element in elements {
        let encoded = bytemuck::cast_slice::<char, u8>(element.as_ref());
        bytes.extend_from_slice(encoded);
    }
    bytes
}

/// Decode a sequence of fixed-length UTF-32 elements from bytes.
fn decode_elements<T, F>(
    bytes_fixed: &[u8],
    length_bytes: usize,
    mut decode: F,
) -> Result<Vec<T>, ElementError>
where
    F: FnMut(&[[u8; 4]]) -> T,
{
    if bytes_fixed.len() % length_bytes != 0 {
        return Err(ElementError::Other(
            "byte length is not a multiple of element size".into(),
        ));
    }
    let mut elements = Vec::with_capacity(bytes_fixed.len() / length_bytes);
    for elem_bytes in bytes_fixed.chunks_exact(length_bytes) {
        let (code_units, remainder) = elem_bytes.as_chunks::<4>();
        debug_assert!(remainder.is_empty());
        elements.push(decode(code_units));
    }
    Ok(elements)
}

/// Get the config parameters for a fixed-length UTF-32 data type.
fn get_config(data_type: &DataType) -> Result<(usize, usize), ElementError> {
    let fixed_length_utf32 = data_type
        .downcast_ref::<data_type::FixedLengthUTF32DataType>()
        .ok_or(IET)?;
    let length_bytes = usize::try_from(fixed_length_utf32.length_bytes().get()).unwrap();
    let capacity = usize::try_from(fixed_length_utf32.capacity_code_points().get()).unwrap();
    Ok((length_bytes, capacity))
}

// -- Element for &[char] --

impl Element for &[char] {
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        if data_type.is::<data_type::FixedLengthUTF32DataType>()
            || data_type
                .compatible_element_types()
                .contains(&TypeId::of::<&[char]>())
        {
            Ok(())
        } else {
            Err(IET)
        }
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError> {
        Self::validate_data_type(data_type)?;
        let (length_bytes, capacity) = get_config(data_type)?;
        encode_variable_length(elements.iter().copied(), length_bytes, capacity).map(Into::into)
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
    }
}

// -- Element for Vec<char> / ElementOwned for Vec<char> --

impl Element for Vec<char> {
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        if data_type.is::<data_type::FixedLengthUTF32DataType>()
            || data_type
                .compatible_element_types()
                .contains(&TypeId::of::<Vec<char>>())
        {
            Ok(())
        } else {
            Err(IET)
        }
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError> {
        Self::validate_data_type(data_type)?;
        let (length_bytes, capacity) = get_config(data_type)?;
        encode_variable_length(elements.iter(), length_bytes, capacity).map(Into::into)
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
    }
}

impl ElementOwned for Vec<char> {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;
        let (length_bytes, _capacity) = get_config(data_type)?;
        let bytes_fixed = bytes.into_fixed()?;
        decode_elements(&bytes_fixed, length_bytes, utf32_ne_bytes_to_trimmed_chars)
    }
}

// -- Element for &[char; N] --

impl<const N: usize> Element for &[char; N] {
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        if data_type.is::<data_type::FixedLengthUTF32DataType>() {
            let fixed_length_utf32 = data_type
                .downcast_ref::<data_type::FixedLengthUTF32DataType>()
                .unwrap();
            if fixed_length_utf32.capacity_code_points().get() as usize == N {
                return Ok(());
            }
        }
        Err(IET)
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError> {
        Self::validate_data_type(data_type)?;
        Ok(encode_fixed_length(elements.iter().copied(), N * 4).into())
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Self::validate_data_type(data_type)?;
        Ok(encode_fixed_length(elements.iter(), N * 4).into())
    }
}

// -- Element for [char; N] / ElementOwned for [char; N] --

impl<const N: usize> Element for [char; N] {
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        <&[char; N] as Element>::validate_data_type(data_type)
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<ArrayBytes<'a>, ElementError> {
        Self::validate_data_type(data_type)?;
        Ok(encode_fixed_length(elements.iter(), N * 4).into())
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Self::validate_data_type(data_type)?;
        Ok(encode_fixed_length(elements.iter().copied(), N * 4).into())
    }
}

impl<const N: usize> ElementOwned for [char; N] {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes_fixed = bytes.into_fixed()?;
        decode_elements(&bytes_fixed, N * 4, utf32_ne_bytes_to_exact_chars)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::data_type;
    use std::sync::Arc;

    use std::num::NonZeroU64;

    fn make_data_type(length_bytes: u32) -> DataType {
        Arc::new(
            data_type::FixedLengthUTF32DataType::new(NonZeroU64::new(length_bytes as u64).unwrap())
                .unwrap(),
        )
        .into()
    }

    #[test]
    fn vec_char_round_trip() {
        let data_type = make_data_type(8);
        let elements: Vec<Vec<char>> = vec![vec!['a'], vec!['a', 'b'], vec!['🎉', '🦀']];

        let bytes = Vec::<char>::to_array_bytes(&data_type, &elements).unwrap();
        let fixed = bytes.clone().into_fixed().unwrap();
        assert_eq!(fixed.len(), 3 * 8);

        let decoded = Vec::<char>::from_array_bytes(&data_type, bytes).unwrap();
        assert_eq!(decoded[0], vec!['a']);
        assert_eq!(decoded[1], vec!['a', 'b']);
        assert_eq!(decoded[2], vec!['🎉', '🦀']);
    }

    #[test]
    fn char_array_2_round_trip() {
        let data_type = make_data_type(8);
        let elements: Vec<[char; 2]> = vec![['a', 'b'], ['🎉', '🦀']];

        let bytes = <[char; 2] as Element>::to_array_bytes(&data_type, &elements).unwrap();
        let fixed = bytes.clone().into_fixed().unwrap();
        assert_eq!(fixed.len(), 2 * 8);

        let decoded = <[char; 2] as ElementOwned>::from_array_bytes(&data_type, bytes).unwrap();
        assert_eq!(decoded[0], ['a', 'b']);
        assert_eq!(decoded[1], ['🎉', '🦀']);
    }

    #[test]
    fn char_array_wrong_size_rejected() {
        let data_type = make_data_type(8);
        assert!(<[char; 3] as Element>::validate_data_type(&data_type).is_err());
    }

    #[test]
    fn ref_char_slice_padding() {
        let data_type = make_data_type(12);
        let elements: Vec<&[char]> = vec![&['a'], &['a', 'b', 'c']];

        let bytes = <&[char] as Element>::to_array_bytes(&data_type, &elements).unwrap();
        let fixed = bytes.into_fixed().unwrap();
        assert_eq!(fixed.len(), 2 * 12);

        let first_elem = &fixed[..12];
        assert_eq!(&first_elem[0..4], &('a' as u32).to_ne_bytes());
        assert_eq!(&first_elem[4..8], &0u32.to_ne_bytes());
        assert_eq!(&first_elem[8..12], &0u32.to_ne_bytes());
    }

    #[test]
    fn overlong_rejected() {
        let data_type = make_data_type(8);
        let elements: Vec<&[char]> = vec![&['a', 'b', 'c']];

        assert!(<&[char] as Element>::to_array_bytes(&data_type, &elements).is_err());
    }

    #[test]
    fn interior_null_preserved() {
        let data_type = make_data_type(12);
        // ['a', '\0', 'b'] with interior U+0000
        let elements: Vec<Vec<char>> = vec![vec!['a', '\0', 'b']];

        let bytes = Vec::<char>::to_array_bytes(&data_type, &elements).unwrap();
        let decoded = Vec::<char>::from_array_bytes(&data_type, bytes).unwrap();
        assert_eq!(decoded[0], vec!['a', '\0', 'b']);
    }

    #[test]
    fn interior_null_round_trip() {
        let data_type = make_data_type(20); // 5 code points
        // Mix of interior and trailing nulls
        let elements: Vec<Vec<char>> = vec![vec!['x', '\0', 'y', '\0', 'z']];

        let bytes = Vec::<char>::to_array_bytes(&data_type, &elements).unwrap();
        let decoded = Vec::<char>::from_array_bytes(&data_type, bytes).unwrap();
        assert_eq!(decoded[0], vec!['x', '\0', 'y', '\0', 'z']);
    }

    #[test]
    fn vec_into_array_bytes_overlong_rejected() {
        let data_type = make_data_type(8); // capacity = 2 code points
        let elements: Vec<Vec<char>> = vec![vec!['a', 'b', 'c']];

        assert!(Vec::<char>::into_array_bytes(&data_type, elements).is_err());
    }
}
