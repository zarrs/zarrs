//! Element implementations for `fixed_length_utf32` data type.

use std::any::TypeId;

use crate::array::{ArrayBytes, DataType, data_type};

use super::{Element, ElementError, ElementOwned};

use ElementError::IncompatibleElementType as IET;

/// Convert UTF-32 bytes (native-endian) to chars, trimming trailing U+0000.
fn utf32_ne_bytes_to_trimmed_chars(bytes: &[u8]) -> Vec<char> {
    let mut chars = Vec::new();
    let chunks = bytes.chunks_exact(4);
    let remainder = chunks.remainder();

    for chunk in chunks {
        let code_unit = u32::from_ne_bytes(chunk.try_into().unwrap());
        if code_unit == 0 {
            break; // Trim trailing U+0000
        }
        if let Some(ch) = char::from_u32(code_unit) {
            chars.push(ch);
        }
    }

    if !remainder.is_empty() {
        let mut padded = [0u8; 4];
        padded[..remainder.len()].copy_from_slice(remainder);
        let code_unit = u32::from_ne_bytes(padded);
        if code_unit != 0 {
            if let Some(ch) = char::from_u32(code_unit) {
                chars.push(ch);
            }
        }
    }

    chars
}

/// Convert UTF-32 bytes (native-endian) to exactly N chars (preserving U+0000).
fn utf32_ne_bytes_to_exact_chars<const N: usize>(bytes: &[u8]) -> [char; N] {
    let mut result = ['\0'; N];
    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        if i < N {
            let code_unit = u32::from_ne_bytes(chunk.try_into().unwrap());
            if let Some(ch) = char::from_u32(code_unit) {
                result[i] = ch;
            }
        }
    }
    result
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
        let Some(fixed_length_utf32) =
            data_type.downcast_ref::<data_type::FixedLengthUTF32DataType>()
        else {
            return Err(IET);
        };

        let length_bytes = fixed_length_utf32.length_bytes() as usize;
        let capacity = fixed_length_utf32.capacity_code_points() as usize;

        let mut bytes = Vec::with_capacity(elements.len() * length_bytes);
        for element in elements {
            let slice: &[char] = *element;
            if slice.len() > capacity {
                return Err(ElementError::InvalidElementValue);
            }
            for &ch in slice {
                bytes.extend_from_slice(&(ch as u32).to_ne_bytes());
            }
            for _ in slice.len()..capacity {
                bytes.extend_from_slice(&0u32.to_ne_bytes());
            }
        }

        Ok(bytes.into())
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
        let Some(fixed_length_utf32) =
            data_type.downcast_ref::<data_type::FixedLengthUTF32DataType>()
        else {
            return Err(IET);
        };

        let length_bytes = fixed_length_utf32.length_bytes() as usize;
        let capacity = fixed_length_utf32.capacity_code_points() as usize;

        let mut bytes = Vec::with_capacity(elements.len() * length_bytes);
        for element in elements {
            if element.len() > capacity {
                return Err(ElementError::InvalidElementValue);
            }
            for &ch in element {
                bytes.extend_from_slice(&(ch as u32).to_ne_bytes());
            }
            for _ in element.len()..capacity {
                bytes.extend_from_slice(&0u32.to_ne_bytes());
            }
        }

        Ok(bytes.into())
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Self::validate_data_type(data_type)?;
        let Some(fixed_length_utf32) =
            data_type.downcast_ref::<data_type::FixedLengthUTF32DataType>()
        else {
            return Err(IET);
        };

        let length_bytes = fixed_length_utf32.length_bytes() as usize;
        let capacity = fixed_length_utf32.capacity_code_points() as usize;

        let mut bytes = Vec::with_capacity(elements.len() * length_bytes);
        for element in &elements {
            for &ch in element {
                bytes.extend_from_slice(&(ch as u32).to_ne_bytes());
            }
            for _ in element.len()..capacity {
                bytes.extend_from_slice(&0u32.to_ne_bytes());
            }
        }

        Ok(bytes.into())
    }
}

impl ElementOwned for Vec<char> {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;
        let Some(fixed_length_utf32) =
            data_type.downcast_ref::<data_type::FixedLengthUTF32DataType>()
        else {
            return Err(IET);
        };

        let length_bytes = fixed_length_utf32.length_bytes() as usize;
        let bytes_fixed = bytes.into_fixed()?;

        let mut elements = Vec::new();
        for chunk in bytes_fixed.chunks_exact(length_bytes) {
            let chars = utf32_ne_bytes_to_trimmed_chars(chunk);
            elements.push(chars);
        }

        Ok(elements)
    }
}

// -- Element for &[char; N] --

impl<const N: usize> Element for &[char; N] {
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        if data_type.is::<data_type::FixedLengthUTF32DataType>() {
            let fixed_length_utf32 = data_type
                .downcast_ref::<data_type::FixedLengthUTF32DataType>()
                .unwrap();
            if fixed_length_utf32.capacity_code_points() as usize == N {
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

        let mut bytes = Vec::with_capacity(elements.len() * N * 4);
        for element in elements {
            let arr: &[char; N] = *element;
            for &ch in arr {
                bytes.extend_from_slice(&(ch as u32).to_ne_bytes());
            }
        }

        Ok(bytes.into())
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes: Vec<u8> = elements
            .into_iter()
            .flat_map(|arr| arr.iter().copied().flat_map(|ch| (ch as u32).to_ne_bytes()))
            .collect();
        Ok(bytes.into())
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
        let bytes: Vec<u8> = elements
            .iter()
            .flat_map(|elem| elem.iter().copied())
            .flat_map(|ch| (ch as u32).to_ne_bytes())
            .collect();
        Ok(bytes.into())
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<ArrayBytes<'static>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes: Vec<u8> = elements
            .into_iter()
            .flat_map(|arr| arr.into_iter().flat_map(|ch| (ch as u32).to_ne_bytes()))
            .collect();
        Ok(bytes.into())
    }
}

impl<const N: usize> ElementOwned for [char; N] {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes_fixed = bytes.into_fixed()?;

        let length = N * 4;
        let num_elements = bytes_fixed.len() / length;
        let mut elements = Vec::with_capacity(num_elements);

        for chunk in bytes_fixed.chunks_exact(length) {
            elements.push(utf32_ne_bytes_to_exact_chars(chunk));
        }

        Ok(elements)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::data_type;
    use std::sync::Arc;

    fn make_data_type(length_bytes: u32) -> DataType {
        Arc::new(data_type::FixedLengthUTF32DataType::new(length_bytes).unwrap()).into()
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
}
