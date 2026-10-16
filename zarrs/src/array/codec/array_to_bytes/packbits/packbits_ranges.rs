//! The byte ranges of elements of a packed chunk, and unpacking elements from them.

#![allow(clippy::similar_names)]

use std::num::NonZeroU64;
use std::ops::Div;

use num::Integer;

use super::PackBitsCodecComponents;
use crate::array::codec::array_to_bytes::packbits::{div_rem_8bit, sign_extend_component};
use crate::array::{ArrayBytes, DataType};
use zarrs_codec::CodecError;
use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;
use zarrs_storage::byte_range::ByteRange;

/// The bit ranges of the elements of `indexer` in a packed chunk of `shape`, and the byte ranges that hold them.
pub(super) fn partial_ranges(
    shape: &[NonZeroU64],
    padding_encoding: PackBitsPaddingEncoding,
    components: PackBitsCodecComponents,
    first_bit: u64,
    last_bit: u64,
    indexer: &dyn crate::array::Indexer,
) -> Result<(Vec<std::ops::Range<u64>>, Vec<ByteRange>), CodecError> {
    // Get the component and element size in bits
    let component_size_bits_extracted = last_bit - first_bit + 1;
    let element_size_bits = component_size_bits_extracted * components.num_components;
    let element_size_bits_usize = usize::try_from(element_size_bits).unwrap();

    let offset = match padding_encoding {
        PackBitsPaddingEncoding::FirstByte => 1,
        PackBitsPaddingEncoding::None | PackBitsPaddingEncoding::LastByte => 0,
    };

    // Get the bit ranges that map to the elements
    let bit_ranges = indexer
        .iter_contiguous_byte_ranges(bytemuck::must_cast_slice(shape), element_size_bits_usize)?
        .collect::<Vec<_>>();

    // Convert to byte ranges, skipping the padding encoding byte
    let byte_ranges = bit_ranges
        .iter()
        .map(|bit_range| {
            let byte_start = offset + bit_range.start.div(8);
            let byte_end = offset + bit_range.end.div_ceil(8);
            ByteRange::new(byte_start..byte_end)
        })
        .collect();
    Ok((bit_ranges, byte_ranges))
}

/// Unpack `num_elements` elements from the `encoded_bytes` read for each of `bit_ranges` (see [`partial_ranges`]).
pub(super) fn partial_unpack<'e>(
    encoded_bytes: impl IntoIterator<Item = &'e [u8]>,
    bit_ranges: &[std::ops::Range<u64>],
    data_type: &DataType,
    components: PackBitsCodecComponents,
    first_bit: u64,
    last_bit: u64,
    num_elements: u64,
) -> Result<ArrayBytes<'static>, CodecError> {
    let PackBitsCodecComponents {
        component_size_bits,
        num_components,
        sign_extension,
    } = components;

    // Get the component and element size in bits
    let component_size_bits_extracted = last_bit - first_bit + 1;
    let element_size_bits = component_size_bits_extracted * num_components;

    let data_type_size_dec = data_type.fixed_size().ok_or_else(|| {
        CodecError::Other("data type must have a fixed size for packbits codec".to_string())
    })?;

    // Convert to elements
    let mut bytes_dec: Vec<u8> =
        vec![0; usize::try_from(num_elements * data_type_size_dec as u64).unwrap()];
    let mut component_idx_outer = 0;
    for (packed_elements, bit_range) in encoded_bytes.into_iter().zip(bit_ranges) {
        // Get the bit range within the entire chunk
        let bit_start = bit_range.start;
        let bit_end = bit_range.end;
        let num_elements = (bit_end - bit_start) / element_size_bits;

        // Get the offset from the start of the byte range encapsulating the bit range
        let bit_offset_from_contiguous_byte_range = bit_start - 8 * bit_start.div(8);

        // Decode the components
        for component_idx in 0..num_elements * num_components {
            let bit_dec0 = (component_idx_outer + component_idx) * component_size_bits + first_bit;
            let bit_enc0 = component_idx * component_size_bits_extracted;
            for bit in 0..component_size_bits_extracted {
                let bit_in = bit_enc0 + bit + bit_offset_from_contiguous_byte_range;
                let bit_out = bit_dec0 + bit;
                let (byte_enc, bit_enc) = bit_in.div_rem(&8);
                let (byte_dec, bit_dec) = div_rem_8bit(bit_out, component_size_bits);
                bytes_dec[usize::try_from(byte_dec).unwrap()] |=
                    ((packed_elements[usize::try_from(byte_enc).unwrap()] >> bit_enc) & 0b1)
                        << bit_dec;
            }
            if sign_extension {
                sign_extend_component(
                    &mut bytes_dec,
                    component_idx_outer + component_idx,
                    last_bit,
                    component_size_bits,
                );
            }
        }
        component_idx_outer += num_elements * num_components;
    }
    Ok(ArrayBytes::new_flen(bytes_dec))
}
