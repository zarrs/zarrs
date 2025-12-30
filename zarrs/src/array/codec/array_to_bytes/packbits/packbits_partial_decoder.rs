#![allow(clippy::similar_names)]

use std::{ops::Div, sync::Arc};

#[cfg(feature = "async")]
use async_generic::async_generic;
use num::Integer;
use std::num::NonZeroU64;

use super::PackBitsCodecComponents;
#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};
use crate::array::{
    ArrayBytes, ChunkShape, DataType, FillValue,
    codec::{
        ArrayPartialDecoderTraits, BytesPartialDecoderTraits, CodecError, CodecOptions,
        array_to_bytes::packbits::{div_rem_8bit, pack_bits_components},
    },
    data_type::DataTypeExt,
};
use crate::metadata_ext::codec::packbits::PackBitsPaddingEncoding;
use crate::storage::{StorageError, byte_range::ByteRange};

// https://github.com/scouten/async-generic/pull/17
#[allow(clippy::too_many_lines)]
#[expect(clippy::too_many_arguments)]
#[cfg_attr(feature = "async", async_generic(
    async_signature(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    shape: &[NonZeroU64],
    data_type: &DataType,
    fill_value: &FillValue,
    padding_encoding: PackBitsPaddingEncoding,
    first_bit: Option<u64>,
    last_bit: Option<u64>,
    indexer: &dyn crate::indexer::Indexer,
    options: &CodecOptions,
)))]
fn partial_decode<'a>(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shape: &[NonZeroU64],
    data_type: &DataType,
    fill_value: &FillValue,
    padding_encoding: PackBitsPaddingEncoding,
    first_bit: Option<u64>,
    last_bit: Option<u64>,
    indexer: &dyn crate::indexer::Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'a>, CodecError> {
    let PackBitsCodecComponents {
        component_size_bits,
        num_components,
        sign_extension,
    } = pack_bits_components(data_type)?;
    let first_bit = first_bit.unwrap_or(0);
    let last_bit = last_bit.unwrap_or(component_size_bits - 1);

    // Get the component and element size in bits
    let component_size_bits_extracted = last_bit - first_bit + 1;
    let element_size_bits = component_size_bits_extracted * num_components;

    let data_type_size_dec = data_type.fixed_size().ok_or_else(|| {
        CodecError::Other("data type must have a fixed size for packbits codec".to_string())
    })?;

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
    let byte_ranges = bit_ranges.iter().map(|bit_range| {
        let byte_start = offset + bit_range.start.div(8);
        let byte_end = offset + bit_range.end.div_ceil(8);
        ByteRange::new(byte_start..byte_end)
    });

    // Retrieve those bytes
    #[cfg(feature = "async")]
    let encoded_bytes = if _async {
        input_handle
            .partial_decode_many(Box::new(byte_ranges), options)
            .await
    } else {
        input_handle.partial_decode_many(Box::new(byte_ranges), options)
    }?;
    #[cfg(not(feature = "async"))]
    let encoded_bytes = input_handle.partial_decode_many(Box::new(byte_ranges), options)?;

    // Convert to elements
    let decoded_bytes = if let Some(encoded_bytes) = encoded_bytes {
        let mut bytes_dec: Vec<u8> =
            vec![0; usize::try_from(indexer.len() * data_type_size_dec as u64).unwrap()];
        let mut component_idx_outer = 0;
        for (packed_elements, bit_range) in encoded_bytes.into_iter().zip(&bit_ranges) {
            // Get the bit range within the entire chunk
            let bit_start = bit_range.start;
            let bit_end = bit_range.end;
            let num_elements = (bit_end - bit_start) / element_size_bits;

            // Get the offset from the start of the byte range encapsulating the bit range
            let bit_offset_from_contiguous_byte_range = bit_start - 8 * bit_start.div(8);

            // Decode the components
            for component_idx in 0..num_elements * num_components {
                let bit_dec0 = (component_idx_outer + component_idx) * component_size_bits;
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
                    let signed: bool = {
                        let (byte_dec, bit_dec) = div_rem_8bit(
                            bit_dec0 + component_size_bits_extracted.saturating_sub(1),
                            component_size_bits,
                        );
                        bytes_dec[usize::try_from(byte_dec).unwrap()] >> bit_dec & 0x1 == 1
                    };
                    if signed {
                        for bit in component_size_bits_extracted..component_size_bits {
                            let (byte_dec, bit_dec) =
                                div_rem_8bit(bit_dec0 + bit, component_size_bits);
                            bytes_dec[usize::try_from(byte_dec).unwrap()] |= 1 << bit_dec;
                        }
                    }
                }
            }
            component_idx_outer += num_elements * num_components;
        }
        ArrayBytes::new_flen(bytes_dec)
    } else {
        ArrayBytes::new_fill_value(data_type, indexer.len(), fill_value)?
    };
    Ok(decoded_bytes)
}

/// Partial decoder for the `packbits` codec.
pub(crate) struct PackBitsPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    shape: ChunkShape,
    data_type: DataType,
    fill_value: FillValue,
    padding_encoding: PackBitsPaddingEncoding,
    first_bit: Option<u64>,
    last_bit: Option<u64>,
}

impl PackBitsPartialDecoder {
    /// Create a new partial decoder for the `packbits` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: ChunkShape,
        data_type: DataType,
        fill_value: FillValue,
        padding_encoding: PackBitsPaddingEncoding,
        first_bit: Option<u64>,
        last_bit: Option<u64>,
    ) -> Self {
        Self {
            input_handle,
            shape,
            data_type,
            fill_value,
            padding_encoding,
            first_bit,
            last_bit,
        }
    }
}

impl ArrayPartialDecoderTraits for PackBitsPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        partial_decode(
            &self.input_handle,
            &self.shape,
            &self.data_type,
            &self.fill_value,
            self.padding_encoding,
            self.first_bit,
            self.last_bit,
            indexer,
            options,
        )
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the `packbits` codec.
pub(crate) struct AsyncPackBitsPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    shape: ChunkShape,
    data_type: DataType,
    fill_value: FillValue,
    padding_encoding: PackBitsPaddingEncoding,
    first_bit: Option<u64>,
    last_bit: Option<u64>,
}

#[cfg(feature = "async")]
impl AsyncPackBitsPartialDecoder {
    /// Create a new partial decoder for the `packbits` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: ChunkShape,
        data_type: DataType,
        fill_value: FillValue,
        padding_encoding: PackBitsPaddingEncoding,
        first_bit: Option<u64>,
        last_bit: Option<u64>,
    ) -> Self {
        Self {
            input_handle,
            shape,
            data_type,
            fill_value,
            padding_encoding,
            first_bit,
            last_bit,
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncPackBitsPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        partial_decode_async(
            &self.input_handle,
            &self.shape,
            &self.data_type,
            &self.fill_value,
            self.padding_encoding,
            self.first_bit,
            self.last_bit,
            indexer,
            options,
        )
        .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}
