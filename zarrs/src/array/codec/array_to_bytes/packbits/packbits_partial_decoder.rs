#![allow(clippy::similar_names)]

use std::ops::Div;
use std::sync::Arc;

use ambisync::ambisync;
use num::Integer;
use std::num::NonZeroU64;

use super::PackBitsCodecComponents;
use crate::array::codec::array_to_bytes::packbits::div_rem_8bit;
use crate::array::{ArrayBytes, ChunkShape, DataType, FillValue};
use zarrs_codec::{ArrayPartialDecoderTraits, BytesPartialDecoderTraits, CodecError, CodecOptions};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};
use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::ByteRange;

#[allow(clippy::too_many_lines)]
#[expect(clippy::too_many_arguments)]
#[ambisync(
    sync(
        name = "partial_decode",
        types(AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits),
    ),
    async(feature = "async"),
)]
async fn partial_decode_async<'a>(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    shape: &[NonZeroU64],
    data_type: &DataType,
    fill_value: &FillValue,
    padding_encoding: PackBitsPaddingEncoding,
    components: PackBitsCodecComponents,
    first_bit: u64,
    last_bit: u64,
    indexer: &dyn crate::array::Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'a>, CodecError> {
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
    let encoded_bytes = input_handle
        .partial_decode_many(Box::new(byte_ranges), options)
        .await?;

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

/// Asynchronous partial decoder for the `packbits` codec.
#[ambisync(
    sync(
        types(
            AsyncPackBitsPartialDecoder => PackBitsPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
pub(crate) struct AsyncPackBitsPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    shape: ChunkShape,
    data_type: DataType,
    fill_value: FillValue,
    padding_encoding: PackBitsPaddingEncoding,
    components: PackBitsCodecComponents,
    first_bit: u64,
    last_bit: u64,
}

#[ambisync(
    sync(
        fns("{}"),
        types(
            AsyncPackBitsPartialDecoder => PackBitsPartialDecoder,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(feature = "async"),
)]
impl AsyncPackBitsPartialDecoder {
    /// Create a new partial decoder for the `packbits` codec.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: ChunkShape,
        data_type: DataType,
        fill_value: FillValue,
        padding_encoding: PackBitsPaddingEncoding,
        components: PackBitsCodecComponents,
        first_bit: u64,
        last_bit: u64,
    ) -> Self {
        Self {
            input_handle,
            shape,
            data_type,
            fill_value,
            padding_encoding,
            components,
            first_bit,
            last_bit,
        }
    }
}

#[ambisync(
    sync(
        fns("{}", partial_decode_async => partial_decode),
        types(
            AsyncPackBitsPartialDecoder => PackBitsPartialDecoder,
            AsyncArrayPartialDecoderTraits => ArrayPartialDecoderTraits,
            AsyncBytesPartialDecoderTraits => BytesPartialDecoderTraits,
        ),
    ),
    async(
        feature = "async",
        flavor = async_trait,
        send = cfg(not(target_arch = "wasm32")),
    ),
)]
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

    async fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<zarrs_chunk_grid::ChunkGrid>>, CodecError> {
        Ok(Vec::new())
    }

    #[sync_signature(
        fn partial_decode(
            &self,
            indexer: &dyn crate::array::Indexer,
            options: &CodecOptions,
        ) -> Result<ArrayBytes<'_>, CodecError>
    )]
    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        partial_decode_async(
            &self.input_handle,
            &self.shape,
            &self.data_type,
            &self.fill_value,
            self.padding_encoding,
            self.components,
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
