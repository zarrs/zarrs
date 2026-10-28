use std::sync::Arc;

#[cfg(feature = "async")]
use async_generic::async_generic;
use std::num::NonZeroU64;
use std::ops::Range;

use super::packbits_codec::PackBitsCodecBound;
use super::unpack;
use crate::array::{ArrayBytes, ChunkShape, CowBytes, DataType};
use zarrs_codec::{
    ArrayPartialDecoderNoSubchunkingTraits, ArrayPartialDecoderTraits, BytesPartialDecoderTraits,
    CodecError, CodecOptions, InvalidBytesLengthError,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};
use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::ByteRange;

/// Retrieve the packed bytes of `bit_ranges` from packed elements starting at `packed_start`.
///
/// If `probe` is set, the encoded bytes must end one byte after it.
/// Returns [`None`] if the encoded bytes do not exist.
///
/// # Errors
/// Returns an error if the bytes cannot be retrieved, or a byte range has an unexpected length.
#[cfg_attr(feature = "async", async_generic(
    async_signature(
    input_handle: &'a Arc<dyn AsyncBytesPartialDecoderTraits>,
    bit_ranges: &'a [Range<u64>],
    packed_start: u64,
    probe: Option<u64>,
    options: &CodecOptions,
)))]
fn retrieve_packed<'a>(
    input_handle: &'a Arc<dyn BytesPartialDecoderTraits>,
    bit_ranges: &'a [Range<u64>],
    packed_start: u64,
    probe: Option<u64>,
    options: &CodecOptions,
) -> Result<Option<Vec<CowBytes<'a>>>, CodecError> {
    let byte_range = |bit_range: &Range<u64>| bit_range.start / 8..bit_range.end.div_ceil(8);
    let byte_ranges = bit_ranges
        .iter()
        .map(move |bit_range| {
            let byte_range = byte_range(bit_range);
            ByteRange::new(packed_start + byte_range.start..packed_start + byte_range.end)
        })
        .chain(probe.map(|probe| ByteRange::FromStart(probe, None)));

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

    let Some(mut encoded_bytes) = encoded_bytes else {
        return Ok(None);
    };
    let num_byte_ranges = bit_ranges.len() + usize::from(probe.is_some());
    if encoded_bytes.len() != num_byte_ranges {
        return Err(CodecError::Other(format!(
            "expected {num_byte_ranges} byte ranges from the packbits codec input, got {}",
            encoded_bytes.len()
        )));
    }
    if let Some(probe) = probe {
        let tail = encoded_bytes.pop().unwrap_or_default();
        if tail.len() != 1 {
            let probe = usize::try_from(probe).unwrap();
            return Err(InvalidBytesLengthError::new(probe + tail.len(), probe + 1).into());
        }
    }
    for (packed_elements, bit_range) in encoded_bytes.iter().zip(bit_ranges) {
        let expected_len = byte_range(bit_range);
        let expected_len = usize::try_from(expected_len.end - expected_len.start).unwrap();
        if packed_elements.len() != expected_len {
            return Err(InvalidBytesLengthError::new(packed_elements.len(), expected_len).into());
        }
    }
    Ok(Some(encoded_bytes))
}

#[cfg_attr(feature = "async", async_generic(
    async_signature(
    input_handle: &Arc<dyn AsyncBytesPartialDecoderTraits>,
    shape: &[NonZeroU64],
    codec: &PackBitsCodecBound,
    indexer: &dyn crate::array::Indexer,
    options: &CodecOptions,
)))]
fn partial_decode<'a>(
    input_handle: &Arc<dyn BytesPartialDecoderTraits>,
    shape: &[NonZeroU64],
    codec: &PackBitsCodecBound,
    indexer: &dyn crate::array::Indexer,
    options: &CodecOptions,
) -> Result<ArrayBytes<'a>, CodecError> {
    let PackBitsCodecBound {
        layout,
        element_size_bits,
        data_type,
        fill_value,
        ..
    } = codec;
    let element_size_bits = *element_size_bits;
    let data_type_size_dec = data_type.fixed_size().ok_or_else(|| {
        CodecError::Other("data type must have a fixed size for packbits codec".to_string())
    })?;
    let sizes = codec.sizes(shape)?;

    let element_size_bits_usize = usize::try_from(element_size_bits).unwrap();

    // Get the bit ranges that map to the elements
    let bit_ranges = indexer
        .iter_contiguous_byte_ranges(bytemuck::must_cast_slice(shape), element_size_bits_usize)?
        .collect::<Vec<_>>();

    // Chunks written by zarrs 0.20-0.23 omitted the padding encoding byte if all bits of whole-byte components
    // are packed. Their packed elements start one byte earlier with `first_byte` padding encoding, so retrieve
    // the last encoded byte to distinguish them, unless the last requested byte range already reaches it.
    // The padding encoding byte and encoded length are otherwise only checked when decoding, as checking them
    // here would add byte ranges to every partial decode.
    let packed_start = sizes.packed.start as u64;
    let last_byte = sizes.encoded_bytes as u64 - 1;
    let legacy = codec
        .legacy_packed(&sizes)
        .filter(|_| codec.padding_encoding == PackBitsPaddingEncoding::FirstByte);
    let probe = (legacy.is_some()
        && bit_ranges
            .last()
            .is_none_or(|bit_range| packed_start + bit_range.end.div_ceil(8) <= last_byte))
    .then_some(last_byte);

    // Retrieve those bytes
    #[cfg(feature = "async")]
    let encoded_bytes = if _async {
        retrieve_packed_async(input_handle, &bit_ranges, packed_start, probe, options).await
    } else {
        retrieve_packed(input_handle, &bit_ranges, packed_start, probe, options)
    };
    #[cfg(not(feature = "async"))]
    let encoded_bytes = retrieve_packed(input_handle, &bit_ranges, packed_start, probe, options);

    // The last encoded byte is absent in legacy chunks, so retrieving it fails or returns no bytes.
    // Retrieve the packed elements again from the start, checking that the chunk has the legacy length.
    // A failure here may not be due to a legacy chunk, so it returns the original error.
    let encoded_bytes = match (encoded_bytes, legacy) {
        (Err(err), Some(legacy)) => {
            let legacy_probe = Some(legacy.end as u64 - 1);
            #[cfg(feature = "async")]
            let encoded_bytes = if _async {
                retrieve_packed_async(input_handle, &bit_ranges, 0, legacy_probe, options).await
            } else {
                retrieve_packed(input_handle, &bit_ranges, 0, legacy_probe, options)
            };
            #[cfg(not(feature = "async"))]
            let encoded_bytes =
                retrieve_packed(input_handle, &bit_ranges, 0, legacy_probe, options);
            match encoded_bytes {
                Ok(Some(encoded_bytes)) => Some(encoded_bytes),
                Ok(None) | Err(_) => return Err(err),
            }
        }
        (encoded_bytes, _) => encoded_bytes?,
    };

    let Some(encoded_bytes) = encoded_bytes else {
        return Ok(ArrayBytes::new_fill_value(
            data_type,
            indexer.len(),
            fill_value,
        )?);
    };

    // Convert to elements
    let decoded_len = usize::try_from(indexer.len() * data_type_size_dec as u64).unwrap();
    let mut bytes_dec: Vec<u8> = Vec::with_capacity(decoded_len);
    let mut bytes_dec_remaining = &mut bytes_dec.spare_capacity_mut()[..decoded_len];
    for (packed_elements, bit_range) in encoded_bytes.into_iter().zip(&bit_ranges) {
        let num_elements = (bit_range.end - bit_range.start) / element_size_bits;
        let (bytes_dec_range, remaining) = std::mem::take(&mut bytes_dec_remaining)
            .split_at_mut_checked(usize::try_from(num_elements).unwrap() * data_type_size_dec)
            .ok_or("the packbits byte ranges exceed the decoded elements")?;
        bytes_dec_remaining = remaining;
        // The packed elements start part way through the first byte
        let bit_offset = (bit_range.start % 8) as u8;
        unpack(*layout, &packed_elements, bit_offset, bytes_dec_range);
    }
    if !bytes_dec_remaining.is_empty() {
        return Err("the packbits byte ranges do not cover the decoded elements".into());
    }
    // SAFETY: `unpack` initialised consecutive ranges of the first `decoded_len <= capacity` elements,
    // up to the empty remainder.
    unsafe { bytes_dec.set_len(decoded_len) };
    let decoded_bytes = ArrayBytes::Fixed(codec.components_to_bytes(bytes_dec)?);
    Ok(decoded_bytes)
}

/// Partial decoder for the `packbits` codec.
pub(crate) struct PackBitsPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    shape: ChunkShape,
    codec: Arc<PackBitsCodecBound>,
}

impl PackBitsPartialDecoder {
    /// Create a new partial decoder for the `packbits` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: ChunkShape,
        codec: Arc<PackBitsCodecBound>,
    ) -> Self {
        Self {
            input_handle,
            shape,
            codec,
        }
    }
}

/// The `packbits` codec encodes a chunk as a whole, so it has no subchunks.
impl ArrayPartialDecoderNoSubchunkingTraits for PackBitsPartialDecoder {}

impl ArrayPartialDecoderTraits for PackBitsPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.codec.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        partial_decode(
            &self.input_handle,
            &self.shape,
            &self.codec,
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
    codec: Arc<PackBitsCodecBound>,
}

#[cfg(feature = "async")]
impl AsyncPackBitsPartialDecoder {
    /// Create a new partial decoder for the `packbits` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: ChunkShape,
        codec: Arc<PackBitsCodecBound>,
    ) -> Self {
        Self {
            input_handle,
            shape,
            codec,
        }
    }
}

#[cfg(feature = "async")]
impl ArrayPartialDecoderNoSubchunkingTraits for AsyncPackBitsPartialDecoder {}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncPackBitsPartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.codec.data_type
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        partial_decode_async(
            &self.input_handle,
            &self.shape,
            &self.codec,
            indexer,
            options,
        )
        .await
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}
