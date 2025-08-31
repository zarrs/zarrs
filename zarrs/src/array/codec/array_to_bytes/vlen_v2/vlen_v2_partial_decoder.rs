// TODO: Support actual partial decoding, coalescing required

use std::sync::Arc;

use crate::array::{
    array_bytes::extract_decoded_regions_vlen,
    codec::{ArrayPartialDecoderTraits, BytesPartialDecoderTraits, CodecError, CodecOptions},
    ArrayBytes, ArraySize, ChunkRepresentation, DataType, DataTypeSize, FillValue, RawBytes,
};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};

/// Partial decoder for the `bytes` codec.
pub(crate) struct VlenV2PartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
}

impl VlenV2PartialDecoder {
    /// Create a new partial decoder for the `bytes` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
        }
    }
}

fn decode_vlen_bytes<'a>(
    bytes: Option<RawBytes>,
    indexer: &dyn crate::indexer::Indexer,
    data_type_size: DataTypeSize,
    fill_value: &FillValue,
    shape: &[u64],
) -> Result<ArrayBytes<'a>, CodecError> {
    if let Some(bytes) = bytes {
        let num_elements = usize::try_from(shape.iter().product::<u64>()).unwrap();
        let (bytes, offsets) = super::get_interleaved_bytes_and_offsets(num_elements, &bytes)?;
        extract_decoded_regions_vlen(&bytes, &offsets, indexer, shape)
    } else {
        // Chunk is empty, all decoded regions are empty
        let array_size = ArraySize::new(data_type_size, indexer.len());
        Ok(ArrayBytes::new_fill_value(array_size, fill_value))
    }
}

impl ArrayPartialDecoderTraits for VlenV2PartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    fn size(&self) -> usize {
        self.input_handle.size()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        // Get all of the input bytes (cached due to CodecTraits::partial_decoder_decodes_all() == true)
        let bytes = self.input_handle.decode(options)?;
        decode_vlen_bytes(
            bytes,
            indexer,
            self.decoded_representation.data_type().size(),
            self.decoded_representation.fill_value(),
            &self.decoded_representation.shape_u64(),
        )
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the `bytes` codec.
pub(crate) struct AsyncVlenV2PartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
}

#[cfg(feature = "async")]
impl AsyncVlenV2PartialDecoder {
    /// Create a new partial decoder for the `bytes` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncVlenV2PartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        // Get all of the input bytes (cached due to CodecTraits::partial_decoder_decodes_all() == true)
        let bytes = self.input_handle.decode(options).await?;
        decode_vlen_bytes(
            bytes,
            indexer,
            self.decoded_representation.data_type().size(),
            self.decoded_representation.fill_value(),
            &self.decoded_representation.shape_u64(),
        )
    }
}
