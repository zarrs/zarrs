// TODO: Support actual partial decoding, coalescing required

use std::sync::Arc;

use zarrs_storage::StorageError;

use crate::array::{
    array_bytes::extract_decoded_regions_vlen,
    codec::{ArrayPartialDecoderTraits, BytesPartialDecoderTraits, CodecError, CodecOptions},
    ArrayBytes, ArrayBytesRaw, ChunkRepresentation, DataType, FillValue,
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
    bytes: Option<ArrayBytesRaw>,
    indexer: &dyn crate::indexer::Indexer,
    data_type: &DataType,
    fill_value: &FillValue,
    shape: &[u64],
) -> Result<ArrayBytes<'a>, CodecError> {
    if let Some(bytes) = bytes {
        let num_elements = usize::try_from(shape.iter().product::<u64>()).unwrap();
        let (bytes, offsets) = super::get_interleaved_bytes_and_offsets(num_elements, &bytes)?;
        extract_decoded_regions_vlen(&bytes, &offsets, indexer, shape)
    } else {
        // Chunk is empty, all decoded regions are empty
        ArrayBytes::new_fill_value(data_type, indexer.len(), fill_value).map_err(CodecError::from)
    }
}

impl ArrayPartialDecoderTraits for VlenV2PartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
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
        // Get all of the input bytes (cached due to PartialDecoderCapability.partial_read == false)
        let bytes = self.input_handle.decode(options)?;
        decode_vlen_bytes(
            bytes,
            indexer,
            self.decoded_representation.data_type(),
            self.decoded_representation.fill_value(),
            &self.decoded_representation.shape_u64(),
        )
    }

    fn supports_partial_decode(&self) -> bool {
        false
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
        // Get all of the input bytes (cached due to PartialDecoderCapability.partial_read == false)
        let bytes = self.input_handle.decode(options).await?;
        decode_vlen_bytes(
            bytes,
            indexer,
            self.decoded_representation.data_type(),
            self.decoded_representation.fill_value(),
            &self.decoded_representation.shape_u64(),
        )
    }

    fn supports_partial_decode(&self) -> bool {
        false
    }
}
