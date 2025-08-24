// TODO: Support actual partial decoding, coalescing required

use std::{num::NonZeroU64, sync::Arc};

use crate::array::{
    array_bytes::extract_decoded_regions_vlen,
    codec::{ArrayPartialDecoderTraits, BytesPartialDecoderTraits, CodecError, CodecOptions},
    ArrayBytes, ArraySize, ChunkRepresentation, CodecChain, DataType, FillValue, RawBytes,
};
use zarrs_metadata_ext::codec::vlen::{VlenIndexDataType, VlenIndexLocation};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};

/// Partial decoder for the `bytes` codec.
pub(crate) struct VlenPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    index_codecs: Arc<CodecChain>,
    data_codecs: Arc<CodecChain>,
    index_data_type: VlenIndexDataType,
    index_location: VlenIndexLocation,
}

impl VlenPartialDecoder {
    /// Create a new partial decoder for the `bytes` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
        index_codecs: Arc<CodecChain>,
        data_codecs: Arc<CodecChain>,
        index_data_type: VlenIndexDataType,
        index_location: VlenIndexLocation,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
            index_codecs,
            data_codecs,
            index_data_type,
            index_location,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_vlen_bytes<'a>(
    index_codecs: &CodecChain,
    data_codecs: &CodecChain,
    index_data_type: VlenIndexDataType,
    index_location: VlenIndexLocation,
    bytes: Option<RawBytes>,
    indexer: &dyn crate::indexer::Indexer,
    fill_value: &FillValue,
    shape: &[u64],
    options: &CodecOptions,
) -> Result<ArrayBytes<'a>, CodecError> {
    if let Some(bytes) = bytes {
        let num_elements = usize::try_from(shape.iter().product::<u64>()).unwrap();
        let index_shape = vec![unsafe { NonZeroU64::new_unchecked(1 + num_elements as u64) }];
        let index_chunk_representation = match index_data_type {
            // VlenIndexDataType::UInt8 => {
            //     ChunkRepresentation::new(index_shape, DataType::UInt8, 0u8)
            // }
            // VlenIndexDataType::UInt16 => {
            //     ChunkRepresentation::new(index_shape, DataType::UInt16, 0u16)
            // }
            VlenIndexDataType::UInt32 => {
                ChunkRepresentation::new(index_shape, DataType::UInt32, 0u32)
            }
            VlenIndexDataType::UInt64 => {
                ChunkRepresentation::new(index_shape, DataType::UInt64, 0u64)
            }
        }
        .expect("all data types/fill values are compatible");
        let (data, index) = super::get_vlen_bytes_and_offsets(
            &index_chunk_representation,
            &bytes,
            index_codecs,
            data_codecs,
            index_location,
            options,
        )?;
        extract_decoded_regions_vlen(&data, &index, indexer, shape)
    } else {
        // Chunk is empty, all decoded regions are empty
        let array_size = ArraySize::Variable {
            num_elements: indexer.len(),
        };
        Ok(ArrayBytes::new_fill_value(array_size, fill_value))
    }
}

impl ArrayPartialDecoderTraits for VlenPartialDecoder {
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
        // Get all the input bytes (cached due to CodecTraits::partial_decoder_decodes_all() == true)
        let bytes = self.input_handle.decode(options)?;
        decode_vlen_bytes(
            &self.index_codecs,
            &self.data_codecs,
            self.index_data_type,
            self.index_location,
            bytes,
            indexer,
            self.decoded_representation.fill_value(),
            &self.decoded_representation.shape_u64(),
            options,
        )
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the `bytes` codec.
pub(crate) struct AsyncVlenPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    index_codecs: Arc<CodecChain>,
    data_codecs: Arc<CodecChain>,
    index_data_type: VlenIndexDataType,
    index_location: VlenIndexLocation,
}

#[cfg(feature = "async")]
impl AsyncVlenPartialDecoder {
    /// Create a new partial decoder for the `bytes` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
        index_codecs: Arc<CodecChain>,
        data_codecs: Arc<CodecChain>,
        index_data_type: VlenIndexDataType,
        index_location: VlenIndexLocation,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
            index_codecs,
            data_codecs,
            index_data_type,
            index_location,
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncVlenPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        // Get all the input bytes (cached due to CodecTraits::partial_decoder_decodes_all() == true)
        let bytes = self.input_handle.decode(options).await?;
        decode_vlen_bytes(
            &self.index_codecs,
            &self.data_codecs,
            self.index_data_type,
            self.index_location,
            bytes,
            indexer,
            self.decoded_representation.fill_value(),
            &self.decoded_representation.shape_u64(),
            options,
        )
    }
}
