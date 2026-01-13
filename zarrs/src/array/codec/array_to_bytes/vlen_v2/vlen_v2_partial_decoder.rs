// TODO: Support actual partial decoding, coalescing required

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::array::{ArrayBytes, ArrayBytesRaw, DataType, FillValue};
use zarrs_codec::{
    ArrayPartialDecoderTraits, BytesPartialDecoderTraits, CodecError, CodecOptions,
    extract_decoded_regions_vlen,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};
use zarrs_storage::StorageError;

/// Partial decoder for the `bytes` codec.
pub(crate) struct VlenV2PartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    shape: Vec<NonZeroU64>,
    data_type: DataType,
    fill_value: FillValue,
}

impl VlenV2PartialDecoder {
    /// Create a new partial decoder for the `bytes` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: Vec<NonZeroU64>,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Self {
        Self {
            input_handle,
            shape,
            data_type,
            fill_value,
        }
    }
}

fn decode_vlen_bytes<'a>(
    bytes: Option<ArrayBytesRaw>,
    indexer: &dyn crate::array::Indexer,
    data_type: &DataType,
    fill_value: &FillValue,
    shape: &[NonZeroU64],
) -> Result<ArrayBytes<'a>, CodecError> {
    if let Some(bytes) = bytes {
        let num_elements =
            usize::try_from(shape.iter().copied().map(NonZeroU64::get).product::<u64>()).unwrap();
        let (bytes, offsets) = super::get_interleaved_bytes_and_offsets(num_elements, &bytes)?;
        Ok(ArrayBytes::Variable(extract_decoded_regions_vlen(
            &bytes, &offsets, indexer, shape,
        )?))
    } else {
        // Chunk is empty, all decoded regions are empty
        ArrayBytes::new_fill_value(data_type, indexer.len(), fill_value).map_err(CodecError::from)
    }
}

impl ArrayPartialDecoderTraits for VlenV2PartialDecoder {
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
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        // Get all of the input bytes (cached due to PartialDecoderCapability.partial_read == false)
        let bytes = self.input_handle.decode(options)?;
        decode_vlen_bytes(
            bytes,
            indexer,
            &self.data_type,
            &self.fill_value,
            &self.shape,
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
    shape: Vec<NonZeroU64>,
    data_type: DataType,
    fill_value: FillValue,
}

#[cfg(feature = "async")]
impl AsyncVlenV2PartialDecoder {
    /// Create a new partial decoder for the `bytes` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: Vec<NonZeroU64>,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Self {
        Self {
            input_handle,
            shape,
            data_type,
            fill_value,
        }
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for AsyncVlenV2PartialDecoder {
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
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        // Get all of the input bytes (cached due to PartialDecoderCapability.partial_read == false)
        let bytes = self.input_handle.decode(options).await?;
        decode_vlen_bytes(
            bytes,
            indexer,
            &self.data_type,
            &self.fill_value,
            &self.shape,
        )
    }

    fn supports_partial_decode(&self) -> bool {
        false
    }
}
