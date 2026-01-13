// TODO: Support actual partial decoding, coalescing required

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::array::codec::{
    ArrayPartialDecoderTraits, BytesPartialDecoderTraits, CodecError, CodecOptions,
};
#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};
use crate::array::{ArrayBytes, ArrayBytesRaw, CodecChain, DataType, FillValue};
use crate::metadata_ext::codec::vlen::{VlenIndexDataType, VlenIndexLocation};
use crate::storage::StorageError;
use zarrs_codec::extract_decoded_regions_vlen;

/// Partial decoder for the `bytes` codec.
pub(crate) struct VlenPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    shape: Vec<NonZeroU64>,
    data_type: DataType,
    fill_value: FillValue,
    index_codecs: Arc<CodecChain>,
    data_codecs: Arc<CodecChain>,
    index_data_type: VlenIndexDataType,
    index_location: VlenIndexLocation,
}

impl VlenPartialDecoder {
    /// Create a new partial decoder for the `bytes` codec.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: Vec<NonZeroU64>,
        data_type: DataType,
        fill_value: FillValue,
        index_codecs: Arc<CodecChain>,
        data_codecs: Arc<CodecChain>,
        index_data_type: VlenIndexDataType,
        index_location: VlenIndexLocation,
    ) -> Self {
        Self {
            input_handle,
            shape,
            data_type,
            fill_value,
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
    bytes: Option<ArrayBytesRaw>,
    indexer: &dyn crate::array::Indexer,
    data_type: &DataType,
    fill_value: &FillValue,
    shape: &[NonZeroU64],
    options: &CodecOptions,
) -> Result<ArrayBytes<'a>, CodecError> {
    if let Some(bytes) = bytes {
        let (data, index) = super::get_vlen_bytes_and_offsets(
            &bytes,
            shape,
            index_data_type,
            index_codecs,
            data_codecs,
            index_location,
            options,
        )?;
        Ok(ArrayBytes::Variable(extract_decoded_regions_vlen(
            &data, &index, indexer, shape,
        )?))
    } else {
        // Chunk is empty, all decoded regions are empty
        ArrayBytes::new_fill_value(data_type, indexer.len(), fill_value).map_err(CodecError::from)
    }
}

impl ArrayPartialDecoderTraits for VlenPartialDecoder {
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
        // Get all the input bytes (cached due to PartialDecoderCapability.partial_read == false)
        let bytes = self.input_handle.decode(options)?;
        decode_vlen_bytes(
            &self.index_codecs,
            &self.data_codecs,
            self.index_data_type,
            self.index_location,
            bytes,
            indexer,
            &self.data_type,
            &self.fill_value,
            &self.shape,
            options,
        )
    }

    fn supports_partial_decode(&self) -> bool {
        false
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the `bytes` codec.
pub(crate) struct AsyncVlenPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    shape: Vec<NonZeroU64>,
    data_type: DataType,
    fill_value: FillValue,
    index_codecs: Arc<CodecChain>,
    data_codecs: Arc<CodecChain>,
    index_data_type: VlenIndexDataType,
    index_location: VlenIndexLocation,
}

#[cfg(feature = "async")]
impl AsyncVlenPartialDecoder {
    /// Create a new partial decoder for the `bytes` codec.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: Vec<NonZeroU64>,
        data_type: DataType,
        fill_value: FillValue,
        index_codecs: Arc<CodecChain>,
        data_codecs: Arc<CodecChain>,
        index_data_type: VlenIndexDataType,
        index_location: VlenIndexLocation,
    ) -> Self {
        Self {
            input_handle,
            shape,
            data_type,
            fill_value,
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
        // Get all the input bytes (cached due to PartialDecoderCapability.partial_read == false)
        let bytes = self.input_handle.decode(options).await?;
        decode_vlen_bytes(
            &self.index_codecs,
            &self.data_codecs,
            self.index_data_type,
            self.index_location,
            bytes,
            indexer,
            &self.data_type,
            &self.fill_value,
            &self.shape,
            options,
        )
    }

    fn supports_partial_decode(&self) -> bool {
        false
    }
}
