// TODO: Support actual partial decoding, coalescing required

use std::num::NonZeroU64;
use std::sync::Arc;

use crate::array::array_bytes_internal::extract_decoded_regions_vlen;
use crate::array::{ArrayBytes, CodecChainBound, CowBytes, DataType, FillValue};
use zarrs_codec::{
    ArrayPartialDecoderNoSubchunkingTraits, CodecError, CodecOptions, Resources,
    SyncArrayPartialDecoderTraits, SyncBytesPartialDecoderTraits,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};
use zarrs_metadata_ext::codec::vlen::VlenIndexLocation;
use zarrs_storage::StorageError;

/// Partial decoder for the `vlen` codec.
pub(crate) struct VlenPartialDecoder<T: ?Sized> {
    input_handle: Arc<T>,
    shape: Vec<NonZeroU64>,
    data_type: DataType,
    fill_value: Option<FillValue>,
    index_codecs: Arc<CodecChainBound>,
    data_codecs: Arc<CodecChainBound>,
    index_location: VlenIndexLocation,
}

impl<T: ?Sized> VlenPartialDecoder<T> {
    /// Create a new partial decoder for the `vlen` codec.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        input_handle: Arc<T>,
        shape: Vec<NonZeroU64>,
        data_type: DataType,
        fill_value: Option<FillValue>,
        index_codecs: Arc<CodecChainBound>,
        data_codecs: Arc<CodecChainBound>,
        index_location: VlenIndexLocation,
    ) -> Self {
        Self {
            input_handle,
            shape,
            data_type,
            fill_value,
            index_codecs,
            data_codecs,
            index_location,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn decode_vlen_bytes<'a>(
    index_codecs: &CodecChainBound,
    data_codecs: &CodecChainBound,
    index_location: VlenIndexLocation,
    bytes: Option<CowBytes>,
    indexer: &dyn crate::array::Indexer,
    data_type: &DataType,
    fill_value: Option<&FillValue>,
    shape: &[NonZeroU64],
    options: &CodecOptions,
    resources: &Resources,
) -> Result<ArrayBytes<'a>, CodecError> {
    if let Some(bytes) = bytes {
        let (data, index) = super::get_vlen_bytes_and_offsets(
            &bytes,
            shape,
            index_codecs,
            data_codecs,
            index_location,
            options,
            resources.compute(),
        )?;
        Ok(ArrayBytes::Variable(extract_decoded_regions_vlen(
            &data, &index, indexer, shape,
        )?))
    } else {
        // Chunk is empty, all decoded regions are empty. The fill value does not touch the
        // indexer, so validate it here.
        indexer.validate(bytemuck::must_cast_slice(shape))?;
        let fill_value = fill_value.ok_or(CodecError::UndefinedFillValue)?;
        ArrayBytes::new_fill_value(data_type, indexer.len(), fill_value).map_err(CodecError::from)
    }
}

/// The `vlen` codec encodes a chunk as a whole, so it has no subchunks.
impl<T: ?Sized> ArrayPartialDecoderNoSubchunkingTraits for VlenPartialDecoder<T> {}

impl<T: SyncBytesPartialDecoderTraits + ?Sized + 'static> SyncArrayPartialDecoderTraits
    for VlenPartialDecoder<T>
{
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
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        // Get all the input bytes (cached due to PartialDecoderCapability.partial_read == false)
        let bytes = self.input_handle.decode(options, resources)?;
        decode_vlen_bytes(
            &self.index_codecs,
            &self.data_codecs,
            self.index_location,
            bytes,
            indexer,
            &self.data_type,
            self.fill_value.as_ref(),
            &self.shape,
            options,
            resources,
        )
    }

    fn supports_partial_decode(&self) -> bool {
        false
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: AsyncBytesPartialDecoderTraits + ?Sized + 'static> AsyncArrayPartialDecoderTraits
    for VlenPartialDecoder<T>
{
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
        resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        // Get all the input bytes (cached due to PartialDecoderCapability.partial_read == false)
        let bytes = self.input_handle.decode(options, resources).await?;
        decode_vlen_bytes(
            &self.index_codecs,
            &self.data_codecs,
            self.index_location,
            bytes,
            indexer,
            &self.data_type,
            self.fill_value.as_ref(),
            &self.shape,
            options,
            resources,
        )
    }

    fn supports_partial_decode(&self) -> bool {
        false
    }
}
