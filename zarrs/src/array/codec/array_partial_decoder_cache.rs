//! A cache for partial decoders.

use std::sync::Arc;

use crate::array::{ArrayBytes, ArraySubset, ChunkGrid, ChunkShape, DataType};
use zarrs_codec::{
    ArrayPartialDecoderSubchunkingTraits, ArrayPartialDecoderTraits, CodecError, CodecOptions,
    EncodedSubchunk,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderSubchunkingTraits, AsyncArrayPartialDecoderTraits};
use zarrs_storage::StorageError;

/// A cache for an [`ArrayPartialDecoderTraits`] or [`AsyncArrayPartialDecoderTraits`] partial decoder.
///
/// The decoded chunk is cached, and encoded subchunks are retrieved through the input handle.
pub(crate) struct ArrayPartialDecoderCache<T: ?Sized> {
    /// The input handle, retained only if it exposes subchunks.
    input_handle: Option<Arc<T>>,
    shape: ChunkShape,
    data_type: DataType,
    cache: ArrayBytes<'static>,
    local_subchunk_grids: Vec<Option<ChunkGrid>>,
}

impl ArrayPartialDecoderCache<dyn ArrayPartialDecoderTraits> {
    /// Create a new partial decoder cache.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation of the partial decoder fails.
    pub(crate) fn new(
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        shape: ChunkShape,
        data_type: DataType,
        options: &CodecOptions,
    ) -> Result<Self, CodecError> {
        let local_subchunk_grids = input_handle.local_subchunk_grids(options)?;
        let bytes = input_handle
            .partial_decode(
                &ArraySubset::new_with_shape(bytemuck::must_cast_slice(&shape)),
                options,
            )?
            .into_owned();
        Ok(Self {
            input_handle: (!local_subchunk_grids.is_empty()).then_some(input_handle),
            shape,
            data_type,
            cache: bytes,
            local_subchunk_grids,
        })
    }
}

#[cfg(feature = "async")]
impl ArrayPartialDecoderCache<dyn AsyncArrayPartialDecoderTraits> {
    /// Create a new asynchronous partial decoder cache.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation of the partial decoder fails.
    pub(crate) async fn async_new(
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        shape: ChunkShape,
        data_type: DataType,
        options: &CodecOptions,
    ) -> Result<Self, CodecError> {
        let local_subchunk_grids = input_handle.local_subchunk_grids(options).await?;
        let bytes = input_handle
            .partial_decode(
                &ArraySubset::new_with_shape(bytemuck::must_cast_slice(&shape)),
                options,
            )
            .await?
            .into_owned();
        Ok(Self {
            input_handle: (!local_subchunk_grids.is_empty()).then_some(input_handle),
            shape,
            data_type,
            cache: bytes,
            local_subchunk_grids,
        })
    }
}

impl ArrayPartialDecoderSubchunkingTraits
    for ArrayPartialDecoderCache<dyn ArrayPartialDecoderTraits>
{
    fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        Ok(self.local_subchunk_grids.clone())
    }

    fn retrieve_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<EncodedSubchunk>, CodecError> {
        let Some(input_handle) = &self.input_handle else {
            return Err(CodecError::UnsupportedEncodedSubchunk);
        };
        input_handle.retrieve_encoded_subchunk_at_level(level, subchunk_indices, options)
    }
}

impl ArrayPartialDecoderTraits for ArrayPartialDecoderCache<dyn ArrayPartialDecoderTraits> {
    fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.input_handle
            .as_ref()
            .map_or(0, |input_handle| input_handle.size_held())
            + self.cache.size()
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::array::Indexer,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let array_shape = bytemuck::must_cast_slice(&self.shape);
        self.cache
            .extract_array_subset(indexer, array_shape, &self.data_type)
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderSubchunkingTraits
    for ArrayPartialDecoderCache<dyn AsyncArrayPartialDecoderTraits>
{
    async fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        Ok(self.local_subchunk_grids.clone())
    }

    async fn retrieve_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<EncodedSubchunk>, CodecError> {
        let Some(input_handle) = &self.input_handle else {
            return Err(CodecError::UnsupportedEncodedSubchunk);
        };
        input_handle
            .retrieve_encoded_subchunk_at_level(level, subchunk_indices, options)
            .await
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits
    for ArrayPartialDecoderCache<dyn AsyncArrayPartialDecoderTraits>
{
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.input_handle
            .as_ref()
            .map_or(0, |input_handle| input_handle.size_held())
            + self.cache.size()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::array::Indexer,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let array_shape = bytemuck::must_cast_slice(&self.shape);
        self.cache
            .extract_array_subset(indexer, array_shape, &self.data_type)
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;
    use crate::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
    use crate::array::{ArrayBuilder, data_type};

    fn sharded_array_builder() -> ArrayBuilder {
        let data_type = data_type::uint16();
        let subchunk_shape = vec![NonZeroU64::new(4).unwrap(); 2];
        let mut builder = ArrayBuilder::new(vec![8, 8], vec![8, 8], data_type.clone(), 0u16);
        builder.array_to_bytes_codec(
            ShardingCodecBuilder::new(subchunk_shape, &data_type).build_arc(),
        );
        builder
    }

    #[test]
    fn array_partial_decoder_cache_forwards_encoded_subchunks()
    -> Result<(), Box<dyn std::error::Error>> {
        let array = sharded_array_builder()
            .build(Arc::new(zarrs_storage::store::MemoryStore::new()), "/")?;
        array.store_chunk(&[0, 0], (0..64u16).collect::<Vec<_>>())?;
        let options = CodecOptions::default();
        let decoder = array.partial_decoder(&[0, 0])?;
        let cache = ArrayPartialDecoderCache::new(
            decoder.clone(),
            array.chunk_shape(&[0, 0])?,
            array.data_type().clone(),
            &options,
        )?;
        let expected = decoder.retrieve_encoded_subchunk_at_level(0, &[1, 0], &options)?;
        assert!(expected.is_some());
        assert_eq!(
            cache.retrieve_encoded_subchunk_at_level(0, &[1, 0], &options)?,
            expected
        );
        Ok(())
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_array_partial_decoder_cache_forwards_encoded_subchunks()
    -> Result<(), Box<dyn std::error::Error>> {
        let array = sharded_array_builder()
            .build(Arc::new(zarrs_storage::store::AsyncMemoryStore::new()), "/")?;
        array
            .async_store_chunk(&[0, 0], (0..64u16).collect::<Vec<_>>())
            .await?;
        let options = CodecOptions::default();
        let decoder = array.async_partial_decoder(&[0, 0]).await?;
        let cache = ArrayPartialDecoderCache::async_new(
            decoder.clone(),
            array.chunk_shape(&[0, 0])?,
            array.data_type().clone(),
            &options,
        )
        .await?;
        let expected = decoder
            .retrieve_encoded_subchunk_at_level(0, &[1, 0], &options)
            .await?;
        assert!(expected.is_some());
        assert_eq!(
            cache
                .retrieve_encoded_subchunk_at_level(0, &[1, 0], &options)
                .await?,
            expected
        );
        Ok(())
    }
}
