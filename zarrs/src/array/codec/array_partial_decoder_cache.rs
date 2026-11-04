//! A cache for partial decoders.

use crate::array::{ArrayBytes, ArraySubset, ChunkGrid, ChunkShape, DataType};
use zarrs_codec::{
    ArrayPartialDecoderSubchunkingTraits, ArrayPartialDecoderTraits, CodecError, CodecOptions,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderSubchunkingTraits, AsyncArrayPartialDecoderTraits};
use zarrs_storage::StorageError;

/// A cache for an [`ArrayPartialDecoderTraits`] partial decoder.
pub(crate) struct ArrayPartialDecoderCache {
    shape: ChunkShape,
    data_type: DataType,
    cache: ArrayBytes<'static>,
    local_subchunk_grids: Vec<Option<ChunkGrid>>,
}

impl ArrayPartialDecoderCache {
    /// Create a new partial decoder cache.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation of the partial decoder fails.
    pub(crate) fn new(
        input_handle: &dyn ArrayPartialDecoderTraits,
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
            shape,
            data_type,
            cache: bytes,
            local_subchunk_grids,
        })
    }

    #[cfg(feature = "async")]
    /// Create a new asynchronous partial decoder cache.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation of the partial decoder fails.
    pub(crate) async fn async_new(
        input_handle: &dyn AsyncArrayPartialDecoderTraits,
        shape: ChunkShape,
        data_type: DataType,
        options: &CodecOptions,
    ) -> Result<ArrayPartialDecoderCache, CodecError> {
        let local_subchunk_grids = input_handle.local_subchunk_grids(options).await?;
        let bytes = input_handle
            .partial_decode(
                &ArraySubset::new_with_shape(bytemuck::must_cast_slice(&shape)),
                options,
            )
            .await?
            .into_owned();
        Ok(Self {
            shape,
            data_type,
            cache: bytes,
            local_subchunk_grids,
        })
    }
}

impl ArrayPartialDecoderSubchunkingTraits for ArrayPartialDecoderCache {
    fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        Ok(self.local_subchunk_grids.clone())
    }
}

impl ArrayPartialDecoderTraits for ArrayPartialDecoderCache {
    fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.cache.size()
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
impl AsyncArrayPartialDecoderSubchunkingTraits for ArrayPartialDecoderCache {
    async fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        Ok(self.local_subchunk_grids.clone())
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for ArrayPartialDecoderCache {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.cache.size()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        ArrayPartialDecoderTraits::partial_decode(self, indexer, options)
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

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

    /// The cache holds only the decoded chunk, so it exposes local subchunk grids but not encoded subchunks.
    #[test]
    fn array_partial_decoder_cache_holds_only_the_decoded_chunk()
    -> Result<(), Box<dyn std::error::Error>> {
        let array = sharded_array_builder()
            .build(Arc::new(zarrs_storage::store::MemoryStore::new()), "/")?;
        array.store_chunk(&[0, 0], (0..64u16).collect::<Vec<_>>())?;
        let options = CodecOptions::default();
        let decoder = array.partial_decoder(&[0, 0])?;
        let cache = ArrayPartialDecoderCache::new(
            decoder.as_ref(),
            array.chunk_shape(&[0, 0])?,
            array.data_type().clone(),
            &options,
        )?;
        assert_eq!(
            ArrayPartialDecoderTraits::size_held(&cache),
            64 * size_of::<u16>()
        );
        let grid_shapes = |grids: Vec<Option<ChunkGrid>>| -> Vec<Option<Vec<u64>>> {
            grids
                .iter()
                .map(|grid| grid.as_ref().map(|grid| grid.grid_shape().to_vec()))
                .collect()
        };
        assert_eq!(
            grid_shapes(ArrayPartialDecoderSubchunkingTraits::local_subchunk_grids(
                &cache, &options
            )?),
            grid_shapes(decoder.local_subchunk_grids(&options)?)
        );
        assert!(matches!(
            ArrayPartialDecoderSubchunkingTraits::retrieve_encoded_subchunk_at_level(
                &cache,
                0,
                &[1, 0],
                &options
            ),
            Err(CodecError::UnsupportedEncodedSubchunk)
        ));
        Ok(())
    }

    #[cfg(feature = "async")]
    #[tokio::test]
    async fn async_array_partial_decoder_cache_holds_only_the_decoded_chunk()
    -> Result<(), Box<dyn std::error::Error>> {
        let array = sharded_array_builder()
            .build(Arc::new(zarrs_storage::store::AsyncMemoryStore::new()), "/")?;
        array
            .async_store_chunk(&[0, 0], (0..64u16).collect::<Vec<_>>())
            .await?;
        let options = CodecOptions::default();
        let decoder = array.async_partial_decoder(&[0, 0]).await?;
        let cache = ArrayPartialDecoderCache::async_new(
            decoder.as_ref(),
            array.chunk_shape(&[0, 0])?,
            array.data_type().clone(),
            &options,
        )
        .await?;
        assert_eq!(
            AsyncArrayPartialDecoderTraits::size_held(&cache),
            64 * size_of::<u16>()
        );
        assert!(matches!(
            AsyncArrayPartialDecoderSubchunkingTraits::retrieve_encoded_subchunk_at_level(
                &cache,
                0,
                &[1, 0],
                &options
            )
            .await,
            Err(CodecError::UnsupportedEncodedSubchunk)
        ));
        Ok(())
    }
}
