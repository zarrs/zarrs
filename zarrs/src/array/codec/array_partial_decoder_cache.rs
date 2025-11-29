//! A cache for partial decoders.

#[cfg(feature = "async")]
use super::AsyncArrayPartialDecoderTraits;
use super::{ArrayPartialDecoderTraits, ArraySubset, CodecError, CodecOptions};
use crate::array::{ArrayBytes, ChunkRepresentation, DataType};
use crate::storage::StorageError;

/// A cache for an [`ArrayPartialDecoderTraits`] partial decoder.
pub(crate) struct ArrayPartialDecoderCache {
    decoded_representation: ChunkRepresentation,
    cache: ArrayBytes<'static>,
}

impl ArrayPartialDecoderCache {
    /// Create a new partial decoder cache.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation of the partial decoder fails.
    pub(crate) fn new(
        input_handle: &dyn ArrayPartialDecoderTraits,
        decoded_representation: ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Self, CodecError> {
        let bytes = input_handle
            .partial_decode(
                &ArraySubset::new_with_shape(decoded_representation.shape_u64()),
                options,
            )?
            .into_owned();
        Ok(Self {
            decoded_representation,
            cache: bytes,
        })
    }

    #[cfg(feature = "async")]
    /// Create a new asynchronous partial decoder cache.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation of the partial decoder fails.
    pub(crate) async fn async_new(
        input_handle: &dyn AsyncArrayPartialDecoderTraits,
        decoded_representation: ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<ArrayPartialDecoderCache, CodecError> {
        let bytes = input_handle
            .partial_decode(
                &ArraySubset::new_with_shape(decoded_representation.shape_u64()),
                options,
            )
            .await?
            .into_owned();
        Ok(Self {
            decoded_representation,
            cache: bytes,
        })
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
        self.decoded_representation.data_type()
    }

    fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let array_shape = self.decoded_representation.shape_u64();
        self.cache.extract_array_subset(
            indexer,
            &array_shape,
            self.decoded_representation.data_type(),
        )
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncArrayPartialDecoderTraits for ArrayPartialDecoderCache {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.cache.size()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        ArrayPartialDecoderTraits::partial_decode(self, indexer, options)
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}
