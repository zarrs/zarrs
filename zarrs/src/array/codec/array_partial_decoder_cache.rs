//! A cache for partial decoders.

use crate::array::{ArrayBytes, ChunkRepresentation, DataType};

use super::{ArrayPartialDecoderTraits, ArraySubset, CodecError, CodecOptions};

#[cfg(feature = "async")]
use super::AsyncArrayPartialDecoderTraits;

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
    fn size(&self) -> usize {
        self.cache.size()
    }

    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    fn partial_decode(
        &self,
        indexer: &ArraySubset,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let array_shape = self.decoded_representation.shape_u64();
        self.cache.extract_array_subset(
            indexer,
            &array_shape,
            self.decoded_representation.data_type(),
        )
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl AsyncArrayPartialDecoderTraits for ArrayPartialDecoderCache {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode(
        &self,
        indexer: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        ArrayPartialDecoderTraits::partial_decode(self, indexer, options)
    }
}
