use std::sync::Arc;

use zarrs_registry::codec::BYTES;

use crate::array::{
    codec::{ArrayPartialDecoderTraits, BytesPartialDecoderTraits, CodecError, CodecOptions},
    ArrayBytes, ArraySize, ChunkRepresentation, DataType,
};

#[cfg(feature = "async")]
use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};

use super::{reverse_endianness, Endianness};

/// Partial decoder for the `bytes` codec.
pub(crate) struct BytesPartialDecoder {
    input_handle: Arc<dyn BytesPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    endian: Option<Endianness>,
}

impl BytesPartialDecoder {
    /// Create a new partial decoder for the `bytes` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
        endian: Option<Endianness>,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
            endian,
        }
    }
}

impl ArrayPartialDecoderTraits for BytesPartialDecoder {
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
        let Some(data_type_size) = self.data_type().fixed_size() else {
            return Err(CodecError::UnsupportedDataType(
                self.data_type().clone(),
                BYTES.to_string(),
            ));
        };

        let chunk_shape = self.decoded_representation.shape_u64();
        // Get byte ranges
        let mut byte_ranges = indexer.byte_ranges(&chunk_shape, data_type_size)?;

        // Decode
        let decoded = self
            .input_handle
            .partial_decode_concat(&mut byte_ranges, options)?
            .map_or_else(
                || {
                    let array_size = ArraySize::new(
                        self.decoded_representation.data_type().size(),
                        indexer.len(),
                    );
                    ArrayBytes::new_fill_value(array_size, self.decoded_representation.fill_value())
                },
                |mut decoded| {
                    if let Some(endian) = &self.endian {
                        if !endian.is_native() {
                            reverse_endianness(
                                decoded.to_mut(),
                                self.decoded_representation.data_type(),
                            );
                        }
                    }
                    ArrayBytes::from(decoded)
                },
            );

        Ok(decoded)
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial decoder for the `bytes` codec.
pub(crate) struct AsyncBytesPartialDecoder {
    input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
    decoded_representation: ChunkRepresentation,
    endian: Option<Endianness>,
}

#[cfg(feature = "async")]
impl AsyncBytesPartialDecoder {
    /// Create a new partial decoder for the `bytes` codec.
    pub(crate) fn new(
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: ChunkRepresentation,
        endian: Option<Endianness>,
    ) -> Self {
        Self {
            input_handle,
            decoded_representation,
            endian,
        }
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl AsyncArrayPartialDecoderTraits for AsyncBytesPartialDecoder {
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        let Some(data_type_size) = self.data_type().fixed_size() else {
            return Err(CodecError::UnsupportedDataType(
                self.data_type().clone(),
                BYTES.to_string(),
            ));
        };

        if indexer.dimensionality() != self.decoded_representation.dimensionality() {
            return Err(CodecError::InvalidIndexerDimensionalityError(
                indexer.dimensionality(),
                self.decoded_representation.dimensionality(),
            ));
        }

        let chunk_shape = self.decoded_representation.shape_u64();

        // Get byte ranges
        let mut byte_ranges = indexer.byte_ranges(&chunk_shape, data_type_size)?;

        // Decode
        let decoded = self
            .input_handle
            .partial_decode_concat(&mut byte_ranges, options)
            .await?
            .map_or_else(
                || {
                    let array_size = ArraySize::new(
                        self.decoded_representation.data_type().size(),
                        indexer.len(),
                    );
                    ArrayBytes::new_fill_value(array_size, self.decoded_representation.fill_value())
                },
                |mut decoded| {
                    if let Some(endian) = &self.endian {
                        if !endian.is_native() {
                            reverse_endianness(
                                decoded.to_mut(),
                                self.decoded_representation.data_type(),
                            );
                        }
                    }
                    ArrayBytes::from(decoded)
                },
            );

        Ok(decoded)
    }
}
