use std::sync::Arc;

use zarrs_registry::codec::BYTES;

use crate::{
    array::{
        codec::{
            ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, BytesPartialDecoderTraits,
            BytesPartialEncoderTraits, CodecError, CodecOptions,
        },
        ArrayBytes, ArraySize, ChunkRepresentation, DataType,
    },
    indexer::IncompatibleIndexerError,
};

#[cfg(feature = "async")]
use crate::array::codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};

use super::{reverse_endianness, Endianness};

/// Partial decoder for the `bytes` codec.
pub(crate) struct BytesCodecPartial<T: ?Sized> {
    input_output_handle: Arc<T>,
    decoded_representation: ChunkRepresentation,
    endian: Option<Endianness>,
}

impl<T: ?Sized> BytesCodecPartial<T> {
    /// Create a new partial decoder for the `bytes` codec.
    pub(crate) fn new(
        input_output_handle: Arc<T>,
        decoded_representation: ChunkRepresentation,
        endian: Option<Endianness>,
    ) -> Self {
        Self {
            input_output_handle,
            decoded_representation,
            endian,
        }
    }
}

impl<T: ?Sized> ArrayPartialDecoderTraits for BytesCodecPartial<T>
where
    T: BytesPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    fn size(&self) -> usize {
        self.input_output_handle.size()
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

        if indexer.dimensionality() != self.decoded_representation.dimensionality() {
            return Err(IncompatibleIndexerError::new_incompatible_dimensionality(
                indexer.dimensionality(),
                self.decoded_representation.dimensionality(),
            )
            .into());
        }

        let chunk_shape = self.decoded_representation.shape_u64();
        // Get byte ranges
        let byte_ranges = indexer.byte_ranges(&chunk_shape, data_type_size)?;

        // Decode
        let decoded = self
            .input_output_handle
            .partial_decode_many(Box::new(byte_ranges), options)?
            .map_or_else(
                || {
                    let array_size = ArraySize::new(
                        self.decoded_representation.data_type().size(),
                        indexer.len(),
                    );
                    ArrayBytes::new_fill_value(array_size, self.decoded_representation.fill_value())
                },
                |decoded| {
                    let mut decoded = decoded.concat();
                    if let Some(endian) = &self.endian {
                        if !endian.is_native() {
                            reverse_endianness(
                                &mut decoded,
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
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialDecoderTraits for BytesCodecPartial<T>
where
    T: AsyncBytesPartialDecoderTraits,
{
    fn data_type(&self) -> &DataType {
        self.decoded_representation.data_type()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let Some(data_type_size) = self.data_type().fixed_size() else {
            return Err(CodecError::UnsupportedDataType(
                self.data_type().clone(),
                BYTES.to_string(),
            ));
        };

        if indexer.dimensionality() != self.decoded_representation.dimensionality() {
            return Err(IncompatibleIndexerError::new_incompatible_dimensionality(
                indexer.dimensionality(),
                self.decoded_representation.dimensionality(),
            )
            .into());
        }

        let chunk_shape = self.decoded_representation.shape_u64();

        // Get byte ranges
        let byte_ranges = indexer.byte_ranges(&chunk_shape, data_type_size)?;

        // Decode
        let decoded = self
            .input_output_handle
            .partial_decode_many(Box::new(byte_ranges), options)
            .await?
            .map_or_else(
                || {
                    let array_size = ArraySize::new(
                        self.decoded_representation.data_type().size(),
                        indexer.len(),
                    );
                    ArrayBytes::new_fill_value(array_size, self.decoded_representation.fill_value())
                },
                |decoded| {
                    let mut decoded = decoded.concat();
                    if let Some(endian) = &self.endian {
                        if !endian.is_native() {
                            reverse_endianness(
                                &mut decoded,
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

impl<T> ArrayPartialEncoderTraits for BytesCodecPartial<T>
where
    T: BytesPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn ArrayPartialDecoderTraits> {
        self
    }

    fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase()
    }

    fn partial_encode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let Some(data_type_size) = self.data_type().fixed_size() else {
            return Err(CodecError::UnsupportedDataType(
                self.data_type().clone(),
                BYTES.to_string(),
            ));
        };

        if indexer.dimensionality() != self.decoded_representation.dimensionality() {
            return Err(IncompatibleIndexerError::new_incompatible_dimensionality(
                indexer.dimensionality(),
                self.decoded_representation.dimensionality(),
            )
            .into());
        }

        let chunk_shape = self.decoded_representation.shape_u64();
        let byte_ranges = indexer.byte_ranges(&chunk_shape, data_type_size)?;

        let mut bytes_to_encode = bytes.clone().into_fixed()?.to_vec();
        if let Some(endian) = &self.endian {
            if !endian.is_native() {
                reverse_endianness(
                    &mut bytes_to_encode,
                    self.decoded_representation.data_type(),
                );
            }
        }

        let total_size = u64::try_from(
            self.decoded_representation
                .size()
                .fixed_size()
                .expect("representation is fixed"),
        )
        .unwrap();
        let offset_bytes: Vec<_> = byte_ranges
            .zip(bytes_to_encode.chunks_exact(data_type_size))
            .map(|(range, chunk)| (range.start(total_size), crate::array::RawBytes::from(chunk)))
            .collect();

        self.input_output_handle
            .partial_encode_many(Box::new(offset_bytes.into_iter()), options)
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T> AsyncArrayPartialEncoderTraits for BytesCodecPartial<T>
where
    T: AsyncBytesPartialEncoderTraits,
{
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncArrayPartialDecoderTraits> {
        self
    }

    async fn erase(&self) -> Result<(), CodecError> {
        self.input_output_handle.erase().await
    }

    async fn partial_encode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let Some(data_type_size) = self.data_type().fixed_size() else {
            return Err(CodecError::UnsupportedDataType(
                self.data_type().clone(),
                BYTES.to_string(),
            ));
        };

        if indexer.dimensionality() != self.decoded_representation.dimensionality() {
            return Err(IncompatibleIndexerError::new_incompatible_dimensionality(
                indexer.dimensionality(),
                self.decoded_representation.dimensionality(),
            )
            .into());
        }

        let chunk_shape = self.decoded_representation.shape_u64();
        let byte_ranges = indexer.byte_ranges(&chunk_shape, data_type_size)?;

        let mut bytes_to_encode = bytes.clone().into_fixed()?.to_vec();
        if let Some(endian) = &self.endian {
            if !endian.is_native() {
                reverse_endianness(
                    &mut bytes_to_encode,
                    self.decoded_representation.data_type(),
                );
            }
        }

        let total_size = u64::try_from(
            self.decoded_representation
                .size()
                .fixed_size()
                .expect("representation is fixed"),
        )
        .unwrap();
        let offset_bytes: Vec<_> = byte_ranges
            .zip(bytes_to_encode.chunks_exact(data_type_size))
            .map(|(range, chunk)| (range.start(total_size), crate::array::RawBytes::from(chunk)))
            .collect();

        self.input_output_handle
            .partial_encode_many(Box::new(offset_bytes.into_iter()), options)
            .await
    }
}
