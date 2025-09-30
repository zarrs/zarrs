use std::{borrow::Cow, sync::Arc};

use zarrs_registry::codec::BYTES;
use zarrs_storage::{byte_range::ByteRange, StorageError};

use crate::{
    array::{
        codec::{
            ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, BytesPartialDecoderTraits,
            BytesPartialEncoderTraits, CodecError, CodecOptions,
        },
        update_array_bytes, ArrayBytes, ArraySize, ChunkRepresentation, DataType,
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

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
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
        let byte_ranges = indexer
            .iter_contiguous_byte_ranges(&chunk_shape, data_type_size)?
            .map(ByteRange::new);

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

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
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

    async fn exists(&self) -> Result<bool, StorageError> {
        self.input_output_handle.exists().await
    }

    fn size_held(&self) -> usize {
        self.input_output_handle.size_held()
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
        let byte_ranges = indexer
            .iter_contiguous_byte_ranges(&chunk_shape, data_type_size)?
            .map(ByteRange::new);

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

    fn supports_partial_decode(&self) -> bool {
        self.input_output_handle.supports_partial_decode()
    }
}

impl<T: ?Sized> ArrayPartialEncoderTraits for BytesCodecPartial<T>
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

        // If the chunk is empty, initialise the chunk with the fill value and update
        if self.input_output_handle.exists()? {
            let chunk_shape = self.decoded_representation.shape_u64();
            let byte_ranges = indexer.iter_contiguous_byte_ranges(&chunk_shape, data_type_size)?;

            let mut bytes_to_encode = bytes.clone().into_fixed()?.to_vec();
            if let Some(endian) = &self.endian {
                if !endian.is_native() {
                    reverse_endianness(
                        &mut bytes_to_encode,
                        self.decoded_representation.data_type(),
                    );
                }
            }

            let offset_bytes: Vec<_> = byte_ranges
                .scan(0usize, |offset_in, range_out| {
                    let len = usize::try_from(range_out.end - range_out.start).unwrap();
                    let range_in = *offset_in..*offset_in + len;
                    *offset_in += len;
                    Some((
                        range_out.start,
                        crate::array::RawBytes::from(&bytes_to_encode[range_in]),
                    ))
                })
                .collect();

            self.input_output_handle
                .partial_encode_many(Box::new(offset_bytes.into_iter()), options)
        } else {
            // Create a chunk filled with the fill value
            let array_size = ArraySize::new(
                self.decoded_representation.data_type().size(),
                self.decoded_representation.num_elements(),
            );
            let chunk_bytes =
                ArrayBytes::new_fill_value(array_size, self.decoded_representation.fill_value());
            let chunk_bytes = update_array_bytes(
                chunk_bytes,
                &self.decoded_representation.shape_u64(),
                indexer,
                bytes,
                self.decoded_representation.data_type().size(),
            )?;
            let mut chunk_bytes: Vec<u8> = chunk_bytes
                .into_fixed()
                .expect("fixed data type")
                .into_owned();

            if let Some(endian) = &self.endian {
                if !endian.is_native() {
                    reverse_endianness(&mut chunk_bytes, self.decoded_representation.data_type());
                }
            }

            self.input_output_handle
                .partial_encode(0, Cow::Owned(chunk_bytes), options)
        }
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<T: ?Sized> AsyncArrayPartialEncoderTraits for BytesCodecPartial<T>
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

        // If the chunk is empty, initialise the chunk with the fill value and update
        if self.input_output_handle.exists().await? {
            let chunk_shape = self.decoded_representation.shape_u64();
            let byte_ranges = indexer.iter_contiguous_byte_ranges(&chunk_shape, data_type_size)?;

            let mut bytes_to_encode = bytes.clone().into_fixed()?.to_vec();
            if let Some(endian) = &self.endian {
                if !endian.is_native() {
                    reverse_endianness(
                        &mut bytes_to_encode,
                        self.decoded_representation.data_type(),
                    );
                }
            }

            let offset_bytes: Vec<_> = byte_ranges
                .scan(0usize, |offset_in, range_out| {
                    let len = usize::try_from(range_out.end - range_out.start).unwrap();
                    let range_in = *offset_in..*offset_in + len;
                    *offset_in += len;
                    Some((
                        range_out.start,
                        crate::array::RawBytes::from(&bytes_to_encode[range_in]),
                    ))
                })
                .collect();

            self.input_output_handle
                .partial_encode_many(Box::new(offset_bytes.into_iter()), options)
                .await
        } else {
            // Create a chunk filled with the fill value
            let array_size = ArraySize::new(
                self.decoded_representation.data_type().size(),
                self.decoded_representation.num_elements(),
            );
            let chunk_bytes =
                ArrayBytes::new_fill_value(array_size, self.decoded_representation.fill_value());
            let chunk_bytes = update_array_bytes(
                chunk_bytes,
                &self.decoded_representation.shape_u64(),
                indexer,
                bytes,
                self.decoded_representation.data_type().size(),
            )?;
            let mut chunk_bytes: Vec<u8> = chunk_bytes
                .into_fixed()
                .expect("fixed data type")
                .into_owned();

            if let Some(endian) = &self.endian {
                if !endian.is_native() {
                    reverse_endianness(&mut chunk_bytes, self.decoded_representation.data_type());
                }
            }

            self.input_output_handle
                .partial_encode(0, Cow::Owned(chunk_bytes), options)
                .await
        }
    }

    fn supports_partial_encode(&self) -> bool {
        self.input_output_handle.supports_partial_encode()
    }
}
