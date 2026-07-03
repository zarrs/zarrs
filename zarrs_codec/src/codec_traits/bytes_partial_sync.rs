use std::any::Any;
use std::borrow::Cow;
use std::sync::{Arc, Mutex};

use zarrs_plugin::{MaybeSend, MaybeSync};
use zarrs_storage::byte_range::{
    ByteRange, ByteRangeIterator, InvalidByteRangeError, extract_byte_ranges,
};
use zarrs_storage::{OffsetBytesIterator, StorageError};

use crate::{ArrayBytesRaw, CodecError, CodecOptions};

/// Partial bytes decoder traits.
pub trait BytesPartialDecoderTraits: Any + MaybeSend + MaybeSync {
    /// Returns whether the chunk exists.
    ///
    /// # Errors
    /// Returns [`StorageError`] if a storage operation fails.
    fn exists(&self) -> Result<bool, StorageError>;

    /// Returns the size of chunk bytes held by the partial decoder.
    ///
    /// Intended for use by size-constrained partial decoder caches.
    fn size_held(&self) -> usize;

    /// Partially decode a byte range.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the byte range is invalid.
    fn partial_decode(
        &self,
        decoded_region: ByteRange,
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytesRaw<'_>>, CodecError> {
        Ok(self
            .partial_decode_many(Box::new([decoded_region].into_iter()), options)?
            .map(|mut v| v.pop().expect("single byte range")))
    }

    /// Partially decode byte ranges.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or a byte range is invalid.
    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError>;

    /// Decode all bytes.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    fn decode(&self, options: &CodecOptions) -> Result<Option<ArrayBytesRaw<'_>>, CodecError> {
        self.partial_decode(ByteRange::FromStart(0, None), options)
    }

    /// Returns whether this decoder supports partial decoding.
    ///
    /// If this returns `true`, the decoder can efficiently handle partial decoding operations.
    /// If this returns `false`, partial decoding will fall back to a full decode operation.
    fn supports_partial_decode(&self) -> bool;
}

/// Partial bytes encoder traits.
pub trait BytesPartialEncoderTraits:
    BytesPartialDecoderTraits + Any + MaybeSend + MaybeSync
{
    /// Return the encoder as an [`Arc<BytesPartialDecoderTraits>`].
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn BytesPartialDecoderTraits>;

    /// Erase the chunk.
    ///
    /// # Errors
    /// Returns an error if there is an underlying store error.
    fn erase(&self) -> Result<(), CodecError>;

    /// Partially encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the byte range is invalid.
    fn partial_encode(
        &self,
        offset: u64,
        bytes: ArrayBytesRaw<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.partial_encode_many(Box::new([(offset, bytes)].into_iter()), options)
    }

    /// Partially encode a chunk from an [`OffsetBytesIterator`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or a byte range is invalid.
    fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<ArrayBytesRaw<'_>>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;

    /// Returns whether this encoder supports partial encoding.
    ///
    /// If this returns `true`, the encoder can efficiently handle partial encoding operations.
    /// If this returns `false`, partial encoding will fall back to a full decode and encode operation.
    fn supports_partial_encode(&self) -> bool;
}

impl BytesPartialDecoderTraits for Cow<'static, [u8]> {
    fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.as_ref().len()
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        _parallel: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError> {
        Ok(Some(
            extract_byte_ranges(self, decoded_regions)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

impl BytesPartialDecoderTraits for Vec<u8> {
    fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.len()
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        _parallel: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError> {
        Ok(Some(
            extract_byte_ranges(self, decoded_regions)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

impl BytesPartialDecoderTraits for Mutex<Option<Vec<u8>>> {
    fn exists(&self) -> Result<bool, StorageError> {
        Ok(self.lock().unwrap().is_some())
    }

    fn size_held(&self) -> usize {
        self.lock().unwrap().as_ref().map_or(0, Vec::len)
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError> {
        if let Some(input) = self.lock().unwrap().as_ref() {
            let size = input.len() as u64;
            let mut outputs = vec![];
            for byte_range in decoded_regions {
                if byte_range.end(size) <= size {
                    outputs.push(Cow::Owned(input[byte_range.to_range_usize(size)].into()));
                } else {
                    return Err(InvalidByteRangeError::new(byte_range, size).into());
                }
            }
            Ok(Some(outputs))
        } else {
            Ok(None)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

impl BytesPartialEncoderTraits for Mutex<Option<Vec<u8>>> {
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn BytesPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), CodecError> {
        *self.lock().unwrap() = None;
        Ok(())
    }

    fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<ArrayBytesRaw<'_>>,
        _options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let mut v = self.lock().unwrap();
        let mut output = v.as_ref().cloned().unwrap_or_default();

        for (offset, value) in offset_values {
            let offset = usize::try_from(offset).unwrap();
            if output.len() < offset + value.len() {
                output.resize(offset + value.len(), 0);
            }
            output[offset..offset + value.len()].copy_from_slice(&value);
        }
        *v = Some(output);
        Ok(())
    }

    fn supports_partial_encode(&self) -> bool {
        true
    }
}
