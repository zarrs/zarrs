//! Zarr codecs.
//!
//! Array chunks can be encoded using a sequence of codecs, each of which specifies a bidirectional transform (an encode transform and a decode transform).
//! A codec can map array to an array, an array to bytes, or bytes to bytes.
//! A codec may support partial decoding to extract a byte range or array subset without needing to decode the entire input.
//!
//! A [`CodecChain`] represents a codec sequence consisting of any number of array to array and bytes to bytes codecs, and one array to bytes codec.
//! A codec chain is itself an array to bytes codec.
//! A cache may be inserted into a codec chain to optimise partial decoding where appropriate.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-encoding>.

pub mod array_to_array;
pub mod array_to_bytes;
pub mod bytes_to_bytes;
mod options;

mod named_codec;
pub use named_codec::{
    NamedArrayToArrayCodec, NamedArrayToBytesCodec, NamedBytesToBytesCodec, NamedCodec,
};

use derive_more::derive::Display;
pub use options::{CodecMetadataOptions, CodecOptions, CodecOptionsBuilder};

// Array to array
#[cfg(feature = "bitround")]
pub use array_to_array::bitround::*;
#[cfg(feature = "transpose")]
pub use array_to_array::transpose::*;
pub use array_to_array::{fixedscaleoffset::*, squeeze::*};

// Array to bytes
#[cfg(feature = "pcodec")]
pub use array_to_bytes::pcodec::*;
#[cfg(feature = "sharding")]
pub use array_to_bytes::sharding::*;
#[cfg(feature = "zfp")]
pub use array_to_bytes::zfp::*;
#[cfg(feature = "zfp")]
pub use array_to_bytes::zfpy::*;
pub use array_to_bytes::{
    bytes::*, codec_chain::CodecChain, packbits::*, vlen::*, vlen_array::*, vlen_bytes::*,
    vlen_utf8::*, vlen_v2::*,
};

// Bytes to bytes
#[cfg(feature = "adler32")]
pub use bytes_to_bytes::adler32::*;
#[cfg(feature = "blosc")]
pub use bytes_to_bytes::blosc::*;
#[cfg(feature = "bz2")]
pub use bytes_to_bytes::bz2::*;
#[cfg(feature = "crc32c")]
pub use bytes_to_bytes::crc32c::*;
#[cfg(feature = "fletcher32")]
pub use bytes_to_bytes::fletcher32::*;
#[cfg(feature = "gdeflate")]
pub use bytes_to_bytes::gdeflate::*;
#[cfg(feature = "gzip")]
pub use bytes_to_bytes::gzip::*;
pub use bytes_to_bytes::shuffle::*;
#[cfg(feature = "zlib")]
pub use bytes_to_bytes::zlib::*;
#[cfg(feature = "zstd")]
pub use bytes_to_bytes::zstd::*;

use thiserror::Error;

mod array_partial_decoder_cache;
mod bytes_partial_decoder_cache;
pub(crate) use array_partial_decoder_cache::ArrayPartialDecoderCache;
pub(crate) use bytes_partial_decoder_cache::BytesPartialDecoderCache;

mod byte_interval_partial_decoder;
pub use byte_interval_partial_decoder::ByteIntervalPartialDecoder;

#[cfg(feature = "async")]
pub use byte_interval_partial_decoder::AsyncByteIntervalPartialDecoder;

mod array_to_array_partial_encoder_default;
pub use array_to_array_partial_encoder_default::ArrayToArrayPartialEncoderDefault;
#[cfg(feature = "async")]
pub use array_to_array_partial_encoder_default::AsyncArrayToArrayPartialEncoderDefault;

mod array_to_bytes_partial_encoder_default;
pub use array_to_bytes_partial_encoder_default::ArrayToBytesPartialEncoderDefault;
#[cfg(feature = "async")]
pub use array_to_bytes_partial_encoder_default::AsyncArrayToBytesPartialEncoderDefault;
use zarrs_metadata::Configuration;

use crate::array_subset::IncompatibleDimensionalityError;
mod array_to_array_partial_decoder_default;
pub use array_to_array_partial_decoder_default::ArrayToArrayPartialDecoderDefault;
#[cfg(feature = "async")]
pub use array_to_array_partial_decoder_default::AsyncArrayToArrayPartialDecoderDefault;

mod array_to_bytes_partial_decoder_default;
pub use array_to_bytes_partial_decoder_default::ArrayToBytesPartialDecoderDefault;
#[cfg(feature = "async")]
pub use array_to_bytes_partial_decoder_default::AsyncArrayToBytesPartialDecoderDefault;

mod bytes_to_bytes_partial_encoder_default;
#[cfg(feature = "async")]
pub use bytes_to_bytes_partial_encoder_default::AsyncBytesToBytesPartialEncoderDefault;
pub use bytes_to_bytes_partial_encoder_default::BytesToBytesPartialEncoderDefault;

mod bytes_to_bytes_partial_decoder_default;
#[cfg(feature = "async")]
pub use bytes_to_bytes_partial_decoder_default::AsyncBytesToBytesPartialDecoderDefault;
pub use bytes_to_bytes_partial_decoder_default::BytesToBytesPartialDecoderDefault;

use zarrs_data_type::{DataTypeExtensionError, DataTypeFillValueError, FillValue};
use zarrs_metadata::{v3::MetadataV3, ArrayShape};
use zarrs_plugin::PluginUnsupportedError;
use zarrs_registry::ExtensionAliasesCodecV3;

use crate::config::global_config;
use crate::storage::{StoreKeyOffsetValue, WritableStorage};
use crate::{
    array_subset::{ArraySubset, IncompatibleArraySubsetAndShapeError},
    byte_range::{extract_byte_ranges_read_seek, ByteOffset, ByteRange, InvalidByteRangeError},
    plugin::{Plugin, PluginCreateError},
    storage::{ReadableStorage, StorageError, StoreKey},
};

#[cfg(feature = "async")]
use crate::storage::AsyncReadableStorage;

use std::any::Any;
use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

use super::ArraySize;
use super::{
    array_bytes::RawBytesOffsetsCreateError, concurrency::RecommendedConcurrency, ArrayBytes,
    ArrayBytesFixedDisjointView, BytesRepresentation, ChunkRepresentation, ChunkShape, DataType,
    RawBytes, RawBytesOffsetsOutOfBoundsError,
};

/// A codec plugin.
#[derive(derive_more::Deref)]
pub struct CodecPlugin(Plugin<Codec, MetadataV3>);
inventory::collect!(CodecPlugin);

impl CodecPlugin {
    /// Create a new [`CodecPlugin`].
    pub const fn new(
        identifier: &'static str,
        match_name_fn: fn(name: &str) -> bool,
        create_fn: fn(metadata: &MetadataV3) -> Result<Codec, PluginCreateError>,
    ) -> Self {
        Self(Plugin::new(identifier, match_name_fn, create_fn))
    }
}

/// A generic array to array, array to bytes, or bytes to bytes codec.
#[derive(Debug)]
pub enum Codec {
    /// An array to array codec.
    ArrayToArray(Arc<dyn ArrayToArrayCodecTraits>),
    /// An array to bytes codec.
    ArrayToBytes(Arc<dyn ArrayToBytesCodecTraits>),
    /// A bytes to bytes codec.
    BytesToBytes(Arc<dyn BytesToBytesCodecTraits>),
}

impl Codec {
    /// Create a codec from metadata.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if the metadata is invalid or not associated with a registered codec plugin.
    pub fn from_metadata(
        metadata: &MetadataV3,
        codec_aliases: &ExtensionAliasesCodecV3,
    ) -> Result<Self, PluginCreateError> {
        let identifier = codec_aliases.identifier(metadata.name());
        for plugin in inventory::iter::<CodecPlugin> {
            if plugin.match_name(identifier) {
                return plugin.create(metadata);
            }
        }
        #[cfg(miri)]
        {
            // Inventory does not work in miri, so manually handle all known codecs
            match metadata.name() {
                #[cfg(feature = "transpose")]
                codec::TRANSPOSE => {
                    return array_to_array::transpose::create_codec_transpose(metadata);
                }
                #[cfg(feature = "bitround")]
                codec::BITROUND => {
                    return array_to_array::bitround::create_codec_bitround(metadata);
                }
                codec::BYTES => {
                    return array_to_bytes::bytes::create_codec_bytes(metadata);
                }
                #[cfg(feature = "pcodec")]
                codec::PCODEC => {
                    return array_to_bytes::pcodec::create_codec_pcodec(metadata);
                }
                #[cfg(feature = "sharding")]
                codec::SHARDING => {
                    return array_to_bytes::sharding::create_codec_sharding(metadata);
                }
                #[cfg(feature = "zfp")]
                codec::ZFP => {
                    return array_to_bytes::zfp::create_codec_zfp(metadata);
                }
                #[cfg(feature = "zfp")]
                codec::ZFPY => {
                    return array_to_bytes::zfpy::create_codec_zfpy(metadata);
                }
                codec::VLEN => {
                    return array_to_bytes::vlen::create_codec_vlen(metadata);
                }
                codec::VLEN_V2 => {
                    return array_to_bytes::vlen_v2::create_codec_vlen_v2(metadata);
                }
                #[cfg(feature = "blosc")]
                codec::BLOSC => {
                    return bytes_to_bytes::blosc::create_codec_blosc(metadata);
                }
                #[cfg(feature = "bz2")]
                codec::BZ2 => {
                    return bytes_to_bytes::bz2::create_codec_bz2(metadata);
                }
                #[cfg(feature = "crc32c")]
                codec::CRC32C => {
                    return bytes_to_bytes::crc32c::create_codec_crc32c(metadata);
                }
                #[cfg(feature = "gdeflate")]
                codec::GDEFLATE => {
                    return bytes_to_bytes::gdeflate::create_codec_gdeflate(metadata);
                }
                #[cfg(feature = "gzip")]
                codec::GZIP => {
                    return bytes_to_bytes::gzip::create_codec_gzip(metadata);
                }
                #[cfg(feature = "zstd")]
                codec::ZSTD => {
                    return bytes_to_bytes::zstd::create_codec_zstd(metadata);
                }
                _ => {}
            }
        }
        Err(PluginUnsupportedError::new(metadata.name().to_string(), "codec".to_string()).into())
    }
}

/// Codec traits.
pub trait CodecTraits: Send + Sync {
    /// Unique identifier for the codec.
    fn identifier(&self) -> &str;

    /// The default name of the codec.
    fn default_name(&self) -> String {
        let identifier = self.identifier();
        global_config()
            .codec_aliases_v3()
            .default_name(identifier)
            .to_string()
    }

    /// Create the codec configuration.
    ///
    /// A hidden codec (e.g. a cache) will return [`None`], since it will not have any associated metadata.
    fn configuration_opt(
        &self,
        name: &str,
        options: &CodecMetadataOptions,
    ) -> Option<Configuration>;

    /// Create the codec configuration with default options.
    ///
    /// A hidden codec (e.g. a cache) will return [`None`], since it will not have any associated metadata.
    fn configuration(&self, name: &str) -> Option<Configuration> {
        self.configuration_opt(name, &CodecMetadataOptions::default())
    }

    /// Indicates if the input to a codecs partial decoder should be cached for optimal performance.
    /// If true, a cache may be inserted *before* it in a [`CodecChain`] partial decoder.
    fn partial_decoder_should_cache_input(&self) -> bool;

    /// Indicates if a partial decoder decodes all bytes from its input handle and its output should be cached for optimal performance.
    /// If true, a cache will be inserted at some point *after* it in a [`CodecChain`] partial decoder.
    fn partial_decoder_decodes_all(&self) -> bool;
}

/// Traits for both array to array and array to bytes codecs.
pub trait ArrayCodecTraits: CodecTraits {
    /// Return the recommended concurrency for the requested decoded representation.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the decoded representation is not valid for the codec.
    fn recommended_concurrency(
        &self,
        decoded_representation: &ChunkRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError>;

    /// Return the partial decode granularity.
    ///
    /// This represents the shape of the smallest subset of a chunk that can be efficiently decoded if the chunk were subdivided into a regular grid.
    /// For most codecs, this is just the shape of the chunk.
    /// It is the shape of the "inner chunks" for the sharding codec.
    fn partial_decode_granularity(
        &self,
        decoded_representation: &ChunkRepresentation,
    ) -> ChunkShape {
        decoded_representation.shape().into()
    }
}

/// Partial bytes decoder traits.
pub trait BytesPartialDecoderTraits: Any + Send + Sync {
    /// Returns the size of chunk bytes held by the partial decoder.
    fn size(&self) -> usize;

    /// Partially decode bytes.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or a byte range is invalid.
    fn partial_decode(
        &self,
        decoded_regions: &[ByteRange],
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError>;

    /// Partially decode bytes and concatenate.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// Codecs can manually implement this method with a preallocated array to reduce internal allocations.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or a byte range is invalid.
    fn partial_decode_concat(
        &self,
        decoded_regions: &[ByteRange],
        options: &CodecOptions,
    ) -> Result<Option<RawBytes<'_>>, CodecError> {
        Ok(self
            .partial_decode(decoded_regions, options)?
            .map(|vecs| Cow::Owned(vecs.concat())))
    }

    /// Decode all bytes.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    fn decode(&self, options: &CodecOptions) -> Result<Option<RawBytes<'_>>, CodecError> {
        Ok(self
            .partial_decode(&[ByteRange::FromStart(0, None)], options)?
            .map(|mut v| v.remove(0)))
    }
}

#[cfg(feature = "async")]
/// Asynchronous partial bytes decoder traits.
#[async_trait::async_trait]
pub trait AsyncBytesPartialDecoderTraits: Any + Send + Sync {
    /// Partially decode bytes.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or a byte range is invalid.
    async fn partial_decode(
        &self,
        decoded_regions: &[ByteRange],
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError>;

    /// Partially decode bytes and concatenate.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or a byte range is invalid.
    async fn partial_decode_concat(
        &self,
        decoded_regions: &[ByteRange],
        options: &CodecOptions,
    ) -> Result<Option<RawBytes<'_>>, CodecError> {
        Ok(self
            .partial_decode(decoded_regions, options)
            .await?
            .map(|vecs| Cow::Owned(vecs.concat())))
    }

    /// Decode all bytes.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    async fn decode(&self, options: &CodecOptions) -> Result<Option<RawBytes<'_>>, CodecError> {
        Ok(self
            .partial_decode(&[ByteRange::FromStart(0, None)], options)
            .await?
            .map(|mut v| v.remove(0)))
    }
}

/// Partial array decoder traits.
pub trait ArrayPartialDecoderTraits: Any + Send + Sync {
    /// Return the data type of the partial decoder.
    fn data_type(&self) -> &DataType;

    /// Returns the size of chunk bytes held by the partial decoder.
    fn size(&self) -> usize;

    /// Partially decode a chunk.
    ///
    /// If the inner `input_handle` is a bytes decoder and partial decoding returns [`None`], then the array subsets have the fill value.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or an array subset is invalid.
    fn partial_decode(
        &self,
        indexer: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError>;

    /// Partially decode into a preallocated output.
    ///
    /// This method is intended for internal use by Array.
    /// It currently only works for fixed length data types.
    ///
    /// The `array_subset` shape and dimensionality does not need to match `output_subset`, but the number of elements must match.
    /// Extracted elements from the `array_subset` are written to the subset of the output in C order.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the number of elements in `array_subset` does not match the number of elements in `output_view`,
    fn partial_decode_into(
        &self,
        indexer: &ArraySubset,
        output_view: &mut ArrayBytesFixedDisjointView<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if indexer.num_elements() != output_view.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                indexer.num_elements(),
                output_view.num_elements(),
            )
            .into());
        }

        let decoded_value = self.partial_decode(indexer, options)?;
        if let ArrayBytes::Fixed(decoded_value) = decoded_value {
            output_view.copy_from_slice(&decoded_value)?;
            Ok(())
        } else {
            Err(CodecError::ExpectedFixedLengthBytes)
        }
    }
}

/// Partial array encoder traits.
pub trait ArrayPartialEncoderTraits: Any + Send + Sync {
    /// Erase the chunk.
    ///
    /// # Errors
    /// Returns an error if there is an underlying store error.
    fn erase(&self) -> Result<(), CodecError>;

    /// Partially encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or an array subset is invalid.
    fn partial_encode(
        &self,
        indexer: &ArraySubset,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;
}

#[cfg(feature = "async")]
/// Asynchronous partial array encoder traits.
#[async_trait::async_trait]
pub trait AsyncArrayPartialEncoderTraits: Any + Send + Sync {
    /// Erase the chunk.
    ///
    /// # Errors
    /// Returns an error if there is an underlying store error.
    async fn erase(&self) -> Result<(), CodecError>;

    /// Partially encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or an array subset is invalid.
    async fn partial_encode(
        &self,
        indexer: &ArraySubset,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;
}

/// Partial bytes encoder traits.
pub trait BytesPartialEncoderTraits: Any + Send + Sync {
    /// Erase the chunk.
    ///
    /// # Errors
    /// Returns an error if there is an underlying store error.
    fn erase(&self) -> Result<(), CodecError>;

    /// Partially encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or an array subset is invalid.
    fn partial_encode(
        &self,
        offsets_and_bytes: &[(ByteOffset, crate::array::RawBytes<'_>)],
        options: &CodecOptions,
    ) -> Result<(), CodecError>;
}

#[cfg(feature = "async")]
/// Asynhronous partial bytes encoder traits.
#[async_trait::async_trait]
pub trait AsyncBytesPartialEncoderTraits: Any + Send + Sync {
    /// Erase the chunk.
    ///
    /// # Errors
    /// Returns an error if there is an underlying store error.
    async fn erase(&self) -> Result<(), CodecError>;

    /// Partially encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or an array subset is invalid.
    async fn partial_encode(
        &self,
        offsets_and_bytes: &[(ByteOffset, crate::array::RawBytes<'_>)],
        options: &CodecOptions,
    ) -> Result<(), CodecError>;
}

#[cfg(feature = "async")]
/// Asynchronous partial array decoder traits.
#[async_trait::async_trait]
pub trait AsyncArrayPartialDecoderTraits: Any + Send + Sync {
    /// Return the data type of the partial decoder.
    fn data_type(&self) -> &DataType;

    /// Partially decode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails, array subset is invalid, or the array subset shape does not match array view subset shape.
    async fn partial_decode(
        &self,
        indexer: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError>;

    /// Async variant of [`ArrayPartialDecoderTraits::partial_decode_into`].
    #[allow(clippy::missing_safety_doc)]
    async fn partial_decode_into(
        &self,
        indexer: &ArraySubset,
        output_view: &mut ArrayBytesFixedDisjointView<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if indexer.num_elements() != output_view.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                output_view.num_elements(),
                indexer.num_elements(),
            )
            .into());
        }
        let decoded_value = self.partial_decode(indexer, options).await?;
        if let ArrayBytes::Fixed(decoded_value) = decoded_value {
            output_view.copy_from_slice(&decoded_value)?;
            Ok(())
        } else {
            Err(CodecError::ExpectedFixedLengthBytes)
        }
    }
}

/// A [`ReadableStorage`] store value partial decoder.
pub struct StoragePartialDecoder {
    storage: ReadableStorage,
    key: StoreKey,
}

impl StoragePartialDecoder {
    /// Create a new storage partial decoder.
    pub fn new(storage: ReadableStorage, key: StoreKey) -> Self {
        Self { storage, key }
    }
}

impl BytesPartialDecoderTraits for StoragePartialDecoder {
    fn size(&self) -> usize {
        0
    }

    fn partial_decode(
        &self,
        decoded_regions: &[ByteRange],
        _options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        Ok(self
            .storage
            .get_partial_values_key(&self.key, decoded_regions)?
            .map(|vec_bytes| {
                vec_bytes
                    .into_iter()
                    .map(|bytes| Cow::Owned(bytes.to_vec()))
                    .collect()
            }))
    }
}

#[cfg(feature = "async")]
/// An [`AsyncReadableStorage`] store value partial decoder.
pub struct AsyncStoragePartialDecoder {
    storage: AsyncReadableStorage,
    key: StoreKey,
}

#[cfg(feature = "async")]
impl AsyncStoragePartialDecoder {
    /// Create a new storage partial decoder.
    pub fn new(storage: AsyncReadableStorage, key: StoreKey) -> Self {
        Self { storage, key }
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl AsyncBytesPartialDecoderTraits for AsyncStoragePartialDecoder {
    async fn partial_decode(
        &self,
        decoded_regions: &[ByteRange],
        _options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        Ok(self
            .storage
            .get_partial_values_key(&self.key, decoded_regions)
            .await?
            .map(|vec_bytes| {
                vec_bytes
                    .into_iter()
                    .map(|bytes| Cow::Owned(bytes.to_vec()))
                    .collect()
            }))
    }
}

/// A [`WritableStorage`] store value partial encoder.
pub struct StoragePartialEncoder {
    storage: WritableStorage,
    key: StoreKey,
}

impl StoragePartialEncoder {
    /// Create a new storage partial encoder.
    pub fn new(storage: WritableStorage, key: StoreKey) -> Self {
        Self { storage, key }
    }
}

impl BytesPartialEncoderTraits for StoragePartialEncoder {
    fn erase(&self) -> Result<(), CodecError> {
        Ok(self.storage.erase(&self.key)?)
    }

    fn partial_encode(
        &self,
        offsets_and_bytes: &[(ByteOffset, crate::array::RawBytes<'_>)],
        _options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let key_offset_values = offsets_and_bytes
            .iter()
            .map(|(offset, bytes)| StoreKeyOffsetValue::new(self.key.clone(), *offset, bytes))
            .collect::<Vec<_>>();
        Ok(self.storage.set_partial_values(&key_offset_values)?)
    }
}

/// Traits for array to array codecs.
#[cfg_attr(feature = "async", async_trait::async_trait)]
pub trait ArrayToArrayCodecTraits: ArrayCodecTraits + core::fmt::Debug {
    /// Return a dynamic version of the codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits>;

    /// Returns the encoded data type for a given decoded data type.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the data type is not supported by this codec.
    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError>;

    /// Returns the encoded fill value for a given decoded fill value
    ///
    /// The encoded fill value is computed by applying [`ArrayToArrayCodecTraits::encode`] to the `decoded_fill_value`.
    /// This may need to be implemented manually if a codec does not support encoding a single element or the encoding is otherwise dependent on the chunk shape.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the data type is not supported by this codec.
    fn encoded_fill_value(
        &self,
        decoded_data_type: &DataType,
        decoded_fill_value: &FillValue,
    ) -> Result<FillValue, CodecError> {
        let element_representation = ChunkRepresentation::new(
            vec![unsafe { NonZeroU64::new_unchecked(1) }],
            decoded_data_type.clone(),
            decoded_fill_value.clone(),
        )
        .map_err(|err| CodecError::Other(err.to_string()))?;

        // Calculate the changed fill value
        let fill_value = self
            .encode(
                ArrayBytes::new_fill_value(
                    ArraySize::new(decoded_data_type.size(), 1),
                    decoded_fill_value,
                ),
                &element_representation,
                &CodecOptions::default(),
            )?
            .into_fixed()?
            .into_owned();
        Ok(FillValue::new(fill_value))
    }

    /// Returns the shape of the encoded chunk for a given decoded chunk shape.
    ///
    /// The default implementation returns the shape unchanged.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the shape is not supported by this codec.
    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        Ok(decoded_shape.to_vec().into())
    }

    /// Returns the shape of the decoded chunk for a given encoded chunk shape.
    ///
    /// The default implementation returns the shape unchanged.
    ///
    /// Returns [`None`] if the decoded shape cannot be determined from the encoded shape.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the shape is not supported by this codec.
    fn decoded_shape(
        &self,
        encoded_shape: &[NonZeroU64],
    ) -> Result<Option<ChunkShape>, CodecError> {
        Ok(Some(encoded_shape.to_vec().into()))
    }

    /// Returns the encoded chunk representation given the decoded chunk representation.
    ///
    /// The default implementation returns the chunk representation from the outputs of
    /// - [`encoded_data_type`](ArrayToArrayCodecTraits::encoded_data_type),
    /// - [`encoded_fill_value`](ArrayToArrayCodecTraits::encoded_fill_value), and
    /// - [`encoded_shape`](ArrayToArrayCodecTraits::encoded_shape).
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the decoded chunk representation is not supported by this codec.
    fn encoded_representation(
        &self,
        decoded_representation: &ChunkRepresentation,
    ) -> Result<ChunkRepresentation, CodecError> {
        Ok(ChunkRepresentation::new(
            self.encoded_shape(decoded_representation.shape())?.into(),
            self.encoded_data_type(decoded_representation.data_type())?,
            self.encoded_fill_value(
                decoded_representation.data_type(),
                decoded_representation.fill_value(),
            )?,
        )?)
    }

    /// Encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or `bytes` is incompatible with `decoded_representation`.
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Decode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the decoded output is incompatible with `decoded_representation`.
    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Initialise a partial decoder.
    ///
    /// The default implementation decodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToArrayPartialDecoderDefault::new(
            input_handle,
            decoded_representation.clone(),
            self.into_dyn(),
        )))
    }

    /// Initialise a partial encoder.
    ///
    /// The default implementation reencodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    fn partial_encoder(
        self: Arc<Self>,
        input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToArrayPartialEncoderDefault::new(
            input_handle,
            output_handle,
            decoded_representation.clone(),
            self.into_dyn(),
        )))
    }

    #[cfg(feature = "async")]
    /// Initialise an asynchronous partial decoder.
    ///
    /// The default implementation decodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(AsyncArrayToArrayPartialDecoderDefault::new(
            input_handle,
            decoded_representation.clone(),
            self.into_dyn(),
        )))
    }

    #[cfg(feature = "async")]
    /// Initialise an asynchronous partial encoder.
    ///
    /// The default implementation reencodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    fn async_partial_encoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(AsyncArrayToArrayPartialEncoderDefault::new(
            input_handle,
            output_handle,
            decoded_representation.clone(),
            self.into_dyn(),
        )))
    }
}

/// Traits for array to bytes codecs.
#[cfg_attr(feature = "async", async_trait::async_trait)]
pub trait ArrayToBytesCodecTraits: ArrayCodecTraits + core::fmt::Debug {
    /// Return a dynamic version of the codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits>;

    /// Returns the size of the encoded representation given a size of the decoded representation.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the decoded representation is not supported by this codec.
    fn encoded_representation(
        &self,
        decoded_representation: &ChunkRepresentation,
    ) -> Result<BytesRepresentation, CodecError>;

    /// Encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or `bytes` is incompatible with `decoded_representation`.
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError>;

    /// Decode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the decoded output is incompatible with `decoded_representation`.
    fn decode<'a>(
        &self,
        bytes: RawBytes<'a>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Decode into a subset of a preallocated output.
    ///
    /// This method is intended for internal use by Array.
    /// It currently only works for fixed length data types.
    ///
    /// The decoded representation shape and dimensionality does not need to match `output_subset`, but the number of elements must match.
    /// Chunk elements are written to the subset of the output in C order.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the number of elements in `decoded_representation` does not match the number of elements in `output_view`,
    fn decode_into(
        &self,
        bytes: RawBytes<'_>,
        decoded_representation: &ChunkRepresentation,
        output_view: &mut ArrayBytesFixedDisjointView<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if decoded_representation.num_elements() != output_view.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                output_view.num_elements(),
                decoded_representation.num_elements(),
            )
            .into());
        }
        let decoded_value = self.decode(bytes, decoded_representation, options)?;
        if let ArrayBytes::Fixed(decoded_value) = decoded_value {
            output_view.copy_from_slice(&decoded_value)?;
        } else {
            return Err(CodecError::ExpectedFixedLengthBytes);
        }
        Ok(())
    }

    /// Initialise a partial decoder.
    ///
    /// The default implementation decodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToBytesPartialDecoderDefault::new(
            input_handle,
            decoded_representation.clone(),
            self.into_dyn(),
        )))
    }

    /// Initialise a partial encoder.
    ///
    /// The default implementation reencodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    fn partial_encoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        output_handle: Arc<dyn BytesPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToBytesPartialEncoderDefault::new(
            input_handle,
            output_handle,
            decoded_representation.clone(),
            self.into_dyn(),
        )))
    }

    #[cfg(feature = "async")]
    /// Initialise an asynchronous partial decoder.
    ///
    /// The default implementation decodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(AsyncArrayToBytesPartialDecoderDefault::new(
            input_handle,
            decoded_representation.clone(),
            self.into_dyn(),
        )))
    }

    #[cfg(feature = "async")]
    /// Initialise an asynchronous partial encoder.
    ///
    /// The default implementation reencodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(AsyncArrayToBytesPartialEncoderDefault::new(
            input_handle,
            output_handle,
            decoded_representation.clone(),
            self.into_dyn(),
        )))
    }
}

/// Traits for bytes to bytes codecs.
#[cfg_attr(feature = "async", async_trait::async_trait)]
pub trait BytesToBytesCodecTraits: CodecTraits + core::fmt::Debug {
    /// Return a dynamic version of the codec.
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits>;

    /// Return the maximum internal concurrency supported for the requested decoded representation.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the decoded representation is not valid for the codec.
    fn recommended_concurrency(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> Result<RecommendedConcurrency, CodecError>;

    /// Returns the size of the encoded representation given a size of the decoded representation.
    fn encoded_representation(
        &self,
        decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation;

    /// Encode chunk bytes.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    fn encode<'a>(
        &self,
        decoded_value: RawBytes<'a>,
        options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError>;

    /// Decode chunk bytes.
    //
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    fn decode<'a>(
        &self,
        encoded_value: RawBytes<'a>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<RawBytes<'a>, CodecError>;

    /// Initialises a partial decoder.
    ///
    /// The default implementation decodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn BytesPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(BytesToBytesPartialDecoderDefault::new(
            input_handle,
            *decoded_representation,
            self.into_dyn(),
        )))
    }

    /// Initialise a partial encoder.
    ///
    /// The default implementation reencodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    fn partial_encoder(
        self: Arc<Self>,
        input_handle: Arc<dyn BytesPartialDecoderTraits>,
        output_handle: Arc<dyn BytesPartialEncoderTraits>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn BytesPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(BytesToBytesPartialEncoderDefault::new(
            input_handle,
            output_handle,
            *decoded_representation,
            self.into_dyn(),
        )))
    }

    #[cfg(feature = "async")]
    /// Initialises an asynchronous partial decoder.
    ///
    /// The default implementation decodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    async fn async_partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncBytesPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(AsyncBytesToBytesPartialDecoderDefault::new(
            input_handle,
            *decoded_representation,
            self.into_dyn(),
        )))
    }

    #[cfg(feature = "async")]
    /// Initialise an asynchronous partial encoder.
    ///
    /// The default implementation reencodes the entire chunk.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if initialisation fails.
    #[allow(unused_variables)]
    async fn async_partial_encoder(
        self: Arc<Self>,
        input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncBytesPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(AsyncBytesToBytesPartialEncoderDefault::new(
            input_handle,
            output_handle,
            *decoded_representation,
            self.into_dyn(),
        )))
    }
}

impl<T> BytesPartialDecoderTraits for T
where
    T: AsRef<[u8]> + Send + Sync + 'static,
{
    fn size(&self) -> usize {
        self.as_ref().len()
    }

    fn partial_decode(
        &self,
        decoded_regions: &[ByteRange],
        _parallel: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        Ok(Some(
            extract_byte_ranges_read_seek(std::io::Cursor::new(self), decoded_regions)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl<T> AsyncBytesPartialDecoderTraits for T
where
    T: AsRef<[u8]> + Send + Sync + 'static,
{
    async fn partial_decode(
        &self,
        decoded_regions: &[ByteRange],
        _parallel: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        Ok(Some(
            extract_byte_ranges_read_seek(std::io::Cursor::new(self), decoded_regions)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
    }
}

/// An error indicating the length of bytes does not match the expected length.
#[derive(Clone, Debug, Display, Error)]
#[display("Invalid bytes len {len}, expected {expected_len}")]
pub struct InvalidBytesLengthError {
    len: usize,
    expected_len: usize,
}

impl InvalidBytesLengthError {
    /// Create a new [`InvalidBytesLengthError`].
    #[must_use]
    pub fn new(len: usize, expected_len: usize) -> Self {
        Self { len, expected_len }
    }
}

/// An error indicating the shape is not compatible with the expected number of elements.
#[derive(Clone, Debug, Display, Error)]
#[display("Invalid shape {shape:?} for number of elements {expected_num_elements}")]
pub struct InvalidArrayShapeError {
    shape: ArrayShape,
    expected_num_elements: usize,
}

impl InvalidArrayShapeError {
    /// Create a new [`InvalidArrayShapeError`].
    #[must_use]
    pub fn new(shape: ArrayShape, expected_num_elements: usize) -> Self {
        Self {
            shape,
            expected_num_elements,
        }
    }
}

/// An error indicating the length of elements does not match the expected length.
#[derive(Clone, Debug, Display, Error)]
#[display("Invalid number of elements {num}, expected {expected}")]
pub struct InvalidNumberOfElementsError {
    num: u64,
    expected: u64,
}

impl InvalidNumberOfElementsError {
    /// Create a new [`InvalidNumberOfElementsError`].
    #[must_use]
    pub fn new(num: u64, expected: u64) -> Self {
        Self { num, expected }
    }
}

/// An array subset is out of bounds.
#[derive(Clone, Debug, Display, Error)]
#[display("Subset {subset} is out of bounds of {must_be_within}")]
pub struct SubsetOutOfBoundsError {
    subset: ArraySubset,
    must_be_within: ArraySubset,
}

impl SubsetOutOfBoundsError {
    /// Create a new [`InvalidNumberOfElementsError`].
    #[must_use]
    pub fn new(subset: ArraySubset, must_be_within: ArraySubset) -> Self {
        Self {
            subset,
            must_be_within,
        }
    }
}

/// A codec error.
#[non_exhaustive]
#[derive(Clone, Debug, Error)]
pub enum CodecError {
    /// An error creating a subset while decoding
    #[error(transparent)]
    IncompatibleDimensionalityError(#[from] IncompatibleDimensionalityError),
    /// An IO error.
    #[error(transparent)]
    IOError(#[from] Arc<std::io::Error>),
    /// An invalid byte range was requested.
    #[error(transparent)]
    InvalidByteRangeError(#[from] InvalidByteRangeError),
    /// An invalid array subset was requested.
    #[error(transparent)]
    InvalidArraySubsetError(#[from] IncompatibleArraySubsetAndShapeError),
    /// An invalid array subset was requested with the wrong dimensionality.
    #[error("the array subset {_0} has the wrong dimensionality, expected {_1}")]
    InvalidArraySubsetDimensionalityError(ArraySubset, usize),
    /// The decoded size of a chunk did not match what was expected.
    #[error("the size of a decoded chunk is {}, expected {}", _0.len, _0.expected_len)]
    UnexpectedChunkDecodedSize(#[from] InvalidBytesLengthError),
    /// An embedded checksum does not match the decoded value.
    #[error("the checksum is invalid")]
    InvalidChecksum,
    /// A store error.
    #[error(transparent)]
    StorageError(#[from] StorageError),
    /// Unsupported data type
    #[error("Unsupported data type {0} for codec {1}")]
    UnsupportedDataType(DataType, String),
    /// Offsets are not [`None`] with a fixed length data type.
    #[error("Offsets are invalid or are not compatible with the data type (e.g. fixed-sized data types)")]
    InvalidOffsets,
    /// Other
    #[error("{_0}")]
    Other(String),
    /// Invalid variable sized array offsets.
    #[error("Invalid variable sized array offsets")]
    InvalidVariableSizedArrayOffsets,
    /// Expected fixed length bytes.
    #[error("Expected fixed length array bytes")]
    ExpectedFixedLengthBytes,
    /// Expected variable length bytes.
    #[error("Expected variable length array bytes")]
    ExpectedVariableLengthBytes,
    /// Invalid array shape.
    #[error(transparent)]
    InvalidArrayShape(#[from] InvalidArrayShapeError),
    /// Invalid number of elements.
    #[error(transparent)]
    InvalidNumberOfElements(#[from] InvalidNumberOfElementsError),
    /// Subset out of bounds.
    #[error(transparent)]
    SubsetOutOfBounds(#[from] SubsetOutOfBoundsError),
    /// Invalid byte offsets for variable length data.
    #[error(transparent)]
    RawBytesOffsetsCreate(#[from] RawBytesOffsetsCreateError),
    /// Variable length array bytes offsets are out of bounds.
    #[error(transparent)]
    RawBytesOffsetsOutOfBounds(#[from] RawBytesOffsetsOutOfBoundsError),
    /// A data type extension error.
    #[error(transparent)]
    DataTypeExtension(#[from] DataTypeExtensionError),
    /// An incompatible fill value error
    #[error(transparent)]
    DataTypeFillValueError(#[from] DataTypeFillValueError),
}

impl From<std::io::Error> for CodecError {
    fn from(err: std::io::Error) -> Self {
        Self::IOError(Arc::new(err))
    }
}

impl From<&str> for CodecError {
    fn from(err: &str) -> Self {
        Self::Other(err.to_string())
    }
}

impl From<String> for CodecError {
    fn from(err: String) -> Self {
        Self::Other(err)
    }
}
