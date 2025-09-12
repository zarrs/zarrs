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
//!
#![doc = include_str!("../../doc/status/codecs.md")]

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

mod codec_partial_default;
pub use codec_partial_default::CodecPartialDefault;
use codec_partial_default::{
    ArrayToArrayCodecPartialDefault, ArrayToBytesCodecPartialDefault,
    BytesToBytesCodecPartialDefault,
};
use zarrs_metadata::Configuration;

use zarrs_data_type::{DataTypeExtensionError, DataTypeFillValueError, FillValue};
use zarrs_metadata::{v3::MetadataV3, ArrayShape};
use zarrs_plugin::PluginUnsupportedError;
use zarrs_registry::ExtensionAliasesCodecV3;
use zarrs_storage::byte_range::extract_byte_ranges;
use zarrs_storage::OffsetBytesIterator;
use zarrs_storage::{MaybeSend, MaybeSync};

use crate::config::global_config;
use crate::{
    array_subset::{ArraySubset, IncompatibleDimensionalityError},
    indexer::IncompatibleIndexerError,
    plugin::{Plugin, PluginCreateError},
    storage::byte_range::{ByteRange, ByteRangeIterator, InvalidByteRangeError},
    storage::{ReadableStorage, ReadableWritableStorage, StorageError, StoreKey},
};

#[cfg(feature = "async")]
use crate::storage::AsyncReadableStorage;

use std::any::Any;
use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::Mutex;

use super::ArraySize;
use super::{
    array_bytes::RawBytesOffsetsCreateError, concurrency::RecommendedConcurrency, ArrayBytes,
    ArrayBytesFixedDisjointView, BytesRepresentation, ChunkRepresentation, ChunkShape, DataType,
    RawBytes, RawBytesOffsetsOutOfBoundsError,
};

/// Describes the partial decoding capabilities of a codec.
///
/// The capability describes:
/// - `partial_read`: Whether the codec can perform partial reading from its input.
/// - `partial_decode`: Whether the codec supports partial decoding decoding, or it must decode the entire input.
///
/// If `partial_read` is false and `partial_decode` is true, input should be cached for optimal performance.
/// If `partial_decode` is false, a cache should be inserted after this codec in a [`CodecChain`] partial decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartialDecoderCapability {
    /// Whether the codec can perform partial reading from its input.
    /// If false, the codec needs to read all input data before decoding.
    pub partial_read: bool,
    /// Whether the codec supports partial decoding operations.
    /// If false, the codec needs to decode the entire input.
    pub partial_decode: bool,
}

/// Describes the partial encoding capabilities of a codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PartialEncoderCapability {
    /// Whether the codec supports partial encoding operations.
    ///
    /// If this returns `true`, the codec can efficiently handle partial encoding operations if supported by the parent codec or storage handle.
    /// If this returns `false`, partial encoding will fall back to a full decode and encode operation.
    pub partial_encode: bool,
}

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
pub trait CodecTraits: MaybeSend + MaybeSync {
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

    /// Returns the partial decoder capability of this codec.
    fn partial_decoder_capability(&self) -> PartialDecoderCapability;

    /// Returns the partial encoder capability of this codec.
    fn partial_encoder_capability(&self) -> PartialEncoderCapability;
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
pub trait BytesPartialDecoderTraits: Any + MaybeSend + MaybeSync {
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
    ) -> Result<Option<RawBytes<'_>>, CodecError> {
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
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError>;

    /// Decode all bytes.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    fn decode(&self, options: &CodecOptions) -> Result<Option<RawBytes<'_>>, CodecError> {
        self.partial_decode(ByteRange::FromStart(0, None), options)
    }

    /// Returns whether this decoder supports partial decoding.
    ///
    /// If this returns `true`, the decoder can efficiently handle partial decoding operations.
    /// If this returns `false`, partial decoding will fall back to a full decode operation.
    fn supports_partial_decode(&self) -> bool;
}

#[cfg(feature = "async")]
/// Asynchronous partial bytes decoder traits.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncBytesPartialDecoderTraits: Any + MaybeSend + MaybeSync {
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
    async fn partial_decode<'a>(
        &'a self,
        decoded_region: ByteRange,
        options: &CodecOptions,
    ) -> Result<Option<RawBytes<'a>>, CodecError> {
        Ok(self
            .partial_decode_many(Box::new([decoded_region].into_iter()), options)
            .await?
            .map(|mut v| v.pop().expect("single byte range")))
    }

    /// Partially decode byte ranges.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or a byte range is invalid.
    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'a>>>, CodecError>;

    /// Decode all bytes.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    async fn decode<'a>(
        &'a self,
        options: &CodecOptions,
    ) -> Result<Option<RawBytes<'a>>, CodecError> {
        self.partial_decode(ByteRange::FromStart(0, None), options)
            .await
    }
}

/// Partial array decoder traits.
pub trait ArrayPartialDecoderTraits: Any + MaybeSend + MaybeSync {
    /// Return the data type of the partial decoder.
    fn data_type(&self) -> &DataType;

    /// Returns the size of chunk bytes held by the partial decoder.
    ///
    /// Intended for use by size-constrained partial decoder caches.
    fn size_held(&self) -> usize;

    /// Partially decode a chunk.
    ///
    /// If the inner `input_handle` is a bytes decoder and partial decoding returns [`None`], then the array subsets have the fill value.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or an array subset is invalid.
    fn partial_decode(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError>;

    /// Partially decode into a preallocated output.
    ///
    /// This method is intended for internal use by Array.
    /// It currently only works for fixed length data types.
    ///
    /// The `indexer` shape and dimensionality does not need to match `output_subset`, but the number of elements must match.
    /// Extracted elements from the `indexer` are written as ordered by the indexer.
    /// For an [`ArraySubset`], that is C order.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the number of elements in `indexer` does not match the number of elements in `output_view`,
    fn partial_decode_into(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        output_view: &mut ArrayBytesFixedDisjointView<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if indexer.len() != output_view.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                indexer.len(),
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
pub trait ArrayPartialEncoderTraits:
    ArrayPartialDecoderTraits + Any + MaybeSend + MaybeSync
{
    /// Return the encoder as an [`Arc<ArrayPartialDecoderTraits>`].
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn ArrayPartialDecoderTraits>;

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
        indexer: &dyn crate::indexer::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;

    /// Returns whether this encoder supports partial encoding.
    ///
    /// If this returns `true`, the encoder can efficiently handle partial encoding operations.
    /// If this returns `false`, partial encoding will fall back to a full decode and encode operation.
    fn supports_partial_encode(&self) -> bool;
}

#[cfg(feature = "async")]
/// Asynchronous partial array encoder traits.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncArrayPartialEncoderTraits:
    AsyncArrayPartialDecoderTraits + Any + MaybeSend + MaybeSync
{
    /// Return the encoder as an [`Arc<AsyncArrayPartialDecoderTraits>`].
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncArrayPartialDecoderTraits>;

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
        indexer: &dyn crate::indexer::Indexer,
        bytes: &ArrayBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;

    /// Returns whether this encoder supports partial encoding.
    ///
    /// If this returns `true`, the encoder can efficiently handle partial encoding operations.
    /// If this returns `false`, partial encoding will fall back to a full decode and encode operation.
    fn supports_partial_encode(&self) -> bool;
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
        bytes: crate::array::RawBytes<'_>,
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
        offset_values: OffsetBytesIterator<crate::array::RawBytes<'_>>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;

    /// Returns whether this encoder supports partial encoding.
    ///
    /// If this returns `true`, the encoder can efficiently handle partial encoding operations.
    /// If this returns `false`, partial encoding will fall back to a full decode and encode operation.
    fn supports_partial_encode(&self) -> bool;
}

#[cfg(feature = "async")]
/// Asynhronous partial bytes encoder traits.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncBytesPartialEncoderTraits:
    AsyncBytesPartialDecoderTraits + Any + MaybeSend + MaybeSync
{
    /// Return the encoder as an [`Arc<AsyncBytesPartialDecoderTraits>`].
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncBytesPartialDecoderTraits>;

    /// Erase the chunk.
    ///
    /// # Errors
    /// Returns an error if there is an underlying store error.
    async fn erase(&self) -> Result<(), CodecError>;

    /// Partially encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the byte range is invalid.
    async fn partial_encode(
        &self,
        offset: u64,
        bytes: crate::array::RawBytes<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        self.partial_encode_many(Box::new([(offset, bytes)].into_iter()), options)
            .await
    }

    /// Partially encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or a byte range is invalid.
    async fn partial_encode_many<'a>(
        &'a self,
        offset_values: OffsetBytesIterator<'a, crate::array::RawBytes<'_>>,
        options: &CodecOptions,
    ) -> Result<(), CodecError>;

    /// Returns whether this encoder supports partial encoding.
    ///
    /// If this returns `true`, the encoder can efficiently handle partial encoding operations.
    /// If this returns `false`, partial encoding will fall back to a full decode and encode operation.
    fn supports_partial_encode(&self) -> bool;
}

#[cfg(feature = "async")]
/// Asynchronous partial array decoder traits.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncArrayPartialDecoderTraits: Any + MaybeSend + MaybeSync {
    /// Return the data type of the partial decoder.
    fn data_type(&self) -> &DataType;

    /// Returns the size of chunk bytes held by the partial decoder.
    ///
    /// Intended for use by size-constrained partial decoder caches.
    fn size_held(&self) -> usize;

    /// Partially decode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails, array subset is invalid, or the array subset shape does not match array view subset shape.
    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::indexer::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Async variant of [`ArrayPartialDecoderTraits::partial_decode_into`].
    #[allow(clippy::missing_safety_doc)]
    async fn partial_decode_into(
        &self,
        indexer: &dyn crate::indexer::Indexer,
        output_view: &mut ArrayBytesFixedDisjointView<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if indexer.len() != output_view.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                output_view.num_elements(),
                indexer.len(),
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
    fn size_held(&self) -> usize {
        0
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let bytes = self.storage.get_partial_many(&self.key, decoded_regions)?;
        if let Some(bytes) = bytes {
            let bytes = bytes
                .map(|b| Ok::<_, StorageError>(Cow::Owned(b?.to_vec())))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Some(bytes))
        } else {
            Ok(None)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.storage.supports_get_partial()
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
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for AsyncStoragePartialDecoder {
    fn size_held(&self) -> usize {
        0
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'a>>>, CodecError> {
        let bytes = self
            .storage
            .get_partial_many(&self.key, decoded_regions)
            .await?;
        Ok(if let Some(bytes) = bytes {
            use futures::{StreamExt, TryStreamExt};
            Some(
                bytes
                    .map(|bytes| Ok::<_, StorageError>(Cow::Owned(bytes?.to_vec())))
                    .try_collect()
                    .await?,
            )
        } else {
            None
        })
    }
}

impl BytesPartialDecoderTraits for Mutex<Option<Vec<u8>>> {
    fn size_held(&self) -> usize {
        self.lock().unwrap().as_ref().map_or(0, Vec::len)
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        if let Some(input) = self.lock().unwrap().as_ref() {
            let size = input.len() as u64;
            let mut outputs = vec![];
            for byte_range in decoded_regions {
                if byte_range.end(size) <= size {
                    outputs.push(Cow::Owned(input[byte_range.to_range_usize(size)].to_vec()));
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
        offset_values: OffsetBytesIterator<crate::array::RawBytes<'_>>,
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

/// A [`ReadableWritableStorage`] store value partial encoder.
pub struct StoragePartialEncoder {
    storage: ReadableWritableStorage,
    key: StoreKey,
}

impl StoragePartialEncoder {
    /// Create a new storage partial encoder.
    pub fn new(storage: ReadableWritableStorage, key: StoreKey) -> Self {
        Self { storage, key }
    }
}

impl BytesPartialDecoderTraits for StoragePartialEncoder {
    fn size_held(&self) -> usize {
        0
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
        let results = self.storage.get_partial_many(&self.key, decoded_regions)?;
        if let Some(results) = results {
            Ok(Some(
                results
                    .into_iter()
                    .map(|bytes| Ok::<_, StorageError>(Cow::Owned(bytes?.to_vec())))
                    .collect::<Result<Vec<_>, _>>()?,
            ))
        } else {
            Ok(None)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.storage.supports_get_partial()
    }
}

impl BytesPartialEncoderTraits for StoragePartialEncoder {
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn BytesPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), CodecError> {
        Ok(self.storage.erase(&self.key)?)
    }

    fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<crate::array::RawBytes<'_>>,
        _options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let offset_values = offset_values
            .into_iter()
            .map(|(offset, bytes)| (offset, bytes::Bytes::from(bytes.into_owned())));
        Ok(self
            .storage
            .set_partial_many(&self.key, Box::new(offset_values))?)
    }

    fn supports_partial_encode(&self) -> bool {
        self.storage.supports_set_partial()
    }
}

/// Traits for array to array codecs.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
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
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
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
        input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_output_handle,
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
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
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
        input_output_handle: Arc<dyn AsyncArrayPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_output_handle,
            decoded_representation.clone(),
            self.into_dyn(),
        )))
    }
}

/// Traits for array to bytes codecs.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
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
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
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
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
            input_output_handle,
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
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
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
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        decoded_representation: &ChunkRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
            input_output_handle,
            decoded_representation.clone(),
            self.into_dyn(),
        )))
    }
}

/// Traits for bytes to bytes codecs.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
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
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new(
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
        input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn BytesPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new(
            input_output_handle,
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
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new(
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
        input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncBytesPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new(
            input_output_handle,
            *decoded_representation,
            self.into_dyn(),
        )))
    }
}

impl BytesPartialDecoderTraits for Cow<'static, [u8]> {
    fn size_held(&self) -> usize {
        self.as_ref().len()
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        _parallel: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
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
    fn size_held(&self) -> usize {
        self.len()
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        _parallel: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'_>>>, CodecError> {
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

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for Cow<'static, [u8]> {
    fn size_held(&self) -> usize {
        self.as_ref().len()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        _parallel: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'a>>>, CodecError> {
        Ok(Some(
            extract_byte_ranges(self, decoded_regions)?
                .into_iter()
                .map(Cow::Owned)
                .collect(),
        ))
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for Vec<u8> {
    fn size_held(&self) -> usize {
        self.len()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        _parallel: &CodecOptions,
    ) -> Result<Option<Vec<RawBytes<'a>>>, CodecError> {
        Ok(Some(
            extract_byte_ranges(self, decoded_regions)?
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
    /// The indexer is invalid (e.g. incorrect dimensionality / out-of-bounds access).
    #[error(transparent)]
    IncompatibleIndexer(#[from] IncompatibleIndexerError),
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
