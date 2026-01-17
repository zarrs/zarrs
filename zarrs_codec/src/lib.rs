//! The codec API for the [`zarrs`](https://docs.rs/zarrs/latest/zarrs/index.html) crate.
//!
//! ## Licence
//! `zarrs_codec` is licensed under either of
//!  - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_codec/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//!  - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_codec/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

mod options;

mod codec_partial_default;
pub use codec_partial_default::CodecPartialDefault;
use codec_partial_default::{
    ArrayToArrayCodecPartialDefault, ArrayToBytesCodecPartialDefault,
    BytesToBytesCodecPartialDefault,
};

mod array_bytes_fixed_disjoint_view;
pub use array_bytes_fixed_disjoint_view::{
    ArrayBytesFixedDisjointView, ArrayBytesFixedDisjointViewCreateError,
};

mod recommended_concurrency;
pub use recommended_concurrency::RecommendedConcurrency;

mod bytes_representation;
pub use bytes_representation::BytesRepresentation;

mod array_bytes;
pub use array_bytes::{
    ArrayBytes, ArrayBytesError, ArrayBytesOffsets, ArrayBytesOptional, ArrayBytesRaw,
    ArrayBytesRawOffsetsCreateError, ArrayBytesRawOffsetsOutOfBoundsError,
    ArrayBytesVariableLength, build_nested_optional_target, copy_fill_value_into,
    decode_into_array_bytes_target, extract_decoded_regions_vlen, merge_chunks_vlen,
    merge_chunks_vlen_optional, optional_nesting_depth, update_array_bytes,
};

mod byte_interval_partial_decoder;
#[cfg(feature = "async")]
pub use byte_interval_partial_decoder::AsyncByteIntervalPartialDecoder;
pub use byte_interval_partial_decoder::ByteIntervalPartialDecoder;

use derive_more::derive::Display;
pub use options::{CodecMetadataOptions, CodecOptions};
use thiserror::Error;
use zarrs_metadata::{ArrayShape, ChunkShape, Configuration};
use zarrs_storage::byte_range::{
    ByteRange, ByteRangeIterator, InvalidByteRangeError, extract_byte_ranges,
};
#[cfg(feature = "async")]
use zarrs_storage::{AsyncReadableStorage, AsyncReadableWritableStorage};
use zarrs_storage::{
    OffsetBytesIterator, ReadableStorage, ReadableWritableStorage, StorageError, StoreKey,
};

use std::any::Any;
use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::{Arc, LazyLock, Mutex};

use zarrs_chunk_grid::{
    ArraySubset, ArraySubsetError, IncompatibleDimensionalityError, Indexer, IndexerError,
};
use zarrs_data_type::{DataType, DataTypeFillValueError, FillValue};
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::{
    ExtensionAliases, ExtensionName, MaybeSend, MaybeSync, Plugin, PluginCreateError,
    PluginUnsupportedError, RuntimePlugin, RuntimeRegistry, ZarrVersion, ZarrVersion2,
    ZarrVersion3,
};

/// A target for decoding array bytes into a preallocated output view.
///
/// This enum mirrors the structure of [`ArrayBytes`] to support decoding fixed-length
/// and optional data types into preallocated views.
// #[non_exhaustive]
pub enum ArrayBytesDecodeIntoTarget<'a> {
    /// Target for non-optional (fixed-length) data.
    Fixed(&'a mut ArrayBytesFixedDisjointView<'a>),

    /// Target for optional data (nested data target + mask view).
    ///
    /// The mask is always fixed-length (one byte per element).
    Optional(
        Box<ArrayBytesDecodeIntoTarget<'a>>,
        &'a mut ArrayBytesFixedDisjointView<'a>,
    ),
}

impl ArrayBytesDecodeIntoTarget<'_> {
    /// Return the number of elements in the target.
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        match self {
            Self::Fixed(data) => data.num_elements(),
            Self::Optional(data, _) => data.num_elements(),
        }
    }
}

impl<'a> From<&'a mut ArrayBytesFixedDisjointView<'a>> for ArrayBytesDecodeIntoTarget<'a> {
    fn from(view: &'a mut ArrayBytesFixedDisjointView<'a>) -> Self {
        Self::Fixed(view)
    }
}

/// Describes the partial decoding capabilities of a codec.
///
/// The capability describes:
/// - `partial_read`: Whether the codec can perform partial reading from its input.
/// - `partial_decode`: Whether the codec supports partial decoding decoding, or it must decode the entire input.
///
/// If `partial_read` is false and `partial_decode` is true, input should be cached for optimal performance.
/// If `partial_decode` is false, a cache should be inserted after this codec in a codec chain partial decoder.
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

/// A Zarr V3 codec plugin.
#[derive(derive_more::Deref)]
pub struct CodecPluginV3(Plugin<Codec, MetadataV3>);
inventory::collect!(CodecPluginV3);

impl CodecPluginV3 {
    /// Create a new [`CodecPluginV3`] for a type implementing [`ExtensionAliases<ZarrVersion3>`].
    pub const fn new<T: ExtensionAliases<ZarrVersion3> + CodecTraitsV3>() -> Self {
        Self(Plugin::new(T::matches_name, T::create))
    }
}

/// A Zarr V2 codec plugin.
#[derive(derive_more::Deref)]
pub struct CodecPluginV2(Plugin<Codec, MetadataV2>);
inventory::collect!(CodecPluginV2);

impl CodecPluginV2 {
    /// Create a new [`CodecPluginV2`] for a type implementing [`ExtensionAliases<ZarrVersion2>`].
    pub const fn new<T: ExtensionAliases<ZarrVersion2> + CodecTraitsV2>() -> Self {
        Self(Plugin::new(T::matches_name, T::create))
    }
}

/// A runtime V3 codec plugin for dynamic registration.
pub type CodecRuntimePluginV3 = RuntimePlugin<Codec, MetadataV3>;

/// A runtime V2 codec plugin for dynamic registration.
pub type CodecRuntimePluginV2 = RuntimePlugin<Codec, MetadataV2>;

/// Global runtime registry for V3 codec plugins.
pub static CODEC_RUNTIME_REGISTRY_V3: LazyLock<RuntimeRegistry<CodecRuntimePluginV3>> =
    LazyLock::new(RuntimeRegistry::new);

/// Global runtime registry for V2 codec plugins.
pub static CODEC_RUNTIME_REGISTRY_V2: LazyLock<RuntimeRegistry<CodecRuntimePluginV2>> =
    LazyLock::new(RuntimeRegistry::new);

/// A handle to a registered V3 codec plugin.
pub type CodecRuntimeRegistryHandleV3 = Arc<CodecRuntimePluginV3>;

/// A handle to a registered V2 codec plugin.
pub type CodecRuntimeRegistryHandleV2 = Arc<CodecRuntimePluginV2>;

/// Register a V3 codec plugin at runtime.
///
/// Runtime-registered plugins take precedence over compile-time registered plugins.
///
/// # Returns
///
/// A handle that can be used to unregister the plugin later.
///
/// # Example
///
/// ```ignore
/// use zarrs::array::codec::{register_codec_v3, CodecRuntimePluginV3, Codec};
///
/// let handle = register_codec_v3(CodecRuntimePluginV3::new(
///     |name| name == "my.custom.codec",
///     |metadata| Ok(Codec::BytesToBytes(Arc::new(MyCodec::from_metadata(metadata)?))),
/// ));
/// ```
pub fn register_codec_v3(plugin: CodecRuntimePluginV3) -> CodecRuntimeRegistryHandleV3 {
    CODEC_RUNTIME_REGISTRY_V3.register(plugin)
}

/// Register a V2 codec plugin at runtime.
///
/// Runtime-registered plugins take precedence over compile-time registered plugins.
///
/// # Returns
///
/// A handle that can be used to unregister the plugin later.
pub fn register_codec_v2(plugin: CodecRuntimePluginV2) -> CodecRuntimeRegistryHandleV2 {
    CODEC_RUNTIME_REGISTRY_V2.register(plugin)
}

/// Unregister a runtime V3 codec plugin.
///
/// # Returns
/// `true` if the plugin was found and removed, `false` otherwise.
pub fn unregister_codec_v3(handle: &CodecRuntimeRegistryHandleV3) -> bool {
    CODEC_RUNTIME_REGISTRY_V3.unregister(handle)
}

/// Unregister a runtime V2 codec plugin.
///
/// # Returns
/// `true` if the plugin was found and removed, `false` otherwise.
pub fn unregister_codec_v2(handle: &CodecRuntimeRegistryHandleV2) -> bool {
    CODEC_RUNTIME_REGISTRY_V2.unregister(handle)
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

/// Codec metadata for different Zarr versions.
#[derive(Debug, Clone, Copy, derive_more::From)]
pub enum CodecMetadata<'a> {
    /// Zarr V3 metadata.
    V3(&'a MetadataV3),
    /// Zarr V2 metadata.
    V2(&'a MetadataV2),
}

impl Codec {
    /// Create a codec from metadata.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if the metadata is invalid or not associated with a registered codec plugin.
    pub fn from_metadata<'a>(
        metadata: impl Into<CodecMetadata<'a>>,
    ) -> Result<Self, PluginCreateError> {
        match metadata.into() {
            CodecMetadata::V2(metadata) => Self::from_metadata_v2(metadata),
            CodecMetadata::V3(metadata) => Self::from_metadata_v3(metadata),
        }
    }

    /// Create a codec from V3 metadata.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if the metadata is invalid or not associated with a registered codec plugin.
    fn from_metadata_v3(metadata: &MetadataV3) -> Result<Self, PluginCreateError> {
        let name = metadata.name();

        // Check runtime registry first (higher priority)
        {
            let result = CODEC_RUNTIME_REGISTRY_V3.with_plugins(|plugins| {
                for plugin in plugins {
                    if plugin.match_name(name) {
                        return Some(plugin.create(metadata));
                    }
                }
                None
            });
            if let Some(result) = result {
                return result;
            }
        }

        // Fall back to compile-time registered plugins
        for plugin in inventory::iter::<CodecPluginV3> {
            if plugin.match_name(name) {
                return plugin.create(metadata);
            }
        }
        Err(PluginUnsupportedError::new(metadata.name().to_string(), "codec".to_string()).into())
    }

    /// Create a codec from V2 metadata.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if the metadata is invalid or not associated with a registered codec plugin.
    fn from_metadata_v2(metadata: &MetadataV2) -> Result<Self, PluginCreateError> {
        let name = metadata.id();

        // Check runtime registry first (higher priority)
        {
            let result = CODEC_RUNTIME_REGISTRY_V2.with_plugins(|plugins| {
                for plugin in plugins {
                    if plugin.match_name(name) {
                        return Some(plugin.create(metadata));
                    }
                }
                None
            });
            if let Some(result) = result {
                return result;
            }
        }

        // Fall back to compile-time registered plugins
        for plugin in inventory::iter::<CodecPluginV2> {
            if plugin.match_name(name) {
                return plugin.create(metadata);
            }
        }
        Err(PluginUnsupportedError::new(metadata.id().to_string(), "codec".to_string()).into())
    }

    /// Create the codec configuration.
    #[must_use]
    pub fn configuration(
        &self,
        version: ZarrVersion,
        options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        match self {
            Self::ArrayToArray(codec) => codec.configuration(version, options),
            Self::ArrayToBytes(codec) => codec.configuration(version, options),
            Self::BytesToBytes(codec) => codec.configuration(version, options),
        }
    }

    /// Create the Zarr V3 codec configuration.
    #[must_use]
    pub fn configuration_v3(&self, options: &CodecMetadataOptions) -> Option<Configuration> {
        self.configuration(ZarrVersion::V3, options)
    }

    /// Create the Zarr V2 codec configuration.
    #[must_use]
    pub fn configuration_v2(&self, options: &CodecMetadataOptions) -> Option<Configuration> {
        self.configuration(ZarrVersion::V2, options)
    }
}

impl ExtensionName for Codec {
    fn name(&self, version: ZarrVersion) -> Option<std::borrow::Cow<'static, str>> {
        match self {
            Self::ArrayToArray(codec) => codec.name(version),
            Self::ArrayToBytes(codec) => codec.name(version),
            Self::BytesToBytes(codec) => codec.name(version),
        }
    }
}

/// Trait for creating a codec from Zarr V2 metadata.
///
/// Types implementing this trait can be registered as V2 codec plugins via [`CodecPluginV2`].
pub trait CodecTraitsV2 {
    /// Create a codec from Zarr V2 metadata.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if the plugin cannot be created.
    fn create(metadata: &MetadataV2) -> Result<Codec, PluginCreateError>
    where
        Self: Sized;
}

/// Trait for creating a codec from Zarr V3 metadata.
///
/// Types implementing this trait can be registered as V3 codec plugins via [`CodecPluginV3`].
pub trait CodecTraitsV3 {
    /// Create a codec from Zarr V3 metadata.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if the plugin cannot be created.
    fn create(metadata: &MetadataV3) -> Result<Codec, PluginCreateError>
    where
        Self: Sized;
}

/// Codec traits.
pub trait CodecTraits: ExtensionName + MaybeSend + MaybeSync {
    /// Returns this codec as [`Any`].
    fn as_any(&self) -> &dyn Any;

    /// Create the codec configuration.
    ///
    /// A hidden codec (e.g. a cache) will return [`None`], since it will not have any associated metadata.
    fn configuration(
        &self,
        version: ZarrVersion,
        options: &CodecMetadataOptions,
    ) -> Option<Configuration>;

    /// Create the Zarr V3 codec configuration.
    fn configuration_v3(&self, options: &CodecMetadataOptions) -> Option<Configuration> {
        self.configuration(ZarrVersion::V3, options)
    }

    /// Create the Zarr V2 codec configuration.
    fn configuration_v2(&self, options: &CodecMetadataOptions) -> Option<Configuration> {
        self.configuration(ZarrVersion::V2, options)
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
        shape: &[NonZeroU64],
        data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError>;

    /// Return the partial decode granularity.
    ///
    /// This represents the shape of the smallest subset of a chunk that can be efficiently decoded if the chunk were subdivided into a regular grid.
    /// For most codecs, this is just the shape of the chunk.
    /// It is the shape of the "inner chunks" for the sharding codec.
    fn partial_decode_granularity(&self, shape: &[NonZeroU64]) -> ChunkShape {
        shape.to_vec()
    }
}

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

#[cfg(feature = "async")]
/// Asynchronous partial bytes decoder traits.
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncBytesPartialDecoderTraits: Any + MaybeSend + MaybeSync {
    /// Returns whether the chunk exists.
    ///
    /// # Errors
    /// Returns [`StorageError`] if a storage operation fails.
    async fn exists(&self) -> Result<bool, StorageError>;

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
    ) -> Result<Option<ArrayBytesRaw<'a>>, CodecError> {
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
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError>;

    /// Decode all bytes.
    ///
    /// Returns [`None`] if partial decoding of the input handle returns [`None`].
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    async fn decode<'a>(
        &'a self,
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytesRaw<'a>>, CodecError> {
        self.partial_decode(ByteRange::FromStart(0, None), options)
            .await
    }

    /// Returns whether this decoder supports partial decoding.
    ///
    /// If this returns `true`, the decoder can efficiently handle partial decoding operations.
    /// If this returns `false`, partial decoding will fall back to a full decode operation.
    fn supports_partial_decode(&self) -> bool;
}

/// Partial array decoder traits.
pub trait ArrayPartialDecoderTraits: Any + MaybeSend + MaybeSync {
    /// Return the data type of the partial decoder.
    fn data_type(&self) -> &DataType;

    /// Returns whether the chunk exists.
    ///
    /// # Errors
    /// Returns [`StorageError`] if a storage operation fails.
    fn exists(&self) -> Result<bool, StorageError>;

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
        indexer: &dyn Indexer,
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
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if indexer.len() != output_target.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                indexer.len(),
                output_target.num_elements(),
            )
            .into());
        }

        let decoded_value = self.partial_decode(indexer, options)?;
        decode_into_array_bytes_target(&decoded_value, output_target)
    }

    /// Returns whether this decoder supports partial decoding.
    ///
    /// If this returns `true`, the decoder can efficiently handle partial decoding operations.
    /// If this returns `false`, partial decoding will fall back to a full decode operation.
    fn supports_partial_decode(&self) -> bool;
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
        indexer: &dyn Indexer,
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
        indexer: &dyn Indexer,
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
        bytes: ArrayBytesRaw<'_>,
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
        offset_values: OffsetBytesIterator<'a, ArrayBytesRaw<'_>>,
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

    /// Returns whether the chunk exists.
    ///
    /// # Errors
    /// Returns [`StorageError`] if a storage operation fails.
    async fn exists(&self) -> Result<bool, StorageError>;

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
        indexer: &dyn Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Async variant of [`ArrayPartialDecoderTraits::partial_decode_into`].
    #[allow(clippy::missing_safety_doc)]
    async fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        if indexer.len() != output_target.num_elements() {
            return Err(InvalidNumberOfElementsError::new(
                indexer.len(),
                output_target.num_elements(),
            )
            .into());
        }

        let decoded_value = self.partial_decode(indexer, options).await?;
        decode_into_array_bytes_target(&decoded_value, output_target)
    }

    /// Returns whether this decoder supports partial decoding.
    ///
    /// If this returns `true`, the decoder can efficiently handle partial decoding operations.
    /// If this returns `false`, partial decoding will fall back to a full decode operation.
    fn supports_partial_decode(&self) -> bool;
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
    fn exists(&self) -> Result<bool, StorageError> {
        Ok(self.storage.size_key(&self.key)?.is_some())
    }

    fn size_held(&self) -> usize {
        0
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError> {
        let bytes = self.storage.get_partial_many(&self.key, decoded_regions)?;
        if let Some(bytes) = bytes {
            let bytes = bytes
                .map(|b| Ok::<_, StorageError>(Cow::Owned(b?.into())))
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
    async fn exists(&self) -> Result<bool, StorageError> {
        Ok(self.storage.size_key(&self.key).await?.is_some())
    }

    fn size_held(&self) -> usize {
        0
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
        let bytes = self
            .storage
            .get_partial_many(&self.key, decoded_regions)
            .await?;
        Ok(if let Some(bytes) = bytes {
            use futures::{StreamExt, TryStreamExt};
            Some(
                bytes
                    .map(|bytes| Ok::<_, StorageError>(Cow::Owned(bytes?.into())))
                    .try_collect()
                    .await?,
            )
        } else {
            None
        })
    }

    fn supports_partial_decode(&self) -> bool {
        self.storage.supports_get_partial()
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

/// A store value partial encoder generic over storage type.
pub struct StoragePartialEncoder<TStorage> {
    storage: TStorage,
    key: StoreKey,
}

impl<TStorage> StoragePartialEncoder<TStorage> {
    /// Create a new storage partial encoder.
    pub fn new(storage: TStorage, key: StoreKey) -> Self {
        Self { storage, key }
    }
}

impl BytesPartialDecoderTraits for StoragePartialEncoder<ReadableWritableStorage> {
    fn exists(&self) -> Result<bool, StorageError> {
        Ok(self.storage.size_key(&self.key)?.is_some())
    }

    fn size_held(&self) -> usize {
        0
    }

    fn partial_decode_many(
        &self,
        decoded_regions: ByteRangeIterator,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'_>>>, CodecError> {
        let results = self.storage.get_partial_many(&self.key, decoded_regions)?;
        if let Some(results) = results {
            Ok(Some(
                results
                    .into_iter()
                    .map(|bytes| Ok::<_, StorageError>(Cow::Owned(bytes?.into())))
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

impl BytesPartialEncoderTraits for StoragePartialEncoder<ReadableWritableStorage> {
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn BytesPartialDecoderTraits> {
        self.clone()
    }

    fn erase(&self) -> Result<(), CodecError> {
        Ok(self.storage.erase(&self.key)?)
    }

    fn partial_encode_many(
        &self,
        offset_values: OffsetBytesIterator<ArrayBytesRaw<'_>>,
        _options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let offset_values = offset_values
            .into_iter()
            .map(|(offset, bytes)| (offset, bytes.into_owned().into()));
        Ok(self
            .storage
            .set_partial_many(&self.key, Box::new(offset_values))?)
    }

    fn supports_partial_encode(&self) -> bool {
        self.storage.supports_set_partial()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for StoragePartialEncoder<AsyncReadableWritableStorage> {
    async fn exists(&self) -> Result<bool, StorageError> {
        Ok(self.storage.size_key(&self.key).await?.is_some())
    }

    fn size_held(&self) -> usize {
        0
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        _options: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
        let results = self
            .storage
            .get_partial_many(&self.key, decoded_regions)
            .await?;
        if let Some(results) = results {
            use futures::{StreamExt, TryStreamExt};
            Ok(Some(
                results
                    .map(|bytes| Ok::<_, StorageError>(Cow::Owned(bytes?.into())))
                    .try_collect()
                    .await?,
            ))
        } else {
            Ok(None)
        }
    }

    fn supports_partial_decode(&self) -> bool {
        self.storage.supports_get_partial()
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialEncoderTraits for StoragePartialEncoder<AsyncReadableWritableStorage> {
    fn into_dyn_decoder(self: Arc<Self>) -> Arc<dyn AsyncBytesPartialDecoderTraits> {
        self.clone()
    }

    async fn erase(&self) -> Result<(), CodecError> {
        Ok(self.storage.erase(&self.key).await?)
    }

    async fn partial_encode_many<'a>(
        &'a self,
        offset_values: OffsetBytesIterator<'a, ArrayBytesRaw<'_>>,
        _options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let offset_values = offset_values
            .into_iter()
            .map(|(offset, bytes)| (offset, bytes.into_owned().into()));
        Ok(self
            .storage
            .set_partial_many(&self.key, Box::new(offset_values))
            .await?)
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
        let element_shape = ChunkShape::from(vec![unsafe { NonZeroU64::new_unchecked(1) }]);

        // Calculate the changed fill value
        let fill_value = self
            .encode(
                ArrayBytes::new_fill_value(decoded_data_type, 1, decoded_fill_value)?,
                &element_shape,
                decoded_data_type,
                decoded_fill_value,
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
        Ok(decoded_shape.to_vec())
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
        Ok(Some(encoded_shape.to_vec()))
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
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
    ) -> Result<(ChunkShape, DataType, FillValue), CodecError> {
        Ok((
            self.encoded_shape(shape)?,
            self.encoded_data_type(data_type)?,
            self.encoded_fill_value(data_type, fill_value)?,
        ))
    }

    /// Encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or `bytes` is incompatible with the decoded representation.
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Decode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the decoded output is incompatible with the decoded representation.
    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
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
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_handle,
            shape.to_vec(),
            data_type.clone(),
            fill_value.clone(),
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
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_output_handle,
            shape.to_vec(),
            data_type.clone(),
            fill_value.clone(),
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
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_handle,
            shape.to_vec(),
            data_type.clone(),
            fill_value.clone(),
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
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToArrayCodecPartialDefault::new(
            input_output_handle,
            shape.to_vec(),
            data_type.clone(),
            fill_value.clone(),
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
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
    ) -> Result<BytesRepresentation, CodecError>;

    /// Encode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or `bytes` is incompatible with the decoded representation.
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError>;

    /// Decode a chunk.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the decoded output is incompatible with the decoded representation.
    fn decode<'a>(
        &self,
        bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError>;

    /// Compact a chunk.
    ///
    /// Takes an encoded representation and compacts it to remove any extraneous data.
    /// The default implementation returns the input `bytes` unchanged.
    ///
    /// Returns `Ok(None)` if no compaction was performed.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or `bytes` is incompatible with the decoded representation.
    #[expect(unused_variables)]
    fn compact<'a>(
        &self,
        bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytesRaw<'a>>, CodecError> {
        Ok(None)
    }

    /// Decode into a subset of a preallocated output.
    ///
    /// This method is intended for internal use by Array.
    /// It works for fixed length data types and optional data types.
    ///
    /// The decoded representation shape and dimensionality does not need to match the output target, but the number of elements must match.
    /// Chunk elements are written to the subset of the output in C order.
    ///
    /// For optional data types, provide an `ArrayBytesDecodeIntoTarget` with a `mask` set to `Some`.
    /// For non-optional data types, convert a fixed view to target using `.into()` or create with `mask: None`.
    ///
    /// # Errors
    /// Returns [`CodecError`] if a codec fails or the number of elements in the decoded representation does not match the number of elements in the output target.
    fn decode_into(
        &self,
        bytes: ArrayBytesRaw<'_>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let num_elements = output_target.num_elements();
        let shape_num_elements: u64 = shape.iter().map(|d| d.get()).product();
        if shape_num_elements != num_elements {
            return Err(InvalidNumberOfElementsError::new(num_elements, shape_num_elements).into());
        }

        let decoded_value = self.decode(bytes, shape, data_type, fill_value, options)?;
        decode_into_array_bytes_target(&decoded_value, output_target)
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
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
            input_handle,
            shape.to_vec(),
            data_type.clone(),
            fill_value.clone(),
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
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
            input_output_handle,
            shape.to_vec(),
            data_type.clone(),
            fill_value.clone(),
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
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
            input_handle,
            shape.to_vec(),
            data_type.clone(),
            fill_value.clone(),
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
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        Ok(Arc::new(ArrayToBytesCodecPartialDefault::new(
            input_output_handle,
            shape.to_vec(),
            data_type.clone(),
            fill_value.clone(),
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
        decoded_value: ArrayBytesRaw<'a>,
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError>;

    /// Decode chunk bytes.
    //
    /// # Errors
    /// Returns [`CodecError`] if a codec fails.
    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        decoded_representation: &BytesRepresentation,
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError>;

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
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new_bytes(
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
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new_bytes(
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
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new_bytes(
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
        Ok(Arc::new(BytesToBytesCodecPartialDefault::new_bytes(
            input_output_handle,
            *decoded_representation,
            self.into_dyn(),
        )))
    }
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

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl AsyncBytesPartialDecoderTraits for Cow<'static, [u8]> {
    async fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.as_ref().len()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        _parallel: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
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
impl AsyncBytesPartialDecoderTraits for Vec<u8> {
    async fn exists(&self) -> Result<bool, StorageError> {
        Ok(true)
    }

    fn size_held(&self) -> usize {
        self.len()
    }

    async fn partial_decode_many<'a>(
        &'a self,
        decoded_regions: ByteRangeIterator<'a>,
        _parallel: &CodecOptions,
    ) -> Result<Option<Vec<ArrayBytesRaw<'a>>>, CodecError> {
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
    IncompatibleIndexer(#[from] IndexerError),
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
    #[error("{}", format_unsupported_data_type(.0, .1))]
    UnsupportedDataType(DataType, String),
    /// Data type does not support a codec.
    #[error(transparent)]
    UnsupportedDataTypeCodec(#[from] zarrs_data_type::DataTypeCodecError),
    /// Offsets are not [`None`] with a fixed length data type.
    #[error(
        "Offsets are invalid or are not compatible with the data type (e.g. fixed-sized data types)"
    )]
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
    /// Expected non-optional bytes.
    #[error("Expected non-optional array bytes")]
    ExpectedNonOptionalBytes,
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
    RawBytesOffsetsCreate(#[from] ArrayBytesRawOffsetsCreateError),
    /// Variable length array bytes offsets are out of bounds.
    #[error(transparent)]
    RawBytesOffsetsOutOfBounds(#[from] ArrayBytesRawOffsetsOutOfBoundsError),
    /// An incompatible fill value error
    #[error(transparent)]
    DataTypeFillValueError(#[from] DataTypeFillValueError),
    /// An array region error.
    #[error(transparent)]
    ArraySubsetError(#[from] ArraySubsetError),
}

fn format_unsupported_data_type(data_type: &DataType, codec_name: &String) -> String {
    let data_type_name = data_type
        .name(zarrs_plugin::ZarrVersion::V3)
        .unwrap_or_default();
    if data_type.is_optional() {
        format!(
            "Unsupported data type {data_type_name} for codec {codec_name}. Use the optional codec to handle optional data types.",
        )
    } else {
        format!("Unsupported data type {data_type_name} for codec {codec_name}",)
    }
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

impl From<zarrs_data_type::codec_traits::BytesCodecEndiannessMissingError> for CodecError {
    fn from(err: zarrs_data_type::codec_traits::BytesCodecEndiannessMissingError) -> Self {
        Self::Other(err.to_string())
    }
}
