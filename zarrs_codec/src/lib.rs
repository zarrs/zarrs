//! The codec API for the [`zarrs`](https://docs.rs/zarrs/latest/zarrs/index.html) crate.
//!
//! ## Licence
//! `zarrs_codec` is licensed under either of
//!  - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_codec/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//!  - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_codec/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

mod codec_specific_options;
mod options;
pub use codec_specific_options::CodecSpecificOptions;

mod codec_partial_default;
use codec_partial_default::BytesToBytesCodecPartialDefault;
pub use codec_partial_default::CodecPartialDefault;

mod array_bytes_fixed_disjoint_view;
pub use array_bytes_fixed_disjoint_view::{
    ArrayBytesFixedDisjointView, ArrayBytesFixedDisjointViewCreateError,
};

mod codec_traits;
pub use codec_traits::array::ArrayCodecTraits;
pub use codec_traits::array_partial_sync::{ArrayPartialDecoderTraits, ArrayPartialEncoderTraits};
pub use codec_traits::array_to_array::{ArrayToArrayCodecTraits, UnboundArrayToArrayCodecTraits};
pub use codec_traits::array_to_bytes::{ArrayToBytesCodecTraits, UnboundArrayToBytesCodecTraits};
pub use codec_traits::bytes_partial_sync::{BytesPartialDecoderTraits, BytesPartialEncoderTraits};
pub use codec_traits::bytes_to_bytes::BytesToBytesCodecTraits;
pub use codec_traits::{CodecTraits, CodecTraitsV2, CodecTraitsV3};

#[cfg(feature = "async")]
pub use codec_traits::array_partial_async::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits,
};
#[cfg(feature = "async")]
pub use codec_traits::bytes_partial_async::{
    AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits,
};

mod recommended_concurrency;
pub use recommended_concurrency::RecommendedConcurrency;

mod bytes_representation;
pub use bytes_representation::BytesRepresentation;

mod array_bytes;
pub use array_bytes::{
    ArrayBytes, ArrayBytesError, ArrayBytesOffsets, ArrayBytesOptional, ArrayBytesRaw,
    ArrayBytesRawOffsetsCreateError, ArrayBytesRawOffsetsOutOfBoundsError,
    ArrayBytesVariableLength, ExpectedFixedLengthBytesError, ExpectedOptionalBytesError,
    ExpectedVariableLengthBytesError, copy_fill_value_into, decode_into_array_bytes_target,
    update_array_bytes,
};

mod byte_interval_partial_decoder;
#[cfg(feature = "async")]
pub use byte_interval_partial_decoder::AsyncByteIntervalPartialDecoder;
pub use byte_interval_partial_decoder::ByteIntervalPartialDecoder;

use derive_more::derive::Display;
pub use options::{CodecMetadataOptions, CodecOptions};
use thiserror::Error;
use zarrs_metadata::{ArrayShape, ChunkShape, Configuration};
use zarrs_storage::byte_range::{ByteRangeIterator, InvalidByteRangeError, extract_byte_ranges};
#[cfg(feature = "async")]
use zarrs_storage::{AsyncReadableStorage, AsyncReadableWritableStorage};
use zarrs_storage::{
    OffsetBytesIterator, ReadableStorage, ReadableWritableStorage, StorageError, StoreKey,
};

use std::borrow::Cow;
use std::sync::{Arc, LazyLock, Mutex};

use zarrs_chunk_grid::{
    ArraySubset, ArraySubsetError, IncompatibleDimensionalityError, Indexer, IndexerError,
};
use zarrs_data_type::{DataType, DataTypeFillValueError, FillValue};
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::{
    ExtensionAliases, ExtensionName, Plugin, PluginCreateError, PluginUnsupportedError,
    RuntimePlugin, RuntimeRegistry, ZarrVersion, ZarrVersion2, ZarrVersion3,
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
pub struct CodecPluginV3(Plugin<Codec, MetadataV3, CodecCreateError>);
inventory::collect!(CodecPluginV3);

impl CodecPluginV3 {
    /// Create a new [`CodecPluginV3`] for a type implementing [`ExtensionAliases<ZarrVersion3>`].
    pub const fn new<T: ExtensionAliases<ZarrVersion3> + CodecTraitsV3>() -> Self {
        Self(Plugin::new(T::matches_name, T::create))
    }
}

/// A Zarr V2 codec plugin.
#[derive(derive_more::Deref)]
pub struct CodecPluginV2(Plugin<Codec, MetadataV2, CodecCreateError>);
inventory::collect!(CodecPluginV2);

impl CodecPluginV2 {
    /// Create a new [`CodecPluginV2`] for a type implementing [`ExtensionAliases<ZarrVersion2>`].
    pub const fn new<T: ExtensionAliases<ZarrVersion2> + CodecTraitsV2>() -> Self {
        Self(Plugin::new(T::matches_name, T::create))
    }
}

/// A runtime V3 codec plugin for dynamic registration.
pub type CodecRuntimePluginV3 = RuntimePlugin<Codec, MetadataV3, CodecCreateError>;

/// A runtime V2 codec plugin for dynamic registration.
pub type CodecRuntimePluginV2 = RuntimePlugin<Codec, MetadataV2, CodecCreateError>;

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
    ArrayToArray(Arc<dyn UnboundArrayToArrayCodecTraits>),
    /// An array to bytes codec.
    ArrayToBytes(Arc<dyn UnboundArrayToBytesCodecTraits>),
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
    ) -> Result<Self, CodecCreateError> {
        match metadata.into() {
            CodecMetadata::V2(metadata) => Self::from_metadata_v2(metadata),
            CodecMetadata::V3(metadata) => Self::from_metadata_v3(metadata),
        }
    }

    /// Create a codec from V3 metadata.
    ///
    /// # Errors
    /// Returns [`CodecCreateError`] if the metadata is invalid or not associated with a registered codec plugin.
    fn from_metadata_v3(metadata: &MetadataV3) -> Result<Self, CodecCreateError> {
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
        Err(PluginCreateError::Unsupported(PluginUnsupportedError::new(
            metadata.name().to_string(),
            "codec".to_string(),
        ))
        .into())
    }

    /// Create a codec from V2 metadata.
    ///
    /// # Errors
    /// Returns [`CodecCreateError`] if the metadata is invalid or not associated with a registered codec plugin.
    fn from_metadata_v2(metadata: &MetadataV2) -> Result<Self, CodecCreateError> {
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
        Err(PluginCreateError::Unsupported(PluginUnsupportedError::new(
            metadata.id().to_string(),
            "codec".to_string(),
        ))
        .into())
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

/// A codec creation error.
///
/// This is used for failures while creating codecs from metadata, reconfiguring codecs, or binding
/// unbound array codecs to a decoded data type and fill value.
#[derive(Clone, Debug, Error)]
pub enum CodecCreateError {
    /// A plugin creation error.
    #[error(transparent)]
    PluginCreateError(#[from] PluginCreateError),
    /// Unsupported data type.
    #[error("{}", format_unsupported_data_type(.0, .1))]
    UnsupportedDataType(DataType, String),
    /// An incompatible fill value error.
    #[error(transparent)]
    DataTypeFillValueError(#[from] DataTypeFillValueError),
    /// Other.
    #[error("{_0}")]
    Other(String),
}

impl From<zarrs_data_type::DataTypeCodecError> for CodecCreateError {
    fn from(error: zarrs_data_type::DataTypeCodecError) -> Self {
        let zarrs_data_type::DataTypeCodecError::UnsupportedDataType {
            data_type,
            codec_name,
        } = error;
        Self::UnsupportedDataType(data_type, codec_name.to_string())
    }
}

impl CodecCreateError {
    /// Create a new [`CodecCreateError::Other`] from a displayable error or message.
    #[must_use]
    pub fn other(error: impl ToString) -> Self {
        Self::Other(error.to_string())
    }
}

impl From<&str> for CodecCreateError {
    fn from(error: &str) -> Self {
        Self::Other(error.to_string())
    }
}

impl From<String> for CodecCreateError {
    fn from(error: String) -> Self {
        Self::Other(error)
    }
}

impl From<Arc<serde_json::Error>> for CodecCreateError {
    fn from(error: Arc<serde_json::Error>) -> Self {
        Self::PluginCreateError(PluginCreateError::from(error))
    }
}

/// A codec runtime error.
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
    /// Unsupported data type.
    #[error("{}", format_unsupported_data_type(.0, .1))]
    UnsupportedDataType(DataType, String),
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
    #[error(transparent)]
    ExpectedFixedLengthBytes(#[from] ExpectedFixedLengthBytesError),
    /// Expected variable length bytes.
    #[error(transparent)]
    ExpectedVariableLengthBytes(#[from] ExpectedVariableLengthBytesError),
    /// Expected optional bytes.
    #[error(transparent)]
    ExpectedOptionalBytes(#[from] ExpectedOptionalBytesError),
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
    /// Codec create error.
    #[error(transparent)]
    CodecCreateError(#[from] CodecCreateError),
}

impl From<zarrs_data_type::DataTypeCodecError> for CodecError {
    fn from(error: zarrs_data_type::DataTypeCodecError) -> Self {
        let zarrs_data_type::DataTypeCodecError::UnsupportedDataType {
            data_type,
            codec_name,
        } = error;
        Self::UnsupportedDataType(data_type, codec_name.to_string())
    }
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
        format!("Unsupported data type {data_type_name} for codec {codec_name}")
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

impl From<zarrs_data_type::codec_traits::bytes::BytesCodecEndiannessMissingError> for CodecError {
    fn from(err: zarrs_data_type::codec_traits::bytes::BytesCodecEndiannessMissingError) -> Self {
        Self::Other(err.to_string())
    }
}
