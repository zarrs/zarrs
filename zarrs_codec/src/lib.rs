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
mod resources;
mod small_copy;
pub use codec_specific_options::CodecSpecificOptions;
pub use resources::{ComputeResources, Resources};

mod codec_partial_default;
use codec_partial_default::BytesToBytesCodecPartialDefault;
pub use codec_partial_default::CodecPartialDefault;

mod array_bytes_fixed_disjoint_view;
pub use array_bytes_fixed_disjoint_view::{
    ArrayBytesFixedDisjointView, ArrayBytesFixedDisjointViewCreateError,
};

mod codec_traits;
pub use codec_traits::array::ArrayCodecTraits;
pub use codec_traits::array_partial_sync::{
    ArrayPartialDecoderNoSubchunkingTraits, ArrayPartialDecoderSubchunkingTraits,
    ArrayPartialDecoderTraits, ArrayPartialEncoderTraits,
};
pub use codec_traits::array_to_array::{
    ArrayToArrayCodecGeometryIdentityTraits, ArrayToArrayCodecGeometryTraits,
    ArrayToArrayCodecTraits, UnboundArrayToArrayCodecTraits, validate_chunk_element_indices,
    validate_chunk_subset,
};
pub use codec_traits::array_to_bytes::{
    ArrayToBytesCodecNoSubchunkingTraits, ArrayToBytesCodecSubchunkingTraits,
    ArrayToBytesCodecTraits, UnboundArrayToBytesCodecTraits,
};
pub use codec_traits::bytes_partial_sync::{BytesPartialDecoderTraits, BytesPartialEncoderTraits};
pub use codec_traits::bytes_to_bytes::{
    BytesToBytesCodecDecodeInto, BytesToBytesCodecTraits, copy_decoded_bytes_into,
    read_decoded_bytes_into,
};
pub use codec_traits::{CodecTraits, CodecTraitsV2, CodecTraitsV3};

#[cfg(feature = "async")]
pub use codec_traits::array_partial_async::{
    AsyncArrayPartialDecoderSubchunkingTraits, AsyncArrayPartialDecoderTraits,
    AsyncArrayPartialEncoderTraits,
};
#[cfg(feature = "async")]
pub use codec_traits::bytes_partial_async::{
    AsyncBytesPartialDecoderTraits, AsyncBytesPartialEncoderTraits,
};

mod recommended_concurrency;
pub use recommended_concurrency::RecommendedConcurrency;

mod bytes_representation;
pub use bytes_representation::BytesRepresentation;

mod array_bytes_decode_into_input;
pub use array_bytes_decode_into_input::{ArrayBytesDecodeIntoInput, BytesDecodeSource};

mod array_bytes;
pub use array_bytes::{
    ArrayBytes, ArrayBytesError, ArrayBytesOffsets, ArrayBytesOffsetsCreateError,
    ArrayBytesOffsetsElement, ArrayBytesOffsetsIter, ArrayBytesOffsetsOutOfBoundsError,
    ArrayBytesOffsetsRangesIter, ArrayBytesOffsetsSlice, ArrayBytesOptional,
    ArrayBytesVariableLength, CowBytes, ExpectedFixedLengthBytesError, ExpectedOptionalBytesError,
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
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::InvalidByteRangeError;

use std::sync::{Arc, LazyLock};

use zarrs_chunk_grid::{
    ArraySubset, ArraySubsetError, ChunkGrid, IncompatibleDimensionalityError, Indexer,
    IndexerError,
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
    Fixed(ArrayBytesFixedDisjointView<'a>),

    /// Target for optional data (nested data target + mask view).
    ///
    /// Created with [`with_optional_mask`](ArrayBytesDecodeIntoTarget::with_optional_mask), which validates the mask against the data.
    Optional(ArrayBytesDecodeIntoTargetOptional<'a>),
}

/// An optional decode target composed of a data target and a validity mask view.
///
/// The fields are private so that the mask always covers the same elements as the data.
pub struct ArrayBytesDecodeIntoTargetOptional<'a> {
    data: Box<ArrayBytesDecodeIntoTarget<'a>>,
    mask: ArrayBytesFixedDisjointView<'a>,
}

impl<'a> ArrayBytesDecodeIntoTargetOptional<'a> {
    /// Return the target for the inner data.
    #[must_use]
    pub fn data(&self) -> &ArrayBytesDecodeIntoTarget<'a> {
        &self.data
    }

    /// Return the validity mask view, with one byte per element.
    #[must_use]
    pub fn mask(&self) -> &ArrayBytesFixedDisjointView<'a> {
        &self.mask
    }

    /// Consume self and return the data target and mask view.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        Box<ArrayBytesDecodeIntoTarget<'a>>,
        ArrayBytesFixedDisjointView<'a>,
    ) {
        (self.data, self.mask)
    }
}

impl<'a> ArrayBytesDecodeIntoTarget<'a> {
    /// Wrap the target in an [`Optional`](Self::Optional) layer with `mask` as its validity mask.
    ///
    /// The target becomes the inner data and `mask` the outermost mask, like [`ArrayBytes::with_optional_mask`].
    ///
    /// # Errors
    /// Returns a [`WithOptionalMaskError`] holding the target and `mask` unless `mask` has one byte per element and views an array with the same shape and [`subset`](Self::subset) as the target.
    /// Map it to an [`IncompatibleOptionalMaskError`] with `From` to discard the inputs, e.g. to propagate it as a `Box<dyn Error>`.
    pub fn with_optional_mask(
        self,
        mask: ArrayBytesFixedDisjointView<'a>,
    ) -> Result<Self, WithOptionalMaskError<'a>> {
        if let Err(error) = self.validate_optional_mask(&mask) {
            return Err(WithOptionalMaskError::new(error, self, mask));
        }
        Ok(self.wrap_optional_mask(mask))
    }

    /// Wrap the target in one [`Optional`](Self::Optional) layer per mask, with `masks` ordered from outermost to innermost.
    ///
    /// # Errors
    /// Returns a [`WithOptionalMaskError`] holding the target and `masks` if any mask is incompatible, see [`with_optional_mask`](Self::with_optional_mask).
    pub fn with_optional_masks(
        self,
        masks: impl IntoIterator<Item = ArrayBytesFixedDisjointView<'a>>,
    ) -> Result<Self, WithOptionalMaskError<'a, Vec<ArrayBytesFixedDisjointView<'a>>>> {
        let masks: Vec<_> = masks.into_iter().collect();
        // Wrapping does not change the innermost data, so every mask can be validated up front
        if let Err(error) = masks
            .iter()
            .try_for_each(|mask| self.validate_optional_mask(mask))
        {
            return Err(WithOptionalMaskError::new(error, self, masks));
        }
        Ok(masks.into_iter().rev().fold(self, Self::wrap_optional_mask))
    }

    /// Check that `mask` is a valid validity mask for the innermost data of the target.
    fn validate_optional_mask(
        &self,
        mask: &ArrayBytesFixedDisjointView<'a>,
    ) -> Result<(), IncompatibleOptionalMaskError> {
        let data = self.data_view();
        if mask.data_type_size() != 1 {
            return Err(IncompatibleOptionalMaskError::ElementSize(
                mask.data_type_size(),
            ));
        }
        if mask.shape() != data.shape() {
            return Err(IncompatibleOptionalMaskError::ArrayShape {
                mask: mask.shape().to_vec(),
                data: data.shape().to_vec(),
            });
        }
        if mask.subset() != data.subset() {
            return Err(IncompatibleOptionalMaskError::Subset {
                mask: Box::new(mask.subset().clone()),
                data: Box::new(data.subset().clone()),
            });
        }
        Ok(())
    }

    /// Wrap the target in an [`Optional`](Self::Optional) layer with a validated `mask`.
    fn wrap_optional_mask(self, mask: ArrayBytesFixedDisjointView<'a>) -> Self {
        Self::Optional(ArrayBytesDecodeIntoTargetOptional {
            data: Box::new(self),
            mask,
        })
    }

    /// Return the innermost data view.
    fn data_view(&self) -> &ArrayBytesFixedDisjointView<'a> {
        match self {
            Self::Fixed(data) => data,
            Self::Optional(optional) => optional.data.data_view(),
        }
    }

    /// Return the subset of the array covered by the target.
    #[must_use]
    pub fn subset(&self) -> &ArraySubset {
        self.data_view().subset()
    }

    /// Create a target with the same structure as this target that covers `subset` of the same array.
    ///
    /// An empty `subset` references no elements, so it is accepted irrespective of its start.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if `subset` has an incompatible dimensionality with, or is out-of-bounds of, the [`subset`](Self::subset) of the target.
    ///
    /// # Safety
    /// The `subset` must not overlap with the `subset` of any other created targets or views that reference the same array bytes.
    pub unsafe fn subdivide(&self, subset: ArraySubset) -> Result<Self, CodecError> {
        match self {
            Self::Fixed(data) => Ok(Self::Fixed(unsafe { data.subdivide(subset) }?)),
            Self::Optional(optional) => {
                let mask = unsafe { optional.mask.subdivide(subset.clone()) }?;
                Ok(Self::Optional(ArrayBytesDecodeIntoTargetOptional {
                    data: Box::new(unsafe { optional.data.subdivide(subset) }?),
                    mask,
                }))
            }
        }
    }

    /// Return the number of elements in the target.
    #[must_use]
    pub fn num_elements(&self) -> u64 {
        self.data_view().num_elements()
    }
}

/// A validity mask view is incompatible with the data of an [`ArrayBytesDecodeIntoTarget`].
#[derive(Clone, Debug, Display, Error)]
pub enum IncompatibleOptionalMaskError {
    /// The mask does not have one byte per element.
    #[display("the optional mask has {_0} bytes per element, expected 1")]
    ElementSize(usize),
    /// The mask and data view arrays with different shapes.
    #[display(
        "the optional mask array shape {mask:?} does not match the data array shape {data:?}"
    )]
    ArrayShape {
        /// The shape of the mask array.
        mask: ArrayShape,
        /// The shape of the data array.
        data: ArrayShape,
    },
    /// The mask and data cover different subsets.
    #[display("the optional mask subset {mask} does not match the data subset {data}")]
    Subset {
        /// The subset of the mask.
        mask: Box<ArraySubset>,
        /// The subset of the data.
        data: Box<ArraySubset>,
    },
}

/// The error returned by [`ArrayBytesDecodeIntoTarget::with_optional_mask`] and [`with_optional_masks`](ArrayBytesDecodeIntoTarget::with_optional_masks), holding the target and mask(s) that were passed in.
pub struct WithOptionalMaskError<'a, M = ArrayBytesFixedDisjointView<'a>>(
    Box<WithOptionalMaskErrorInner<'a, M>>,
);

/// Boxed to keep the error small.
struct WithOptionalMaskErrorInner<'a, M> {
    error: IncompatibleOptionalMaskError,
    target: ArrayBytesDecodeIntoTarget<'a>,
    masks: M,
}

impl<'a, M> WithOptionalMaskError<'a, M> {
    fn new(
        error: IncompatibleOptionalMaskError,
        target: ArrayBytesDecodeIntoTarget<'a>,
        masks: M,
    ) -> Self {
        Self(Box::new(WithOptionalMaskErrorInner {
            error,
            target,
            masks,
        }))
    }
}

impl<'a, M> WithOptionalMaskError<'a, M> {
    /// Return the reason the mask is incompatible.
    #[must_use]
    pub fn error(&self) -> &IncompatibleOptionalMaskError {
        &self.0.error
    }

    /// Consume self and return the target and mask(s) that were passed in.
    #[must_use]
    pub fn into_parts(self) -> (ArrayBytesDecodeIntoTarget<'a>, M) {
        (self.0.target, self.0.masks)
    }
}

impl<M> std::fmt::Debug for WithOptionalMaskError<'_, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WithOptionalMaskError")
            .field("error", &self.0.error)
            .finish_non_exhaustive()
    }
}

impl<M> std::fmt::Display for WithOptionalMaskError<'_, M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0.error, f)
    }
}

impl<M> std::error::Error for WithOptionalMaskError<'_, M> {}

impl<M> From<WithOptionalMaskError<'_, M>> for IncompatibleOptionalMaskError {
    fn from(error: WithOptionalMaskError<'_, M>) -> Self {
        error.0.error
    }
}

impl<M> From<WithOptionalMaskError<'_, M>> for CodecError {
    fn from(error: WithOptionalMaskError<'_, M>) -> Self {
        Self::IncompatibleOptionalMask(error.0.error)
    }
}

impl<'a> From<ArrayBytesFixedDisjointView<'a>> for ArrayBytesDecodeIntoTarget<'a> {
    fn from(view: ArrayBytesFixedDisjointView<'a>) -> Self {
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

mod chunk_grid_mapped {
    use super::ChunkGrid;

    /// Describes a chunk grid mapped to a representation in a codec chain.
    #[derive(Debug, Clone)]
    pub enum ChunkGridMapped {
        /// No chunk grid is available.
        None,
        /// A chunk grid is resolvable for the whole array.
        Array(ChunkGrid),
        /// No global grid exists, but a grid is resolvable for each chunk via
        /// [`ArrayPartialDecoderSubchunkingTraits::local_subchunk_grid`](crate::ArrayPartialDecoderSubchunkingTraits::local_subchunk_grid).
        ChunkLocal,
    }

    impl ChunkGridMapped {
        /// Return the chunk grid if it is resolvable for the whole array.
        ///
        /// Returns [`None`] for [`ChunkGridMapped::None`] and [`ChunkGridMapped::ChunkLocal`],
        /// which are distinct: the latter indicates that a grid is resolvable per chunk.
        #[must_use]
        pub fn as_chunk_grid(&self) -> Option<&ChunkGrid> {
            match self {
                Self::Array(chunk_grid) => Some(chunk_grid),
                Self::None | Self::ChunkLocal => None,
            }
        }
    }

    /// A borrowed chunk grid mapped to a representation in a codec chain.
    #[derive(Debug, Clone, Copy)]
    pub enum ChunkGridMappedRef<'a> {
        /// No chunk grid is available.
        None,
        /// A chunk grid is resolvable for the whole array.
        Array(&'a ChunkGrid),
        /// No global grid exists, but a grid is resolvable for each chunk.
        ChunkLocal,
    }

    impl<'a> ChunkGridMappedRef<'a> {
        /// Return the chunk grid if it is resolvable for the whole array.
        ///
        /// Returns [`None`] for [`ChunkGridMappedRef::None`] and [`ChunkGridMappedRef::ChunkLocal`],
        /// which are distinct: the latter indicates that a grid is resolvable per chunk.
        #[must_use]
        pub fn as_chunk_grid(self) -> Option<&'a ChunkGrid> {
            match self {
                Self::Array(chunk_grid) => Some(chunk_grid),
                Self::None | Self::ChunkLocal => None,
            }
        }
    }

    impl From<Option<ChunkGrid>> for ChunkGridMapped {
        fn from(chunk_grid: Option<ChunkGrid>) -> Self {
            chunk_grid.map_or(Self::None, Self::Array)
        }
    }

    impl<'a> From<&'a ChunkGrid> for ChunkGridMappedRef<'a> {
        fn from(chunk_grid: &'a ChunkGrid) -> Self {
            Self::Array(chunk_grid)
        }
    }

    impl<'a> From<&'a ChunkGridMapped> for ChunkGridMappedRef<'a> {
        fn from(chunk_grid: &'a ChunkGridMapped) -> Self {
            match chunk_grid {
                ChunkGridMapped::None => Self::None,
                ChunkGridMapped::Array(chunk_grid) => Self::Array(chunk_grid),
                ChunkGridMapped::ChunkLocal => Self::ChunkLocal,
            }
        }
    }

    impl From<ChunkGridMappedRef<'_>> for ChunkGridMapped {
        fn from(chunk_grid: ChunkGridMappedRef<'_>) -> Self {
            match chunk_grid {
                ChunkGridMappedRef::None => Self::None,
                ChunkGridMappedRef::Array(chunk_grid) => Self::Array(chunk_grid.clone()),
                ChunkGridMappedRef::ChunkLocal => Self::ChunkLocal,
            }
        }
    }
}

/// A chunk grid mapped to the decoded representation of a codec.
pub use chunk_grid_mapped::ChunkGridMapped as ChunkGridDecoded;

/// A chunk grid mapped to the encoded representation of a codec.
pub use chunk_grid_mapped::ChunkGridMapped as ChunkGridEncoded;

/// A borrowed chunk grid mapped to the decoded representation of a codec.
pub use chunk_grid_mapped::ChunkGridMappedRef as ChunkGridDecodedRef;

/// A borrowed chunk grid mapped to the encoded representation of a codec.
pub use chunk_grid_mapped::ChunkGridMappedRef as ChunkGridEncodedRef;

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
    // Boxed, as array subsets are stored inline and would enlarge every result with a codec error
    subset: Box<ArraySubset>,
    must_be_within: Box<ArraySubset>,
}

impl SubsetOutOfBoundsError {
    /// Create a new [`SubsetOutOfBoundsError`].
    #[must_use]
    pub fn new(subset: ArraySubset, must_be_within: ArraySubset) -> Self {
        Self {
            subset: Box::new(subset),
            must_be_within: Box::new(must_be_within),
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
    #[allow(
        clippy::needless_pass_by_value,
        reason = "taking `impl ToString` by value keeps `.map_err(Self::other)` and string literals ergonomic"
    )]
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
    /// An optional mask is incompatible with the data of a decode target.
    #[error(transparent)]
    IncompatibleOptionalMask(#[from] IncompatibleOptionalMaskError),
    /// Invalid byte offsets for variable length data.
    #[error(transparent)]
    ArrayBytesOffsetsCreate(#[from] ArrayBytesOffsetsCreateError),
    /// Variable length array bytes offsets are out of bounds.
    #[error(transparent)]
    ArrayBytesOffsetsOutOfBounds(#[from] ArrayBytesOffsetsOutOfBoundsError),
    /// An incompatible fill value error
    #[error(transparent)]
    DataTypeFillValueError(#[from] DataTypeFillValueError),
    /// An array region error.
    #[error(transparent)]
    ArraySubsetError(#[from] ArraySubsetError),
    /// Codec create error.
    #[error(transparent)]
    CodecCreateError(#[from] CodecCreateError),
    /// Missing input cannot be decoded, or partially updated, because the fill value is undefined.
    ///
    /// See [`UnboundArrayToBytesCodecTraits::with_context`].
    #[error(
        "missing input cannot be decoded or partially updated because the fill value is undefined"
    )]
    UndefinedFillValue,
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

/// Copy `src` into `dst`, which may be uninitialised, and return the initialised `dst`.
///
/// Polyfill for `<[MaybeUninit<u8>]>::write_copy_of_slice`; remove at MSRV 1.93.
#[doc(hidden)]
#[inline]
pub fn write_copy_of_slice<'a>(
    dst: &'a mut [std::mem::MaybeUninit<u8>],
    src: &[u8],
) -> &'a mut [u8] {
    // SAFETY: `MaybeUninit<u8>` has the same layout as `u8`, and `src` is only read.
    let src = unsafe { &*(std::ptr::from_ref::<[u8]>(src) as *const [std::mem::MaybeUninit<u8>]) };
    dst.copy_from_slice(src);
    // SAFETY: `dst` was initialised by the copy.
    unsafe { slice_assume_init_mut(dst) }
}

/// Reinterpret initialised bytes as `u8`.
///
/// Polyfill for `<[MaybeUninit<u8>]>::assume_init_mut`; remove at MSRV 1.93.
///
/// # Safety
/// Every byte of `bytes` must be initialised.
#[doc(hidden)]
#[inline]
pub unsafe fn slice_assume_init_mut(bytes: &mut [std::mem::MaybeUninit<u8>]) -> &mut [u8] {
    // SAFETY: `MaybeUninit<u8>` has the same layout as `u8`, and the caller guarantees the bytes are initialised.
    unsafe { &mut *(std::ptr::from_mut(bytes) as *mut [u8]) }
}
