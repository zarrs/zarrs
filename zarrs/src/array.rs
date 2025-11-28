//! Zarr arrays.
//!
//! An array is a node in a Zarr hierarchy used to hold multidimensional array data and associated metadata.
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array>.
//!
//! A Zarr V3 array is defined by the following parameters (which are encoded in its JSON metadata):
//!  - **shape**: defines the length of the array dimensions,
//!  - **data type**: defines the numerical representation array elements,
//!  - **chunk grid**: defines how the array is subdivided into chunks,
//!  - **chunk key encoding**: defines how chunk grid cell coordinates are mapped to keys in a store,
//!  - **fill value**: an element value to use for uninitialised portions of the array,
//!  - **codecs**: used to encode and decode chunks.
//!  - (optional) **attributes**: user-defined attributes,
//!  - (optional) **storage transformers**: used to intercept and alter the storage keys and bytes of an array before they reach the underlying physical storage, and
//!  - (optional) **dimension names**: defines the names of the array dimensions.
//!
//! See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#array-metadata> for more information on array metadata.
//!
//! `zarrs` supports a subset of Zarr V2 arrays which are a compatible subset of Zarr V3 arrays.
//! This encompasses Zarr V2 array that use supported codecs and **could** be converted to a Zarr V3 array with only a metadata change.
//!
//! The documentation for [`Array`] details how to interact with arrays.

mod array_builder;
mod array_bytes;
mod array_bytes_fixed_disjoint_view;
mod array_errors;
mod array_metadata_options;
mod array_representation;
mod bytes_representation;
mod chunk_cache;
mod element;

pub mod chunk_grid;
pub mod chunk_key_encoding;
pub mod codec;
pub mod concurrency;
pub mod data_type;
pub mod storage_transformer;

#[cfg(feature = "dlpack")]
mod array_dlpack_ext;
#[cfg(feature = "sharding")]
mod array_sharded_ext;
#[cfg(feature = "sharding")]
mod array_sync_sharded_readable_ext;

use std::sync::Arc;

pub use self::{
    array_builder::{
        ArrayBuilder, ArrayBuilderChunkGrid, ArrayBuilderChunkGridMetadata, ArrayBuilderDataType,
        ArrayBuilderFillValue,
    },
    array_bytes::{
        copy_fill_value_into, update_array_bytes, ArrayBytes, ArrayBytesError, ArrayBytesOptional,
        RawBytes, RawBytesOffsets, RawBytesOffsetsCreateError, RawBytesOffsetsOutOfBoundsError,
        VariableLengthBytes,
    },
    array_bytes_fixed_disjoint_view::{
        ArrayBytesFixedDisjointView, ArrayBytesFixedDisjointViewCreateError,
    },
    array_errors::{AdditionalFieldUnsupportedError, ArrayCreateError, ArrayError},
    array_metadata_options::ArrayMetadataOptions,
    array_representation::{ArrayRepresentation, ArrayRepresentationBase, ChunkRepresentation},
    bytes_representation::BytesRepresentation,
    chunk_grid::ChunkGrid,
    chunk_key_encoding::{ChunkKeyEncoding, ChunkKeySeparator},
    codec::ArrayCodecTraits,
    codec::CodecChain,
    concurrency::RecommendedConcurrency,
    element::{Element, ElementFixedLength, ElementOwned},
    storage_transformer::StorageTransformerChain,
};
pub use data_type::{DataType, FillValue, NamedDataType};
use zarrs_registry::chunk_grid::REGULAR;

pub use crate::metadata::v2::{ArrayMetadataV2, FillValueMetadataV2};
pub use crate::metadata::v3::{
    ArrayMetadataV3, FillValueMetadataV3, ZARR_NAN_BF16, ZARR_NAN_F16, ZARR_NAN_F32, ZARR_NAN_F64,
};
pub use crate::metadata::{
    ArrayMetadata, ArrayShape, ChunkShape, DataTypeSize, DimensionName, Endianness,
};
use crate::plugin::PluginCreateError;
use crate::{
    array::chunk_grid::{RegularBoundedChunkGridConfiguration, RegularChunkGrid},
    config::global_config,
};
use zarrs_metadata_ext::v2_to_v3::ArrayMetadataV2ToV3Error;

pub use chunk_cache::{
    chunk_cache_lru::*, ChunkCache, ChunkCacheType, ChunkCacheTypeDecoded, ChunkCacheTypeEncoded,
    ChunkCacheTypePartialDecoder,
};

#[cfg(all(feature = "sharding", feature = "async"))]
pub use array_async_sharded_readable_ext::{
    AsyncArrayShardedReadableExt, AsyncArrayShardedReadableExtCache,
};
#[cfg(feature = "dlpack")]
pub use array_dlpack_ext::{
    ArrayDlPackExt, ArrayDlPackExtError, AsyncArrayDlPackExt, RawBytesDlPack,
};
#[cfg(feature = "sharding")]
pub use array_sharded_ext::ArrayShardedExt;
#[cfg(feature = "sharding")]
pub use array_sync_sharded_readable_ext::{ArrayShardedReadableExt, ArrayShardedReadableExtCache};

use zarrs_metadata::v2::DataTypeMetadataV2;

use crate::{
    array_subset::{ArraySubset, IncompatibleDimensionalityError},
    config::MetadataConvertVersion,
    node::{data_key, NodePath},
    storage::StoreKey,
};
use zarrs_metadata_ext::v2_to_v3::array_metadata_v2_to_v3;

/// An ND index to an element in an array.
pub type ArrayIndices = Vec<u64>;

/// Convert a [`ChunkShape`] reference to an [`ArrayShape`].
#[must_use]
pub fn chunk_shape_to_array_shape(chunk_shape: &[std::num::NonZeroU64]) -> ArrayShape {
    chunk_shape.iter().map(|i| i.get()).collect()
}

/// A Zarr array.
///
/// ## Initilisation
/// The easiest way to create a *new* Zarr V3 array is with an [`ArrayBuilder`].
/// Alternatively, a new Zarr V2 or Zarr V3 array can be created with [`Array::new_with_metadata`].
///
/// An *existing* Zarr V2 or Zarr V3 array can be initialised with [`Array::open`] or [`Array::open_opt`] with metadata read from the store.
///
/// [`Array`] initialisation will error if [`ArrayMetadata`] contains:
///  - unsupported extension points, including extensions which are supported by `zarrs` but have not been enabled with the appropriate features gates, or
///  - incompatible codecs (e.g. codecs in wrong order, codecs incompatible with data type, etc.),
///  - a chunk grid incompatible with the array shape,
///  - a fill value incompatible with the data type, or
///  - the metadata is in invalid in some other way.
///
/// ## Array Metadata
/// Array metadata **must be explicitly stored** with [`store_metadata`](Array::store_metadata) or [`store_metadata_opt`](Array::store_metadata_opt) if an array is newly created or its metadata has been mutated.
///
/// The underlying metadata of an [`Array`] can be accessed with [`metadata`](Array::metadata) or [`metadata_opt`](Array::metadata_opt).
/// The latter accepts [`ArrayMetadataOptions`] that can be used to convert array metadata from Zarr V2 to V3, for example.
/// [`metadata_opt`](Array::metadata_opt) is used internally by [`store_metadata`](Array::store_metadata) / [`store_metadata_opt`](Array::store_metadata_opt).
/// Use [`serde_json::to_string`] or [`serde_json::to_string_pretty`] on [`ArrayMetadata`] to convert it to a JSON string.
///
/// ### Immutable Array Metadata / Properties
///  - [`metadata`](Array::metadata): the underlying [`ArrayMetadata`] structure containing all array metadata
///  - [`data_type`](Array::data_type)
///  - [`fill_value`](Array::fill_value)
///  - [`chunk_grid`](Array::chunk_grid)
///  - [`chunk_key_encoding`](Array::chunk_key_encoding)
///  - [`codecs`](Array::codecs)
///  - [`storage_transformers`](Array::storage_transformers)
///  - [`path`](Array::path)
///
/// ### Mutable Array Metadata
/// Do not forget to store metadata after mutation.
///  - [`shape`](Array::shape) / [`set_shape`](Array::set_shape) / [`set_shape_and_chunk_grid`](Array::set_shape_and_chunk_grid)
///  - [`attributes`](Array::attributes) / [`attributes_mut`](Array::attributes_mut)
///  - [`dimension_names`](Array::dimension_names) / [`set_dimension_names`](Array::set_dimension_names)
///
/// ### `zarrs` Metadata
/// By default, the `zarrs` version and a link to its source code is written to the `_zarrs` attribute in array metadata when calling [`store_metadata`](Array::store_metadata).
/// Override this behaviour globally with [`Config::set_include_zarrs_metadata`](crate::config::Config::set_include_zarrs_metadata) or call [`store_metadata_opt`](Array::store_metadata_opt) with an explicit [`ArrayMetadataOptions`].
///
/// ## Array Data
/// Array operations are divided into several categories based on the traits implemented for the backing [storage](crate::storage).
/// The core array methods are:
///  - [`[Async]ReadableStorageTraits`](crate::storage::ReadableStorageTraits): read array data and metadata
///    - [`retrieve_chunk_if_exists`](Array::retrieve_chunk_if_exists)
///    - [`retrieve_chunk`](Array::retrieve_chunk)
///    - [`retrieve_chunks`](Array::retrieve_chunks)
///    - [`retrieve_chunk_subset`](Array::retrieve_chunk_subset)
///    - [`retrieve_array_subset`](Array::retrieve_array_subset)
///    - [`retrieve_encoded_chunk`](Array::retrieve_encoded_chunk)
///    - [`partial_decoder`](Array::partial_decoder)
///  - [`[Async]WritableStorageTraits`](crate::storage::WritableStorageTraits): store/erase array data and metadata
///    - [`store_metadata`](Array::store_metadata)
///    - [`erase_metadata`](Array::erase_metadata)
///    - [`store_chunk`](Array::store_chunk)
///    - [`store_chunks`](Array::store_chunks)
///    - [`store_encoded_chunk`](Array::store_encoded_chunk)
///    - [`erase_chunk`](Array::erase_chunk)
///    - [`erase_chunks`](Array::erase_chunks)
///  - [`[Async]ReadableWritableStorageTraits`](crate::storage::ReadableWritableStorageTraits): store operations requiring reading *and* writing
///    - [`store_chunk_subset`](Array::store_chunk_subset)
///    - [`store_array_subset`](Array::store_array_subset)
///    - [`partial_encoder`](Array::partial_encoder)
///
/// Many `retrieve` and `store` methods have multiple variants:
///   - Standard variants store or retrieve data represented as [`ArrayBytes`] (representing fixed or variable length bytes).
///   - `_elements` suffix variants can store or retrieve chunks with a known type.
///   - `_ndarray` suffix variants can store or retrieve [`ndarray::Array`]s (requires `ndarray` feature).
///   - `_opt` suffix variants have a [`CodecOptions`](crate::array::codec::CodecOptions) parameter for fine-grained concurrency control and more.
///   - Variants without the `_opt` suffix use default [`CodecOptions`](crate::array::codec::CodecOptions).
///   - **Experimental**: `async_` prefix variants can be used with async stores (requires `async` feature).
///
/// Additional [`Array`] methods are offered by extension traits:
///  - [`ArrayShardedExt`] and [`ArrayShardedReadableExt`]: see [Reading Sharded Arrays](#reading-sharded-arrays).
///  - [`[Async]ArrayDlPackExt`](ArrayDlPackExt): methods for [`DLPack`](https://arrow.apache.org/docs/python/dlpack.html) tensor interop.
///
/// [`ChunkCache`] implementations offer a similar API to [`Array::ReadableStorageTraits`](crate::storage::ReadableStorageTraits), except with [Chunk Caching](#chunk-caching) support.
///
/// ### Chunks and Array Subsets
/// Several convenience methods are available for querying the underlying chunk grid:
///  - [`chunk_origin`](Array::chunk_origin)
///  - [`chunk_shape`](Array::chunk_shape)
///  - [`chunk_subset`](Array::chunk_subset)
///  - [`chunk_subset_bounded`](Array::chunk_subset_bounded)
///  - [`chunks_subset`](Array::chunks_subset) / [`chunks_subset_bounded`](Array::chunks_subset_bounded)
///  - [`chunks_in_array_subset`](Array::chunks_in_array_subset)
///
/// An [`ArraySubset`] spanning the entire array can be retrieved with [`subset_all`](Array::subset_all).
///
/// ## Example: Update an Array Chunk-by-Chunk (in Parallel)
/// In the below example, an array is updated chunk-by-chunk in parallel.
/// This makes use of [`chunk_subset_bounded`](Array::chunk_subset_bounded) to retrieve and store only the subset of chunks that are within the array bounds.
/// This can occur when a regular chunk grid does not evenly divide the array shape, for example.
///
/// ```rust
/// # use std::sync::Arc;
/// # use zarrs::array::{Array, ArrayBytes};
/// # use zarrs::array_subset::ArraySubset;
/// # use zarrs::array_subset::iterators::Indices;
/// # use zarrs::indexer::Indexer;
/// # use rayon::iter::{IntoParallelIterator, ParallelIterator};
/// # let store = Arc::new(zarrs_filesystem::FilesystemStore::new("tests/data/array_write_read.zarr")?);
/// # let array = Array::open(store, "/group/array")?;
/// // Get an iterator over the chunk indices
/// //   The array shape must have been set (i.e. non-zero), otherwise the
/// //   iterator will be empty
/// let chunk_grid_shape = array.chunk_grid_shape();
/// let chunks: Indices = ArraySubset::new_with_shape(chunk_grid_shape.to_vec()).indices();
///
/// // Iterate over chunk indices (in parallel)
/// chunks.into_par_iter().try_for_each(|chunk_indices: Vec<u64>| {
///     // Retrieve the array subset of the chunk within the array bounds
///     //   This partially decodes chunks that extend beyond the array end
///     let subset: ArraySubset = array.chunk_subset_bounded(&chunk_indices)?;
///     let chunk_bytes: ArrayBytes = array.retrieve_array_subset(&subset)?;
///
///     // ... Update the chunk bytes
///
///     // Write the updated chunk
///     //   Elements beyond the array bounds in straddling chunks are left
///     //   unmodified or set to the fill value if the chunk did not exist.
///     array.store_array_subset(&subset, chunk_bytes)
/// })?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// ## Optimising Writes
/// For optimum write performance, an array should be written using [`store_chunk`](Array::store_chunk) or [`store_chunks`](Array::store_chunks) where possible.
///
/// [`store_chunk_subset`](Array::store_chunk_subset) and [`store_array_subset`](Array::store_array_subset) may incur decoding overhead, and they require careful usage if executed in parallel (see [Parallel Writing](#parallel-writing) below).
/// However, these methods will use a fast path and avoid decoding if the subset covers entire chunks.
///
/// ### Direct IO (Linux)
/// If using Linux, enabling direct IO with the [`FilesystemStore`](https://docs.rs/zarrs_filesystem/latest/zarrs_filesystem/struct.FilesystemStore.html) may improve write performance.
///
/// Currently, the most performant path for uncompressed writing is to reuse page aligned buffers via [`store_encoded_chunk`](Array::store_encoded_chunk).
/// See [`zarrs` GitHub issue #58](https://github.com/zarrs/zarrs/pull/58) for a discussion on this method.
//  TODO: Add example?
///
/// ### Parallel Writing
/// `zarrs` does not currently offer a "synchronisation" API for locking chunks or array subsets.
///
/// **It is the responsibility of `zarrs` consumers to ensure that chunks are not written to concurrently**.
///
/// If a chunk is written more than once, its element values depend on whichever operation wrote to the chunk last.
/// The [`store_chunk_subset`](Array::store_chunk_subset) and [`store_array_subset`](Array::store_array_subset) methods and their variants internally retrieve, update, and store chunks.
/// So do [`partial_encoder`](Array::partial_encoder)s, which may used internally by the above methods.
///
/// It is the responsibility of `zarrs` consumers to ensure that:
///   - [`store_array_subset`](Array::store_array_subset) is not called concurrently on array subsets sharing chunks,
///   - [`store_chunk_subset`](Array::store_chunk_subset) is not called concurrently on the same chunk,
///   - [`partial_encoder`](Array::partial_encoder)s are created or used concurrently for the same chunk,
///   - or any combination of the above are called concurrently on the same chunk.
///
/// **Partial writes to a chunk may be lost if these rules are not respected.**
///
/// ## Optimising Reads
/// It is fastest to load arrays using [`retrieve_chunk`](Array::retrieve_chunk) or [`retrieve_chunks`](Array::retrieve_chunks) where possible.
/// In contrast, the [`retrieve_chunk_subset`](Array::retrieve_chunk_subset) and [`retrieve_array_subset`](Array::retrieve_array_subset) may use partial decoders which can be less efficient with some codecs/stores.
/// Like their write counterparts, these methods will use a fast path if subsets cover entire chunks.
///
/// **Standard [`Array`] retrieve methods do not perform any caching**.
/// For this reason, retrieving multiple subsets in a chunk with [`retrieve_chunk_subset`](Array::store_chunk_subset) is very inefficient and strongly discouraged.
/// For example, consider that a compressed chunk may need to be retrieved and decoded in its entirety even if only a small part of the data is needed.
/// In such situations, prefer to initialise a partial decoder for a chunk with [`partial_decoder`](Array::partial_decoder) and then retrieve multiple chunk subsets with [`partial_decode`](codec::ArrayPartialDecoderTraits::partial_decode).
/// The underlying codec chain will use a cache where efficient to optimise multiple partial decoding requests (see [`CodecChain`]).
/// Another alternative is to use [Chunk Caching](#chunk-caching).
///
/// ### Chunk Caching
/// `zarrs` supports three types of chunk caches:
/// - [`ChunkCacheTypeDecoded`]: caches decoded chunks.
///   - Preferred where decoding is expensive and memory is abundant.
/// - [`ChunkCacheTypeEncoded`]: caches encoded chunks.
///   - Preferred where decoding is cheap and memory is scarce, provided that data is well compressed/sparse.
/// - [`ChunkCacheTypePartialDecoder`]: caches partial decoders.
///   - Preferred where chunks are repeatedly *partially retrieved*.
///   - Useful for retrieval of inner chunks from sharded arrays, as the partial decoder caches shard indexes (but **not** inner chunks).
///   - Memory usage of this cache is highly dependent on the array codecs and whether the codec chain ([`Array::codecs`]) ends up decoding entire chunks or caching inputs based on their [`PartialDecoderCapability`](crate::array::codec::PartialDecoderCapability).
///
/// `zarrs` implements the following Least Recently Used (LRU) chunk caches:
///  - [`ChunkCacheDecodedLruChunkLimit`]: a decoded chunk cache with a fixed chunk capacity..
///  - [`ChunkCacheDecodedLruSizeLimit`]: a decoded chunk cache with a fixed size in bytes.
///  - [`ChunkCacheEncodedLruChunkLimit`]: an encoded chunk cache with a fixed chunk capacity.
///  - [`ChunkCacheEncodedLruSizeLimit`]: an encoded chunk cache with a fixed size in bytes.
///  - [`ChunkCachePartialDecoderLruChunkLimit`]: a partial decoder chunk cache with a fixed chunk capacity
///  - [`ChunkCachePartialDecoderLruSizeLimit`]: a partial decoder chunk cache with a fixed size in bytes.
///
/// There are also `ThreadLocal` suffixed variants of all of these caches that have a per-thread cache.
/// `zarrs` consumers can create custom caches by implementing the [`ChunkCache`] trait.
///
/// Chunk caches implement the [`ChunkCache`] trait which has cached versions of the equivalent [`Array`] methods:
///  - [`retrieve_chunk`](ChunkCache::retrieve_chunk)
///  - [`retrieve_chunks`](ChunkCache::retrieve_chunks)
///  - [`retrieve_chunk_subset`](ChunkCache::retrieve_chunk_subset)
///  - [`retrieve_array_subset`](ChunkCache::retrieve_array_subset)
///
/// `_elements` and `_ndarray` variants are also available.
///
/// Chunk caching is likely to be effective for remote stores where redundant retrievals are costly.
/// Chunk caching may not outperform disk caching with a filesystem store.
/// The above caches use internal locking to support multithreading, which has a performance overhead.
/// **Prefer not to use a chunk cache if chunks are not accessed repeatedly**.
/// Aside from [`ChunkCacheTypePartialDecoder`]-based caches, caches do not use partial decoders and any intersected chunk is fully retrieved if not present in the cache.
///
/// For many access patterns, chunk caching may reduce performance.
/// **Benchmark your algorithm/data.**
///
/// ## Reading Sharded Arrays
/// The `sharding_indexed` codec ([`ShardingCodec`](codec::array_to_bytes::sharding)) enables multiple sub-chunks ("inner chunks") to be stored in a single chunk ("shard").
/// With a sharded array, the [`chunk_grid`](Array::chunk_grid) and chunk indices in store/retrieve methods reference the chunks ("shards") of an array.
///
/// The [`ArrayShardedExt`] trait provides additional methods to [`Array`] to query if an array is sharded and retrieve the inner chunk shape.
/// Additionally, the *inner chunk grid* can be queried, which is a [`ChunkGrid`](chunk_grid) where chunk indices refer to inner chunks rather than shards.
///
/// The [`ArrayShardedReadableExt`] trait adds [`Array`] methods to conveniently and efficiently access the data in a sharded array (with `_elements` and `_ndarray` variants):
///  - [`retrieve_inner_chunk_opt`](ArrayShardedReadableExt::retrieve_inner_chunk_opt)
///  - [`retrieve_inner_chunks_opt`](ArrayShardedReadableExt::retrieve_inner_chunks_opt)
///  - [`retrieve_array_subset_sharded_opt`](ArrayShardedReadableExt::retrieve_array_subset_sharded_opt)
///
/// For unsharded arrays, these methods gracefully fallback to referencing standard chunks.
/// Each method has a `cache` parameter ([`ArrayShardedReadableExtCache`]) that stores shard indexes so that they do not have to be repeatedly retrieved and decoded.
///
/// ## Parallelism and Concurrency
/// ### Sync API
/// Codecs run in parallel using a dedicated threadpool.
/// Array store and retrieve methods will also run in parallel when they involve multiple chunks.
/// `zarrs` will automatically choose where to prioritise parallelism between codecs/chunks based on the codecs and number of chunks.
///
/// By default, all available CPU cores will be used (where possible/efficient).
/// Concurrency can be limited globally with [`Config::set_codec_concurrent_target`](crate::config::Config::set_codec_concurrent_target) or as required using `_opt` methods with [`CodecOptions`](crate::array::codec::CodecOptions) manipulated with [`CodecOptions::set_concurrent_target`](crate::array::codec::CodecOptions::set_concurrent_target).
///
/// ### Async API
/// This crate is async runtime-agnostic.
/// Async methods do not spawn tasks internally, so asynchronous storage calls are concurrent but not parallel.
/// Codec encoding and decoding operations still execute in parallel (where supported) in an asynchronous context.
///
/// Due the lack of parallelism, methods like [`async_retrieve_array_subset`](Array::async_retrieve_array_subset) or [`async_retrieve_chunks`](Array::async_retrieve_chunks) do not parallelise over chunks and can be slow compared to the sync API.
/// Parallelism over chunks can be achieved by spawning tasks outside of `zarrs`.
/// A crate like [`async-scoped`](https://crates.io/crates/async-scoped) can enable spawning non-`'static` futures.
/// If executing many tasks concurrently, consider reducing the codec [`concurrent_target`](crate::array::codec::CodecOptions::set_concurrent_target).
#[derive(Debug)]
pub struct Array<TStorage: ?Sized> {
    /// The storage (including storage transformers).
    storage: Arc<TStorage>,
    /// The path of the array in a store.
    path: NodePath,
    /// The data type of the Zarr array.
    data_type: NamedDataType,
    /// The chunk grid of the Zarr array.
    chunk_grid: ChunkGrid,
    /// The mapping from chunk grid cell coordinates to keys in the underlying store.
    chunk_key_encoding: ChunkKeyEncoding,
    /// Provides an element value to use for uninitialised portions of the Zarr array. It encodes the underlying data type.
    fill_value: FillValue,
    /// Specifies a list of codecs to be used for encoding and decoding chunks.
    codecs: Arc<CodecChain>,
    /// An optional list of storage transformers.
    storage_transformers: StorageTransformerChain,
    /// An optional list of dimension names.
    dimension_names: Option<Vec<DimensionName>>,
    /// Metadata used to create the array
    metadata: ArrayMetadata,
}

impl<TStorage: ?Sized> Array<TStorage> {
    /// Replace the storage backing an array.
    pub fn with_storage<TStorage2: ?Sized>(&self, storage: Arc<TStorage2>) -> Array<TStorage2> {
        Array {
            storage,
            path: self.path.clone(),
            data_type: self.data_type.clone(),
            chunk_grid: self.chunk_grid.clone(),
            chunk_key_encoding: self.chunk_key_encoding.clone(),
            fill_value: self.fill_value.clone(),
            codecs: self.codecs.clone(),
            storage_transformers: self.storage_transformers.clone(),
            dimension_names: self.dimension_names.clone(),
            metadata: self.metadata.clone(),
        }
    }

    /// Create an array in `storage` at `path` with `metadata`.
    /// This does **not** write to the store, use [`store_metadata`](Array<WritableStorageTraits>::store_metadata) to write `metadata` to `storage`.
    ///
    /// # Errors
    /// Returns [`ArrayCreateError`] if:
    ///  - any metadata is invalid or,
    ///  - a plugin (e.g. data type/chunk grid/chunk key encoding/codec/storage transformer) is invalid.
    pub fn new_with_metadata(
        storage: Arc<TStorage>,
        path: &str,
        metadata: ArrayMetadata,
    ) -> Result<Self, ArrayCreateError> {
        let path = NodePath::new(path)?;

        // Convert V2 metadata to V3 if it is a compatible subset
        let metadata_v3 = {
            let config = global_config();
            match &metadata {
                ArrayMetadata::V3(v3) => Ok(v3.clone()),
                ArrayMetadata::V2(v2) => array_metadata_v2_to_v3(
                    v2,
                    config.codec_aliases_v2(),
                    config.codec_aliases_v3(),
                    config.data_type_aliases_v2(),
                    config.data_type_aliases_v3(),
                )
                .map_err(|err| ArrayCreateError::UnsupportedZarrV2Array(err.to_string())),
            }?
        };

        let data_type = DataType::from_metadata(
            &metadata_v3.data_type,
            global_config().data_type_aliases_v3(),
        )
        .map_err(ArrayCreateError::DataTypeCreateError)?;
        let data_type = NamedDataType::new(metadata_v3.data_type.name().to_string(), data_type);
        let chunk_grid = ChunkGrid::from_metadata(&metadata_v3.chunk_grid, &metadata_v3.shape)
            .map_err(ArrayCreateError::ChunkGridCreateError)?;
        if chunk_grid.dimensionality() != metadata_v3.shape.len() {
            return Err(ArrayCreateError::InvalidChunkGridDimensionality(
                chunk_grid.dimensionality(),
                metadata_v3.shape.len(),
            ));
        }
        let fill_value = data_type
            .fill_value_from_metadata(&metadata_v3.fill_value)
            .map_err(ArrayCreateError::InvalidFillValueMetadata)?;
        let codecs = Arc::new(
            CodecChain::from_metadata(&metadata_v3.codecs)
                .map_err(ArrayCreateError::CodecsCreateError)?,
        );
        let storage_transformers =
            StorageTransformerChain::from_metadata(&metadata_v3.storage_transformers, &path)
                .map_err(ArrayCreateError::StorageTransformersCreateError)?;
        let chunk_key_encoding = ChunkKeyEncoding::from_metadata(&metadata_v3.chunk_key_encoding)
            .map_err(ArrayCreateError::ChunkKeyEncodingCreateError)?;
        if let Some(dimension_names) = &metadata_v3.dimension_names {
            if dimension_names.len() != metadata_v3.shape.len() {
                return Err(ArrayCreateError::InvalidDimensionNames(
                    dimension_names.len(),
                    metadata_v3.shape.len(),
                ));
            }
        }

        Ok(Self {
            storage,
            path,
            // shape: metadata_v3.shape,
            data_type,
            chunk_grid,
            chunk_key_encoding,
            fill_value,
            codecs,
            // attributes: metadata_v3.attributes,
            // additional_fields: metadata_v3.additional_fields,
            storage_transformers,
            dimension_names: metadata_v3.dimension_names,
            metadata,
        })
    }

    /// Get the underlying storage backing the array.
    #[must_use]
    pub fn storage(&self) -> Arc<TStorage> {
        self.storage.clone()
    }

    /// Get the node path.
    #[must_use]
    pub const fn path(&self) -> &NodePath {
        &self.path
    }

    /// Get the data type.
    #[must_use]
    pub const fn data_type(&self) -> &DataType {
        self.data_type.data_type()
    }

    /// Get the fill value.
    #[must_use]
    pub const fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    /// Get the array shape.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        self.chunk_grid().array_shape()
    }

    /// Set the array shape.
    ///
    /// # Errors
    /// Returns an [`ArrayCreateError`] if the chunk grid is not compatible with `array_shape`.
    pub fn set_shape(&mut self, array_shape: ArrayShape) -> Result<&mut Self, ArrayCreateError> {
        self.chunk_grid =
            ChunkGrid::from_metadata(&self.chunk_grid.create_metadata(), &array_shape)
                .map_err(ArrayCreateError::ChunkGridCreateError)?;
        match &mut self.metadata {
            ArrayMetadata::V3(metadata) => {
                metadata.shape = array_shape;
            }
            ArrayMetadata::V2(metadata) => {
                metadata.shape = array_shape;
            }
        }
        Ok(self)
    }

    /// Set the array shape and chunk grid from chunk grid metadata.
    ///
    /// This method allows setting both the array shape and chunk grid simultaneously.
    /// Some chunk grids depend on the array shape (e.g. `rectilinear`), so this method ensures that the chunk grid is correctly configured for the new array shape.
    ///
    /// # Errors
    /// Returns an [`ArrayCreateError`] if:
    ///  - the chunk grid is not compatible with `array_shape`, or
    ///  - the chunk grid metadata is invalid.
    ///
    /// # Safety
    /// This method does not validate that existing chunks in the store are compatible with the new chunk grid.
    /// If the chunk grid is changed such that existing chunks are no longer valid, subsequent read or write operations may fail or produce incorrect results.
    ///
    /// It is the caller's responsibility to ensure that the new chunk grid is compatible with any existing data in the store.
    /// This may involve deleting or rewriting existing chunks to match the new chunk grid.
    /// Use with caution!
    pub unsafe fn set_shape_and_chunk_grid(
        &mut self,
        array_shape: ArrayShape,
        chunk_grid_metadata: impl Into<ArrayBuilderChunkGridMetadata>,
    ) -> Result<&mut Self, ArrayCreateError> {
        let chunk_grid_metadata: ArrayBuilderChunkGridMetadata = chunk_grid_metadata.into();
        let chunk_grid_metadata = chunk_grid_metadata.to_metadata()?;

        // Create the new chunk grid
        self.chunk_grid = ChunkGrid::from_metadata(&chunk_grid_metadata, &array_shape)
            .map_err(ArrayCreateError::ChunkGridCreateError)?;

        // Update metadata based on version
        match &mut self.metadata {
            ArrayMetadata::V3(metadata) => {
                metadata.shape = array_shape;
                metadata.chunk_grid = chunk_grid_metadata;
            }
            ArrayMetadata::V2(metadata) => {
                let err = || {
                    ArrayCreateError::ChunkGridCreateError(PluginCreateError::Other(
                        "Only regular chunk grids are supported in Zarr V2".to_string(),
                    ))
                };
                if chunk_grid_metadata.name() != REGULAR {
                    return Err(err());
                }
                let regular_chunk_grid_configuration = chunk_grid_metadata
                    .to_configuration::<RegularBoundedChunkGridConfiguration>()
                    .map_err(|_| err())?;
                let regular_chunk_grid = RegularChunkGrid::new(
                    array_shape.clone(),
                    regular_chunk_grid_configuration.chunk_shape,
                )
                .map_err(|_| {
                    ArrayCreateError::ChunkGridCreateError(PluginCreateError::Other(
                        "Chunk grid is not compatible with array shape".to_string(),
                    ))
                })?;
                metadata.shape = array_shape;
                metadata.chunks = regular_chunk_grid.chunk_shape().to_vec().into();
            }
        }
        Ok(self)
    }

    /// Get the array dimensionality.
    #[must_use]
    pub fn dimensionality(&self) -> usize {
        self.shape().len()
    }

    /// Get the codecs.
    #[must_use]
    pub fn codecs(&self) -> Arc<CodecChain> {
        self.codecs.clone()
    }

    /// Get the chunk grid.
    #[must_use]
    pub const fn chunk_grid(&self) -> &ChunkGrid {
        &self.chunk_grid
    }

    /// Get the chunk key encoding.
    #[must_use]
    pub const fn chunk_key_encoding(&self) -> &ChunkKeyEncoding {
        &self.chunk_key_encoding
    }

    /// Get the storage transformers.
    #[must_use]
    pub const fn storage_transformers(&self) -> &StorageTransformerChain {
        &self.storage_transformers
    }

    /// Get the dimension names.
    #[must_use]
    pub const fn dimension_names(&self) -> &Option<Vec<DimensionName>> {
        &self.dimension_names
    }

    /// Set the dimension names.
    pub fn set_dimension_names(
        &mut self,
        dimension_names: Option<Vec<DimensionName>>,
    ) -> &mut Self {
        self.dimension_names = dimension_names;
        self
    }

    /// Get the attributes.
    #[must_use]
    pub const fn attributes(&self) -> &serde_json::Map<String, serde_json::Value> {
        match &self.metadata {
            ArrayMetadata::V3(metadata) => &metadata.attributes,
            ArrayMetadata::V2(metadata) => &metadata.attributes,
        }
    }

    /// Mutably borrow the array attributes.
    #[must_use]
    pub fn attributes_mut(&mut self) -> &mut serde_json::Map<String, serde_json::Value> {
        match &mut self.metadata {
            ArrayMetadata::V3(metadata) => &mut metadata.attributes,
            ArrayMetadata::V2(metadata) => &mut metadata.attributes,
        }
    }

    /// Return the underlying array metadata.
    #[must_use]
    pub fn metadata(&self) -> &ArrayMetadata {
        &self.metadata
    }

    /// Return a new [`ArrayMetadata`] with [`ArrayMetadataOptions`] applied.
    ///
    /// This method is used internally by [`Array::store_metadata`] and [`Array::store_metadata_opt`].
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn metadata_opt(&self, options: &ArrayMetadataOptions) -> ArrayMetadata {
        use ArrayMetadata as AM;
        use MetadataConvertVersion as V;
        let mut metadata = self.metadata.clone();

        // Attribute manipulation
        if options.include_zarrs_metadata() {
            #[derive(serde::Serialize)]
            struct ZarrsMetadata {
                description: String,
                repository: String,
                version: String,
            }
            let zarrs_metadata = ZarrsMetadata {
                description: "This array was created with zarrs".to_string(),
                repository: env!("CARGO_PKG_REPOSITORY").to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            };
            let attributes = match &mut metadata {
                AM::V3(metadata) => &mut metadata.attributes,
                AM::V2(metadata) => &mut metadata.attributes,
            };
            attributes.insert("_zarrs".to_string(), unsafe {
                serde_json::to_value(zarrs_metadata).unwrap_unchecked()
            });
        }

        // Codec metadata manipulation
        match &mut metadata {
            ArrayMetadata::V3(metadata) => {
                metadata.codecs = self.codecs().create_metadatas_opt(options.codec_options());
            }
            ArrayMetadata::V2(_metadata) => {
                // NOTE: The codec related options in ArrayMetadataOptions do not impact V2 codecs
            }
        }

        // Convert version
        let mut metadata = match (metadata, options.metadata_convert_version()) {
            (AM::V3(metadata), V::Default | V::V3) => ArrayMetadata::V3(metadata),
            (AM::V2(metadata), V::Default) => ArrayMetadata::V2(metadata),
            (AM::V2(metadata), V::V3) => {
                let metadata = {
                    let config = global_config();
                    array_metadata_v2_to_v3(
                        &metadata,
                        config.codec_aliases_v2(),
                        config.codec_aliases_v3(),
                        config.data_type_aliases_v2(),
                        config.data_type_aliases_v3(),
                    )
                    .expect("conversion succeeded on array creation")
                };
                AM::V3(metadata)
            }
        };

        // Convert aliased extension names
        if options.convert_aliased_extension_names() {
            let config = global_config();
            match &mut metadata {
                AM::V3(metadata) => {
                    let codec_aliases = config.codec_aliases_v3();
                    metadata.codecs.iter_mut().for_each(|codec| {
                        let identifier = codec_aliases.identifier(codec.name());
                        codec.set_name(codec_aliases.default_name(identifier).to_string());
                    });
                    let data_type_aliases = config.data_type_aliases_v3();
                    {
                        let name = metadata.data_type.name();
                        let identifier = data_type_aliases.identifier(name);
                        metadata
                            .data_type
                            .set_name(data_type_aliases.default_name(identifier).to_string());
                    }
                }
                AM::V2(metadata) => {
                    let codec_aliases = config.codec_aliases_v2();
                    {
                        if let Some(filters) = &mut metadata.filters {
                            for filter in filters.iter_mut() {
                                let identifier = codec_aliases.identifier(filter.id());
                                filter.set_id(codec_aliases.default_name(identifier).to_string());
                            }
                        }
                        if let Some(compressor) = &mut metadata.compressor {
                            let identifier = codec_aliases.identifier(compressor.id());
                            compressor.set_id(codec_aliases.default_name(identifier).to_string());
                        }
                    }
                    let data_type_aliases = config.data_type_aliases_v2();
                    {
                        match &mut metadata.dtype {
                            DataTypeMetadataV2::Simple(dtype) => {
                                let identifier = data_type_aliases.identifier(dtype);
                                *dtype = data_type_aliases.default_name(identifier).to_string();
                            }
                            DataTypeMetadataV2::Structured(_) => {
                                // FIXME: structured data type support
                            }
                        }
                    }
                }
            }
        }

        metadata
    }

    pub(crate) fn fill_value_metadata_v3(&self) -> FillValueMetadataV3 {
        self.data_type
            .metadata_fill_value(&self.fill_value)
            .expect("data type and fill value are compatible")
    }

    /// Create an array builder matching the parameters of this array.
    #[must_use]
    pub fn builder(&self) -> ArrayBuilder {
        ArrayBuilder::from_array(self)
    }

    /// Return the shape of the chunk grid (i.e., the number of chunks).
    #[must_use]
    pub fn chunk_grid_shape(&self) -> &ArrayShape {
        self.chunk_grid().grid_shape()
    }

    /// Return the [`StoreKey`] of the chunk at `chunk_indices`.
    #[must_use]
    pub fn chunk_key(&self, chunk_indices: &[u64]) -> StoreKey {
        data_key(self.path(), &self.chunk_key_encoding.encode(chunk_indices))
    }

    /// Return the origin of the chunk at `chunk_indices`.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    pub fn chunk_origin(&self, chunk_indices: &[u64]) -> Result<ArrayIndices, ArrayError> {
        self.chunk_grid()
            .chunk_origin(chunk_indices)
            .map_err(|_| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))
    }

    /// Return the shape of the chunk at `chunk_indices`.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    pub fn chunk_shape(&self, chunk_indices: &[u64]) -> Result<ChunkShape, ArrayError> {
        self.chunk_grid()
            .chunk_shape(chunk_indices)
            .map_err(|_| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))
    }

    /// Return an array subset that spans the entire array.
    #[must_use]
    pub fn subset_all(&self) -> ArraySubset {
        ArraySubset::new_with_shape(self.shape().to_vec())
    }

    /// Return the shape of the chunk at `chunk_indices`.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    ///
    /// # Panics
    /// Panics if any component of the chunk shape exceeds [`usize::MAX`].
    pub fn chunk_shape_usize(&self, chunk_indices: &[u64]) -> Result<Vec<usize>, ArrayError> {
        Ok(self
            .chunk_shape(chunk_indices)?
            .iter()
            .map(|d| usize::try_from(d.get()).unwrap())
            .collect())
    }

    /// Return the array subset of the chunk at `chunk_indices`.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    pub fn chunk_subset(&self, chunk_indices: &[u64]) -> Result<ArraySubset, ArrayError> {
        self.chunk_grid()
            .subset(chunk_indices)
            .map_err(|_| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))
    }

    /// Return the array subset of the chunk at `chunk_indices` bounded by the array shape.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    pub fn chunk_subset_bounded(&self, chunk_indices: &[u64]) -> Result<ArraySubset, ArrayError> {
        let chunk_subset = self.chunk_subset(chunk_indices)?;
        Ok(chunk_subset.bound(self.shape())?)
    }

    /// Return the array subset of `chunks`.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if a chunk in `chunks` is incompatible with the chunk grid.
    #[allow(clippy::similar_names)]
    pub fn chunks_subset(&self, chunks: &ArraySubset) -> Result<ArraySubset, ArrayError> {
        match chunks.end_inc() {
            Some(end) => {
                let chunk0 = self.chunk_subset(chunks.start())?;
                let chunk1 = self.chunk_subset(&end)?;
                let start = chunk0.start().to_vec();
                let end = chunk1.end_exc();
                ArraySubset::new_with_start_end_exc(start, end).map_err(std::convert::Into::into)
            }
            None => Ok(ArraySubset::new_empty(chunks.dimensionality())),
        }
    }

    /// Return the array subset of `chunks` bounded by the array shape.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    pub fn chunks_subset_bounded(&self, chunks: &ArraySubset) -> Result<ArraySubset, ArrayError> {
        let chunks_subset = self.chunks_subset(chunks)?;
        Ok(chunks_subset.bound(self.shape())?)
    }

    /// Get the chunk array representation at `chunk_index`.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    pub fn chunk_array_representation(
        &self,
        chunk_indices: &[u64],
    ) -> Result<ChunkRepresentation, ArrayError> {
        self.chunk_grid().chunk_shape(chunk_indices)?.map_or_else(
            || {
                Err(ArrayError::InvalidChunkGridIndicesError(
                    chunk_indices.to_vec(),
                ))
            },
            |chunk_shape| {
                Ok(unsafe {
                    ChunkRepresentation::new_unchecked(
                        chunk_shape.to_vec(),
                        self.data_type().clone(),
                        self.fill_value().clone(),
                    )
                })
            },
        )
    }

    /// Return an array subset indicating the chunks intersecting `array_subset`.
    ///
    /// Returns [`None`] if the intersecting chunks cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if the array subset has an incorrect dimensionality.
    pub fn chunks_in_array_subset(
        &self,
        array_subset: &ArraySubset,
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        self.chunk_grid.chunks_in_array_subset(array_subset)
    }

    /// Calculate the recommended codec concurrency.
    fn recommended_codec_concurrency(
        &self,
        chunk_representation: &ChunkRepresentation,
    ) -> Result<RecommendedConcurrency, ArrayError> {
        Ok(self
            .codecs()
            .recommended_concurrency(chunk_representation)?)
    }

    /// Convert the array to Zarr V3.
    ///
    /// # Errors
    /// Returns a [`ArrayMetadataV2ToV3Error`] if the metadata is not compatible with Zarr V3 metadata.
    pub fn to_v3(self) -> Result<Self, ArrayMetadataV2ToV3Error> {
        match self.metadata {
            ArrayMetadata::V2(metadata) => {
                let metadata: ArrayMetadata = {
                    let config = global_config();
                    array_metadata_v2_to_v3(
                        &metadata,
                        config.codec_aliases_v2(),
                        config.codec_aliases_v3(),
                        config.data_type_aliases_v2(),
                        config.data_type_aliases_v3(),
                    )?
                    .into()
                };
                Ok(Self {
                    storage: self.storage,
                    path: self.path,
                    data_type: self.data_type,
                    chunk_grid: self.chunk_grid,
                    chunk_key_encoding: self.chunk_key_encoding,
                    fill_value: self.fill_value,
                    codecs: self.codecs,
                    storage_transformers: self.storage_transformers,
                    dimension_names: self.dimension_names,
                    metadata,
                })
            }
            ArrayMetadata::V3(_) => Ok(self),
        }
    }

    /// Reject the array if it contains unsupported extensions or additional fields with `"must_understand": true`.
    fn validate_metadata(metadata: &ArrayMetadata) -> Result<(), ArrayCreateError> {
        match &metadata {
            ArrayMetadata::V2(_) => {}
            ArrayMetadata::V3(_metadata) => {
                // for extension in &metadata.extensions {
                //     if extension.must_understand() {
                //         return Err(ArrayCreateError::AdditionalFieldUnsupportedError(
                //             AdditionalFieldUnsupportedError::new(
                //                 extension.name().to_string(),
                //                 extension
                //                     .configuration()
                //                     .map(|configuration| {
                //                         serde_json::Value::Object(configuration.clone().into())
                //                     })
                //                     .unwrap_or_default(),
                //             ),
                //         ));
                //     }
                // }
            }
        }

        match metadata {
            ArrayMetadata::V2(_metadata) => {}
            ArrayMetadata::V3(metadata) => {
                let additional_fields = &metadata.additional_fields;
                for (name, field) in additional_fields {
                    if field.must_understand() {
                        return Err(ArrayCreateError::AdditionalFieldUnsupportedError(
                            AdditionalFieldUnsupportedError::new(
                                name.clone(),
                                field.as_value().clone(),
                            ),
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(feature = "ndarray")]
/// Convert an ndarray into a vec with standard layout
fn ndarray_into_vec<T: Clone, D: ndarray::Dimension>(array: ndarray::Array<T, D>) -> Vec<T> {
    #[allow(deprecated)]
    if array.is_standard_layout() {
        array
    } else {
        array.as_standard_layout().into_owned()
    }
    .into_raw_vec()
}

mod array_sync_readable;

mod array_sync_writable;

mod array_sync_readable_writable;

#[cfg(feature = "async")]
mod array_async_readable;

#[cfg(feature = "async")]
mod array_async_writable;

#[cfg(feature = "async")]
mod array_async_readable_writable;

#[cfg(feature = "async")]
mod array_async_sharded_readable_ext;

/// Transmute from `Vec<u8>` to `Vec<T>`.
#[must_use]
pub fn convert_from_bytes_slice<T: bytemuck::Pod>(from: &[u8]) -> Vec<T> {
    bytemuck::allocation::pod_collect_to_vec(from)
}

/// Transmute from `Vec<u8>` to `Vec<T>`.
#[must_use]
pub fn transmute_from_bytes_vec<T: bytemuck::Pod>(from: Vec<u8>) -> Vec<T> {
    bytemuck::allocation::try_cast_vec(from)
        .unwrap_or_else(|(_err, from)| convert_from_bytes_slice(&from))
}

/// Convert from `&[T]` to `Vec<u8>`.
#[must_use]
pub fn convert_to_bytes_vec<T: bytemuck::NoUninit>(from: &[T]) -> Vec<u8> {
    bytemuck::allocation::pod_collect_to_vec(from)
}

/// Transmute from `Vec<T>` to `Vec<u8>`.
#[must_use]
pub fn transmute_to_bytes_vec<T: bytemuck::NoUninit>(from: Vec<T>) -> Vec<u8> {
    bytemuck::allocation::try_cast_vec(from)
        .unwrap_or_else(|(_err, from)| convert_to_bytes_vec(&from))
}

/// Transmute from `&[T]` to `&[u8]`.
#[must_use]
pub fn transmute_to_bytes<T: bytemuck::NoUninit>(from: &[T]) -> &[u8] {
    bytemuck::must_cast_slice(from)
}

/// Unravel a linearised index to ND indices.
#[must_use]
pub fn unravel_index(mut index: u64, shape: &[u64]) -> Option<ArrayIndices> {
    let len = shape.len();
    let mut indices: ArrayIndices = Vec::with_capacity(len);
    for (indices_i, &dim) in std::iter::zip(
        indices.spare_capacity_mut().iter_mut().rev(),
        shape.iter().rev(),
    ) {
        indices_i.write(index % dim);
        index /= dim;
    }
    unsafe { indices.set_len(len) };
    if index == 0 {
        Some(indices)
    } else {
        None
    }
}

/// Ravel ND indices to a linearised index.
///
/// Returns [`None`] if any `indices` are out-of-bounds of `shape`.
#[must_use]
pub fn ravel_indices(indices: &[u64], shape: &[u64]) -> Option<u64> {
    let mut index: u64 = 0;
    let mut count = 1;
    for (i, s) in std::iter::zip(indices, shape).rev() {
        if i >= s {
            return None;
        }
        index += i * count;
        count *= s;
    }
    Some(index)
}

#[cfg(feature = "ndarray")]
fn iter_u64_to_usize<'a, I: Iterator<Item = &'a u64>>(iter: I) -> Vec<usize> {
    iter.map(|v| usize::try_from(*v).unwrap())
        .collect::<Vec<_>>()
}

#[cfg(feature = "ndarray")]
/// Convert a vector of elements to an [`ndarray::ArrayD`].
///
/// # Errors
/// Returns an error if the length of `elements` is not equal to the product of the components in `shape`.
pub fn elements_to_ndarray<T>(
    shape: &[u64],
    elements: Vec<T>,
) -> Result<ndarray::ArrayD<T>, ArrayError> {
    let length = elements.len();
    ndarray::ArrayD::<T>::from_shape_vec(iter_u64_to_usize(shape.iter()), elements).map_err(|_| {
        ArrayError::CodecError(codec::InvalidArrayShapeError::new(shape.to_vec(), length).into())
    })
}

#[cfg(feature = "ndarray")]
/// Convert a vector of bytes to an [`ndarray::ArrayD`].
///
/// # Errors
/// Returns an error if the length of `bytes` is not equal to the product of the components in `shape` and the size of `T`.
pub fn bytes_to_ndarray<T: bytemuck::Pod>(
    shape: &[u64],
    bytes: Vec<u8>,
) -> Result<ndarray::ArrayD<T>, ArrayError> {
    let expected_len = shape.iter().product::<u64>() * size_of::<T>() as u64;
    if bytes.len() as u64 != expected_len {
        return Err(ArrayError::InvalidBytesInputSize(bytes.len(), expected_len));
    }
    let elements = transmute_from_bytes_vec::<T>(bytes);
    elements_to_ndarray(shape, elements)
}

#[cfg(test)]
mod tests {
    use crate::storage::store::MemoryStore;
    use zarrs_filesystem::FilesystemStore;
    use zarrs_metadata::v3::{AdditionalFieldV3, AdditionalFieldsV3};

    use super::*;

    #[test]
    fn test_array_metadata_write_read() {
        let store = Arc::new(MemoryStore::new());

        let array_path = "/array";
        let array = ArrayBuilder::new(vec![8, 8], vec![4, 4], DataType::UInt8, 0u8)
            .build(store.clone(), array_path)
            .unwrap();
        array.store_metadata().unwrap();
        let stored_metadata = array.metadata_opt(&ArrayMetadataOptions::default());

        let array_other = Array::open(store, array_path).unwrap();
        assert_eq!(array_other.metadata(), &stored_metadata);
    }

    #[test]
    fn array_set_shape_and_attributes() {
        let store = MemoryStore::new();
        let array_path = "/group/array";
        let mut array = ArrayBuilder::new(
            vec![8, 8], // array shape
            vec![4, 4],
            DataType::Float32,
            ZARR_NAN_F32,
        )
        .bytes_to_bytes_codecs(vec![
            #[cfg(feature = "gzip")]
            Arc::new(codec::GzipCodec::new(5).unwrap()),
        ])
        .build(store.into(), array_path)
        .unwrap();

        array.set_shape(vec![16, 16]).unwrap();
        array
            .attributes_mut()
            .insert("test".to_string(), "apple".into());

        assert_eq!(array.shape(), &[16, 16]);
        assert_eq!(
            array.attributes().get_key_value("test"),
            Some((
                &"test".to_string(),
                &serde_json::Value::String("apple".to_string())
            ))
        );
    }

    #[test]
    fn array_set_shape_and_chunk_grid() {
        use crate::array::chunk_grid::RectangularChunkGridConfiguration;
        use zarrs_metadata::v3::MetadataV3;

        let store = MemoryStore::new();
        let array_path = "/group/array";
        let mut array = ArrayBuilder::new(
            vec![8, 8], // array shape
            vec![4, 4], // chunk shape
            DataType::UInt8,
            0u8,
        )
        .build(store.into(), array_path)
        .unwrap();

        // Create chunk grid metadata for a rectangular chunk that is an acceptable expansion of the array
        let chunk_grid_metadata = MetadataV3::new_with_configuration(
            "rectangular",
            RectangularChunkGridConfiguration {
                chunk_shape: vec![
                    [4, 4, 6].try_into().unwrap(),          // varying sizes for dimension 0
                    [4, 4, 4, 3, 2, 1].try_into().unwrap(), // varying sizes for dimension 1
                ],
            },
        );

        // Set new array shape and chunk grid
        unsafe {
            // SAFETY: The new chunk grid does not change the shape of existing chunks, so existing data remains valid.
            array
                .set_shape_and_chunk_grid(vec![14, 18], chunk_grid_metadata)
                .unwrap();
        }

        // Verify new state
        assert_eq!(array.shape(), &[14, 18]);

        // Verify chunk shapes for different chunks
        assert_eq!(
            array.chunk_shape(&[0, 0]).unwrap().as_slice(),
            &[
                std::num::NonZeroU64::new(4).unwrap(),
                std::num::NonZeroU64::new(4).unwrap()
            ]
        );
        assert_eq!(
            array.chunk_shape(&[2, 3]).unwrap().as_slice(),
            &[
                std::num::NonZeroU64::new(6).unwrap(),
                std::num::NonZeroU64::new(3).unwrap()
            ]
        );
    }

    #[test]
    fn array_subset_round_trip() {
        let store = Arc::new(MemoryStore::default());
        let array_path = "/array";
        let array = ArrayBuilder::new(
            vec![8, 8], // array shape
            vec![4, 4], // regular chunk shape
            DataType::Float32,
            1f32,
        )
        .bytes_to_bytes_codecs(vec![
            #[cfg(feature = "gzip")]
            Arc::new(codec::GzipCodec::new(5).unwrap()),
        ])
        // .storage_transformers(vec![].into())
        .build(store, array_path)
        .unwrap();

        array
            .store_array_subset_elements::<f32>(
                &ArraySubset::new_with_ranges(&[3..6, 3..6]),
                &[1.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            )
            .unwrap();

        let subset_all = array.subset_all();
        let data_all = array
            .retrieve_array_subset_elements::<f32>(&subset_all)
            .unwrap();
        assert_eq!(
            data_all,
            vec![
                //     (0,0)       |     (0, 1)
                //0  1    2    3   |4    5    6    7
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 0
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 1
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 2
                1.0, 1.0, 1.0, 1.0, 0.2, 0.3, 1.0, 1.0, //_3____________
                1.0, 1.0, 1.0, 0.4, 0.5, 0.6, 1.0, 1.0, // 4
                1.0, 1.0, 1.0, 0.7, 0.8, 0.9, 1.0, 1.0, // 5 (1, 1)
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 6
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // 7
            ]
        );
        assert!(array
            .retrieve_chunk_elements_if_exists::<f32>(&[0; 2])
            .unwrap()
            .is_none());
        #[cfg(feature = "ndarray")]
        assert!(array
            .retrieve_chunk_ndarray_if_exists::<f32>(&[0; 2])
            .unwrap()
            .is_none());
    }

    #[allow(dead_code)]
    fn array_v2_to_v3(path_in: &str, path_out: &str) {
        let store = Arc::new(FilesystemStore::new(path_in).unwrap());
        let array_in = Array::open(store, "/").unwrap();

        println!("{array_in:?}");

        let subset_all = ArraySubset::new_with_shape(array_in.shape().to_vec());
        let elements = array_in
            .retrieve_array_subset_elements::<f32>(&subset_all)
            .unwrap();

        assert_eq!(
            &elements,
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
                10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, //
                20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, //
                30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, //
                40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, //
                50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, //
                60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, //
                70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, //
                80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, //
                90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, //
            ],
        );

        let store = Arc::new(FilesystemStore::new(path_out).unwrap());
        let array_out = Array::new_with_metadata(store, "/", array_in.metadata().clone()).unwrap();
        array_out
            .store_array_subset_elements::<f32>(&subset_all, &elements)
            .unwrap();

        // Store V2 and V3 metadata
        for version in [MetadataConvertVersion::Default, MetadataConvertVersion::V3] {
            array_out
                .store_metadata_opt(
                    &ArrayMetadataOptions::default()
                        .with_metadata_convert_version(version)
                        .with_include_zarrs_metadata(false),
                )
                .unwrap();
        }
    }

    #[test]
    fn array_v2_none_c() {
        array_v2_to_v3(
            "tests/data/v2/array_none_C.zarr",
            "tests/data/v3/array_none.zarr",
        )
    }

    #[cfg(feature = "transpose")]
    #[test]
    fn array_v2_none_f() {
        array_v2_to_v3(
            "tests/data/v2/array_none_F.zarr",
            "tests/data/v3/array_none_transpose.zarr",
        )
    }

    #[cfg(feature = "blosc")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v2_blosc_c() {
        array_v2_to_v3(
            "tests/data/v2/array_blosc_C.zarr",
            "tests/data/v3/array_blosc.zarr",
        )
    }

    #[cfg(feature = "blosc")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v2_blosc_f() {
        array_v2_to_v3(
            "tests/data/v2/array_blosc_F.zarr",
            "tests/data/v3/array_blosc_transpose.zarr",
        )
    }

    #[cfg(feature = "gzip")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v2_gzip_c() {
        array_v2_to_v3(
            "tests/data/v2/array_gzip_C.zarr",
            "tests/data/v3/array_gzip.zarr",
        )
    }

    #[cfg(feature = "bz2")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v2_bz2_c() {
        array_v2_to_v3(
            "tests/data/v2/array_bz2_C.zarr",
            "tests/data/v3/array_bz2.zarr",
        )
    }

    #[cfg(feature = "zfp")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v2_zfpy_c() {
        array_v2_to_v3(
            "tests/data/v2/array_zfpy_C.zarr",
            "tests/data/v3/array_zfpy.zarr",
        )
    }

    #[cfg(feature = "zstd")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v2_zstd_c() {
        array_v2_to_v3(
            "tests/data/v2/array_zstd_C.zarr",
            "tests/data/v3/array_zstd.zarr",
        )
    }

    #[cfg(feature = "pcodec")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v2_pcodec_c() {
        array_v2_to_v3(
            "tests/data/v2/array_pcodec_C.zarr",
            "tests/data/v3/array_pcodec.zarr",
        )
    }

    #[allow(dead_code)]
    fn array_v3_numcodecs(path_in: &str) {
        let store = Arc::new(FilesystemStore::new(path_in).unwrap());
        let array_in = Array::open(store, "/").unwrap();

        println!(
            "{:?}",
            array_in.metadata_opt(
                &ArrayMetadataOptions::default()
                    .with_metadata_convert_version(MetadataConvertVersion::V3)
            )
        );

        println!("{array_in:?}");

        let subset_all = ArraySubset::new_with_shape(array_in.shape().to_vec());
        let elements = array_in
            .retrieve_array_subset_elements::<f32>(&subset_all)
            .unwrap();

        assert_eq!(
            &elements,
            &[
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, //
                10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, //
                20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, //
                30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, //
                40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, //
                50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, //
                60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, //
                70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, //
                80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, //
                90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, //
            ],
        );
    }

    #[test]
    fn array_v3_none() {
        array_v3_numcodecs("tests/data/v3_zarr_python/array_none.zarr")
    }

    #[cfg(feature = "blosc")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v3_blosc() {
        array_v3_numcodecs("tests/data/v3_zarr_python/array_blosc.zarr")
    }

    #[cfg(feature = "bz2")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v3_bz2() {
        array_v3_numcodecs("tests/data/v3_zarr_python/array_bz2.zarr")
    }

    #[cfg(feature = "fletcher32")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v3_fletcher32() {
        array_v3_numcodecs("tests/data/v3_zarr_python/array_fletcher32.zarr")
    }

    #[cfg(feature = "adler32")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v3_adler32() {
        array_v3_numcodecs("tests/data/v3_zarr_python/array_adler32.zarr")
    }

    #[cfg(feature = "zlib")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v3_zlib() {
        array_v3_numcodecs("tests/data/v3_zarr_python/array_zlib.zarr")
    }

    #[cfg(feature = "gzip")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v3_gzip() {
        array_v3_numcodecs("tests/data/v3_zarr_python/array_gzip.zarr")
    }

    #[cfg(feature = "pcodec")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v3_pcodec() {
        array_v3_numcodecs("tests/data/v3_zarr_python/array_pcodec.zarr")
    }

    #[cfg(feature = "zfp")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v3_zfpy() {
        array_v3_numcodecs("tests/data/v3_zarr_python/array_zfpy.zarr")
    }

    #[cfg(feature = "zstd")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn array_v3_zstd() {
        array_v3_numcodecs("tests/data/v3_zarr_python/array_zstd.zarr")
    }

    // fn array_subset_locking(locks: StoreLocks, expect_equal: bool) {
    //     let store = Arc::new(MemoryStore::new_with_locks(locks));

    //     let array_path = "/array";
    //     let array = ArrayBuilder::new(
    //         vec![100, 4],
    //         DataType::UInt8,
    //         vec![10, 2].try_into().unwrap(),
    //         0u8,
    //     )
    //     .build(store, array_path)
    //     .unwrap();

    //     let mut any_not_equal = false;
    //     for j in 1..10 {
    //         (0..100).into_par_iter().for_each(|i| {
    //             let subset = ArraySubset::new_with_ranges(&[i..i + 1, 0..4]);
    //             array.store_array_subset(&subset, vec![j; 4]).unwrap();
    //         });
    //         let subset_all = array.subset_all();
    //         let data_all = array.retrieve_array_subset(&subset_all).unwrap();
    //         let all_equal = data_all.iter().all_equal_value() == Ok(&j);
    //         if expect_equal {
    //             assert!(all_equal);
    //         } else {
    //             any_not_equal |= !all_equal;
    //         }
    //     }
    //     if !expect_equal {
    //         assert!(any_not_equal);
    //     }
    // }

    // #[test]
    // #[cfg_attr(miri, ignore)]
    // fn array_subset_locking_default() {
    //     array_subset_locking(Arc::new(DefaultStoreLocks::default()), true);
    // }

    // // Due to the nature of this test, it can fail sometimes. It was used for development but is now disabled.
    // #[test]
    // fn array_subset_locking_disabled() {
    //     array_subset_locking(
    //         Arc::new(crate::storage::store_lock::DisabledStoreLocks::default()),
    //         false,
    //     );
    // }

    #[test]
    fn array_additional_fields() {
        let store = Arc::new(MemoryStore::new());
        let array_path = "/group/array";

        for must_understand in [true, false] {
            let additional_field = serde_json::Map::new();
            let additional_field = AdditionalFieldV3::new(additional_field, must_understand);
            let mut additional_fields = AdditionalFieldsV3::new();
            additional_fields.insert("key".to_string(), additional_field);

            // Permit array creation with manually added additional fields
            let array = ArrayBuilder::new(
                vec![8, 8], // array shape
                vec![4, 4],
                DataType::Float32,
                ZARR_NAN_F32,
            )
            .bytes_to_bytes_codecs(vec![
                #[cfg(feature = "gzip")]
                Arc::new(codec::GzipCodec::new(5).unwrap()),
            ])
            .additional_fields(additional_fields)
            .build(store.clone(), array_path)
            .unwrap();
            array.store_metadata().unwrap();

            let array = Array::open(store.clone(), array_path);
            if must_understand {
                // Disallow array opening with unknown `"must_understand": true` additional fields
                assert!(array.is_err());
            } else {
                assert!(array.is_ok());
            }
        }
    }
}
