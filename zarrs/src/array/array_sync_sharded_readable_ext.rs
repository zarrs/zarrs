use std::{collections::HashMap, sync::Arc};

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon_iter_concurrent_limit::iter_concurrent_limit;
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_metadata::ConfigurationSerialize;
use zarrs_metadata_ext::codec::sharding::ShardingCodecConfiguration;
use zarrs_storage::byte_range::ByteRange;
use zarrs_storage::StorageHandle;

use super::array_bytes::merge_chunks_vlen;
use super::codec::array_to_bytes::sharding::ShardingPartialDecoder;
use super::codec::{CodecError, ShardingCodec};
use super::element::ElementOwned;
use super::{
    codec::CodecOptions, concurrency::concurrency_chunks_and_codec, Array, ArrayError,
    ArrayShardedExt, ChunkGrid,
};
use super::{ArrayBytes, ArrayBytesFixedDisjointView, ArraySize, DataTypeSize};
use crate::array::codec::StoragePartialDecoder;
use crate::storage::ReadableStorageTraits;
use crate::{array::codec::ArrayPartialDecoderTraits, array_subset::ArraySubset};

// TODO: Remove with trait upcasting
#[derive(Clone)]
enum MaybeShardingPartialDecoder {
    Sharding(Arc<ShardingPartialDecoder>),
    Other(Arc<dyn ArrayPartialDecoderTraits>),
}

impl MaybeShardingPartialDecoder {
    fn partial_decode(
        &self,
        indexer: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        match self {
            Self::Sharding(partial_decoder) => partial_decoder.partial_decode(indexer, options),
            Self::Other(partial_decoder) => partial_decoder.partial_decode(indexer, options),
        }
    }
}

type PartialDecoderHashMap = HashMap<Vec<u64>, MaybeShardingPartialDecoder>;

/// A cache used for methods in the [`ArrayShardedReadableExt`] trait.
pub struct ArrayShardedReadableExtCache {
    array_is_sharded: bool,
    array_is_exclusively_sharded: bool,
    inner_chunk_grid: ChunkGrid,
    cache: Arc<std::sync::Mutex<PartialDecoderHashMap>>,
}

impl ArrayShardedReadableExtCache {
    /// Create a new cache for an array.
    #[must_use]
    pub fn new<TStorage: ?Sized + ReadableStorageTraits>(array: &Array<TStorage>) -> Self {
        let inner_chunk_grid = array.inner_chunk_grid();
        Self {
            array_is_sharded: array.is_sharded(),
            array_is_exclusively_sharded: array.is_exclusively_sharded(),
            inner_chunk_grid,
            cache: Arc::new(std::sync::Mutex::new(HashMap::default())),
        }
    }

    /// Returns true if the array is sharded.
    ///
    /// This is cheaper than calling [`ArrayShardedExt::is_sharded`] repeatedly.
    #[must_use]
    pub fn array_is_sharded(&self) -> bool {
        self.array_is_sharded
    }

    /// Returns true if the array is exclusively sharded (no array-to-array or bytes-to-bytes codecs).
    ///
    /// This is cheaper than calling [`ArrayShardedExt::is_exclusively_sharded`] repeatedly.
    #[must_use]
    pub fn array_is_exclusively_sharded(&self) -> bool {
        self.array_is_exclusively_sharded
    }

    fn inner_chunk_grid(&self) -> &ChunkGrid {
        &self.inner_chunk_grid
    }

    /// Return the number of shard indexes cached.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn len(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// Returns true if the cache contains no cached shard indexes.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn is_empty(&self) -> bool {
        self.cache.lock().unwrap().is_empty()
    }

    /// Clear the cache.
    #[allow(clippy::missing_panics_doc)]
    pub fn clear(&self) {
        self.cache.lock().unwrap().clear();
    }

    fn retrieve<TStorage: ?Sized + ReadableStorageTraits + 'static>(
        &self,
        array: &Array<TStorage>,
        shard_indices: &[u64],
    ) -> Result<MaybeShardingPartialDecoder, ArrayError> {
        let mut cache = self.cache.lock().unwrap();
        if let Some(partial_decoder) = cache.get(shard_indices) {
            Ok(partial_decoder.clone())
        } else if self.array_is_exclusively_sharded() {
            // Create the sharding partial decoder directly, without a codec chain
            let storage_handle = Arc::new(StorageHandle::new(array.storage.clone()));
            let storage_transformer = array
                .storage_transformers()
                .create_readable_transformer(storage_handle)?;
            let input_handle = Arc::new(StoragePartialDecoder::new(
                storage_transformer,
                array.chunk_key(shard_indices),
            ));

            // --- Workaround for lack of trait upcasting ---
            let chunk_representation = array.chunk_array_representation(shard_indices)?;
            let sharding_codec_configuration = array
                .codecs()
                .array_to_bytes_codec()
                .configuration()
                .expect("valid sharding metadata");
            let sharding_codec_configuration =
                ShardingCodecConfiguration::try_from_configuration(sharding_codec_configuration)
                    .expect("valid sharding configuration");
            let sharding_codec = Arc::new(
                ShardingCodec::new_with_configuration(&sharding_codec_configuration).expect(
                    "supported sharding codec configuration, already instantiated in array",
                ),
            );
            let partial_decoder =
                MaybeShardingPartialDecoder::Sharding(Arc::new(ShardingPartialDecoder::new(
                    input_handle,
                    chunk_representation,
                    &sharding_codec.chunk_shape,
                    sharding_codec.inner_codecs.clone(),
                    &sharding_codec.index_codecs,
                    sharding_codec.index_location,
                    &CodecOptions::default(),
                )?));
            // // TODO: Trait upcasting
            // let partial_decoder = array
            //     .codecs()
            //     .array_to_bytes_codec()
            //     .clone()
            //     .partial_decoder(
            //         input_handle,
            //         &chunk_representation,
            //         &CodecOptions::default(),
            //     )?;
            cache.insert(shard_indices.to_vec(), partial_decoder.clone());
            Ok(partial_decoder)
        } else {
            let partial_decoder =
                MaybeShardingPartialDecoder::Other(array.partial_decoder(shard_indices)?);
            cache.insert(shard_indices.to_vec(), partial_decoder.clone());
            Ok(partial_decoder)
        }
    }
}

/// An [`Array`] extension trait to efficiently read data (e.g. inner chunks) from arrays using the `sharding_indexed` codec.
///
/// Sharding indexes are cached in a [`ArrayShardedReadableExtCache`] enabling faster retrieval.
// TODO: Add default methods? Or change to options: Option<&CodecOptions>? Should really do this for array (breaking)...
pub trait ArrayShardedReadableExt<TStorage: ?Sized + ReadableStorageTraits + 'static>:
    private::Sealed
{
    /// Retrieve the byte range of an encoded inner chunk.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] on failure, such as if decoding the shard index fails.
    fn inner_chunk_byte_range(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunk_indices: &[u64],
    ) -> Result<Option<ByteRange>, ArrayError>;

    /// Retrieve the encoded bytes of an inner chunk.
    ///
    /// See [`Array::retrieve_encoded_chunk`].
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_encoded_inner_chunk(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunk_indices: &[u64],
    ) -> Result<Option<Vec<u8>>, ArrayError>;

    // TODO: retrieve_encoded_inner_chunks

    /// Read and decode the inner chunk at `chunk_indices` into its bytes.
    ///
    /// See [`Array::retrieve_chunk_opt`].
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_inner_chunk_opt(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError>;

    /// Read and decode the inner chunk at `chunk_indices` into a vector of its elements.
    ///
    /// See [`Array::retrieve_chunk_elements_opt`].
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_inner_chunk_elements_opt<T: ElementOwned>(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    /// Read and decode the chunk at `chunk_indices` into an [`ndarray::ArrayD`].
    ///
    /// See [`Array::retrieve_chunk_ndarray_opt`].
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_inner_chunk_ndarray_opt<T: ElementOwned>(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;

    /// Read and decode the chunks at `chunks` into their bytes.
    ///
    /// See [`Array::retrieve_chunks_opt`].
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_inner_chunks_opt(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError>;

    /// Read and decode the inner chunks at `inner_chunks` into a vector of their elements.
    ///
    /// See [`Array::retrieve_chunks_elements_opt`].
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_inner_chunks_elements_opt<T: ElementOwned>(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    /// Read and decode the inner chunks at `inner_chunks` into an [`ndarray::ArrayD`].
    ///
    /// See [`Array::retrieve_chunks_ndarray_opt`].
    #[cfg(feature = "ndarray")]
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_inner_chunks_ndarray_opt<T: ElementOwned>(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;

    /// Read and decode the `array_subset` of array into its bytes.
    ///
    /// See [`Array::retrieve_array_subset_opt`].
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_array_subset_sharded_opt(
        &self,
        cache: &ArrayShardedReadableExtCache,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError>;

    /// Read and decode the `array_subset` of array into a vector of its elements.
    ///
    /// See [`Array::retrieve_array_subset_elements_opt`].
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_array_subset_elements_sharded_opt<T: ElementOwned>(
        &self,
        cache: &ArrayShardedReadableExtCache,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    /// Read and decode the `array_subset` of array into an [`ndarray::ArrayD`].
    ///
    /// See [`Array::retrieve_array_subset_ndarray_opt`].
    #[allow(clippy::missing_errors_doc)]
    fn retrieve_array_subset_ndarray_sharded_opt<T: ElementOwned>(
        &self,
        cache: &ArrayShardedReadableExtCache,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;
}

fn inner_chunk_shard_index_and_subset<TStorage: ?Sized + ReadableStorageTraits + 'static>(
    array: &Array<TStorage>,
    inner_chunk_grid: &ChunkGrid,
    inner_chunk_indices: &[u64],
) -> Result<(Vec<u64>, ArraySubset), ArrayError> {
    // TODO: Can this logic be simplified?
    let array_subset = inner_chunk_grid
        .subset(inner_chunk_indices)?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(inner_chunk_indices.to_vec()))?;
    let shards = array
        .chunks_in_array_subset(&array_subset)?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(inner_chunk_indices.to_vec()))?;
    if shards.num_elements() != 1 {
        // This should not happen, but it is checked just in case.
        return Err(ArrayError::InvalidChunkGridIndicesError(
            inner_chunk_indices.to_vec(),
        ));
    }
    let shard_indices = shards.start();
    let shard_origin = array.chunk_origin(shard_indices)?;
    let shard_subset = array_subset.relative_to(&shard_origin)?;
    Ok((shard_indices.to_vec(), shard_subset))
}

fn inner_chunk_shard_index_and_chunk_index<TStorage: ?Sized + ReadableStorageTraits + 'static>(
    array: &Array<TStorage>,
    inner_chunk_grid: &ChunkGrid,
    inner_chunk_indices: &[u64],
) -> Result<(Vec<u64>, Vec<u64>), ArrayError> {
    // TODO: Simplify this?
    let (shard_indices, shard_subset) =
        inner_chunk_shard_index_and_subset(array, inner_chunk_grid, inner_chunk_indices)?;
    let effective_inner_chunk_shape = array
        .effective_inner_chunk_shape()
        .expect("array is sharded");
    let chunk_indices: Vec<u64> = shard_subset
        .start()
        .iter()
        .zip(effective_inner_chunk_shape.as_slice())
        .map(|(o, s)| o / s.get())
        .collect();
    Ok((shard_indices, chunk_indices))
}

impl<TStorage: ?Sized + ReadableStorageTraits + 'static> ArrayShardedReadableExt<TStorage>
    for Array<TStorage>
{
    fn inner_chunk_byte_range(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunk_indices: &[u64],
    ) -> Result<Option<ByteRange>, ArrayError> {
        if cache.array_is_exclusively_sharded() {
            let (shard_indices, chunk_indices) = inner_chunk_shard_index_and_chunk_index(
                self,
                cache.inner_chunk_grid(),
                inner_chunk_indices,
            )?;
            let partial_decoder = cache.retrieve(self, &shard_indices)?;
            let MaybeShardingPartialDecoder::Sharding(partial_decoder) = partial_decoder else {
                unreachable!("exlusively sharded")
            };
            // TODO: trait upcasting
            // let partial_decoder: Arc<dyn Any + Send + Sync> = partial_decoder.clone();
            // let partial_decoder = partial_decoder
            //     .downcast::<ShardingPartialDecoder>()
            //     .expect("array is exclusively sharded");

            Ok(partial_decoder.inner_chunk_byte_range(&chunk_indices)?)
        } else {
            Err(ArrayError::UnsupportedMethod(
                "the array is not exclusively sharded".to_string(),
            ))
        }
    }

    fn retrieve_encoded_inner_chunk(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunk_indices: &[u64],
    ) -> Result<Option<Vec<u8>>, ArrayError> {
        if cache.array_is_exclusively_sharded() {
            let (shard_indices, chunk_indices) = inner_chunk_shard_index_and_chunk_index(
                self,
                cache.inner_chunk_grid(),
                inner_chunk_indices,
            )?;
            let partial_decoder = cache.retrieve(self, &shard_indices)?;
            let MaybeShardingPartialDecoder::Sharding(partial_decoder) = partial_decoder else {
                unreachable!("exlusively sharded")
            };
            // TODO: trait upcasting
            // let partial_decoder: Arc<dyn Any + Send + Sync> = partial_decoder.clone();
            // let partial_decoder = partial_decoder
            //     .downcast::<ShardingPartialDecoder>()
            //     .expect("array is exclusively sharded");

            Ok(partial_decoder
                .retrieve_inner_chunk_encoded(&chunk_indices)?
                .map(Vec::from))
        } else {
            Err(ArrayError::UnsupportedMethod(
                "the array is not exclusively sharded".to_string(),
            ))
        }
    }

    fn retrieve_inner_chunk_opt(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        if cache.array_is_sharded() {
            let (shard_indices, shard_subset) = inner_chunk_shard_index_and_subset(
                self,
                cache.inner_chunk_grid(),
                inner_chunk_indices,
            )?;
            let partial_decoder = cache.retrieve(self, &shard_indices)?;
            let bytes = partial_decoder
                .partial_decode(&shard_subset, options)?
                .into_owned();
            Ok(bytes)
        } else {
            self.retrieve_chunk_opt(inner_chunk_indices, options)
        }
    }

    fn retrieve_inner_chunk_elements_opt<T: ElementOwned>(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            self.retrieve_inner_chunk_opt(cache, inner_chunk_indices, options)?,
        )
    }

    #[cfg(feature = "ndarray")]
    fn retrieve_inner_chunk_ndarray_opt<T: ElementOwned>(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        if let Some(inner_chunk_shape) = self.inner_chunk_shape() {
            super::elements_to_ndarray(
                &inner_chunk_shape.to_array_shape(),
                self.retrieve_inner_chunk_elements_opt::<T>(cache, inner_chunk_indices, options)?,
            )
        } else {
            self.retrieve_chunk_ndarray_opt(inner_chunk_indices, options)
        }
    }

    fn retrieve_inner_chunks_opt(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        if cache.array_is_sharded() {
            let inner_chunk_grid = cache.inner_chunk_grid();
            let array_subset = inner_chunk_grid
                .chunks_subset(inner_chunks)?
                .ok_or_else(|| {
                    ArrayError::InvalidArraySubset(
                        inner_chunks.clone(),
                        inner_chunk_grid.grid_shape().clone(),
                    )
                })?;
            self.retrieve_array_subset_sharded_opt(cache, &array_subset, options)
        } else {
            self.retrieve_chunks_opt(inner_chunks, options)
        }
    }

    fn retrieve_inner_chunks_elements_opt<T: ElementOwned>(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            self.retrieve_inner_chunks_opt(cache, inner_chunks, options)?,
        )
    }

    #[cfg(feature = "ndarray")]
    fn retrieve_inner_chunks_ndarray_opt<T: ElementOwned>(
        &self,
        cache: &ArrayShardedReadableExtCache,
        inner_chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let inner_chunk_grid = cache.inner_chunk_grid();
        let array_subset = inner_chunk_grid
            .chunks_subset(inner_chunks)?
            .ok_or_else(|| {
                ArrayError::InvalidArraySubset(
                    inner_chunks.clone(),
                    inner_chunk_grid.grid_shape().clone(),
                )
            })?;
        let elements =
            self.retrieve_inner_chunks_elements_opt::<T>(cache, inner_chunks, options)?;
        super::elements_to_ndarray(array_subset.shape(), elements)
    }

    #[allow(clippy::too_many_lines)]
    fn retrieve_array_subset_sharded_opt(
        &self,
        cache: &ArrayShardedReadableExtCache,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        if cache.array_is_sharded() {
            // Find the shards intersecting this array subset
            let shards = self.chunks_in_array_subset(array_subset)?;
            let Some(shards) = shards else {
                return Err(ArrayError::InvalidArraySubset(
                    array_subset.clone(),
                    self.shape().to_vec(),
                ));
            };

            // Retrieve chunk bytes
            let num_shards = shards.num_elements_usize();
            if num_shards == 0 {
                let array_size =
                    ArraySize::new(self.data_type().size(), array_subset.num_elements());
                Ok(ArrayBytes::new_fill_value(array_size, self.fill_value()))
            } else {
                // Calculate chunk/codec concurrency
                let chunk_representation =
                    self.chunk_array_representation(&vec![0; self.dimensionality()])?;
                let codec_concurrency =
                    self.recommended_codec_concurrency(&chunk_representation)?;
                let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                    options.concurrent_target(),
                    num_shards,
                    options,
                    &codec_concurrency,
                );

                match self.data_type().size() {
                    DataTypeSize::Variable => {
                        let retrieve_inner_chunk = |shard_indices: Vec<u64>| -> Result<
                            (ArrayBytes<'_>, ArraySubset),
                            ArrayError,
                        > {
                            let shard_subset = self.chunk_subset(&shard_indices)?;
                            let shard_subset_overlap = shard_subset.overlap(array_subset)?;
                            let bytes = cache
                                .retrieve(self, &shard_indices)?
                                .partial_decode(
                                    &shard_subset_overlap.relative_to(shard_subset.start())?,
                                    &options,
                                )?
                                .into_owned();
                            Ok((
                                bytes,
                                shard_subset_overlap.relative_to(array_subset.start())?,
                            ))
                        };

                        let indices = shards.indices();
                        let chunk_bytes_and_subsets = iter_concurrent_limit!(
                            chunk_concurrent_limit,
                            indices,
                            map,
                            retrieve_inner_chunk
                        )
                        .collect::<Result<Vec<_>, _>>()?;

                        Ok(merge_chunks_vlen(
                            chunk_bytes_and_subsets,
                            array_subset.shape(),
                        )?)
                    }
                    DataTypeSize::Fixed(data_type_size) => {
                        let size_output = array_subset.num_elements_usize() * data_type_size;
                        if size_output == 0 {
                            return Ok(ArrayBytes::new_flen(vec![]));
                        }
                        let mut output = Vec::with_capacity(size_output);
                        {
                            let output =
                                UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut output);
                            let retrieve_shard_into_slice = |shard_indices: Vec<u64>| {
                                let shard_subset = self.chunk_subset(&shard_indices)?;
                                let shard_subset_overlap = shard_subset.overlap(array_subset)?;
                                // let shard_subset_bytes = self.retrieve_chunk_subset_opt(
                                //     &shard_indices,
                                //     &shard_subset_overlap.relative_to(shard_subset.start())?,
                                //     &options,
                                // )?;
                                let bytes = cache
                                    .retrieve(self, &shard_indices)?
                                    .partial_decode(
                                        &shard_subset_overlap.relative_to(shard_subset.start())?,
                                        &options,
                                    )?
                                    .into_owned();
                                let mut output_view = unsafe {
                                    // SAFETY: chunks represent disjoint array subsets
                                    ArrayBytesFixedDisjointView::new(
                                        output,
                                        data_type_size,
                                        array_subset.shape(),
                                        shard_subset_overlap.relative_to(array_subset.start())?,
                                    )?
                                };
                                output_view
                                    .copy_from_slice(&bytes.into_fixed()?)
                                    .map_err(CodecError::from)?;
                                Ok::<_, ArrayError>(())
                            };
                            let indices = shards.indices();
                            iter_concurrent_limit!(
                                chunk_concurrent_limit,
                                indices,
                                try_for_each,
                                retrieve_shard_into_slice
                            )?;
                        }
                        unsafe { output.set_len(size_output) };
                        Ok(ArrayBytes::from(output))
                    }
                }
            }
        } else {
            self.retrieve_array_subset_opt(array_subset, options)
        }
    }

    fn retrieve_array_subset_elements_sharded_opt<T: ElementOwned>(
        &self,
        cache: &ArrayShardedReadableExtCache,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        T::from_array_bytes(
            self.data_type(),
            self.retrieve_array_subset_sharded_opt(cache, array_subset, options)?,
        )
    }

    #[cfg(feature = "ndarray")]
    fn retrieve_array_subset_ndarray_sharded_opt<T: ElementOwned>(
        &self,
        cache: &ArrayShardedReadableExtCache,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        let elements =
            self.retrieve_array_subset_elements_sharded_opt::<T>(cache, array_subset, options)?;
        super::elements_to_ndarray(array_subset.shape(), elements)
    }
}

mod private {
    use super::{Array, ReadableStorageTraits};

    pub trait Sealed {}

    impl<TStorage: ?Sized + ReadableStorageTraits + 'static> Sealed for Array<TStorage> {}
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use zarrs_metadata_ext::codec::transpose::TransposeOrder;

    use crate::{
        array::{
            codec::{array_to_bytes::sharding::ShardingCodecBuilder, TransposeCodec},
            ArrayBuilder, DataType,
        },
        array_subset::ArraySubset,
        storage::{
            storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter,
            store::MemoryStore,
        },
    };

    use super::*;

    fn array_sharded_ext_impl(sharded: bool) -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(MemoryStore::default());
        let array_path = "/array";
        let mut builder = ArrayBuilder::new(
            vec![8, 8], // array shape
            vec![4, 4], // regular chunk shape
            DataType::UInt16,
            0u16,
        );
        if sharded {
            builder.array_to_bytes_codec(Arc::new(
                ShardingCodecBuilder::new(vec![2, 2].try_into()?)
                    .bytes_to_bytes_codecs(vec![
                        #[cfg(feature = "gzip")]
                        Arc::new(crate::array::codec::GzipCodec::new(5)?),
                    ])
                    .build(),
            ));
        }
        let array = builder.build(store, array_path)?;

        let data: Vec<u16> = (0..array.shape().into_iter().product())
            .map(|i| i as u16)
            .collect();

        array.store_array_subset_elements(&array.subset_all(), &data)?;

        let cache = ArrayShardedReadableExtCache::new(&array);
        assert_eq!(array.is_sharded(), sharded);
        let inner_chunk_grid = array.inner_chunk_grid();
        if sharded {
            assert_eq!(array.inner_chunk_shape(), Some(vec![2, 2].try_into()?));
            assert_eq!(inner_chunk_grid.grid_shape(), &[4, 4]);

            let compare =
                array.retrieve_array_subset_elements::<u16>(&ArraySubset::new_with_ranges(&[
                    4..6,
                    6..8,
                ]))?;
            let test = array.retrieve_inner_chunk_elements_opt::<u16>(
                &cache,
                &[2, 3],
                &CodecOptions::default(),
            )?;
            assert_eq!(compare, test);
            assert_eq!(cache.len(), 1);

            #[cfg(feature = "ndarray")]
            {
                let compare = array.retrieve_array_subset_ndarray::<u16>(
                    &ArraySubset::new_with_ranges(&[4..6, 6..8]),
                )?;
                let test = array.retrieve_inner_chunk_ndarray_opt::<u16>(
                    &cache,
                    &[2, 3],
                    &CodecOptions::default(),
                )?;
                assert_eq!(compare, test);
            }

            cache.clear();
            assert_eq!(cache.len(), 0);

            let subset = ArraySubset::new_with_ranges(&[3..7, 3..7]);
            let compare = array.retrieve_array_subset_elements::<u16>(&subset)?;
            let test = array.retrieve_array_subset_elements_sharded_opt::<u16>(
                &cache,
                &subset,
                &CodecOptions::default(),
            )?;
            assert_eq!(compare, test);
            assert_eq!(cache.len(), 4);

            #[cfg(feature = "ndarray")]
            {
                let subset = ArraySubset::new_with_ranges(&[3..7, 3..7]);
                let compare = array.retrieve_array_subset_ndarray::<u16>(&subset)?;
                let test = array.retrieve_array_subset_ndarray_sharded_opt::<u16>(
                    &cache,
                    &subset,
                    &CodecOptions::default(),
                )?;
                assert_eq!(compare, test);
            }

            let subset = ArraySubset::new_with_ranges(&[2..6, 2..6]);
            let inner_chunks = ArraySubset::new_with_ranges(&[1..3, 1..3]);
            let compare = array.retrieve_array_subset_elements::<u16>(&subset)?;
            let test = array.retrieve_inner_chunks_elements_opt::<u16>(
                &cache,
                &inner_chunks,
                &CodecOptions::default(),
            )?;
            assert_eq!(compare, test);
            assert_eq!(cache.len(), 4);

            #[cfg(feature = "ndarray")]
            {
                let subset = ArraySubset::new_with_ranges(&[2..6, 2..6]);
                let inner_chunks = ArraySubset::new_with_ranges(&[1..3, 1..3]);
                let compare = array.retrieve_array_subset_ndarray::<u16>(&subset)?;
                let test = array.retrieve_inner_chunks_ndarray_opt::<u16>(
                    &cache,
                    &inner_chunks,
                    &CodecOptions::default(),
                )?;
                assert_eq!(compare, test);
                assert_eq!(cache.len(), 4);
            }

            let encoded_inner_chunk = array
                .retrieve_encoded_inner_chunk(&cache, &[0, 0])?
                .unwrap();
            assert_eq!(
                array
                    .inner_chunk_byte_range(&cache, &[0, 0])?
                    .unwrap()
                    .length(u64::MAX),
                encoded_inner_chunk.len() as u64
            );
            // assert_eq!(
            //     u16::from_array_bytes(array.data_type(), encoded_inner_chunk.into())?,
            //     array.retrieve_chunk_elements::<u16>(&[0, 0])?
            // );
        } else {
            assert_eq!(array.inner_chunk_shape(), None);
            assert_eq!(inner_chunk_grid.grid_shape(), &[2, 2]);

            let compare =
                array.retrieve_array_subset_elements::<u16>(&ArraySubset::new_with_ranges(&[
                    4..8,
                    4..8,
                ]))?;
            let test = array.retrieve_inner_chunk_elements_opt::<u16>(
                &cache,
                &[1, 1],
                &CodecOptions::default(),
            )?;
            assert_eq!(compare, test);

            let subset = ArraySubset::new_with_ranges(&[3..7, 3..7]);
            let compare = array.retrieve_array_subset_elements::<u16>(&subset)?;
            let test = array.retrieve_array_subset_elements_sharded_opt::<u16>(
                &cache,
                &subset,
                &CodecOptions::default(),
            )?;
            assert_eq!(compare, test);
            assert!(cache.is_empty());

            assert!(array.retrieve_encoded_inner_chunk(&cache, &[0, 0]).is_err());
            assert!(array.inner_chunk_byte_range(&cache, &[0, 0]).is_err());
        }

        Ok(())
    }

    #[test]
    fn array_sharded_ext_sharded() -> Result<(), Box<dyn std::error::Error>> {
        array_sharded_ext_impl(true)
    }

    #[test]
    fn array_sharded_ext_unsharded() -> Result<(), Box<dyn std::error::Error>> {
        array_sharded_ext_impl(false)
    }

    fn array_sharded_ext_impl_transpose(
        valid_inner_chunk_shape: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(MemoryStore::default());
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(store));

        let array_path = "/array";
        let mut builder = ArrayBuilder::new(
            vec![16, 16, 9], // array shape
            vec![8, 4, 3],   // regular chunk shape
            DataType::UInt32,
            0u32,
        );
        builder.array_to_array_codecs(vec![Arc::new(TransposeCodec::new(TransposeOrder::new(
            &[1, 0, 2],
        )?))]);
        builder.array_to_bytes_codec(Arc::new(
            ShardingCodecBuilder::new(
                vec![1, if valid_inner_chunk_shape { 2 } else { 3 }, 3].try_into()?,
            )
            .bytes_to_bytes_codecs(vec![
                #[cfg(feature = "gzip")]
                Arc::new(crate::array::codec::GzipCodec::new(5)?),
            ])
            .build(),
        ));
        let array = builder.build(store.clone(), array_path)?;

        let inner_chunk_grid = array.inner_chunk_grid();
        if valid_inner_chunk_shape {
            //  Config:
            //  16 x 16 x 9 Array shape
            //   8 x  4 x 3 Chunk (shard) shape
            //   1 x  2 x 3 Inner chunk shape
            //      [1,0,2] Transpose order
            //  Calculations:
            //   2 x  4 x 3 Number of shards (chunk grid shape)
            //   4 x  8 x 3 Transposed shard shape
            //   4 x  4 x 1 Inner chunks per (transposed) shard
            //   8 x 16 x 3 Inner grid shape
            //   2 x  1 x 3 Effective inner chunk shape (read granularity)

            assert_eq!(array.chunk_grid_shape(), &[2, 4, 3]);
            assert_eq!(array.inner_chunk_shape(), Some(vec![1, 2, 3].try_into()?));
            assert_eq!(
                array.effective_inner_chunk_shape(),
                Some(vec![2, 1, 3].try_into()?)
            ); // NOTE: transposed
            assert_eq!(inner_chunk_grid.grid_shape(), &[8, 16, 3]);
        } else {
            // skip above tests if the inner chunk shape is invalid, below calls fail with
            // CodecError(Other("invalid inner chunk shape [1, 3, 3], it must evenly divide [4, 8, 3]"))
        }

        let data: Vec<u32> = (0..array.shape().into_iter().product())
            .map(|i| i as u32)
            .collect();
        array.store_array_subset_elements(&array.subset_all(), &data)?;

        // Retrieving an inner chunk should be exactly 2 reads: index + chunk
        let inner_chunk_subset = inner_chunk_grid.subset(&[0, 0, 0])?.unwrap();
        let inner_chunk_data = array.retrieve_array_subset_elements::<u32>(&inner_chunk_subset)?;
        assert_eq!(inner_chunk_data, &[0, 1, 2, 144, 145, 146]);
        assert_eq!(store.reads(), 2);

        Ok(())
    }

    #[test]
    fn array_sharded_ext_impl_transpose_valid_inner_chunk_shape() {
        assert!(array_sharded_ext_impl_transpose(true).is_ok())
    }

    #[test]
    fn array_sharded_ext_impl_transpose_invalid_inner_chunk_shape() {
        assert_eq!(
            array_sharded_ext_impl_transpose(false)
                .unwrap_err()
                .to_string(),
            "invalid inner chunk shape [1, 3, 3], it must evenly divide [4, 8, 3]"
        )
    }
}
