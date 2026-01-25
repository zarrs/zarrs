use std::collections::HashMap;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};
use unsafe_cell_slice::UnsafeCellSlice;

use super::codec::ShardingCodec;
use super::codec::array_to_bytes::sharding::AsyncShardingPartialDecoder;
use super::concurrency::concurrency_chunks_and_codec;
use super::element::ElementOwned;
use super::from_array_bytes::FromArrayBytes;
use super::{
    Array, ArrayBytes, ArrayBytesFixedDisjointView, ArrayError, ArrayIndicesTinyVec,
    ArrayShardedExt, ChunkGrid, DataTypeSize,
};
use crate::array::{ArraySubset, ArraySubsetTraits};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, AsyncArrayPartialDecoderTraits, AsyncStoragePartialDecoder,
    CodecError, CodecOptions, merge_chunks_vlen,
};
use zarrs_metadata::ConfigurationSerialize;
use zarrs_metadata_ext::codec::sharding::ShardingCodecConfiguration;
use zarrs_storage::byte_range::ByteRange;
use zarrs_storage::{AsyncReadableStorageTraits, MaybeSend, MaybeSync, StorageHandle};

// TODO: Remove with trait upcasting
#[derive(Clone)]
enum MaybeShardingPartialDecoder {
    Sharding(Arc<AsyncShardingPartialDecoder>),
    Other(Arc<dyn AsyncArrayPartialDecoderTraits>),
}

impl MaybeShardingPartialDecoder {
    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn crate::array::Indexer,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        match self {
            Self::Sharding(partial_decoder) => {
                partial_decoder.partial_decode(indexer, options).await
            }
            Self::Other(partial_decoder) => partial_decoder.partial_decode(indexer, options).await,
        }
    }

    async fn partial_decode_into(
        &self,
        indexer: &dyn crate::array::Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        match self {
            Self::Sharding(partial_decoder) => {
                partial_decoder
                    .partial_decode_into(indexer, output_target, options)
                    .await
            }
            Self::Other(partial_decoder) => {
                partial_decoder
                    .partial_decode_into(indexer, output_target, options)
                    .await
            }
        }
    }
}

type PartialDecoderHashMap = HashMap<Vec<u64>, MaybeShardingPartialDecoder>;

/// A cache used for methods in the [`AsyncArrayShardedReadableExt`] trait.
pub struct AsyncArrayShardedReadableExtCache {
    array_is_sharded: bool,
    array_is_exclusively_sharded: bool,
    subchunk_grid: ChunkGrid,
    cache: Arc<async_lock::Mutex<PartialDecoderHashMap>>,
}

impl AsyncArrayShardedReadableExtCache {
    /// Create a new cache for an array.
    #[must_use]
    pub fn new<TStorage: ?Sized + AsyncReadableStorageTraits>(array: &Array<TStorage>) -> Self {
        let subchunk_grid = array.subchunk_grid();
        Self {
            array_is_sharded: array.is_sharded(),
            array_is_exclusively_sharded: array.is_exclusively_sharded(),
            subchunk_grid,
            cache: Arc::new(async_lock::Mutex::new(HashMap::default())),
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

    fn subchunk_grid(&self) -> &ChunkGrid {
        &self.subchunk_grid
    }

    /// Return the number of shard indexes cached.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub async fn len(&self) -> usize {
        self.cache.lock().await.len()
    }

    /// Returns true if the cache contains no cached shard indexes.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub async fn is_empty(&self) -> bool {
        self.cache.lock().await.is_empty()
    }

    /// Clear the cache.
    #[allow(clippy::missing_panics_doc)]
    pub async fn clear(&self) {
        self.cache.lock().await.clear();
    }

    async fn retrieve<TStorage: ?Sized + AsyncReadableStorageTraits + 'static>(
        &self,
        array: &Array<TStorage>,
        shard_indices: &[u64],
    ) -> Result<MaybeShardingPartialDecoder, ArrayError> {
        let mut cache = self.cache.lock().await;
        if let Some(partial_decoder) = cache.get(shard_indices) {
            Ok(partial_decoder.clone())
        } else if self.array_is_exclusively_sharded() {
            // Create the sharding partial decoder directly, without a codec chain
            let storage_handle = Arc::new(StorageHandle::new(array.storage.clone()));
            let storage_transformer = array
                .storage_transformers()
                .create_async_readable_transformer(storage_handle)
                .await?;
            let input_handle = Arc::new(AsyncStoragePartialDecoder::new(
                storage_transformer,
                array.chunk_key(shard_indices),
            ));

            // --- Workaround for lack of trait upcasting ---
            let chunk_shape = array.chunk_shape(shard_indices)?;
            let sharding_codec_configuration = array
                .codecs()
                .array_to_bytes_codec()
                .configuration_v3(array.metadata_options.codec_metadata_options())
                .expect("valid sharding metadata");
            let sharding_codec_configuration =
                ShardingCodecConfiguration::try_from_configuration(sharding_codec_configuration)
                    .expect("valid sharding configuration");
            let sharding_codec = Arc::new(
                ShardingCodec::new_with_configuration(&sharding_codec_configuration).expect(
                    "supported sharding codec configuration, already instantiated in array",
                ),
            );
            let partial_decoder = MaybeShardingPartialDecoder::Sharding(Arc::new(
                AsyncShardingPartialDecoder::new(
                    input_handle,
                    array.data_type().clone(),
                    array.fill_value().clone(),
                    chunk_shape.clone(),
                    sharding_codec.subchunk_shape.clone(),
                    sharding_codec.inner_codecs.clone(),
                    &sharding_codec.index_codecs,
                    sharding_codec.index_location,
                    &array.codec_options,
                )
                .await?,
            ));
            // // TODO: Trait upcasting
            // let partial_decoder = array
            //     .codecs()
            //     .array_to_bytes_codec()
            //     .clone()
            //     .partial_decoder(
            //         input_handle,
            //         &chunk_representation,
            //         &self.codec_options,
            //     )?;
            cache.insert(shard_indices.to_vec(), partial_decoder.clone());
            Ok(partial_decoder)
        } else {
            let partial_decoder = MaybeShardingPartialDecoder::Other(
                array.async_partial_decoder(shard_indices).await?,
            );
            cache.insert(shard_indices.to_vec(), partial_decoder.clone());
            Ok(partial_decoder)
        }
    }
}

/// An [`Array`] extension trait to efficiently read data (e.g. subchunks) from arrays using the `sharding_indexed` codec.
///s
/// Sharding indexes are cached in a [`AsyncArrayShardedReadableExtCache`] enabling faster retrieval.
// TODO: Add default methods? Or change to options: Option<&CodecOptions>? Should really do this for array (breaking)...
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub trait AsyncArrayShardedReadableExt<TStorage: ?Sized + AsyncReadableStorageTraits + 'static>:
    private::Sealed
{
    /// Retrieve the byte range of an encoded subchunk.
    ///
    /// # Errors
    /// Returns an [`ArrayError`] on failure, such as if decoding the shard index fails.
    async fn async_subchunk_byte_range(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunk_indices: &[u64],
    ) -> Result<Option<ByteRange>, ArrayError>;

    /// Retrieve the encoded bytes of a subchunk.
    ///
    /// See [`Array::retrieve_encoded_chunk`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_encoded_subchunk(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunk_indices: &[u64],
    ) -> Result<Option<Vec<u8>>, ArrayError>;

    // TODO: retrieve_encoded_subchunks

    /// Read and decode the subchunk at `subchunk_indices` into its bytes.
    ///
    /// See [`Array::retrieve_chunk_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_subchunk_opt<T: FromArrayBytes + MaybeSend>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_subchunk_opt::<Vec<T>>() instead"
    )]
    /// Read and decode the subchunk at `subchunk_indices` into a vector of its elements.
    ///
    /// See [`Array::retrieve_chunk_elements_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_subchunk_elements_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_subchunk_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Read and decode the subchunk at `subchunk_indices` into an [`ndarray::ArrayD`].
    ///
    /// See [`Array::retrieve_chunk_ndarray_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_subchunk_ndarray_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;

    /// Read and decode the chunks at `chunks`.
    ///
    /// See [`Array::retrieve_chunks_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_subchunks_opt<T: FromArrayBytes + MaybeSend>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_subchunks_opt::<Vec<T>>() instead"
    )]
    /// Read and decode the subchunks at `subchunks` into a vector of their elements.
    ///
    /// See [`Array::retrieve_chunks_elements_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_subchunks_elements_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_subchunks_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Read and decode the subchunks at `subchunks` into an [`ndarray::ArrayD`].
    ///
    /// See [`Array::retrieve_chunks_ndarray_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_subchunks_ndarray_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;

    /// Read and decode the `array_subset` of array.
    ///
    /// See [`Array::retrieve_array_subset_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_array_subset_sharded_opt<T: FromArrayBytes + MaybeSend>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_array_subset_sharded_opt::<Vec<T>>() instead"
    )]
    /// Read and decode the `array_subset` of array into a vector of its elements.
    ///
    /// See [`Array::retrieve_array_subset_elements_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_array_subset_elements_sharded_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError>;

    #[cfg(feature = "ndarray")]
    #[deprecated(
        since = "0.23.0",
        note = "Use async_retrieve_array_subset_sharded_opt::<ndarray::ArrayD<T>>() instead"
    )]
    /// Read and decode the `array_subset` of array into an [`ndarray::ArrayD`].
    ///
    /// See [`Array::retrieve_array_subset_ndarray_opt`].
    #[allow(clippy::missing_errors_doc)]
    async fn async_retrieve_array_subset_ndarray_sharded_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError>;
}

fn subchunk_shard_index_and_subset<TStorage: ?Sized + AsyncReadableStorageTraits + 'static>(
    array: &Array<TStorage>,
    cache: &AsyncArrayShardedReadableExtCache,
    subchunk_indices: &[u64],
) -> Result<(Vec<u64>, ArraySubset), ArrayError> {
    // TODO: Can this logic be simplified?
    let array_subset = cache
        .subchunk_grid()
        .subset(subchunk_indices)?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?;
    let shards = array
        .chunks_in_array_subset(&array_subset)?
        .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(subchunk_indices.to_vec()))?;
    if shards.num_elements() != 1 {
        // This should not happen, but it is checked just in case.
        return Err(ArrayError::InvalidChunkGridIndicesError(
            subchunk_indices.to_vec(),
        ));
    }
    let shard_indices = shards.start();
    let shard_origin = array.chunk_origin(shard_indices)?;
    let shard_subset = array_subset.relative_to(&shard_origin)?;
    Ok((shard_indices.to_vec(), shard_subset))
}

fn subchunk_shard_index_and_chunk_index<TStorage: ?Sized + AsyncReadableStorageTraits + 'static>(
    array: &Array<TStorage>,
    cache: &AsyncArrayShardedReadableExtCache,
    subchunk_indices: &[u64],
) -> Result<(Vec<u64>, Vec<u64>), ArrayError> {
    // TODO: Simplify this?
    let (shard_indices, shard_subset) =
        subchunk_shard_index_and_subset(array, cache, subchunk_indices)?;
    let effective_subchunk_shape = array.effective_subchunk_shape().expect("array is sharded");
    let chunk_indices: Vec<u64> = shard_subset
        .start()
        .iter()
        .zip(effective_subchunk_shape.as_slice())
        .map(|(o, s)| o / s.get())
        .collect();
    Ok((shard_indices, chunk_indices))
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> AsyncArrayShardedReadableExt<TStorage>
    for Array<TStorage>
{
    async fn async_subchunk_byte_range(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunk_indices: &[u64],
    ) -> Result<Option<ByteRange>, ArrayError> {
        if cache.array_is_exclusively_sharded() {
            let (shard_indices, chunk_indices) =
                subchunk_shard_index_and_chunk_index(self, cache, subchunk_indices)?;
            let partial_decoder = cache.retrieve(self, &shard_indices).await?;
            let MaybeShardingPartialDecoder::Sharding(partial_decoder) = partial_decoder else {
                unreachable!("exlusively sharded")
            };
            // TODO: trait upcasting
            // let partial_decoder: Arc<dyn Any + MaybeSend + MaybeSync> = partial_decoder.clone();
            // let partial_decoder = partial_decoder
            //     .downcast::<AsyncShardingPartialDecoder>()
            //     .expect("array is exclusively sharded");

            Ok(partial_decoder.subchunk_byte_range(&chunk_indices)?)
        } else {
            Err(ArrayError::UnsupportedMethod(
                "the array is not exclusively sharded".to_string(),
            ))
        }
    }

    async fn async_retrieve_encoded_subchunk(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunk_indices: &[u64],
    ) -> Result<Option<Vec<u8>>, ArrayError> {
        if cache.array_is_exclusively_sharded() {
            let (shard_indices, chunk_indices) =
                subchunk_shard_index_and_chunk_index(self, cache, subchunk_indices)?;
            let partial_decoder = cache.retrieve(self, &shard_indices).await?;
            let MaybeShardingPartialDecoder::Sharding(partial_decoder) = partial_decoder else {
                unreachable!("exlusively sharded")
            };
            // TODO: trait upcasting
            // let partial_decoder: Arc<dyn Any + MaybeSend + MaybeSync> = partial_decoder.clone();
            // let partial_decoder = partial_decoder
            //     .downcast::<AsyncShardingPartialDecoder>()
            //     .expect("array is exclusively sharded");

            Ok(partial_decoder
                .retrieve_subchunk_encoded(&chunk_indices)
                .await?
                .map(Vec::from))
        } else {
            Err(ArrayError::UnsupportedMethod(
                "the array is not exclusively sharded".to_string(),
            ))
        }
    }

    async fn async_retrieve_subchunk_opt<T: FromArrayBytes + MaybeSend>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        if cache.array_is_sharded() {
            let (shard_indices, shard_subset) =
                subchunk_shard_index_and_subset(self, cache, subchunk_indices)?;
            let partial_decoder = cache.retrieve(self, &shard_indices).await?;
            let bytes = partial_decoder
                .partial_decode(&shard_subset, options)
                .await?
                .into_owned();
            T::from_array_bytes(bytes, shard_subset.shape(), self.data_type())
        } else {
            self.async_retrieve_chunk_opt(subchunk_indices, options)
                .await
        }
    }

    async fn async_retrieve_subchunk_elements_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_subchunk_opt(cache, subchunk_indices, options)
            .await
    }

    #[cfg(feature = "ndarray")]
    async fn async_retrieve_subchunk_ndarray_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_subchunk_opt(cache, subchunk_indices, options)
            .await
    }

    async fn async_retrieve_subchunks_opt<T: FromArrayBytes + MaybeSend>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        if cache.array_is_sharded() {
            let subchunk_grid = cache.subchunk_grid();
            let array_subset = subchunk_grid.chunks_subset(subchunks)?.ok_or_else(|| {
                ArrayError::InvalidArraySubset(
                    subchunks.to_array_subset(),
                    subchunk_grid.grid_shape().to_vec(),
                )
            })?;
            self.async_retrieve_array_subset_sharded_opt(cache, &array_subset, options)
                .await
        } else {
            self.async_retrieve_chunks_opt(subchunks, options).await
        }
    }

    async fn async_retrieve_subchunks_elements_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_subchunks_opt(cache, subchunks, options)
            .await
    }

    #[cfg(feature = "ndarray")]
    async fn async_retrieve_subchunks_ndarray_opt<T: ElementOwned + MaybeSend + MaybeSync>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_subchunks_opt(cache, subchunks, options)
            .await
    }

    #[allow(clippy::too_many_lines)]
    async fn async_retrieve_array_subset_sharded_opt<T: FromArrayBytes + MaybeSend>(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        if cache.array_is_sharded() {
            // Find the shards intersecting this array subset
            let shards = self.chunks_in_array_subset(array_subset)?;
            let Some(shards) = shards else {
                return Err(ArrayError::InvalidArraySubset(
                    array_subset.to_array_subset(),
                    self.shape().to_vec(),
                ));
            };

            // Retrieve chunk bytes
            let num_shards = shards.num_elements_usize();
            let array_subset_start = array_subset.start();
            let array_subset_shape = array_subset.shape();
            let bytes = if num_shards == 0 {
                ArrayBytes::new_fill_value(
                    self.data_type(),
                    array_subset.num_elements(),
                    self.fill_value(),
                )
                .map_err(CodecError::from)
                .map_err(ArrayError::from)?
            } else {
                // Calculate chunk/codec concurrency
                let chunk_shape = self.chunk_shape(&vec![0; self.dimensionality()])?;
                let codec_concurrency =
                    self.recommended_codec_concurrency(&chunk_shape, self.data_type())?;
                let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                    options.concurrent_target(),
                    num_shards,
                    options,
                    &codec_concurrency,
                );
                let options = Arc::new(options);

                match self.data_type().size() {
                    DataTypeSize::Variable => {
                        let retrieve_subchunk = |shard_indices: ArrayIndicesTinyVec| {
                            let options = options.clone();
                            let array_subset_start = &array_subset_start;
                            async move {
                                let shard_subset = self.chunk_subset(&shard_indices)?;
                                let shard_subset_overlap = shard_subset.overlap(array_subset)?;
                                let bytes = cache
                                    .retrieve(self, &shard_indices)
                                    .await?
                                    .partial_decode(
                                        &shard_subset_overlap.relative_to(shard_subset.start())?,
                                        &options,
                                    )
                                    .await?
                                    .into_owned()
                                    .into_variable()?;
                                Ok::<_, ArrayError>((
                                    bytes,
                                    shard_subset_overlap.relative_to(array_subset_start)?,
                                ))
                            }
                        };

                        let indices = shards.indices();
                        let futures = indices.into_iter().map(retrieve_subchunk);
                        let chunk_bytes_and_subsets = futures::stream::iter(futures)
                            .buffered(chunk_concurrent_limit)
                            .try_collect()
                            .await?;
                        ArrayBytes::Variable(merge_chunks_vlen(
                            chunk_bytes_and_subsets,
                            &array_subset_shape,
                        )?)
                    }
                    DataTypeSize::Fixed(data_type_size) => {
                        let size_output = array_subset.num_elements_usize() * data_type_size;
                        if size_output == 0 {
                            ArrayBytes::new_flen(vec![])
                        } else {
                            let mut output = Vec::with_capacity(size_output);
                            {
                                let output =
                                    UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut output);
                                let retrieve_shard_into_slice =
                                    |shard_indices: ArrayIndicesTinyVec| {
                                        let options = options.clone();
                                        let array_subset_start = &array_subset_start;
                                        let array_subset_shape = &array_subset_shape;
                                        async move {
                                            let shard_subset = self.chunk_subset(&shard_indices)?;
                                            let shard_subset_overlap =
                                                shard_subset.overlap(array_subset)?;
                                            let mut output_view = unsafe {
                                                // SAFETY: chunks represent disjoint array subsets
                                                ArrayBytesFixedDisjointView::new(
                                                    output,
                                                    data_type_size,
                                                    array_subset_shape.as_ref(),
                                                    shard_subset_overlap
                                                        .relative_to(array_subset_start)?,
                                                )?
                                            };
                                            cache
                                                .retrieve(self, &shard_indices)
                                                .await?
                                                .partial_decode_into(
                                                    &shard_subset_overlap
                                                        .relative_to(shard_subset.start())?,
                                                    (&mut output_view).into(),
                                                    &options,
                                                )
                                                .await?;
                                            Ok::<_, ArrayError>(())
                                        }
                                    };

                                futures::stream::iter(&shards.indices())
                                    .map(Ok)
                                    .try_for_each_concurrent(
                                        Some(chunk_concurrent_limit),
                                        retrieve_shard_into_slice,
                                    )
                                    .await?;
                            }
                            unsafe { output.set_len(size_output) };
                            ArrayBytes::from(output)
                        }
                    }
                }
            };
            T::from_array_bytes(bytes, &array_subset_shape, self.data_type())
        } else {
            self.async_retrieve_array_subset_opt(array_subset, options)
                .await
        }
    }

    async fn async_retrieve_array_subset_elements_sharded_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<T>, ArrayError> {
        self.async_retrieve_array_subset_sharded_opt(cache, array_subset, options)
            .await
    }

    #[cfg(feature = "ndarray")]
    async fn async_retrieve_array_subset_ndarray_sharded_opt<
        T: ElementOwned + MaybeSend + MaybeSync,
    >(
        &self,
        cache: &AsyncArrayShardedReadableExtCache,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<ndarray::ArrayD<T>, ArrayError> {
        self.async_retrieve_array_subset_sharded_opt(cache, array_subset, options)
            .await
    }
}

mod private {
    use super::{Array, AsyncReadableStorageTraits};

    pub trait Sealed {}

    impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> Sealed for Array<TStorage> {}
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use super::*;
    use crate::array::codec::TransposeCodec;
    use crate::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
    use crate::array::{ArrayBuilder, ArraySubset, data_type};
    use zarrs_metadata_ext::codec::transpose::TransposeOrder;
    use zarrs_storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;

    async fn array_sharded_ext_impl(sharded: bool) -> Result<(), Box<dyn std::error::Error>> {
        let store = object_store::memory::InMemory::new();
        let store = Arc::new(zarrs_object_store::AsyncObjectStore::new(store));
        let array_path = "/array";
        let mut builder = ArrayBuilder::new(
            vec![8, 8], // array shape
            vec![4, 4], // regular chunk shape
            data_type::uint16(),
            0u16,
        );
        builder.bytes_to_bytes_codecs(vec![
            #[cfg(feature = "gzip")]
            Arc::new(crate::array::codec::GzipCodec::new(5)?),
        ]);
        if sharded {
            builder.subchunk_shape(vec![2, 2]);
        }
        let array = builder.build(store, array_path)?;

        let data: Vec<u16> = (0..array.shape().iter().product())
            .map(|i| i as u16)
            .collect();

        array
            .async_store_array_subset(&array.subset_all(), &data)
            .await?;

        let cache = AsyncArrayShardedReadableExtCache::new(&array);
        assert_eq!(array.is_sharded(), sharded);
        let subchunk_grid = array.subchunk_grid();
        if sharded {
            assert_eq!(
                array.subchunk_shape(),
                Some(vec![NonZeroU64::new(2).unwrap(); 2])
            );
            assert_eq!(subchunk_grid.grid_shape(), &[4, 4]);

            let compare = array
                .async_retrieve_array_subset::<Vec<u16>>(&[4..6, 6..8])
                .await?;
            let test = array
                .async_retrieve_subchunk_opt::<Vec<u16>>(&cache, &[2, 3], &CodecOptions::default())
                .await?;
            assert_eq!(compare, test);
            assert_eq!(cache.len().await, 1);

            #[cfg(feature = "ndarray")]
            {
                let compare = array
                    .async_retrieve_array_subset::<ndarray::ArrayD<u16>>(&[4..6, 6..8])
                    .await?;
                let test = array
                    .async_retrieve_subchunk_opt::<ndarray::ArrayD<u16>>(
                        &cache,
                        &[2, 3],
                        &CodecOptions::default(),
                    )
                    .await?;
                assert_eq!(compare, test);
            }

            cache.clear().await;
            assert_eq!(cache.len().await, 0);

            let subset = ArraySubset::new_with_ranges(&[3..7, 3..7]);
            let compare = array
                .async_retrieve_array_subset::<Vec<u16>>(&subset)
                .await?;
            let test = array
                .async_retrieve_array_subset_sharded_opt::<Vec<u16>>(
                    &cache,
                    &subset,
                    &CodecOptions::default(),
                )
                .await?;
            assert_eq!(compare, test);
            assert_eq!(cache.len().await, 4);

            #[cfg(feature = "ndarray")]
            {
                let subset = ArraySubset::new_with_ranges(&[3..7, 3..7]);
                let compare = array
                    .async_retrieve_array_subset::<ndarray::ArrayD<u16>>(&subset)
                    .await?;
                let test = array
                    .async_retrieve_array_subset_sharded_opt::<ndarray::ArrayD<u16>>(
                        &cache,
                        &subset,
                        &CodecOptions::default(),
                    )
                    .await?;
                assert_eq!(compare, test);
            }

            let subset = ArraySubset::new_with_ranges(&[2..6, 2..6]);
            let subchunks = ArraySubset::new_with_ranges(&[1..3, 1..3]);
            let compare = array
                .async_retrieve_array_subset::<Vec<u16>>(&subset)
                .await?;
            let test = array
                .async_retrieve_subchunks_opt::<Vec<u16>>(
                    &cache,
                    &subchunks,
                    &CodecOptions::default(),
                )
                .await?;
            assert_eq!(compare, test);
            assert_eq!(cache.len().await, 4);

            #[cfg(feature = "ndarray")]
            {
                let subset = ArraySubset::new_with_ranges(&[2..6, 2..6]);
                let subchunks = ArraySubset::new_with_ranges(&[1..3, 1..3]);
                let compare = array
                    .async_retrieve_array_subset::<ndarray::ArrayD<u16>>(&subset)
                    .await?;
                let test = array
                    .async_retrieve_subchunks_opt::<ndarray::ArrayD<u16>>(
                        &cache,
                        &subchunks,
                        &CodecOptions::default(),
                    )
                    .await?;
                assert_eq!(compare, test);
                assert_eq!(cache.len().await, 4);
            }

            let encoded_subchunk = array
                .async_retrieve_encoded_subchunk(&cache, &[0, 0])
                .await?
                .unwrap();
            assert_eq!(
                array
                    .async_subchunk_byte_range(&cache, &[0, 0])
                    .await?
                    .unwrap()
                    .length(u64::MAX),
                encoded_subchunk.len() as u64
            );
            // assert_eq!(
            //     u16::from_array_bytes(array.data_type(), encoded_subchunk.into())?,
            //     array.async_retrieve_chunk_elements::<u16>(&[0, 0])?
            // );
        } else {
            assert_eq!(array.subchunk_shape(), None);
            assert_eq!(subchunk_grid.grid_shape(), &[2, 2]);

            let compare = array
                .async_retrieve_array_subset::<Vec<u16>>(&[4..8, 4..8])
                .await?;
            let test = array
                .async_retrieve_subchunk_opt::<Vec<u16>>(&cache, &[1, 1], &CodecOptions::default())
                .await?;
            assert_eq!(compare, test);

            let subset = ArraySubset::new_with_ranges(&[3..7, 3..7]);
            let compare = array
                .async_retrieve_array_subset::<Vec<u16>>(&subset)
                .await?;
            let test = array
                .async_retrieve_array_subset_sharded_opt::<Vec<u16>>(
                    &cache,
                    &subset,
                    &CodecOptions::default(),
                )
                .await?;
            assert_eq!(compare, test);
            assert!(cache.is_empty().await);

            assert!(
                array
                    .async_retrieve_encoded_subchunk(&cache, &[0, 0])
                    .await
                    .is_err()
            );
            assert!(
                array
                    .async_subchunk_byte_range(&cache, &[0, 0])
                    .await
                    .is_err()
            );
        }

        Ok(())
    }

    #[tokio::test]
    async fn async_array_sharded_ext_sharded() -> Result<(), Box<dyn std::error::Error>> {
        array_sharded_ext_impl(true).await
    }

    #[tokio::test]
    async fn async_array_sharded_ext_unsharded() -> Result<(), Box<dyn std::error::Error>> {
        array_sharded_ext_impl(false).await
    }

    async fn array_sharded_ext_impl_transpose(
        valid_subchunk_shape: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let store = object_store::memory::InMemory::new();
        let store = Arc::new(zarrs_object_store::AsyncObjectStore::new(store));
        let store = Arc::new(PerformanceMetricsStorageAdapter::new(store));

        let array_path = "/array";
        let mut builder = ArrayBuilder::new(
            vec![16, 16, 9], // array shape
            vec![8, 4, 3],   // regular chunk shape
            data_type::uint32(),
            0u32,
        );
        builder.array_to_array_codecs(vec![Arc::new(TransposeCodec::new(TransposeOrder::new(
            &[1, 0, 2],
        )?))]);
        builder.array_to_bytes_codec(Arc::new(
            ShardingCodecBuilder::new(
                vec![
                    NonZeroU64::new(1).unwrap(),
                    if valid_subchunk_shape {
                        NonZeroU64::new(2).unwrap()
                    } else {
                        NonZeroU64::new(3).unwrap()
                    },
                    NonZeroU64::new(3).unwrap(),
                ],
                &data_type::uint32(),
            )
            .bytes_to_bytes_codecs(vec![
                #[cfg(feature = "gzip")]
                Arc::new(crate::array::codec::GzipCodec::new(5)?),
            ])
            .build(),
        ));
        let array = builder.build(store.clone(), array_path)?;

        let subchunk_grid = array.subchunk_grid();
        if valid_subchunk_shape {
            //  Config:
            //  16 x 16 x 9 Array shape
            //   8 x  4 x 3 Chunk (shard) shape
            //   1 x  2 x 3 Subchunk shape
            //      [1,0,2] Transpose order
            //  Calculations:
            //   2 x  4 x 3 Number of shards (chunk grid shape)
            //   4 x  8 x 3 Transposed shard shape
            //   4 x  4 x 1 Subchunks per (transposed) shard
            //   8 x 16 x 3 Subchunk grid shape
            //   2 x  1 x 3 Effective subchunk shape (read granularity)

            assert_eq!(array.chunk_grid_shape(), &[2, 4, 3]);
            assert_eq!(
                array.subchunk_shape(),
                Some(vec![
                    NonZeroU64::new(1).unwrap(),
                    NonZeroU64::new(2).unwrap(),
                    NonZeroU64::new(3).unwrap()
                ])
            );
            assert_eq!(
                array.effective_subchunk_shape(),
                Some(vec![
                    NonZeroU64::new(2).unwrap(),
                    NonZeroU64::new(1).unwrap(),
                    NonZeroU64::new(3).unwrap()
                ])
            ); // NOTE: transposed
            assert_eq!(subchunk_grid.grid_shape(), &[8, 16, 3]);
        } else {
            // skip above tests if the subchunk shape is invalid, below calls fail with
            // CodecError(Other("invalid subchunk shape [1, 3, 3], it must evenly divide [4, 8, 3]"))
        }

        let data: Vec<u32> = (0..array.shape().iter().product())
            .map(|i| i as u32)
            .collect();
        array
            .async_store_array_subset(&array.subset_all(), &data)
            .await?;

        // Retrieving a subchunk should be exactly 2 reads: index + chunk
        let subchunk_subset = subchunk_grid.subset(&[0, 0, 0])?.unwrap();
        let subchunk_data = array
            .async_retrieve_array_subset::<Vec<u32>>(&subchunk_subset)
            .await?;
        assert_eq!(subchunk_data, &[0, 1, 2, 144, 145, 146]);
        assert_eq!(store.reads(), 2);

        Ok(())
    }

    #[tokio::test]
    async fn async_array_sharded_ext_impl_transpose_valid_subchunk_shape() {
        assert!(array_sharded_ext_impl_transpose(true).await.is_ok());
    }

    #[tokio::test]
    async fn async_array_sharded_ext_impl_transpose_invalid_subchunk_shape() {
        assert_eq!(
            array_sharded_ext_impl_transpose(false)
                .await
                .unwrap_err()
                .to_string(),
            "invalid subchunk shape [1, 3, 3], it must evenly divide shard shape [4, 8, 3]"
        );
    }
}
