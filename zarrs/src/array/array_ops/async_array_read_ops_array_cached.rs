use inherent::inherent;
use std::sync::Arc;

use futures::{StreamExt, TryStreamExt};

use super::array_read_ops_array_cached::copy_cached_chunk_into;
use super::async_array_read_ops_common::AsyncRetrieveInto;
use super::{AsyncArrayReadOps, *};
use crate::array::array_bytes_internal::merge_cached_chunks;
use crate::array::chunk_cache::{
    AsyncChunkCache, Sealed, SealedAsync, async_retrieve_chunk_bytes,
    async_retrieve_chunk_overlap_bytes, fill_value_bytes,
};
use crate::array::{ArrayBytes, ArrayIndices};
use zarrs_codec::plan::{SharedOutput, TargetViews};
use zarrs_codec::{ArrayBytesDecodeIntoTarget, AsyncArrayPartialDecoderTraits, CodecError};
use zarrs_storage::Bytes;

async fn async_retrieve_array_subset_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncChunkCache + ?Sized,
{
    if array_subset.dimensionality() != array.dimensionality() {
        return Err(ArrayError::InvalidArraySubset(
            Box::new(array_subset.to_array_subset()),
            array.shape().to_vec(),
        ));
    }
    let Some(chunks) = array.chunks_in_array_subset(array_subset)? else {
        return Err(ArrayError::InvalidArraySubset(
            Box::new(array_subset.to_array_subset()),
            array.shape().to_vec(),
        ));
    };
    match chunks.num_elements_usize() {
        0 => fill_value_bytes(array, array_subset.num_elements()),
        1 => {
            let chunk_indices = chunks.start();
            let chunk_subset = array.chunk_subset(chunk_indices)?;
            async_retrieve_chunk_overlap_bytes(
                cache,
                array,
                chunk_indices,
                &chunk_subset,
                array_subset,
                options,
                resources,
            )
            .await
        }
        num_chunks => {
            let (chunk_concurrent_limit, resources) = split_chunk_resources(num_chunks, resources);
            if array.data_type().is_fixed() {
                async_retrieve_multi_chunk_fixed(
                    cache,
                    array,
                    array_subset,
                    &chunks,
                    chunk_concurrent_limit,
                    options,
                    &resources,
                )
                .await
            } else {
                async_retrieve_multi_chunk_variable(
                    cache,
                    array,
                    array_subset,
                    &chunks,
                    chunk_concurrent_limit,
                    options,
                    &resources,
                )
                .await
            }
        }
    }
}

async fn async_retrieve_multi_chunk_variable<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &dyn ArraySubsetTraits,
    chunk_concurrent_limit: usize,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncChunkCache + ?Sized,
{
    let retrieve_chunk = |chunk_indices: ArrayIndices| async move {
        let chunk_subset = array.chunk_subset(&chunk_indices)?;
        let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
        let bytes = async_retrieve_chunk_overlap_bytes(
            cache,
            array,
            &chunk_indices,
            &chunk_subset,
            &chunk_subset_overlap,
            options,
            resources,
        )
        .await?;
        Ok::<_, ArrayError>((
            bytes,
            chunk_subset_overlap.relative_to(&array_subset.start())?,
        ))
    };

    let chunk_bytes_and_subsets: Vec<_> = futures::stream::iter(chunks.indices().iter())
        .map(retrieve_chunk)
        .buffered(chunk_concurrent_limit)
        .try_collect()
        .await?;

    Ok(merge_cached_chunks(
        chunk_bytes_and_subsets,
        &array_subset.shape(),
        array.data_type(),
    )?
    .into())
}

/// The source of the elements of a chunk in a cached read.
enum AsyncChunkSource {
    /// The decoded chunk, or [`None`] if it does not exist.
    Chunk(Option<Arc<ArrayBytes<'static>>>),
    /// A partial decoder of the chunk.
    Decoder(Arc<dyn AsyncArrayPartialDecoderTraits>),
}

impl AsyncChunkSource {
    async fn retrieve<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Self, ArrayError>
    where
        TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
        C: AsyncChunkCache + ?Sized,
    {
        Ok(if C::Value::HOLDS_DECODED {
            // The elements are copied from the decoded chunk, rather than extracted then copied
            Self::Chunk(
                C::Value::async_retrieve_chunk_bytes_if_exists(
                    cache,
                    array,
                    chunk_indices,
                    options,
                    resources,
                )
                .await?,
            )
        } else {
            // The elements are decoded into the output, rather than decoded then copied
            Self::Decoder(
                C::Value::async_partial_decoder(cache, array, chunk_indices, options, resources)
                    .await?,
            )
        })
    }

    /// Decode the elements of `indexer` of the chunk of `chunk_shape` into `target`.
    async fn decode_into<TStorage: ?Sized>(
        &self,
        array: &Array<TStorage>,
        chunk_shape: &[u64],
        indexer: &dyn Indexer,
        target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        match self {
            Self::Chunk(chunk) => {
                copy_cached_chunk_into(array, chunk.as_deref(), chunk_shape, indexer, target)
            }
            Self::Decoder(decoder) => {
                decoder
                    .partial_decode_into(indexer, target, options, resources)
                    .await
            }
        }
    }

    /// Decode the elements of `indexer` of the chunk of `chunk_shape` into `views`.
    async fn decode_into_views<TStorage: ?Sized>(
        &self,
        array: &Array<TStorage>,
        chunk_shape: &[u64],
        indexer: &dyn Indexer,
        views: TargetViews<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        match self {
            Self::Chunk(chunk) => views.with_target(|target| {
                copy_cached_chunk_into(array, chunk.as_deref(), chunk_shape, indexer, target)
            }),
            Self::Decoder(decoder) => {
                views
                    .async_partial_decode_into(&**decoder, indexer, options, resources)
                    .await
            }
        }
    }
}

async fn async_retrieve_multi_chunk_fixed<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &dyn ArraySubsetTraits,
    chunk_concurrent_limit: usize,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncChunkCache + ?Sized,
{
    let (output, views) = SharedOutput::for_data_type(array.data_type(), &array_subset.shape())
        .expect("fixed data type");
    let array_subset_start = array_subset.start();
    let retrieve_chunk = |chunk_indices: ArrayIndices| {
        let (array_subset_start, views) = (&array_subset_start, &views);
        async move {
            let chunk_subset = array.chunk_subset(&chunk_indices)?;
            let overlap = chunk_subset.overlap(array_subset)?;
            let output_subset = overlap.relative_to(array_subset_start)?;
            let chunk_overlap = overlap.relative_to(chunk_subset.start())?;
            let source =
                AsyncChunkSource::retrieve(cache, array, &chunk_indices, options, resources)
                    .await?;
            // SAFETY: the chunks are disjoint
            let views = unsafe { views.subdivide(output_subset) }?;
            source
                .decode_into_views(
                    array,
                    chunk_subset.shape(),
                    &chunk_overlap,
                    views,
                    options,
                    resources,
                )
                .await?;
            Ok::<_, ArrayError>(())
        }
    };
    futures::stream::iter(&chunks.indices())
        .map(Ok)
        .try_for_each_concurrent(Some(chunk_concurrent_limit), retrieve_chunk)
        .await?;
    drop(views);
    // SAFETY: the chunks cover the array subset
    let bytes = unsafe { output.into_array_bytes() }
        .map_err(|_| CodecError::Other("an output view outlived its chunk".to_string()))?;
    Ok(bytes.into())
}

impl<TStorage, C> ArrayCached<TStorage, C>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncChunkCache,
{
    pub(in crate::array) async fn async_retrieve_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.array().chunk_shape(chunk_indices)?;
        self.async_retrieve_partial_chunk_into_with_options(
            chunk_indices,
            &ArraySubset::new_with_shape(bytemuck::must_cast_slice::<_, u64>(&chunk_shape)),
            output_target,
            options,
            resources,
        )
        .await
    }

    pub(in crate::array) async fn async_retrieve_partial_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let source = AsyncChunkSource::retrieve(
            self.cache(),
            self.array(),
            chunk_indices,
            options,
            resources,
        )
        .await?;
        let chunk_shape = self.array().chunk_shape(chunk_indices)?;
        source
            .decode_into(
                self.array(),
                bytemuck::must_cast_slice(&chunk_shape),
                indexer,
                output_target,
                options,
                resources,
            )
            .await
            .map_err(ArrayError::CodecError)
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage, C> AsyncRetrieveInto for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncChunkCache,
{
    async fn retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.async_retrieve_chunk_into_with_options(
            chunk_indices,
            output_target,
            options,
            resources,
        )
        .await
    }

    async fn retrieve_partial_chunk_into(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.async_retrieve_partial_chunk_into_with_options(
            chunk_indices,
            indexer,
            output_target,
            options,
            resources,
        )
        .await
    }
}

#[inherent]
impl<TStorage, C> AsyncArrayReadOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncChunkCache,
{
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let options = self.codec_options();
        let bytes = async_retrieve_chunk_bytes(
            self.cache(),
            self.array(),
            chunk_indices,
            options,
            resources,
        )
        .await?;
        let shape = self.array().chunk_shape(chunk_indices)?;
        T::from_array_bytes_arc(
            bytes,
            bytemuck::must_cast_slice(&shape),
            self.array().data_type(),
        )
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.async_retrieve_chunk_into_with_options(
            chunk_indices,
            output_target,
            self.codec_options(),
            resources,
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_partial_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let bytes = C::Value::async_retrieve_partial_chunk_bytes(
            self.cache(),
            self.array(),
            chunk_indices,
            indexer,
            self.codec_options(),
            resources,
        )
        .await?;
        T::from_array_bytes_arc(bytes, &indexer.output_shape(), self.array().data_type())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_partial_chunk_into(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.async_retrieve_partial_chunk_into_with_options(
            chunk_indices,
            indexer,
            output_target,
            self.codec_options(),
            resources,
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let bytes = async_retrieve_array_subset_bytes(
            self.cache(),
            self.array(),
            array_subset,
            self.codec_options(),
            resources,
        )
        .await?;
        T::from_array_bytes_arc(bytes, &array_subset.shape(), self.array().data_type())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<T>, ArrayError> {
        let options = self.codec_options();
        let Some(bytes) = C::Value::async_retrieve_chunk_bytes_if_exists(
            self.cache(),
            self.array(),
            chunk_indices,
            options,
            resources,
        )
        .await?
        else {
            return Ok(None);
        };
        let shape = self.array().chunk_shape(chunk_indices)?;
        T::from_array_bytes_arc(
            bytes,
            bytemuck::must_cast_slice(&shape),
            self.array().data_type(),
        )
        .map(Some)
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<EncodedSubchunk>, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<EncodedSubchunk>, ArrayError> {
        self.array()
            .async_retrieve_encoded_subchunk_at_level(level, subchunk_indices, resources)
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Bytes>, ArrayError> {
        self.array()
            .async_retrieve_encoded_chunk(chunk_indices)
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_encoded_chunks(
        &self,
        chunks: &dyn Indexer,
        resources: &Resources,
    ) -> Result<Vec<Option<Bytes>>, ArrayError> {
        self.array()
            .async_retrieve_encoded_chunks(chunks, resources)
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunk<T: FromArrayBytes>(
        &self,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunk_at_level<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunks<T: FromArrayBytes>(
        &self,
        subchunks: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunks_at_level<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunks: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        super::async_array_read_ops_common::retrieve_array_subset_into(
            self.array().as_ref(),
            self,
            array_subset,
            output_target,
            self.codec_options(),
            resources,
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_partial_decoder(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        C::Value::async_partial_decoder(
            self.cache(),
            self.array(),
            chunk_indices,
            self.codec_options(),
            resources,
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_local_subchunk_grid(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<ChunkGrid>, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_local_subchunk_grid_at_level(
        &self,
        level: usize,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<ChunkGrid>, ArrayError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::chunk_cache::{
        AsyncChunkCacheDecodedLruChunkLimit, AsyncChunkCacheDecodedLruSizeLimit,
        AsyncChunkCacheEncodedLruChunkLimit, AsyncChunkCacheEncodedLruSizeLimit,
        AsyncChunkCachePartialDecoderLruChunkLimit, AsyncChunkCachePartialDecoderLruSizeLimit,
    };
    use crate::array::{ArrayBuilder, ArrayBytesFixedDisjointView, FillValue, data_type};
    use unsafe_cell_slice::UnsafeCellSlice;
    use zarrs_storage::store::AsyncMemoryStore;

    async fn test_cache_async<C>(cache: C)
    where
        C: AsyncChunkCache + 'static,
    {
        let store = Arc::new(AsyncMemoryStore::new());
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array
            .async_store_chunk(&[0], &[1u8, 2], &Resources::default())
            .await
            .unwrap();

        let cached = ArrayCached::new(array, cache);
        assert_eq!(
            cached
                .async_retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                .await
                .unwrap(),
            vec![1, 2]
        );
        assert_eq!(
            cached
                .async_retrieve_partial_chunk::<Vec<u8>>(&[0], &[1..2], &Resources::default())
                .await
                .unwrap(),
            vec![2]
        );
        assert_eq!(
            cached
                .async_retrieve_chunk_if_exists::<Vec<u8>>(&[1], &Resources::default())
                .await
                .unwrap(),
            None
        );
        assert_eq!(
            cached
                .async_partial_decoder(&[0], &Resources::default())
                .await
                .unwrap()
                .partial_decode(
                    &ArraySubset::new_with_ranges(&[0..1]),
                    &CodecOptions::default(),
                    &Resources::default(),
                )
                .await
                .unwrap(),
            vec![1].into()
        );
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u8>>(&[1..3], &Resources::default())
                .await
                .unwrap(),
            vec![2, 0]
        );
        assert!(
            cached
                .async_retrieve_chunk::<Vec<u8>>(&[2], &Resources::default())
                .await
                .is_err()
        );
        assert!(!cached.cache().is_empty().await);

        // Write operations invalidate affected cached chunks
        cached
            .async_store_chunk(&[0], &[3u8, 4], &Resources::default())
            .await
            .unwrap();
        assert_eq!(
            cached
                .async_retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                .await
                .unwrap(),
            vec![3, 4]
        );
        cached
            .async_store_array_subset(&[0..1], &[5u8], &Resources::default())
            .await
            .unwrap();
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u8>>(&[0..4], &Resources::default())
                .await
                .unwrap(),
            vec![5, 4, 0, 0]
        );
        cached.async_erase_chunk(&[0]).await.unwrap();
        assert_eq!(
            cached
                .async_retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                .await
                .unwrap(),
            vec![0, 0]
        );

        cached.cache().invalidate().await;
        assert!(cached.cache().is_empty().await);
    }

    async fn test_cache_sharded_async<C>(cache: C)
    where
        C: AsyncChunkCache + 'static,
    {
        let store = Arc::new(AsyncMemoryStore::new());
        let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type::uint16(), 0u16);
        builder.subchunk_shape(vec![2, 2]);
        let array = builder.build_arc(store, "/").unwrap();
        let data: Vec<u16> = (0..64).collect();
        array
            .async_store_array_subset(&array.subset_all(), &data, &Resources::default())
            .await
            .unwrap();

        // Exercise the cached reads under a non-default concurrency target.
        let cached = ArrayCached::new(array, cache);
        let resources = Resources::default().with_compute_target(1);
        assert_eq!(
            cached
                .async_retrieve_subchunk::<Vec<u16>>(&[2, 3], &resources)
                .await
                .unwrap(),
            vec![38, 39, 46, 47]
        );
        assert_eq!(
            cached
                .async_retrieve_subchunks::<Vec<u16>>(&[1..3, 1..3], &resources)
                .await
                .unwrap(),
            vec![
                18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45,
            ]
        );
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u16>>(&[0..8, 0..8], &resources)
                .await
                .unwrap(),
            data
        );
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u16>>(&[1..8, 0..8], &resources)
                .await
                .unwrap(),
            data[8..]
        );
        assert!(
            cached
                .async_retrieve_subchunk::<Vec<u16>>(&[0], &resources)
                .await
                .is_err()
        );
        assert!(
            cached
                .async_retrieve_subchunks::<Vec<u16>>(&[0..1], &resources)
                .await
                .is_err()
        );
        assert!(!cached.cache().is_empty().await);
    }

    async fn test_cache_into_async<C>(cache: C)
    where
        C: AsyncChunkCache + 'static,
    {
        let store = Arc::new(AsyncMemoryStore::new());
        let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array
            .async_store_array_subset(
                &array.subset_all(),
                &(0..16u8).collect::<Vec<u8>>(),
                &Resources::default(),
            )
            .await
            .unwrap();

        let cached = ArrayCached::new(array, cache);
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
        let shape = subset.shape().to_vec();
        let mut buf = vec![0u8; subset.num_elements_usize()];
        {
            let slice = UnsafeCellSlice::new(&mut buf);
            let mut view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    slice,
                    1,
                    &shape,
                    ArraySubset::new_with_shape(shape.clone()),
                )
                .unwrap()
            };
            cached
                .async_retrieve_array_subset_into(
                    &subset,
                    ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                    &Resources::default(),
                )
                .await
                .unwrap();
        }
        assert_eq!(buf, vec![5, 6, 9, 10]);
        assert!(!cached.cache().is_empty().await);
    }

    /// Merge complete and partial variable-length chunks, including missing chunks.
    async fn test_cache_variable_async<C>(cache: C)
    where
        C: AsyncChunkCache + 'static,
    {
        let store = Arc::new(AsyncMemoryStore::new());
        let array = ArrayBuilder::new(vec![6, 6], vec![2, 2], data_type::string(), "")
            .build_arc(store, "/")
            .unwrap();
        let data: Vec<String> = (0..36)
            .map(|i| if i < 24 { i.to_string() } else { String::new() })
            .collect();
        array
            .async_store_array_subset(&[0..4, 0..6], &data[..24], &Resources::default())
            .await
            .unwrap();

        let cached = ArrayCached::new(array, cache);
        // Repeat the reads to exercise both cold and populated caches.
        for _ in 0..2 {
            assert_eq!(
                cached
                    .async_retrieve_array_subset::<Vec<String>>(
                        &[0..6, 0..6],
                        &Resources::default()
                    )
                    .await
                    .unwrap(),
                data
            );
            let expected: Vec<String> = (1..6)
                .flat_map(|row| data[row * 6 + 1..row * 6 + 6].iter().cloned())
                .collect();
            assert_eq!(
                cached
                    .async_retrieve_array_subset::<Vec<String>>(
                        &[1..6, 1..6],
                        &Resources::default()
                    )
                    .await
                    .unwrap(),
                expected
            );
        }
    }

    /// Retrieving a subset spanning multiple chunks must handle nested optional data types.
    async fn test_cache_nested_optional_async<C>(cache: C)
    where
        C: AsyncChunkCache + 'static,
    {
        let store = Arc::new(AsyncMemoryStore::new());
        let array = ArrayBuilder::new(
            vec![6],
            vec![2],
            data_type::uint8().to_optional().to_optional(),
            FillValue::from(None::<Option<u8>>),
        )
        .build_arc(store, "/")
        .unwrap();
        array
            .async_store_chunk(&[0], &[Some(Some(1u8)), Some(None)], &Resources::default())
            .await
            .unwrap();
        array
            .async_store_chunk(&[1], &[None, Some(Some(4u8))], &Resources::default())
            .await
            .unwrap();
        // chunk 2 is unwritten, so it decodes to the fill value (`None`)

        let cached = ArrayCached::new(array, cache);
        // Spans all three chunks
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<Option<Option<u8>>>>(
                    &[0..6],
                    &Resources::default()
                )
                .await
                .unwrap(),
            vec![Some(Some(1)), Some(None), None, Some(Some(4)), None, None]
        );
        // Partial overlap of two chunks
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<Option<Option<u8>>>>(
                    &[1..4],
                    &Resources::default()
                )
                .await
                .unwrap(),
            vec![Some(None), None, Some(Some(4))]
        );
    }

    #[tokio::test]
    async fn async_lru_caches_support_encoded_values() {
        test_cache_async(AsyncChunkCacheEncodedLruChunkLimit::new(2)).await;
        test_cache_async(AsyncChunkCacheEncodedLruSizeLimit::new(1024)).await;
        test_cache_sharded_async(AsyncChunkCacheEncodedLruChunkLimit::new(4)).await;
        test_cache_into_async(AsyncChunkCacheEncodedLruChunkLimit::new(4)).await;
        test_cache_nested_optional_async(AsyncChunkCacheEncodedLruChunkLimit::new(4)).await;
        test_cache_variable_async(AsyncChunkCacheEncodedLruChunkLimit::new(9)).await;
        test_cache_variable_async(AsyncChunkCacheEncodedLruSizeLimit::new(4096)).await;
    }

    #[tokio::test]
    async fn async_lru_caches_support_decoded_values() {
        test_cache_async(AsyncChunkCacheDecodedLruChunkLimit::new(2)).await;
        test_cache_async(AsyncChunkCacheDecodedLruSizeLimit::new(1024)).await;
        test_cache_sharded_async(AsyncChunkCacheDecodedLruChunkLimit::new(4)).await;
        test_cache_into_async(AsyncChunkCacheDecodedLruChunkLimit::new(4)).await;
        test_cache_nested_optional_async(AsyncChunkCacheDecodedLruChunkLimit::new(4)).await;
        test_cache_variable_async(AsyncChunkCacheDecodedLruChunkLimit::new(9)).await;
        test_cache_variable_async(AsyncChunkCacheDecodedLruSizeLimit::new(4096)).await;
    }

    #[tokio::test]
    async fn async_lru_caches_support_async_partial_decoder_values() {
        test_cache_async(AsyncChunkCachePartialDecoderLruChunkLimit::new(2)).await;
        test_cache_async(AsyncChunkCachePartialDecoderLruSizeLimit::new(1024)).await;
        test_cache_sharded_async(AsyncChunkCachePartialDecoderLruChunkLimit::new(4)).await;
        test_cache_into_async(AsyncChunkCachePartialDecoderLruChunkLimit::new(4)).await;
        test_cache_nested_optional_async(AsyncChunkCachePartialDecoderLruChunkLimit::new(4)).await;
        test_cache_variable_async(AsyncChunkCachePartialDecoderLruChunkLimit::new(9)).await;
        test_cache_variable_async(AsyncChunkCachePartialDecoderLruSizeLimit::new(4096)).await;
    }

    /// Cached async retrievals must be `Send` so that they can be used with
    /// executors that move tasks between threads (e.g. `tokio::spawn`).
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_cache_spawn_async<C>(cache: C)
    where
        C: AsyncChunkCache + 'static,
    {
        let store = Arc::new(AsyncMemoryStore::new());
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array
            .async_store_chunk(&[0], &[1u8, 2], &Resources::default())
            .await
            .unwrap();

        let cached = Arc::new(ArrayCached::new(array, cache));
        let handles = (0..4).map(|_| {
            let cached = cached.clone();
            tokio::spawn(async move {
                cached
                    .async_retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                    .await
                    .unwrap()
            })
        });
        for handle in handles {
            assert_eq!(handle.await.unwrap(), vec![1, 2]);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn async_lru_caches_are_send() {
        test_cache_spawn_async(AsyncChunkCacheEncodedLruChunkLimit::new(2)).await;
        test_cache_spawn_async(AsyncChunkCacheDecodedLruChunkLimit::new(2)).await;
        test_cache_spawn_async(AsyncChunkCachePartialDecoderLruChunkLimit::new(2)).await;
    }
}
