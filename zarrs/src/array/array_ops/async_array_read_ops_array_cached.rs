use inherent::inherent;
use std::sync::Arc;
use zarrs_codec::Resources;

use futures::{StreamExt, TryStreamExt};

use super::async_array_read_ops_common::{AsyncRetrieveInto, retrieve_multi_chunk_fixed_into};
use super::{
    Array, ArrayCached, ArrayError, ArraySubset, ArraySubsetTraits, AsyncArrayReadOps,
    AsyncReadableStorageTraits, COrderBytes, ChunkGrid, CodecOptions, FromArrayBytes, Indexer,
    recommended_codec_concurrency,
};
use crate::array::array_bytes_internal::{
    FixedDecodeBuffers, merge_cached_chunks, optional_innermost_with_depth,
};
use crate::array::cache::{
    AsyncArrayCache, async_retrieve_chunk_overlap_bytes, async_retrieve_chunk_overlap_into,
};
use crate::array::concurrency::concurrency_chunks_and_codec;
use crate::array::{ArrayBytes, ArrayIndices, DataType};
use zarrs_codec::{ArrayBytesTarget, AsyncArrayPartialDecoderTraits};
use zarrs_storage::Bytes;

/// Retrieve `array_subset`, which spans `chunks`, with elements in C order.
///
/// `chunks` are the chunks intersecting `array_subset` (see [`array_subset_chunks`](super::array_subset_chunks)).
async fn async_retrieve_array_subset_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &ArraySubset,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncArrayCache + ?Sized,
{
    match chunks.num_elements_usize() {
        0 => super::fill_value_bytes(
            array.data_type(),
            array.fill_value(),
            array_subset.num_elements(),
        )
        .map(Arc::new),
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
            let chunk_shape = array.chunk_shape(chunks.start())?;
            let codec_concurrency = recommended_codec_concurrency(array, &chunk_shape)?;
            let (chunk_concurrent_limit, resources) = concurrency_chunks_and_codec(
                num_chunks,
                array.chunk_concurrent_minimum,
                resources,
                &codec_concurrency,
            );
            if let (DataType::Fixed(fixed), optional_depth) =
                optional_innermost_with_depth(array.data_type())
            {
                async_retrieve_multi_chunk_fixed(
                    cache,
                    array,
                    array_subset,
                    chunks,
                    fixed.size(),
                    optional_depth,
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
                    chunks,
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
    C: AsyncArrayCache + ?Sized,
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

    let chunk_bytes_and_subsets = {
        let indices = chunks.indices();
        let indices = indices.iter();
        let mut values = Vec::with_capacity(indices.size_hint().0);
        let mut results = futures::stream::iter(indices)
            .map(retrieve_chunk)
            .buffered(chunk_concurrent_limit);

        while let Some(value) = results.try_next().await? {
            values.push(value);
        }
        values
    };

    Ok(merge_cached_chunks(
        chunk_bytes_and_subsets,
        &array_subset.shape(),
        array.data_type(),
    )?
    .into())
}

#[expect(clippy::too_many_arguments)]
async fn async_retrieve_multi_chunk_fixed<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &dyn ArraySubsetTraits,
    data_type_size: usize,
    optional_depth: usize,
    chunk_concurrent_limit: usize,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncArrayCache + ?Sized,
{
    // Zero-sized data has no bytes to decode, unlike the masks of zero-sized optional data
    if optional_depth == 0 && data_type_size == 0 {
        return Ok(Arc::new(ArrayBytes::new_flen(vec![])));
    }
    let mut buffers = FixedDecodeBuffers::with_size_and_depth(
        data_type_size,
        optional_depth,
        &array_subset.shape(),
    );
    retrieve_multi_chunk_fixed_into(
        array,
        array_subset,
        chunks,
        chunk_concurrent_limit,
        &buffers.target()?,
        |chunk_indices, chunk_subset, overlap, mut target| async move {
            async_retrieve_chunk_overlap_into(
                cache,
                array,
                &chunk_indices,
                &chunk_subset,
                &overlap,
                &mut target,
                options,
                resources,
            )
            .await
        },
    )
    .await?;
    // SAFETY: decoding covers the array subset and satisfies the target's output requirements
    Ok(unsafe { buffers.into_array_bytes() }.into())
}

impl<TStorage, C> ArrayCached<TStorage, C>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncArrayCache + ?Sized,
{
    pub(in crate::array) async fn async_retrieve_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.cache()
            .async_cached_chunk_into(
                self.array().as_ref(),
                chunk_indices,
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
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.cache()
            .async_cached_partial_chunk_into(
                self.array().as_ref(),
                chunk_indices,
                indexer,
                output_target,
                options,
                resources,
            )
            .await
    }
}

#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<TStorage, C> AsyncRetrieveInto for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncArrayCache + ?Sized,
{
    async fn retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesTarget<'_>,
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
        output_target: &mut ArrayBytesTarget<'_>,
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
    C: AsyncArrayCache + ?Sized,
{
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let options = self.codec_options();
        let chunk_shape = self.array().chunk_shape(chunk_indices)?;
        super::async_retrieve_in_order(
            self.array().data_type(),
            self.element_order(),
            bytemuck::must_cast_slice(&chunk_shape),
            async |target| {
                self.async_retrieve_chunk_into_with_options(
                    chunk_indices,
                    target,
                    options,
                    resources,
                )
                .await
            },
            async || {
                self.cache()
                    .async_cached_chunk_bytes(
                        self.array().as_ref(),
                        chunk_indices,
                        options,
                        resources,
                    )
                    .await
            },
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesTarget<'_>,
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
        let options = self.codec_options();
        let chunk_shape = self.array().chunk_shape(chunk_indices)?;
        super::validate_partial_chunk_subset(indexer, bytemuck::must_cast_slice(&chunk_shape))?;
        super::async_retrieve_in_order(
            self.array().data_type(),
            super::indexer_element_order(indexer, self.element_order()),
            &indexer.output_shape(),
            async |target| {
                self.async_retrieve_partial_chunk_into_with_options(
                    chunk_indices,
                    indexer,
                    target,
                    options,
                    resources,
                )
                .await
            },
            async || {
                self.cache()
                    .async_cached_partial_chunk_bytes(
                        self.array().as_ref(),
                        chunk_indices,
                        indexer,
                        options,
                        resources,
                    )
                    .await
            },
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_partial_chunk_into(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
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
        let options = self.codec_options();
        let chunks = super::array_subset_chunks(self, array_subset)?;
        super::async_retrieve_in_order(
            self.array().data_type(),
            self.element_order(),
            &array_subset.shape(),
            async |target| {
                self.async_retrieve_array_subset_into(array_subset, target, resources)
                    .await
            },
            async || {
                async_retrieve_array_subset_bytes(
                    self.cache(),
                    self.array(),
                    array_subset,
                    &chunks,
                    options,
                    resources,
                )
                .await
            },
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<T>, ArrayError> {
        let options = self.codec_options();
        let Some(bytes) = self
            .cache()
            .async_cached_chunk_bytes_if_exists(
                self.array().as_ref(),
                chunk_indices,
                options,
                resources,
            )
            .await?
        else {
            return Ok(None);
        };
        let chunk_shape = self.array().chunk_shape(chunk_indices)?;
        let shape = bytemuck::must_cast_slice(&chunk_shape);
        if self.element_order().is_reordered(shape) {
            super::from_c_order_in_f_order(&bytes, shape, self.array().data_type())
        } else {
            bytes.into_elements(shape, self.array().data_type(), self.element_order())
        }
        .map(Some)
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
        output_target: &mut ArrayBytesTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        super::async_array_read_ops_common::retrieve_array_subset_into(
            self.array().as_ref(),
            self,
            array_subset,
            output_target,
            self.array().chunk_concurrent_minimum,
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
        self.cache()
            .async_cached_partial_decoder(
                self.array().as_ref(),
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
    use crate::array::cache::{
        AsyncChunkCacheDecoded, AsyncChunkCacheEncoded, AsyncChunkCachePartialDecoder,
        AsyncChunkCacheTraits,
    };
    use crate::array::{
        ArrayBuilder, ArrayBytesFixedDisjointView, ArraySubset, FillValue, data_type,
    };
    use unsafe_cell_slice::UnsafeCellSlice;
    use zarrs_storage::store::AsyncMemoryStore;

    async fn test_cache_async<C>(cache: Box<C>)
    where
        C: AsyncChunkCacheTraits + ?Sized + 'static,
    {
        let store = Arc::new(AsyncMemoryStore::new());
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array
            .async_store_chunk(&[0], &[1u8, 2], &Resources::default())
            .await
            .unwrap();

        let cached = ArrayCached::new_boxed(array, cache);
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
        C: AsyncChunkCacheTraits + 'static,
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
        C: AsyncChunkCacheTraits + 'static,
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
            let view = unsafe {
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
                    &mut ArrayBytesTarget::Fixed(view),
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
        C: AsyncChunkCacheTraits + 'static,
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
        C: AsyncChunkCacheTraits + 'static,
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
    async fn async_chunk_caches_support_encoded_values() {
        test_cache_async(Box::new(AsyncChunkCacheEncoded::new_chunk_limit(2))).await;
        test_cache_async(Box::new(AsyncChunkCacheEncoded::new_size_limit(1024, None))).await;
        test_cache_sharded_async(AsyncChunkCacheEncoded::new_chunk_limit(4)).await;
        test_cache_into_async(AsyncChunkCacheEncoded::new_chunk_limit(4)).await;
        test_cache_nested_optional_async(AsyncChunkCacheEncoded::new_chunk_limit(4)).await;
        test_cache_variable_async(AsyncChunkCacheEncoded::new_chunk_limit(9)).await;
        test_cache_variable_async(AsyncChunkCacheEncoded::new_size_limit(4096, None)).await;
    }

    #[tokio::test]
    async fn async_chunk_caches_support_decoded_values() {
        test_cache_async(Box::new(AsyncChunkCacheDecoded::new_chunk_limit(2))).await;
        test_cache_async(Box::new(AsyncChunkCacheDecoded::new_size_limit(1024, None))).await;
        test_cache_sharded_async(AsyncChunkCacheDecoded::new_chunk_limit(4)).await;
        test_cache_into_async(AsyncChunkCacheDecoded::new_chunk_limit(4)).await;
        test_cache_nested_optional_async(AsyncChunkCacheDecoded::new_chunk_limit(4)).await;
        test_cache_variable_async(AsyncChunkCacheDecoded::new_chunk_limit(9)).await;
        test_cache_variable_async(AsyncChunkCacheDecoded::new_size_limit(4096, None)).await;
    }

    #[tokio::test]
    async fn async_chunk_caches_support_async_partial_decoder_values() {
        test_cache_async(Box::new(AsyncChunkCachePartialDecoder::new_chunk_limit(2))).await;
        test_cache_async(Box::new(AsyncChunkCachePartialDecoder::new_size_limit(
            1024, None,
        )))
        .await;
        test_cache_sharded_async(AsyncChunkCachePartialDecoder::new_chunk_limit(4)).await;
        test_cache_into_async(AsyncChunkCachePartialDecoder::new_chunk_limit(4)).await;
        test_cache_nested_optional_async(AsyncChunkCachePartialDecoder::new_chunk_limit(4)).await;
        test_cache_variable_async(AsyncChunkCachePartialDecoder::new_chunk_limit(9)).await;
        test_cache_variable_async(AsyncChunkCachePartialDecoder::new_size_limit(4096, None)).await;
    }

    /// Cached async retrievals must be `Send` so that they can be used with
    /// executors that move tasks between threads (e.g. `tokio::spawn`).
    #[cfg(not(target_arch = "wasm32"))]
    async fn test_cache_spawn_async<C>(cache: C)
    where
        C: AsyncChunkCacheTraits + 'static,
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
    async fn async_chunk_caches_are_send() {
        test_cache_spawn_async(AsyncChunkCacheEncoded::new_chunk_limit(2)).await;
        test_cache_spawn_async(AsyncChunkCacheDecoded::new_chunk_limit(2)).await;
        test_cache_spawn_async(AsyncChunkCachePartialDecoder::new_chunk_limit(2)).await;
    }

    #[tokio::test]
    async fn dyn_async_array_cache() {
        use crate::array::cache::{ChunkCacheTypeDecoded, NoCache, SubchunkCache};

        let store = Arc::new(AsyncMemoryStore::new());
        let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint8(), 0u8)
            .subchunk_shape(vec![1, 1])
            .build_arc(store, "/")
            .unwrap();
        let data: Vec<u8> = (0..16).collect();
        array
            .async_store_array_subset(&array.subset_all(), data.as_slice(), &Resources::default())
            .await
            .unwrap();

        let cached: ArrayCached<
            AsyncMemoryStore,
            dyn AsyncChunkCacheTraits<Value = ChunkCacheTypeDecoded>,
        > = ArrayCached::new_boxed(
            array.clone(),
            Box::new(AsyncChunkCacheDecoded::new_chunk_limit(16)),
        );
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u8>>(&array.subset_all(), &Resources::default())
                .await
                .unwrap(),
            data
        );
        assert_eq!(cached.cache().len().await, 4);
        cached.async_invalidate().await;
        assert!(cached.cache().is_empty().await);

        test_cache_async::<dyn AsyncChunkCacheTraits<Value = ChunkCacheTypeDecoded>>(Box::new(
            AsyncChunkCacheDecoded::new_chunk_limit(16),
        ))
        .await;

        let dyn_subchunks: Box<dyn AsyncChunkCacheTraits<Value = ChunkCacheTypeDecoded>> =
            Box::new(AsyncChunkCacheDecoded::new_chunk_limit(16));
        let dyn_chunks: Box<dyn AsyncArrayCache> =
            Box::new(AsyncChunkCachePartialDecoder::new_chunk_limit(4));
        let subchunk_cache = SubchunkCache::new_boxed(0, dyn_subchunks, dyn_chunks);

        let caches: Vec<Box<dyn AsyncArrayCache>> = vec![
            Box::new(AsyncChunkCacheEncoded::new_chunk_limit(4)),
            Box::new(AsyncChunkCacheDecoded::new_chunk_limit(4)),
            Box::new(AsyncChunkCachePartialDecoder::new_chunk_limit(4)),
            Box::new(subchunk_cache),
            Box::new(NoCache),
        ];
        for cache in caches {
            let cached: ArrayCached<AsyncMemoryStore, dyn AsyncArrayCache> =
                ArrayCached::new_boxed(array.clone(), cache);
            assert_eq!(
                cached
                    .async_retrieve_array_subset::<Vec<u8>>(
                        &array.subset_all(),
                        &Resources::default()
                    )
                    .await
                    .unwrap(),
                data
            );
            cached
                .async_store_chunk(&[0, 0], &[0u8, 1, 4, 5], &Resources::default())
                .await
                .unwrap();
            assert_eq!(
                cached
                    .async_retrieve_chunk::<Vec<u8>>(&[0, 0], &Resources::default())
                    .await
                    .unwrap(),
                vec![0, 1, 4, 5]
            );

            // A write that bypasses the cache is read once its chunk is invalidated
            array
                .async_store_chunk(&[0, 0], &[9u8; 4], &Resources::default())
                .await
                .unwrap();
            cached.async_invalidate_chunk(&[0, 0]).await;
            assert_eq!(
                cached
                    .async_retrieve_chunk::<Vec<u8>>(&[0, 0], &Resources::default())
                    .await
                    .unwrap(),
                vec![9; 4]
            );
            cached
                .async_store_chunk(&[0, 0], &[0u8, 1, 4, 5], &Resources::default())
                .await
                .unwrap();
        }
    }
}
