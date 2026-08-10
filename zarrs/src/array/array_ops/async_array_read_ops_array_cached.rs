use inherent::inherent;
use std::sync::Arc;
use unsafe_cell_slice::UnsafeCellSlice;

use futures::{StreamExt, TryStreamExt};

use super::async_array_read_ops_common::AsyncRetrieveInto;
use super::{AsyncArrayReadOps, *};
use crate::array::array_bytes_internal::{
    build_nested_optional_target, merge_cached_chunks_vlen, optional_nesting_depth,
    wrap_optional_masks,
};
use crate::array::chunk_cache::{
    AsyncChunkCache, AsyncChunkCacheType, async_retrieve_chunk_bytes, fill_value_bytes,
};
use crate::array::concurrency::concurrency_chunks_and_codec;
use crate::array::{ArrayBytes, ArrayBytesFixedDisjointView, ArrayIndicesTinyVec};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, AsyncArrayPartialDecoderTraits, decode_into_array_bytes_target,
};
use zarrs_storage::Bytes;

async fn async_retrieve_array_subset_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncChunkCache + ?Sized,
{
    if array_subset.dimensionality() != array.dimensionality() {
        return Err(ArrayError::InvalidArraySubset(
            array_subset.to_array_subset(),
            array.shape().to_vec(),
        ));
    }
    let Some(chunks) = array.chunks_in_array_subset(array_subset)? else {
        return Err(ArrayError::InvalidArraySubset(
            array_subset.to_array_subset(),
            array.shape().to_vec(),
        ));
    };
    match chunks.num_elements_usize() {
        0 => fill_value_bytes(array, array_subset.num_elements()),
        1 => {
            let chunk_indices = chunks.start();
            let chunk_subset = array.chunk_subset(chunk_indices)?;
            if chunk_subset == array_subset {
                async_retrieve_chunk_bytes(cache, array, chunk_indices, options).await
            } else {
                C::Value::async_retrieve_chunk_subset_bytes(
                    cache,
                    array,
                    chunk_indices,
                    &array_subset.relative_to(chunk_subset.start())?,
                    options,
                )
                .await
            }
        }
        num_chunks => {
            let chunk_shape = array.chunk_shape(chunks.start())?;
            let codec_concurrency = recommended_codec_concurrency(array, &chunk_shape)?;
            let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                options.concurrent_target(),
                num_chunks,
                options,
                &codec_concurrency,
            );
            if array.data_type().is_fixed() {
                async_retrieve_multi_chunk_fixed(
                    cache,
                    array,
                    array_subset,
                    &chunks,
                    chunk_concurrent_limit,
                    &options,
                )
                .await
            } else {
                async_retrieve_multi_chunk_variable(
                    cache,
                    array,
                    array_subset,
                    &chunks,
                    chunk_concurrent_limit,
                    &options,
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
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncChunkCache + ?Sized,
{
    let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| async move {
        let chunk_subset = array.chunk_subset(&chunk_indices)?;
        let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
        let bytes = C::Value::async_retrieve_chunk_subset_bytes(
            cache,
            array,
            &chunk_indices,
            &chunk_subset_overlap.relative_to(chunk_subset.start())?,
            options,
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

    Ok(merge_cached_chunks_vlen(
        chunk_bytes_and_subsets,
        &array_subset.shape(),
        array.data_type(),
    )?
    .into())
}

async fn async_retrieve_multi_chunk_fixed<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &dyn ArraySubsetTraits,
    chunk_concurrent_limit: usize,
    options: &CodecOptions,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: AsyncChunkCache + ?Sized,
{
    let data_type_size = array.data_type().fixed_size().expect("fixed data type");
    let num_elements = array_subset.num_elements_usize();
    let size_output = num_elements * data_type_size;
    let nesting_depth = optional_nesting_depth(array.data_type());
    if size_output == 0 {
        return Ok(
            wrap_optional_masks(ArrayBytes::new_flen(vec![]), vec![vec![]; nesting_depth]).into(),
        );
    }
    let mut data_output = Vec::with_capacity(size_output);
    let mut mask_outputs: Vec<Vec<u8>> = (0..nesting_depth)
        .map(|_| Vec::with_capacity(num_elements))
        .collect();

    {
        let data_slice = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut data_output);
        let mask_slices: Vec<_> = mask_outputs
            .iter_mut()
            .map(UnsafeCellSlice::new_from_vec_with_spare_capacity)
            .collect();
        let mask_slices = mask_slices.as_slice();
        let array_subset_start = array_subset.start();
        let array_subset_shape = array_subset.shape();
        let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
            let array_subset_start = &array_subset_start;
            let array_subset_shape = &array_subset_shape;
            async move {
                let chunk_subset = array.chunk_subset(&chunk_indices)?;
                let overlap = chunk_subset.overlap(array_subset)?;
                let output_subset = overlap.relative_to(array_subset_start)?;
                let bytes = C::Value::async_retrieve_chunk_subset_bytes(
                    cache,
                    array,
                    &chunk_indices,
                    &overlap.relative_to(chunk_subset.start())?,
                    options,
                )
                .await?;
                let mut data_view = unsafe {
                    ArrayBytesFixedDisjointView::new(
                        data_slice,
                        data_type_size,
                        array_subset_shape,
                        output_subset.clone(),
                    )?
                };
                let mut mask_views: Vec<ArrayBytesFixedDisjointView<'_>> = mask_slices
                    .iter()
                    .map(|mask_slice| unsafe {
                        ArrayBytesFixedDisjointView::new(
                            *mask_slice,
                            1,
                            array_subset_shape,
                            output_subset.clone(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let target =
                    build_nested_optional_target(&mut data_view, mask_views.as_mut_slice());
                decode_into_array_bytes_target(&bytes, target).map_err(ArrayError::CodecError)
            }
        };
        futures::stream::iter(&chunks.indices())
            .map(Ok)
            .try_for_each_concurrent(Some(chunk_concurrent_limit), retrieve_chunk)
            .await?;
    }

    unsafe { data_output.set_len(size_output) };
    for mask in &mut mask_outputs {
        unsafe { mask.set_len(num_elements) };
    }
    Ok(wrap_optional_masks(ArrayBytes::new_flen(data_output), mask_outputs).into())
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
    ) -> Result<(), ArrayError> {
        let bytes =
            async_retrieve_chunk_bytes(self.cache(), self.array(), chunk_indices, options).await?;
        decode_into_array_bytes_target(&bytes, output_target).map_err(ArrayError::CodecError)
    }

    pub(in crate::array) async fn async_retrieve_chunk_subset_into_with_options(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let bytes = C::Value::async_retrieve_chunk_subset_bytes(
            self.cache(),
            self.array(),
            chunk_indices,
            chunk_subset,
            options,
        )
        .await?;
        decode_into_array_bytes_target(&bytes, output_target).map_err(ArrayError::CodecError)
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
    ) -> Result<(), ArrayError> {
        self.async_retrieve_chunk_into_with_options(chunk_indices, output_target, options)
            .await
    }

    async fn retrieve_chunk_subset_into(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.async_retrieve_chunk_subset_into_with_options(
            chunk_indices,
            chunk_subset,
            output_target,
            options,
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
    ) -> Result<T, ArrayError> {
        let options = self.codec_options();
        let bytes =
            async_retrieve_chunk_bytes(self.cache(), self.array(), chunk_indices, options).await?;
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
    ) -> Result<(), ArrayError> {
        self.async_retrieve_chunk_into_with_options(
            chunk_indices,
            output_target,
            self.codec_options(),
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_subset<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        let bytes = C::Value::async_retrieve_chunk_subset_bytes(
            self.cache(),
            self.array(),
            chunk_indices,
            chunk_subset,
            self.codec_options(),
        )
        .await?;
        T::from_array_bytes_arc(bytes, &chunk_subset.shape(), self.array().data_type())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_subset_into(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError> {
        self.async_retrieve_chunk_subset_into_with_options(
            chunk_indices,
            chunk_subset,
            output_target,
            self.codec_options(),
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        let bytes = async_retrieve_array_subset_bytes(
            self.cache(),
            self.array(),
            array_subset,
            self.codec_options(),
        )
        .await?;
        T::from_array_bytes_arc(bytes, &array_subset.shape(), self.array().data_type())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<T>, ArrayError> {
        let options = self.codec_options();
        let Some(bytes) = C::Value::async_retrieve_chunk_bytes_if_exists(
            self.cache(),
            self.array(),
            chunk_indices,
            options,
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
    pub async fn async_retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Bytes>, StorageError> {
        self.array()
            .async_retrieve_encoded_chunk(chunk_indices)
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_encoded_chunks(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<Vec<Option<Bytes>>, StorageError> {
        self.array().async_retrieve_encoded_chunks(chunks).await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunk<T: FromArrayBytes>(
        &self,
        subchunk_indices: &[u64],
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunk_at_level<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunk_indices: &[u64],
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunks<T: FromArrayBytes>(
        &self,
        subchunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunks_at_level<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError> {
        super::async_array_read_ops_common::retrieve_array_subset_into(
            self.array().as_ref(),
            self,
            array_subset,
            output_target,
            self.codec_options(),
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_partial_decoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        C::Value::async_partial_decoder(
            self.cache(),
            self.array(),
            chunk_indices,
            self.codec_options(),
        )
        .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_local_subchunk_grid(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<ChunkGrid>, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_local_subchunk_grid_at_level(
        &self,
        level: usize,
        chunk_indices: &[u64],
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
    use crate::array::{ArrayBuilder, FillValue, data_type};
    use zarrs_storage::store::AsyncMemoryStore;

    #[expect(clippy::single_range_in_vec_init)]
    async fn test_cache_async<C>(cache: C)
    where
        C: AsyncChunkCache + 'static,
    {
        let store = Arc::new(AsyncMemoryStore::new());
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array.async_store_chunk(&[0], &[1u8, 2]).await.unwrap();

        let cached = ArrayCached::new(array, cache);
        assert_eq!(
            cached.async_retrieve_chunk::<Vec<u8>>(&[0]).await.unwrap(),
            vec![1, 2]
        );
        assert_eq!(
            cached
                .async_retrieve_chunk_subset::<Vec<u8>>(&[0], &[1..2])
                .await
                .unwrap(),
            vec![2]
        );
        assert_eq!(
            cached
                .async_retrieve_chunk_if_exists::<Vec<u8>>(&[1])
                .await
                .unwrap(),
            None
        );
        assert_eq!(
            cached
                .async_partial_decoder(&[0])
                .await
                .unwrap()
                .partial_decode(
                    &ArraySubset::new_with_ranges(&[0..1]),
                    &CodecOptions::default()
                )
                .await
                .unwrap(),
            vec![1].into()
        );
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u8>>(&[1..3])
                .await
                .unwrap(),
            vec![2, 0]
        );
        assert!(cached.async_retrieve_chunk::<Vec<u8>>(&[2]).await.is_err());
        assert!(!cached.cache().is_empty().await);

        // Write operations invalidate affected cached chunks
        cached.async_store_chunk(&[0], &[3u8, 4]).await.unwrap();
        assert_eq!(
            cached.async_retrieve_chunk::<Vec<u8>>(&[0]).await.unwrap(),
            vec![3, 4]
        );
        cached
            .async_store_array_subset(&[0..1], &[5u8])
            .await
            .unwrap();
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<u8>>(&[0..4])
                .await
                .unwrap(),
            vec![5, 4, 0, 0]
        );
        cached.async_erase_chunk(&[0]).await.unwrap();
        assert_eq!(
            cached.async_retrieve_chunk::<Vec<u8>>(&[0]).await.unwrap(),
            vec![0, 0]
        );

        cached.cache().invalidate().await;
        assert!(cached.cache().is_empty().await);
    }

    #[expect(clippy::single_range_in_vec_init)]
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
            .async_store_array_subset(&array.subset_all(), &data)
            .await
            .unwrap();

        // Exercise the cached reads under a non-default concurrency target. The derived
        // array shares the cache, so the assertions below still observe it.
        let cached = ArrayCached::new(array, cache)
            .with_codec_options(CodecOptions::default().with_concurrent_target(1));
        assert_eq!(
            cached
                .async_retrieve_subchunk::<Vec<u16>>(&[2, 3])
                .await
                .unwrap(),
            vec![38, 39, 46, 47]
        );
        assert_eq!(
            cached
                .async_retrieve_subchunks::<Vec<u16>>(&[1..3, 1..3])
                .await
                .unwrap(),
            vec![
                18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45,
            ]
        );
        assert!(
            cached
                .async_retrieve_subchunk::<Vec<u16>>(&[0])
                .await
                .is_err()
        );
        assert!(
            cached
                .async_retrieve_subchunks::<Vec<u16>>(&[0..1])
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
            .async_store_array_subset(&array.subset_all(), &(0..16u8).collect::<Vec<u8>>())
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
                )
                .await
                .unwrap();
        }
        assert_eq!(buf, vec![5, 6, 9, 10]);
        assert!(!cached.cache().is_empty().await);
    }

    /// Retrieving a subset spanning multiple chunks must handle nested optional data types.
    #[expect(clippy::single_range_in_vec_init)]
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
            .async_store_chunk(&[0], &[Some(Some(1u8)), Some(None)])
            .await
            .unwrap();
        array
            .async_store_chunk(&[1], &[None, Some(Some(4u8))])
            .await
            .unwrap();
        // chunk 2 is unwritten, so it decodes to the fill value (`None`)

        let cached = ArrayCached::new(array, cache);
        // Spans all three chunks
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<Option<Option<u8>>>>(&[0..6])
                .await
                .unwrap(),
            vec![Some(Some(1)), Some(None), None, Some(Some(4)), None, None]
        );
        // Partial overlap of two chunks
        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<Option<Option<u8>>>>(&[1..4])
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
    }

    #[tokio::test]
    async fn async_lru_caches_support_decoded_values() {
        test_cache_async(AsyncChunkCacheDecodedLruChunkLimit::new(2)).await;
        test_cache_async(AsyncChunkCacheDecodedLruSizeLimit::new(1024)).await;
        test_cache_sharded_async(AsyncChunkCacheDecodedLruChunkLimit::new(4)).await;
        test_cache_into_async(AsyncChunkCacheDecodedLruChunkLimit::new(4)).await;
        test_cache_nested_optional_async(AsyncChunkCacheDecodedLruChunkLimit::new(4)).await;
    }

    #[tokio::test]
    async fn async_lru_caches_support_async_partial_decoder_values() {
        test_cache_async(AsyncChunkCachePartialDecoderLruChunkLimit::new(2)).await;
        test_cache_async(AsyncChunkCachePartialDecoderLruSizeLimit::new(1024)).await;
        test_cache_sharded_async(AsyncChunkCachePartialDecoderLruChunkLimit::new(4)).await;
        test_cache_into_async(AsyncChunkCachePartialDecoderLruChunkLimit::new(4)).await;
        test_cache_nested_optional_async(AsyncChunkCachePartialDecoderLruChunkLimit::new(4)).await;
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
        array.async_store_chunk(&[0], &[1u8, 2]).await.unwrap();

        let cached = Arc::new(ArrayCached::new(array, cache));
        let handles = (0..4).map(|_| {
            let cached = cached.clone();
            tokio::spawn(async move { cached.async_retrieve_chunk::<Vec<u8>>(&[0]).await.unwrap() })
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
