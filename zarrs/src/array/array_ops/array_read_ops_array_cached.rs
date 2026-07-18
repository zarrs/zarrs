use ambisync::ambisync;
use inherent::inherent;
use std::sync::Arc;
use unsafe_cell_slice::UnsafeCellSlice;

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[cfg(feature = "async")]
use futures::{StreamExt, TryStreamExt};

#[cfg(feature = "async")]
use super::AsyncArrayReadOps;
use super::{ArrayReadOps, *};
#[cfg(feature = "async")]
use crate::array::array_bytes_internal::{build_nested_optional_target, extract_target_views};
use crate::array::array_bytes_internal::{
    merge_chunks_vlen, merge_chunks_vlen_optional, optional_nesting_depth,
};
#[cfg(feature = "async")]
use crate::array::chunk_cache::{AsyncChunkCacheType, async_retrieve_chunk_bytes};
use crate::array::chunk_cache::{SyncChunkCacheType, fill_value_bytes, retrieve_chunk_bytes};
use crate::array::concurrency::concurrency_chunks_and_codec;
use crate::array::{ArrayBytes, ArrayBytesFixedDisjointView, ArrayIndicesTinyVec};
use crate::iter_concurrent_limit;
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayPartialDecoderTraits, CodecError,
    decode_into_array_bytes_target,
};
#[cfg(feature = "async")]
use zarrs_codec::{
    AsyncArrayPartialDecoderTraits, InvalidNumberOfElementsError, copy_fill_value_into,
};
#[cfg(feature = "async")]
use zarrs_storage::Bytes;

#[ambisync(
    sync(
        fns("async_{}"),
        types(
            AsyncChunkCacheType => SyncChunkCacheType,
            AsyncReadableStorageTraits => ReadableStorageTraits,
        ),
    ),
    async(feature = "async"),
)]
#[allow(clippy::too_many_lines)]
async fn async_retrieve_array_subset_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: ChunkCache + ?Sized,
    C::Value: AsyncChunkCacheType,
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
    let chunk_shape0 = array.chunk_shape(&vec![0; array.dimensionality()])?;
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
            let codec_concurrency = array.recommended_codec_concurrency(&chunk_shape0)?;
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

#[ambisync(
    sync(
        fns("async_{}"),
        types(
            AsyncChunkCacheType => SyncChunkCacheType,
            AsyncReadableStorageTraits => ReadableStorageTraits,
        ),
    ),
    async(feature = "async"),
)]
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
    C: ChunkCache + ?Sized,
    C::Value: AsyncChunkCacheType,
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

    let chunk_bytes_and_subsets: Vec<_> = ambisync::alt!(
        sync => iter_concurrent_limit!(
            chunk_concurrent_limit,
            chunks.indices(),
            map,
            retrieve_chunk
        )
        .collect::<Result<Vec<_>, ArrayError>>()?,
        async => futures::stream::iter(chunks.indices().iter())
            .map(|chunk_indices| retrieve_chunk(chunk_indices.clone()))
            .buffered(chunk_concurrent_limit)
            .try_collect()
            .await?,
    );

    let nesting_depth = optional_nesting_depth(array.data_type());
    if nesting_depth > 0 {
        let chunks = chunk_bytes_and_subsets
            .iter()
            .map(|(bytes, subset)| {
                (
                    ArrayBytes::clone(bytes)
                        .into_optional()
                        .expect("run on vlen data"),
                    subset.clone(),
                )
            })
            .collect();
        Ok(ArrayBytes::Optional(merge_chunks_vlen_optional(
            chunks,
            &array_subset.shape(),
            nesting_depth,
        )?)
        .into())
    } else {
        let chunks = chunk_bytes_and_subsets
            .iter()
            .map(|(bytes, subset)| {
                (
                    ArrayBytes::clone(bytes)
                        .into_variable()
                        .expect("run on vlen data"),
                    subset.clone(),
                )
            })
            .collect();
        Ok(ArrayBytes::Variable(merge_chunks_vlen(chunks, &array_subset.shape())).into())
    }
}

#[ambisync(
    sync(
        fns("async_{}"),
        types(
            AsyncChunkCacheType => SyncChunkCacheType,
            AsyncReadableStorageTraits => ReadableStorageTraits,
        ),
    ),
    async(feature = "async"),
)]
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
    C: ChunkCache + ?Sized,
    C::Value: AsyncChunkCacheType,
{
    let data_type_size = array.data_type().fixed_size().expect("fixed data type");
    let num_elements = array_subset.num_elements_usize();
    let size_output = num_elements * data_type_size;
    if size_output == 0 {
        return Ok(ArrayBytes::new_flen(vec![]).into());
    }
    let mut data_output = Vec::with_capacity(size_output);
    let mut mask_output = array
        .data_type()
        .is_optional()
        .then(|| Vec::with_capacity(num_elements));

    {
        let data_slice = UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut data_output);
        let mask_slice = mask_output
            .as_mut()
            .map(UnsafeCellSlice::new_from_vec_with_spare_capacity);
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
                let mut mask_view = mask_slice
                    .map(|slice| unsafe {
                        ArrayBytesFixedDisjointView::new(
                            slice,
                            1,
                            array_subset_shape,
                            output_subset.clone(),
                        )
                    })
                    .transpose()?;
                match bytes.as_ref() {
                    ArrayBytes::Fixed(bytes) => {
                        data_view.copy_from_slice(bytes).map_err(CodecError::from)?;
                    }
                    ArrayBytes::Optional(bytes) => {
                        let ArrayBytes::Fixed(data) = bytes.data() else {
                            unreachable!("optional fixed data contains fixed bytes");
                        };
                        data_view.copy_from_slice(data).map_err(CodecError::from)?;
                        if let Some(mask_view) = &mut mask_view {
                            mask_view
                                .copy_from_slice(bytes.mask())
                                .map_err(CodecError::from)?;
                        }
                    }
                    ArrayBytes::Variable(_) => unreachable!("fixed data contains fixed bytes"),
                }
                Ok::<_, ArrayError>(())
            }
        };
        ambisync::alt!(
            sync => iter_concurrent_limit!(
                chunk_concurrent_limit,
                chunks.indices(),
                try_for_each,
                retrieve_chunk
            )?,
            async => futures::stream::iter(&chunks.indices())
                .map(Ok)
                .try_for_each_concurrent(Some(chunk_concurrent_limit), retrieve_chunk)
                .await?,
        );
    }

    unsafe { data_output.set_len(size_output) };
    if let Some(mask) = &mut mask_output {
        unsafe { mask.set_len(num_elements) };
    }
    let bytes = ArrayBytes::from(data_output);
    Ok(if let Some(mask) = mask_output {
        bytes.with_optional_mask(mask).into()
    } else {
        bytes.into()
    })
}

#[cfg(feature = "async")]
async fn async_retrieve_multi_chunk_fixed_into<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &dyn ArraySubsetTraits,
    chunk_concurrent_limit: usize,
    output_target: &ArrayBytesDecodeIntoTarget<'_>,
    options: &CodecOptions,
) -> Result<(), ArrayError>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: ChunkCache + ?Sized,
    C::Value: AsyncChunkCacheType,
{
    let (data_view_ref, mask_view_refs) = extract_target_views(output_target);
    let parent_start = data_view_ref.subset().start().to_vec();
    let array_subset_start = array_subset.start();

    let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
        let array_subset_start = &array_subset_start;
        let parent_start = &parent_start;
        let mask_view_refs = &mask_view_refs;
        async move {
            let chunk_subset = array.chunk_subset(&chunk_indices)?;
            let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
            let chunk_subset_in_array = chunk_subset_overlap.relative_to(array_subset_start)?;

            let chunk_start_in_view: Vec<u64> = chunk_subset_in_array
                .start()
                .iter()
                .zip(parent_start.iter())
                .map(|(&c, &p)| c + p)
                .collect();
            let chunk_subset_in_view = ArraySubset::new_with_start_shape(
                chunk_start_in_view,
                chunk_subset_in_array.shape().to_vec(),
            )?;

            let mut data_sub = unsafe {
                // SAFETY: chunks represent disjoint array subsets
                data_view_ref.subdivide(chunk_subset_in_view.clone())?
            };

            let mut mask_subs: Vec<ArrayBytesFixedDisjointView<'_>> = mask_view_refs
                .iter()
                .map(|mask_view| unsafe {
                    // SAFETY: chunks represent disjoint array subsets
                    mask_view.subdivide(chunk_subset_in_view.clone())
                })
                .collect::<Result<Vec<_>, _>>()?;

            let target = build_nested_optional_target(&mut data_sub, mask_subs.as_mut_slice());

            let bytes = C::Value::async_retrieve_chunk_subset_bytes(
                cache,
                array,
                &chunk_indices,
                &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                options,
            )
            .await?;
            decode_into_array_bytes_target(&bytes, target).map_err(ArrayError::CodecError)
        }
    };

    futures::stream::iter(&chunks.indices())
        .map(Ok)
        .try_for_each_concurrent(Some(chunk_concurrent_limit), retrieve_chunk)
        .await?;

    Ok(())
}

#[ambisync(
    sync(
        fns("async_{}"),
        types(
            AsyncArrayReadOps => ArrayReadOps,
            AsyncChunkCacheType => SyncChunkCacheType,
            AsyncReadableStorageTraits => ReadableStorageTraits,
            AsyncArrayPartialDecoderTraits => ArrayPartialDecoderTraits,
            Bytes => Vec<u8>,
        ),
    ),
    async(feature = "async"),
)]
#[inherent]
impl<TStorage, C> AsyncArrayReadOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + AsyncReadableStorageTraits + 'static,
    C: ChunkCache,
    C::Value: AsyncChunkCacheType,
{
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<T, ArrayError> {
        self.async_retrieve_chunk_opt(chunk_indices, self.codec_options())
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        let bytes =
            async_retrieve_chunk_bytes(self.cache(), self.array(), chunk_indices, options).await?;
        let shape = self.array().chunk_shape(chunk_indices)?;
        T::from_array_bytes_arc(
            bytes,
            bytemuck::must_cast_slice(&shape),
            self.array().data_type(),
        )
    }

    #[sync_only]
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let bytes = retrieve_chunk_bytes(self.cache(), self.array(), chunk_indices, options)?;
        decode_into_array_bytes_target(&bytes, output_target).map_err(ArrayError::CodecError)
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        self.async_retrieve_chunks_opt(chunks, self.codec_options())
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunks_opt<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_subset<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        self.async_retrieve_chunk_subset_opt(chunk_indices, chunk_subset, self.codec_options())
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_subset_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        let bytes = C::Value::async_retrieve_chunk_subset_bytes(
            self.cache(),
            self.array(),
            chunk_indices,
            chunk_subset,
            options,
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

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError> {
        self.async_retrieve_array_subset_opt(array_subset, self.codec_options())
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset_opt<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        let bytes =
            async_retrieve_array_subset_bytes(self.cache(), self.array(), array_subset, options)
                .await?;
        T::from_array_bytes_arc(bytes, &array_subset.shape(), self.array().data_type())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<T>, ArrayError> {
        self.async_retrieve_chunk_if_exists_opt(chunk_indices, self.codec_options())
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_chunk_if_exists_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<T>, ArrayError> {
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
    ) -> Result<Option<Bytes>, StorageError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_encoded_chunk_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Bytes>, StorageError> {
        self.array()
            .async_retrieve_encoded_chunk_opt(chunk_indices, options)
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_encoded_chunks(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<Vec<Option<Bytes>>, StorageError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_encoded_chunks_opt(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<Option<Bytes>>, StorageError> {
        self.array()
            .async_retrieve_encoded_chunks_opt(chunks, options)
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunk_opt<T: FromArrayBytes>(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunk_at_level_opt<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunks_opt<T: FromArrayBytes>(
        &self,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_subchunks_at_level_opt<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError> {
        self.async_retrieve_array_subset_into_opt(array_subset, output_target, self.codec_options())
            .await
    }

    #[sync_only]
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_array_subset_into_opt(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        super::array_read_ops_common::retrieve_array_subset_into(
            self.array(),
            array_subset,
            output_target,
            options,
            |chunk_indices, output_target, options| {
                self.retrieve_chunk_into(chunk_indices, output_target, options)
            },
            |chunk_indices, chunk_subset, output_target, options| {
                self.retrieve_chunk_subset_into(chunk_indices, chunk_subset, output_target, options)
            },
        )
    }

    #[async_only]
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_retrieve_array_subset_into_opt(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let array = self.array();
        if array_subset.dimensionality() != array.dimensionality() {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.to_array_subset(),
                array.shape().to_vec(),
            ));
        }

        if !array.data_type().is_fixed() {
            return Err(ArrayError::CodecError(CodecError::Other(
                "retrieve_array_subset_into does not support variable-length data types"
                    .to_string(),
            )));
        }

        if output_target.num_elements() != array_subset.num_elements() {
            return Err(ArrayError::CodecError(
                InvalidNumberOfElementsError::new(
                    output_target.num_elements(),
                    array_subset.num_elements(),
                )
                .into(),
            ));
        }

        let Some(chunks) = array.chunks_in_array_subset(array_subset)? else {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.to_array_subset(),
                array.shape().to_vec(),
            ));
        };

        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => copy_fill_value_into(array.data_type(), array.fill_value(), output_target)
                .map_err(ArrayError::CodecError),
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = array.chunk_subset(chunk_indices)?;
                if chunk_subset == array_subset {
                    let bytes =
                        async_retrieve_chunk_bytes(self.cache(), array, chunk_indices, options)
                            .await?;
                    decode_into_array_bytes_target(&bytes, output_target)
                        .map_err(ArrayError::CodecError)
                } else {
                    self.async_retrieve_chunk_subset_into(
                        chunk_indices,
                        &array_subset.relative_to(chunk_subset.start())?,
                        output_target,
                        options,
                    )
                    .await
                }
            }
            _ => {
                let chunk_shape = array.chunk_shape(chunks.start())?;
                let codec_concurrency = array.recommended_codec_concurrency(&chunk_shape)?;
                let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                    options.concurrent_target(),
                    num_chunks,
                    options,
                    &codec_concurrency,
                );
                async_retrieve_multi_chunk_fixed_into(
                    self.cache(),
                    array,
                    array_subset,
                    &chunks,
                    chunk_concurrent_limit,
                    &output_target,
                    &options,
                )
                .await
            }
        }
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_partial_decoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        self.async_partial_decoder_opt(chunk_indices, self.codec_options())
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_partial_decoder_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        C::Value::async_partial_decoder(self.cache(), self.array(), chunk_indices, options).await
    }
}

#[cfg(test)]
#[ambisync::scope(
    sync(
        fns("async_{}", "{}_async"),
        types(AsyncChunkCacheType => SyncChunkCacheType),
    ),
    async(feature = "async"),
)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;
    #[cfg(feature = "async")]
    use crate::array::chunk_cache::{
        ChunkCacheAsyncPartialDecoderLruChunkLimit,
        ChunkCacheAsyncPartialDecoderLruChunkLimitThreadLocal,
        ChunkCacheAsyncPartialDecoderLruSizeLimit,
        ChunkCacheAsyncPartialDecoderLruSizeLimitThreadLocal,
    };
    use crate::array::chunk_cache::{
        ChunkCacheDecodedLruChunkLimit, ChunkCacheDecodedLruChunkLimitThreadLocal,
        ChunkCacheDecodedLruSizeLimit, ChunkCacheDecodedLruSizeLimitThreadLocal,
        ChunkCacheEncodedLruChunkLimit, ChunkCacheEncodedLruChunkLimitThreadLocal,
        ChunkCacheEncodedLruSizeLimit, ChunkCacheEncodedLruSizeLimitThreadLocal,
        ChunkCachePartialDecoderLruChunkLimit, ChunkCachePartialDecoderLruChunkLimitThreadLocal,
        ChunkCachePartialDecoderLruSizeLimit, ChunkCachePartialDecoderLruSizeLimitThreadLocal,
        ChunkCacheTypeDecoded,
    };
    use crate::array::{ArrayBuilder, ArrayError, data_type};
    #[cfg(feature = "async")]
    use object_store::memory::InMemory;
    #[cfg(feature = "async")]
    use zarrs_object_store::AsyncObjectStore;
    use zarrs_storage::store::MemoryStore;

    #[expect(clippy::single_range_in_vec_init)]
    #[ambisync]
    async fn test_cache_async<C>(cache: C)
    where
        C: ChunkCache + 'static,
        C::Value: AsyncChunkCacheType,
    {
        let store = ambisync::alt!(
            sync => Arc::new(MemoryStore::default()),
            async => Arc::new(AsyncObjectStore::new(InMemory::new())),
        );
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
        assert!(matches!(
            cached
                .async_retrieve_subchunk_opt::<Vec<u8>>(&[0], &CodecOptions::default())
                .await
                .unwrap_err(),
            ArrayError::MissingSubchunkGrid
        ));
        assert!(matches!(
            cached
                .async_retrieve_subchunks_opt::<Vec<u8>>(&[0..2], &CodecOptions::default())
                .await
                .unwrap_err(),
            ArrayError::MissingSubchunkGrid
        ));
        assert!(
            cached
                .async_retrieve_subchunk_opt::<Vec<u8>>(&[0, 0], &CodecOptions::default())
                .await
                .is_err()
        );
        assert!(
            cached
                .async_retrieve_subchunks_opt::<Vec<u8>>(&[0..1, 0..1], &CodecOptions::default())
                .await
                .is_err()
        );
        assert!(cached.async_retrieve_chunk::<Vec<u8>>(&[2]).await.is_err());
        assert!(!cached.cache().is_empty());

        ambisync::alt!(
            sync => {},
            async => {
                // Async write operations invalidate affected cached chunks.
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
            },
        );

        assert!(cached.cache().invalidate_chunk(&[0]));
        cached.cache().invalidate();
        assert!(cached.cache().is_empty());
    }

    #[expect(clippy::single_range_in_vec_init)]
    #[ambisync]
    async fn test_cache_sharded_async<C>(cache: C)
    where
        C: ChunkCache + 'static,
        C::Value: AsyncChunkCacheType,
    {
        let store = ambisync::alt!(
            sync => Arc::new(MemoryStore::default()),
            async => Arc::new(AsyncObjectStore::new(InMemory::new())),
        );
        let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type::uint16(), 0u16);
        builder.subchunk_shape(vec![2, 2]);
        let array = builder.build_arc(store, "/").unwrap();
        let data: Vec<u16> = (0..64).collect();
        array
            .async_store_array_subset(&array.subset_all(), &data)
            .await
            .unwrap();

        let cached = ArrayCached::new(array, cache);
        let options = CodecOptions::default().with_concurrent_target(1);
        assert_eq!(
            cached
                .async_retrieve_subchunk_opt::<Vec<u16>>(&[2, 3], &options)
                .await
                .unwrap(),
            vec![38, 39, 46, 47]
        );
        assert_eq!(
            cached
                .async_retrieve_subchunks_opt::<Vec<u16>>(&[1..3, 1..3], &options)
                .await
                .unwrap(),
            vec![
                18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45,
            ]
        );
        assert!(
            cached
                .async_retrieve_subchunk_opt::<Vec<u16>>(&[0], &options)
                .await
                .is_err()
        );
        assert!(
            cached
                .async_retrieve_subchunks_opt::<Vec<u16>>(&[0..1], &options)
                .await
                .is_err()
        );
        assert!(!cached.cache().is_empty());
    }

    #[ambisync]
    async fn test_cache_into_async<C>(cache: C)
    where
        C: ChunkCache + 'static,
        C::Value: AsyncChunkCacheType,
    {
        let store = ambisync::alt!(
            sync => Arc::new(MemoryStore::default()),
            async => Arc::new(AsyncObjectStore::new(InMemory::new())),
        );
        let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array
            .async_store_array_subset(&array.subset_all(), (0..16u8).collect::<Vec<u8>>())
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
        assert!(!cached.cache().is_empty());
    }

    #[ambisync::test(sync(name = "all_lru_policies_support_encoded_values"))]
    async fn async_lru_caches_support_encoded_values() {
        test_cache_async(ChunkCacheEncodedLruChunkLimit::new(2)).await;
        test_cache_async(ChunkCacheEncodedLruChunkLimitThreadLocal::new(2)).await;
        test_cache_async(ChunkCacheEncodedLruSizeLimit::new(1024)).await;
        test_cache_async(ChunkCacheEncodedLruSizeLimitThreadLocal::new(1024)).await;
        test_cache_sharded_async(ChunkCacheEncodedLruChunkLimit::new(4)).await;
        test_cache_sharded_async(ChunkCacheEncodedLruChunkLimitThreadLocal::new(4)).await;
        test_cache_sharded_async(ChunkCacheEncodedLruSizeLimit::new(4096)).await;
        test_cache_sharded_async(ChunkCacheEncodedLruSizeLimitThreadLocal::new(4096)).await;
        test_cache_into_async(ChunkCacheEncodedLruChunkLimit::new(4)).await;
    }

    #[ambisync::test(sync(name = "all_lru_policies_support_decoded_values"))]
    async fn async_lru_caches_support_decoded_values() {
        test_cache_async(ChunkCacheDecodedLruChunkLimit::new(2)).await;
        test_cache_async(ChunkCacheDecodedLruChunkLimitThreadLocal::new(2)).await;
        test_cache_async(ChunkCacheDecodedLruSizeLimit::new(1024)).await;
        test_cache_async(ChunkCacheDecodedLruSizeLimitThreadLocal::new(1024)).await;
        test_cache_sharded_async(ChunkCacheDecodedLruChunkLimit::new(4)).await;
        test_cache_sharded_async(ChunkCacheDecodedLruChunkLimitThreadLocal::new(4)).await;
        test_cache_sharded_async(ChunkCacheDecodedLruSizeLimit::new(4096)).await;
        test_cache_sharded_async(ChunkCacheDecodedLruSizeLimitThreadLocal::new(4096)).await;
        test_cache_into_async(ChunkCacheDecodedLruChunkLimit::new(4)).await;
    }

    #[ambisync::test(sync(
        name = "all_lru_policies_support_partial_decoder_values",
        types(
            ChunkCacheAsyncPartialDecoderLruChunkLimit => ChunkCachePartialDecoderLruChunkLimit,
            ChunkCacheAsyncPartialDecoderLruChunkLimitThreadLocal => ChunkCachePartialDecoderLruChunkLimitThreadLocal,
            ChunkCacheAsyncPartialDecoderLruSizeLimit => ChunkCachePartialDecoderLruSizeLimit,
            ChunkCacheAsyncPartialDecoderLruSizeLimitThreadLocal => ChunkCachePartialDecoderLruSizeLimitThreadLocal,
        ),
    ))]
    async fn async_lru_caches_support_async_partial_decoder_values() {
        test_cache_async(ChunkCacheAsyncPartialDecoderLruChunkLimit::new(2)).await;
        test_cache_async(ChunkCacheAsyncPartialDecoderLruChunkLimitThreadLocal::new(
            2,
        ))
        .await;
        test_cache_async(ChunkCacheAsyncPartialDecoderLruSizeLimit::new(1024)).await;
        test_cache_async(ChunkCacheAsyncPartialDecoderLruSizeLimitThreadLocal::new(
            1024,
        ))
        .await;
        test_cache_sharded_async(ChunkCacheAsyncPartialDecoderLruChunkLimit::new(4)).await;
        test_cache_sharded_async(ChunkCacheAsyncPartialDecoderLruChunkLimitThreadLocal::new(
            4,
        ))
        .await;
        test_cache_sharded_async(ChunkCacheAsyncPartialDecoderLruSizeLimit::new(4096)).await;
        test_cache_sharded_async(ChunkCacheAsyncPartialDecoderLruSizeLimitThreadLocal::new(
            4096,
        ))
        .await;
        test_cache_into_async(ChunkCacheAsyncPartialDecoderLruChunkLimit::new(4)).await;
    }

    #[derive(Default)]
    struct CustomDecodedCache {
        values: Mutex<HashMap<Vec<u64>, ChunkCacheTypeDecoded>>,
    }

    impl ChunkCache for CustomDecodedCache {
        type Value = ChunkCacheTypeDecoded;

        fn get(&self, chunk_indices: &[u64]) -> Option<Self::Value> {
            self.values.lock().unwrap().get(chunk_indices).cloned()
        }

        fn try_get_or_insert_with<F>(
            &self,
            chunk_indices: Vec<u64>,
            f: F,
        ) -> Result<Self::Value, Arc<ArrayError>>
        where
            F: FnOnce() -> Result<Self::Value, ArrayError>,
        {
            let mut values = self.values.lock().unwrap();
            if let Some(value) = values.get(&chunk_indices) {
                return Ok(value.clone());
            }
            let value = f().map_err(Arc::new)?;
            values.insert(chunk_indices, value.clone());
            Ok(value)
        }

        fn invalidate(&self) -> usize {
            let mut values = self.values.lock().unwrap();
            let len = values.len();
            values.clear();
            len
        }

        fn invalidate_chunk(&self, chunk_indices: &[u64]) -> bool {
            self.values.lock().unwrap().remove(chunk_indices).is_some()
        }

        fn len(&self) -> usize {
            self.values.lock().unwrap().len()
        }
    }

    #[ambisync::test(sync(name = "custom_policy_gets_reads_from_its_value_type"))]
    async fn async_custom_policy_gets_reads_from_its_value_type() {
        test_cache_async(CustomDecodedCache::default()).await;
        test_cache_sharded_async(CustomDecodedCache::default()).await;
    }
}
