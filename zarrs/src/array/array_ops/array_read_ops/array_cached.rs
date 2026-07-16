use inherent::inherent;
use std::sync::Arc;
use unsafe_cell_slice::UnsafeCellSlice;

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use super::super::*;
use super::ArrayReadOps;
use crate::array::array_bytes_internal::{
    merge_chunks_vlen, merge_chunks_vlen_optional, optional_nesting_depth,
};
use crate::array::chunk_cache::{ChunkCacheType, fill_value_bytes, retrieve_chunk_bytes};
use crate::array::concurrency::concurrency_chunks_and_codec;
use crate::array::{ArrayBytes, ArrayBytesFixedDisjointView, ArrayIndicesTinyVec};
use crate::iter_concurrent_limit;
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayPartialDecoderTraits, CodecError,
    decode_into_array_bytes_target,
};

#[allow(clippy::too_many_lines)]
fn retrieve_array_subset_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: ChunkCache + ?Sized,
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
                retrieve_chunk_bytes(cache, array, chunk_indices, options)
            } else {
                C::Value::retrieve_chunk_subset_bytes(
                    cache,
                    array,
                    chunk_indices,
                    &array_subset.relative_to(chunk_subset.start())?,
                    options,
                )
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
                retrieve_multi_chunk_fixed(
                    cache,
                    array,
                    array_subset,
                    &chunks,
                    chunk_concurrent_limit,
                    &options,
                )
            } else {
                retrieve_multi_chunk_variable(
                    cache,
                    array,
                    array_subset,
                    &chunks,
                    chunk_concurrent_limit,
                    &options,
                )
            }
        }
    }
}

fn retrieve_multi_chunk_variable<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &dyn ArraySubsetTraits,
    chunk_concurrent_limit: usize,
    options: &CodecOptions,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: ChunkCache + ?Sized,
{
    let indices = chunks.indices();
    let chunk_bytes_and_subsets =
        iter_concurrent_limit!(chunk_concurrent_limit, indices, map, |chunk_indices| {
            let chunk_subset = array.chunk_subset(&chunk_indices)?;
            let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
            let bytes = C::Value::retrieve_chunk_subset_bytes(
                cache,
                array,
                &chunk_indices,
                &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                options,
            )?;
            Ok((
                bytes,
                chunk_subset_overlap.relative_to(&array_subset.start())?,
            ))
        })
        .collect::<Result<Vec<_>, ArrayError>>()?;

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

fn retrieve_multi_chunk_fixed<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &dyn ArraySubsetTraits,
    chunk_concurrent_limit: usize,
    options: &CodecOptions,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: ChunkCache + ?Sized,
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
            let chunk_subset = array.chunk_subset(&chunk_indices)?;
            let overlap = chunk_subset.overlap(array_subset)?;
            let output_subset = overlap.relative_to(&array_subset_start)?;
            let bytes = C::Value::retrieve_chunk_subset_bytes(
                cache,
                array,
                &chunk_indices,
                &overlap.relative_to(chunk_subset.start())?,
                options,
            )?;
            let mut data_view = unsafe {
                ArrayBytesFixedDisjointView::new(
                    data_slice,
                    data_type_size,
                    &array_subset_shape,
                    output_subset.clone(),
                )?
            };
            let mut mask_view = mask_slice
                .map(|slice| unsafe {
                    ArrayBytesFixedDisjointView::new(
                        slice,
                        1,
                        &array_subset_shape,
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
        };
        iter_concurrent_limit!(
            chunk_concurrent_limit,
            chunks.indices(),
            try_for_each,
            retrieve_chunk
        )?;
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

#[inherent]
impl<TStorage, C> ArrayReadOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: ChunkCache,
{
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk<T: FromArrayBytes>(&self, chunk_indices: &[u64])
    -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        let bytes = retrieve_chunk_bytes(self.cache(), self.array(), chunk_indices, options)?;
        let shape = self.array().chunk_shape(chunk_indices)?;
        T::from_array_bytes_arc(
            bytes,
            bytemuck::must_cast_slice(&shape),
            self.array().data_type(),
        )
    }

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
    pub fn retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunks_opt<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_subset<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_subset_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        let bytes = C::Value::retrieve_chunk_subset_bytes(
            self.cache(),
            self.array(),
            chunk_indices,
            chunk_subset,
            options,
        )?;
        T::from_array_bytes_arc(bytes, &chunk_subset.shape(), self.array().data_type())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_subset_into(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let bytes = C::Value::retrieve_chunk_subset_bytes(
            self.cache(),
            self.array(),
            chunk_indices,
            chunk_subset,
            options,
        )?;
        decode_into_array_bytes_target(&bytes, output_target).map_err(ArrayError::CodecError)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_array_subset_opt<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        let bytes = retrieve_array_subset_bytes(self.cache(), self.array(), array_subset, options)?;
        T::from_array_bytes_arc(bytes, &array_subset.shape(), self.array().data_type())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<T>, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_if_exists_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<T>, ArrayError> {
        let Some(bytes) = C::Value::retrieve_chunk_bytes_if_exists(
            self.cache(),
            self.array(),
            chunk_indices,
            options,
        )?
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
    pub fn retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Vec<u8>>, StorageError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_encoded_chunk_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Vec<u8>>, StorageError> {
        self.array()
            .retrieve_encoded_chunk_opt(chunk_indices, options)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_encoded_chunks(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<Vec<Option<Vec<u8>>>, StorageError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_encoded_chunks_opt(
        &self,
        chunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<Vec<Option<Vec<u8>>>, StorageError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_subchunk_opt<T: FromArrayBytes>(
        &self,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_subchunk_at_level_opt<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_subchunks_opt<T: FromArrayBytes>(
        &self,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_subchunks_at_level_opt<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_array_subset_into_opt(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        super::common::retrieve_array_subset_into(
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

    #[allow(clippy::missing_errors_doc)]
    pub fn partial_decoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn partial_decoder_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        C::Value::partial_decoder(self.cache(), self.array(), chunk_indices, options)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;
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
    use zarrs_storage::store::MemoryStore;

    #[expect(clippy::single_range_in_vec_init)]
    fn test_cache<C>(cache: C)
    where
        C: ChunkCache,
    {
        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array.store_chunk(&[0], &[1u8, 2]).unwrap();

        let cached = ArrayCached::new(array, cache);
        assert_eq!(cached.retrieve_chunk::<Vec<u8>>(&[0]).unwrap(), vec![1, 2]);
        assert_eq!(
            cached
                .retrieve_chunk_subset::<Vec<u8>>(&[0], &[1..2])
                .unwrap(),
            vec![2]
        );
        assert_eq!(
            cached.retrieve_chunk_if_exists::<Vec<u8>>(&[1]).unwrap(),
            None
        );
        assert_eq!(
            cached
                .partial_decoder(&[0])
                .unwrap()
                .partial_decode(
                    &ArraySubset::new_with_ranges(&[0..1]),
                    &CodecOptions::default()
                )
                .unwrap(),
            vec![1].into()
        );
        assert_eq!(
            cached.retrieve_array_subset::<Vec<u8>>(&[1..3]).unwrap(),
            vec![2, 0]
        );
        assert!(matches!(
            cached
                .retrieve_subchunk_opt::<Vec<u8>>(&[0], &CodecOptions::default())
                .unwrap_err(),
            ArrayError::MissingSubchunkGrid
        ));
        assert!(matches!(
            cached
                .retrieve_subchunks_opt::<Vec<u8>>(&[0..2], &CodecOptions::default())
                .unwrap_err(),
            ArrayError::MissingSubchunkGrid
        ));
        assert!(
            cached
                .retrieve_subchunk_opt::<Vec<u8>>(&[0, 0], &CodecOptions::default())
                .is_err()
        );
        assert!(
            cached
                .retrieve_subchunks_opt::<Vec<u8>>(&[0..1, 0..1], &CodecOptions::default())
                .is_err()
        );
        assert!(cached.retrieve_chunk::<Vec<u8>>(&[2]).is_err());
        assert!(!cached.cache().is_empty());
        assert!(cached.cache().invalidate_chunk(&[0]));
        cached.cache().invalidate();
        assert!(cached.cache().is_empty());
    }

    #[expect(clippy::single_range_in_vec_init)]
    fn test_cache_sharded<C>(cache: C)
    where
        C: ChunkCache,
    {
        let store = Arc::new(MemoryStore::default());
        let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type::uint16(), 0u16);
        builder.subchunk_shape(vec![2, 2]);
        let array = builder.build_arc(store, "/").unwrap();
        let data: Vec<u16> = (0..64).collect();
        array
            .store_array_subset(&array.subset_all(), &data)
            .unwrap();

        let cached = ArrayCached::new(array, cache);
        let options = CodecOptions::default().with_concurrent_target(1);
        assert_eq!(
            cached
                .retrieve_subchunk_opt::<Vec<u16>>(&[2, 3], &options)
                .unwrap(),
            vec![38, 39, 46, 47]
        );
        assert_eq!(
            cached
                .retrieve_subchunks_opt::<Vec<u16>>(&[1..3, 1..3], &options)
                .unwrap(),
            vec![
                18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45,
            ]
        );
        assert!(
            cached
                .retrieve_subchunk_opt::<Vec<u16>>(&[0], &options)
                .is_err()
        );
        assert!(
            cached
                .retrieve_subchunks_opt::<Vec<u16>>(&[0..1], &options)
                .is_err()
        );
        assert!(!cached.cache().is_empty());
    }

    #[test]
    fn all_lru_policies_support_encoded_values() {
        test_cache(ChunkCacheEncodedLruChunkLimit::new(2));
        test_cache(ChunkCacheEncodedLruChunkLimitThreadLocal::new(2));
        test_cache(ChunkCacheEncodedLruSizeLimit::new(1024));
        test_cache(ChunkCacheEncodedLruSizeLimitThreadLocal::new(1024));
        test_cache_sharded(ChunkCacheEncodedLruChunkLimit::new(4));
        test_cache_sharded(ChunkCacheEncodedLruChunkLimitThreadLocal::new(4));
        test_cache_sharded(ChunkCacheEncodedLruSizeLimit::new(4096));
        test_cache_sharded(ChunkCacheEncodedLruSizeLimitThreadLocal::new(4096));
    }

    #[test]
    fn all_lru_policies_support_decoded_values() {
        test_cache(ChunkCacheDecodedLruChunkLimit::new(2));
        test_cache(ChunkCacheDecodedLruChunkLimitThreadLocal::new(2));
        test_cache(ChunkCacheDecodedLruSizeLimit::new(1024));
        test_cache(ChunkCacheDecodedLruSizeLimitThreadLocal::new(1024));
        test_cache_sharded(ChunkCacheDecodedLruChunkLimit::new(4));
        test_cache_sharded(ChunkCacheDecodedLruChunkLimitThreadLocal::new(4));
        test_cache_sharded(ChunkCacheDecodedLruSizeLimit::new(4096));
        test_cache_sharded(ChunkCacheDecodedLruSizeLimitThreadLocal::new(4096));
    }

    #[test]
    fn all_lru_policies_support_partial_decoder_values() {
        test_cache(ChunkCachePartialDecoderLruChunkLimit::new(2));
        test_cache(ChunkCachePartialDecoderLruChunkLimitThreadLocal::new(2));
        test_cache(ChunkCachePartialDecoderLruSizeLimit::new(1024));
        test_cache(ChunkCachePartialDecoderLruSizeLimitThreadLocal::new(1024));
        test_cache_sharded(ChunkCachePartialDecoderLruChunkLimit::new(4));
        test_cache_sharded(ChunkCachePartialDecoderLruChunkLimitThreadLocal::new(4));
        test_cache_sharded(ChunkCachePartialDecoderLruSizeLimit::new(4096));
        test_cache_sharded(ChunkCachePartialDecoderLruSizeLimitThreadLocal::new(4096));
    }

    #[derive(Default)]
    struct CustomDecodedCache {
        values: Mutex<HashMap<Vec<u64>, ChunkCacheTypeDecoded>>,
    }

    impl ChunkCache for CustomDecodedCache {
        type Value = ChunkCacheTypeDecoded;

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

    #[test]
    fn custom_policy_gets_reads_from_its_value_type() {
        test_cache(CustomDecodedCache::default());
        test_cache_sharded(CustomDecodedCache::default());
    }
}
