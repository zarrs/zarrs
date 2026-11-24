use inherent::inherent;
use std::sync::Arc;
use zarrs_codec::Resources;

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::ParallelIterator;

use super::array_read_ops_common::retrieve_multi_chunk_fixed_into;
use super::{
    Array, ArrayCached, ArrayError, ArrayReadOps, ArraySubset, ArraySubsetTraits, COrderBytes,
    ChunkGrid, CodecOptions, FromArrayBytes, Indexer, ReadableStorageTraits,
    recommended_codec_concurrency,
};
use crate::IntoConcurrentLimitIterator;
use crate::array::array_bytes_internal::{
    FixedDecodeBuffers, merge_cached_chunks, optional_innermost_with_depth,
};
use crate::array::cache::{
    SyncArrayCache, retrieve_chunk_overlap_bytes, retrieve_chunk_overlap_into,
};
use crate::array::concurrency::concurrency_chunks_and_codec;
use crate::array::{ArrayBytes, DataType};
use zarrs_codec::{ArrayBytesTarget, ArrayPartialDecoderTraits};
use zarrs_storage::Bytes;

/// Retrieve `array_subset`, which spans `chunks`, with elements in C order.
///
/// `chunks` are the chunks intersecting `array_subset` (see [`array_subset_chunks`](super::array_subset_chunks)).
fn retrieve_array_subset_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    chunks: &ArraySubset,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: SyncArrayCache + ?Sized,
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
            retrieve_chunk_overlap_bytes(
                cache,
                array,
                chunk_indices,
                &chunk_subset,
                array_subset,
                options,
                resources,
            )
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
                retrieve_multi_chunk_fixed(
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
            } else {
                retrieve_multi_chunk_variable(
                    cache,
                    array,
                    array_subset,
                    chunks,
                    chunk_concurrent_limit,
                    options,
                    &resources,
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
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: SyncArrayCache + ?Sized,
{
    let indices = chunks.indices();
    let chunk_bytes_and_subsets = indices
        .concurrent_limit(chunk_concurrent_limit)
        .map(|chunk_indices| {
            let chunk_subset = array.chunk_subset(&chunk_indices)?;
            let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
            let bytes = retrieve_chunk_overlap_bytes(
                cache,
                array,
                &chunk_indices,
                &chunk_subset,
                &chunk_subset_overlap,
                options,
                resources,
            )?;
            Ok((
                bytes,
                chunk_subset_overlap.relative_to(&array_subset.start())?,
            ))
        })
        .collect::<Result<Vec<_>, ArrayError>>()?;

    Ok(merge_cached_chunks(
        chunk_bytes_and_subsets,
        &array_subset.shape(),
        array.data_type(),
    )?
    .into())
}

#[expect(clippy::too_many_arguments)]
fn retrieve_multi_chunk_fixed<TStorage, C>(
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
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: SyncArrayCache + ?Sized,
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
        options,
        resources,
        &|chunk_indices, chunk_subset, overlap, target, options, resources| {
            retrieve_chunk_overlap_into(
                cache,
                array,
                chunk_indices,
                chunk_subset,
                overlap,
                target,
                options,
                resources,
            )
        },
    )?;
    // SAFETY: decoding covers the array subset and satisfies the target's output requirements
    Ok(unsafe { buffers.into_array_bytes() }.into())
}

impl<TStorage, C> ArrayCached<TStorage, C>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: SyncArrayCache + ?Sized,
{
    pub(in crate::array) fn retrieve_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.cache().cached_chunk_into(
            self.array().as_ref(),
            chunk_indices,
            output_target,
            options,
            resources,
        )
    }

    pub(in crate::array) fn retrieve_partial_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.cache().cached_partial_chunk_into(
            self.array().as_ref(),
            chunk_indices,
            indexer,
            output_target,
            options,
            resources,
        )
    }
}

#[inherent]
impl<TStorage, C> ArrayReadOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: SyncArrayCache + ?Sized,
{
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let options = self.codec_options();
        let chunk_shape = self.array().chunk_shape(chunk_indices)?;
        super::retrieve_in_order(
            self.array().data_type(),
            self.element_order(),
            bytemuck::must_cast_slice(&chunk_shape),
            |target| {
                self.retrieve_chunk_into_with_options(chunk_indices, target, options, resources)
            },
            || {
                self.cache().cached_chunk_bytes(
                    self.array().as_ref(),
                    chunk_indices,
                    options,
                    resources,
                )
            },
        )
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.retrieve_chunk_into_with_options(
            chunk_indices,
            output_target,
            self.codec_options(),
            resources,
        )
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunks<T: FromArrayBytes>(
        &self,
        chunks: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_partial_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let options = self.codec_options();
        let chunk_shape = self.array().chunk_shape(chunk_indices)?;
        super::validate_partial_chunk_subset(indexer, bytemuck::must_cast_slice(&chunk_shape))?;
        super::retrieve_in_order(
            self.array().data_type(),
            super::indexer_element_order(indexer, self.element_order()),
            &indexer.output_shape(),
            |target| {
                self.retrieve_partial_chunk_into_with_options(
                    chunk_indices,
                    indexer,
                    target,
                    options,
                    resources,
                )
            },
            || {
                self.cache().cached_partial_chunk_bytes(
                    self.array().as_ref(),
                    chunk_indices,
                    indexer,
                    options,
                    resources,
                )
            },
        )
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_partial_chunk_into(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.retrieve_partial_chunk_into_with_options(
            chunk_indices,
            indexer,
            output_target,
            self.codec_options(),
            resources,
        )
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let options = self.codec_options();
        let chunks = super::array_subset_chunks(self, array_subset)?;
        super::retrieve_in_order(
            self.array().data_type(),
            self.element_order(),
            &array_subset.shape(),
            |target| self.retrieve_array_subset_into(array_subset, target, resources),
            || {
                retrieve_array_subset_bytes(
                    self.cache(),
                    self.array(),
                    array_subset,
                    &chunks,
                    options,
                    resources,
                )
            },
        )
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<T>, ArrayError> {
        let options = self.codec_options();
        let Some(bytes) = self.cache().cached_chunk_bytes_if_exists(
            self.array().as_ref(),
            chunk_indices,
            options,
            resources,
        )?
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
    pub fn retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Bytes>, ArrayError> {
        self.array().retrieve_encoded_chunk(chunk_indices)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_encoded_chunks(
        &self,
        chunks: &dyn Indexer,
        resources: &Resources,
    ) -> Result<Vec<Option<Bytes>>, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_subchunk<T: FromArrayBytes>(
        &self,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_subchunk_at_level<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_subchunks<T: FromArrayBytes>(
        &self,
        subchunks: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_subchunks_at_level<T: FromArrayBytes>(
        &self,
        level: usize,
        subchunks: &dyn ArraySubsetTraits,
        resources: &Resources,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: &mut ArrayBytesTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        super::array_read_ops_common::retrieve_array_subset_into(
            self,
            array_subset,
            output_target,
            self.array().chunk_concurrent_minimum,
            self.codec_options(),
            resources,
            |chunk_indices, output_target, options, resources| {
                self.retrieve_chunk_into_with_options(
                    chunk_indices,
                    output_target,
                    options,
                    resources,
                )
            },
            |chunk_indices, chunk_subset, output_target, options, resources| {
                self.retrieve_partial_chunk_into_with_options(
                    chunk_indices,
                    chunk_subset,
                    output_target,
                    options,
                    resources,
                )
            },
        )
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn partial_decoder(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        let options = self.codec_options();
        self.cache().cached_partial_decoder(
            self.array().as_ref(),
            chunk_indices,
            options,
            resources,
        )
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn local_subchunk_grid(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<ChunkGrid>, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn local_subchunk_grid_at_level(
        &self,
        level: usize,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<ChunkGrid>, ArrayError>;
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::*;
    use crate::array::cache::{
        ChunkCacheDecoded, ChunkCacheEncoded, ChunkCachePartialDecoder, ChunkCacheType,
        ChunkCacheTypeDecoded, SyncChunkCacheTraits,
    };
    use crate::array::{ArrayBuilder, ArrayError, ArrayIndices, ArraySubset, FillValue, data_type};
    use zarrs_storage::store::MemoryStore;

    fn test_cache<C>(cache: Box<C>)
    where
        C: SyncChunkCacheTraits + ?Sized,
    {
        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array
            .store_chunk(&[0], &[1u8, 2], &Resources::default())
            .unwrap();

        let cached = ArrayCached::new_boxed(array, cache);
        assert_eq!(
            cached
                .retrieve_chunk::<Vec<u8>>(&[0], &Resources::default())
                .unwrap(),
            vec![1, 2]
        );
        assert_eq!(
            cached
                .retrieve_partial_chunk::<Vec<u8>>(&[0], &[1..2], &Resources::default())
                .unwrap(),
            vec![2]
        );
        assert_eq!(
            cached
                .retrieve_chunk_if_exists::<Vec<u8>>(&[1], &Resources::default())
                .unwrap(),
            None
        );
        assert_eq!(
            cached
                .partial_decoder(&[0], &Resources::default())
                .unwrap()
                .partial_decode(
                    &ArraySubset::new_with_ranges(&[0..1]),
                    &CodecOptions::default(),
                    &Resources::default(),
                )
                .unwrap(),
            vec![1].into()
        );
        assert_eq!(
            cached
                .retrieve_array_subset::<Vec<u8>>(&[1..3], &Resources::default())
                .unwrap(),
            vec![2, 0]
        );
        // Complete chunks, in a single chunk and across multiple chunks
        assert_eq!(
            cached
                .retrieve_array_subset::<Vec<u8>>(&[0..2], &Resources::default())
                .unwrap(),
            vec![1, 2]
        );
        assert_eq!(
            cached
                .retrieve_array_subset::<Vec<u8>>(&[0..4], &Resources::default())
                .unwrap(),
            vec![1, 2, 0, 0]
        );
        assert!(matches!(
            cached
                .retrieve_subchunk::<Vec<u8>>(&[0], &Resources::default())
                .unwrap_err(),
            ArrayError::MissingSubchunkGrid
        ));
        assert!(matches!(
            cached
                .retrieve_subchunks::<Vec<u8>>(&[0..2], &Resources::default())
                .unwrap_err(),
            ArrayError::MissingSubchunkGrid
        ));
        assert!(
            cached
                .retrieve_subchunk::<Vec<u8>>(&[0, 0], &Resources::default())
                .is_err()
        );
        assert!(
            cached
                .retrieve_subchunks::<Vec<u8>>(&[0..1, 0..1], &Resources::default())
                .is_err()
        );
        assert!(
            cached
                .retrieve_chunk::<Vec<u8>>(&[2], &Resources::default())
                .is_err()
        );
        assert!(!cached.cache().is_empty());
        assert!(cached.cache().invalidate_chunk(&[0]));
        cached.cache().invalidate();
        assert!(cached.cache().is_empty());
    }

    fn test_cache_sharded<C>(cache: C)
    where
        C: SyncChunkCacheTraits,
    {
        let store = Arc::new(MemoryStore::default());
        let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type::uint16(), 0u16);
        builder.subchunk_shape(vec![2, 2]);
        let array = builder.build_arc(store, "/").unwrap();
        let data: Vec<u16> = (0..64).collect();
        array
            .store_array_subset(&array.subset_all(), &data, &Resources::default())
            .unwrap();

        // Exercise the cached reads under a non-default concurrency target.
        let cached = ArrayCached::new(array, cache);
        let resources = Resources::default().with_compute_target(1);
        assert_eq!(
            cached
                .retrieve_subchunk::<Vec<u16>>(&[2, 3], &resources)
                .unwrap(),
            vec![38, 39, 46, 47]
        );
        assert_eq!(
            cached
                .retrieve_subchunks::<Vec<u16>>(&[1..3, 1..3], &resources)
                .unwrap(),
            vec![
                18, 19, 20, 21, 26, 27, 28, 29, 34, 35, 36, 37, 42, 43, 44, 45,
            ]
        );
        assert!(
            cached
                .retrieve_subchunk::<Vec<u16>>(&[0], &resources)
                .is_err()
        );
        assert!(
            cached
                .retrieve_subchunks::<Vec<u16>>(&[0..1], &resources)
                .is_err()
        );
        assert!(!cached.cache().is_empty());
    }

    /// Merge complete and partial variable-length chunks, including missing chunks.
    fn test_cache_variable<C>(cache: C)
    where
        C: SyncChunkCacheTraits,
    {
        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(vec![6, 6], vec![2, 2], data_type::string(), "")
            .build_arc(store, "/")
            .unwrap();
        let data: Vec<String> = (0..36)
            .map(|i| if i < 24 { i.to_string() } else { String::new() })
            .collect();
        array
            .store_array_subset(&[0..4, 0..6], &data[..24], &Resources::default())
            .unwrap();

        let cached = ArrayCached::new(array, cache);
        // Repeat the reads to exercise both cold and populated caches.
        for _ in 0..2 {
            assert_eq!(
                cached
                    .retrieve_array_subset::<Vec<String>>(&[0..6, 0..6], &Resources::default())
                    .unwrap(),
                data
            );
            let expected: Vec<String> = (1..6)
                .flat_map(|row| data[row * 6 + 1..row * 6 + 6].iter().cloned())
                .collect();
            assert_eq!(
                cached
                    .retrieve_array_subset::<Vec<String>>(&[1..6, 1..6], &Resources::default())
                    .unwrap(),
                expected
            );
        }
    }

    /// Retrieving a subset spanning multiple chunks must handle nested optional data types.
    fn test_cache_nested_optional<C>(cache: C)
    where
        C: SyncChunkCacheTraits,
    {
        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(
            vec![6],
            vec![2],
            data_type::uint8().to_optional().to_optional(),
            FillValue::from(None::<Option<u8>>),
        )
        .build_arc(store, "/")
        .unwrap();
        array
            .store_chunk(&[0], &[Some(Some(1u8)), Some(None)], &Resources::default())
            .unwrap();
        array
            .store_chunk(&[1], &[None, Some(Some(4u8))], &Resources::default())
            .unwrap();
        // chunk 2 is unwritten, so it decodes to the fill value (`None`)

        let cached = ArrayCached::new(array, cache);
        // Spans all three chunks
        assert_eq!(
            cached
                .retrieve_array_subset::<Vec<Option<Option<u8>>>>(&[0..6], &Resources::default())
                .unwrap(),
            vec![Some(Some(1)), Some(None), None, Some(Some(4)), None, None]
        );
        // Partial overlap of two chunks
        assert_eq!(
            cached
                .retrieve_array_subset::<Vec<Option<Option<u8>>>>(&[1..4], &Resources::default())
                .unwrap(),
            vec![Some(None), None, Some(Some(4))]
        );
    }

    #[test]
    fn chunk_caches_support_encoded_values() {
        test_cache(Box::new(ChunkCacheEncoded::new_chunk_limit(2)));
        test_cache(Box::new(ChunkCacheEncoded::new_size_limit(1024, None)));
        test_cache_sharded(ChunkCacheEncoded::new_chunk_limit(4));
        test_cache_sharded(ChunkCacheEncoded::new_size_limit(4096, None));
        test_cache_nested_optional(ChunkCacheEncoded::new_chunk_limit(4));
        test_cache_nested_optional(ChunkCacheEncoded::new_size_limit(4096, None));
        test_cache_variable(ChunkCacheEncoded::new_chunk_limit(9));
        test_cache_variable(ChunkCacheEncoded::new_size_limit(4096, None));
    }

    #[test]
    fn chunk_caches_support_decoded_values() {
        test_cache(Box::new(ChunkCacheDecoded::new_chunk_limit(2)));
        test_cache(Box::new(ChunkCacheDecoded::new_size_limit(1024, None)));
        test_cache_sharded(ChunkCacheDecoded::new_chunk_limit(4));
        test_cache_sharded(ChunkCacheDecoded::new_size_limit(4096, None));
        test_cache_nested_optional(ChunkCacheDecoded::new_chunk_limit(4));
        test_cache_nested_optional(ChunkCacheDecoded::new_size_limit(4096, None));
        test_cache_variable(ChunkCacheDecoded::new_chunk_limit(9));
        test_cache_variable(ChunkCacheDecoded::new_size_limit(4096, None));
    }

    #[test]
    fn chunk_caches_support_partial_decoder_values() {
        test_cache(Box::new(ChunkCachePartialDecoder::new_chunk_limit(2)));
        test_cache(Box::new(ChunkCachePartialDecoder::new_size_limit(
            1024, None,
        )));
        test_cache_sharded(ChunkCachePartialDecoder::new_chunk_limit(4));
        test_cache_sharded(ChunkCachePartialDecoder::new_size_limit(4096, None));
        test_cache_nested_optional(ChunkCachePartialDecoder::new_chunk_limit(4));
        test_cache_nested_optional(ChunkCachePartialDecoder::new_size_limit(4096, None));
        test_cache_variable(ChunkCachePartialDecoder::new_chunk_limit(9));
        test_cache_variable(ChunkCachePartialDecoder::new_size_limit(4096, None));
    }

    #[derive(Default)]
    struct CustomDecodedCache {
        values: Mutex<HashMap<ArrayIndices, ChunkCacheTypeDecoded>>,
    }

    impl SyncChunkCacheTraits for CustomDecodedCache {
        type Value = ChunkCacheTypeDecoded;

        fn try_get_or_insert_with(
            &self,
            chunk_indices: ArrayIndices,
            f: &mut dyn FnMut() -> Result<Self::Value, ArrayError>,
        ) -> Result<Self::Value, Arc<ArrayError>> {
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

        fn size(&self) -> usize {
            self.values
                .lock()
                .unwrap()
                .values()
                .map(ChunkCacheType::size)
                .sum()
        }
    }

    #[test]
    fn custom_policy_gets_reads_from_its_value_type() {
        test_cache(Box::new(CustomDecodedCache::default()));
        test_cache_sharded(CustomDecodedCache::default());
    }

    #[test]
    fn dyn_chunk_cache_and_sync_array_cache() {
        use crate::array::cache::{NoCache, SubchunkCache};

        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint8(), 0u8)
            .subchunk_shape(vec![1, 1])
            .build_arc(store, "/")
            .unwrap();
        let data: Vec<u8> = (0..16).collect();
        array
            .store_array_subset(&array.subset_all(), data.as_slice(), &Resources::default())
            .unwrap();

        // dyn SyncChunkCacheTraits<Value = ChunkCacheTypeDecoded>
        let cached: ArrayCached<
            MemoryStore,
            dyn SyncChunkCacheTraits<Value = ChunkCacheTypeDecoded>,
        > = ArrayCached::new_boxed(
            array.clone(),
            Box::new(ChunkCacheDecoded::new_chunk_limit(4)),
        );
        assert_eq!(
            cached
                .retrieve_array_subset::<Vec<u8>>(&array.subset_all(), &Resources::default())
                .unwrap(),
            data
        );
        assert_eq!(cached.cache().len(), 4);
        cached.invalidate();
        assert!(cached.cache().is_empty());

        test_cache::<dyn SyncChunkCacheTraits<Value = ChunkCacheTypeDecoded>>(Box::new(
            ChunkCacheDecoded::new_chunk_limit(4),
        ));

        // SubchunkCache over dyn SyncChunkCacheTraits and dyn SyncArrayCache
        let dyn_subchunks: Box<dyn SyncChunkCacheTraits<Value = ChunkCacheTypeDecoded>> =
            Box::new(ChunkCacheDecoded::new_chunk_limit(4));
        let dyn_chunks: Box<dyn SyncArrayCache> =
            Box::new(ChunkCachePartialDecoder::new_chunk_limit(4));
        let subchunk_cache = SubchunkCache::new_boxed(0, dyn_subchunks, dyn_chunks);

        // Heterogeneous caches behind ArrayCached<MemoryStore, dyn SyncArrayCache>
        let caches: Vec<Box<dyn SyncArrayCache>> = vec![
            Box::new(ChunkCacheEncoded::new_chunk_limit(4)),
            Box::new(ChunkCacheDecoded::new_chunk_limit(4)),
            Box::new(ChunkCachePartialDecoder::new_chunk_limit(4)),
            Box::new(subchunk_cache),
            Box::new(NoCache),
        ];
        for cache in caches {
            let cached: ArrayCached<MemoryStore, dyn SyncArrayCache> =
                ArrayCached::new_boxed(array.clone(), cache);
            assert_eq!(
                cached
                    .retrieve_array_subset::<Vec<u8>>(&array.subset_all(), &Resources::default())
                    .unwrap(),
                data
            );
            cached
                .store_chunk(&[0, 0], &[0u8, 1, 4, 5], &Resources::default())
                .unwrap();
            assert_eq!(
                cached
                    .retrieve_chunk::<Vec<u8>>(&[0, 0], &Resources::default())
                    .unwrap(),
                vec![0, 1, 4, 5]
            );

            // A write that bypasses the cache is read once its chunk is invalidated
            array
                .store_chunk(&[0, 0], &[9u8; 4], &Resources::default())
                .unwrap();
            cached.invalidate_chunk(&[0, 0]);
            assert_eq!(
                cached
                    .retrieve_chunk::<Vec<u8>>(&[0, 0], &Resources::default())
                    .unwrap(),
                vec![9; 4]
            );
            cached
                .store_chunk(&[0, 0], &[0u8, 1, 4, 5], &Resources::default())
                .unwrap();
        }
    }
}
