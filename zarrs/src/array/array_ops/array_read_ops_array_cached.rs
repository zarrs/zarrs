use inherent::inherent;
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::ParallelIterator;

use super::{ArrayReadOps, *};
use crate::IntoConcurrentLimitIterator;
use crate::array::array_bytes_internal::merge_cached_chunks;
use crate::array::chunk_cache::{
    Sealed, SealedSync, fill_value_bytes, retrieve_chunk_bytes, retrieve_chunk_overlap_bytes,
};
use crate::array::{ArrayBytes, ArrayIndicesTinyVec};
use zarrs_codec::plan::SharedOutput;
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayPartialDecoderTraits, CodecError,
    InvalidNumberOfElementsError, copy_array_subset_into_target, copy_fill_value_into,
};
use zarrs_storage::Bytes;

fn retrieve_array_subset_bytes<TStorage, C>(
    cache: &C,
    array: &Array<TStorage>,
    array_subset: &dyn ArraySubsetTraits,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<Arc<ArrayBytes<'static>>, ArrayError>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: ChunkCache + ?Sized,
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
            let (chunk_concurrent_limit, resources) = split_chunk_resources(num_chunks, resources);
            if array.data_type().is_fixed() {
                retrieve_multi_chunk_fixed(
                    cache,
                    array,
                    array_subset,
                    &chunks,
                    chunk_concurrent_limit,
                    options,
                    &resources,
                )
            } else {
                retrieve_multi_chunk_variable(
                    cache,
                    array,
                    array_subset,
                    &chunks,
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
    C: ChunkCache + ?Sized,
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

/// Copy the elements of `indexer` of a decoded chunk of `chunk_shape` (or the fill value if it does not exist) into
/// `target`.
pub(super) fn copy_cached_chunk_into<TStorage: ?Sized>(
    array: &Array<TStorage>,
    chunk: Option<&ArrayBytes<'_>>,
    chunk_shape: &[u64],
    indexer: &dyn Indexer,
    target: ArrayBytesDecodeIntoTarget<'_>,
) -> Result<(), CodecError> {
    if indexer.len() != target.num_elements() {
        return Err(InvalidNumberOfElementsError::new(indexer.len(), target.num_elements()).into());
    }
    if let Some(chunk) = chunk {
        copy_array_subset_into_target(chunk, chunk_shape, indexer, array.data_type(), target)
    } else {
        indexer.validate(chunk_shape)?;
        copy_fill_value_into(array.data_type(), array.fill_value(), target)
    }
}

/// The source of the elements of a chunk in a cached read.
enum ChunkSource {
    /// The decoded chunk, or [`None`] if it does not exist.
    Chunk(Option<Arc<ArrayBytes<'static>>>),
    /// A partial decoder of the chunk.
    Decoder(Arc<dyn ArrayPartialDecoderTraits>),
}

impl ChunkSource {
    fn retrieve<TStorage, C>(
        cache: &C,
        array: &Array<TStorage>,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Self, ArrayError>
    where
        TStorage: ?Sized + ReadableStorageTraits + 'static,
        C: ChunkCache + ?Sized,
    {
        Ok(if C::Value::HOLDS_DECODED {
            // The elements are copied from the decoded chunk, rather than extracted then copied
            Self::Chunk(C::Value::retrieve_chunk_bytes_if_exists(
                cache,
                array,
                chunk_indices,
                options,
                resources,
            )?)
        } else {
            // The elements are decoded into the output, rather than decoded then copied
            Self::Decoder(C::Value::partial_decoder(
                cache,
                array,
                chunk_indices,
                options,
                resources,
            )?)
        })
    }

    /// Decode the elements of `indexer` of the chunk of `chunk_shape` into `target`.
    fn decode_into<TStorage: ?Sized>(
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
                decoder.partial_decode_into(indexer, target, options, resources)
            }
        }
    }
}

fn retrieve_multi_chunk_fixed<TStorage, C>(
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
    C: ChunkCache + ?Sized,
{
    let (output, views) = SharedOutput::for_data_type(array.data_type(), &array_subset.shape())
        .expect("fixed data type");
    let array_subset_start = array_subset.start();
    let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
        let chunk_subset = array.chunk_subset(&chunk_indices)?;
        let overlap = chunk_subset.overlap(array_subset)?;
        let output_subset = overlap.relative_to(&array_subset_start)?;
        let chunk_overlap = overlap.relative_to(chunk_subset.start())?;
        let source = ChunkSource::retrieve(cache, array, &chunk_indices, options, resources)?;
        // SAFETY: the chunks are disjoint
        unsafe {
            views.with_subdivided(output_subset, |views| {
                views.with_target(|target| {
                    source.decode_into(
                        array,
                        chunk_subset.shape(),
                        &chunk_overlap,
                        target,
                        options,
                        resources,
                    )
                })
            })
        }??;
        Ok::<_, ArrayError>(())
    };
    chunks
        .indices()
        .concurrent_limit(chunk_concurrent_limit)
        .try_for_each(retrieve_chunk)?;
    drop(views);
    // SAFETY: the chunks cover the array subset
    let bytes = unsafe { output.into_array_bytes() }
        .map_err(|_| CodecError::Other("an output view outlived its chunk".to_string()))?;
    Ok(bytes.into())
}

impl<TStorage, C> ArrayCached<TStorage, C>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: ChunkCache,
{
    pub(in crate::array) fn retrieve_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.array().chunk_shape(chunk_indices)?;
        self.retrieve_partial_chunk_into_with_options(
            chunk_indices,
            &ArraySubset::new_with_shape(bytemuck::must_cast_slice::<_, u64>(&chunk_shape)),
            output_target,
            options,
            resources,
        )
    }

    pub(in crate::array) fn retrieve_partial_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let source = ChunkSource::retrieve(
            self.cache(),
            self.array(),
            chunk_indices,
            options,
            resources,
        )?;
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
            .map_err(ArrayError::CodecError)
    }
}

#[inherent]
impl<TStorage, C> ArrayReadOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + ReadableStorageTraits + 'static,
    C: ChunkCache,
{
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let options = self.codec_options();
        let bytes = retrieve_chunk_bytes(
            self.cache(),
            self.array(),
            chunk_indices,
            options,
            resources,
        )?;
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
        let bytes = C::Value::retrieve_partial_chunk_bytes(
            self.cache(),
            self.array(),
            chunk_indices,
            indexer,
            options,
            resources,
        )?;
        T::from_array_bytes_arc(bytes, &indexer.output_shape(), self.array().data_type())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_partial_chunk_into(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
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
        let bytes = retrieve_array_subset_bytes(
            self.cache(),
            self.array(),
            array_subset,
            options,
            resources,
        )?;
        T::from_array_bytes_arc(bytes, &array_subset.shape(), self.array().data_type())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<T>, ArrayError> {
        let options = self.codec_options();
        let Some(bytes) = C::Value::retrieve_chunk_bytes_if_exists(
            self.cache(),
            self.array(),
            chunk_indices,
            options,
            resources,
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
    ) -> Result<Option<Bytes>, ArrayError> {
        self.array().retrieve_encoded_chunk(chunk_indices)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_encoded_subchunk(
        &self,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<EncodedSubchunk>, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_encoded_subchunk_at_level(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<EncodedSubchunk>, ArrayError> {
        self.array()
            .retrieve_encoded_subchunk_at_level(level, subchunk_indices, resources)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_encoded_chunks(
        &self,
        chunks: &dyn Indexer,
        resources: &Resources,
    ) -> Result<Vec<Option<Bytes>>, ArrayError> {
        self.array().retrieve_encoded_chunks(chunks, resources)
    }

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
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        super::array_read_ops_common::retrieve_array_subset_into(
            self,
            array_subset,
            output_target,
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
        C::Value::partial_decoder(
            self.cache(),
            self.array(),
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
    use crate::array::chunk_cache::{
        ChunkCacheDecodedLruChunkLimit, ChunkCacheDecodedLruChunkLimitThreadLocal,
        ChunkCacheDecodedLruSizeLimit, ChunkCacheDecodedLruSizeLimitThreadLocal,
        ChunkCacheEncodedLruChunkLimit, ChunkCacheEncodedLruChunkLimitThreadLocal,
        ChunkCacheEncodedLruSizeLimit, ChunkCacheEncodedLruSizeLimitThreadLocal,
        ChunkCachePartialDecoderLruChunkLimit, ChunkCachePartialDecoderLruChunkLimitThreadLocal,
        ChunkCachePartialDecoderLruSizeLimit, ChunkCachePartialDecoderLruSizeLimitThreadLocal,
        ChunkCacheTypeDecoded,
    };
    use crate::array::{ArrayBuilder, ArrayError, FillValue, data_type};
    use zarrs_storage::store::MemoryStore;

    fn test_cache<C>(cache: C)
    where
        C: ChunkCache,
    {
        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(vec![4], vec![2], data_type::uint8(), 0u8)
            .build_arc(store, "/")
            .unwrap();
        array
            .store_chunk(&[0], &[1u8, 2], &Resources::default())
            .unwrap();

        let cached = ArrayCached::new(array, cache);
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
        C: ChunkCache,
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
        C: ChunkCache,
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
        C: ChunkCache,
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
    fn all_lru_policies_support_encoded_values() {
        test_cache(ChunkCacheEncodedLruChunkLimit::new(2));
        test_cache(ChunkCacheEncodedLruChunkLimitThreadLocal::new(2));
        test_cache(ChunkCacheEncodedLruSizeLimit::new(1024));
        test_cache(ChunkCacheEncodedLruSizeLimitThreadLocal::new(1024));
        test_cache_sharded(ChunkCacheEncodedLruChunkLimit::new(4));
        test_cache_sharded(ChunkCacheEncodedLruChunkLimitThreadLocal::new(4));
        test_cache_sharded(ChunkCacheEncodedLruSizeLimit::new(4096));
        test_cache_sharded(ChunkCacheEncodedLruSizeLimitThreadLocal::new(4096));
        test_cache_nested_optional(ChunkCacheEncodedLruChunkLimit::new(4));
        test_cache_nested_optional(ChunkCacheEncodedLruSizeLimitThreadLocal::new(4096));
        test_cache_variable(ChunkCacheEncodedLruChunkLimit::new(9));
        test_cache_variable(ChunkCacheEncodedLruSizeLimitThreadLocal::new(4096));
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
        test_cache_nested_optional(ChunkCacheDecodedLruChunkLimit::new(4));
        test_cache_nested_optional(ChunkCacheDecodedLruSizeLimitThreadLocal::new(4096));
        test_cache_variable(ChunkCacheDecodedLruChunkLimit::new(9));
        test_cache_variable(ChunkCacheDecodedLruSizeLimitThreadLocal::new(4096));
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
        test_cache_nested_optional(ChunkCachePartialDecoderLruChunkLimit::new(4));
        test_cache_nested_optional(ChunkCachePartialDecoderLruSizeLimitThreadLocal::new(4096));
        test_cache_variable(ChunkCachePartialDecoderLruChunkLimit::new(9));
        test_cache_variable(ChunkCachePartialDecoderLruSizeLimitThreadLocal::new(4096));
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
