//! Decoded subchunks of a [`SubchunkCache`].

#[cfg(feature = "async")]
use std::future::Future;
use std::sync::{Arc, OnceLock};

#[cfg(feature = "async")]
use futures::{StreamExt, TryStreamExt};
#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::ParallelIterator;

#[cfg(feature = "async")]
use super::super::array_cache::array_cache_sealed::AsyncArrayCacheOps;
use super::super::array_cache::array_cache_sealed::SyncArrayCacheOps;
use super::super::chunk_cache_type::{cache_error, validate_chunk_grid_indices};
use super::{Generation, SubchunkCache};
use crate::IntoConcurrentLimitIterator;
use crate::array::array_bytes_internal::{
    FixedDecodeBuffers, merge_cached_chunks, optional_innermost, optional_innermost_with_depth,
};
#[cfg(feature = "async")]
use crate::array::chunk_cache::{AsyncArrayCache, AsyncCachedArray, AsyncChunkCache};
use crate::array::chunk_cache::{
    CachedArrayMeta, ChunkCache, ChunkCacheTypeDecoded, SyncArrayCache, SyncCachedArray,
};
use crate::array::concurrency::split_resources;
use crate::array::{
    ArrayBytes, ArrayError, ArrayIndices, ArrayShape, ArraySubset, ArraySubsetTraits, ChunkGrid,
    CodecOptions, DataType, FillValue, Indexer,
};
use zarrs_codec::{
    ArrayBytesTarget, ArrayPartialDecoderSubchunkingTraits, ArrayPartialDecoderTraits, CodecError,
    RecommendedConcurrency, Resources, copy_fill_value_into, decode_into_array_bytes_target,
};
#[cfg(feature = "async")]
use zarrs_codec::{AsyncArrayPartialDecoderSubchunkingTraits, AsyncArrayPartialDecoderTraits};
#[cfg(feature = "async")]
use zarrs_storage::MaybeSend;
use zarrs_storage::{MaybeSync, StorageError};

/// A (synchronous or asynchronous) partial decoder `D` of a chunk, opened at most once and only if needed.
///
/// It is opened at most once only if it is not opened concurrently, which [`ChunkSubchunks::assemble`] ensures.
struct LazyDecoder<D: ?Sized, F> {
    open: F,
    decoder: OnceLock<Arc<D>>,
}

impl<D: ?Sized, F> LazyDecoder<D, F> {
    fn new(open: F) -> Self {
        Self {
            open,
            decoder: OnceLock::new(),
        }
    }

    fn is_open(&self) -> bool {
        self.decoder.get().is_some()
    }
}

impl<F> LazyDecoder<dyn ArrayPartialDecoderTraits, F>
where
    F: Fn() -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>,
{
    fn get(&self) -> Result<&Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        if let Some(decoder) = self.decoder.get() {
            return Ok(decoder);
        }
        let decoder = (self.open)()?;
        Ok(self.decoder.get_or_init(|| decoder))
    }
}

#[cfg(feature = "async")]
impl<F, Fut> LazyDecoder<dyn AsyncArrayPartialDecoderTraits, F>
where
    F: Fn() -> Fut,
    Fut: Future<Output = Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>>,
{
    async fn get(&self) -> Result<&Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        if let Some(decoder) = self.decoder.get() {
            return Ok(decoder);
        }
        let decoder = (self.open)().await?;
        Ok(self.decoder.get_or_init(|| decoder))
    }
}

/// The decoded subchunks of a chunk, located in the global subchunk grid of an array.
struct ChunkSubchunks<'a> {
    grid: &'a ChunkGrid,
    /// The subset of the chunk in the array.
    chunk_subset: ArraySubset,
    array_shape: &'a [u64],
    data_type: &'a DataType,
    fill_value: &'a FillValue,
}

/// The subchunks overlapping a chunk-local region.
struct RegionPlan {
    /// The region in the array.
    array_region: ArraySubset,
    /// The region of the subchunk grid overlapping the region.
    subchunks: ArraySubset,
}

/// A decoded subchunk overlapping a region: its subset in the array and its cached value.
type RegionSubchunk = (ArraySubset, ChunkCacheTypeDecoded);

/// A region of a chunk assembled from decoded subchunks.
enum AssembledRegion {
    /// The decoded region.
    Bytes(Arc<ArrayBytes<'static>>),
    /// Every subchunk overlapping the region is missing or entirely the fill value.
    Fill,
}

impl<'a> ChunkSubchunks<'a> {
    fn new(
        array: &'a (impl CachedArrayMeta + ?Sized),
        level: usize,
        chunk_indices: &[u64],
    ) -> Result<Self, ArrayError> {
        validate_chunk_grid_indices(array, chunk_indices)?;
        Ok(Self {
            grid: array.subchunk_grid_at_level(level)?,
            chunk_subset: array.chunk_subset(chunk_indices)?,
            array_shape: array.shape(),
            data_type: array.data_type(),
            fill_value: array.fill_value(),
        })
    }

    fn chunk_shape(&self) -> &[u64] {
        self.chunk_subset.shape()
    }

    fn whole_chunk(&self) -> ArraySubset {
        ArraySubset::new_with_shape(self.chunk_shape())
    }

    /// Locate the subchunks overlapping the chunk-local `region`.
    ///
    /// Returns [`None`] if the region cannot be addressed by subchunks, because it extends beyond the array shape.
    fn plan(&self, region: &ArraySubset) -> Result<Option<RegionPlan>, ArrayError> {
        if region.is_empty() {
            return Ok(None);
        }
        let array_region = region.offset(self.chunk_subset.start())?;
        if !array_region.inbounds_shape(self.array_shape) {
            return Ok(None);
        }
        Ok(self
            .grid
            .chunks_in_array_subset(&array_region)?
            .map(|subchunks| RegionPlan {
                array_region,
                subchunks,
            }))
    }

    /// Return the subset of the subchunk at `subchunk_indices` in the array.
    ///
    /// Returns [`None`] if the subchunk is not within the chunk, as the subchunk grid does not refine it.
    fn subchunk_subset(&self, subchunk_indices: &[u64]) -> Result<Option<ArraySubset>, ArrayError> {
        Ok(self
            .grid
            .subset(subchunk_indices)?
            .filter(|subchunk_subset| subchunk_subset.inbounds(&self.chunk_subset)))
    }

    /// Return the subset of the subchunk at `subchunk_subset` in the array, relative to the chunk.
    fn local_subset(&self, subchunk_subset: &ArraySubset) -> Result<ArraySubset, ArrayError> {
        Ok(subchunk_subset.relative_to(self.chunk_subset.start())?)
    }

    /// Return the cached value of a decoded subchunk, which is [`None`] if it is entirely the fill value.
    fn value(&self, bytes: ArrayBytes<'static>) -> ChunkCacheTypeDecoded {
        (!bytes.is_fill_value(self.fill_value)).then(|| Arc::new(bytes))
    }

    /// Return `num_elements` elements of the fill value.
    fn fill(&self, num_elements: u64) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
        Ok(Arc::new(
            ArrayBytes::new_fill_value(self.data_type, num_elements, self.fill_value)
                .map_err(CodecError::from)?,
        ))
    }

    /// Return the bytes of an assembled `region` of `num_elements` elements.
    fn region_bytes(
        &self,
        region: AssembledRegion,
        num_elements: u64,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
        match region {
            AssembledRegion::Bytes(bytes) => Ok(bytes),
            AssembledRegion::Fill => self.fill(num_elements),
        }
    }

    /// Return the elements of `subchunk` in its `overlap` with a region.
    fn piece(
        &self,
        (subchunk_subset, value): &RegionSubchunk,
        overlap: &ArraySubset,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
        Ok(match value {
            None => self.fill(overlap.num_elements())?,
            Some(bytes) if overlap == subchunk_subset => bytes.clone(),
            Some(bytes) => Arc::new(
                bytes
                    .extract_array_subset(
                        &overlap.relative_to(subchunk_subset.start())?,
                        subchunk_subset.shape(),
                        self.data_type,
                    )?
                    .into_owned(),
            ),
        })
    }

    /// Compose the region of `plan` from its overlapping `subchunks`.
    ///
    /// Each subchunk contributes the piece of the region it overlaps, so the pieces tile the region.
    fn compose(
        &self,
        plan: &RegionPlan,
        subchunks: &[RegionSubchunk],
    ) -> Result<AssembledRegion, ArrayError> {
        if subchunks.iter().all(|(_, value)| value.is_none()) {
            return Ok(AssembledRegion::Fill);
        }
        let region = &plan.array_region;
        if let [subchunk] = subchunks {
            let overlap = subchunk.0.overlap(region)?;
            return Ok(AssembledRegion::Bytes(self.piece(subchunk, &overlap)?));
        }
        // Writing pieces directly into the region avoids copying whole cached subchunks, but costs more than merging
        // pieces that are all extracted from subchunks
        if let (DataType::Fixed(fixed), optional_depth) =
            optional_innermost_with_depth(self.data_type)
            && subchunks
                .iter()
                .any(|(subchunk_subset, value)| value.is_some() && subchunk_subset.inbounds(region))
        {
            return self.compose_fixed(region, subchunks, fixed.size(), optional_depth);
        }
        let pieces = subchunks
            .iter()
            .map(|subchunk| {
                let overlap = subchunk.0.overlap(region)?;
                Ok((
                    self.piece(subchunk, &overlap)?,
                    overlap.relative_to(region.start())?,
                ))
            })
            .collect::<Result<Vec<_>, ArrayError>>()?;
        Ok(AssembledRegion::Bytes(Arc::new(merge_cached_chunks(
            pieces,
            region.shape(),
            self.data_type,
        )?)))
    }

    /// Compose the `array_region` of fixed length elements from its overlapping `subchunks`, writing each piece
    /// directly into the region rather than merging separately allocated pieces.
    fn compose_fixed(
        &self,
        array_region: &ArraySubset,
        subchunks: &[RegionSubchunk],
        data_type_size: usize,
        optional_depth: usize,
    ) -> Result<AssembledRegion, ArrayError> {
        // Zero-sized data has no bytes to write, unlike the masks of zero-sized optional data
        if optional_depth == 0 && data_type_size == 0 {
            return Ok(AssembledRegion::Bytes(Arc::new(ArrayBytes::new_flen(
                vec![],
            ))));
        }
        let mut buffers = FixedDecodeBuffers::with_size_and_depth(
            data_type_size,
            optional_depth,
            array_region.shape(),
        );
        self.compose_into(array_region, subchunks, &mut buffers.target()?)?;
        // SAFETY: composing writes every element of the region, satisfying the target's output requirements
        Ok(AssembledRegion::Bytes(Arc::new(unsafe {
            buffers.into_array_bytes()
        })))
    }

    /// Compose the `array_region` of fixed length elements from its overlapping `subchunks` into `target`, which has
    /// the shape of the region.
    ///
    /// Every element of `target` is written if this succeeds.
    fn compose_into(
        &self,
        array_region: &ArraySubset,
        subchunks: &[RegionSubchunk],
        target: &mut ArrayBytesTarget<'_>,
    ) -> Result<(), ArrayError> {
        if subchunks.iter().all(|(_, value)| value.is_none()) {
            return Ok(copy_fill_value_into(
                self.data_type,
                self.fill_value,
                target,
            )?);
        }
        let mut written = 0;
        for (subchunk_subset, value) in subchunks {
            let overlap = subchunk_subset.overlap(array_region)?;
            // SAFETY: subchunks are disjoint, so their overlaps with the region are disjoint
            let mut piece_target =
                unsafe { target.subdivide(overlap.relative_to(array_region.start())?) }?;
            match value {
                None => {
                    copy_fill_value_into(self.data_type, self.fill_value, &mut piece_target)?;
                }
                Some(bytes) if overlap == *subchunk_subset => {
                    decode_into_array_bytes_target(bytes, &mut piece_target)?;
                }
                Some(bytes) => decode_into_array_bytes_target(
                    &bytes.extract_array_subset(
                        &overlap.relative_to(subchunk_subset.start())?,
                        subchunk_subset.shape(),
                        self.data_type,
                    )?,
                    &mut piece_target,
                )?,
            }
            written += overlap.num_elements();
        }
        // The disjoint pieces tile the region if they cover as many elements, so every element was written
        if written != array_region.num_elements() {
            return Err(ArrayError::CodecError(CodecError::Other(
                "the subchunks overlapping a region do not tile it".to_string(),
            )));
        }
        Ok(())
    }

    /// Return the chunk-local region selected by `indexer`, or [`None`] if it does not select a region.
    ///
    /// Elements selected by other indexers are served by the chunk cache, rather than by assembling the whole chunk.
    fn indexer_region(&self, indexer: &dyn Indexer) -> Result<Option<ArraySubset>, ArrayError> {
        indexer
            .validate(self.chunk_shape())
            .map_err(CodecError::from)?;
        Ok(indexer
            .as_array_subset()
            .map(ArraySubsetTraits::to_array_subset))
    }

    /// Return the cached value of the subchunk at `subchunk_indices`, decoding it with `decoder` if it is not cached.
    ///
    /// Returns [`None`] if the subchunk is not within the chunk.
    fn subchunk<S, F>(
        &self,
        subchunks: &S,
        subchunk_indices: ArrayIndices,
        decoder: &LazyDecoder<dyn ArrayPartialDecoderTraits, F>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<RegionSubchunk>, ArrayError>
    where
        S: ChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
        F: Fn() -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>,
    {
        let Some(subchunk_subset) = self.subchunk_subset(&subchunk_indices)? else {
            return Ok(None);
        };
        let value = subchunks
            .try_get_or_insert_with(subchunk_indices, &mut || {
                let bytes = decoder
                    .get()?
                    .partial_decode(&self.local_subset(&subchunk_subset)?, options, resources)?
                    .into_owned();
                Ok(self.value(bytes))
            })
            .map_err(cache_error)?;
        Ok(Some((subchunk_subset, value)))
    }

    /// Assemble the chunk-local `region` from decoded subchunks, decoding missing subchunks with `decoder`.
    ///
    /// Returns [`None`] if the region cannot be addressed by subchunks.
    fn assemble<S, F>(
        &self,
        subchunks: &S,
        region: &ArraySubset,
        decoder: &LazyDecoder<dyn ArrayPartialDecoderTraits, F>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<AssembledRegion>, ArrayError>
    where
        S: ChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
        F: Fn() -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> + MaybeSync,
    {
        self.assemble_with(
            subchunks,
            region,
            decoder,
            options,
            resources,
            |plan, region_subchunks| self.compose(plan, region_subchunks),
        )
    }

    /// Retrieve the decoded subchunks overlapping the chunk-local `region`, decoding missing subchunks with `decoder`,
    /// and `compose` the region from them.
    ///
    /// Returns [`None`] if the region cannot be addressed by subchunks.
    fn assemble_with<S, F, R>(
        &self,
        subchunks: &S,
        region: &ArraySubset,
        decoder: &LazyDecoder<dyn ArrayPartialDecoderTraits, F>,
        options: &CodecOptions,
        resources: &Resources,
        compose: impl FnOnce(&RegionPlan, &[RegionSubchunk]) -> Result<R, ArrayError>,
    ) -> Result<Option<R>, ArrayError>
    where
        S: ChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
        F: Fn() -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> + MaybeSync,
    {
        let Some(plan) = self.plan(region)? else {
            return Ok(None);
        };
        if plan.subchunks.num_elements_usize() == 1 {
            let subchunk_indices = plan.subchunks.start().into();
            let Some(subchunk) =
                self.subchunk(subchunks, subchunk_indices, decoder, options, resources)?
            else {
                return Ok(None);
            };
            return compose(&plan, std::slice::from_ref(&subchunk)).map(Some);
        }
        let mut indices = plan.subchunks.indices().into_iter();
        let mut region_subchunks = Vec::with_capacity(plan.subchunks.num_elements_usize());
        // Retrieve subchunks in turn until a miss opens the chunk, so that it is opened once and not by a concurrent
        // retrieval, which could reenter its opening on the same thread
        while let Some(subchunk_indices) = indices.next() {
            let Some(subchunk) =
                self.subchunk(subchunks, subchunk_indices, decoder, options, resources)?
            else {
                return Ok(None);
            };
            region_subchunks.push(subchunk);
            if decoder.is_open() {
                // The chunk is missing subchunks, so retrieve the rest concurrently
                let remaining = indices.collect();
                if !self.subchunks_concurrently(
                    subchunks,
                    remaining,
                    decoder,
                    &mut region_subchunks,
                    options,
                    resources,
                )? {
                    return Ok(None);
                }
                break;
            }
        }
        compose(&plan, &region_subchunks).map(Some)
    }

    /// Retrieve the subchunks at `subchunk_indices` concurrently with an open `decoder`, appending them to
    /// `region_subchunks`.
    ///
    /// Returns false if a subchunk is not within the chunk.
    // Not inlined, to keep the retrieval of cached subchunks in `assemble` lean
    #[inline(never)]
    fn subchunks_concurrently<S, F>(
        &self,
        subchunks: &S,
        subchunk_indices: Vec<ArrayIndices>,
        decoder: &LazyDecoder<dyn ArrayPartialDecoderTraits, F>,
        region_subchunks: &mut Vec<RegionSubchunk>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<bool, ArrayError>
    where
        S: ChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
        F: Fn() -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> + MaybeSync,
    {
        if subchunk_indices.is_empty() {
            return Ok(true);
        }
        let (concurrent_limit, resources) = remaining_resources(subchunk_indices.len(), resources);
        let retrieved = subchunk_indices
            .concurrent_limit(concurrent_limit)
            .map(|subchunk_indices| {
                self.subchunk(subchunks, subchunk_indices, decoder, options, &resources)
            })
            .collect::<Result<Vec<_>, _>>()?;
        for subchunk in retrieved {
            let Some(subchunk) = subchunk else {
                return Ok(false);
            };
            region_subchunks.push(subchunk);
        }
        Ok(true)
    }

    /// Retrieve the elements of the chunk selected by `indexer` from decoded subchunks.
    ///
    /// Returns [`None`] if they cannot be addressed by subchunks.
    fn retrieve_indexer<S, F>(
        &self,
        subchunks: &S,
        indexer: &dyn Indexer,
        decoder: &LazyDecoder<dyn ArrayPartialDecoderTraits, F>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
    where
        S: ChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
        F: Fn() -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> + MaybeSync,
    {
        let Some(region) = self.indexer_region(indexer)? else {
            return Ok(None);
        };
        // An empty selection needs no subchunks, and the chunk cache need not open the chunk
        if region.is_empty() {
            return self.fill(0).map(Some);
        }
        self.assemble(subchunks, &region, decoder, options, resources)?
            .map(|assembled| self.region_bytes(assembled, region.num_elements()))
            .transpose()
    }

    /// Decode the elements of the chunk selected by `indexer` from decoded subchunks into `output_target`, which has the
    /// shape of the selection.
    ///
    /// Returns false if they cannot be addressed by subchunks.
    fn retrieve_indexer_into<S, F>(
        &self,
        subchunks: &S,
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        decoder: &LazyDecoder<dyn ArrayPartialDecoderTraits, F>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<bool, ArrayError>
    where
        S: ChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
        F: Fn() -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> + MaybeSync,
    {
        let Some(region) = self.indexer_region(indexer)? else {
            return Ok(false);
        };
        // An empty selection has nothing to decode, and the chunk cache need not open the chunk
        if region.is_empty() {
            return Ok(true);
        }
        Ok(self
            .assemble_with(
                subchunks,
                &region,
                decoder,
                options,
                resources,
                |plan, region_subchunks| {
                    self.compose_into(&plan.array_region, region_subchunks, output_target)
                },
            )?
            .is_some())
    }

    /// Asynchronously return the cached value of the subchunk at `subchunk_indices`, decoding it with `decoder` if it
    /// is not cached.
    ///
    /// Returns [`None`] if the subchunk is not within the chunk.
    #[cfg(feature = "async")]
    async fn async_subchunk<S, F, Fut>(
        &self,
        subchunks: &S,
        subchunk_indices: ArrayIndices,
        decoder: &LazyDecoder<dyn AsyncArrayPartialDecoderTraits, F>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<RegionSubchunk>, ArrayError>
    where
        S: AsyncChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
        F: Fn() -> Fut + MaybeSync,
        Fut: Future<Output = Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>>
            + MaybeSend,
    {
        let Some(subchunk_subset) = self.subchunk_subset(&subchunk_indices)? else {
            return Ok(None);
        };
        let value = subchunks
            .try_get_or_insert_with(
                subchunk_indices,
                std::pin::pin!(async {
                    let bytes = decoder
                        .get()
                        .await?
                        .partial_decode(&self.local_subset(&subchunk_subset)?, options, resources)
                        .await?
                        .into_owned();
                    Ok(self.value(bytes))
                }),
            )
            .await
            .map_err(cache_error)?;
        Ok(Some((subchunk_subset, value)))
    }

    /// Asynchronously assemble the chunk-local `region` from decoded subchunks, decoding missing subchunks with
    /// `decoder`.
    ///
    /// Returns [`None`] if the region cannot be addressed by subchunks.
    #[cfg(feature = "async")]
    async fn async_assemble<S, F, Fut>(
        &self,
        subchunks: &S,
        region: &ArraySubset,
        decoder: &LazyDecoder<dyn AsyncArrayPartialDecoderTraits, F>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<AssembledRegion>, ArrayError>
    where
        S: AsyncChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
        F: Fn() -> Fut + MaybeSync,
        Fut: Future<Output = Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>>
            + MaybeSend,
    {
        self.async_assemble_with(
            subchunks,
            region,
            decoder,
            options,
            resources,
            |plan, region_subchunks| self.compose(plan, region_subchunks),
        )
        .await
    }

    /// Asynchronously retrieve the decoded subchunks overlapping the chunk-local `region`, decoding missing subchunks
    /// with `decoder`, and `compose` the region from them.
    ///
    /// Returns [`None`] if the region cannot be addressed by subchunks.
    #[cfg(feature = "async")]
    async fn async_assemble_with<S, F, Fut, R>(
        &self,
        subchunks: &S,
        region: &ArraySubset,
        decoder: &LazyDecoder<dyn AsyncArrayPartialDecoderTraits, F>,
        options: &CodecOptions,
        resources: &Resources,
        compose: impl FnOnce(&RegionPlan, &[RegionSubchunk]) -> Result<R, ArrayError> + MaybeSend,
    ) -> Result<Option<R>, ArrayError>
    where
        S: AsyncChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
        F: Fn() -> Fut + MaybeSync,
        Fut: Future<Output = Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>>
            + MaybeSend,
    {
        let Some(plan) = self.plan(region)? else {
            return Ok(None);
        };
        if plan.subchunks.num_elements_usize() == 1 {
            let subchunk_indices = plan.subchunks.start().into();
            let Some(subchunk) = self
                .async_subchunk(subchunks, subchunk_indices, decoder, options, resources)
                .await?
            else {
                return Ok(None);
            };
            return compose(&plan, std::slice::from_ref(&subchunk)).map(Some);
        }
        let mut indices = plan.subchunks.indices().into_iter();
        let mut region_subchunks = Vec::with_capacity(plan.subchunks.num_elements_usize());
        // Retrieve subchunks in turn until a miss opens the chunk, so that it is opened once
        for subchunk_indices in &mut indices {
            let Some(subchunk) = self
                .async_subchunk(subchunks, subchunk_indices, decoder, options, resources)
                .await?
            else {
                return Ok(None);
            };
            region_subchunks.push(subchunk);
            if decoder.is_open() {
                break;
            }
        }
        // Retrieve the remaining subchunks concurrently, as the chunk is missing subchunks
        let remaining = indices.collect::<Vec<_>>();
        if !remaining.is_empty() {
            let (concurrent_limit, resources) = remaining_resources(remaining.len(), resources);
            let mut remaining = futures::stream::iter(remaining)
                .map(|subchunk_indices| {
                    self.async_subchunk(subchunks, subchunk_indices, decoder, options, &resources)
                })
                .buffered(concurrent_limit);
            while let Some(subchunk) = remaining.try_next().await? {
                let Some(subchunk) = subchunk else {
                    return Ok(None);
                };
                region_subchunks.push(subchunk);
            }
        }
        compose(&plan, &region_subchunks).map(Some)
    }

    /// Asynchronously retrieve the elements of the chunk selected by `indexer` from decoded subchunks.
    ///
    /// Returns [`None`] if they cannot be addressed by subchunks.
    #[cfg(feature = "async")]
    async fn async_retrieve_indexer<S, F, Fut>(
        &self,
        subchunks: &S,
        indexer: &dyn Indexer,
        decoder: &LazyDecoder<dyn AsyncArrayPartialDecoderTraits, F>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError>
    where
        S: AsyncChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
        F: Fn() -> Fut + MaybeSync,
        Fut: Future<Output = Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>>
            + MaybeSend,
    {
        let Some(region) = self.indexer_region(indexer)? else {
            return Ok(None);
        };
        // An empty selection needs no subchunks, and the chunk cache need not open the chunk
        if region.is_empty() {
            return self.fill(0).map(Some);
        }
        self.async_assemble(subchunks, &region, decoder, options, resources)
            .await?
            .map(|assembled| self.region_bytes(assembled, region.num_elements()))
            .transpose()
    }

    /// Asynchronously decode the elements of the chunk selected by `indexer` from decoded subchunks into
    /// `output_target`, which has the shape of the selection.
    ///
    /// Returns false if they cannot be addressed by subchunks.
    #[cfg(feature = "async")]
    async fn async_retrieve_indexer_into<S, F, Fut>(
        &self,
        subchunks: &S,
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        decoder: &LazyDecoder<dyn AsyncArrayPartialDecoderTraits, F>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<bool, ArrayError>
    where
        S: AsyncChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized,
        F: Fn() -> Fut + MaybeSync,
        Fut: Future<Output = Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError>>
            + MaybeSend,
    {
        let Some(region) = self.indexer_region(indexer)? else {
            return Ok(false);
        };
        // An empty selection has nothing to decode, and the chunk cache need not open the chunk
        if region.is_empty() {
            return Ok(true);
        }
        Ok(self
            .async_assemble_with(
                subchunks,
                &region,
                decoder,
                options,
                resources,
                |plan, region_subchunks| {
                    self.compose_into(&plan.array_region, region_subchunks, output_target)
                },
            )
            .await?
            .is_some())
    }
}

impl<S, C> SyncArrayCacheOps for SubchunkCache<S, C>
where
    S: ChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized + 'static,
    C: SyncArrayCache,
{
    fn cached_partial_decoder(
        &self,
        array: &dyn SyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        let chunk = ChunkSubchunks::new(array, self.level, chunk_indices)?;
        // Before the chunk is opened, so that an invalidation while it opens retires the decoder
        let generation = self.generations.snapshot(chunk_indices);
        let inner = self
            .chunks
            .cached_partial_decoder(array, chunk_indices, options, resources)?;
        Ok(Arc::new(SubchunkCachePartialDecoder::new(
            self.subchunks.clone(),
            &chunk,
            inner,
            generation,
        )))
    }

    fn cached_chunk_bytes_if_exists(
        &self,
        array: &dyn SyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError> {
        let chunk = ChunkSubchunks::new(array, self.level, chunk_indices)?;
        let decoder = LazyDecoder::new(|| {
            self.chunks
                .cached_partial_decoder(array, chunk_indices, options, resources)
        });
        let whole_chunk = chunk.whole_chunk();
        match chunk.assemble(&*self.subchunks, &whole_chunk, &decoder, options, resources)? {
            Some(AssembledRegion::Bytes(bytes)) => Ok(Some(bytes)),
            // A chunk of only fill subchunks may not be stored
            Some(AssembledRegion::Fill) => decoder
                .get()?
                .exists()?
                .then(|| chunk.fill(whole_chunk.num_elements()))
                .transpose(),
            None => {
                self.chunks
                    .cached_chunk_bytes_if_exists(array, chunk_indices, options, resources)
            }
        }
    }

    fn cached_chunk_bytes(
        &self,
        array: &dyn SyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
        let chunk = ChunkSubchunks::new(array, self.level, chunk_indices)?;
        let decoder = LazyDecoder::new(|| {
            self.chunks
                .cached_partial_decoder(array, chunk_indices, options, resources)
        });
        let whole_chunk = chunk.whole_chunk();
        match chunk.assemble(&*self.subchunks, &whole_chunk, &decoder, options, resources)? {
            // A chunk of only fill subchunks is the fill value whether or not it is stored
            Some(assembled) => chunk.region_bytes(assembled, whole_chunk.num_elements()),
            None => self
                .chunks
                .cached_chunk_bytes(array, chunk_indices, options, resources),
        }
    }

    fn cached_partial_chunk_bytes(
        &self,
        array: &dyn SyncCachedArray,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
        let chunk = ChunkSubchunks::new(array, self.level, chunk_indices)?;
        let decoder = LazyDecoder::new(|| {
            self.chunks
                .cached_partial_decoder(array, chunk_indices, options, resources)
        });
        if let Some(bytes) =
            chunk.retrieve_indexer(&*self.subchunks, indexer, &decoder, options, resources)?
        {
            Ok(bytes)
        } else {
            self.chunks.cached_partial_chunk_bytes(
                array,
                chunk_indices,
                indexer,
                options,
                resources,
            )
        }
    }

    fn cached_partial_chunk_into(
        &self,
        array: &dyn SyncCachedArray,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        // Subchunks are placed by their position in the selection, which needs a fixed length target of its shape
        if !optional_innermost(array.data_type()).is_fixed()
            || output_target.shape() != indexer.output_shape()
        {
            let bytes =
                self.cached_partial_chunk_bytes(array, chunk_indices, indexer, options, resources)?;
            return decode_into_array_bytes_target(&bytes, output_target)
                .map_err(ArrayError::CodecError);
        }
        let chunk = ChunkSubchunks::new(array, self.level, chunk_indices)?;
        let decoder = LazyDecoder::new(|| {
            self.chunks
                .cached_partial_decoder(array, chunk_indices, options, resources)
        });
        if chunk.retrieve_indexer_into(
            &*self.subchunks,
            indexer,
            output_target,
            &decoder,
            options,
            resources,
        )? {
            Ok(())
        } else {
            self.chunks.cached_partial_chunk_into(
                array,
                chunk_indices,
                indexer,
                output_target,
                options,
                resources,
            )
        }
    }

    fn invalidate_cached_all(&self, array: &dyn CachedArrayMeta) {
        self.chunks.invalidate_cached_all(array);
        self.invalidate_all_subchunks();
    }

    fn invalidate_cached_chunk(&self, array: &dyn CachedArrayMeta, chunk_indices: &[u64]) {
        self.chunks.invalidate_cached_chunk(array, chunk_indices);
        self.invalidate_chunk_subchunks(array, chunk_indices);
    }

    fn invalidate_cached_chunks(&self, array: &dyn CachedArrayMeta, chunks: &dyn Indexer) {
        self.chunks.invalidate_cached_chunks(array, chunks);
        self.invalidate_chunks_subchunks(array, chunks);
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<S, C> AsyncArrayCacheOps for SubchunkCache<S, C>
where
    S: AsyncChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized + 'static,
    C: AsyncArrayCache,
{
    async fn async_cached_partial_decoder(
        &self,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, ArrayError> {
        let chunk = ChunkSubchunks::new(array, self.level, chunk_indices)?;
        // Before the chunk is opened, so that an invalidation while it opens retires the decoder
        let generation = self.generations.snapshot(chunk_indices);
        let inner = self
            .chunks
            .async_cached_partial_decoder(array, chunk_indices, options, resources)
            .await?;
        Ok(Arc::new(SubchunkCachePartialDecoder::new(
            self.subchunks.clone(),
            &chunk,
            inner,
            generation,
        )))
    }

    async fn async_cached_chunk_bytes_if_exists(
        &self,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<Arc<ArrayBytes<'static>>>, ArrayError> {
        let chunk = ChunkSubchunks::new(array, self.level, chunk_indices)?;
        let decoder = LazyDecoder::new(|| {
            self.chunks
                .async_cached_partial_decoder(array, chunk_indices, options, resources)
        });
        let whole_chunk = chunk.whole_chunk();
        match chunk
            .async_assemble(&*self.subchunks, &whole_chunk, &decoder, options, resources)
            .await?
        {
            Some(AssembledRegion::Bytes(bytes)) => Ok(Some(bytes)),
            // A chunk of only fill subchunks may not be stored
            Some(AssembledRegion::Fill) => decoder
                .get()
                .await?
                .exists()
                .await?
                .then(|| chunk.fill(whole_chunk.num_elements()))
                .transpose(),
            None => {
                self.chunks
                    .async_cached_chunk_bytes_if_exists(array, chunk_indices, options, resources)
                    .await
            }
        }
    }

    async fn async_cached_chunk_bytes(
        &self,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
        let chunk = ChunkSubchunks::new(array, self.level, chunk_indices)?;
        let decoder = LazyDecoder::new(|| {
            self.chunks
                .async_cached_partial_decoder(array, chunk_indices, options, resources)
        });
        let whole_chunk = chunk.whole_chunk();
        match chunk
            .async_assemble(&*self.subchunks, &whole_chunk, &decoder, options, resources)
            .await?
        {
            // A chunk of only fill subchunks is the fill value whether or not it is stored
            Some(assembled) => chunk.region_bytes(assembled, whole_chunk.num_elements()),
            None => {
                self.chunks
                    .async_cached_chunk_bytes(array, chunk_indices, options, resources)
                    .await
            }
        }
    }

    async fn async_cached_partial_chunk_bytes(
        &self,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<ArrayBytes<'static>>, ArrayError> {
        let chunk = ChunkSubchunks::new(array, self.level, chunk_indices)?;
        let decoder = LazyDecoder::new(|| {
            self.chunks
                .async_cached_partial_decoder(array, chunk_indices, options, resources)
        });
        if let Some(bytes) = chunk
            .async_retrieve_indexer(&*self.subchunks, indexer, &decoder, options, resources)
            .await?
        {
            Ok(bytes)
        } else {
            self.chunks
                .async_cached_partial_chunk_bytes(array, chunk_indices, indexer, options, resources)
                .await
        }
    }

    async fn async_cached_partial_chunk_into(
        &self,
        array: &dyn AsyncCachedArray,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        // Subchunks are placed by their position in the selection, which needs a fixed length target of its shape
        if !optional_innermost(array.data_type()).is_fixed()
            || output_target.shape() != indexer.output_shape()
        {
            let bytes = self
                .async_cached_partial_chunk_bytes(array, chunk_indices, indexer, options, resources)
                .await?;
            return decode_into_array_bytes_target(&bytes, output_target)
                .map_err(ArrayError::CodecError);
        }
        let chunk = ChunkSubchunks::new(array, self.level, chunk_indices)?;
        let decoder = LazyDecoder::new(|| {
            self.chunks
                .async_cached_partial_decoder(array, chunk_indices, options, resources)
        });
        if chunk
            .async_retrieve_indexer_into(
                &*self.subchunks,
                indexer,
                output_target,
                &decoder,
                options,
                resources,
            )
            .await?
        {
            Ok(())
        } else {
            self.chunks
                .async_cached_partial_chunk_into(
                    array,
                    chunk_indices,
                    indexer,
                    output_target,
                    options,
                    resources,
                )
                .await
        }
    }

    async fn async_invalidate_cached_all(&self, array: &dyn CachedArrayMeta) {
        self.chunks.async_invalidate_cached_all(array).await;
        self.async_invalidate_all_subchunks().await;
    }

    async fn async_invalidate_cached_chunk(
        &self,
        array: &dyn CachedArrayMeta,
        chunk_indices: &[u64],
    ) {
        self.chunks
            .async_invalidate_cached_chunk(array, chunk_indices)
            .await;
        self.async_invalidate_chunk_subchunks(array, chunk_indices)
            .await;
    }

    async fn async_invalidate_cached_chunks(
        &self,
        array: &dyn CachedArrayMeta,
        chunks: &dyn Indexer,
    ) {
        self.chunks
            .async_invalidate_cached_chunks(array, chunks)
            .await;
        self.async_invalidate_chunks_subchunks(array, chunks).await;
    }
}

/// Return the concurrent limit and the resources of each retrieval of `num_subchunks` subchunks retrieved concurrently.
fn remaining_resources(num_subchunks: usize, resources: &Resources) -> (usize, Resources) {
    let (concurrent_limit, resources) = split_resources(
        resources,
        &RecommendedConcurrency::new_maximum(num_subchunks),
        &RecommendedConcurrency::new_maximum(1),
    );
    (concurrent_limit.max(1), resources)
}

/// Convert an [`ArrayError`] of a partial decoder operation to a [`CodecError`].
///
/// Errors other than codec and storage errors are reported by their message.
#[allow(clippy::wildcard_enum_match_arm)]
fn codec_error(error: ArrayError) -> CodecError {
    match error {
        ArrayError::CodecError(error) => error,
        ArrayError::StorageError(error) => CodecError::StorageError(error),
        error => CodecError::Other(error.to_string()),
    }
}

/// A partial decoder of a chunk that assembles decoded subchunks from a subchunk cache, and decodes the subchunks it
/// does not hold with the (synchronous or asynchronous) partial decoder `D` of the chunk cache.
///
/// This holds its own copy of the array metadata of [`ChunkSubchunks`], as it may outlive the array.
struct SubchunkCachePartialDecoder<S: ?Sized, D: ?Sized> {
    subchunks: Arc<S>,
    grid: ChunkGrid,
    chunk_subset: ArraySubset,
    array_shape: ArrayShape,
    data_type: DataType,
    fill_value: FillValue,
    inner: Arc<D>,
    /// The generation of the chunk when `inner` was opened.
    generation: Generation,
}

impl<S: ?Sized, D: ?Sized> SubchunkCachePartialDecoder<S, D> {
    /// Create a partial decoder of `chunk` over the partial decoder `inner` of the chunk cache, which was opened in
    /// the `generation` of the chunk.
    fn new(
        subchunks: Arc<S>,
        chunk: &ChunkSubchunks<'_>,
        inner: Arc<D>,
        generation: Generation,
    ) -> Self {
        Self {
            subchunks,
            grid: chunk.grid.clone(),
            chunk_subset: chunk.chunk_subset.clone(),
            array_shape: chunk.array_shape.to_vec(),
            data_type: chunk.data_type.clone(),
            fill_value: chunk.fill_value.clone(),
            inner,
            generation,
        }
    }

    fn chunk(&self) -> ChunkSubchunks<'_> {
        ChunkSubchunks {
            grid: &self.grid,
            chunk_subset: self.chunk_subset.clone(),
            array_shape: &self.array_shape,
            data_type: &self.data_type,
            fill_value: &self.fill_value,
        }
    }
}

impl<S> ArrayPartialDecoderSubchunkingTraits
    for SubchunkCachePartialDecoder<S, dyn ArrayPartialDecoderTraits>
where
    S: ChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized + 'static,
{
    fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.inner.local_subchunk_grids(options, resources)
    }
}

impl<S> ArrayPartialDecoderTraits for SubchunkCachePartialDecoder<S, dyn ArrayPartialDecoderTraits>
where
    S: ChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized + 'static,
{
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.inner.exists()
    }

    fn size_held(&self) -> usize {
        self.inner.size_held()
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        if !self.generation.is_current() {
            return self.inner.partial_decode(indexer, options, resources);
        }
        let decoder = LazyDecoder::new(|| Ok(Arc::clone(&self.inner)));
        match self
            .chunk()
            .retrieve_indexer(&*self.subchunks, indexer, &decoder, options, resources)
            .map_err(codec_error)?
        {
            Some(bytes) => Ok(Arc::unwrap_or_clone(bytes)),
            None => self.inner.partial_decode(indexer, options, resources),
        }
    }

    fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        if !self.generation.is_current() {
            return self
                .inner
                .partial_decode_into(indexer, output_target, options, resources);
        }
        let decoder = LazyDecoder::new(|| Ok(Arc::clone(&self.inner)));
        match self
            .chunk()
            .retrieve_indexer(&*self.subchunks, indexer, &decoder, options, resources)
            .map_err(codec_error)?
        {
            Some(bytes) => decode_into_array_bytes_target(&bytes, output_target),
            None => self
                .inner
                .partial_decode_into(indexer, output_target, options, resources),
        }
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<S> AsyncArrayPartialDecoderSubchunkingTraits
    for SubchunkCachePartialDecoder<S, dyn AsyncArrayPartialDecoderTraits>
where
    S: AsyncChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized + 'static,
{
    async fn local_subchunk_grids(
        &self,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        self.inner.local_subchunk_grids(options, resources).await
    }
}

#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
impl<S> AsyncArrayPartialDecoderTraits
    for SubchunkCachePartialDecoder<S, dyn AsyncArrayPartialDecoderTraits>
where
    S: AsyncChunkCache<Value = ChunkCacheTypeDecoded> + ?Sized + 'static,
{
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    async fn exists(&self) -> Result<bool, StorageError> {
        self.inner.exists().await
    }

    fn size_held(&self) -> usize {
        self.inner.size_held()
    }

    async fn partial_decode<'a>(
        &'a self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        if !self.generation.is_current() {
            return self.inner.partial_decode(indexer, options, resources).await;
        }
        let decoder = LazyDecoder::new(|| async { Ok(Arc::clone(&self.inner)) });
        match self
            .chunk()
            .async_retrieve_indexer(&*self.subchunks, indexer, &decoder, options, resources)
            .await
            .map_err(codec_error)?
        {
            Some(bytes) => Ok(Arc::unwrap_or_clone(bytes)),
            None => self.inner.partial_decode(indexer, options, resources).await,
        }
    }

    async fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        if !self.generation.is_current() {
            return self
                .inner
                .partial_decode_into(indexer, output_target, options, resources)
                .await;
        }
        let decoder = LazyDecoder::new(|| async { Ok(Arc::clone(&self.inner)) });
        match self
            .chunk()
            .async_retrieve_indexer(&*self.subchunks, indexer, &decoder, options, resources)
            .await
            .map_err(codec_error)?
        {
            Some(bytes) => decode_into_array_bytes_target(&bytes, output_target),
            None => {
                self.inner
                    .partial_decode_into(indexer, output_target, options, resources)
                    .await
            }
        }
    }

    fn supports_partial_decode(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;
    use crate::array::array_bytes_internal::reordered;
    use crate::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
    use crate::array::{ArrayBuilder, ElementOrder, IntoArrayBytes, data_type};
    use zarrs_storage::store::MemoryStore;

    /// Compose a region overlapping every 4x4 subchunk of an 8x8 `chunk`, and covering subchunk `[1, 1]`, from its
    /// decoded subchunks, and compare it with the region extracted from the chunk, in C order and into an F order
    /// target.
    fn check_compose(
        data_type: &DataType,
        fill_value: FillValue,
        chunk: impl IntoArrayBytes<'static>,
    ) {
        let nz = |n| NonZeroU64::new(n).unwrap();
        let mut builder = ArrayBuilder::new(vec![8, 8], vec![8, 8], data_type.clone(), fill_value);
        builder.array_to_bytes_codec(
            ShardingCodecBuilder::new(vec![nz(4), nz(4)], data_type).build_arc(),
        );
        let array = builder
            .build(Arc::new(MemoryStore::default()), "/array")
            .unwrap();
        let chunk_subchunks = ChunkSubchunks::new(&array, 0, &[0, 0]).unwrap();
        let chunk = chunk
            .into_array_bytes(data_type, ElementOrder::C)
            .unwrap()
            .into_owned();
        let region = ArraySubset::new_with_ranges(&[2..8, 3..8]);
        let plan = chunk_subchunks.plan(&region).unwrap().unwrap();
        let subchunks = plan
            .subchunks
            .indices()
            .into_iter()
            .map(|subchunk_indices| {
                let subset = chunk_subchunks
                    .subchunk_subset(&subchunk_indices)
                    .unwrap()
                    .unwrap();
                let bytes = chunk
                    .extract_array_subset(&subset, &[8, 8], data_type)
                    .unwrap()
                    .into_owned();
                (subset, chunk_subchunks.value(bytes))
            })
            .collect::<Vec<_>>();
        assert!(
            subchunks[0].1.is_none(),
            "subchunk [0, 0] is the fill value"
        );
        let expected = chunk
            .extract_array_subset(&region, &[8, 8], data_type)
            .unwrap()
            .into_owned();
        let Ok(AssembledRegion::Bytes(bytes)) = chunk_subchunks.compose(&plan, &subchunks) else {
            panic!("the region is not the fill value");
        };
        assert_eq!(*bytes, expected);

        // A strided target is written in place
        let mut buffers = FixedDecodeBuffers::new(data_type, region.shape()).unwrap();
        chunk_subchunks
            .compose_into(
                &region,
                &subchunks,
                &mut buffers.target_in_order(ElementOrder::F).unwrap(),
            )
            .unwrap();
        // SAFETY: composing writes every element of the region
        let f_order = unsafe { buffers.into_array_bytes() };
        let expected = reordered(
            &expected,
            data_type,
            region.shape(),
            ElementOrder::C,
            ElementOrder::F,
        )
        .unwrap();
        assert_eq!(f_order, expected);
    }

    /// Whether the element at `index` of an 8x8 chunk is in its top left 4x4 subchunk.
    fn in_fill_subchunk(index: u16) -> bool {
        index / 8 < 4 && index % 8 < 4
    }

    #[test]
    fn compose_fixed_length_subchunks() {
        let chunk: Vec<u16> = (0..64)
            .map(|i| if in_fill_subchunk(i) { 0 } else { i })
            .collect();
        check_compose(&data_type::uint16(), 0u16.into(), chunk);
    }

    #[test]
    fn compose_optional_subchunks() {
        let chunk: Vec<Option<u16>> = (0..64)
            .map(|i| (!in_fill_subchunk(i) && i % 3 != 0).then_some(i))
            .collect();
        check_compose(
            &data_type::uint16().to_optional(),
            None::<u16>.into(),
            chunk,
        );
    }
}
