//! Shared subchunking forwarding for array-to-array partial codecs.
//!
//! An array-to-array codec exposes the encoded subchunks of the decoder it wraps: the bytes and
//! the subchunk codecs are those of the inner decoder, and only the *addressing* changes, because
//! the codec may map coordinates between its decoded and encoded domains.
//!
//! Every such codec therefore forwards the subchunking surface identically once the subchunk grid
//! has been mapped outward and the subchunk indices translated back inward. [`SubchunkMapping`]
//! carries those two mappings, which are the only parts that differ between codecs, and
//! [`impl_subchunk_forwarding`] generates the forwarding from them.
//!
//! A codec that maps coordinates identically (e.g. `bitround` and `cast_value`, which alter values
//! but not positions) maps the grid and the indices to themselves.

use std::sync::Arc;

use zarrs_chunk_grid::ChunkGrid;
use zarrs_codec::{CodecError, CodecOptions};

/// The subchunk grid and index mappings of an array-to-array partial codec.
///
/// The two mappings are inverses: [`map_local_subchunk_grid`](Self::map_local_subchunk_grid) maps a
/// grid of the wrapped decoder outward into this codec's decoded domain, and
/// [`remap_subchunk_indices`](Self::remap_subchunk_indices) maps indices of that outward grid back
/// inward.
pub(super) trait SubchunkMapping {
    /// The type of the wrapped input handle.
    type Inner: ?Sized;

    /// Return the wrapped input handle.
    fn inner(&self) -> &Arc<Self::Inner>;

    /// Map a subchunk grid of [`inner`](Self::inner) outward into this codec's decoded domain.
    ///
    /// Returns [`None`] if the grid cannot be represented in the decoded domain.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the grid is incompatible with this codec.
    fn map_local_subchunk_grid(&self, grid: &ChunkGrid) -> Result<Option<ChunkGrid>, CodecError>;
}

/// Map subchunk indices inward, the inverse of
/// [`SubchunkMapping::map_local_subchunk_grid`].
///
/// This is separate from [`SubchunkMapping`] because a codec whose mapping depends on the subchunk
/// grid of the decoder it wraps must query that decoder, which bounds the handle type differently
/// in the synchronous and asynchronous cases.
pub(super) trait SubchunkRemap {
    /// Map `subchunk_indices` of `level` inward into the domain of the wrapped handle.
    ///
    /// The `level` is the subchunk grid level being addressed, which a codec whose mapping depends
    /// on the grid must resolve against.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the indices cannot be mapped.
    fn remap_subchunk_indices(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<u64>, CodecError>;
}

/// The asynchronous counterpart of [`SubchunkRemap`].
///
/// This is separate because a codec whose mapping depends on the subchunk grid of the decoder it
/// wraps must await that grid, which the synchronous method cannot do.
#[cfg(feature = "async")]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
pub(super) trait AsyncSubchunkRemap:
    zarrs_storage::MaybeSend + zarrs_storage::MaybeSync
{
    /// Map `subchunk_indices` of `level` back into the domain of the wrapped handle.
    ///
    /// # Errors
    /// Returns [`CodecError`] if the indices cannot be mapped.
    async fn async_remap_subchunk_indices(
        &self,
        level: usize,
        subchunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Vec<u64>, CodecError>;
}

/// Map a subchunk grid hierarchy of the wrapped decoder outward, preserving unresolvable levels.
pub(super) fn map_local_subchunk_grids<M: SubchunkMapping + ?Sized>(
    mapping: &M,
    grids: Vec<Option<ChunkGrid>>,
) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
    grids
        .into_iter()
        .map(|grid| grid.map_or(Ok(None), |grid| mapping.map_local_subchunk_grid(&grid)))
        .collect()
}

/// Implement the subchunking traits of an array-to-array partial codec in terms of its
/// [`SubchunkMapping`].
///
/// Both the synchronous and asynchronous traits are generated. A blanket implementation is not
/// possible: the subchunking traits are also implemented directly by sharding, the chunk caches,
/// and the partial decoder cache, so a blanket implementation over a marker trait would conflict
/// with all of them.
macro_rules! impl_subchunk_forwarding {
    ($codec:ident) => {
        const _: () = {
            use std::sync::Arc;

            use zarrs_chunk_grid::ChunkGrid;
            use zarrs_codec::{
                ArrayBytesRaw, ArrayPartialDecoderSubchunkingTraits, ArrayToBytesCodecTraits,
                BytesPartialDecoderTraits, CodecError, CodecOptions, EncodedSubchunk,
            };
            use zarrs_metadata::ChunkShape;

            use $crate::array::codec::array_to_array::subchunk_forwarding::{
                SubchunkMapping, SubchunkRemap, map_local_subchunk_grids,
            };

            impl<T: ?Sized> ArrayPartialDecoderSubchunkingTraits for $codec<T>
            where
                T: ArrayPartialDecoderSubchunkingTraits,
                $codec<T>: SubchunkMapping<Inner = T> + SubchunkRemap,
            {
                fn local_subchunk_grids(
                    &self,
                    options: &CodecOptions,
                ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
                    map_local_subchunk_grids(self, self.inner().local_subchunk_grids(options)?)
                }

                fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
                    self.inner().subchunk_codecs()
                }

                fn encoded_subchunk_shape_at_level(
                    &self,
                    level: usize,
                    subchunk_indices: &[u64],
                    options: &CodecOptions,
                ) -> Result<ChunkShape, CodecError> {
                    let inner = self.remap_subchunk_indices(level, subchunk_indices, options)?;
                    self.inner()
                        .encoded_subchunk_shape_at_level(level, &inner, options)
                }

                fn retrieve_encoded_subchunk_bytes(
                    &self,
                    subchunk_indices: &[u64],
                    options: &CodecOptions,
                ) -> Result<Option<ArrayBytesRaw<'_>>, CodecError> {
                    let inner = self.remap_subchunk_indices(0, subchunk_indices, options)?;
                    self.inner()
                        .retrieve_encoded_subchunk_bytes(&inner, options)
                }

                fn retrieve_encoded_subchunk(
                    &self,
                    subchunk_indices: &[u64],
                    options: &CodecOptions,
                ) -> Result<Option<EncodedSubchunk<'_>>, CodecError> {
                    // Remap once and let the inner decoder resolve the bytes and the shape from
                    // the same indices, rather than remapping for each
                    let inner = self.remap_subchunk_indices(0, subchunk_indices, options)?;
                    self.inner().retrieve_encoded_subchunk(&inner, options)
                }

                fn encoded_subchunk_partial_decoder(
                    &self,
                    subchunk_indices: &[u64],
                    options: &CodecOptions,
                ) -> Result<Option<Arc<dyn BytesPartialDecoderTraits>>, CodecError> {
                    let inner = self.remap_subchunk_indices(0, subchunk_indices, options)?;
                    self.inner()
                        .encoded_subchunk_partial_decoder(&inner, options)
                }

                fn retrieve_encoded_subchunk_at_level(
                    &self,
                    level: usize,
                    subchunk_indices: &[u64],
                    options: &CodecOptions,
                ) -> Result<Option<EncodedSubchunk<'static>>, CodecError> {
                    // Delegate the whole descent so the inner decoder descends in one consistent
                    // domain
                    let inner = self.remap_subchunk_indices(level, subchunk_indices, options)?;
                    self.inner()
                        .retrieve_encoded_subchunk_at_level(level, &inner, options)
                }
            }
        };

        #[cfg(feature = "async")]
        const _: () = {
            use std::sync::Arc;

            use zarrs_chunk_grid::ChunkGrid;
            use zarrs_codec::{
                ArrayBytesRaw, ArrayToBytesCodecTraits, AsyncArrayPartialDecoderSubchunkingTraits,
                AsyncBytesPartialDecoderTraits, CodecError, CodecOptions, EncodedSubchunk,
            };
            use zarrs_metadata::ChunkShape;

            use $crate::array::codec::array_to_array::subchunk_forwarding::{
                AsyncSubchunkRemap, SubchunkMapping, map_local_subchunk_grids,
            };

            #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
            #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
            impl<T: ?Sized> AsyncArrayPartialDecoderSubchunkingTraits for $codec<T>
            where
                T: AsyncArrayPartialDecoderSubchunkingTraits,
                $codec<T>: SubchunkMapping<Inner = T> + AsyncSubchunkRemap,
            {
                async fn local_subchunk_grids(
                    &self,
                    options: &CodecOptions,
                ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
                    let grids = self.inner().local_subchunk_grids(options).await?;
                    map_local_subchunk_grids(self, grids)
                }

                fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
                    self.inner().subchunk_codecs()
                }

                async fn encoded_subchunk_shape_at_level(
                    &self,
                    level: usize,
                    subchunk_indices: &[u64],
                    options: &CodecOptions,
                ) -> Result<ChunkShape, CodecError> {
                    let inner = self
                        .async_remap_subchunk_indices(level, subchunk_indices, options)
                        .await?;
                    self.inner()
                        .encoded_subchunk_shape_at_level(level, &inner, options)
                        .await
                }

                async fn retrieve_encoded_subchunk_bytes<'a>(
                    &'a self,
                    subchunk_indices: &[u64],
                    options: &CodecOptions,
                ) -> Result<Option<ArrayBytesRaw<'a>>, CodecError> {
                    let inner = self
                        .async_remap_subchunk_indices(0, subchunk_indices, options)
                        .await?;
                    self.inner()
                        .retrieve_encoded_subchunk_bytes(&inner, options)
                        .await
                }

                async fn retrieve_encoded_subchunk<'a>(
                    &'a self,
                    subchunk_indices: &[u64],
                    options: &CodecOptions,
                ) -> Result<Option<EncodedSubchunk<'a>>, CodecError> {
                    // Remap once and let the inner decoder resolve the bytes and the shape from
                    // the same indices, rather than remapping for each
                    let inner = self
                        .async_remap_subchunk_indices(0, subchunk_indices, options)
                        .await?;
                    self.inner().retrieve_encoded_subchunk(&inner, options).await
                }

                async fn encoded_subchunk_partial_decoder(
                    &self,
                    subchunk_indices: &[u64],
                    options: &CodecOptions,
                ) -> Result<Option<Arc<dyn AsyncBytesPartialDecoderTraits>>, CodecError> {
                    let inner = self
                        .async_remap_subchunk_indices(0, subchunk_indices, options)
                        .await?;
                    self.inner()
                        .encoded_subchunk_partial_decoder(&inner, options)
                        .await
                }

                async fn retrieve_encoded_subchunk_at_level(
                    &self,
                    level: usize,
                    subchunk_indices: &[u64],
                    options: &CodecOptions,
                ) -> Result<Option<EncodedSubchunk<'static>>, CodecError> {
                    // Delegate the whole descent so the inner decoder descends in one consistent
                    // domain
                    let inner = self
                        .async_remap_subchunk_indices(level, subchunk_indices, options)
                        .await?;
                    self.inner()
                        .retrieve_encoded_subchunk_at_level(level, &inner, options)
                        .await
                }
            }
        };
    };
}

pub(super) use impl_subchunk_forwarding;
