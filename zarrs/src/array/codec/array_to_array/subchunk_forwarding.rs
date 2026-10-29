//! Subchunk forwarding for array-to-array partial codecs.
//!
//! An array-to-array partial codec exposes the subchunks of the partial codec it wraps: the encoded subchunks are
//! those of the wrapped codec, and only their *addressing* changes, because the codec may map coordinates between its
//! decoded and encoded domains. The subchunk grids of the wrapped codec are mapped outward with
//! [`ArrayToArrayCodecSubchunkingTraits::decoded_local_subchunk_grid`], and subchunk indices are mapped back inward
//! with [`ArrayToArrayCodecSubchunkingTraits::encoded_subchunk_indices`].
//!
//! [`ArrayToArrayCodecSubchunkingTraits::decoded_local_subchunk_grid`]: zarrs_codec::ArrayToArrayCodecSubchunkingTraits::decoded_local_subchunk_grid
//! [`ArrayToArrayCodecSubchunkingTraits::encoded_subchunk_indices`]: zarrs_codec::ArrayToArrayCodecSubchunkingTraits::encoded_subchunk_indices

use std::num::NonZeroU64;

use zarrs_codec::{ArrayToArrayCodecTraits, CodecError};

use crate::array::{ArrayIndices, ChunkGrid};

/// Map the chunk-local subchunk grids of a wrapped partial codec through `codec`.
pub(super) fn decoded_local_subchunk_grids(
    codec: &dyn ArrayToArrayCodecTraits,
    decoded_shape: &[NonZeroU64],
    encoded_subchunk_grids: Vec<Option<ChunkGrid>>,
) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
    encoded_subchunk_grids
        .into_iter()
        .map(|grid| {
            grid.map_or(Ok(None), |grid| {
                codec.decoded_local_subchunk_grid(decoded_shape, &grid)
            })
        })
        .collect()
}

/// Map subchunk indices of the `level` chunk-local subchunk grid to the `encoded_subchunk_grid` of a wrapped partial
/// codec at that level.
pub(super) fn encoded_subchunk_indices(
    codec: &dyn ArrayToArrayCodecTraits,
    decoded_shape: &[NonZeroU64],
    level: usize,
    encoded_subchunk_grid: Option<ChunkGrid>,
    subchunk_indices: &[u64],
) -> Result<ArrayIndices, CodecError> {
    let encoded_subchunk_grid = encoded_subchunk_grid.ok_or_else(|| {
        CodecError::Other(format!(
            "there is no resolvable subchunk grid at level {level}"
        ))
    })?;
    codec.encoded_subchunk_indices(decoded_shape, &encoded_subchunk_grid, subchunk_indices)
}

/// Implement the subchunking traits of an array-to-array partial codec by forwarding to the partial codec it wraps.
///
/// - `impl_subchunk_forwarding!(Partial, inner, codec, shape)` maps grids and indices through the
///   `codec: Arc<dyn ArrayToArrayCodecTraits>` field, with the decoded chunk shape in the `shape` field.
/// - `impl_subchunk_forwarding!(Partial, inner, identity)` forwards grids and indices unchanged, for a codec that
///   alters values but not positions.
///
/// The wrapped partial codec is in the `inner` field.
macro_rules! impl_subchunk_forwarding {
    ($partial:ident, $inner:ident, $codec:ident, $shape:ident) => {
        impl<T: ?Sized> zarrs_codec::ArrayPartialDecoderSubchunkingTraits for $partial<T>
        where
            T: zarrs_codec::ArrayPartialDecoderSubchunkingTraits,
        {
            fn local_subchunk_grids(
                &self,
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Vec<Option<$crate::array::ChunkGrid>>, zarrs_codec::CodecError> {
                $crate::array::codec::array_to_array::subchunk_forwarding::decoded_local_subchunk_grids(
                    self.$codec.as_ref(),
                    &self.$shape,
                    self.$inner.local_subchunk_grids(options)?,
                )
            }

            fn retrieve_encoded_subchunk_at_level(
                &self,
                level: usize,
                subchunk_indices: &[u64],
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Option<zarrs_codec::EncodedSubchunk>, zarrs_codec::CodecError> {
                let subchunk_indices =
                    $crate::array::codec::array_to_array::subchunk_forwarding::encoded_subchunk_indices(
                        self.$codec.as_ref(),
                        &self.$shape,
                        level,
                        self.$inner.local_subchunk_grid_at_level(level, options)?,
                        subchunk_indices,
                    )?;
                self.$inner
                    .retrieve_encoded_subchunk_at_level(level, &subchunk_indices, options)
            }
        }

        #[cfg(feature = "async")]
        #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
        #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
        impl<T: ?Sized> zarrs_codec::AsyncArrayPartialDecoderSubchunkingTraits for $partial<T>
        where
            T: zarrs_codec::AsyncArrayPartialDecoderSubchunkingTraits,
        {
            async fn local_subchunk_grids(
                &self,
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Vec<Option<$crate::array::ChunkGrid>>, zarrs_codec::CodecError> {
                $crate::array::codec::array_to_array::subchunk_forwarding::decoded_local_subchunk_grids(
                    self.$codec.as_ref(),
                    &self.$shape,
                    self.$inner.local_subchunk_grids(options).await?,
                )
            }

            async fn retrieve_encoded_subchunk_at_level(
                &self,
                level: usize,
                subchunk_indices: &[u64],
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Option<zarrs_codec::EncodedSubchunk>, zarrs_codec::CodecError> {
                let subchunk_indices =
                    $crate::array::codec::array_to_array::subchunk_forwarding::encoded_subchunk_indices(
                        self.$codec.as_ref(),
                        &self.$shape,
                        level,
                        self.$inner
                            .local_subchunk_grid_at_level(level, options)
                            .await?,
                        subchunk_indices,
                    )?;
                self.$inner
                    .retrieve_encoded_subchunk_at_level(level, &subchunk_indices, options)
                    .await
            }
        }
    };

    ($partial:ident, $inner:ident, identity) => {
        impl<T: ?Sized> zarrs_codec::ArrayPartialDecoderSubchunkingTraits for $partial<T>
        where
            T: zarrs_codec::ArrayPartialDecoderSubchunkingTraits,
        {
            fn local_subchunk_grids(
                &self,
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Vec<Option<$crate::array::ChunkGrid>>, zarrs_codec::CodecError> {
                self.$inner.local_subchunk_grids(options)
            }

            fn retrieve_encoded_subchunk_at_level(
                &self,
                level: usize,
                subchunk_indices: &[u64],
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Option<zarrs_codec::EncodedSubchunk>, zarrs_codec::CodecError> {
                self.$inner
                    .retrieve_encoded_subchunk_at_level(level, subchunk_indices, options)
            }
        }

        #[cfg(feature = "async")]
        #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
        #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
        impl<T: ?Sized> zarrs_codec::AsyncArrayPartialDecoderSubchunkingTraits for $partial<T>
        where
            T: zarrs_codec::AsyncArrayPartialDecoderSubchunkingTraits,
        {
            async fn local_subchunk_grids(
                &self,
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Vec<Option<$crate::array::ChunkGrid>>, zarrs_codec::CodecError> {
                self.$inner.local_subchunk_grids(options).await
            }

            async fn retrieve_encoded_subchunk_at_level(
                &self,
                level: usize,
                subchunk_indices: &[u64],
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Option<zarrs_codec::EncodedSubchunk>, zarrs_codec::CodecError> {
                self.$inner
                    .retrieve_encoded_subchunk_at_level(level, subchunk_indices, options)
                    .await
            }
        }
    };
}

pub(super) use impl_subchunk_forwarding;
