//! Subchunk forwarding for array-to-array partial codecs.
//!
//! An array-to-array partial codec exposes the subchunks of the partial codec it wraps: the encoded subchunks are
//! those of the wrapped codec, and only their *addressing* changes, because the codec may map coordinates between its
//! decoded and encoded domains. The subchunk grids of the wrapped codec are mapped outward with
//! [`ArrayToArrayCodecSubchunkingTraits::decoded_local_subchunk_grids`], and a subchunk is addressed inward by its
//! subset, mapped with [`ArrayToArrayCodecSubchunkingTraits::encoded_subset`].
//!
//! A subchunk addressed by indices is located in the subchunk grid of this codec once, and then forwarded by subset, so
//! addressing is linear in the number of codecs.
//!
//! [`ArrayToArrayCodecSubchunkingTraits::decoded_local_subchunk_grids`]: zarrs_codec::ArrayToArrayCodecSubchunkingTraits::decoded_local_subchunk_grids
//! [`ArrayToArrayCodecSubchunkingTraits::encoded_subset`]: zarrs_codec::ArrayToArrayCodecSubchunkingTraits::encoded_subset

/// Implement the subchunking traits of an array-to-array partial codec by forwarding to the partial codec it wraps.
///
/// - `impl_subchunk_forwarding!(Partial, inner, codec, shape)` maps grids and subsets through the
///   `codec: Arc<dyn ArrayToArrayCodecTraits>` field, with the decoded chunk shape in the `shape` field.
/// - `impl_subchunk_forwarding!(Partial, inner, identity)` forwards grids and subsets unchanged, for a codec that
///   alters values but not positions.
///
/// The wrapped partial codec is in the `inner` field.
macro_rules! impl_subchunk_forwarding {
    ($partial:ident, $inner:ident, $codec:ident, $shape:ident) => {
        impl<T: ?Sized> $partial<T> {
            /// Map a chunk-local subchunk `subset` to the domain of the partial codec this wraps.
            fn encoded_subset<'a>(
                &self,
                subset: &'a $crate::array::ArraySubset,
            ) -> Result<std::borrow::Cow<'a, $crate::array::ArraySubset>, zarrs_codec::CodecError>
            {
                Ok(std::borrow::Cow::Owned(
                    zarrs_codec::ArrayToArrayCodecSubchunkingTraits::encoded_subset(
                        self.$codec.as_ref(),
                        &self.$shape,
                        subset,
                    )?,
                ))
            }
        }

        $crate::array::codec::array_to_array::subchunk_forwarding::impl_subchunk_forwarding!(
            @decoder $partial,
            $inner,
            |partial: &Self, inner_grids| {
                zarrs_codec::ArrayToArrayCodecSubchunkingTraits::decoded_local_subchunk_grids(
                    partial.$codec.as_ref(),
                    &partial.$shape,
                    inner_grids,
                )
            }
        );
    };

    ($partial:ident, $inner:ident, identity) => {
        impl<T: ?Sized> $partial<T> {
            /// Map a chunk-local subchunk `subset` to the domain of the partial codec this wraps.
            #[allow(clippy::unnecessary_wraps, clippy::unused_self)]
            fn encoded_subset<'a>(
                &self,
                subset: &'a $crate::array::ArraySubset,
            ) -> Result<std::borrow::Cow<'a, $crate::array::ArraySubset>, zarrs_codec::CodecError>
            {
                Ok(std::borrow::Cow::Borrowed(subset))
            }
        }

        $crate::array::codec::array_to_array::subchunk_forwarding::impl_subchunk_forwarding!(
            @decoder $partial,
            $inner,
            |_partial: &Self, inner_grids| Ok(inner_grids)
        );
    };

    (@decoder $partial:ident, $inner:ident, $map_grids:expr) => {
        impl<T: ?Sized> zarrs_codec::ArrayPartialDecoderSubchunkingTraits for $partial<T>
        where
            T: zarrs_codec::ArrayPartialDecoderSubchunkingTraits,
        {
            fn local_subchunk_grids(
                &self,
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Vec<Option<$crate::array::ChunkGrid>>, zarrs_codec::CodecError> {
                let map_grids: fn(
                    &Self,
                    Vec<Option<$crate::array::ChunkGrid>>,
                )
                    -> Result<Vec<Option<$crate::array::ChunkGrid>>, zarrs_codec::CodecError> =
                    $map_grids;
                map_grids(self, self.$inner.local_subchunk_grids(options)?)
            }

            fn retrieve_encoded_subchunk_at_level(
                &self,
                level: usize,
                subchunk_indices: &[u64],
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Option<zarrs_codec::EncodedSubchunk>, zarrs_codec::CodecError> {
                let subset =
                    zarrs_codec::local_subchunk_subset(self, level, subchunk_indices, options)?;
                self.retrieve_encoded_subchunk_by_subset(level, &subset, options)
            }

            fn retrieve_encoded_subchunk_by_subset(
                &self,
                level: usize,
                subset: &$crate::array::ArraySubset,
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Option<zarrs_codec::EncodedSubchunk>, zarrs_codec::CodecError> {
                self.$inner.retrieve_encoded_subchunk_by_subset(
                    level,
                    &*self.encoded_subset(subset)?,
                    options,
                )
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
                let map_grids: fn(
                    &Self,
                    Vec<Option<$crate::array::ChunkGrid>>,
                )
                    -> Result<Vec<Option<$crate::array::ChunkGrid>>, zarrs_codec::CodecError> =
                    $map_grids;
                map_grids(self, self.$inner.local_subchunk_grids(options).await?)
            }

            async fn retrieve_encoded_subchunk_at_level(
                &self,
                level: usize,
                subchunk_indices: &[u64],
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Option<zarrs_codec::EncodedSubchunk>, zarrs_codec::CodecError> {
                let subset =
                    zarrs_codec::async_local_subchunk_subset(self, level, subchunk_indices, options)
                        .await?;
                self.retrieve_encoded_subchunk_by_subset(level, &subset, options)
                    .await
            }

            async fn retrieve_encoded_subchunk_by_subset(
                &self,
                level: usize,
                subset: &$crate::array::ArraySubset,
                options: &zarrs_codec::CodecOptions,
            ) -> Result<Option<zarrs_codec::EncodedSubchunk>, zarrs_codec::CodecError> {
                let encoded_subset = self.encoded_subset(subset)?;
                self.$inner
                    .retrieve_encoded_subchunk_by_subset(level, &encoded_subset, options)
                    .await
            }
        }
    };
}

pub(super) use impl_subchunk_forwarding;

/// Implement the encoded subchunk methods of [`ArrayPartialEncoderTraits`](zarrs_codec::ArrayPartialEncoderTraits)
/// for an array-to-array partial codec by forwarding to the partial codec in the `inner` field.
///
/// This is invoked within the trait implementation, and requires [`impl_subchunk_forwarding`].
/// The asynchronous equivalent cannot be a macro invoked within an `async_trait` implementation, so asynchronous
/// encoders implement the same methods directly.
macro_rules! encoder_subchunk_forwarding {
    ($inner:ident) => {
        unsafe fn store_encoded_subchunk_at_level(
            &self,
            level: usize,
            subchunk_indices: &[u64],
            bytes: zarrs_codec::CowBytes<'_>,
            options: &zarrs_codec::CodecOptions,
        ) -> Result<(), zarrs_codec::CodecError> {
            let subset =
                zarrs_codec::local_subchunk_subset(self, level, subchunk_indices, options)?;
            // SAFETY: The caller upholds the safety requirements.
            unsafe { self.store_encoded_subchunk_by_subset(level, &subset, bytes, options) }
        }

        unsafe fn store_encoded_subchunk_by_subset(
            &self,
            level: usize,
            subset: &$crate::array::ArraySubset,
            bytes: zarrs_codec::CowBytes<'_>,
            options: &zarrs_codec::CodecOptions,
        ) -> Result<(), zarrs_codec::CodecError> {
            // SAFETY: The subchunk is unchanged, only its subset is mapped.
            unsafe {
                self.$inner.store_encoded_subchunk_by_subset(
                    level,
                    &*self.encoded_subset(subset)?,
                    bytes,
                    options,
                )
            }
        }

        fn erase_subchunk_at_level(
            &self,
            level: usize,
            subchunk_indices: &[u64],
            options: &zarrs_codec::CodecOptions,
        ) -> Result<(), zarrs_codec::CodecError> {
            let subset =
                zarrs_codec::local_subchunk_subset(self, level, subchunk_indices, options)?;
            self.erase_subchunk_by_subset(level, &subset, options)
        }

        fn erase_subchunk_by_subset(
            &self,
            level: usize,
            subset: &$crate::array::ArraySubset,
            options: &zarrs_codec::CodecOptions,
        ) -> Result<(), zarrs_codec::CodecError> {
            self.$inner
                .erase_subchunk_by_subset(level, &*self.encoded_subset(subset)?, options)
        }
    };
}

pub(super) use encoder_subchunk_forwarding;
