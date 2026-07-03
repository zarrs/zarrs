use inherent::inherent;
use std::borrow::Cow;
use std::sync::Arc;

use super::super::super::array_bytes_internal::{
    build_nested_optional_target, merge_chunks_vlen, merge_chunks_vlen_optional,
    optional_nesting_depth,
};
use super::super::super::concurrency::concurrency_chunks_and_codec;
use super::super::super::{ArrayBytesFixedDisjointView, ArrayIndicesTinyVec};
use super::super::*;
use super::ArrayReadOps;
use crate::array::ArrayBytes;
use crate::iter_concurrent_limit;
#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use unsafe_cell_slice::UnsafeCellSlice;
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayBytesOptional, ArrayBytesVariableLength,
    ArrayPartialDecoderTraits, ArrayToBytesCodecTraits, CodecError, StoragePartialDecoder,
    copy_fill_value_into,
};
use zarrs_storage::StorageHandle;

#[inherent]
impl<TStorage: ?Sized + ReadableStorageTraits + 'static> ArrayReadOps for Array<TStorage> {
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn retrieve_chunk<T: FromArrayBytes>(&self, chunk_indices: &[u64])
    -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_into(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        if chunk_indices.len() != self.dimensionality() {
            return Err(ArrayError::InvalidChunkGridIndicesError(
                chunk_indices.to_vec(),
            ));
        }
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_readable_transformer(storage_handle)?;
        let chunk_encoded = storage_transformer
            .get(&self.chunk_key(chunk_indices))
            .map_err(ArrayError::StorageError)?;
        if let Some(chunk_encoded) = chunk_encoded {
            self.codecs_bound()
                .decode_into(
                    Cow::Owned(chunk_encoded.into()),
                    &self.chunk_shape(chunk_indices)?,
                    output_target,
                    options,
                )
                .map_err(ArrayError::CodecError)
        } else {
            copy_fill_value_into(self.data_type(), self.fill_value(), output_target)
                .map_err(ArrayError::CodecError)
        }
    }

    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
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

    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
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
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let chunk_shape_u64 = bytemuck::must_cast_slice(&chunk_shape);
        if !chunk_subset.inbounds_shape(chunk_shape_u64) {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        }

        let bytes = if chunk_subset.start().iter().all(|&o| o == 0)
            && chunk_subset.shape() == chunk_shape_u64
        {
            // Fast path if `chunk_subset` encompasses the whole chunk
            return self.retrieve_chunk_opt(chunk_indices, options);
        } else {
            let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
            let storage_transformer = self
                .storage_transformers()
                .create_readable_transformer(storage_handle)?;
            let input_handle = Arc::new(StoragePartialDecoder::new(
                storage_transformer,
                self.chunk_key(chunk_indices),
            ));

            self.codecs_bound()
                .partial_decoder(input_handle, &chunk_shape, options)?
                .partial_decode(chunk_subset, options)?
                .into_owned()
        };
        bytes.validate(chunk_subset.num_elements(), self.data_type())?;
        T::from_array_bytes(bytes, &chunk_subset.shape(), self.data_type())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_subset_into(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let chunk_shape_u64 = bytemuck::must_cast_slice(&chunk_shape);
        if !chunk_subset.inbounds_shape(chunk_shape_u64) {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        }

        if chunk_subset.start().iter().all(|&o| o == 0) && chunk_subset.shape() == chunk_shape_u64 {
            self.retrieve_chunk_into(chunk_indices, output_target, options)
        } else {
            let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
            let storage_transformer = self
                .storage_transformers()
                .create_readable_transformer(storage_handle)?;
            let input_handle = Arc::new(StoragePartialDecoder::new(
                storage_transformer,
                self.chunk_key(chunk_indices),
            ));
            self.codecs_bound()
                .partial_decoder(input_handle, &chunk_shape, options)?
                .partial_decode_into(chunk_subset, output_target, options)?;
            Ok(())
        }
    }

    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn retrieve_array_subset<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn retrieve_array_subset_opt<T: FromArrayBytes>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError> {
        if array_subset.dimensionality() != self.dimensionality() {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        }

        // Find the chunks intersecting this array subset
        let chunks = self.chunks_in_array_subset(array_subset)?;
        let Some(chunks) = chunks else {
            return Err(ArrayError::InvalidArraySubset(
                array_subset.to_array_subset(),
                self.shape().to_vec(),
            ));
        };

        // Retrieve chunk bytes
        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => {
                let bytes = ArrayBytes::new_fill_value(
                    self.data_type(),
                    array_subset.num_elements(),
                    self.fill_value(),
                )
                .map_err(CodecError::from)
                .map_err(ArrayError::from)?;
                T::from_array_bytes(bytes, &array_subset.shape(), self.data_type())
            }
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = self.chunk_subset(chunk_indices)?;
                if chunk_subset == array_subset {
                    // Single chunk fast path if the array subset domain matches the chunk domain
                    self.retrieve_chunk_opt(chunk_indices, options)
                } else {
                    let array_subset_in_chunk_subset =
                        array_subset.relative_to(chunk_subset.start())?;
                    self.retrieve_chunk_subset_opt(
                        chunk_indices,
                        &array_subset_in_chunk_subset,
                        options,
                    )
                }
            }
            _ => {
                let chunk_shape = self.chunk_shape(chunks.start())?;

                // Calculate chunk/codec concurrency
                let codec_concurrency = self.recommended_codec_concurrency(&chunk_shape)?;
                let (chunk_concurrent_limit, options) = concurrency_chunks_and_codec(
                    options.concurrent_target(),
                    num_chunks,
                    options,
                    &codec_concurrency,
                );

                // Delegate to appropriate helper based on data type size
                let bytes = if self.data_type().is_fixed() {
                    self.retrieve_multi_chunk_fixed(
                        array_subset,
                        &chunks,
                        self.data_type(),
                        chunk_concurrent_limit,
                        &options,
                    )?
                } else {
                    self.retrieve_multi_chunk_variable(
                        array_subset,
                        &chunks,
                        self.data_type(),
                        chunk_concurrent_limit,
                        &options,
                    )?
                };
                bytes.validate(array_subset.num_elements(), self.data_type())?;
                T::from_array_bytes(bytes.into_owned(), &array_subset.shape(), self.data_type())
            }
        }
    }
    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<T>, ArrayError>;

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
        let _ = options;
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_readable_transformer(storage_handle)?;

        storage_transformer
            .get(&self.chunk_key(chunk_indices))
            .map(|maybe_bytes| maybe_bytes.map(Into::into))
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
    pub fn retrieve_subchunks_opt<T: FromArrayBytes>(
        &self,
        subchunks: &dyn ArraySubsetTraits,
        options: &CodecOptions,
    ) -> Result<T, ArrayError>;

    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
    ) -> Result<(), ArrayError>;

    #[allow(clippy::missing_errors_doc)]
    pub fn partial_decoder(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError>;

    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_if_exists_opt<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<T>, ArrayError> {
        if chunk_indices.len() != self.dimensionality() {
            return Err(ArrayError::InvalidChunkGridIndicesError(
                chunk_indices.to_vec(),
            ));
        }
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_readable_transformer(storage_handle)?;
        let chunk_encoded = storage_transformer
            .get(&self.chunk_key(chunk_indices))
            .map_err(ArrayError::StorageError)?;
        if let Some(chunk_encoded) = chunk_encoded {
            let chunk_shape = self.chunk_shape(chunk_indices)?;
            let bytes = self
                .codecs_bound()
                .decode(Cow::Owned(chunk_encoded.into()), &chunk_shape, options)
                .map_err(ArrayError::CodecError)?;
            Ok(Some(T::from_array_bytes(
                bytes.into_owned(),
                bytemuck::must_cast_slice(&chunk_shape),
                self.data_type(),
            )?))
        } else {
            Ok(None)
        }
    }

    #[allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
    pub fn retrieve_array_subset_into_opt(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        super::common::retrieve_array_subset_into(
            self,
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
    pub fn partial_decoder_opt(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_readable_transformer(storage_handle)?;
        let input_handle = Arc::new(StoragePartialDecoder::new(
            storage_transformer,
            self.chunk_key(chunk_indices),
        ));
        Ok(self.codecs_bound().partial_decoder(
            input_handle,
            &self.chunk_shape(chunk_indices)?,
            options,
        )?)
    }
}

impl<TStorage: ?Sized + ReadableStorageTraits + 'static> Array<TStorage> {
    /// Helper method to retrieve multiple chunks with variable-length data types.
    /// Also handles optional data types with variable-length inner types (including nested optionals).
    pub(crate) fn retrieve_multi_chunk_variable(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        data_type: &DataType,
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        let nesting_depth = optional_nesting_depth(data_type);

        let chunk_indices = chunks.indices();
        if nesting_depth > 0 {
            let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| -> Result<
                (ArrayBytesOptional<'static>, ArraySubset),
                ArrayError,
            > {
                let chunk_subset = self.chunk_subset(&chunk_indices)?;
                let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                Ok((
                    self.retrieve_chunk_subset_opt::<ArrayBytes<'static>>(
                        &chunk_indices,
                        &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                        options,
                    )?
                    .into_optional()?,
                    chunk_subset_overlap.relative_to(&array_subset.start())?,
                ))
            };
            let chunk_bytes_and_subsets =
                iter_concurrent_limit!(chunk_concurrent_limit, chunk_indices, map, retrieve_chunk)
                    .collect::<Result<Vec<_>, _>>()?;
            Ok(ArrayBytes::Optional(merge_chunks_vlen_optional(
                chunk_bytes_and_subsets,
                &array_subset.shape(),
                nesting_depth,
            )?))
        } else {
            let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| -> Result<
                (ArrayBytesVariableLength<'static>, ArraySubset),
                ArrayError,
            > {
                let chunk_subset = self.chunk_subset(&chunk_indices)?;
                let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                Ok((
                    self.retrieve_chunk_subset_opt::<ArrayBytes<'static>>(
                        &chunk_indices,
                        &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                        options,
                    )?
                    .into_variable()?,
                    chunk_subset_overlap.relative_to(&array_subset.start())?,
                ))
            };
            let chunk_bytes_and_subsets =
                iter_concurrent_limit!(chunk_concurrent_limit, chunk_indices, map, retrieve_chunk)
                    .collect::<Result<Vec<_>, _>>()?;
            Ok(ArrayBytes::Variable(merge_chunks_vlen(
                chunk_bytes_and_subsets,
                &array_subset.shape(),
            )))
        }
    }

    /// Helper method to retrieve multiple chunks with fixed-length data types.
    /// Also handles optional data types with fixed-length inner types (including nested optionals).
    pub(crate) fn retrieve_multi_chunk_fixed(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        data_type: &DataType,
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        let data_type_size = data_type
            .fixed_size()
            .expect("data_type must have fixed size");
        let num_elements = array_subset.num_elements_usize();
        let size_output = num_elements * data_type_size;
        let nesting_depth = optional_nesting_depth(data_type);
        let mut data_output = Vec::with_capacity(size_output);
        let mut mask_outputs: Vec<Vec<u8>> = (0..nesting_depth)
            .map(|_| Vec::with_capacity(num_elements))
            .collect();

        {
            let data_output_slice =
                UnsafeCellSlice::new_from_vec_with_spare_capacity(&mut data_output);
            let mask_output_slices: Vec<_> = mask_outputs
                .iter_mut()
                .map(UnsafeCellSlice::new_from_vec_with_spare_capacity)
                .collect();

            let retrieve_chunk = |chunk_indices: ArrayIndicesTinyVec| {
                let chunk_subset = self.chunk_subset(&chunk_indices)?;
                let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
                let chunk_subset_in_array =
                    chunk_subset_overlap.relative_to(&array_subset.start())?;

                let array_subset_shape = array_subset.shape();
                let mut data_view = unsafe {
                    // SAFETY: chunks represent disjoint array subsets
                    ArrayBytesFixedDisjointView::new(
                        data_output_slice,
                        data_type_size,
                        &array_subset_shape,
                        chunk_subset_in_array.clone(),
                    )?
                };

                let mut mask_views: Vec<ArrayBytesFixedDisjointView<'_>> = mask_output_slices
                    .iter()
                    .map(|mask_slice| unsafe {
                        // SAFETY: chunks represent disjoint array subsets
                        ArrayBytesFixedDisjointView::new(
                            *mask_slice,
                            1,
                            &array_subset_shape,
                            chunk_subset_in_array.clone(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                let target =
                    build_nested_optional_target(&mut data_view, mask_views.as_mut_slice());

                self.retrieve_chunk_subset_into(
                    &chunk_indices,
                    &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                    target,
                    options,
                )?;
                Ok::<_, ArrayError>(())
            };

            let indices = chunks.indices();
            iter_concurrent_limit!(
                chunk_concurrent_limit,
                indices,
                try_for_each,
                retrieve_chunk
            )?;
        }

        unsafe { data_output.set_len(size_output) };
        for mask in &mut mask_outputs {
            unsafe { mask.set_len(num_elements) };
        }

        let mut array_bytes = ArrayBytes::new_flen(data_output);
        for mask in mask_outputs.into_iter().rev() {
            array_bytes = array_bytes.with_optional_mask(mask);
        }
        Ok(array_bytes)
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::single_range_in_vec_init)]
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use super::*;
    use crate::array::{ArrayBuilder, ArraySubset, data_type};
    use zarrs_storage::store::MemoryStore;

    fn array_read_ops_subchunks_impl(sharded: bool) -> Result<(), Box<dyn std::error::Error>> {
        let store = Arc::new(MemoryStore::default());
        let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type::uint16(), 0u16);
        if sharded {
            builder.subchunk_shape(vec![2, 2]);
        }
        let array = builder.build(store, "/array")?;

        let data: Vec<u16> = (0..array.shape().iter().product())
            .map(|i| i as u16)
            .collect();
        array.store_array_subset(&array.subset_all(), &data)?;

        if sharded {
            let subchunk_grid = array.subchunk_grid().unwrap();
            assert_eq!(
                subchunk_grid.chunk_shape(&vec![0; array.dimensionality()])?,
                Some(vec![NonZeroU64::new(2).unwrap(); 2])
            );
            assert_eq!(subchunk_grid.grid_shape(), &[4, 4]);

            let compare = array.retrieve_array_subset::<Vec<u16>>(&[4..6, 6..8])?;
            let test =
                array.retrieve_subchunk_opt::<Vec<u16>>(&[2, 3], &CodecOptions::default())?;
            assert_eq!(compare, test);

            let local_subchunk_grid =
                array.local_subchunk_grid(&[0, 0], &CodecOptions::default())?;
            assert_eq!(
                local_subchunk_grid.unwrap().chunk_shape(&[0, 0])?.unwrap(),
                vec![NonZeroU64::new(2).unwrap(); 2]
            );

            #[cfg(feature = "ndarray")]
            {
                let compare = array.retrieve_array_subset::<ndarray::ArrayD<u16>>(&[4..6, 6..8])?;
                let test = array.retrieve_subchunk_opt::<ndarray::ArrayD<u16>>(
                    &[2, 3],
                    &CodecOptions::default(),
                )?;
                assert_eq!(compare, test);
            }

            let subset = ArraySubset::new_with_ranges(&[2..6, 2..6]);
            let subchunks = ArraySubset::new_with_ranges(&[1..3, 1..3]);
            let compare = array.retrieve_array_subset::<Vec<u16>>(&subset)?;
            let test =
                array.retrieve_subchunks_opt::<Vec<u16>>(&subchunks, &CodecOptions::default())?;
            assert_eq!(compare, test);

            #[cfg(feature = "ndarray")]
            {
                let compare = array.retrieve_array_subset::<ndarray::ArrayD<u16>>(&subset)?;
                let test = array.retrieve_subchunks_opt::<ndarray::ArrayD<u16>>(
                    &subchunks,
                    &CodecOptions::default(),
                )?;
                assert_eq!(compare, test);
            }
        } else {
            assert!(array.subchunk_grid().is_none());
            assert!(
                array
                    .local_subchunk_grid(&[0, 0], &CodecOptions::default())?
                    .is_none()
            );

            let chunks = ArraySubset::new_with_ranges(&[0..2, 0..2]);
            assert!(matches!(
                array.retrieve_subchunk_opt::<Vec<u16>>(&[1, 1], &CodecOptions::default()),
                Err(ArrayError::MissingSubchunkGrid)
            ));
            assert!(matches!(
                array.retrieve_subchunks_opt::<Vec<u16>>(&chunks, &CodecOptions::default()),
                Err(ArrayError::MissingSubchunkGrid)
            ));
        }

        assert!(
            array
                .retrieve_subchunk_opt::<Vec<u16>>(&[0], &CodecOptions::default())
                .is_err()
        );
        assert!(
            array
                .retrieve_subchunks_opt::<Vec<u16>>(&[0..1], &CodecOptions::default())
                .is_err()
        );

        Ok(())
    }

    #[test]
    fn array_read_ops_subchunks_sharded() -> Result<(), Box<dyn std::error::Error>> {
        array_read_ops_subchunks_impl(true)
    }

    #[test]
    fn array_read_ops_subchunks_unsharded() -> Result<(), Box<dyn std::error::Error>> {
        array_read_ops_subchunks_impl(false)
    }
}
