use inherent::inherent;
use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs_codec::Resources;

use super::super::ArrayIndices;
use super::super::array_bytes_internal::{FixedDecodeBuffers, merge_chunks};
use super::super::concurrency::concurrency_chunks_and_codec;
use super::array_read_ops_common::retrieve_multi_chunk_fixed_into;
use super::{
    Array, ArrayError, ArrayReadOps, ArraySubsetTraits, ChunkGrid, CodecOptions, FromArrayBytes,
    Indexer, SyncReadableStorageTraits, recommended_codec_concurrency,
};
use crate::IntoConcurrentLimitIterator;
use crate::array::{ArrayBytes, ArraySubset, ChunkShapeTraits};
#[cfg(not(target_arch = "wasm32"))]
use rayon::iter::ParallelIterator;
use zarrs_codec::{
    ArrayBytesTarget, ArrayPartialDecoderTraits, ArrayToBytesCodecTraits,
    InvalidNumberOfElementsError, copy_fill_value_into,
};
use zarrs_storage::Bytes;

#[inherent]
impl<TStorage: ?Sized + SyncReadableStorageTraits + 'static> ArrayReadOps for Array<TStorage> {
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let options = self.codec_options();
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        super::retrieve_in_order(
            self.data_type(),
            self.element_order(),
            bytemuck::must_cast_slice(&chunk_shape),
            |target| {
                self.retrieve_chunk_into_with_options(chunk_indices, target, options, resources)
            },
            || self.retrieve_chunk_with_shape(chunk_indices, &chunk_shape, options, resources),
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
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        super::validate_partial_chunk_subset(indexer, bytemuck::must_cast_slice(&chunk_shape))?;
        super::retrieve_in_order(
            self.data_type(),
            super::indexer_element_order(indexer, self.element_order()),
            &indexer.output_shape(),
            |target| {
                self.retrieve_partial_chunk_into_with_shape(
                    chunk_indices,
                    &chunk_shape,
                    indexer,
                    target,
                    options,
                    resources,
                )
            },
            || {
                self.retrieve_partial_chunk_with_shape(
                    chunk_indices,
                    &chunk_shape,
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
        let chunks = super::array_subset_chunks(self, array_subset)?;
        let array_subset_shape = array_subset.shape();
        super::retrieve_in_order(
            self.data_type(),
            self.element_order(),
            &array_subset_shape,
            |target| self.retrieve_array_subset_into(array_subset, target, resources),
            || {
                self.retrieve_array_subset_c_order(
                    array_subset,
                    &array_subset_shape,
                    &chunks,
                    resources,
                )
            },
        )
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Bytes>, ArrayError> {
        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.readable_storage()?;

        Ok(storage_transformer.get(&chunk_key)?)
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

    /////////////////////////////////////////////////////////////////////////////
    // Advanced methods
    /////////////////////////////////////////////////////////////////////////////

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk_if_exists<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<Option<T>, ArrayError> {
        let Some(chunk_encoded) = self.retrieve_encoded_chunk(chunk_indices)? else {
            return Ok(None);
        };
        let options = self.codec_options();
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        // One retrieval decodes the chunk, and takes it so that its bytes can be decoded in place if they are unique
        let chunk_encoded = std::cell::Cell::new(Some(chunk_encoded));
        let take_encoded = || chunk_encoded.take().expect("one retrieval");
        super::retrieve_in_order(
            self.data_type(),
            self.element_order(),
            bytemuck::must_cast_slice(&chunk_shape),
            |target| {
                self.decode_chunk_into(take_encoded(), &chunk_shape, target, options, resources)
            },
            || self.decode_chunk(take_encoded(), &chunk_shape, options, resources),
        )
        .map(Some)
    }

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
            self.chunk_concurrent_minimum,
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
        self.partial_decoder_with_options(chunk_indices, self.codec_options(), resources)
    }
}

impl<TStorage: ?Sized + SyncReadableStorageTraits + 'static> Array<TStorage> {
    /// Retrieve `array_subset`, which spans `chunks`, with elements in C order.
    ///
    /// `chunks` are the chunks intersecting `array_subset` (see [`array_subset_chunks`](super::array_subset_chunks)).
    fn retrieve_array_subset_c_order(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        array_subset_shape: &[u64],
        chunks: &ArraySubset,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        let options = self.codec_options();
        let num_chunks = chunks.num_elements_usize();
        match num_chunks {
            0 => super::fill_value_bytes(
                self.data_type(),
                self.fill_value(),
                array_subset.num_elements(),
            ),
            1 => {
                let chunk_indices = chunks.start();
                let chunk_subset = self.chunk_subset(chunk_indices)?;
                if chunk_subset == array_subset {
                    // Single chunk fast path if the array subset domain matches the chunk domain
                    self.retrieve_chunk_with_options(chunk_indices, options, resources)
                } else {
                    let array_subset_in_chunk_subset =
                        array_subset.relative_to(chunk_subset.start())?;
                    self.retrieve_partial_chunk_with_options(
                        chunk_indices,
                        &array_subset_in_chunk_subset,
                        options,
                        resources,
                    )
                }
            }
            _ => {
                let chunk_shape = self.chunk_shape(chunks.start())?;

                // Calculate chunk/codec concurrency
                let codec_concurrency = recommended_codec_concurrency(self, &chunk_shape)?;
                let (chunk_concurrent_limit, resources) = concurrency_chunks_and_codec(
                    num_chunks,
                    self.chunk_concurrent_minimum,
                    resources,
                    &codec_concurrency,
                );

                // Delegate to appropriate helper based on data type size
                let bytes = if let Some(buffers) =
                    FixedDecodeBuffers::new(self.data_type(), array_subset_shape)
                {
                    self.retrieve_multi_chunk_fixed(
                        array_subset,
                        chunks,
                        buffers,
                        chunk_concurrent_limit,
                        options,
                        &resources,
                    )?
                } else {
                    self.retrieve_multi_chunk_variable(
                        array_subset,
                        chunks,
                        array_subset_shape,
                        chunk_concurrent_limit,
                        options,
                        &resources,
                    )?
                };
                bytes.validate(array_subset.num_elements(), self.data_type())?;
                Ok(bytes.into_owned())
            }
        }
    }

    /// Retrieve the chunk at `chunk_indices` in C order, or its fill value if it does not exist.
    pub(in crate::array) fn retrieve_chunk_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        self.retrieve_chunk_with_shape(
            chunk_indices,
            &self.chunk_shape(chunk_indices)?,
            options,
            resources,
        )
    }

    /// Retrieve the chunk at `chunk_indices` with `chunk_shape` in C order, or its fill value if it does not exist.
    fn retrieve_chunk_with_shape(
        &self,
        chunk_indices: &[u64],
        chunk_shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        if let Some(chunk_encoded) = self.retrieve_encoded_chunk(chunk_indices)? {
            self.decode_chunk(chunk_encoded, chunk_shape, options, resources)
        } else {
            super::fill_value_bytes(
                self.data_type(),
                self.fill_value(),
                chunk_shape.num_elements_u64(),
            )
        }
    }

    pub(in crate::array) fn retrieve_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        if output_target.num_elements() != chunk_shape.num_elements_u64() {
            return Err(ArrayError::CodecError(
                InvalidNumberOfElementsError::new(
                    output_target.num_elements(),
                    chunk_shape.num_elements_u64(),
                )
                .into(),
            ));
        }
        if let Some(chunk_encoded) = self.retrieve_encoded_chunk(chunk_indices)? {
            self.decode_chunk_into(
                chunk_encoded,
                &chunk_shape,
                output_target,
                options,
                resources,
            )
        } else {
            copy_fill_value_into(self.data_type(), self.fill_value(), output_target)
                .map_err(ArrayError::CodecError)
        }
    }

    /// Retrieve the elements of `indexer` in the chunk at `chunk_indices` in C order.
    pub(in crate::array) fn retrieve_partial_chunk_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        super::validate_partial_chunk_subset(indexer, bytemuck::must_cast_slice(&chunk_shape))?;
        self.retrieve_partial_chunk_with_shape(
            chunk_indices,
            &chunk_shape,
            indexer,
            options,
            resources,
        )
    }

    /// Retrieve the elements of `indexer` in the chunk at `chunk_indices` with `chunk_shape` in C order.
    ///
    /// `indexer` must have been validated against `chunk_shape` (see [`validate_partial_chunk_subset`](super::validate_partial_chunk_subset)).
    fn retrieve_partial_chunk_with_shape(
        &self,
        chunk_indices: &[u64],
        chunk_shape: &[NonZeroU64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, ArrayError> {
        if indexer.covers_array(bytemuck::must_cast_slice(chunk_shape)) {
            return self.retrieve_chunk_with_shape(chunk_indices, chunk_shape, options, resources);
        }

        let bytes = {
            let chunk_key = self.chunk_key(chunk_indices)?;
            let storage_transformer = self.readable_storage()?;
            let input_handle = Arc::new((storage_transformer, chunk_key));
            self.codecs_bound()
                .partial_decoder(input_handle, chunk_shape, options, resources)?
                .partial_decode(indexer, options, resources)?
                .into_owned()
        };
        bytes.validate(indexer.len(), self.data_type())?;
        Ok(bytes)
    }

    pub(in crate::array) fn retrieve_partial_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        super::validate_partial_chunk_subset(indexer, bytemuck::must_cast_slice(&chunk_shape))?;
        self.retrieve_partial_chunk_into_with_shape(
            chunk_indices,
            &chunk_shape,
            indexer,
            output_target,
            options,
            resources,
        )
    }

    /// Retrieve the elements of `indexer` in the chunk at `chunk_indices` with `chunk_shape` into `output_target`.
    ///
    /// `indexer` must have been validated against `chunk_shape` (see [`validate_partial_chunk_subset`](super::validate_partial_chunk_subset)).
    fn retrieve_partial_chunk_into_with_shape(
        &self,
        chunk_indices: &[u64],
        chunk_shape: &[NonZeroU64],
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        if indexer.covers_array(bytemuck::must_cast_slice(chunk_shape)) {
            return self.retrieve_chunk_into_with_options(
                chunk_indices,
                output_target,
                options,
                resources,
            );
        }

        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.readable_storage()?;
        let input_handle = Arc::new((storage_transformer, chunk_key));
        self.codecs_bound()
            .partial_decoder(input_handle, chunk_shape, options, resources)?
            .partial_decode_into(indexer, output_target, options, resources)?;
        Ok(())
    }

    /// Retrieve the chunk at `chunk_indices` in C order if it exists.
    pub(in crate::array) fn retrieve_chunk_if_exists_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<ArrayBytes<'static>>, ArrayError> {
        let Some(chunk_encoded) = self.retrieve_encoded_chunk(chunk_indices)? else {
            return Ok(None);
        };
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        self.decode_chunk(chunk_encoded, &chunk_shape, options, resources)
            .map(Some)
    }

    pub(in crate::array) fn partial_decoder_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        let chunk_key = self.chunk_key(chunk_indices)?;
        let storage_transformer = self.readable_storage()?;
        let input_handle = Arc::new((storage_transformer, chunk_key));
        Ok(self.codecs_bound().partial_decoder(
            input_handle,
            &self.chunk_shape(chunk_indices)?,
            options,
            resources,
        )?)
    }

    /// Helper method to retrieve multiple chunks with variable-length data types.
    /// Also handles optional data types with variable-length inner types (including nested optionals).
    pub(crate) fn retrieve_multi_chunk_variable(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        array_subset_shape: &[u64],
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        let retrieve_chunk = |chunk_indices: ArrayIndices| {
            let chunk_subset = self.chunk_subset(&chunk_indices)?;
            let chunk_subset_overlap = chunk_subset.overlap(array_subset)?;
            Ok::<_, ArrayError>((
                self.retrieve_partial_chunk_with_options(
                    &chunk_indices,
                    &chunk_subset_overlap.relative_to(chunk_subset.start())?,
                    options,
                    resources,
                )?,
                chunk_subset_overlap.relative_to(&array_subset.start())?,
            ))
        };
        let chunk_bytes_and_subsets = chunks
            .indices()
            .concurrent_limit(chunk_concurrent_limit)
            .map(retrieve_chunk)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(merge_chunks(
            chunk_bytes_and_subsets,
            array_subset_shape,
            self.data_type(),
        )?)
    }

    /// Helper method to retrieve multiple chunks with fixed-length data types.
    /// Also handles optional data types with fixed-length inner types (including nested optionals).
    pub(crate) fn retrieve_multi_chunk_fixed(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        chunks: &dyn ArraySubsetTraits,
        mut buffers: FixedDecodeBuffers,
        chunk_concurrent_limit: usize,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, ArrayError> {
        retrieve_multi_chunk_fixed_into(
            self,
            array_subset,
            chunks,
            chunk_concurrent_limit,
            &buffers.target()?,
            options,
            resources,
            &|chunk_indices, chunk_subset, overlap, target, options, resources| {
                self.retrieve_partial_chunk_into_with_options(
                    chunk_indices,
                    &overlap.relative_to(chunk_subset.start())?,
                    target,
                    options,
                    resources,
                )
            },
        )?;
        // SAFETY: decoding covers the array subset and satisfies the target's output requirements
        Ok(unsafe { buffers.into_array_bytes() })
    }
}

#[cfg(test)]
#[allow(clippy::cast_possible_truncation)]
mod tests {
    #![expect(clippy::single_range_in_vec_init)]
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use super::*;
    use crate::array::{ArrayBuilder, ArraySubset, ChunkGridDecodedRef, data_type};
    use zarrs_storage::store::MemoryStore;

    /// Reads in F order return F-ordered elements, including the fill value of missing chunks.
    #[test]
    fn retrieve_labels_f_order() -> Result<(), Box<dyn std::error::Error>> {
        use crate::array::{ElementOrder, Tensor};
        let store = Arc::new(MemoryStore::default());
        let array = ArrayBuilder::new(vec![2, 9], vec![2, 3], data_type::uint8(), 0u8)
            .build(store, "/array")?;
        let c_order: Vec<u8> = (1..13).collect();
        array.store_array_subset(&[0..2, 0..6], &c_order, &Resources::default())?;
        let array = array.with_element_order(ElementOrder::F);
        let resources = Resources::default();

        let tensor: Tensor<'static> = array.retrieve_chunk(&[0, 1], &resources)?;
        assert_eq!(tensor.order(), ElementOrder::F);
        assert_eq!(tensor.as_parts().0, [4, 10, 5, 11, 6, 12]);
        let tensor: Tensor<'static> = array.retrieve_chunk(&[0, 2], &resources)?;
        assert_eq!(tensor.order(), ElementOrder::F);
        assert_eq!(tensor.as_parts().0, [0; 6]);
        let tensor: Tensor<'static> = array.retrieve_array_subset(&[0..2, 1..5], &resources)?;
        assert_eq!(tensor.order(), ElementOrder::F);
        assert_eq!(tensor.as_parts().0, [2, 8, 3, 9, 4, 10, 5, 11]);
        let tensor: Tensor<'static> = array.retrieve_partial_chunk(
            &[0, 0],
            &ArraySubset::new_with_ranges(&[0..2, 1..3]),
            &resources,
        )?;
        assert_eq!(tensor.order(), ElementOrder::F);
        assert_eq!(tensor.as_parts().0, [2, 8, 3, 9]);

        // The C and F orders of a single row coincide
        let tensor: Tensor<'static> = array.retrieve_array_subset(&[1..2, 1..5], &resources)?;
        assert_eq!(tensor.order(), ElementOrder::F);
        assert_eq!(tensor.as_parts().0, [8, 9, 10, 11]);
        Ok(())
    }

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
        array.store_array_subset(&array.subset_all(), &data, &Resources::default())?;

        if sharded {
            let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap();
            assert_eq!(
                subchunk_grid.chunk_shape(&vec![0; array.dimensionality()])?,
                Some(vec![NonZeroU64::new(2).unwrap(); 2])
            );
            assert_eq!(subchunk_grid.grid_shape(), &[4, 4]);

            let compare =
                array.retrieve_array_subset::<Vec<u16>>(&[4..6, 6..8], &Resources::default())?;
            let test = array.retrieve_subchunk::<Vec<u16>>(&[2, 3], &Resources::default())?;
            assert_eq!(compare, test);

            let local_subchunk_grid = array.local_subchunk_grid(&[0, 0], &Resources::default())?;
            assert_eq!(
                local_subchunk_grid.unwrap().chunk_shape(&[0, 0])?.unwrap(),
                vec![NonZeroU64::new(2).unwrap(); 2]
            );

            #[cfg(feature = "ndarray")]
            {
                let compare = array.retrieve_array_subset::<ndarray::ArrayD<u16>>(
                    &[4..6, 6..8],
                    &Resources::default(),
                )?;
                let test = array
                    .retrieve_subchunk::<ndarray::ArrayD<u16>>(&[2, 3], &Resources::default())?;
                assert_eq!(compare, test);
            }

            let subset = ArraySubset::new_with_ranges(&[2..6, 2..6]);
            let subchunks = ArraySubset::new_with_ranges(&[1..3, 1..3]);
            let compare =
                array.retrieve_array_subset::<Vec<u16>>(&subset, &Resources::default())?;
            let test = array.retrieve_subchunks::<Vec<u16>>(&subchunks, &Resources::default())?;
            assert_eq!(compare, test);

            #[cfg(feature = "ndarray")]
            {
                let compare = array.retrieve_array_subset::<ndarray::ArrayD<u16>>(
                    &subset,
                    &Resources::default(),
                )?;
                let test = array.retrieve_subchunks::<ndarray::ArrayD<u16>>(
                    &subchunks,
                    &Resources::default(),
                )?;
                assert_eq!(compare, test);
            }
        } else {
            assert!(matches!(array.subchunk_grid(), ChunkGridDecodedRef::None));
            assert!(
                array
                    .local_subchunk_grid(&[0, 0], &Resources::default())?
                    .is_none()
            );

            let chunks = ArraySubset::new_with_ranges(&[0..2, 0..2]);
            assert!(matches!(
                array.retrieve_subchunk::<Vec<u16>>(&[1, 1], &Resources::default()),
                Err(ArrayError::MissingSubchunkGrid)
            ));
            assert!(matches!(
                array.retrieve_subchunks::<Vec<u16>>(&chunks, &Resources::default()),
                Err(ArrayError::MissingSubchunkGrid)
            ));
        }

        assert!(
            array
                .retrieve_subchunk::<Vec<u16>>(&[0], &Resources::default())
                .is_err()
        );
        assert!(
            array
                .retrieve_subchunks::<Vec<u16>>(&[0..1], &Resources::default())
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
