use inherent::inherent;
use std::sync::Arc;

use super::{ArrayReadOps, *};
use zarrs_codec::{ArrayBytesDecodeIntoTarget, ArrayPartialDecoderTraits};
use zarrs_storage::{Bytes, StorageHandle};

#[inherent]
impl<TStorage: ?Sized + ReadableStorageTraits + 'static> ArrayReadOps for Array<TStorage> {
    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_chunk<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        self.retrieve_chunk_with_options(chunk_indices, self.codec_options(), resources)
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
        self.retrieve_partial_chunk_with_options(
            chunk_indices,
            indexer,
            self.codec_options(),
            resources,
        )
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

        let bytes = self.retrieve_array_subset_planned(array_subset, options, resources)?;
        T::from_array_bytes(bytes, &array_subset.shape(), self.data_type())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_encoded_chunk(
        &self,
        chunk_indices: &[u64],
    ) -> Result<Option<Bytes>, ArrayError> {
        let options = self.codec_options();
        let _ = options;
        let storage_handle = Arc::new(StorageHandle::new(self.storage.clone()));
        let storage_transformer = self
            .storage_transformers()
            .create_readable_transformer(storage_handle)?;

        Ok(storage_transformer.get(&self.chunk_key(chunk_indices)?)?)
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
        self.retrieve_encoded_subchunk_planned(level, subchunk_indices, resources)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_encoded_chunks(
        &self,
        chunks: &dyn Indexer,
        resources: &Resources,
    ) -> Result<Vec<Option<Bytes>>, ArrayError> {
        chunks
            .validate(self.chunk_grid_shape())
            .map_err(zarrs_codec::CodecError::from)?;
        self.retrieve_encoded_chunks_planned(chunks, resources)
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
        self.retrieve_chunk_if_exists_with_options(chunk_indices, self.codec_options(), resources)
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn retrieve_array_subset_into(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.retrieve_array_subset_into_planned(
            array_subset,
            output_target,
            self.codec_options(),
            resources,
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

impl<TStorage: ?Sized + ReadableStorageTraits + 'static> Array<TStorage> {
    pub(in crate::array) fn retrieve_chunk_with_options<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let (bytes, shape) = self.retrieve_chunk_planned(chunk_indices, options, resources)?;
        T::from_array_bytes(bytes, bytemuck::must_cast_slice(&shape), self.data_type())
    }

    pub(in crate::array) fn retrieve_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        self.retrieve_chunk_into_planned(chunk_indices, output_target, options, resources)
    }

    pub(in crate::array) fn retrieve_partial_chunk_with_options<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<T, ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let chunk_shape_u64 = bytemuck::must_cast_slice(&chunk_shape);
        if let Some(chunk_subset) = indexer.as_array_subset()
            && !chunk_subset.inbounds_shape(chunk_shape_u64)
        {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.to_array_subset(),
                chunk_shape_u64.to_vec(),
            ));
        }

        // Whole chunks are recognised by the plan
        let bytes = self.retrieve_partial_chunk_planned(
            chunk_indices,
            &chunk_shape,
            indexer,
            options,
            resources,
        )?;
        T::from_array_bytes(bytes, &indexer.output_shape(), self.data_type())
    }

    pub(in crate::array) fn retrieve_partial_chunk_into_with_options(
        &self,
        chunk_indices: &[u64],
        indexer: &dyn Indexer,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), ArrayError> {
        let chunk_shape = self.chunk_shape(chunk_indices)?;
        let chunk_shape_u64 = bytemuck::must_cast_slice(&chunk_shape);
        if let Some(chunk_subset) = indexer.as_array_subset()
            && !chunk_subset.inbounds_shape(chunk_shape_u64)
        {
            return Err(ArrayError::InvalidArraySubset(
                chunk_subset.to_array_subset(),
                chunk_shape_u64.to_vec(),
            ));
        }

        // Whole chunks are recognised by the plan
        self.retrieve_partial_chunk_into_planned(
            chunk_indices,
            &chunk_shape,
            indexer,
            output_target,
            options,
            resources,
        )
    }

    pub(in crate::array) fn retrieve_chunk_if_exists_with_options<T: FromArrayBytes>(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<T>, ArrayError> {
        if chunk_indices.len() != self.dimensionality() {
            return Err(ArrayError::InvalidChunkGridIndicesError(
                chunk_indices.to_vec(),
            ));
        }

        let shape = self.chunk_shape(chunk_indices)?;
        self.retrieve_chunk_if_exists_planned(chunk_indices, options, resources)?
            .map(|bytes| {
                T::from_array_bytes(bytes, bytemuck::must_cast_slice(&shape), self.data_type())
            })
            .transpose()
    }

    pub(in crate::array) fn partial_decoder_with_options(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, ArrayError> {
        self.planned_partial_decoder(chunk_indices, options, resources)
    }
}

#[cfg(test)]
mod tests {
    #![expect(clippy::single_range_in_vec_init)]
    use std::num::NonZeroU64;
    use std::sync::Arc;

    use super::*;
    use crate::array::{ArrayBuilder, ArraySubset, ChunkGridDecodedRef, data_type};
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
