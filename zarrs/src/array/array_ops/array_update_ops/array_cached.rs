use inherent::inherent;
use std::sync::Arc;

use super::super::*;
use super::ArrayUpdateOps;
use zarrs_codec::ArrayPartialEncoderTraits;

#[inherent]
impl<TStorage, C> ArrayUpdateOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + ReadableWritableStorageTraits + 'static,
    C: ChunkCache,
{
    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunk_subset<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
    ) -> Result<(), ArrayError> {
        self.store_chunk_subset_opt(
            chunk_indices,
            chunk_subset,
            chunk_subset_data,
            &CodecOptions::default(),
        )
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn store_chunk_subset_opt<'a, T: IntoArrayBytes<'a>>(
        &self,
        chunk_indices: &[u64],
        chunk_subset: &dyn ArraySubsetTraits,
        chunk_subset_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.array().store_chunk_subset_opt(
            chunk_indices,
            chunk_subset,
            chunk_subset_data,
            options,
        )?;
        self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn store_array_subset<'a, T: IntoArrayBytes<'a>>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
    ) -> Result<(), ArrayError> {
        self.store_array_subset_opt(array_subset, subset_data, &CodecOptions::default())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn store_array_subset_opt<'a, T: IntoArrayBytes<'a>>(
        &self,
        array_subset: &dyn ArraySubsetTraits,
        subset_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.array()
            .store_array_subset_opt(array_subset, subset_data, options)?;
        if let Some(chunks) = self.array().chunks_in_array_subset(array_subset)? {
            self.cache().invalidate_chunks(&chunks);
        } else {
            self.cache().invalidate();
        }
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn compact_chunk(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<bool, ArrayError> {
        let compacted = self.array().compact_chunk(chunk_indices, options)?;
        if compacted {
            self.cache().invalidate_chunk(chunk_indices);
        }
        Ok(compacted)
    }

    pub fn readable(&self) -> Array<dyn ReadableStorageTraits> {
        self.array().readable()
    }

    #[allow(clippy::missing_errors_doc)]
    pub fn partial_encoder(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, ArrayError> {
        self.array().partial_encoder(chunk_indices, options)
    }
}
