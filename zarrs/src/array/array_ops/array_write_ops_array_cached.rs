use ambisync::ambisync;
use inherent::inherent;

#[cfg(feature = "async")]
use super::AsyncArrayWriteOps;
use super::{ArrayWriteOps, *};

#[ambisync(
    sync(
        fns("async_{}"),
        types(
            AsyncArrayWriteOps => ArrayWriteOps,
            AsyncWritableStorageTraits => WritableStorageTraits,
        ),
    ),
    async(feature = "async"),
)]
#[inherent]
impl<TStorage, C> AsyncArrayWriteOps for ArrayCached<TStorage, C>
where
    TStorage: ?Sized + AsyncWritableStorageTraits + 'static,
    C: ChunkCache,
{
    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_metadata(&self) -> Result<(), StorageError> {
        self.async_store_metadata_opt(self.metadata_options()).await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_metadata_opt(
        &self,
        options: &ArrayMetadataOptions,
    ) -> Result<(), StorageError> {
        self.array().async_store_metadata_opt(options).await?;
        self.cache().invalidate();
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_metadata(&self) -> Result<(), StorageError> {
        self.async_erase_metadata_opt(self.metadata_erase_version())
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_metadata_opt(
        &self,
        options: MetadataEraseVersion,
    ) -> Result<(), StorageError> {
        self.array().async_erase_metadata_opt(options).await?;
        self.cache().invalidate();
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
    ) -> Result<(), ArrayError> {
        self.async_store_chunk_opt(chunk_indices, chunk_data, self.codec_options())
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunk_opt<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        chunk_indices: &[u64],
        chunk_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.array()
            .async_store_chunk_opt(chunk_indices, chunk_data, options)
            .await?;
        self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunks<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
    ) -> Result<(), ArrayError> {
        self.async_store_chunks_opt(chunks, chunks_data, self.codec_options())
            .await
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_store_chunks_opt<
        'a,
        #[sync_bounds(IntoArrayBytes<'a>)] T: IntoArrayBytes<'a> + MaybeSend,
    >(
        &self,
        chunks: &dyn ArraySubsetTraits,
        chunks_data: T,
        options: &CodecOptions,
    ) -> Result<(), ArrayError> {
        self.array()
            .async_store_chunks_opt(chunks, chunks_data, options)
            .await?;
        self.cache().invalidate_chunks(chunks);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_chunk(&self, chunk_indices: &[u64]) -> Result<(), StorageError> {
        self.array().async_erase_chunk(chunk_indices).await?;
        let _ = self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc)]
    pub async fn async_erase_chunks(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<(), StorageError> {
        self.array().async_erase_chunks(chunks).await?;
        let _ = self.cache().invalidate_chunks(chunks);
        Ok(())
    }

    #[allow(clippy::missing_errors_doc, clippy::missing_safety_doc)]
    pub async unsafe fn async_store_encoded_chunk(
        &self,
        chunk_indices: &[u64],
        encoded_chunk_bytes: bytes::Bytes,
    ) -> Result<(), ArrayError> {
        unsafe {
            self.array()
                .async_store_encoded_chunk(chunk_indices, encoded_chunk_bytes)
                .await?;
        }
        self.cache().invalidate_chunk(chunk_indices);
        Ok(())
    }
}
