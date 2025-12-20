use std::num::NonZeroU64;

#[cfg(doc)]
use crate::array::TensorError;
use crate::array::{Array, ArrayError, Tensor, codec::CodecOptions};
use crate::array_subset::ArraySubset;
use crate::storage::AsyncReadableStorageTraits;

/// An async [`Array`] extension trait with methods that return `DLPack` managed tensors.
#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
pub trait AsyncArrayDlPackExt<TStorage: ?Sized + AsyncReadableStorageTraits + 'static>:
    private::Sealed
{
    /// Read and decode the `array_subset` of array into a `DLPack` tensor.
    ///
    /// See [`Array::retrieve_array_subset_opt`].
    ///
    /// # Errors
    /// Returns a [`TensorError`] if the chunk cannot be represented as a `DLPack` tensor.
    /// Otherwise returns standard [`Array::retrieve_array_subset_opt`] errors.
    async fn retrieve_array_subset_dlpack(
        &self,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Tensor, ArrayError>;

    /// Read and decode the chunk at `chunk_indices` into a `DLPack` tensor if it exists.
    ///
    /// See [`Array::retrieve_chunk_if_exists_opt`].
    ///
    /// # Errors
    /// Returns a [`TensorError`] if the chunk cannot be represented as a `DLPack` tensor.
    /// Otherwise returns standard [`Array::retrieve_chunk_if_exists_opt`] errors.
    async fn retrieve_chunk_if_exists_dlpack(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Tensor>, ArrayError>;

    /// Read and decode the chunk at `chunk_indices` into a `DLPack` tensor.
    ///
    /// See [`Array::retrieve_chunk_opt`].
    ///
    /// # Errors
    /// Returns a [`TensorError`] if the chunk cannot be represented as a `DLPack` tensor.
    /// Otherwise returns standard [`Array::retrieve_chunk_opt`] errors.
    async fn retrieve_chunk_dlpack(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Tensor, ArrayError>;

    /// Read and decode the chunks at `chunks` into a `DLPack` tensor.
    ///
    /// See [`Array::retrieve_chunks_opt`].
    ///
    /// # Errors
    /// Returns a [`TensorError`] if the chunk cannot be represented as a `DLPack` tensor.
    /// Otherwise returns standard [`Array::retrieve_chunks_opt`] errors.
    async fn retrieve_chunks_dlpack(
        &self,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Tensor, ArrayError>;
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> AsyncArrayDlPackExt<TStorage>
    for Array<TStorage>
{
    async fn retrieve_array_subset_dlpack(
        &self,
        array_subset: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Tensor, ArrayError> {
        let bytes = self
            .async_retrieve_array_subset_opt(array_subset, options)
            .await?
            .into_owned();
        let bytes = bytes.into_fixed()?;

        let shape: Vec<u64> = array_subset
            .shape()
            .iter()
            .map(|s| {
                NonZeroU64::new(*s).ok_or_else(|| {
                    ArrayError::InvalidArraySubset(array_subset.clone(), self.shape().to_vec())
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .map(|s| s.get())
            .collect();

        Ok(Tensor::new(bytes, self.data_type().clone(), shape))
    }

    async fn retrieve_chunk_if_exists_dlpack(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Option<Tensor>, ArrayError> {
        let Some(bytes) = self
            .async_retrieve_chunk_if_exists_opt(chunk_indices, options)
            .await?
        else {
            return Ok(None);
        };
        let bytes = bytes.into_owned();
        let bytes = bytes.into_fixed()?;
        let representation = self.chunk_array_representation(chunk_indices)?;
        let shape: Vec<u64> = representation.shape().iter().map(|s| s.get()).collect();
        Ok(Some(Tensor::new(bytes, self.data_type().clone(), shape)))
    }

    async fn retrieve_chunk_dlpack(
        &self,
        chunk_indices: &[u64],
        options: &CodecOptions,
    ) -> Result<Tensor, ArrayError> {
        let bytes = self
            .async_retrieve_chunk_opt(chunk_indices, options)
            .await?
            .into_owned();
        let bytes = bytes.into_fixed()?;
        let representation = self.chunk_array_representation(chunk_indices)?;
        let shape: Vec<u64> = representation.shape().iter().map(|s| s.get()).collect();
        Ok(Tensor::new(bytes, self.data_type().clone(), shape))
    }

    async fn retrieve_chunks_dlpack(
        &self,
        chunks: &ArraySubset,
        options: &CodecOptions,
    ) -> Result<Tensor, ArrayError> {
        let array_subset = self.chunks_subset(chunks)?;
        self.retrieve_array_subset_dlpack(&array_subset, options)
            .await
    }
}

mod private {
    use super::{Array, AsyncReadableStorageTraits};

    pub trait Sealed {}

    impl<TStorage: ?Sized + AsyncReadableStorageTraits + 'static> Sealed for Array<TStorage> {}
}
