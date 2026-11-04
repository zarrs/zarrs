#![allow(dead_code)]
use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::{CodecChain, CodecChainBound};
use zarrs::storage::StoreKey;
use zarrs_codec::{ArrayCodecTraits, CodecOptions, Resources};
pub(crate) struct TestCodec {
    pub chain: CodecChain,
    pub bound: Arc<CodecChainBound>,
}
impl std::ops::Deref for TestCodec {
    type Target = CodecChainBound;
    fn deref(&self) -> &Self::Target {
        &self.bound
    }
}
impl TestCodec {
    fn array<S: ?Sized + 'static>(
        &self,
        storage: Arc<S>,
        shape: &[NonZeroU64],
    ) -> zarrs::array::Array<S> {
        let shape: Vec<u64> = shape.iter().map(|s| s.get()).collect();
        let mut builder = zarrs::array::ArrayBuilder::new(
            shape.clone(),
            shape,
            self.bound.data_type().clone(),
            self.bound.fill_value().unwrap().clone(),
        );
        builder
            .array_to_array_codecs(self.chain.array_to_array_codecs().to_vec())
            .array_to_bytes_codec(self.chain.array_to_bytes_codec().clone())
            .bytes_to_bytes_codecs(self.chain.bytes_to_bytes_codecs().to_vec());
        builder.build(storage, "/").unwrap()
    }
    pub(crate) fn partial_encoder<S: zarrs::storage::ReadableWritableStorageTraits + 'static>(
        &self,
        handle: Arc<(Arc<S>, StoreKey)>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn zarrs_codec::ArrayPartialEncoderTraits>, zarrs::array::ArrayError> {
        self.array(handle.0.clone(), shape)
            .partial_encoder(&[0], &Resources::default())
    }
    pub(crate) fn partial_decoder<S: zarrs::storage::ReadableStorageTraits + 'static>(
        &self,
        handle: Arc<(Arc<S>, StoreKey)>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn zarrs_codec::ArrayPartialDecoderTraits>, zarrs::array::ArrayError> {
        self.array(handle.0.clone(), shape)
            .partial_decoder(&[0], &Resources::default())
    }
    #[cfg(feature = "async")]
    pub(crate) async fn async_partial_encoder<
        S: zarrs::storage::AsyncReadableWritableStorageTraits + 'static,
    >(
        &self,
        handle: Arc<(Arc<S>, StoreKey)>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn zarrs_codec::AsyncArrayPartialEncoderTraits>, zarrs::array::ArrayError>
    {
        self.array(handle.0.clone(), shape)
            .async_partial_encoder(&[0], &Resources::default())
            .await
    }
    #[cfg(feature = "async")]
    pub(crate) async fn async_partial_decoder<
        S: zarrs::storage::AsyncReadableStorageTraits + 'static,
    >(
        &self,
        handle: Arc<(Arc<S>, StoreKey)>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<Arc<dyn zarrs_codec::AsyncArrayPartialDecoderTraits>, zarrs::array::ArrayError>
    {
        self.array(handle.0.clone(), shape)
            .async_partial_decoder(&[0], &Resources::default())
            .await
    }
}
