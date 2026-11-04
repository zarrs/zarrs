#![allow(missing_docs)]

use std::error::Error;
use std::sync::Arc;

use zarrs::array::storage_transformer::{
    StorageTransformer, StorageTransformerChain, StorageTransformerPlugin, StorageTransformerTraits,
};
use zarrs::array::{Array, ArrayBuilder, data_type};
use zarrs::metadata::Configuration;
use zarrs::metadata::v3::MetadataV3;
use zarrs::node::{NodePath, meta_key_v3};
use zarrs::plugin::PluginCreateError;
use zarrs::storage::store::MemoryStore;
#[cfg(feature = "async")]
use zarrs::storage::{
    AsyncListableStorage, AsyncReadableStorage, AsyncReadableWritableStorage, AsyncWritableStorage,
};
use zarrs::storage::{
    ListableStorage, ReadableStorage, ReadableWritableStorage, StorageError, WritableStorage,
    WritableStorageTraits,
};

type TestResult = Result<(), Box<dyn Error>>;

zarrs::plugin::impl_extension_aliases!(UnusableStorageTransformer,
    v3: "zarrs.test_unusable", []
);

inventory::submit! {
    StorageTransformerPlugin::new::<UnusableStorageTransformer>()
}

/// A storage transformer that cannot create a transformed storage.
///
/// Any operation that goes through the storage transformers of an array fails.
#[derive(Debug)]
struct UnusableStorageTransformer;

fn unusable() -> StorageError {
    StorageError::Other("the storage transformer is unusable".to_string())
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(
    all(feature = "async", target_arch = "wasm32"),
    async_trait::async_trait(?Send)
)]
impl StorageTransformerTraits for UnusableStorageTransformer {
    fn create(
        _metadata: &MetadataV3,
        _path: &NodePath,
    ) -> Result<StorageTransformer, PluginCreateError> {
        Ok(Arc::new(Self))
    }

    fn configuration(&self) -> Configuration {
        Configuration::default()
    }

    fn create_readable_transformer(
        self: Arc<Self>,
        _storage: ReadableStorage,
    ) -> Result<ReadableStorage, StorageError> {
        Err(unusable())
    }

    fn create_writable_transformer(
        self: Arc<Self>,
        _storage: WritableStorage,
    ) -> Result<WritableStorage, StorageError> {
        Err(unusable())
    }

    fn create_readable_writable_transformer(
        self: Arc<Self>,
        _storage: ReadableWritableStorage,
    ) -> Result<ReadableWritableStorage, StorageError> {
        Err(unusable())
    }

    fn create_listable_transformer(
        self: Arc<Self>,
        _storage: ListableStorage,
    ) -> Result<ListableStorage, StorageError> {
        Err(unusable())
    }

    #[cfg(feature = "async")]
    async fn create_async_readable_transformer(
        self: Arc<Self>,
        _storage: AsyncReadableStorage,
    ) -> Result<AsyncReadableStorage, StorageError> {
        Err(unusable())
    }

    #[cfg(feature = "async")]
    async fn create_async_writable_transformer(
        self: Arc<Self>,
        _storage: AsyncWritableStorage,
    ) -> Result<AsyncWritableStorage, StorageError> {
        Err(unusable())
    }

    #[cfg(feature = "async")]
    async fn create_async_readable_writable_transformer(
        self: Arc<Self>,
        _storage: AsyncReadableWritableStorage,
    ) -> Result<AsyncReadableWritableStorage, StorageError> {
        Err(unusable())
    }

    #[cfg(feature = "async")]
    async fn create_async_listable_transformer(
        self: Arc<Self>,
        _storage: AsyncListableStorage,
    ) -> Result<AsyncListableStorage, StorageError> {
        Err(unusable())
    }
}

fn array_builder() -> ArrayBuilder {
    let mut builder = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint8(), 0u8);
    builder.storage_transformers(StorageTransformerChain::new(vec![Arc::new(
        UnusableStorageTransformer,
    )]));
    builder
}

/// All array I/O goes through the storage transformers of an array, including storing and erasing
/// metadata. The exception is reading metadata in `Array::[async_]open`, because the storage
/// transformers of an array are only known after its metadata is read.
#[test]
fn metadata_uses_storage_transformers() -> TestResult {
    let store = Arc::new(MemoryStore::default());
    let array = array_builder().build(store.clone(), "/array")?;
    assert!(array.store_metadata().is_err());

    let key = meta_key_v3(array.path());
    store.set(&key, serde_json::to_vec(array.metadata())?.into())?;
    let opened = Array::open(store.clone(), "/array")?;
    assert_eq!(
        opened.storage_transformers().create_metadatas(),
        array.storage_transformers().create_metadatas()
    );

    assert!(opened.erase_metadata().is_err());
    Array::open(store, "/array")?;
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_metadata_uses_storage_transformers() -> TestResult {
    use zarrs::storage::AsyncWritableStorageTraits;

    let store = Arc::new(zarrs::storage::store::AsyncMemoryStore::new());
    let array = array_builder().build(store.clone(), "/array")?;
    assert!(array.async_store_metadata().await.is_err());

    let key = meta_key_v3(array.path());
    store
        .set(&key, serde_json::to_vec(array.metadata())?.into())
        .await?;
    let opened = Array::async_open(store.clone(), "/array").await?;
    assert_eq!(
        opened.storage_transformers().create_metadatas(),
        array.storage_transformers().create_metadatas()
    );

    assert!(opened.async_erase_metadata().await.is_err());
    Array::async_open(store, "/array").await?;
    Ok(())
}
