#![allow(missing_docs)]

use std::error::Error;

use zarrs_object_store::AsyncObjectStore;

#[tokio::test]
async fn memory() -> Result<(), Box<dyn Error>> {
    let store = AsyncObjectStore::new(object_store::memory::InMemory::new());
    zarrs_storage::store_test::async_store_write(&store).await?;
    zarrs_storage::store_test::async_store_read(&store).await?;
    zarrs_storage::store_test::async_store_list(&store).await?;
    Ok(())
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn filesystem() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let store = AsyncObjectStore::new(object_store::local::LocalFileSystem::new_with_prefix(
        path.path(),
    )?);
    zarrs_storage::store_test::async_store_write(&store).await?;
    zarrs_storage::store_test::async_store_read(&store).await?;
    zarrs_storage::store_test::async_store_list(&store).await?;
    Ok(())
}
