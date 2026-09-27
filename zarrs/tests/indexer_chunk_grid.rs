//! Chunk-grid operations with generic indexers.

use std::sync::Arc;

use zarrs::array::{ArrayBuilder, ArrayIndices, data_type};
use zarrs::storage::store::MemoryStore;

#[test]
fn chunk_grid_ops_accept_scattered_indices() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint16(), 0u16)
        .build(store, "/array")?;
    for chunk in [[0u64, 0], [0, 1], [1, 0], [1, 1]] {
        array.store_chunk(&chunk, &[1u16, 1, 1, 1])?;
    }

    let scattered: Vec<ArrayIndices> = vec![vec![1, 0], vec![0, 1]];
    let encoded = array.retrieve_encoded_chunks(&scattered)?;
    assert_eq!(encoded[0], array.retrieve_encoded_chunk(&[1, 0])?);
    assert_eq!(encoded[1], array.retrieve_encoded_chunk(&[0, 1])?);
    array.erase_chunks(&scattered)?;
    assert!(array.retrieve_encoded_chunk(&[0, 0])?.is_some());
    assert!(array.retrieve_encoded_chunk(&[0, 1])?.is_none());
    assert!(array.retrieve_encoded_chunk(&[1, 0])?.is_none());
    assert!(array.retrieve_encoded_chunk(&[1, 1])?.is_some());

    for invalid in [vec![vec![9, 9]], vec![vec![0]]] {
        assert!(array.retrieve_encoded_chunks(&invalid).is_err());
        assert!(array.erase_chunks(&invalid).is_err());
    }
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_chunk_grid_ops_accept_scattered_indices() -> Result<(), Box<dyn std::error::Error>> {
    use zarrs::storage::store::AsyncMemoryStore;

    let store = Arc::new(AsyncMemoryStore::new());
    let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::uint16(), 0u16)
        .build(store, "/array")?;
    array.async_store_chunk(&[0, 1], &[1u16, 1, 1, 1]).await?;
    array.async_store_chunk(&[1, 0], &[2u16, 2, 2, 2]).await?;

    let scattered: Vec<ArrayIndices> = vec![vec![1, 0], vec![0, 1]];
    let encoded = array.async_retrieve_encoded_chunks(&scattered).await?;
    assert_eq!(
        encoded[0],
        array.async_retrieve_encoded_chunk(&[1, 0]).await?
    );
    assert_eq!(
        encoded[1],
        array.async_retrieve_encoded_chunk(&[0, 1]).await?
    );
    array.async_erase_chunks(&scattered).await?;
    assert!(
        array
            .async_retrieve_encoded_chunks(&scattered)
            .await?
            .iter()
            .all(Option::is_none)
    );
    Ok(())
}
