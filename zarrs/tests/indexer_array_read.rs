//! Array chunk reads with generic indexers.

use std::sync::Arc;

use zarrs::array::{ArrayBuilder, ArrayIndices, ArraySubset, data_type};
use zarrs::storage::store::MemoryStore;

#[test]
fn chunk_read_accepts_scattered_indices() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint16(), 0u16)
        .build(store, "/array")?;
    array.store_chunk(&[0, 0], (0u16..16).collect::<Vec<_>>())?;

    let scattered: Vec<ArrayIndices> = vec![vec![0, 1], vec![3, 3], vec![1, 0], vec![2, 2]];
    assert_eq!(
        array.retrieve_chunk_subset::<Vec<u16>>(&[0, 0], &scattered)?,
        [1, 15, 4, 10]
    );
    assert_eq!(
        array.retrieve_chunk_subset::<Vec<u16>>(
            &[0, 0],
            &ArraySubset::new_with_ranges(&[0..2, 0..2])
        )?,
        [0, 1, 4, 5]
    );

    let oob: Vec<ArrayIndices> = vec![vec![0, 4]];
    assert!(
        array
            .retrieve_chunk_subset::<Vec<u16>>(&[0, 0], &oob)
            .is_err()
    );
    Ok(())
}

#[test]
fn decoded_cache_validates_absent_chunk_indexer() -> Result<(), Box<dyn std::error::Error>> {
    use zarrs::array::ArrayCached;
    use zarrs::array::chunk_cache::ChunkCacheDecodedLruChunkLimit;

    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![8, 8], vec![2, 2], data_type::uint16(), 0u16)
        .build(store, "/array")?;
    let cached = ArrayCached::new(Arc::new(array), ChunkCacheDecodedLruChunkLimit::new(4));
    let scattered: Vec<ArrayIndices> = vec![vec![0, 1], vec![1, 1]];
    assert_eq!(
        cached.retrieve_chunk_subset::<Vec<u16>>(&[0, 0], &scattered)?,
        [0, 0]
    );
    let oob: Vec<ArrayIndices> = vec![vec![0, 2]];
    assert!(
        cached
            .retrieve_chunk_subset::<Vec<u16>>(&[0, 0], &oob)
            .is_err()
    );
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_chunk_read_accepts_scattered_indices() -> Result<(), Box<dyn std::error::Error>> {
    use zarrs::storage::store::AsyncMemoryStore;

    let store = Arc::new(AsyncMemoryStore::new());
    let array = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint16(), 0u16)
        .build(store, "/array")?;
    array
        .async_store_chunk(&[0, 0], &(0u16..16).collect::<Vec<_>>())
        .await?;
    let scattered: Vec<ArrayIndices> = vec![vec![3, 3], vec![0, 1]];
    assert_eq!(
        array
            .async_retrieve_chunk_subset::<Vec<u16>>(&[0, 0], &scattered)
            .await?,
        [15, 1]
    );
    Ok(())
}
