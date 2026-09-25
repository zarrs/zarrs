//! Array chunk writes with generic indexers.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::{ShardingCodecBuilder, SqueezeCodec};
use zarrs::array::{ArrayBuilder, ArrayIndices, data_type};
use zarrs::storage::store::MemoryStore;
use zarrs_codec::CodecOptions;

#[test]
fn chunk_write_updates_scattered_indices() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint16(), 0u16)
        .build(store, "/array")?;
    array.store_chunk(&[0, 0], (0u16..16).collect::<Vec<_>>())?;
    let scattered: Vec<ArrayIndices> = vec![vec![0, 1], vec![3, 3], vec![1, 0], vec![2, 2]];
    array.store_chunk_subset(&[0, 0], &scattered, &[100u16, 101, 102, 103])?;
    assert_eq!(
        array.retrieve_chunk::<Vec<u16>>(&[0, 0])?,
        [0, 100, 2, 3, 102, 5, 6, 7, 8, 9, 103, 11, 12, 13, 14, 101]
    );
    Ok(())
}

#[test]
fn partial_encoding_rejects_sharded_scattered_write() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint16(), 0u16)
        .array_to_bytes_codec(
            ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type::uint16())
                .build_arc(),
        )
        .build(store, "/array")?
        .with_codec_options(CodecOptions::default().with_experimental_partial_encoding(true));
    let original = (0u16..16).collect::<Vec<_>>();
    array.store_chunk(&[0, 0], &original)?;
    let scattered: Vec<ArrayIndices> = vec![vec![0, 1], vec![3, 3]];
    assert!(
        array
            .store_chunk_subset(&[0, 0], &scattered, &[100u16, 101])
            .is_err()
    );
    assert_eq!(array.retrieve_chunk::<Vec<u16>>(&[0, 0])?, original);
    Ok(())
}

#[test]
fn squeeze_rejects_out_of_bounds_scattered_write() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 1, 4], vec![4, 1, 4], data_type::uint16(), 0u16)
        .array_to_array_codecs(vec![Arc::new(SqueezeCodec::new())])
        .build(store, "/array")?
        .with_codec_options(CodecOptions::default().with_experimental_partial_encoding(true));
    let original = (0u16..16).collect::<Vec<_>>();
    array.store_chunk(&[0, 0, 0], &original)?;
    let oob: Vec<ArrayIndices> = vec![vec![0, 7, 2]];
    assert!(
        array
            .store_chunk_subset(&[0, 0, 0], &oob, &[999u16])
            .is_err()
    );
    assert_eq!(array.retrieve_chunk::<Vec<u16>>(&[0, 0, 0])?, original);
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_chunk_write_updates_scattered_indices() -> Result<(), Box<dyn std::error::Error>> {
    use zarrs::storage::store::AsyncMemoryStore;

    let store = Arc::new(AsyncMemoryStore::new());
    let array = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint16(), 0u16)
        .build(store, "/array")?;
    array
        .async_store_chunk(&[0, 0], &(0u16..16).collect::<Vec<_>>())
        .await?;
    let scattered: Vec<ArrayIndices> = vec![vec![3, 3], vec![0, 1]];
    array
        .async_store_chunk_subset(&[0, 0], &scattered, &[99u16, 88])
        .await?;
    let chunk = array.async_retrieve_chunk::<Vec<u16>>(&[0, 0]).await?;
    assert_eq!(chunk[15], 99);
    assert_eq!(chunk[1], 88);
    Ok(())
}
