//! Array chunk writes with generic indexers.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::builder::ArrayBuilderFillValue;
use zarrs::array::codec::{ShardingCodecBuilder, SqueezeCodec};
use zarrs::array::{Array, ArrayBuilder, ArrayIndices, DataType, data_type};
use zarrs::storage::store::MemoryStore;
use zarrs_codec::CodecOptions;

#[test]
fn chunk_write_updates_scattered_indices() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint16(), 0u16)
        .build(store, "/array")?;
    array.store_chunk(&[0, 0], (0u16..16).collect::<Vec<_>>())?;
    let scattered: Vec<ArrayIndices> = vec![vec![0, 1], vec![3, 3], vec![1, 0], vec![2, 2]];
    array.store_partial_chunk(&[0, 0], &scattered, &[100u16, 101, 102, 103])?;
    assert_eq!(
        array.retrieve_chunk::<Vec<u16>>(&[0, 0])?,
        [0, 100, 2, 3, 102, 5, 6, 7, 8, 9, 103, 11, 12, 13, 14, 101]
    );
    Ok(())
}

/// A 4x4 array with a single shard of 2x2 subchunks.
fn sharded_array(
    data_type: &DataType,
    fill_value: impl Into<ArrayBuilderFillValue>,
    partial_encoding: bool,
) -> Result<Array<MemoryStore>, Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type.clone(), fill_value)
        .array_to_bytes_codec(
            ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], data_type).build_arc(),
        )
        .build(store, "/array")?
        .with_codec_options(
            CodecOptions::default().with_experimental_partial_encoding(partial_encoding),
        );
    Ok(array)
}

#[test]
fn partial_encoding_sharded_scattered_write() -> Result<(), Box<dyn std::error::Error>> {
    let expected = sharded_array(&data_type::uint16(), 0u16, false)?;
    let array = sharded_array(&data_type::uint16(), 0u16, true)?;

    // The bottom-right subchunk is empty
    let mut original = (0u16..16).collect::<Vec<_>>();
    for i in [10, 11, 14, 15] {
        original[i] = 0;
    }

    // Writes spanning multiple subchunks (including the empty subchunk) with a duplicate index,
    // then a write that empties the top-right subchunk
    let writes: [(Vec<ArrayIndices>, Vec<u16>); 2] = [
        (
            vec![vec![0, 1], vec![3, 3], vec![1, 0], vec![2, 2], vec![0, 1]],
            vec![100, 101, 102, 103, 104],
        ),
        (
            vec![vec![0, 2], vec![0, 3], vec![1, 2], vec![1, 3]],
            vec![0, 0, 0, 0],
        ),
    ];
    for array in [&expected, &array] {
        array.store_chunk(&[0, 0], &original)?;
    }
    for (indices, values) in writes {
        for array in [&expected, &array] {
            array.store_partial_chunk(&[0, 0], &indices, &values)?;
        }
        assert_eq!(
            array.retrieve_chunk::<Vec<u16>>(&[0, 0])?,
            expected.retrieve_chunk::<Vec<u16>>(&[0, 0])?
        );
    }
    assert_eq!(
        array.retrieve_chunk::<Vec<u16>>(&[0, 0])?,
        [0, 104, 0, 0, 102, 5, 0, 0, 8, 9, 103, 0, 12, 13, 0, 101]
    );
    Ok(())
}

#[test]
fn partial_encoding_sharded_scattered_write_string() -> Result<(), Box<dyn std::error::Error>> {
    let expected = sharded_array(&data_type::string(), "", false)?;
    let array = sharded_array(&data_type::string(), "", true)?;
    let original = [
        "a", "bb", "", "", "ccc", "d", "", "", "", "", "", "", "", "", "", "",
    ];
    let indices: Vec<ArrayIndices> = vec![vec![3, 3], vec![0, 0], vec![1, 3], vec![0, 0]];
    let values = ["xyz", "q", "long string", "w"];
    for array in [&expected, &array] {
        array.store_chunk(&[0, 0], &original)?;
        array.store_partial_chunk(&[0, 0], &indices, &values)?;
    }
    let chunk = array.retrieve_chunk::<Vec<String>>(&[0, 0])?;
    assert_eq!(chunk, expected.retrieve_chunk::<Vec<String>>(&[0, 0])?);
    assert_eq!(chunk[0], "w");
    assert_eq!(chunk[7], "long string");
    assert_eq!(chunk[15], "xyz");
    Ok(())
}

#[test]
fn partial_encoding_sharded_rejects_out_of_bounds_scattered_write()
-> Result<(), Box<dyn std::error::Error>> {
    let array = sharded_array(&data_type::uint16(), 0u16, true)?;
    let original = (0u16..16).collect::<Vec<_>>();
    array.store_chunk(&[0, 0], &original)?;
    let scattered: Vec<ArrayIndices> = vec![vec![0, 1], vec![3, 4]];
    assert!(
        array
            .store_partial_chunk(&[0, 0], &scattered, &[100u16, 101])
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
            .store_partial_chunk(&[0, 0, 0], &oob, &[999u16])
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
        .async_store_partial_chunk(&[0, 0], &scattered, &[99u16, 88])
        .await?;
    let chunk = array.async_retrieve_chunk::<Vec<u16>>(&[0, 0]).await?;
    assert_eq!(chunk[15], 99);
    assert_eq!(chunk[1], 88);
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_partial_encoding_sharded_scattered_write() -> Result<(), Box<dyn std::error::Error>>
{
    use zarrs::storage::store::AsyncMemoryStore;

    let store = Arc::new(AsyncMemoryStore::new());
    let array = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint16(), 0u16)
        .array_to_bytes_codec(
            ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type::uint16())
                .build_arc(),
        )
        .build(store, "/array")?
        .with_codec_options(CodecOptions::default().with_experimental_partial_encoding(true));
    array
        .async_store_chunk(&[0, 0], &(0u16..16).collect::<Vec<_>>())
        .await?;
    let scattered: Vec<ArrayIndices> = vec![vec![3, 3], vec![0, 1], vec![2, 0], vec![0, 1]];
    array
        .async_store_partial_chunk(&[0, 0], &scattered, &[99u16, 88, 77, 66])
        .await?;
    assert_eq!(
        array.async_retrieve_chunk::<Vec<u16>>(&[0, 0]).await?,
        [0, 66, 2, 3, 4, 5, 6, 7, 77, 9, 10, 11, 12, 13, 14, 99]
    );
    Ok(())
}
