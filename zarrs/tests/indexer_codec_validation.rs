//! Regression tests for partial codec indexer validation.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::{ShardingCodecBuilder, SqueezeCodec};
#[cfg(feature = "transpose")]
use zarrs::array::codec::{TransposeCodec, TransposeOrder};
use zarrs::array::{ArrayBuilder, ArrayIndices, data_type};
use zarrs::storage::store::MemoryStore;
use zarrs_codec::CodecOptions;

#[test]
fn squeeze_partial_codec_rejects_discarded_out_of_bounds_index()
-> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 1, 4], vec![4, 1, 4], data_type::uint16(), 0u16)
        .array_to_array_codecs(vec![Arc::new(SqueezeCodec::new())])
        .build(store, "/array")?;
    let original = (0u16..16).collect::<Vec<_>>();
    array.store_chunk(&[0, 0, 0], &original)?;

    let oob: Vec<ArrayIndices> = vec![vec![0, 7, 2]];
    let options = CodecOptions::default();
    assert!(
        array
            .partial_decoder(&[0, 0, 0])?
            .partial_decode(&oob, &options)
            .is_err()
    );
    let encoder = array.partial_encoder(&[0, 0, 0])?;
    assert!(
        encoder
            .partial_encode(&oob, &vec![99u8, 0].into(), &options)
            .is_err()
    );
    assert_eq!(array.retrieve_chunk::<Vec<u16>>(&[0, 0, 0])?, original);
    Ok(())
}

#[test]
fn absent_shard_partial_codec_rejects_out_of_bounds_index() -> Result<(), Box<dyn std::error::Error>>
{
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![4, 4], vec![4, 4], data_type::uint16(), 0u16)
        .array_to_bytes_codec(
            ShardingCodecBuilder::new(vec![NonZeroU64::new(2).unwrap(); 2], &data_type::uint16())
                .build_arc(),
        )
        .build(store, "/array")?;
    let oob: Vec<ArrayIndices> = vec![vec![0, 4]];
    assert!(
        array
            .partial_decoder(&[0, 0])?
            .partial_decode(&oob, &CodecOptions::default())
            .is_err()
    );
    Ok(())
}

#[cfg(feature = "transpose")]
#[test]
fn transpose_partial_codec_reports_decoded_shape() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let array = ArrayBuilder::new(vec![2, 8], vec![2, 8], data_type::uint16(), 0u16)
        .array_to_array_codecs(vec![Arc::new(TransposeCodec::new(TransposeOrder::new(
            &[1, 0],
        )?))])
        .build(store, "/array")?;
    array.store_chunk(&[0, 0], &(0u16..16).collect::<Vec<_>>())?;
    let oob: Vec<ArrayIndices> = vec![vec![3, 0]];
    let error = array
        .partial_decoder(&[0, 0])?
        .partial_decode(&oob, &CodecOptions::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("[2, 8]"), "{error}");
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_squeeze_partial_codec_rejects_discarded_index()
-> Result<(), Box<dyn std::error::Error>> {
    use zarrs::storage::store::AsyncMemoryStore;

    let store = Arc::new(AsyncMemoryStore::new());
    let array = ArrayBuilder::new(vec![4, 1, 4], vec![4, 1, 4], data_type::uint16(), 0u16)
        .array_to_array_codecs(vec![Arc::new(SqueezeCodec::new())])
        .build(store, "/array")?;
    array
        .async_store_chunk(&[0, 0, 0], &(0u16..16).collect::<Vec<_>>())
        .await?;
    let oob: Vec<ArrayIndices> = vec![vec![0, 7, 2]];
    assert!(
        array
            .async_partial_decoder(&[0, 0, 0])
            .await?
            .partial_decode(&oob, &CodecOptions::default())
            .await
            .is_err()
    );
    Ok(())
}
