#![allow(missing_docs)]
#![cfg(feature = "async")]

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use object_store::memory::InMemory;
use zarrs::array::codec::array_to_bytes::sharding::{
    AsyncShardingPartialDecoder, ShardingCodecBound, ShardingCodecBuilder,
};
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView,
    ArrayMetadataOptions, ArraySubset, AsyncArrayReadOps, AsyncArrayUpdateOps, CodecOptions,
    FillValue, data_type,
};
use zarrs::config::MetadataEraseVersion;
use zarrs::storage::{AsyncReadableStorageTraits, StorageHandle};
use zarrs_object_store::AsyncObjectStore;

type AsyncStore = AsyncObjectStore<InMemory>;
type TestResult = Result<(), Box<dyn Error>>;

const fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

fn fixed_fixture() -> Array<AsyncStore> {
    let store = Arc::new(AsyncObjectStore::new(InMemory::new()));
    let mut builder = ArrayBuilder::new(vec![5, 5], vec![3, 3], data_type::uint8(), 0u8);
    builder.subchunk_shape(vec![1, 1]);
    builder.build(store, "/array").unwrap()
}

async fn populate<A: AsyncArrayUpdateOps>(array: &A) -> TestResult {
    array
        .async_store_chunk(&[0, 0], &[1u8, 2, 3, 6, 7, 8, 11, 12, 13])
        .await?;
    array
        .async_store_chunk_opt(
            &[0, 1],
            &[4u8, 5, 0, 9, 10, 0, 14, 15, 0],
            &CodecOptions::default(),
        )
        .await?;
    array
        .async_store_chunks(
            &ArraySubset::new_with_ranges(&[1..2, 0..2]),
            &[
                16u8, 17, 18, 19, 20, 0, 21, 22, 23, 24, 25, 0, 0, 0, 0, 0, 0, 0,
            ],
        )
        .await?;
    Ok(())
}

async fn retrieve_array_subset_into<A: AsyncArrayReadOps>(
    array: &A,
    subset: &ArraySubset,
    explicit_options: bool,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let shape = subset.shape().to_vec();
    let mut output = vec![0; subset.num_elements_usize()];
    {
        let output_slice = unsafe_cell_slice::UnsafeCellSlice::new(&mut output);
        let full_subset = ArraySubset::new_with_shape(shape.clone());
        let mut view = unsafe {
            // SAFETY: this is the only view over output and covers it exactly.
            ArrayBytesFixedDisjointView::new(output_slice, 1, &shape, full_subset)?
        };
        let target = ArrayBytesDecodeIntoTarget::Fixed(&mut view);
        if explicit_options {
            array
                .async_retrieve_array_subset_into_opt(subset, target, &CodecOptions::default())
                .await?;
        } else {
            array
                .async_retrieve_array_subset_into(subset, target)
                .await?;
        }
    }
    Ok(output)
}

async fn exercise_async_read_ops<A: AsyncArrayUpdateOps>(array: &A) -> TestResult {
    populate(array).await?;
    let options = CodecOptions::default();
    let chunk_subset = ArraySubset::new_with_ranges(&[1..3, 1..3]);
    let chunks = ArraySubset::new_with_ranges(&[0..1, 0..2]);
    let array_subset = ArraySubset::new_with_ranges(&[1..4, 1..4]);

    assert_eq!(
        array.async_retrieve_chunk::<Vec<u8>>(&[0, 0]).await?,
        [1, 2, 3, 6, 7, 8, 11, 12, 13]
    );
    assert_eq!(
        array
            .async_retrieve_chunk_opt::<Vec<u8>>(&[0, 1], &options)
            .await?,
        [4, 5, 0, 9, 10, 0, 14, 15, 0]
    );
    assert_eq!(
        array
            .async_retrieve_chunk_if_exists::<Vec<u8>>(&[1, 0])
            .await?,
        Some(vec![16, 17, 18, 21, 22, 23, 0, 0, 0])
    );
    array.async_erase_chunk(&[1, 1]).await?;
    assert_eq!(
        array
            .async_retrieve_chunk_if_exists_opt::<Vec<u8>>(&[1, 1], &options)
            .await?,
        None
    );
    assert_eq!(
        array
            .async_retrieve_chunk_subset::<Vec<u8>>(&[0, 0], &chunk_subset)
            .await?,
        [7, 8, 12, 13]
    );
    assert_eq!(
        array
            .async_retrieve_chunk_subset_opt::<Vec<u8>>(&[0, 0], &chunk_subset, &options)
            .await?,
        [7, 8, 12, 13]
    );
    assert_eq!(
        array.async_retrieve_chunks::<Vec<u8>>(&chunks).await?,
        [1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 15, 0]
    );
    assert_eq!(
        array
            .async_retrieve_chunks_opt::<Vec<u8>>(&chunks, &options)
            .await?,
        [1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 15, 0]
    );
    assert_eq!(
        array
            .async_retrieve_array_subset::<Vec<u8>>(&array_subset)
            .await?,
        [7, 8, 9, 12, 13, 14, 17, 18, 0]
    );
    assert_eq!(
        array
            .async_retrieve_array_subset_opt::<Vec<u8>>(&array_subset, &options)
            .await?,
        [7, 8, 9, 12, 13, 14, 17, 18, 0]
    );
    assert_eq!(
        retrieve_array_subset_into(array, &array_subset, false).await?,
        [7, 8, 9, 12, 13, 14, 17, 18, 0]
    );
    assert_eq!(
        retrieve_array_subset_into(array, &array_subset, true).await?,
        [7, 8, 9, 12, 13, 14, 17, 18, 0]
    );
    assert_eq!(
        array
            .async_retrieve_subchunk_opt::<Vec<u8>>(&[1, 1], &options)
            .await?,
        [7]
    );
    assert_eq!(
        array
            .async_retrieve_subchunks_opt::<Vec<u8>>(
                &ArraySubset::new_with_ranges(&[1..3, 1..3]),
                &options,
            )
            .await?,
        [7, 8, 12, 13]
    );
    assert!(array.async_retrieve_encoded_chunk(&[0, 0]).await?.is_some());
    assert_eq!(
        array
            .async_retrieve_encoded_chunks_opt(&chunks, &options)
            .await?
            .len(),
        chunks.num_elements_usize()
    );
    let decoder = array.async_partial_decoder(&[0, 0]).await?;
    assert!(decoder.exists().await?);
    assert_eq!(
        decoder
            .partial_decode(&chunk_subset, &options)
            .await?
            .into_fixed()?
            .as_ref(),
        &[7, 8, 12, 13]
    );
    assert!(
        array
            .async_partial_decoder_opt(&[0, 0], &options)
            .await?
            .exists()
            .await?
    );
    Ok(())
}

async fn sharding_partial_decoder<A: AsyncArrayReadOps>(
    array: &A,
) -> Result<AsyncShardingPartialDecoder, Box<dyn Error>>
where
    A::Storage: AsyncReadableStorageTraits + 'static,
{
    let codecs_bound = array.codecs_bound();
    let sharding_codec = codecs_bound
        .array_to_bytes_codec()
        .as_any()
        .downcast_ref::<ShardingCodecBound>()
        .ok_or("array-to-bytes codec is not sharding")?;
    let storage_handle = Arc::new(StorageHandle::new(array.storage()));
    let storage_transformer = array
        .storage_transformers()
        .create_async_readable_transformer(storage_handle)
        .await?;
    let input_handle = Arc::new((storage_transformer, array.chunk_key(&[0, 0])));

    Ok(AsyncShardingPartialDecoder::new(
        input_handle,
        array.chunk_shape(&[0, 0])?,
        sharding_codec.subchunk_shape().clone(),
        sharding_codec.inner_codecs().clone(),
        sharding_codec.index_codecs(),
        sharding_codec.index_location(),
        array.codec_options(),
        sharding_codec.options().clone(),
    )
    .await?)
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn sharding_partial_decoder_retrieve_subchunk_encoded() -> TestResult {
    let array = fixed_fixture();
    populate(&array).await?;

    let decoder = sharding_partial_decoder(&array).await?;

    assert!(decoder.retrieve_subchunk_encoded(&[1, 1]).await?.is_some());
    assert!(decoder.retrieve_subchunk_encoded(&[3, 3]).await.is_err());
    Ok(())
}

#[tokio::test]
#[cfg_attr(miri, ignore)]
async fn sharding_partial_decoder_retrieve_subchunk_encoded_missing() -> TestResult {
    let array = fixed_fixture();

    let decoder = sharding_partial_decoder(&array).await?;
    assert_eq!(decoder.retrieve_subchunk_encoded(&[0, 0]).await?, None);

    array
        .async_store_chunk(&[0, 0], &[1u8, 0, 0, 0, 0, 0, 0, 0, 0])
        .await?;
    let decoder = sharding_partial_decoder(&array).await?;
    assert_eq!(decoder.retrieve_subchunk_encoded(&[0, 1]).await?, None);
    Ok(())
}

async fn exercise_async_write_update_ops<A: AsyncArrayUpdateOps>(array: &A) -> TestResult {
    let options = CodecOptions::default();
    array.async_store_metadata().await?;
    array
        .async_store_metadata_opt(&ArrayMetadataOptions::default())
        .await?;
    array.async_erase_metadata().await?;
    array.async_store_metadata().await?;
    array
        .async_erase_metadata_opt(MetadataEraseVersion::All)
        .await?;
    array.async_store_metadata().await?;

    array.async_store_chunk(&[0, 0], &[1u8; 9]).await?;
    array
        .async_store_chunk_opt(&[0, 1], &[2u8; 9], &options)
        .await?;
    array
        .async_store_chunks_opt(
            &ArraySubset::new_with_ranges(&[1..2, 0..2]),
            &[3u8; 18],
            &options,
        )
        .await?;

    array
        .async_store_chunk_subset(
            &[0, 0],
            &ArraySubset::new_with_ranges(&[1..2, 1..3]),
            &[4u8, 5],
        )
        .await?;
    array
        .async_store_chunk_subset_opt(
            &[0, 0],
            &ArraySubset::new_with_ranges(&[2..3, 0..1]),
            &[6u8],
            &options,
        )
        .await?;
    array
        .async_store_array_subset(&ArraySubset::new_with_ranges(&[0..1, 0..2]), &[7u8, 8])
        .await?;
    array
        .async_store_array_subset_opt(
            &ArraySubset::new_with_ranges(&[4..5, 4..5]),
            &[9u8],
            &options,
        )
        .await?;
    assert_eq!(
        array
            .async_readable()
            .async_retrieve_chunk::<Vec<u8>>(&[1, 1])
            .await?,
        [3, 3, 3, 3, 9, 3, 3, 3, 3]
    );

    let encoder = array.async_partial_encoder(&[0, 0], &options).await?;
    let _ = encoder.supports_partial_encode();
    encoder
        .partial_encode(
            &ArraySubset::new_with_ranges(&[0..1, 2..3]),
            &vec![10u8].into(),
            &options,
        )
        .await?;
    assert_eq!(array.async_retrieve_chunk::<Vec<u8>>(&[0, 0]).await?[2], 10);

    let encoded = array.async_retrieve_encoded_chunk(&[0, 0]).await?.unwrap();
    unsafe {
        // SAFETY: bytes were produced by the same array and chunk configuration.
        array.async_store_encoded_chunk(&[0, 1], encoded).await?;
    }
    let _ = array.async_compact_chunk(&[0, 0], &options).await?;
    array.async_erase_chunk(&[0, 1]).await?;
    assert!(
        array
            .async_retrieve_chunk_if_exists::<Vec<u8>>(&[0, 1])
            .await?
            .is_none()
    );
    array
        .async_erase_chunks(&ArraySubset::new_with_ranges(&[1..2, 0..2]))
        .await?;
    assert!(
        array
            .async_retrieve_chunk_if_exists::<Vec<u8>>(&[1, 0])
            .await?
            .is_none()
    );
    Ok(())
}

async fn exercise_variable_length_ops<A: AsyncArrayUpdateOps>(array: &A) -> TestResult {
    array
        .async_store_chunk(&[0, 0], &["a", "bb", "ccc", "dddd"])
        .await?;
    array
        .async_store_chunk(&[0, 1], &["eeeee", "ffffff", "ggggggg", "hhhhhhhh"])
        .await?;
    assert_eq!(
        array
            .async_retrieve_array_subset::<Vec<String>>(&ArraySubset::new_with_ranges(&[
                1..3,
                1..3
            ]))
            .await?,
        ["dddd", "ggggggg", "", ""]
    );
    array
        .async_store_array_subset(
            &ArraySubset::new_with_ranges(&[1..3, 1..3]),
            &["updated", "values", "across", "chunks"],
        )
        .await?;
    assert_eq!(
        array
            .async_retrieve_array_subset::<Vec<String>>(&ArraySubset::new_with_ranges(&[
                1..3,
                1..3
            ]))
            .await?,
        ["updated", "values", "across", "chunks"]
    );
    Ok(())
}

async fn exercise_optional_ops<A: AsyncArrayUpdateOps>(array: &A) -> TestResult {
    array
        .async_store_chunk(&[0, 0], &[Some(1u8), None, Some(3), Some(4)])
        .await?;
    array
        .async_store_chunk(&[0, 1], &[None, Some(6u8), None, Some(8)])
        .await?;
    assert_eq!(
        array
            .async_retrieve_array_subset::<Vec<Option<u8>>>(&ArraySubset::new_with_ranges(&[
                0..2,
                1..3
            ]))
            .await?,
        [None, None, Some(4), None]
    );
    array
        .async_store_array_subset(
            &ArraySubset::new_with_ranges(&[0..2, 1..3]),
            &[Some(9u8), None, None, Some(10)],
        )
        .await?;
    assert_eq!(
        array
            .async_retrieve_array_subset::<Vec<Option<u8>>>(&ArraySubset::new_with_ranges(&[
                0..2,
                1..3
            ]))
            .await?,
        [Some(9), None, None, Some(10)]
    );
    Ok(())
}

#[tokio::test]
async fn array_implements_complete_async_operation_suite() -> TestResult {
    let array = fixed_fixture();
    exercise_async_read_ops(&array).await?;
    exercise_async_write_update_ops(&array).await
}

#[tokio::test]
async fn array_supports_async_variable_length_operation_suite() -> TestResult {
    let store = Arc::new(AsyncObjectStore::new(InMemory::new()));
    let array = ArrayBuilder::new(vec![4, 4], vec![2, 2], data_type::string(), "")
        .build(store, "/array")?;
    exercise_variable_length_ops(&array).await
}

#[tokio::test]
async fn array_supports_async_optional_operation_suite() -> TestResult {
    let store = Arc::new(AsyncObjectStore::new(InMemory::new()));
    let array = ArrayBuilder::new(
        vec![4, 4],
        vec![2, 2],
        data_type::uint8().to_optional(),
        FillValue::from(None::<u8>),
    )
    .build(store, "/array")?;
    exercise_optional_ops(&array).await
}

#[tokio::test]
async fn array_supports_async_nested_subchunk_grid_levels() -> TestResult {
    let store = Arc::new(AsyncObjectStore::new(InMemory::new()));
    let data_type = data_type::uint16();
    let inner = ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build_arc();
    let mut outer = ShardingCodecBuilder::new(vec![nz(4), nz(4)], &data_type);
    outer.array_to_bytes_codec(inner);
    let mut builder = ArrayBuilder::new(vec![8, 8], vec![8, 8], data_type, 0u16);
    builder.array_to_bytes_codec(outer.build_arc());
    let array = builder.build(store, "/nested")?;
    array
        .async_store_array_subset(&array.subset_all(), &(0..64).collect::<Vec<u16>>())
        .await?;

    assert_eq!(array.subchunk_grids().len(), 2);
    assert_eq!(array.subchunk_shape_at_level(1), Some(vec![nz(2), nz(2)]));
    let options = CodecOptions::default();
    assert_eq!(
        array
            .async_retrieve_subchunk_at_level_opt::<Vec<u16>>(1, &[2, 3], &options)
            .await?,
        [38, 39, 46, 47]
    );
    assert_eq!(
        array
            .async_local_subchunk_grid_at_level(1, &[0, 0], &options)
            .await?
            .unwrap()
            .grid_shape(),
        &[4, 4]
    );

    Ok(())
}
