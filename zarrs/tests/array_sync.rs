#![allow(missing_docs)]
#![cfg(feature = "ndarray")]

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::CodecOptions;
use zarrs::array::{Array, ArrayBuilder, ArrayBytes, codec::ArrayCodecTraits, data_type};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::store::MemoryStore;

#[rustfmt::skip]
fn array_sync_read(array: &Array<MemoryStore>) -> Result<(), Box<dyn std::error::Error>> {
    assert!(array.data_type().eq(data_type::uint8().as_ref()));
    assert_eq!(array.fill_value().as_ne_bytes(), &[0u8]);
    assert_eq!(array.shape(), &[4, 4]);
    assert_eq!(array.chunk_shape(&[0, 0]).unwrap(), [NonZeroU64::new(2).unwrap(); 2]);
    assert_eq!(array.chunk_grid_shape(), &[2, 2]);

    let options = CodecOptions::default();

    // 1  2 | 3  4
    // 5  6 | 7  8
    // -----|-----
    // 9 10 | 0  0
    // 0  0 | 0  0
    array.store_chunk(&[0, 0], &[1u8, 2, 0, 0])?;
    array.store_chunk(&[0, 1], &[3u8, 4, 7, 8])?;
    array.store_array_subset(&ArraySubset::new_with_ranges(&[1..3, 0..2]), &[5u8, 6, 9, 10])?;

    assert!(array.retrieve_chunk::<ArrayBytes>(&[0, 0, 0]).is_err());
    assert_eq!(array.retrieve_chunk::<ArrayBytes>(&[0, 0])?, vec![1, 2, 5, 6].into());
    assert_eq!(array.retrieve_chunk::<ArrayBytes>(&[0, 1])?, vec![3, 4, 7, 8].into());
    assert_eq!(array.retrieve_chunk::<ArrayBytes>(&[1, 0])?, vec![9, 10, 0, 0].into());
    assert_eq!(array.retrieve_chunk::<ArrayBytes>(&[1, 1])?, vec![0, 0, 0, 0].into());

    assert!(array.retrieve_chunk_if_exists::<ArrayBytes>(&[0, 0, 0]).is_err());
    assert_eq!(array.retrieve_chunk_if_exists::<ArrayBytes>(&[0, 0])?, Some(vec![1, 2, 5, 6].into()));
    assert_eq!(array.retrieve_chunk_if_exists::<ArrayBytes>(&[0, 1])?, Some(vec![3, 4, 7, 8].into()));
    assert_eq!(array.retrieve_chunk_if_exists::<ArrayBytes>(&[1, 0])?, Some(vec![9, 10, 0, 0].into()));
    assert_eq!(array.retrieve_chunk_if_exists::<ArrayBytes>(&[1, 1])?, None);

    assert!(array.retrieve_chunk::<ndarray::ArrayD<u16>>(&[0, 0]).is_err());
    assert_eq!(array.retrieve_chunk::<ndarray::ArrayD<u8>>(&[0, 0])?, ndarray::array![[1, 2], [5, 6]].into_dyn());
    assert_eq!(array.retrieve_chunk::<ndarray::ArrayD<u8>>(&[0, 1])?, ndarray::array![[3, 4], [7, 8]].into_dyn());
    assert_eq!(array.retrieve_chunk::<ndarray::ArrayD<u8>>(&[1, 0])?, ndarray::array![[9, 10], [0, 0]].into_dyn());
    assert_eq!(array.retrieve_chunk::<ndarray::ArrayD<u8>>(&[1, 1])?, ndarray::array![[0, 0], [0, 0]].into_dyn());

    assert_eq!(array.retrieve_chunk_if_exists::<ndarray::ArrayD<u8>>(&[0, 0])?, Some(ndarray::array![[1, 2], [5, 6]].into_dyn()));
    assert_eq!(array.retrieve_chunk_if_exists::<ndarray::ArrayD<u8>>(&[0, 1])?, Some(ndarray::array![[3, 4], [7, 8]].into_dyn()));
    assert_eq!(array.retrieve_chunk_if_exists::<ndarray::ArrayD<u8>>(&[1, 0])?, Some(ndarray::array![[9, 10], [0, 0]].into_dyn()));
    assert_eq!(array.retrieve_chunk_if_exists::<ndarray::ArrayD<u8>>(&[1, 1])?, None);

    assert!(array.retrieve_chunk_subset::<ArrayBytes>(&[0, 0], &ArraySubset::new_with_ranges(&[0..2])).is_err());
    assert!(array.retrieve_chunk_subset::<ArrayBytes>(&[0, 0], &ArraySubset::new_with_ranges(&[0..3, 0..3])).is_err());
    assert_eq!(array.retrieve_chunk_subset::<ArrayBytes>(&[0, 0], &ArraySubset::new_with_ranges(&[0..2, 0..2]))?, vec![1, 2, 5, 6].into());
    assert_eq!(array.retrieve_chunk_subset::<ArrayBytes>(&[0, 0], &ArraySubset::new_with_ranges(&[0..1, 0..2]))?, vec![1, 2].into());
    assert_eq!(array.retrieve_chunk_subset::<ArrayBytes>(&[0, 0], &ArraySubset::new_with_ranges(&[0..2, 1..2]))?, vec![2, 6].into());

    assert!(array.retrieve_chunk_subset::<ndarray::ArrayD<u8>>(&[0, 0], &ArraySubset::new_with_ranges(&[0..3, 0..3])).is_err());
    assert!(array.retrieve_chunk_subset::<ndarray::ArrayD<u16>>(&[0, 0], &ArraySubset::new_with_ranges(&[0..2, 0..2])).is_err());
    assert_eq!(array.retrieve_chunk_subset::<ndarray::ArrayD<u8>>(&[0, 0], &ArraySubset::new_with_ranges(&[0..2, 0..2]))?, ndarray::array![[1, 2], [5, 6]].into_dyn());
    assert_eq!(array.retrieve_chunk_subset::<ndarray::ArrayD<u8>>(&[0, 0], &ArraySubset::new_with_ranges(&[0..1, 0..2]))?, ndarray::array![[1, 2]].into_dyn());
    assert_eq!(array.retrieve_chunk_subset::<ndarray::ArrayD<u8>>(&[0, 0], &ArraySubset::new_with_ranges(&[0..2, 1..2]))?, ndarray::array![[2], [6]].into_dyn());

    assert!(array.retrieve_chunks::<ArrayBytes>(&ArraySubset::new_with_ranges(&[0..2])).is_err());
    assert_eq!(array.retrieve_chunks::<ArrayBytes>(&ArraySubset::new_with_ranges(&[0..0, 0..0]))?, vec![].into());
    assert_eq!(array.retrieve_chunks::<ArrayBytes>(&ArraySubset::new_with_ranges(&[0..1, 0..1]))?, vec![1, 2, 5, 6].into());
    assert_eq!(array.retrieve_chunks::<ArrayBytes>(&ArraySubset::new_with_ranges(&[0..2, 0..2]))?, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0].into());
    assert_eq!(array.retrieve_chunks::<ArrayBytes>(&ArraySubset::new_with_ranges(&[0..2, 1..2]))?, vec![3, 4, 7, 8, 0, 0, 0, 0].into());
    assert_eq!(array.retrieve_chunks::<ArrayBytes>(&ArraySubset::new_with_ranges(&[0..1, 1..3]))?, vec![3, 4, 0, 0, 7, 8, 0, 0].into());

    assert!(array.retrieve_chunks::<ndarray::ArrayD<u8>>(&ArraySubset::new_with_ranges(&[0..2])).is_err());
    assert!(array.retrieve_chunks::<ndarray::ArrayD<u16>>(&ArraySubset::new_with_ranges(&[0..2, 0..2])).is_err());
    assert_eq!(array.retrieve_chunks::<ndarray::ArrayD<u8>>(&ArraySubset::new_with_ranges(&[0..2, 0..2]))?, ndarray::array![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 0, 0], [0, 0, 0, 0]].into_dyn());
    assert_eq!(array.retrieve_chunks::<ndarray::ArrayD<u8>>(&ArraySubset::new_with_ranges(&[0..2, 1..2]))?, ndarray::array![[3, 4], [7, 8], [0, 0], [0, 0]].into_dyn());
    assert_eq!(array.retrieve_chunks::<ndarray::ArrayD<u8>>(&ArraySubset::new_with_ranges(&[0..1, 1..3]))?, ndarray::array![[3, 4, 0, 0], [7, 8, 0, 0]].into_dyn());

    assert!(array.retrieve_array_subset::<ArrayBytes>(&ArraySubset::new_with_ranges(&[0..4])).is_err());
    assert_eq!(array.retrieve_array_subset::<ArrayBytes>(&ArraySubset::new_with_ranges(&[0..0, 0..0]))?, vec![].into());
    assert_eq!(array.retrieve_array_subset::<ArrayBytes>(&ArraySubset::new_with_ranges(&[0..2, 0..2]))?, vec![1, 2, 5, 6].into());
    assert_eq!(array.retrieve_array_subset::<ArrayBytes>(&ArraySubset::new_with_ranges(&[0..4, 0..4]))?, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0].into());
    assert_eq!(array.retrieve_array_subset::<ArrayBytes>(&ArraySubset::new_with_ranges(&[1..3, 1..3]))?, vec![6, 7, 10 ,0].into());
    assert_eq!(array.retrieve_array_subset::<ArrayBytes>(&ArraySubset::new_with_ranges(&[5..7, 5..6]))?, vec![0, 0].into()); // OOB -> fill value
    assert_eq!(array.retrieve_array_subset::<ArrayBytes>(&ArraySubset::new_with_ranges(&[0..5, 0..5]))?, vec![1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].into()); // OOB -> fill value

    assert!(array.retrieve_array_subset::<ndarray::ArrayD<u8>>(&ArraySubset::new_with_ranges(&[0..4])).is_err());
    assert!(array.retrieve_array_subset::<ndarray::ArrayD<u16>>(&ArraySubset::new_with_ranges(&[0..4, 0..4])).is_err());
    assert_eq!(array.retrieve_array_subset::<ndarray::ArrayD<u8>>(&ArraySubset::new_with_ranges(&[0..0, 0..0]))?, ndarray::Array2::<u8>::zeros((0, 0)).into_dyn());
    assert_eq!(array.retrieve_array_subset::<ndarray::ArrayD<u8>>(&ArraySubset::new_with_ranges(&[0..4, 0..4]))?, ndarray::array![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 0, 0], [0, 0, 0, 0]].into_dyn());
    assert_eq!(array.retrieve_array_subset::<ndarray::ArrayD<u8>>(&ArraySubset::new_with_ranges(&[1..3, 1..3]))?, ndarray::array![[6, 7], [10 ,0]].into_dyn());
    assert_eq!(array.retrieve_array_subset::<ndarray::ArrayD<u8>>(&ArraySubset::new_with_ranges(&[5..7, 5..6]))?, ndarray::array![[0], [0]].into_dyn()); // OOB -> fill value
    assert_eq!(array.retrieve_array_subset::<ndarray::ArrayD<u8>>(&ArraySubset::new_with_ranges(&[0..5, 0..5]))?, ndarray::array![[1, 2, 3, 4, 0], [5, 6, 7, 8, 0], [9, 10, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]].into_dyn()); // OOB -> fill value

    assert!(array.partial_decoder(&[0]).is_err());
    assert!(array.partial_decoder(&[0, 0])?.partial_decode(&ArraySubset::new_with_ranges(&[0..1]), &options).is_err());
    assert_eq!(array.partial_decoder(&[5, 0])?.partial_decode(&ArraySubset::new_with_ranges(&[0..1, 0..2]), &options)?, vec![0, 0].into()); // OOB -> fill value
    assert_eq!(array.partial_decoder(&[0, 0])?.partial_decode(&ArraySubset::new_with_ranges(&[0..1, 0..2]), &options)?, vec![1, 2].into());
    assert_eq!(array.partial_decoder(&[0, 0])?.partial_decode(&ArraySubset::new_with_ranges(&[0..2, 1..2]), &options)?, vec![2, 6].into());

    Ok(())
}

#[test]
fn array_sync_read_uncompressed() -> Result<(), Box<dyn std::error::Error>> {
    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let array = ArrayBuilder::new(
        vec![4, 4], // array shape
        vec![2, 2], // regular chunk shape
        data_type::uint8(),
        0u8,
    )
    .bytes_to_bytes_codecs(vec![])
    // .storage_transformers(vec![].into())
    .build(store, array_path)
    .unwrap();

    let chunk_shape = array.chunk_shape(&vec![0; array.dimensionality()])?;
    assert_eq!(
        array.codecs().partial_decode_granularity(&chunk_shape),
        [NonZeroU64::new(2).unwrap(); 2]
    );

    array_sync_read(&array)?;

    // uncompressed partial decoder holds no data
    assert_eq!(array.partial_decoder(&[0, 0])?.size_held(), 0);

    Ok(())
}

#[cfg(feature = "sharding")]
#[test]
#[cfg_attr(miri, ignore)]
fn array_sync_read_shard_compress() -> Result<(), Box<dyn std::error::Error>> {
    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let mut builder = ArrayBuilder::new(
        vec![4, 4], // array shape
        vec![2, 2], // regular chunk shape
        data_type::uint8(),
        0u8,
    );
    builder
        .subchunk_shape(vec![1, 1])
        .bytes_to_bytes_codecs(vec![
            #[cfg(feature = "gzip")]
            Arc::new(zarrs::array::codec::GzipCodec::new(5)?),
        ]);
    // .storage_transformers(vec![].into())

    let array = builder.build(store, array_path).unwrap();

    let chunk_shape = array.chunk_shape(&vec![0; array.dimensionality()])?;
    assert_eq!(
        array.codecs().partial_decode_granularity(&chunk_shape),
        [NonZeroU64::new(1).unwrap(); 2]
    );

    array_sync_read(&array)?;

    // sharding partial decoder holds the shard index, which has double the number of elements
    assert_eq!(
        array.partial_decoder(&[0, 0])?.size_held(),
        size_of::<u64>() * 4 * 2
    );
    // this chunk is empty, so it has no shard index
    assert_eq!(array.partial_decoder(&[1, 1])?.size_held(), 0);

    Ok(())
}

fn array_str_impl(array: Array<MemoryStore>) -> Result<(), Box<dyn std::error::Error>> {
    // Store a single chunk
    array.store_chunk(&[0, 0], &["a", "bb", "ccc", "dddd"])?;
    assert_eq!(
        array.retrieve_chunk::<Vec<String>>(&[0, 0])?,
        &["a", "bb", "ccc", "dddd"]
    );

    // Write array subset with full chunks
    array.store_array_subset(
        &ArraySubset::new_with_ranges(&[2..4, 0..4]),
        &[
            "1", "22", "333", "4444", "55555", "666666", "7777777", "88888888",
        ],
    )?;
    assert_eq!(
        array.retrieve_chunk::<Vec<String>>(&[1, 0])?,
        &["1", "22", "55555", "666666"]
    );
    assert_eq!(
        array.retrieve_chunk::<Vec<String>>(&[1, 1])?,
        &["333", "4444", "7777777", "88888888"]
    );

    // Write array subset with partial chunks
    array.store_array_subset(
        &ArraySubset::new_with_ranges(&[1..3, 1..3]),
        &["S1", "S22", "S333", "S4444"],
    )?;
    assert_eq!(
        array.retrieve_chunk::<Vec<String>>(&[0, 0])?,
        &["a", "bb", "ccc", "S1"]
    );
    assert_eq!(
        array.retrieve_chunk::<Vec<String>>(&[0, 1])?,
        &["", "", "S22", ""]
    );
    assert_eq!(
        array.retrieve_chunk::<Vec<String>>(&[1, 0])?,
        &["1", "S333", "55555", "666666"]
    );
    assert_eq!(
        array.retrieve_chunk::<Vec<String>>(&[1, 1])?,
        &["S4444", "4444", "7777777", "88888888"]
    );

    // Write multiple chunks
    array.store_chunks(
        &ArraySubset::new_with_ranges(&[0..1, 0..2]),
        &["a", "bb", "ccc", "dddd", "C0", "C11", "C222", "C3333"],
    )?;
    assert_eq!(
        array.retrieve_chunk::<Vec<String>>(&[0, 0])?,
        &["a", "bb", "C0", "C11"]
    );
    assert_eq!(
        array.retrieve_chunk::<Vec<String>>(&[0, 1])?,
        &["ccc", "dddd", "C222", "C3333"]
    );
    assert_eq!(
        array.retrieve_chunks::<Vec<String>>(&ArraySubset::new_with_ranges(&[0..1, 0..2]))?,
        &["a", "bb", "ccc", "dddd", "C0", "C11", "C222", "C3333"]
    );

    // Full chunk requests
    assert_eq!(
        array.retrieve_array_subset::<Vec<String>>(&ArraySubset::new_with_ranges(&[0..4, 0..4]))?,
        &[
            "a", "bb", "ccc", "dddd", "C0", "C11", "C222", "C3333", //
            "1", "S333", "S4444", "4444", "55555", "666666", "7777777", "88888888" //
        ]
    );

    // Partial chunk requests
    assert_eq!(
        array.retrieve_array_subset::<Vec<String>>(&ArraySubset::new_with_ranges(&[1..3, 1..3]))?,
        &["C11", "C222", "S333", "S4444"]
    );

    // Incompatible chunks / bytes
    assert!(
        array
            .store_chunks(&ArraySubset::new_with_ranges(&[0..0, 0..2]), &["a", "bb"])
            .is_err()
    );
    assert!(
        array
            .store_chunks(&ArraySubset::new_with_ranges(&[0..1, 0..2]), &["a", "bb"])
            .is_err()
    );

    Ok(())
}

#[test]
fn array_str_sync_simple() -> Result<(), Box<dyn std::error::Error>> {
    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let mut builder = ArrayBuilder::new(
        vec![4, 4], // array shape
        vec![2, 2], // regular chunk shape
        data_type::string(),
        "",
    );
    builder.bytes_to_bytes_codecs(vec![
        #[cfg(feature = "gzip")]
        Arc::new(zarrs::array::codec::GzipCodec::new(5)?),
    ]);

    let array = builder.build(store, array_path).unwrap();

    array_str_impl(array)
}

#[cfg(feature = "sharding")]
#[test]
fn array_str_sync_sharded_transpose() -> Result<(), Box<dyn std::error::Error>> {
    use zarrs::array::codec::{TransposeCodec, TransposeOrder, array_to_bytes::vlen::VlenCodec};

    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let mut builder = ArrayBuilder::new(
        vec![4, 4], // array shape
        vec![2, 2], // regular chunk shape
        data_type::string(),
        "",
    );
    builder.array_to_array_codecs(vec![Arc::new(TransposeCodec::new(
        TransposeOrder::new(&[1, 0]).unwrap(),
    ))]);
    builder.array_to_bytes_codec(Arc::new(
        zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder::new(
            vec![NonZeroU64::new(2).unwrap(), NonZeroU64::new(1).unwrap()],
            &data_type::string(),
        )
        .array_to_bytes_codec(Arc::<VlenCodec>::default())
        .build(),
    ));
    builder.bytes_to_bytes_codecs(vec![
        #[cfg(feature = "gzip")]
        Arc::new(zarrs::array::codec::GzipCodec::new(5)?),
    ]);

    let array = builder.build(store, array_path).unwrap();

    array_str_impl(array)
}

#[rustfmt::skip]
#[test]
fn array_binary() -> Result<(), Box<dyn std::error::Error>> {
    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let mut builder = ArrayBuilder::new(
        vec![4, 4], // array shape
        vec![2, 2], // regular chunk shape
        data_type::bytes(),
        [],
    );
    builder.bytes_to_bytes_codecs(vec![
        #[cfg(feature = "gzip")]
        Arc::new(zarrs::array::codec::GzipCodec::new(5)?),
    ]);

    let array = builder.build(store, array_path).unwrap();

    array.store_array_subset(
        &ArraySubset::new_with_ranges(&[1..3, 1..3]),
        &[&[0u8][..], &[0u8, 1][..], &[0u8, 1, 2][..], &[0u8, 1, 2, 3][..]],
    )?;
    assert_eq!(
        array.retrieve_chunk::<Vec<Vec<u8>>>(&[0, 0])?,
        vec![
            vec![], vec![],
            vec![], vec![0]
        ],
    );
    assert_eq!(
        array.retrieve_chunk::<Vec<Vec<u8>>>(&[0, 1])?,
        vec![
            vec![], vec![],
            vec![0, 1], vec![]
        ],
    );
    assert_eq!(
        array.retrieve_chunk::<Vec<Vec<u8>>>(&[1, 0])?,
        vec![
            vec![], vec![0, 1, 2],
            vec![], vec![]
        ],
    );
    assert_eq!(
        array.retrieve_chunk::<Vec<Vec<u8>>>(&[1, 1])?,
        vec![
            vec![0, 1, 2, 3], vec![],
            vec![], vec![]
        ],
    );
    assert_eq!(
        array.retrieve_array_subset::<Vec<Vec<u8>>>(&ArraySubset::new_with_ranges(&[1..3, 0..4]))?,
        vec![
            vec![], vec![0], vec![0, 1], vec![],
            vec![], vec![0, 1, 2], vec![0, 1, 2, 3], vec![],
        ],
    );

    Ok(())
}

#[cfg(feature = "zfp")]
#[test]
fn array_5d_zfp() -> Result<(), Box<dyn std::error::Error>> {
    use zarrs::metadata_ext::codec::reshape::ReshapeShape;
    use zarrs_data_type::FillValue;

    let store = std::sync::Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new(
        vec![4, 4, 4, 4, 4],
        vec![1, 1, 4, 4, 4],
        data_type::uint32(),
        FillValue::from(0u32),
    );
    builder.array_to_array_codecs(vec![Arc::new(zarrs::array::codec::ReshapeCodec::new(
        ReshapeShape::new([[0, 1, 2].into(), [3].into(), [4].into()])?,
    ))]);
    builder.array_to_bytes_codec(Arc::new(zarrs::array::codec::ZfpCodec::new_reversible()));
    let array = builder.build(store.clone(), "/")?;
    array.store_metadata()?;

    let elements: Vec<u32> = (0..array.shape().iter().product())
        .map(|u| u as u32)
        .collect();

    array.store_array_subset(&array.subset_all(), &elements)?;
    assert_eq!(
        array.retrieve_array_subset::<Vec<u32>>(&array.subset_all())?,
        elements
    );

    // check reshape is registered
    let _array = Array::open(store.clone(), "/")?;

    Ok(())
}
