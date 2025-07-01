#![allow(missing_docs)]
#![cfg(feature = "ndarray")]

use std::sync::Arc;

use zarrs::array::codec::CodecOptions;
use zarrs::array::{Array, ArrayBuilder, ArrayCodecTraits, DataType};
use zarrs::array_subset::ArraySubset;
use zarrs::storage::store::MemoryStore;

#[rustfmt::skip]
fn array_sync_read(array: Array<MemoryStore>) -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(array.data_type(), &DataType::UInt8);
    assert_eq!(array.fill_value().as_ne_bytes(), &[0u8]);
    assert_eq!(array.shape(), &[4, 4]);
    assert_eq!(array.chunk_shape(&[0, 0]).unwrap(), [2, 2].try_into().unwrap());
    assert_eq!(array.chunk_grid_shape(), &[2, 2]);

    let options = CodecOptions::default();

    // 1  2 | 3  4 
    // 5  6 | 7  8
    // -----|-----
    // 9 10 | 0  0
    // 0  0 | 0  0
    array.store_chunk(&[0, 0], &[1, 2, 0, 0])?;
    array.store_chunk(&[0, 1], &[3, 4, 7, 8])?;
    array.store_array_subset(&ArraySubset::new_with_ranges(&[1..3, 0..2]), &[5, 6, 9, 10])?;

    assert!(array.retrieve_chunk(&[0, 0, 0]).is_err());
    assert_eq!(array.retrieve_chunk(&[0, 0])?, vec![1, 2, 5, 6].into());
    assert_eq!(array.retrieve_chunk(&[0, 1])?, vec![3, 4, 7, 8].into());
    assert_eq!(array.retrieve_chunk(&[1, 0])?, vec![9, 10, 0, 0].into());
    assert_eq!(array.retrieve_chunk(&[1, 1])?, vec![0, 0, 0, 0].into());

    assert!(array.retrieve_chunk_if_exists(&[0, 0, 0]).is_err());
    assert_eq!(array.retrieve_chunk_if_exists(&[0, 0])?, Some(vec![1, 2, 5, 6].into()));
    assert_eq!(array.retrieve_chunk_if_exists(&[0, 1])?, Some(vec![3, 4, 7, 8].into()));
    assert_eq!(array.retrieve_chunk_if_exists(&[1, 0])?, Some(vec![9, 10, 0, 0].into()));
    assert_eq!(array.retrieve_chunk_if_exists(&[1, 1])?, None);

    assert!(array.retrieve_chunk_ndarray::<u16>(&[0, 0]).is_err());
    assert_eq!(array.retrieve_chunk_ndarray::<u8>(&[0, 0])?, ndarray::array![[1, 2], [5, 6]].into_dyn());
    assert_eq!(array.retrieve_chunk_ndarray::<u8>(&[0, 1])?, ndarray::array![[3, 4], [7, 8]].into_dyn());
    assert_eq!(array.retrieve_chunk_ndarray::<u8>(&[1, 0])?, ndarray::array![[9, 10], [0, 0]].into_dyn());
    assert_eq!(array.retrieve_chunk_ndarray::<u8>(&[1, 1])?, ndarray::array![[0, 0], [0, 0]].into_dyn());

    assert_eq!(array.retrieve_chunk_ndarray_if_exists::<u8>(&[0, 0])?, Some(ndarray::array![[1, 2], [5, 6]].into_dyn()));
    assert_eq!(array.retrieve_chunk_ndarray_if_exists::<u8>(&[0, 1])?, Some(ndarray::array![[3, 4], [7, 8]].into_dyn()));
    assert_eq!(array.retrieve_chunk_ndarray_if_exists::<u8>(&[1, 0])?, Some(ndarray::array![[9, 10], [0, 0]].into_dyn()));
    assert_eq!(array.retrieve_chunk_ndarray_if_exists::<u8>(&[1, 1])?, None);

    assert!(array.retrieve_chunk_subset(&[0, 0], &ArraySubset::new_with_ranges(&[0..2])).is_err());
    assert!(array.retrieve_chunk_subset(&[0, 0], &ArraySubset::new_with_ranges(&[0..3, 0..3])).is_err());
    assert_eq!(array.retrieve_chunk_subset(&[0, 0], &ArraySubset::new_with_ranges(&[0..2, 0..2]))?, vec![1, 2, 5, 6].into());
    assert_eq!(array.retrieve_chunk_subset(&[0, 0], &ArraySubset::new_with_ranges(&[0..1, 0..2]))?, vec![1, 2].into());
    assert_eq!(array.retrieve_chunk_subset(&[0, 0], &ArraySubset::new_with_ranges(&[0..2, 1..2]))?, vec![2, 6].into());

    assert!(array.retrieve_chunk_subset_ndarray::<u8>(&[0, 0], &ArraySubset::new_with_ranges(&[0..3, 0..3])).is_err());
    assert!(array.retrieve_chunk_subset_ndarray::<u16>(&[0, 0], &ArraySubset::new_with_ranges(&[0..2, 0..2])).is_err());
    assert_eq!(array.retrieve_chunk_subset_ndarray::<u8>(&[0, 0], &ArraySubset::new_with_ranges(&[0..2, 0..2]))?, ndarray::array![[1, 2], [5, 6]].into_dyn());
    assert_eq!(array.retrieve_chunk_subset_ndarray::<u8>(&[0, 0], &ArraySubset::new_with_ranges(&[0..1, 0..2]))?, ndarray::array![[1, 2]].into_dyn());
    assert_eq!(array.retrieve_chunk_subset_ndarray::<u8>(&[0, 0], &ArraySubset::new_with_ranges(&[0..2, 1..2]))?, ndarray::array![[2], [6]].into_dyn());

    assert!(array.retrieve_chunks(&ArraySubset::new_with_ranges(&[0..2])).is_err());
    assert_eq!(array.retrieve_chunks(&ArraySubset::new_with_ranges(&[0..0, 0..0]))?, vec![].into());
    assert_eq!(array.retrieve_chunks(&ArraySubset::new_with_ranges(&[0..1, 0..1]))?, vec![1, 2, 5, 6].into());
    assert_eq!(array.retrieve_chunks(&ArraySubset::new_with_ranges(&[0..2, 0..2]))?, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0].into());
    assert_eq!(array.retrieve_chunks(&ArraySubset::new_with_ranges(&[0..2, 1..2]))?, vec![3, 4, 7, 8, 0, 0, 0, 0].into());
    assert_eq!(array.retrieve_chunks(&ArraySubset::new_with_ranges(&[0..1, 1..3]))?, vec![3, 4, 0, 0, 7, 8, 0, 0].into());

    assert!(array.retrieve_chunks_ndarray::<u8>(&ArraySubset::new_with_ranges(&[0..2])).is_err());
    assert!(array.retrieve_chunks_ndarray::<u16>(&ArraySubset::new_with_ranges(&[0..2, 0..2])).is_err());
    assert_eq!(array.retrieve_chunks_ndarray::<u8>(&ArraySubset::new_with_ranges(&[0..2, 0..2]))?, ndarray::array![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 0, 0], [0, 0, 0, 0]].into_dyn());
    assert_eq!(array.retrieve_chunks_ndarray::<u8>(&ArraySubset::new_with_ranges(&[0..2, 1..2]))?, ndarray::array![[3, 4], [7, 8], [0, 0], [0, 0]].into_dyn());
    assert_eq!(array.retrieve_chunks_ndarray::<u8>(&ArraySubset::new_with_ranges(&[0..1, 1..3]))?, ndarray::array![[3, 4, 0, 0], [7, 8, 0, 0]].into_dyn());

    assert!(array.retrieve_array_subset(&ArraySubset::new_with_ranges(&[0..4])).is_err());
    assert_eq!(array.retrieve_array_subset(&ArraySubset::new_with_ranges(&[0..0, 0..0]))?, vec![].into());
    assert_eq!(array.retrieve_array_subset(&ArraySubset::new_with_ranges(&[0..2, 0..2]))?, vec![1, 2, 5, 6].into());
    assert_eq!(array.retrieve_array_subset(&ArraySubset::new_with_ranges(&[0..4, 0..4]))?, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0, 0, 0, 0, 0].into());
    assert_eq!(array.retrieve_array_subset(&ArraySubset::new_with_ranges(&[1..3, 1..3]))?, vec![6, 7, 10 ,0].into());
    assert_eq!(array.retrieve_array_subset(&ArraySubset::new_with_ranges(&[5..7, 5..6]))?, vec![0, 0].into()); // OOB -> fill value
    assert_eq!(array.retrieve_array_subset(&ArraySubset::new_with_ranges(&[0..5, 0..5]))?, vec![1, 2, 3, 4, 0, 5, 6, 7, 8, 0, 9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].into()); // OOB -> fill value

    assert!(array.retrieve_array_subset_ndarray::<u8>(&ArraySubset::new_with_ranges(&[0..4])).is_err());
    assert!(array.retrieve_array_subset_ndarray::<u16>(&ArraySubset::new_with_ranges(&[0..4, 0..4])).is_err());
    assert_eq!(array.retrieve_array_subset_ndarray::<u8>(&ArraySubset::new_with_ranges(&[0..0, 0..0]))?, ndarray::Array2::<u8>::zeros((0, 0)).into_dyn());
    assert_eq!(array.retrieve_array_subset_ndarray::<u8>(&ArraySubset::new_with_ranges(&[0..4, 0..4]))?, ndarray::array![[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 0, 0], [0, 0, 0, 0]].into_dyn());
    assert_eq!(array.retrieve_array_subset_ndarray::<u8>(&ArraySubset::new_with_ranges(&[1..3, 1..3]))?, ndarray::array![[6, 7], [10 ,0]].into_dyn());
    assert_eq!(array.retrieve_array_subset_ndarray::<u8>(&ArraySubset::new_with_ranges(&[5..7, 5..6]))?, ndarray::array![[0], [0]].into_dyn()); // OOB -> fill value
    assert_eq!(array.retrieve_array_subset_ndarray::<u8>(&ArraySubset::new_with_ranges(&[0..5, 0..5]))?, ndarray::array![[1, 2, 3, 4, 0], [5, 6, 7, 8, 0], [9, 10, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]].into_dyn()); // OOB -> fill value

    assert!(array.partial_decoder(&[0]).is_err());
    assert!(array.partial_decoder(&[0, 0])?.partial_decode(&[ArraySubset::new_with_ranges(&[0..1])], &options).is_err());
    assert_eq!(array.partial_decoder(&[0, 0])?.partial_decode(&[], &options)?, []);
    assert_eq!(array.partial_decoder(&[5, 0])?.partial_decode(&[ArraySubset::new_with_ranges(&[0..1, 0..2])], &options)?, [vec![0, 0].into()]); // OOB -> fill value
    assert_eq!(array.partial_decoder(&[0, 0])?.partial_decode(&[ArraySubset::new_with_ranges(&[0..1, 0..2]), ArraySubset::new_with_ranges(&[0..2, 1..2])], &options)?, [vec![1, 2].into(), vec![2, 6].into()]);

    Ok(())
}

#[test]
fn array_sync_read_uncompressed() -> Result<(), Box<dyn std::error::Error>> {
    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let array = ArrayBuilder::new(
        vec![4, 4], // array shape
        vec![2, 2], // regular chunk shape
        DataType::UInt8,
        0u8,
    )
    .bytes_to_bytes_codecs(vec![])
    // .storage_transformers(vec![].into())
    .build(store, array_path)
    .unwrap();

    let chunk_representation =
        array.chunk_array_representation(&vec![0; array.dimensionality()])?;
    assert_eq!(
        array
            .codecs()
            .partial_decode_granularity(&chunk_representation),
        [2, 2].try_into().unwrap()
    );

    array_sync_read(array)
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
        DataType::UInt8,
        0u8,
    );
    builder.array_to_bytes_codec(Arc::new(
        zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder::new(
            vec![1, 1].try_into().unwrap(),
        )
        .bytes_to_bytes_codecs(vec![
            #[cfg(feature = "gzip")]
            Arc::new(zarrs::array::codec::GzipCodec::new(5)?),
        ])
        .build(),
    ));
    // .storage_transformers(vec![].into())

    let array = builder.build(store, array_path).unwrap();

    let chunk_representation =
        array.chunk_array_representation(&vec![0; array.dimensionality()])?;
    assert_eq!(
        array
            .codecs()
            .partial_decode_granularity(&chunk_representation),
        [1, 1].try_into().unwrap()
    );

    array_sync_read(array)
}

fn array_str_impl(array: Array<MemoryStore>) -> Result<(), Box<dyn std::error::Error>> {
    // Store a single chunk
    array.store_chunk_elements(&[0, 0], &["a", "bb", "ccc", "dddd"])?;
    assert_eq!(
        array.retrieve_chunk_elements::<String>(&[0, 0])?,
        &["a", "bb", "ccc", "dddd"]
    );

    // Write array subset with full chunks
    array.store_array_subset_elements(
        &ArraySubset::new_with_ranges(&[2..4, 0..4]),
        &[
            "1", "22", "333", "4444", "55555", "666666", "7777777", "88888888",
        ],
    )?;
    assert_eq!(
        array.retrieve_chunk_elements::<String>(&[1, 0])?,
        &["1", "22", "55555", "666666"]
    );
    assert_eq!(
        array.retrieve_chunk_elements::<String>(&[1, 1])?,
        &["333", "4444", "7777777", "88888888"]
    );

    // Write array subset with partial chunks
    array.store_array_subset_elements(
        &ArraySubset::new_with_ranges(&[1..3, 1..3]),
        &["S1", "S22", "S333", "S4444"],
    )?;
    assert_eq!(
        array.retrieve_chunk_elements::<String>(&[0, 0])?,
        &["a", "bb", "ccc", "S1"]
    );
    assert_eq!(
        array.retrieve_chunk_elements::<String>(&[0, 1])?,
        &["", "", "S22", ""]
    );
    assert_eq!(
        array.retrieve_chunk_elements::<String>(&[1, 0])?,
        &["1", "S333", "55555", "666666"]
    );
    assert_eq!(
        array.retrieve_chunk_elements::<String>(&[1, 1])?,
        &["S4444", "4444", "7777777", "88888888"]
    );

    // Write multiple chunks
    array.store_chunks_elements(
        &ArraySubset::new_with_ranges(&[0..1, 0..2]),
        &["a", "bb", "ccc", "dddd", "C0", "C11", "C222", "C3333"],
    )?;
    assert_eq!(
        array.retrieve_chunk_elements::<String>(&[0, 0])?,
        &["a", "bb", "C0", "C11"]
    );
    assert_eq!(
        array.retrieve_chunk_elements::<String>(&[0, 1])?,
        &["ccc", "dddd", "C222", "C3333"]
    );
    assert_eq!(
        array.retrieve_chunks_elements::<String>(&ArraySubset::new_with_ranges(&[0..1, 0..2]))?,
        &["a", "bb", "ccc", "dddd", "C0", "C11", "C222", "C3333"]
    );

    // Full chunk requests
    assert_eq!(
        array.retrieve_array_subset_elements::<String>(&ArraySubset::new_with_ranges(&[
            0..4,
            0..4
        ]))?,
        &[
            "a", "bb", "ccc", "dddd", "C0", "C11", "C222", "C3333", //
            "1", "S333", "S4444", "4444", "55555", "666666", "7777777", "88888888" //
        ]
    );

    // Partial chunk requests
    assert_eq!(
        array.retrieve_array_subset_elements::<String>(&ArraySubset::new_with_ranges(&[
            1..3,
            1..3
        ]))?,
        &["C11", "C222", "S333", "S4444"]
    );

    // Incompatible chunks / bytes
    assert!(array
        .store_chunks_elements(&ArraySubset::new_with_ranges(&[0..0, 0..2]), &["a", "bb"])
        .is_err());
    assert!(array
        .store_chunks_elements(&ArraySubset::new_with_ranges(&[0..1, 0..2]), &["a", "bb"])
        .is_err());

    Ok(())
}

#[test]
fn array_str_sync_simple() -> Result<(), Box<dyn std::error::Error>> {
    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let mut builder = ArrayBuilder::new(
        vec![4, 4], // array shape
        vec![2, 2], // regular chunk shape
        DataType::String,
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
    use zarrs::array::codec::{array_to_bytes::vlen::VlenCodec, TransposeCodec, TransposeOrder};

    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let mut builder = ArrayBuilder::new(
        vec![4, 4], // array shape
        vec![2, 2], // regular chunk shape
        DataType::String,
        "",
    );
    builder.array_to_array_codecs(vec![Arc::new(TransposeCodec::new(
        TransposeOrder::new(&[1, 0]).unwrap(),
    ))]);
    builder.array_to_bytes_codec(Arc::new(
        zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder::new(
            vec![2, 1].try_into().unwrap(),
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
        DataType::Bytes,
        [],
    );
    builder.bytes_to_bytes_codecs(vec![
        #[cfg(feature = "gzip")]
        Arc::new(zarrs::array::codec::GzipCodec::new(5)?),
    ]);

    let array = builder.build(store, array_path).unwrap();

    array.store_array_subset_elements::<&[u8]>(
        &ArraySubset::new_with_ranges(&[1..3, 1..3]),
        &[&[0], &[0, 1], &[0, 1, 2], &[0, 1, 2, 3]],
    )?;
    assert_eq!(
        array.retrieve_chunk_elements::<Vec<u8>>(&[0, 0])?,
        vec![
            vec![], vec![],
            vec![], vec![0]
        ],
    );
    assert_eq!(
        array.retrieve_chunk_elements::<Vec<u8>>(&[0, 1])?,
        vec![
            vec![], vec![],
            vec![0, 1], vec![]
        ],
    );
    assert_eq!(
        array.retrieve_chunk_elements::<Vec<u8>>(&[1, 0])?,
        vec![
            vec![], vec![0, 1, 2],
            vec![], vec![]
        ],
    );
    assert_eq!(
        array.retrieve_chunk_elements::<Vec<u8>>(&[1, 1])?,
        vec![
            vec![0, 1, 2, 3], vec![],
            vec![], vec![]
        ],
    );
    assert_eq!(
        array.retrieve_array_subset_elements::<Vec<u8>>(&ArraySubset::new_with_ranges(&[1..3, 0..4]))?,
        vec![
            vec![], vec![0], vec![0, 1], vec![],
            vec![], vec![0, 1, 2], vec![0, 1, 2, 3], vec![],
        ],
    );

    Ok(())
}
