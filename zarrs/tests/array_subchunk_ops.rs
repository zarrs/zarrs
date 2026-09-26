#![allow(missing_docs)]

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::chunk_grid::{RectangularChunkGrid, RectilinearChunkGrid};
use zarrs::array::codec::array_to_array::transpose::{TransposeCodec, TransposeOrder};
use zarrs::array::codec::array_to_bytes::sharding::{
    ShardingCodecBuilder, ShardingCodecOptions, ShardingIndexLocation, SubchunkWriteOrder,
};
use zarrs::array::{
    Array, ArrayBuilder, ArrayCreateError, CodecChain, CodecSpecificOptions, data_type,
};
use zarrs_chunk_grid::{ArraySubset, ChunkGrid, ChunkGridCreateError};
use zarrs_metadata_ext::chunk_grid::rectangular::RectangularChunkGridDimensionConfiguration;
use zarrs_metadata_ext::chunk_grid::rectilinear::{ChunkEdgeLengths, RunLengthElement};
use zarrs_storage::WritableStorageTraits;
use zarrs_storage::store::MemoryStore;

fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

fn build_array_with_chunk_grid(
    chunk_grid: impl Into<ChunkGrid>,
    subchunk_shape: Vec<u64>,
) -> Result<Array<MemoryStore>, Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new_with_chunk_grid(chunk_grid, data_type::uint16(), 0u16);
    builder.subchunk_shape(subchunk_shape);
    Ok(builder.build(store, "/array")?)
}

fn assert_subchunk_grid(
    array: &Array<MemoryStore>,
    expected_array_shape: &[u64],
    expected_grid_shape: &[u64],
    expected_edge_lengths: &[NonZeroU64],
) -> Result<ChunkGrid, Box<dyn std::error::Error>> {
    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap().clone();
    assert_eq!(subchunk_grid.array_shape(), expected_array_shape);
    assert_eq!(subchunk_grid.grid_shape(), expected_grid_shape);
    assert_eq!(subchunk_grid.chunk_edge_lengths(0)?, expected_edge_lengths);
    Ok(subchunk_grid)
}

#[test]
fn subchunk_grid_regular_outer_uses_repeat_chunk_grid() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type::uint16(), 0u16);
    builder.subchunk_shape(vec![2, 2]);
    let array = builder.build(store, "/array")?;

    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap();
    assert_eq!(
        array.subchunk_shape(),
        Some(vec![NonZeroU64::new(2).unwrap(); 2])
    );
    assert_eq!(subchunk_grid.name_v3(), None);
    assert_eq!(subchunk_grid.array_shape(), &[8, 8]);
    assert_eq!(subchunk_grid.grid_shape(), &[4, 4]);
    assert_eq!(
        subchunk_grid.subset(&[2, 3])?,
        Some(ArraySubset::new_with_ranges(&[4..6, 6..8]))
    );

    Ok(())
}

#[test]
fn subchunk_grid_regular_outer_covers_full_repeated_shard_extent()
-> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new(vec![7, 7], vec![4, 4], data_type::uint16(), 0u16);
    builder.subchunk_shape(vec![2, 2]);
    let array = builder.build(store, "/array")?;

    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap();
    assert_eq!(
        array.subchunk_shape(),
        Some(vec![NonZeroU64::new(2).unwrap(); 2])
    );
    assert_eq!(array.shape(), &[7, 7]);
    assert_eq!(array.chunk_grid_shape(), &[2, 2]);
    assert_eq!(subchunk_grid.name_v3(), None);
    assert_eq!(subchunk_grid.array_shape(), &[8, 8]);
    assert_eq!(subchunk_grid.grid_shape(), &[4, 4]);
    assert_eq!(
        subchunk_grid.subset(&[3, 3])?,
        Some(ArraySubset::new_with_ranges(&[6..8, 6..8]))
    );

    Ok(())
}

#[test]
fn subchunk_grid_rejects_non_even_sharding_chunk_shape() -> Result<(), Box<dyn std::error::Error>> {
    // Creating arrays with non evenly divisible subchunks requires opting in
    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new(vec![10], vec![5], data_type::uint16(), 0u16);
    builder.subchunk_shape(vec![3]);
    assert!(builder.build_metadata().is_err());
    let err = builder.build(store, "/array").unwrap_err();

    assert!(matches!(
        err,
        ArrayCreateError::ChunkGridCreateError(ChunkGridCreateError::Other(ref str)) if str.contains("must evenly divide shard shape")
    ));
    assert!(
        err.to_string()
            .contains("ShardingCodecOptions::with_allow_nondivisible_subchunks")
    );

    Ok(())
}

#[test]
fn subchunk_grid_rejects_non_even_sharding_in_codec_chain() -> Result<(), Box<dyn std::error::Error>>
{
    // Sharding in a codec chain used as the array-to-bytes codec
    // The subchunk shape applies to the transposed shard shape [6, 8]
    let data_type = data_type::uint16();
    let codec_chain = |subchunk_shape: Vec<NonZeroU64>| {
        Arc::new(CodecChain::new(
            vec![Arc::new(TransposeCodec::new(
                TransposeOrder::new(&[1, 0]).unwrap(),
            ))],
            ShardingCodecBuilder::new(subchunk_shape, &data_type).build_arc(),
            vec![],
        ))
    };
    let mut builder = ArrayBuilder::new(vec![8, 6], vec![8, 6], data_type.clone(), 0u16);
    builder.array_to_bytes_codec(codec_chain(vec![nz(4), nz(3)]));
    assert!(builder.build_metadata().is_err());
    builder.codec_specific_options(
        CodecSpecificOptions::default()
            .with_option(ShardingCodecOptions::default().with_allow_nondivisible_subchunks(true)),
    );
    builder.build(Arc::new(MemoryStore::default()), "/array")?;

    let mut builder = ArrayBuilder::new(vec![8, 6], vec![8, 6], data_type.clone(), 0u16);
    builder.array_to_bytes_codec(codec_chain(vec![nz(3), nz(4)]));
    builder.build(Arc::new(MemoryStore::default()), "/array")?;

    Ok(())
}

#[test]
fn subchunk_grid_nondivisible_subchunks_per_codec() -> Result<(), Box<dyn std::error::Error>> {
    // Only the outer sharding codec permits non-divisible subchunks
    // The inner [2, 2] subchunks do not evenly divide the outer [3, 3] subchunks
    let data_type = data_type::uint16();
    let allow = ShardingCodecOptions::default().with_allow_nondivisible_subchunks(true);
    let inner = ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build();
    let outer = |inner: Arc<zarrs::array::codec::ShardingCodec>| {
        Arc::new(
            ShardingCodecBuilder::new(vec![nz(3), nz(3)], &data_type)
                .array_to_bytes_codec(inner)
                .build()
                .with_options(allow.clone()),
        )
    };

    let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type.clone(), 0u16);
    builder.array_to_bytes_codec(outer(Arc::new(inner.clone())));
    assert!(builder.build_metadata().is_err());

    builder.array_to_bytes_codec(outer(Arc::new(inner.with_options(allow.clone()))));
    builder.build(Arc::new(MemoryStore::default()), "/array")?;

    Ok(())
}

#[test]
fn subchunk_grid_builds_empty_array_with_sharding() -> Result<(), Box<dyn std::error::Error>> {
    let mut builder = ArrayBuilder::new(vec![0, 8], vec![4, 4], data_type::uint16(), 0u16);
    builder.subchunk_shape(vec![2, 2]);
    builder.build(Arc::new(MemoryStore::default()), "/array")?;
    Ok(())
}

#[test]
fn subchunk_grid_rejects_non_even_nested_and_transposed_sharding()
-> Result<(), Box<dyn std::error::Error>> {
    let data_type = data_type::uint16();

    // Nested sharding with inner subchunks that do not evenly divide the outer subchunks
    let mut builder = ArrayBuilder::new(vec![8, 8], vec![4, 4], data_type.clone(), 0u16);
    builder.array_to_bytes_codec(
        ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type)
            .array_to_bytes_codec(
                ShardingCodecBuilder::new(vec![nz(3), nz(3)], &data_type).build_arc(),
            )
            .build_arc(),
    );
    assert!(builder.build_metadata().is_err());

    // The subchunk shape applies to the transposed shard shape [6, 8]
    let transpose = Arc::new(TransposeCodec::new(TransposeOrder::new(&[1, 0])?));
    let mut builder = ArrayBuilder::new(vec![8, 6], vec![8, 6], data_type.clone(), 0u16);
    builder
        .array_to_array_codecs(vec![transpose])
        .array_to_bytes_codec(
            ShardingCodecBuilder::new(vec![nz(4), nz(3)], &data_type).build_arc(),
        );
    assert!(builder.build_metadata().is_err());
    builder.array_to_bytes_codec(
        ShardingCodecBuilder::new(vec![nz(3), nz(4)], &data_type).build_arc(),
    );
    assert!(builder.build_metadata().is_ok());

    Ok(())
}

#[test]
#[allow(clippy::single_range_in_vec_init)]
fn subchunk_grid_opens_non_even_sharding_chunk_shape() -> Result<(), Box<dyn std::error::Error>> {
    // Existing arrays with non evenly divisible subchunks can be opened without opting in
    let store = Arc::new(MemoryStore::default());
    let metadata = r#"{
        "zarr_format": 3,
        "node_type": "array",
        "shape": [10],
        "data_type": "uint16",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [5]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [{
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": [3],
                "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "index_location": "end"
            }
        }]
    }"#;
    store.set(
        &zarrs_storage::StoreKey::new("array/zarr.json")?,
        metadata.as_bytes().to_vec().into(),
    )?;
    let array = Array::open(store, "/array")?;
    // Setting other sharding options retains the permission for non-divisible subchunks
    let array = array.with_codec_specific_options(&CodecSpecificOptions::default().with_option(
        ShardingCodecOptions::default().with_subchunk_write_order(SubchunkWriteOrder::C),
    ))?;
    assert_eq!(
        array
            .subchunk_grid()
            .as_chunk_grid()
            .unwrap()
            .chunk_edge_lengths(0)?,
        vec![nz(3), nz(2), nz(3), nz(2)]
    );

    let data: Vec<u16> = (0..10).collect();
    array.store_array_subset(&ArraySubset::new_with_ranges(&[0..10]), data.as_slice())?;
    assert_eq!(
        array.retrieve_array_subset::<Vec<u16>>(&ArraySubset::new_with_ranges(&[0..10]))?,
        data
    );

    Ok(())
}

#[test]
#[allow(clippy::single_range_in_vec_init)]
fn subchunk_grid_from_varying_shard_edges() -> Result<(), Box<dyn std::error::Error>> {
    let arrays = [
        build_array_with_chunk_grid(
            RectilinearChunkGrid::new(
                vec![15],
                &[ChunkEdgeLengths::Varying(vec![
                    RunLengthElement::Single(nz(6)),
                    RunLengthElement::Single(nz(9)),
                ])],
            )?,
            vec![3],
        )?,
        build_array_with_chunk_grid(
            RectangularChunkGrid::new(
                vec![15],
                &[RectangularChunkGridDimensionConfiguration::Varying(vec![
                    nz(6),
                    nz(9),
                ])],
            )?,
            vec![3],
        )?,
    ];

    for array in arrays {
        let subchunk_grid =
            assert_subchunk_grid(&array, &[15], &[5], &[nz(3), nz(3), nz(3), nz(3), nz(3)])?;
        assert_eq!(array.subchunk_shape(), Some(vec![nz(3)]));
        assert_eq!(
            subchunk_grid.subset(&[1])?,
            Some(ArraySubset::new_with_ranges(&[3..6]))
        );
        assert_eq!(
            subchunk_grid.subset(&[2])?,
            Some(ArraySubset::new_with_ranges(&[6..9]))
        );
    }

    Ok(())
}

#[test]
fn subchunk_grid_accounts_for_transpose_before_sharding() -> Result<(), Box<dyn std::error::Error>>
{
    let store = Arc::new(MemoryStore::default());
    let data_type = data_type::uint16();
    let sharding_codec = ShardingCodecBuilder::new(vec![nz(3), nz(2)], &data_type)
        .index_location(ShardingIndexLocation::End)
        .build();
    let mut builder = ArrayBuilder::new(vec![8, 6], vec![4, 6], data_type, 0u16);
    builder
        .array_to_array_codecs(vec![Arc::new(TransposeCodec::new(TransposeOrder::new(
            &[1, 0],
        )?))])
        .array_to_bytes_codec(Arc::new(sharding_codec));
    let array = builder.build(store, "/array")?;

    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap();
    assert_eq!(array.subchunk_shape(), Some(vec![nz(2), nz(3)]));
    assert_eq!(subchunk_grid.array_shape(), &[8, 6]);
    assert_eq!(subchunk_grid.grid_shape(), &[4, 2]);
    assert_eq!(
        subchunk_grid.chunk_edge_lengths(0)?,
        vec![nz(2), nz(2), nz(2), nz(2)]
    );
    assert_eq!(subchunk_grid.chunk_edge_lengths(1)?, vec![nz(3), nz(3)]);

    Ok(())
}
