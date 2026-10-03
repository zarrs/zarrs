#![allow(missing_docs)]

//! Sharding with subchunk shapes that do not evenly divide the shard shape.
//!
//! Subchunks straddling the shard boundary are clipped to the shard shape.
//! See <https://github.com/zarr-developers/zarr-specs/pull/370>.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::chunk_grid::{RectilinearChunkGrid, RegularChunkGrid};
use zarrs::array::codec::array_to_array::transpose::{TransposeCodec, TransposeOrder};
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::codec::{BytesCodec, OptionalCodec, VlenCodec};
use zarrs::array::{
    Array, ArrayBuilder, ArrayIndices, ArrayMetadata, CodecChain, CodecOptions, DataType,
    Endianness, FillValue, FromArrayBytes, IntoArrayBytes, data_type,
};
use zarrs_chunk_grid::{ArraySubset, ChunkGrid};
use zarrs_codec::UnboundArrayToBytesCodecTraits;
use zarrs_filesystem::FilesystemStore;
use zarrs_metadata_ext::chunk_grid::rectilinear::{ChunkEdgeLengths, RunLengthElement};
use zarrs_metadata_ext::codec::vlen::{VlenIndexDataType, VlenIndexLocation};
use zarrs_storage::store::MemoryStore;

fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

/// Create an array from metadata, since `ArrayBuilder::build` rejects non evenly divisible subchunks.
fn build_from_metadata<TStorage: ?Sized>(
    builder: &ArrayBuilder,
    store: Arc<TStorage>,
) -> Result<Array<TStorage>, Box<dyn std::error::Error>> {
    let metadata = ArrayMetadata::V3(builder.build_metadata()?);
    Ok(Array::new_with_metadata(store, "/array", metadata)?)
}

fn build_array(
    chunk_grid: impl Into<ChunkGrid>,
    subchunk_shape: Vec<u64>,
) -> Result<Array<MemoryStore>, Box<dyn std::error::Error>> {
    let store = Arc::new(MemoryStore::default());
    let mut builder = ArrayBuilder::new_with_chunk_grid(chunk_grid, data_type::uint16(), 0u16);
    builder.subchunk_shape(subchunk_shape);
    build_from_metadata(&builder, store)
}

#[test]
#[allow(clippy::single_range_in_vec_init)]
fn sharding_nondivisible_subchunk_grid_regular() -> Result<(), Box<dyn std::error::Error>> {
    let array = build_array(RegularChunkGrid::new(vec![10], vec![nz(5)])?, vec![3])?;

    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap();
    // The subchunk grid has varying edge lengths
    assert_eq!(array.subchunk_shape(), None);
    assert_eq!(subchunk_grid.array_shape(), &[10]);
    assert_eq!(subchunk_grid.grid_shape(), &[4]);
    assert_eq!(
        subchunk_grid.chunk_edge_lengths(0)?,
        vec![nz(3), nz(2), nz(3), nz(2)]
    );
    assert_eq!(
        subchunk_grid.subset(&[3])?,
        Some(ArraySubset::new_with_ranges(&[8..10]))
    );
    Ok(())
}

#[test]
#[allow(clippy::single_range_in_vec_init)]
fn sharding_nondivisible_subchunk_grid_rectilinear() -> Result<(), Box<dyn std::error::Error>> {
    let array = build_array(
        RectilinearChunkGrid::new(
            vec![15],
            &[ChunkEdgeLengths::Varying(vec![
                RunLengthElement::Single(nz(6)),
                RunLengthElement::Single(nz(9)),
            ])],
        )?,
        vec![4],
    )?;

    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap();
    assert_eq!(subchunk_grid.array_shape(), &[15]);
    assert_eq!(
        subchunk_grid.chunk_edge_lengths(0)?,
        vec![nz(4), nz(2), nz(4), nz(4), nz(1)]
    );
    assert_eq!(
        subchunk_grid.subset(&[4])?,
        Some(ArraySubset::new_with_ranges(&[14..15]))
    );
    Ok(())
}

const SHAPE: [u64; 2] = [20, 17];
const SHARD_SHAPE: [u64; 2] = [8, 6];

/// The array layouts under test.
#[derive(Clone, Copy, Debug)]
enum Layout {
    /// Subchunks `[3, 4]`.
    Sharded,
    /// Subchunks `[3, 4]` with nested subchunks `[2, 3]`.
    Nested,
    /// A transpose before sharding with subchunks `[4, 3]` (in the transposed shard shape).
    Transposed,
}

fn sharding_codec(layout: Layout) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
    let data_type = data_type::uint16();
    match layout {
        Layout::Sharded => ShardingCodecBuilder::new(vec![nz(3), nz(4)], &data_type).build_arc(),
        Layout::Nested => ShardingCodecBuilder::new(vec![nz(3), nz(4)], &data_type)
            .array_to_bytes_codec(
                ShardingCodecBuilder::new(vec![nz(2), nz(3)], &data_type).build_arc(),
            )
            .build_arc(),
        Layout::Transposed => ShardingCodecBuilder::new(vec![nz(4), nz(3)], &data_type).build_arc(),
    }
}

fn array_builder(layout: Option<Layout>) -> ArrayBuilder {
    let mut builder = ArrayBuilder::new(
        SHAPE.to_vec(),
        SHARD_SHAPE.to_vec(),
        data_type::uint16(),
        0u16,
    );
    if let Some(layout) = layout {
        if matches!(layout, Layout::Transposed) {
            builder.array_to_array_codecs(vec![Arc::new(TransposeCodec::new(
                TransposeOrder::new(&[1, 0]).unwrap(),
            ))]);
        }
        builder.array_to_bytes_codec(sharding_codec(layout));
    }
    builder
}

/// A sharded array.
fn sharded_array<TStorage: ?Sized>(
    store: Arc<TStorage>,
    layout: Layout,
    partial_encoding: bool,
) -> Result<Array<TStorage>, Box<dyn std::error::Error>> {
    let options = CodecOptions::default().with_experimental_partial_encoding(partial_encoding);
    Ok(build_from_metadata(&array_builder(Some(layout)), store)?.with_codec_options(options))
}

/// An unsharded reference array.
fn reference_array() -> Result<Array<MemoryStore>, Box<dyn std::error::Error>> {
    Ok(array_builder(None).build(Arc::new(MemoryStore::default()), "/array")?)
}

fn subset_data(subset: &ArraySubset, offset: u16) -> Vec<u16> {
    (0..subset.num_elements())
        .map(|i| u16::try_from(i).unwrap() + offset)
        .collect()
}

/// Array subset writes, including writes at the final shard boundary and writes of the fill value.
fn subset_writes() -> Vec<(ArraySubset, Vec<u16>)> {
    let whole = ArraySubset::new_with_shape(SHAPE.to_vec());
    let final_shard = ArraySubset::new_with_ranges(&[16..20, 12..17]);
    let final_elements = ArraySubset::new_with_ranges(&[18..20, 15..17]);
    let crossing = ArraySubset::new_with_ranges(&[5..13, 3..9]);
    let emptied = ArraySubset::new_with_ranges(&[0..3, 0..4]);
    let emptied_edge = ArraySubset::new_with_ranges(&[6..8, 4..6]);
    vec![
        (whole.clone(), subset_data(&whole, 1)),
        (final_shard.clone(), subset_data(&final_shard, 1000)),
        (final_elements.clone(), subset_data(&final_elements, 2000)),
        (crossing.clone(), subset_data(&crossing, 3000)),
        (emptied.clone(), vec![0; emptied.num_elements_usize()]),
        (
            emptied_edge.clone(),
            vec![0; emptied_edge.num_elements_usize()],
        ),
    ]
}

/// Scattered writes within the final shard (`[2, 2]`), which extends beyond the array shape.
fn scattered_write() -> (Vec<ArrayIndices>, Vec<u16>) {
    (
        vec![vec![3, 4], vec![0, 0], vec![2, 5], vec![3, 4], vec![1, 3]],
        vec![4000, 4001, 4002, 4003, 4004],
    )
}

fn read_subsets() -> Vec<ArraySubset> {
    vec![
        ArraySubset::new_with_shape(SHAPE.to_vec()),
        ArraySubset::new_with_ranges(&[19..20, 16..17]),
        ArraySubset::new_with_ranges(&[4..19, 5..16]),
    ]
}

/// Generic indexer reads within each listed chunk.
fn partial_chunk_reads() -> (Vec<[u64; 2]>, Vec<ArrayIndices>) {
    (
        vec![[0, 0], [1, 2], [2, 2]],
        vec![vec![7, 5], vec![0, 0], vec![6, 4], vec![3, 3]],
    )
}

fn assert_arrays_eq(
    array: &Array<MemoryStore>,
    reference: &Array<MemoryStore>,
) -> Result<(), Box<dyn std::error::Error>> {
    for subset in read_subsets() {
        assert_eq!(
            array.retrieve_array_subset::<Vec<u16>>(&subset)?,
            reference.retrieve_array_subset::<Vec<u16>>(&subset)?,
            "{subset:?}"
        );
    }

    let (chunks, indices) = partial_chunk_reads();
    for chunk_indices in chunks {
        assert_eq!(
            array.retrieve_partial_chunk::<Vec<u16>>(&chunk_indices, &indices)?,
            reference.retrieve_partial_chunk::<Vec<u16>>(&chunk_indices, &indices)?,
        );
    }

    // Subchunks within the array
    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap().clone();
    for subchunk_indices in subchunk_grid.iter_chunk_indices() {
        let subset = subchunk_grid.subset(&subchunk_indices)?.unwrap();
        if std::iter::zip(subset.end_exc(), SHAPE).all(|(end, shape)| end <= shape) {
            assert_eq!(
                array.retrieve_subchunk::<Vec<u16>>(&subchunk_indices)?,
                reference.retrieve_array_subset::<Vec<u16>>(&subset)?,
            );
        }
    }
    Ok(())
}

fn sharding_nondivisible_array_impl(
    layout: Layout,
    partial_encoding: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let array = sharded_array(Arc::new(MemoryStore::default()), layout, partial_encoding)?;
    let reference = reference_array()?;
    for (subset, data) in subset_writes() {
        array.store_array_subset(&subset, data.as_slice())?;
        reference.store_array_subset(&subset, data.as_slice())?;
        assert_arrays_eq(&array, &reference)?;
    }

    let (indices, data) = scattered_write();
    array.store_partial_chunk(&[2, 2], &indices, data.as_slice())?;
    reference.store_partial_chunk(&[2, 2], &indices, data.as_slice())?;
    assert_arrays_eq(&array, &reference)?;

    for chunk_indices in array.chunk_grid().iter_chunk_indices() {
        array.compact_chunk(&chunk_indices)?;
    }
    assert_arrays_eq(&array, &reference)?;
    Ok(())
}

#[test]
fn sharding_nondivisible_array() -> Result<(), Box<dyn std::error::Error>> {
    for layout in [Layout::Sharded, Layout::Nested, Layout::Transposed] {
        for partial_encoding in [false, true] {
            sharding_nondivisible_array_impl(layout, partial_encoding)?;
        }
    }
    Ok(())
}

#[test]
fn sharding_nondivisible_subchunk_grid_nested_and_transposed()
-> Result<(), Box<dyn std::error::Error>> {
    let array = sharded_array(Arc::new(MemoryStore::default()), Layout::Nested, false)?;
    let inner_grid = array
        .subchunk_grid_at_level(1)
        .as_chunk_grid()
        .unwrap()
        .clone();
    // Each [3, 4] subchunk (clipped to [3|2, 4|2]) contains [2, 3] subchunks, also clipped
    assert_eq!(
        inner_grid.chunk_edge_lengths(0)?,
        [2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2]
            .map(nz)
            .to_vec()
    );
    assert_eq!(
        inner_grid.chunk_edge_lengths(1)?,
        [3, 1, 2, 3, 1, 2, 3, 1, 2].map(nz).to_vec()
    );

    let array = sharded_array(Arc::new(MemoryStore::default()), Layout::Transposed, false)?;
    let subchunk_grid = array.subchunk_grid().as_chunk_grid().unwrap().clone();
    assert_eq!(
        subchunk_grid.chunk_edge_lengths(0)?,
        [3, 3, 2, 3, 3, 2, 3, 3, 2].map(nz).to_vec()
    );
    assert_eq!(
        subchunk_grid.chunk_edge_lengths(1)?,
        [4, 2, 4, 2, 4, 2].map(nz).to_vec()
    );
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn sharding_nondivisible_array_async() -> Result<(), Box<dyn std::error::Error>> {
    use zarrs_storage::store::AsyncMemoryStore;

    for layout in [Layout::Sharded, Layout::Nested, Layout::Transposed] {
        for partial_encoding in [false, true] {
            let array = sharded_array(Arc::new(AsyncMemoryStore::new()), layout, partial_encoding)?;
            let reference = reference_array()?;
            let assert_eq_reference = async |array: &Array<AsyncMemoryStore>| {
                for subset in read_subsets() {
                    assert_eq!(
                        array
                            .async_retrieve_array_subset::<Vec<u16>>(&subset)
                            .await
                            .unwrap(),
                        reference
                            .retrieve_array_subset::<Vec<u16>>(&subset)
                            .unwrap(),
                    );
                }
                let (chunks, indices) = partial_chunk_reads();
                for chunk_indices in chunks {
                    assert_eq!(
                        array
                            .async_retrieve_partial_chunk::<Vec<u16>>(&chunk_indices, &indices)
                            .await
                            .unwrap(),
                        reference
                            .retrieve_partial_chunk::<Vec<u16>>(&chunk_indices, &indices)
                            .unwrap(),
                    );
                }
            };

            for (subset, data) in subset_writes() {
                array
                    .async_store_array_subset(&subset, data.as_slice())
                    .await?;
                reference.store_array_subset(&subset, data.as_slice())?;
                assert_eq_reference(&array).await;
            }

            let (indices, data) = scattered_write();
            array
                .async_store_partial_chunk(&[2, 2], &indices, data.as_slice())
                .await?;
            reference.store_partial_chunk(&[2, 2], &indices, data.as_slice())?;
            assert_eq_reference(&array).await;

            for chunk_indices in array.chunk_grid().iter_chunk_indices() {
                array.async_compact_chunk(&chunk_indices).await?;
            }
            assert_eq_reference(&array).await;
        }
    }
    Ok(())
}

fn bytes_codec_chain() -> Arc<CodecChain> {
    Arc::new(CodecChain::new(
        vec![],
        Arc::new(BytesCodec::new(Some(Endianness::Little))),
        vec![],
    ))
}

/// A codec chain with a sharding codec with `[2]` subchunks.
fn sharding_codec_chain(data_type: &DataType) -> Arc<CodecChain> {
    Arc::new(CodecChain::new(
        vec![],
        ShardingCodecBuilder::new(vec![nz(2)], data_type).build_arc(),
        vec![],
    ))
}

/// Store and retrieve `data` in the whole array.
fn assert_store_round_trip<T>(
    array: &Array<MemoryStore>,
    data: T,
) -> Result<(), Box<dyn std::error::Error>>
where
    T: IntoArrayBytes<'static> + FromArrayBytes + Clone + PartialEq + std::fmt::Debug,
{
    let subset = ArraySubset::new_with_shape(array.shape().to_vec());
    array.store_array_subset(&subset, data.clone())?;
    assert_eq!(array.retrieve_array_subset::<T>(&subset)?, data);
    Ok(())
}

#[test]
fn sharding_nondivisible_nested_in_vlen_and_optional() -> Result<(), Box<dyn std::error::Error>> {
    // The inner data codecs encode a 1D array with a data-dependent length (5 bytes, 3 valid elements)
    let vlen = VlenCodec::new(
        bytes_codec_chain(),
        sharding_codec_chain(&data_type::uint8()),
        VlenIndexDataType::UInt64,
        VlenIndexLocation::Start,
    );
    let mut builder = ArrayBuilder::new(vec![4], vec![4], data_type::string(), "");
    builder.array_to_bytes_codec(Arc::new(vlen));
    let array = builder.build(Arc::new(MemoryStore::default()), "/array")?;
    let data = ["a", "bb", "", "cc"].map(String::from).to_vec();
    assert_store_round_trip(&array, data)?;

    let optional = OptionalCodec::new(
        bytes_codec_chain(),
        sharding_codec_chain(&data_type::uint16()),
    );
    let mut builder = ArrayBuilder::new(
        vec![4],
        vec![4],
        data_type::uint16().to_optional(),
        FillValue::from(None::<u16>),
    );
    builder.array_to_bytes_codec(Arc::new(optional));
    let array = builder.build(Arc::new(MemoryStore::default()), "/array")?;
    let data = vec![Some(1u16), None, Some(3), Some(4)];
    assert_store_round_trip(&array, data)?;
    Ok(())
}

/// Arrays written by zarr-python (`tests/data/sharding_nondivisible.py`) can be read.
#[test]
fn sharding_nondivisible_zarr_python_compat() -> Result<(), Box<dyn std::error::Error>> {
    for name in ["1d", "2d", "larger_than_shard", "nested", "rectilinear"] {
        let path = format!("tests/data/zarr_python_compat/sharding_nondivisible_{name}.zarr");
        let array = Array::open(Arc::new(FilesystemStore::new(path)?), "/")?;
        // The elements are the linearised indices, `0..N`
        let subsets = [
            array.subset_all(),
            // A selection crossing clipped subchunks and shard boundaries
            ArraySubset::from(array.shape().iter().map(|&s| 1..s - 1)),
        ];
        for subset in subsets {
            let expected: Vec<u16> = subset
                .linearised_indices(array.shape())?
                .into_iter()
                .map(u16::try_from)
                .collect::<Result<_, _>>()?;
            assert_eq!(
                array.retrieve_array_subset::<Vec<u16>>(&subset)?,
                expected,
                "{name}"
            );
        }
    }
    Ok(())
}
