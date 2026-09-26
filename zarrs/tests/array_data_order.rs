#![allow(missing_docs)]
#![cfg(feature = "transpose")]

use std::error::Error;
use std::fmt::Debug;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::chunk_cache::{
    ChunkCache, ChunkCacheDecodedLruChunkLimit, ChunkCacheEncodedLruChunkLimit,
    ChunkCachePartialDecoderLruChunkLimit,
};
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::codec::{TransposeCodec, TransposeOrder};
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView, ArrayCached,
    ArrayDataOrder, ArrayError, ArrayMetadata, ArrayReadOps, ArraySubset, ArrayUpdateOps,
    ArrayWriteOps, CodecOptions, DataType, ElementOwned, FillValue, IntoArrayBytes, data_type,
};
use zarrs::metadata::v2::ArrayMetadataV2;
use zarrs::storage::store::MemoryStore;

type TestResult = Result<(), Box<dyn Error>>;

const SHAPE: [u64; 3] = [5, 7, 6];
const CHUNK_SHAPE: [u64; 3] = [2, 3, 4];

/// Convert a C-order buffer with `shape` to F order.
fn to_f<T: Clone>(c: &[T], shape: &[u64]) -> Vec<T> {
    let shape_reversed: Vec<u64> = shape.iter().rev().copied().collect();
    ArraySubset::new_with_shape(shape_reversed)
        .indices()
        .into_iter()
        .map(|indices_reversed| {
            let indices: Vec<u64> = indices_reversed.iter().rev().copied().collect();
            let linear = zarrs::array::ravel_indices(&indices, shape).unwrap();
            c[usize::try_from(linear).unwrap()].clone()
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Codecs {
    Bytes,
    TransposeReversed,
    TransposePermuted,
    ShardingTransposeReversed,
}

fn nz(shape: &[u64]) -> Vec<NonZeroU64> {
    shape.iter().map(|&s| NonZeroU64::new(s).unwrap()).collect()
}

fn transpose(order: &[usize]) -> Arc<TransposeCodec> {
    Arc::new(TransposeCodec::new(TransposeOrder::new(order).unwrap()))
}

fn build_array(
    codecs: Codecs,
    data_type: DataType,
    fill_value: FillValue,
) -> Result<Array<MemoryStore>, Box<dyn Error>> {
    build_array_in(Arc::new(MemoryStore::new()), codecs, data_type, fill_value)
}

fn build_array_in<S: ?Sized>(
    store: Arc<S>,
    codecs: Codecs,
    data_type: DataType,
    fill_value: FillValue,
) -> Result<Array<S>, Box<dyn Error>> {
    let mut builder = ArrayBuilder::new(
        SHAPE.to_vec(),
        CHUNK_SHAPE.to_vec(),
        data_type.clone(),
        fill_value,
    );
    match codecs {
        Codecs::Bytes => {}
        Codecs::TransposeReversed => {
            builder.array_to_array_codecs(vec![transpose(&[2, 1, 0])]);
        }
        Codecs::TransposePermuted => {
            builder.array_to_array_codecs(vec![transpose(&[1, 0, 2])]);
        }
        Codecs::ShardingTransposeReversed => {
            // The subchunk shape is in the transposed representation of the chunk
            let mut sharding = ShardingCodecBuilder::new(nz(&[2, 3, 1]), &data_type);
            sharding.array_to_array_codecs(vec![transpose(&[2, 1, 0])]);
            builder.array_to_bytes_codec(sharding.build_arc());
        }
    }
    Ok(builder.build(store, "/array")?)
}

fn build_array_v2_f() -> Result<Array<MemoryStore>, Box<dyn Error>> {
    let metadata: ArrayMetadataV2 = serde_json::from_str(
        r#"{
            "zarr_format": 2,
            "shape": [5, 7, 6],
            "chunks": [2, 3, 4],
            "dtype": "<u2",
            "compressor": null,
            "fill_value": 0,
            "order": "F",
            "filters": null
        }"#,
    )?;
    Ok(Array::new_with_metadata(
        Arc::new(MemoryStore::new()),
        "/array",
        ArrayMetadata::V2(metadata),
    )?)
}

/// Check that every read of `array_f` is the F-order equivalent of the corresponding read of `array_c`.
fn check_reads<T, A>(array_c: &A, array_f: &A) -> TestResult
where
    T: ElementOwned + Clone + PartialEq + Debug,
    A: ArrayReadOps,
{
    // Chunks, including a missing chunk
    for chunk_indices in [[0, 0, 0], [1, 1, 1], [2, 2, 1]] {
        let c = array_c.retrieve_chunk::<Vec<T>>(&chunk_indices)?;
        let f = array_f.retrieve_chunk::<Vec<T>>(&chunk_indices)?;
        assert_eq!(f, to_f(&c, &CHUNK_SHAPE), "chunk {chunk_indices:?}");
    }

    // Partial chunk with an array subset
    let chunk_subset = ArraySubset::new_with_ranges(&[0..2, 1..3, 1..4]);
    let c = array_c.retrieve_partial_chunk::<Vec<T>>(&[1, 0, 0], &chunk_subset)?;
    let f = array_f.retrieve_partial_chunk::<Vec<T>>(&[1, 0, 0], &chunk_subset)?;
    assert_eq!(f, to_f(&c, chunk_subset.shape()));

    // Partial chunk with a list of indices (a 1D output is identical in C and F order)
    let indices: Vec<Vec<u64>> = vec![vec![0, 0, 0], vec![1, 2, 3], vec![1, 0, 2], vec![0, 1, 1]];
    let c = array_c.retrieve_partial_chunk::<Vec<T>>(&[0, 1, 0], &indices)?;
    let f = array_f.retrieve_partial_chunk::<Vec<T>>(&[0, 1, 0], &indices)?;
    assert_eq!(f, c);

    // Array subsets: a single partial chunk, multiple chunks (including a missing chunk), the whole array
    for subset in [
        ArraySubset::new_with_ranges(&[0..2, 0..3, 1..3]),
        ArraySubset::new_with_ranges(&[1..5, 2..7, 0..6]),
        ArraySubset::new_with_shape(SHAPE.to_vec()),
    ] {
        let c = array_c.retrieve_array_subset::<Vec<T>>(&subset)?;
        let f = array_f.retrieve_array_subset::<Vec<T>>(&subset)?;
        assert_eq!(f, to_f(&c, subset.shape()), "subset {subset:?}");
    }

    // Chunks
    let chunks = ArraySubset::new_with_ranges(&[0..2, 1..3, 0..2]);
    let c = array_c.retrieve_chunks::<Vec<T>>(&chunks)?;
    let f = array_f.retrieve_chunks::<Vec<T>>(&chunks)?;
    assert_eq!(f, to_f(&c, array_c.chunks_subset(&chunks)?.shape()));

    Ok(())
}

/// Read into a preallocated `u16` buffer with the given shape.
fn read_into_u16(
    shape: &[u64],
    read: impl FnOnce(ArrayBytesDecodeIntoTarget<'_>) -> Result<(), ArrayError>,
) -> Result<Vec<u16>, Box<dyn Error>> {
    let num_elements = usize::try_from(shape.iter().product::<u64>())?;
    let mut output = vec![0u8; num_elements * 2];
    {
        let output_slice = unsafe_cell_slice::UnsafeCellSlice::new(&mut output);
        let mut view = unsafe {
            // SAFETY: this is the only view over output and covers it exactly.
            ArrayBytesFixedDisjointView::new(
                output_slice,
                2,
                shape,
                ArraySubset::new_with_shape(shape.to_vec()),
            )?
        };
        read(ArrayBytesDecodeIntoTarget::Fixed(&mut view))?;
    }
    Ok(output
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_ne_bytes([b[0], b[1]]))
        .collect())
}

/// Check the `_into` reads and `ndarray` reads for a `uint16` array.
fn check_reads_u16<A: ArrayReadOps>(array_c: &A, array_f: &A) -> TestResult {
    // The output buffer of an F-order read has the reversed shape
    let reversed = |shape: &[u64]| shape.iter().rev().copied().collect::<Vec<_>>();

    let c = array_c.retrieve_chunk::<Vec<u16>>(&[1, 1, 0])?;
    let f = read_into_u16(&reversed(&CHUNK_SHAPE), |target| {
        array_f.retrieve_chunk_into(&[1, 1, 0], target)
    })?;
    assert_eq!(f, to_f(&c, &CHUNK_SHAPE));

    let chunk_subset = ArraySubset::new_with_ranges(&[1..2, 0..3, 1..3]);
    let c = array_c.retrieve_partial_chunk::<Vec<u16>>(&[0, 1, 1], &chunk_subset)?;
    let f = read_into_u16(&reversed(chunk_subset.shape()), |target| {
        array_f.retrieve_partial_chunk_into(&[0, 1, 1], &chunk_subset, target)
    })?;
    assert_eq!(f, to_f(&c, chunk_subset.shape()));

    for subset in [
        ArraySubset::new_with_ranges(&[0..2, 0..3, 1..3]),
        ArraySubset::new_with_ranges(&[1..5, 2..7, 0..6]),
    ] {
        let c = array_c.retrieve_array_subset::<Vec<u16>>(&subset)?;
        let f = read_into_u16(&reversed(subset.shape()), |target| {
            array_f.retrieve_array_subset_into(&subset, target)
        })?;
        assert_eq!(f, to_f(&c, subset.shape()), "subset {subset:?}");
    }

    #[cfg(feature = "ndarray")]
    {
        let subset = ArraySubset::new_with_ranges(&[1..5, 2..7, 0..6]);
        let c = array_c.retrieve_array_subset::<ndarray::ArrayD<u16>>(&subset)?;
        let f = array_f.retrieve_array_subset::<ndarray::ArrayD<u16>>(&subset)?;
        assert_eq!(c, f);
        assert!(c.is_standard_layout());
        assert!(f.t().is_standard_layout(), "expected an F-order ndarray");

        let c = array_c.retrieve_chunk::<ndarray::Array3<u16>>(&[0, 0, 0])?;
        let f = array_f.retrieve_chunk::<ndarray::Array3<u16>>(&[0, 0, 0])?;
        assert_eq!(c, f);
        assert!(f.t().is_standard_layout(), "expected an F-order ndarray");
    }

    Ok(())
}

fn populate<T, A>(array: &A, elements: impl Fn(usize) -> T) -> TestResult
where
    A: ArrayWriteOps + ArrayUpdateOps,
    Vec<T>: IntoArrayBytes<'static>,
{
    let num_elements = usize::try_from(SHAPE.iter().product::<u64>())?;
    let data: Vec<T> = (0..num_elements).map(elements).collect();
    array.store_array_subset(&ArraySubset::new_with_shape(SHAPE.to_vec()), data)?;
    array.erase_chunk(&[1, 1, 1])?;
    Ok(())
}

fn data_order_impl<T>(
    array: &Array<MemoryStore>,
    elements: impl Fn(usize) -> T,
    check_u16: bool,
) -> TestResult
where
    T: ElementOwned + Clone + PartialEq + Debug,
    Vec<T>: IntoArrayBytes<'static>,
{
    populate(array, elements)?;
    let array_f = array.with_data_order(ArrayDataOrder::F)?;
    assert_eq!(array_f.data_order(), ArrayDataOrder::F);
    check_reads::<T, _>(array, &array_f)?;
    if check_u16 {
        check_reads_u16(array, &array_f)?;
    }

    Ok(())
}

const NUM_WRITE_STEPS: usize = 7;

/// Data for a write of `shape` in `data_order`, with values offset by `offset`.
fn write_data<T: Clone>(
    shape: &[u64],
    data_order: ArrayDataOrder,
    offset: usize,
    elements: &impl Fn(usize) -> T,
) -> Vec<T> {
    let num_elements = usize::try_from(shape.iter().product::<u64>()).unwrap();
    let data: Vec<T> = (0..num_elements).map(|i| elements(i + offset)).collect();
    match data_order {
        ArrayDataOrder::C => data,
        ArrayDataOrder::F => to_f(&data, shape),
    }
}

/// Apply write `step` to `array` with data in `data_order`.
fn write_step<T, A>(
    array: &A,
    step: usize,
    data_order: ArrayDataOrder,
    elements: &impl Fn(usize) -> T,
) -> TestResult
where
    T: Clone,
    Vec<T>: IntoArrayBytes<'static>,
    A: ArrayWriteOps + ArrayUpdateOps,
{
    let offset = step * 1000 + 1;
    match step {
        0 => {
            let subset = ArraySubset::new_with_shape(SHAPE.to_vec());
            array.store_array_subset(
                &subset,
                write_data(subset.shape(), data_order, offset, elements),
            )?;
        }
        1 => {
            array.store_chunk(
                &[0, 0, 0],
                write_data(&CHUNK_SHAPE, data_order, offset, elements),
            )?;
        }
        2 => {
            let chunks = ArraySubset::new_with_ranges(&[0..2, 1..3, 0..2]);
            let shape = array.chunks_subset(&chunks)?.shape().to_vec();
            array.store_chunks(&chunks, write_data(&shape, data_order, offset, elements))?;
        }
        3 => {
            let chunk_subset = ArraySubset::new_with_ranges(&[0..2, 1..3, 1..4]);
            array.store_partial_chunk(
                &[1, 0, 0],
                &chunk_subset,
                write_data(chunk_subset.shape(), data_order, offset, elements),
            )?;
        }
        4 => {
            // A 1D output is identical in C and F order
            let indices: Vec<Vec<u64>> = vec![vec![0, 0, 0], vec![1, 2, 3], vec![1, 0, 2]];
            array.store_partial_chunk(
                &[0, 1, 0],
                &indices,
                write_data(&[3], data_order, offset, elements),
            )?;
        }
        5 => {
            let subset = ArraySubset::new_with_ranges(&[1..5, 2..7, 0..6]);
            array.store_array_subset(
                &subset,
                write_data(subset.shape(), data_order, offset, elements),
            )?;
        }
        6 => {
            let subset = ArraySubset::new_with_ranges(&[0..2, 0..3, 1..3]);
            array.store_array_subset(
                &subset,
                write_data(subset.shape(), data_order, offset, elements),
            )?;
        }
        _ => unreachable!(),
    }
    Ok(())
}

/// Check that every F-order write is equivalent to the corresponding C-order write.
fn check_writes<T>(
    new_array: impl Fn() -> Result<Array<MemoryStore>, Box<dyn Error>>,
    elements: impl Fn(usize) -> T,
) -> TestResult
where
    T: ElementOwned + Clone + PartialEq + Debug,
    Vec<T>: IntoArrayBytes<'static>,
{
    for partial_encoding in [false, true] {
        let codec_options =
            CodecOptions::default().with_experimental_partial_encoding(partial_encoding);
        let array_c = new_array()?.with_codec_options(codec_options);
        let array_f_storage = new_array()?.with_codec_options(codec_options);
        let array_f = array_f_storage.with_data_order(ArrayDataOrder::F)?;
        let subset_all = ArraySubset::new_with_shape(SHAPE.to_vec());
        for step in 0..NUM_WRITE_STEPS {
            write_step(&array_c, step, ArrayDataOrder::C, &elements)?;
            write_step(&array_f, step, ArrayDataOrder::F, &elements)?;
            assert_eq!(
                array_f_storage.retrieve_array_subset::<Vec<T>>(&subset_all)?,
                array_c.retrieve_array_subset::<Vec<T>>(&subset_all)?,
                "partial_encoding={partial_encoding} step {step}"
            );
        }
    }
    Ok(())
}

const ALL_CODECS: [Codecs; 4] = [
    Codecs::Bytes,
    Codecs::TransposeReversed,
    Codecs::TransposePermuted,
    Codecs::ShardingTransposeReversed,
];

// The `sharding` codec does not support optional data types
const OPTIONAL_CODECS: [Codecs; 3] = [
    Codecs::Bytes,
    Codecs::TransposeReversed,
    Codecs::TransposePermuted,
];

#[test]
fn array_data_order_uint16() -> TestResult {
    for codecs in ALL_CODECS {
        let array = build_array(codecs, data_type::uint16(), FillValue::from(0u16))?;
        data_order_impl(&array, |i| u16::try_from(i).unwrap() + 1, true)?;
        check_writes(
            || build_array(codecs, data_type::uint16(), FillValue::from(0u16)),
            |i| u16::try_from(i % 65536).unwrap(),
        )?;
    }
    Ok(())
}

#[test]
fn array_data_order_uint16_v2_f() -> TestResult {
    let array = build_array_v2_f()?;
    data_order_impl(&array, |i| u16::try_from(i).unwrap() + 1, true)?;
    check_writes(build_array_v2_f, |i| u16::try_from(i % 65536).unwrap())
}

#[test]
fn array_data_order_string() -> TestResult {
    for codecs in ALL_CODECS {
        let array = build_array(codecs, data_type::string(), FillValue::from(""))?;
        data_order_impl(&array, |i| "x".repeat(i % 5), false)?;
        check_writes(
            || build_array(codecs, data_type::string(), FillValue::from("")),
            |i| format!("s{i}"),
        )?;
    }
    Ok(())
}

#[test]
fn array_data_order_optional_uint8() -> TestResult {
    for codecs in OPTIONAL_CODECS {
        let array = build_array(
            codecs,
            data_type::uint8().to_optional(),
            FillValue::from(None::<u8>),
        )?;
        data_order_impl(
            &array,
            |i| (i % 3 != 0).then(|| u8::try_from(i % 256).unwrap()),
            false,
        )?;
        check_writes(
            || {
                build_array(
                    codecs,
                    data_type::uint8().to_optional(),
                    FillValue::from(None::<u8>),
                )
            },
            |i| (i % 3 != 0).then(|| u8::try_from(i % 256).unwrap()),
        )?;
    }
    Ok(())
}

#[test]
fn array_data_order_optional_string() -> TestResult {
    for codecs in OPTIONAL_CODECS {
        let array = build_array(
            codecs,
            data_type::string().to_optional(),
            FillValue::new_optional_null(),
        )?;
        data_order_impl(&array, |i| (i % 4 != 0).then(|| "y".repeat(i % 3)), false)?;
        check_writes(
            || {
                build_array(
                    codecs,
                    data_type::string().to_optional(),
                    FillValue::new_optional_null(),
                )
            },
            |i| (i % 4 != 0).then(|| format!("y{i}")),
        )?;
    }
    Ok(())
}

#[test]
fn array_data_order_1d() -> TestResult {
    // F order is identical to C order for 1D arrays
    let array = ArrayBuilder::new(vec![10], vec![3], data_type::uint16(), 0u16)
        .build(Arc::new(MemoryStore::new()), "/array")?;
    let data: Vec<u16> = (0..10).collect();
    array.store_array_subset(&array.subset_all(), data.clone())?;
    let array_f = array.with_data_order(ArrayDataOrder::F)?;
    assert_eq!(
        array_f.retrieve_array_subset::<Vec<u16>>(&array.subset_all())?,
        data
    );
    Ok(())
}

#[cfg(feature = "async")]
mod async_tests {
    use super::*;
    use zarrs::array::AsyncArrayReadOps;
    use zarrs::array::chunk_cache::{
        AsyncChunkCache, AsyncChunkCacheDecodedLruChunkLimit, AsyncChunkCacheEncodedLruChunkLimit,
        AsyncChunkCachePartialDecoderLruChunkLimit,
    };
    use zarrs::storage::store::AsyncMemoryStore;

    async fn check_reads_async<T, A>(array_c: &A, array_f: &A) -> TestResult
    where
        T: ElementOwned + Clone + PartialEq + Debug + Send,
        A: AsyncArrayReadOps,
    {
        for chunk_indices in [[0, 0, 0], [1, 1, 1], [2, 2, 1]] {
            let c = array_c
                .async_retrieve_chunk::<Vec<T>>(&chunk_indices)
                .await?;
            let f = array_f
                .async_retrieve_chunk::<Vec<T>>(&chunk_indices)
                .await?;
            assert_eq!(f, to_f(&c, &CHUNK_SHAPE), "chunk {chunk_indices:?}");
        }

        let chunk_subset = ArraySubset::new_with_ranges(&[0..2, 1..3, 1..4]);
        let c = array_c
            .async_retrieve_partial_chunk::<Vec<T>>(&[1, 0, 0], &chunk_subset)
            .await?;
        let f = array_f
            .async_retrieve_partial_chunk::<Vec<T>>(&[1, 0, 0], &chunk_subset)
            .await?;
        assert_eq!(f, to_f(&c, chunk_subset.shape()));

        let indices: Vec<Vec<u64>> = vec![vec![0, 0, 0], vec![1, 2, 3], vec![1, 0, 2]];
        let c = array_c
            .async_retrieve_partial_chunk::<Vec<T>>(&[0, 1, 0], &indices)
            .await?;
        let f = array_f
            .async_retrieve_partial_chunk::<Vec<T>>(&[0, 1, 0], &indices)
            .await?;
        assert_eq!(f, c);

        for subset in [
            ArraySubset::new_with_ranges(&[0..2, 0..3, 1..3]),
            ArraySubset::new_with_ranges(&[1..5, 2..7, 0..6]),
        ] {
            let c = array_c
                .async_retrieve_array_subset::<Vec<T>>(&subset)
                .await?;
            let f = array_f
                .async_retrieve_array_subset::<Vec<T>>(&subset)
                .await?;
            assert_eq!(f, to_f(&c, subset.shape()), "subset {subset:?}");
        }
        Ok(())
    }

    async fn check_reads_async_u16<A: AsyncArrayReadOps>(array_c: &A, array_f: &A) -> TestResult {
        let reversed = |shape: &[u64]| shape.iter().rev().copied().collect::<Vec<_>>();
        for subset in [
            ArraySubset::new_with_ranges(&[0..2, 0..3, 1..3]),
            ArraySubset::new_with_ranges(&[1..5, 2..7, 0..6]),
        ] {
            let c = array_c
                .async_retrieve_array_subset::<Vec<u16>>(&subset)
                .await?;
            let shape = reversed(subset.shape());
            let num_elements = usize::try_from(shape.iter().product::<u64>())?;
            let mut output = vec![0u8; num_elements * 2];
            {
                let output_slice = unsafe_cell_slice::UnsafeCellSlice::new(&mut output);
                let mut view = unsafe {
                    // SAFETY: this is the only view over output and covers it exactly.
                    ArrayBytesFixedDisjointView::new(
                        output_slice,
                        2,
                        &shape,
                        ArraySubset::new_with_shape(shape.clone()),
                    )?
                };
                array_f
                    .async_retrieve_array_subset_into(
                        &subset,
                        ArrayBytesDecodeIntoTarget::Fixed(&mut view),
                    )
                    .await?;
            }
            let f: Vec<u16> = output
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_ne_bytes([b[0], b[1]]))
                .collect();
            assert_eq!(f, to_f(&c, subset.shape()), "subset {subset:?}");
        }
        Ok(())
    }

    /// Async variant of [`write_step`](super::write_step).
    async fn write_step_async<T>(
        array: &Array<AsyncMemoryStore>,
        step: usize,
        data_order: ArrayDataOrder,
        elements: &impl Fn(usize) -> T,
    ) -> TestResult
    where
        T: Clone + Send,
        Vec<T>: IntoArrayBytes<'static> + Send,
    {
        let offset = step * 1000 + 1;
        match step {
            0 => {
                let subset = ArraySubset::new_with_shape(SHAPE.to_vec());
                let data = write_data(subset.shape(), data_order, offset, elements);
                array.async_store_array_subset(&subset, data).await?;
            }
            1 => {
                let data = write_data(&CHUNK_SHAPE, data_order, offset, elements);
                array.async_store_chunk(&[0, 0, 0], data).await?;
            }
            2 => {
                let chunks = ArraySubset::new_with_ranges(&[0..2, 1..3, 0..2]);
                let shape = array.chunks_subset(&chunks)?.shape().to_vec();
                let data = write_data(&shape, data_order, offset, elements);
                array.async_store_chunks(&chunks, data).await?;
            }
            3 => {
                let chunk_subset = ArraySubset::new_with_ranges(&[0..2, 1..3, 1..4]);
                let data = write_data(chunk_subset.shape(), data_order, offset, elements);
                array
                    .async_store_partial_chunk(&[1, 0, 0], &chunk_subset, data)
                    .await?;
            }
            4 => {
                let indices: Vec<Vec<u64>> = vec![vec![0, 0, 0], vec![1, 2, 3], vec![1, 0, 2]];
                let data = write_data(&[3], data_order, offset, elements);
                array
                    .async_store_partial_chunk(&[0, 1, 0], &indices, data)
                    .await?;
            }
            5 => {
                let subset = ArraySubset::new_with_ranges(&[1..5, 2..7, 0..6]);
                let data = write_data(subset.shape(), data_order, offset, elements);
                array.async_store_array_subset(&subset, data).await?;
            }
            6 => {
                let subset = ArraySubset::new_with_ranges(&[0..2, 0..3, 1..3]);
                let data = write_data(subset.shape(), data_order, offset, elements);
                array.async_store_array_subset(&subset, data).await?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Async variant of [`check_writes`](super::check_writes).
    async fn check_writes_async<T>(
        new_array: impl Fn() -> Result<Array<AsyncMemoryStore>, Box<dyn Error>>,
        elements: &impl Fn(usize) -> T,
    ) -> TestResult
    where
        T: ElementOwned + Clone + PartialEq + Debug + Send,
        Vec<T>: IntoArrayBytes<'static> + Send,
    {
        for partial_encoding in [false, true] {
            let codec_options =
                CodecOptions::default().with_experimental_partial_encoding(partial_encoding);
            let array_c = new_array()?.with_codec_options(codec_options);
            let array_f_storage = new_array()?.with_codec_options(codec_options);
            let array_f = array_f_storage.with_data_order(ArrayDataOrder::F)?;
            let subset_all = ArraySubset::new_with_shape(SHAPE.to_vec());
            for step in 0..NUM_WRITE_STEPS {
                write_step_async(&array_c, step, ArrayDataOrder::C, elements).await?;
                write_step_async(&array_f, step, ArrayDataOrder::F, elements).await?;
                assert_eq!(
                    array_f_storage
                        .async_retrieve_array_subset::<Vec<T>>(&subset_all)
                        .await?,
                    array_c
                        .async_retrieve_array_subset::<Vec<T>>(&subset_all)
                        .await?,
                    "partial_encoding={partial_encoding} step {step}"
                );
            }
        }
        Ok(())
    }

    async fn data_order_async_impl<T>(
        codecs: Codecs,
        data_type: DataType,
        fill_value: FillValue,
        elements: impl Fn(usize) -> T,
        check_u16: bool,
    ) -> TestResult
    where
        T: ElementOwned + Clone + PartialEq + Debug + Send,
        Vec<T>: IntoArrayBytes<'static> + Send,
    {
        let array = build_array_in(
            Arc::new(AsyncMemoryStore::new()),
            codecs,
            data_type,
            fill_value,
        )?;
        let num_elements = usize::try_from(SHAPE.iter().product::<u64>())?;
        let data: Vec<T> = (0..num_elements).map(&elements).collect();
        array
            .async_store_array_subset(&ArraySubset::new_with_shape(SHAPE.to_vec()), data)
            .await?;
        array.async_erase_chunk(&[1, 1, 1]).await?;

        let array_f = array.with_data_order(ArrayDataOrder::F)?;
        check_reads_async::<T, _>(&array, &array_f).await?;
        if check_u16 {
            check_reads_async_u16(&array, &array_f).await?;
        }
        check_writes_async(
            || {
                build_array_in(
                    Arc::new(AsyncMemoryStore::new()),
                    codecs,
                    array.data_type().clone(),
                    array.fill_value().clone(),
                )
            },
            &elements,
        )
        .await?;
        Ok(())
    }

    #[tokio::test]
    async fn array_data_order_async() -> TestResult {
        for codecs in ALL_CODECS {
            data_order_async_impl(
                codecs,
                data_type::uint16(),
                FillValue::from(0u16),
                |i| u16::try_from(i).unwrap() + 1,
                true,
            )
            .await?;
            data_order_async_impl(
                codecs,
                data_type::string(),
                FillValue::from(""),
                |i| "x".repeat(i % 5),
                false,
            )
            .await?;
        }
        for codecs in OPTIONAL_CODECS {
            data_order_async_impl(
                codecs,
                data_type::uint8().to_optional(),
                FillValue::from(None::<u8>),
                |i| (i % 3 != 0).then(|| u8::try_from(i % 256).unwrap()),
                false,
            )
            .await?;
        }
        Ok(())
    }

    async fn data_order_async_cached_impl<C: AsyncChunkCache>(
        new_cache: impl Fn() -> C,
    ) -> TestResult {
        for codecs in ALL_CODECS {
            let array = build_array_in(
                Arc::new(AsyncMemoryStore::new()),
                codecs,
                data_type::uint16(),
                FillValue::from(0u16),
            )?;
            let num_elements = usize::try_from(SHAPE.iter().product::<u64>())?;
            let data: Vec<u16> = (0..num_elements)
                .map(|i| u16::try_from(i).unwrap() + 1)
                .collect();
            array
                .async_store_array_subset(&ArraySubset::new_with_shape(SHAPE.to_vec()), data)
                .await?;
            array.async_erase_chunk(&[1, 1, 1]).await?;

            // The C and F order arrays share the cache
            let array_c = ArrayCached::new(Arc::new(array), new_cache());
            let array_f = array_c.with_data_order(ArrayDataOrder::F)?;
            check_reads_async::<u16, _>(&array_c, &array_f).await?;
            check_reads_async_u16(&array_c, &array_f).await?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn array_data_order_async_cached() -> TestResult {
        data_order_async_cached_impl(|| AsyncChunkCacheEncodedLruChunkLimit::new(4)).await?;
        data_order_async_cached_impl(|| AsyncChunkCacheDecodedLruChunkLimit::new(4)).await?;
        data_order_async_cached_impl(|| AsyncChunkCachePartialDecoderLruChunkLimit::new(4)).await
    }
}

fn data_order_cached_impl<C: ChunkCache + 'static>(new_cache: impl Fn() -> C) -> TestResult {
    for codecs in ALL_CODECS {
        let array = build_array(codecs, data_type::uint16(), FillValue::from(0u16))?;
        populate(&array, |i| u16::try_from(i).unwrap() + 1)?;
        // The C and F order arrays share the cache
        let array_c = ArrayCached::new(Arc::new(array), new_cache());
        let array_f = array_c.with_data_order(ArrayDataOrder::F)?;
        check_reads::<u16, _>(&array_c, &array_f)?;
        check_reads_u16(&array_c, &array_f)?;
        // Read in the opposite order so the F order array populates the cache first
        let array_c = ArrayCached::new(array_c.array().clone(), new_cache());
        let array_f = array_c.with_data_order(ArrayDataOrder::F)?;
        let subset = ArraySubset::new_with_ranges(&[1..5, 2..7, 0..6]);
        let f = array_f.retrieve_array_subset::<Vec<u16>>(&subset)?;
        let c = array_c.retrieve_array_subset::<Vec<u16>>(&subset)?;
        assert_eq!(f, to_f(&c, subset.shape()));
        assert_eq!(
            c,
            array_c.array().retrieve_array_subset::<Vec<u16>>(&subset)?
        );

        // Write in F order through the cache, then read in C order from the shared cache
        let data_c: Vec<u16> = (0..subset.num_elements_usize())
            .map(|i| u16::try_from(i).unwrap() + 5000)
            .collect();
        array_f.store_array_subset(&subset, to_f(&data_c, subset.shape()))?;
        assert_eq!(array_c.retrieve_array_subset::<Vec<u16>>(&subset)?, data_c);
        assert_eq!(
            array_f.retrieve_array_subset::<Vec<u16>>(&subset)?,
            to_f(&data_c, subset.shape())
        );
    }
    Ok(())
}

#[test]
fn array_data_order_cached_string_optional() -> TestResult {
    for codecs in OPTIONAL_CODECS {
        let array = build_array(codecs, data_type::string(), FillValue::from(""))?;
        populate(&array, |i| "x".repeat(i % 5))?;
        let array_c = ArrayCached::new(Arc::new(array), ChunkCacheDecodedLruChunkLimit::new(4));
        let array_f = array_c.with_data_order(ArrayDataOrder::F)?;
        check_reads::<String, _>(&array_c, &array_f)?;

        let array = build_array(
            codecs,
            data_type::uint8().to_optional(),
            FillValue::from(None::<u8>),
        )?;
        populate(&array, |i| {
            (i % 3 != 0).then(|| u8::try_from(i % 256).unwrap())
        })?;
        let array_c = ArrayCached::new(Arc::new(array), ChunkCacheDecodedLruChunkLimit::new(4));
        let array_f = array_c.with_data_order(ArrayDataOrder::F)?;
        check_reads::<Option<u8>, _>(&array_c, &array_f)?;
    }
    Ok(())
}

#[test]
fn array_data_order_cached() -> TestResult {
    data_order_cached_impl(|| ChunkCacheEncodedLruChunkLimit::new(4))?;
    data_order_cached_impl(|| ChunkCacheDecodedLruChunkLimit::new(4))?;
    data_order_cached_impl(|| ChunkCachePartialDecoderLruChunkLimit::new(4))
}
