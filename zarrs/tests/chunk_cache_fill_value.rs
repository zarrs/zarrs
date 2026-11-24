#![allow(missing_docs)]
#![cfg(feature = "bitround")]

//! Missing chunks retain the array fill when a codec transforms its encoded fill.

use std::error::Error;
use std::sync::Arc;

use unsafe_cell_slice::UnsafeCellSlice;
use zarrs::array::chunk_cache::ChunkCachePartialDecoderLruChunkLimit;
use zarrs::array::codec::BitroundCodec;
use zarrs::array::{
    ArrayBuilder, ArrayBytesFixedDisjointView, ArrayBytesTarget, ArrayCached, ArraySubset,
    Resources, data_type,
};
use zarrs::storage::store::MemoryStore;

const FILL_VALUE: f32 = 1.234_567_9;

fn builder() -> ArrayBuilder {
    let mut builder = ArrayBuilder::new(vec![4], vec![2], data_type::float32(), FILL_VALUE);
    builder.array_to_array_codecs(vec![Arc::new(BitroundCodec::new(3))]);
    builder
}

fn output_target<'a>(
    output: &'a mut [f32],
    shape: &'a [u64],
) -> Result<ArrayBytesTarget<'a>, Box<dyn Error>> {
    // SAFETY: this is the only view of output and covers it exactly.
    let view = unsafe {
        ArrayBytesFixedDisjointView::new(
            UnsafeCellSlice::new(bytemuck::cast_slice_mut::<f32, u8>(output)),
            size_of::<f32>(),
            shape,
            ArraySubset::new_with_shape(shape),
        )?
    };
    Ok(ArrayBytesTarget::Fixed(view))
}

#[test]
fn missing_cached_chunks_preserve_lossy_fill_value() -> Result<(), Box<dyn Error>> {
    let array = builder().build_arc(Arc::new(MemoryStore::default()), "/array")?;
    let cached = ArrayCached::new(array, ChunkCachePartialDecoderLruChunkLimit::new(4));
    let resources = Resources::default();
    let subset = cached.array().subset_all();

    // Check both newly created and cached partial decoders.
    for _ in 0..2 {
        let mut chunk = [0.0; 2];
        cached.retrieve_chunk_into(&[0], &mut output_target(&mut chunk, &[2])?, &resources)?;
        assert_eq!(chunk, [FILL_VALUE; 2]);
        assert_eq!(cached.retrieve_chunk::<Vec<f32>>(&[0], &resources)?, chunk);

        assert_eq!(
            cached.retrieve_array_subset::<Vec<f32>>(&subset, &resources)?,
            [FILL_VALUE; 4]
        );
    }
    Ok(())
}

#[cfg(feature = "async")]
#[tokio::test]
async fn async_missing_cached_chunks_preserve_lossy_fill_value() -> Result<(), Box<dyn Error>> {
    use zarrs::array::chunk_cache::AsyncChunkCachePartialDecoderLruChunkLimit;
    use zarrs::storage::store::AsyncMemoryStore;

    let array = builder().build_arc(Arc::new(AsyncMemoryStore::new()), "/array")?;
    let cached = ArrayCached::new(array, AsyncChunkCachePartialDecoderLruChunkLimit::new(4));
    let resources = Resources::default();
    let subset = cached.array().subset_all();

    for _ in 0..2 {
        let mut chunk = [0.0; 2];
        cached
            .async_retrieve_chunk_into(&[0], &mut output_target(&mut chunk, &[2])?, &resources)
            .await?;
        assert_eq!(chunk, [FILL_VALUE; 2]);
        assert_eq!(
            cached
                .async_retrieve_chunk::<Vec<f32>>(&[0], &resources)
                .await?,
            chunk
        );

        assert_eq!(
            cached
                .async_retrieve_array_subset::<Vec<f32>>(&subset, &resources)
                .await?,
            [FILL_VALUE; 4]
        );
    }
    Ok(())
}
