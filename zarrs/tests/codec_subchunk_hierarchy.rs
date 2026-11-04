#![allow(missing_docs)]

use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::chunk_cache::ChunkCacheDecodedLruChunkLimit;
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::codec::{CodecChainBound, TransposeCodec, TransposeOrder};
use zarrs::array::{ArrayBuilder, ArrayCached, ArraySubset, CodecOptions, data_type};
use zarrs::storage::storage_adapter::performance_metrics::PerformanceMetricsStorageAdapter;
use zarrs::storage::store::MemoryStore;

fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}

#[test]
fn hierarchy_is_pure_ordered_and_shape_independent() -> Result<(), Box<dyn std::error::Error>> {
    let store = Arc::new(PerformanceMetricsStorageAdapter::new(Arc::new(
        MemoryStore::default(),
    )));
    let dt = data_type::uint8();
    let inner = ShardingCodecBuilder::new(vec![nz(2), nz(3)], &dt).build_arc();
    let mut outer = ShardingCodecBuilder::new(vec![nz(4), nz(6)], &dt);
    outer.array_to_bytes_codec(inner);
    let mut builder = ArrayBuilder::new(vec![12, 8], vec![12, 8], dt, 17u8);
    builder
        .array_to_array_codecs(vec![Arc::new(TransposeCodec::new(TransposeOrder::new(
            &[1, 0],
        )?))])
        .array_to_bytes_codec(outer.build_arc());
    let array = builder.build_arc(store.clone(), "/array")?;
    let cached = ArrayCached::new(array.clone(), ChunkCacheDecodedLruChunkLimit::new(1));
    let codecs = array.subchunk_codecs();
    assert_eq!(codecs.len(), array.subchunk_grids().len());
    assert_eq!(codecs.len(), 2);
    assert_eq!(codecs[0].subchunk_codecs().len(), 1);
    assert!(codecs[1].subchunk_codecs().is_empty());
    for (level, codec) in codecs.iter().enumerate() {
        let at_level = array.subchunk_codecs_at_level(level).unwrap();
        let cached_codec = cached.subchunk_codecs_at_level(level).unwrap();
        assert!(Arc::ptr_eq(codec, &at_level));
        let chain = at_level.as_any().downcast_ref::<CodecChainBound>().unwrap();
        assert!(chain.array_to_array_codecs().is_empty());
        assert!(Arc::ptr_eq(codec, &cached_codec));
        assert_eq!(at_level.data_type(), &data_type::uint8());
        assert_eq!(at_level.fill_value().unwrap().as_ne_bytes(), &[17]);
        // Include clipped/varying shapes. The outer transpose must not enter these codecs.
        for shape in [vec![nz(2), nz(3)], vec![nz(1), nz(2)], vec![nz(3), nz(1)]] {
            let bytes: Vec<u8> =
                (0..u8::try_from(shape.iter().map(|x| x.get()).product::<u64>())?).collect();
            let encoded =
                at_level.encode(bytes.clone().into(), &shape, &CodecOptions::default())?;
            assert_eq!(
                at_level
                    .decode(encoded, &shape, &CodecOptions::default())?
                    .into_fixed()?
                    .as_ref(),
                bytes
            );
        }
    }
    // A leaf subchunk is raw uint8 bytes in encoded-domain order. Decoding it must not
    // apply the outer transpose, even though the outward grid has its axes reversed.
    assert_eq!(
        codecs[1]
            .decode(
                vec![0, 1, 2, 3, 4, 5].into(),
                &[nz(2), nz(3)],
                &CodecOptions::default()
            )?
            .into_fixed()?
            .as_ref(),
        &[0, 1, 2, 3, 4, 5]
    );
    assert!(array.subchunk_codecs_at_level(2).is_none());
    assert!(cached.subchunk_codecs_at_level(2).is_none());
    assert_eq!(store.reads(), 0);
    let plain = ArrayBuilder::new(vec![4], vec![4], data_type::uint8(), 0u8)
        .build(store.clone(), "/plain")?;
    assert!(plain.subchunk_codecs().is_empty());
    assert!(plain.subchunk_codecs_at_level(0).is_none());
    assert_eq!(store.reads(), 0);
    Ok(())
}

#[test]
fn nested_codecs_decode_into_inner_chain_domain() -> Result<(), Box<dyn std::error::Error>> {
    let dt = data_type::uint8();
    // Level 0 subchunks are [4, 6]. They are transposed to [6, 4] by the inner chain before
    // the inner sharding codec splits them into [3, 2] level 1 subchunks.
    let inner = ShardingCodecBuilder::new(vec![nz(3), nz(2)], &dt).build_arc();
    let mut outer = ShardingCodecBuilder::new(vec![nz(4), nz(6)], &dt);
    outer
        .array_to_array_codecs(vec![Arc::new(TransposeCodec::new(TransposeOrder::new(
            &[1, 0],
        )?))])
        .array_to_bytes_codec(inner);
    let mut builder = ArrayBuilder::new(vec![4, 6], vec![4, 6], dt, 0u8);
    builder.array_to_bytes_codec(outer.build_arc());
    let array = builder.build_arc(Arc::new(MemoryStore::default()), "/array")?;

    // The level 1 grid is in the array domain: [3, 2] subchunks transposed back to [2, 3]
    let level1 = array.subchunk_grid_at_level(1).as_chunk_grid().unwrap();
    assert_eq!(
        level1.subset(&[0, 0])?,
        Some(ArraySubset::new_with_ranges(&[0..2, 0..3]))
    );

    let codecs = array.subchunk_codecs();
    assert_eq!(codecs.len(), 2);
    let level0 = codecs[0]
        .as_any()
        .downcast_ref::<CodecChainBound>()
        .unwrap();
    assert_eq!(level0.array_to_array_codecs().len(), 1);

    // Encode a level 0 subchunk and extract the bytes of its first level 1 subchunk via the
    // inner shard index (4 subchunks of (offset, nbytes) u64 pairs, then a crc32c checksum)
    let values: Vec<u8> = (0..24).collect();
    let shard = codecs[0].encode(values.into(), &[nz(4), nz(6)], &CodecOptions::default())?;
    let index_start = shard.len() - 4 - 4 * 16;
    let entry = |i: usize| {
        u64::from_le_bytes(
            shard[index_start + i * 8..index_start + i * 8 + 8]
                .try_into()
                .unwrap(),
        )
    };
    let (offset, nbytes) = (usize::try_from(entry(0))?, usize::try_from(entry(1))?);
    let encoded_subchunk = shard[offset..offset + nbytes].to_vec();

    // The level 1 codec decodes it in the transposed domain of the inner chain: [3, 2] rather
    // than the array-domain [2, 3], with elements in transposed order.
    let decoded = codecs[1]
        .decode(
            encoded_subchunk.into(),
            &[nz(3), nz(2)],
            &CodecOptions::default(),
        )?
        .into_fixed()?;
    assert_eq!(decoded.as_ref(), &[0, 6, 1, 7, 2, 8]);
    Ok(())
}
