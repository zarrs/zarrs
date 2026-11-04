#![allow(missing_docs)]
//! Encoded subchunks are addressed in time linear in the depth of the codecs.
//!
//! An array addresses a subchunk by its subset of a chunk, which each array-to-array codec maps with two coordinate
//! mappings and each sharding level locates in its own subchunk grid, so no subchunk grid hierarchy is resolved. An
//! index-addressed partial decoder resolves the hierarchy once, at the top.

mod identity_codec;

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use identity_codec::IdentityCodec;
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::{Array, ArrayBuilder, CodecOptions, data_type};
use zarrs::storage::store::MemoryStore;

type TestResult = Result<(), Box<dyn Error>>;

/// The number of identity codecs above the outer sharding codec, and inside it above the middle sharding codec.
const CODECS: usize = 4;

/// The number of subchunk grid levels: 8x8, 4x4 and 2x2 subchunks of a 16x16 chunk.
const LEVELS: usize = 3;

fn nz(n: u64) -> NonZeroU64 {
    NonZeroU64::new(n).unwrap()
}

/// An array, and the identity codecs above its outer and middle sharding codecs.
type DeepArray = (
    Array<MemoryStore>,
    Vec<Arc<IdentityCodec>>,
    Vec<Arc<IdentityCodec>>,
);

/// A 16x16 single chunk array of three nested shards, with [`CODECS`] identity codecs above the outer and the middle
/// sharding codecs.
fn deep_array() -> Result<DeepArray, Box<dyn Error>> {
    let identity_codecs = || -> Vec<Arc<IdentityCodec>> {
        (0..CODECS)
            .map(|_| Arc::new(IdentityCodec::new(true)))
            .collect()
    };
    let (outer_codecs, inner_codecs) = (identity_codecs(), identity_codecs());
    let data_type = data_type::uint16();
    let inner = ShardingCodecBuilder::new(vec![nz(2), nz(2)], &data_type).build_arc();
    let mut middle = ShardingCodecBuilder::new(vec![nz(4), nz(4)], &data_type);
    middle.array_to_bytes_codec(inner);
    let mut outer = ShardingCodecBuilder::new(vec![nz(8), nz(8)], &data_type);
    outer
        .array_to_array_codecs(
            inner_codecs
                .iter()
                .map(|codec| codec.clone() as _)
                .collect(),
        )
        .array_to_bytes_codec(middle.build_arc());
    let mut builder = ArrayBuilder::new(vec![16, 16], vec![16, 16], data_type, 0u16);
    builder
        .array_to_array_codecs(
            outer_codecs
                .iter()
                .map(|codec| codec.clone() as _)
                .collect(),
        )
        .array_to_bytes_codec(outer.build_arc());
    let array = builder.build(Arc::new(MemoryStore::default()), "/array")?;
    array.store_chunk(&[0, 0], (0..256u16).collect::<Vec<_>>())?;
    Ok((array, outer_codecs, inner_codecs))
}

/// Return and reset the total counts of calls to `decoded_local_subchunk_grid` and `encoded_coordinates` of `codecs`.
fn take_calls(codecs: &[Arc<IdentityCodec>]) -> (usize, usize) {
    codecs.iter().fold((0, 0), |(grids, coordinates), codec| {
        let (codec_grids, codec_coordinates) = codec.calls.take();
        (grids + codec_grids, coordinates + codec_coordinates)
    })
}

/// The identity codecs above the subchunk at `level`, which map its subset.
fn codecs_above(level: usize) -> usize {
    if level == 0 { CODECS } else { 2 * CODECS }
}

#[test]
fn array_encoded_subchunk_addressing_is_linear() -> TestResult {
    let (array, outer_codecs, inner_codecs) = deep_array()?;
    let codecs = [outer_codecs, inner_codecs].concat();
    take_calls(&codecs);

    for (level, indices) in [(0, [1, 0]), (1, [0, 3]), (2, [2, 1])] {
        assert!(
            array
                .retrieve_encoded_subchunk_at_level(level, &indices)?
                .is_some()
        );
        // Each identity codec above the subchunk maps the first and last elements of its subset, and no subchunk grid
        // hierarchy is resolved
        assert_eq!(take_calls(&codecs), (0, 2 * codecs_above(level)));

        let encoded = array
            .retrieve_encoded_subchunk_at_level(level, &indices)?
            .expect("the subchunk is stored");
        take_calls(&codecs);
        // SAFETY: The subchunk is stored back where it was retrieved from.
        unsafe {
            array.store_encoded_subchunk_at_level(level, &indices, encoded.into_bytes().into())?;
        }
        assert_eq!(take_calls(&codecs), (0, 2 * codecs_above(level)));
        array.erase_subchunk_at_level(level, &indices)?;
        assert_eq!(take_calls(&codecs), (0, 2 * codecs_above(level)));
    }
    Ok(())
}

#[test]
fn partial_decoder_encoded_subchunk_addressing_is_linear() -> TestResult {
    let (array, outer_codecs, inner_codecs) = deep_array()?;
    let options = CodecOptions::default();
    let partial_decoder = array.partial_decoder(&[0, 0])?;
    take_calls(&outer_codecs);
    take_calls(&inner_codecs);

    for (level, indices) in [(0, [1, 0]), (1, [0, 3]), (2, [2, 1])] {
        assert!(
            partial_decoder
                .retrieve_encoded_subchunk_at_level(level, &indices, &options)?
                .is_some()
        );
        // Addressing by indices resolves the subchunk grid hierarchy once, through each outer identity codec, which
        // then forwards the subchunk by subset. The inner identity codecs only map subsets.
        let (outer_grids, outer_coordinates) = take_calls(&outer_codecs);
        assert_eq!(outer_grids, CODECS * LEVELS);
        assert_eq!(outer_coordinates, 2 * CODECS);
        assert_eq!(
            take_calls(&inner_codecs),
            (0, 2 * (codecs_above(level) - CODECS))
        );
    }
    Ok(())
}
