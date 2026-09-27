#![allow(missing_docs)]

use std::error::Error;
use std::sync::Arc;

use itertools::Itertools;
use zarrs::array::chunk_key_encoding::api::{
    ChunkKeyEncoding, ChunkKeyEncodingError, ChunkKeyEncodingPlugin, ChunkKeyEncodingTraits,
};
use zarrs::array::{Array, ArrayBuilder, ArrayError, ArrayShape, data_type};
use zarrs::metadata::Configuration;
use zarrs::metadata::v3::MetadataV3;
use zarrs::plugin::PluginCreateError;
use zarrs::storage::store::MemoryStore;

type TestResult = Result<(), Box<dyn Error>>;

/// The dimensionality that [`ThreeDimensionalChunkKeyEncoding`] can encode.
const DIMENSIONALITY: usize = 3;

zarrs::plugin::impl_extension_aliases!(ThreeDimensionalChunkKeyEncoding,
    v3: "zarrs.test_three_dimensional", []
);

inventory::submit! {
    ChunkKeyEncodingPlugin::new::<ThreeDimensionalChunkKeyEncoding>()
}

/// A chunk key encoding that can only encode 3 dimensional chunk grid indices.
///
/// This stands in for the proposed `generic` chunk key encoding, whose format string may
/// reference a dimension that the chunk grid does not have. That mismatch is only detectable
/// when a chunk key is encoded, which is why [`ChunkKeyEncodingTraits::encode`] is fallible.
#[derive(Debug)]
struct ThreeDimensionalChunkKeyEncoding;

impl ChunkKeyEncodingTraits for ThreeDimensionalChunkKeyEncoding {
    fn create(_metadata: &MetadataV3) -> Result<ChunkKeyEncoding, PluginCreateError> {
        Ok(Self.into())
    }

    fn configuration(&self) -> Configuration {
        Configuration::default()
    }

    fn encode(&self, chunk_grid_indices: &[u64]) -> Result<String, ChunkKeyEncodingError> {
        if chunk_grid_indices.len() == DIMENSIONALITY {
            Ok(chunk_grid_indices.iter().join("/"))
        } else {
            Err(ChunkKeyEncodingError::IncompatibleDimensionality(
                chunk_grid_indices.len(),
                format!("only {DIMENSIONALITY} dimensional chunk grid indices are supported"),
            ))
        }
    }
}

fn array(shape: ArrayShape, chunk_shape: ArrayShape) -> Result<Array<MemoryStore>, Box<dyn Error>> {
    let mut builder = ArrayBuilder::new(shape, chunk_shape, data_type::uint8(), 0u8);
    builder.chunk_key_encoding(ThreeDimensionalChunkKeyEncoding);
    Ok(builder.build(Arc::new(MemoryStore::default()), "/array")?)
}

fn array_3d() -> Result<Array<MemoryStore>, Box<dyn Error>> {
    array(vec![4, 4, 4], vec![2, 2, 2])
}

/// An array whose dimensionality the chunk key encoding cannot handle.
fn array_2d() -> Result<Array<MemoryStore>, Box<dyn Error>> {
    array(vec![4, 4], vec![2, 2])
}

#[track_caller]
fn assert_encoding_error(error: ArrayError) {
    assert!(
        matches!(error, ArrayError::ChunkKeyEncodingError(_)),
        "expected a chunk key encoding error, got {error:?}"
    );
}

#[test]
fn chunk_key_surfaces_an_encoding_error() -> TestResult {
    assert_eq!(array_3d()?.chunk_key(&[0, 1, 2])?.as_str(), "array/0/1/2");
    assert_encoding_error(array_2d()?.chunk_key(&[0, 0]).unwrap_err());
    Ok(())
}

#[test]
fn array_ops_surface_an_encoding_error_rather_than_panicking() -> TestResult {
    let array = array_2d()?;
    assert_encoding_error(array.store_chunk(&[0, 0], vec![0u8; 4]).unwrap_err());
    assert_encoding_error(array.retrieve_chunk::<Vec<u8>>(&[0, 0]).unwrap_err());
    assert_encoding_error(array.retrieve_encoded_chunk(&[0, 0]).unwrap_err());
    assert_encoding_error(array.erase_chunk(&[0, 0]).unwrap_err());
    Ok(())
}

#[test]
fn a_supported_dimensionality_still_round_trips() -> TestResult {
    let array = array_3d()?;
    let chunk = vec![4u8, 5, 6, 7, 8, 9, 10, 11];
    array.store_chunk(&[1, 1, 1], chunk.clone())?;
    assert_eq!(array.retrieve_chunk::<Vec<u8>>(&[1, 1, 1])?, chunk);
    assert!(array.retrieve_encoded_chunk(&[1, 1, 1])?.is_some());
    array.erase_chunk(&[1, 1, 1])?;
    assert!(array.retrieve_encoded_chunk(&[1, 1, 1])?.is_none());
    Ok(())
}
