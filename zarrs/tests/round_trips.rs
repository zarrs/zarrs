#![allow(missing_docs)]

use std::{error::Error, path::Path};

use zarrs::{
    array::{
        ChunkKeyEncoding,
        chunk_key_encoding::{DefaultChunkKeyEncoding, V2ChunkKeyEncoding},
    },
    node::{data_key, meta_key_v3},
    storage::{ReadableStorageTraits, WritableStorageTraits, store::MemoryStore},
};
use zarrs_filesystem::FilesystemStore;

pub fn to_json(data: &[u8]) -> serde_json::Value {
    let data = std::str::from_utf8(data).unwrap();
    serde_json::from_str(data).unwrap()
}

#[test]
fn metadata_round_trip_memory() -> Result<(), Box<dyn Error>> {
    let store = MemoryStore::new();
    let metadata_in = include_bytes!("data/array_metadata.json");
    store.set(
        &meta_key_v3(&"/group/array".try_into()?),
        metadata_in.to_vec().into(),
    )?;
    let metadata_out = store
        .get(&meta_key_v3(&"/group/array".try_into()?))?
        .unwrap();
    assert_eq!(metadata_in.as_slice(), metadata_out);
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn metadata_round_trip_filesystem() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let store = FilesystemStore::new(path.path())?;
    let metadata_in = include_bytes!("data/array_metadata.json");
    store.set(
        &meta_key_v3(&"/group/array".try_into()?),
        metadata_in.to_vec().into(),
    )?;
    let metadata_out = store
        .get(&meta_key_v3(&"/group/array".try_into()?))?
        .unwrap();
    assert_eq!(metadata_in.as_slice(), metadata_out);
    Ok(())
}

fn filesystem_chunk_round_trip_impl(
    path: &Path,
    chunk_key_encoding: &ChunkKeyEncoding,
) -> Result<(), Box<dyn Error>> {
    let store = FilesystemStore::new(path)?;
    let data_serialised_in: Vec<u8> = vec![0, 1, 2];
    store.set(
        &data_key(
            &"/group/array".try_into()?,
            &chunk_key_encoding.encode(&[0, 0, 0]),
        ),
        data_serialised_in.clone().into(),
    )?;
    let data_serialised_out = store
        .get(&data_key(
            &"/group/array".try_into()?,
            &chunk_key_encoding.encode(&[0, 0, 0]),
        ))?
        .unwrap()
        .to_vec();
    assert_eq!(data_serialised_in, data_serialised_out);
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn chunk_round_trip_filesystem_key_encoding_default_slash() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let chunk_key_encoding = ChunkKeyEncoding::new(DefaultChunkKeyEncoding::default());
    filesystem_chunk_round_trip_impl(path.path(), &chunk_key_encoding)?;
    let mut path_expect = path.path().to_owned();
    path_expect.push("group/array/c/0/0/0");
    assert!(path_expect.is_file());
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn chunk_round_trip_filesystem_key_encoding_default_dot() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let chunk_key_encoding = ChunkKeyEncoding::new(DefaultChunkKeyEncoding::new_dot());
    filesystem_chunk_round_trip_impl(path.path(), &chunk_key_encoding)?;
    let mut path_expect = path.path().to_owned();
    path_expect.push("group/array/c.0.0.0");
    assert!(path_expect.is_file());
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn chunk_round_trip_filesystem_key_encoding_v2_dot() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let chunk_key_encoding = ChunkKeyEncoding::new(V2ChunkKeyEncoding::default());
    filesystem_chunk_round_trip_impl(path.path(), &chunk_key_encoding)?;
    let mut path_expect = path.path().to_owned();
    path_expect.push("group/array/0.0.0");
    assert!(path_expect.is_file());
    Ok(())
}

#[test]
#[cfg_attr(miri, ignore)]
fn chunk_round_trip_filesystem_key_encoding_v2_slash() -> Result<(), Box<dyn Error>> {
    let path = tempfile::TempDir::new()?;
    let chunk_key_encoding = ChunkKeyEncoding::new(V2ChunkKeyEncoding::new_slash());
    filesystem_chunk_round_trip_impl(path.path(), &chunk_key_encoding)?;
    let mut path_expect = path.path().to_owned();
    path_expect.push("group/array/0/0/0");
    assert!(path_expect.is_file());
    Ok(())
}
