#![allow(missing_docs)]
#![cfg(all(feature = "sharding", feature = "zstd"))]

use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    sync::Arc,
};

use zarrs::{
    array::{
        ArrayBuilder, ArrayBytes, ArrayMetadataOptions,
        codec::{
            ArrayToBytesCodecTraits, VlenCodecConfiguration, ZstdCodec,
            array_to_bytes::{vlen::VlenCodec, vlen_utf8::VlenUtf8Codec},
        },
        data_type,
    },
    storage::{ReadableWritableListableStorage, store::MemoryStore},
};
use zarrs_filesystem::FilesystemStore;

fn read_cities() -> std::io::Result<Vec<String>> {
    let reader = BufReader::new(File::open("tests/data/cities.csv")?);
    let mut cities = Vec::with_capacity(47868);
    for line in reader.lines() {
        cities.push(line?);
    }
    Ok(cities)
}

fn cities_impl(
    cities: &[String],
    compression_level: Option<i32>,
    chunk_size: u64,
    shard_size: Option<u64>,
    vlen_codec: Arc<dyn ArrayToBytesCodecTraits>,
    write_to_file: bool,
) -> Result<u64, Box<dyn Error>> {
    let store: ReadableWritableListableStorage = if write_to_file {
        Arc::new(FilesystemStore::new("tests/data/v3/cities.zarr")?)
    } else {
        Arc::new(MemoryStore::default())
    };
    store.erase_prefix(&"".try_into().unwrap())?;

    let mut builder = ArrayBuilder::new(
        vec![cities.len() as u64], // array shape
        vec![chunk_size],          // regular chunk shape
        data_type::string(),
        "",
    );
    builder.array_to_bytes_codec(vlen_codec);
    if let Some(shard_size) = shard_size {
        builder.subchunk_shape(vec![shard_size]);
    }
    if let Some(compression_level) = compression_level {
        builder.bytes_to_bytes_codecs(vec![
            #[cfg(feature = "zstd")]
            Arc::new(ZstdCodec::new(compression_level, false)),
        ]);
    }

    let array = builder.build(store.clone(), "/")?;
    array
        .store_metadata_opt(&ArrayMetadataOptions::default().with_include_zarrs_metadata(false))?;

    let subset_all = array.subset_all();
    array.store_array_subset(&subset_all, cities)?;
    let cities_out = array.retrieve_array_subset::<Vec<String>>(&subset_all)?;
    assert_eq!(cities, cities_out);

    let last_block: ArrayBytes =
        array.retrieve_chunk(&[(cities.len() as u64).div_ceil(chunk_size)])?;
    let variable_length_bytes = last_block.into_variable()?;
    assert_eq!(variable_length_bytes.offsets().len() as u64, chunk_size + 1);

    Ok(store.size_prefix(&"c/".try_into().unwrap())?) // only chunks
}

#[rustfmt::skip]
#[test]
#[cfg_attr(miri, ignore)]
fn cities() -> Result<(), Box<dyn Error>> {
    let cities = read_cities()?;
    assert_eq!(cities.len(), 47868);
    assert_eq!(cities[0], "Tokyo");
    assert_eq!(cities[47862], "Sariwŏn-si");
    assert_eq!(cities[47867], "Charlotte Amalie");

    let vlen_utf8 = Arc::new(VlenUtf8Codec::new());

    // let vlen = Arc::new(VlenCodec::default());
    let vlen_configuration: VlenCodecConfiguration = serde_json::from_str(r#"{
        "data_codecs": [{"name": "bytes"}],
        "index_codecs": [{"name": "bytes","configuration": { "endian": "little" }}],
        "index_data_type": "uint32",
        "index_location": "start"
    }"#)?;
    let vlen = Arc::new(VlenCodec::new_with_configuration(&vlen_configuration)?);

    let vlen_compressed_configuration: VlenCodecConfiguration = serde_json::from_str(r#"{
        "data_codecs": [{"name": "bytes"},{"name": "blosc","configuration": {"cname": "zstd", "clevel":5,"shuffle": "bitshuffle", "typesize":1,"blocksize":0}}],
        "index_codecs": [{"name": "bytes","configuration": { "endian": "little" }},{"name": "blosc","configuration":{"cname": "zstd", "clevel":5,"shuffle": "shuffle", "typesize":4,"blocksize":0}}],
        "index_data_type": "uint32",
        "index_location": "end"
    }"#)?;
    let vlen_compressed = Arc::new(VlenCodec::new_with_configuration(&vlen_compressed_configuration)?);

    println!("| encoding         | compression | size   |");
    println!("| ---------------- | ----------- | ------ |");
    println!("| vlen_utf8 |             | {} |", cities_impl(&cities, None, 1000, None, vlen_utf8.clone(), true)?);
    println!("| vlen_utf8 | zstd 5      | {} |", cities_impl(&cities, Some(5), 1000, None, vlen_utf8.clone(), false)?);
    println!("| vlen             |             | {} |", cities_impl(&cities, None, 1000, None, vlen.clone(), false)?);
    println!("| vlen             | zstd 5      | {} |", cities_impl(&cities, None, 1000, None, vlen_compressed.clone(), false)?);
    println!();
    // panic!();

    // | encoding         | compression | size   |
    // | ---------------- | ----------- | ------ |
    // | vlen_utf8 |             | 642196 |
    // | vlen_utf8 | zstd 5      | 362626 |
    // | vlen             |             | 642580 |
    // | vlen             | zstd 5      | 346950 |

    Ok(())
}

#[test]
fn cities_zarr_python_v2_compat() -> Result<(), Box<dyn Error>> {
    let store = Arc::new(FilesystemStore::new(
        "tests/data/zarr_python_compat/cities_v2.zarr",
    )?);
    let array = zarrs::array::Array::open(store.clone(), "/")?;
    let subset_all = array.subset_all();
    let cities_out = array.retrieve_array_subset::<Vec<String>>(&subset_all)?;

    let cities = read_cities()?;
    assert_eq!(cities, cities_out);

    Ok(())
}

#[test]
fn cities_zarr_python_v3_compat() -> Result<(), Box<dyn Error>> {
    let store = Arc::new(FilesystemStore::new(
        "tests/data/zarr_python_compat/cities_v3.zarr",
    )?);
    let array = zarrs::array::Array::open(store.clone(), "/")?;
    let subset_all = array.subset_all();
    let cities_out = array.retrieve_array_subset::<Vec<String>>(&subset_all)?;

    let cities = read_cities()?;
    assert_eq!(cities, cities_out);

    Ok(())
}
