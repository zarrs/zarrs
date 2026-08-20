#![allow(missing_docs)]

use std::error::Error;
use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::codec::{BytesCodec, GzipCodec, PackBitsCodec, ShardingCodecBuilder};
use zarrs::array::{
    ArrayBuilder, ArrayError, ElementError, ElementLayout, ElementPacking, Tensor, data_type,
};
use zarrs::storage::store::MemoryStore;
use zarrs_metadata_ext::codec::packbits::PackBitsPaddingEncoding;

/// int4 elements -1..=6, which are stored sign extended one per byte in memory.
const INT4_ELEMENTS: [i8; 8] = [-1, 7, -8, 1, 0, -2, 3, 6];

fn packbits_array(
    padding_encoding: PackBitsPaddingEncoding,
) -> Result<zarrs::array::Array<MemoryStore>, Box<dyn Error>> {
    let mut builder = ArrayBuilder::new(vec![8], vec![8], data_type::int4(), 0i8);
    builder.array_to_bytes_codec(Arc::new(PackBitsCodec::new(padding_encoding, None, None)?));
    let array = builder.build(MemoryStore::new().into(), "/")?;
    array.store_chunk(&[0], INT4_ELEMENTS.to_vec())?;
    Ok(array)
}

#[test]
fn stored_layout_packbits_is_zero_copy() -> Result<(), Box<dyn Error>> {
    let array = packbits_array(PackBitsPaddingEncoding::None)?;

    let stored = array.retrieve_chunk_stored_layout(&[0])?.unwrap();
    assert_eq!(stored.layout().packing, ElementPacking::PackedLsb0);
    assert_eq!(stored.layout().byte_offset, 0);
    assert_eq!(stored.shape(), &[8]);
    assert_eq!(stored.data_type(), &data_type::int4());

    // Eight 4-bit elements pack into four bytes, half of the padded form
    assert_eq!(stored.bytes().len(), 4);

    // The zero-copy and pack-on-demand paths must agree
    let decoded: Tensor = array.retrieve_chunk(&[0])?;
    assert_eq!(decoded.bytes().len(), 8);
    assert_eq!(decoded.into_packed()?.bytes(), stored.bytes());

    Ok(())
}

#[test]
fn stored_layout_packbits_padding_encoding() -> Result<(), Box<dyn Error>> {
    let reference = packbits_array(PackBitsPaddingEncoding::None)?
        .retrieve_chunk_stored_layout(&[0])?
        .unwrap();

    // The padding bit count is prepended, so the elements start one byte in
    let first = packbits_array(PackBitsPaddingEncoding::FirstByte)?
        .retrieve_chunk_stored_layout(&[0])?
        .unwrap();
    assert_eq!(first.layout().byte_offset, 1);
    assert_eq!(&first.bytes()[1..], reference.bytes());

    // The padding bit count is appended, past the last element
    let last = packbits_array(PackBitsPaddingEncoding::LastByte)?
        .retrieve_chunk_stored_layout(&[0])?
        .unwrap();
    assert_eq!(last.layout().byte_offset, 0);
    assert_eq!(&last.bytes()[..4], reference.bytes());

    Ok(())
}

#[test]
fn stored_layout_decodes_bytes_to_bytes_codecs() -> Result<(), Box<dyn Error>> {
    // A compressed chunk must still yield the packed buffer
    let mut builder = ArrayBuilder::new(vec![8], vec![8], data_type::int4(), 0i8);
    builder
        .array_to_bytes_codec(Arc::new(PackBitsCodec::default()))
        .bytes_to_bytes_codecs(vec![Arc::new(GzipCodec::new(5)?)]);
    let array = builder.build(MemoryStore::new().into(), "/")?;
    array.store_chunk(&[0], INT4_ELEMENTS.to_vec())?;

    let stored = array.retrieve_chunk_stored_layout(&[0])?.unwrap();
    assert_eq!(stored.layout().packing, ElementPacking::PackedLsb0);
    assert_eq!(stored.bytes().len(), 4);

    // The gzipped chunk is not the packed buffer
    let encoded = array.retrieve_encoded_chunk(&[0])?.unwrap();
    assert_ne!(encoded, stored.bytes());

    Ok(())
}

#[test]
fn stored_layout_bytes_codec_is_padded() -> Result<(), Box<dyn Error>> {
    let mut builder = ArrayBuilder::new(vec![8], vec![8], data_type::int4(), 0i8);
    builder.array_to_bytes_codec(Arc::new(BytesCodec::new(None)));
    let array = builder.build(MemoryStore::new().into(), "/")?;
    array.store_chunk(&[0], INT4_ELEMENTS.to_vec())?;

    // The `bytes` codec stores sub-byte elements padded to one byte each
    let stored = array.retrieve_chunk_stored_layout(&[0])?.unwrap();
    assert_eq!(stored.layout(), ElementLayout::default());
    assert_eq!(stored.bytes().len(), 8);

    Ok(())
}

#[test]
fn stored_layout_missing_chunk_is_none() -> Result<(), Box<dyn Error>> {
    let mut builder = ArrayBuilder::new(vec![8], vec![8], data_type::int4(), 0i8);
    builder.array_to_bytes_codec(Arc::new(PackBitsCodec::default()));
    let array = builder.build(MemoryStore::new().into(), "/")?;

    assert!(array.retrieve_chunk_stored_layout(&[0])?.is_none());

    Ok(())
}

#[cfg(feature = "dlpack")]
#[test]
fn stored_layout_dlpack_export_is_packed() -> Result<(), Box<dyn Error>> {
    use dlpark::{Builder, DlpackFlags, legacy, versioned};

    let array = packbits_array(PackBitsPaddingEncoding::None)?;
    let stored = array.retrieve_chunk_stored_layout(&[0])?.unwrap();
    let packed_bytes = stored.bytes().to_vec();

    let dlpack: versioned::Dlpack = Builder::try_from(Box::new(stored))?.try_build()?;
    // A packed tensor is what DLPack assumes, so no flag is needed and `num_bytes` agrees
    assert_eq!(dlpack.flags(), DlpackFlags::empty());
    assert_eq!(dlpack.num_bytes()?, 4);
    assert_eq!(dlpack.shape()?, &[8]);
    drop(dlpack);

    // With no flag to lose, the legacy ABI is also safe for a sub-byte data type
    let stored = array.retrieve_chunk_stored_layout(&[0])?.unwrap();
    let dlpack: legacy::Dlpack = Builder::try_from(Box::new(stored))?.try_build()?;
    assert_eq!(dlpack.num_bytes()?, 4);
    drop(dlpack);

    // The decoded tensor is padded instead, and is twice the size
    let decoded: Tensor = array.retrieve_chunk(&[0])?;
    let dlpack: versioned::Dlpack = Builder::try_from(Box::new(decoded))?.try_build()?;
    assert_eq!(dlpack.flags(), DlpackFlags::IS_SUBBYTE_TYPE_PADDED);
    assert_eq!(packed_bytes.len() * 2, 8);

    Ok(())
}

#[test]
fn stored_layout_cannot_be_stored() -> Result<(), Box<dyn Error>> {
    let array = packbits_array(PackBitsPaddingEncoding::None)?;
    let stored = array.retrieve_chunk_stored_layout(&[0])?.unwrap();

    // Array bytes are always in the default layout, and storing does not convert, so a packed
    // tensor must be rejected rather than reinterpreted as padded
    assert!(matches!(
        array.store_chunk(&[0], &stored),
        Err(ArrayError::ElementError(
            ElementError::IncompatibleElementLayout
        ))
    ));
    assert!(matches!(
        array.store_chunk(&[0], stored),
        Err(ArrayError::ElementError(
            ElementError::IncompatibleElementLayout
        ))
    ));

    // A decoded tensor stores as usual, and the array is unchanged by the failures above
    let decoded: Tensor = array.retrieve_chunk(&[0])?;
    array.store_chunk(&[0], decoded)?;
    assert_eq!(
        array.retrieve_chunk::<Vec<i8>>(&[0])?,
        INT4_ELEMENTS.to_vec()
    );

    Ok(())
}

#[test]
fn stored_layout_unsupported_codecs() -> Result<(), Box<dyn Error>> {
    // Sharding does not declare an element layout, and is rejected without special casing
    let mut sharding =
        ShardingCodecBuilder::new(vec![NonZeroU64::new(4).unwrap()], &data_type::int4());
    sharding.array_to_bytes_codec(Arc::new(PackBitsCodec::default()));
    let mut builder = ArrayBuilder::new(vec![8], vec![8], data_type::int4(), 0i8);
    builder.array_to_bytes_codec(sharding.build_arc());
    let array = builder.build(MemoryStore::new().into(), "/")?;
    array.store_chunk(&[0], INT4_ELEMENTS.to_vec())?;
    assert!(matches!(
        array.retrieve_chunk_stored_layout(&[0]),
        Err(ArrayError::NoStoredLayout)
    ));

    // A restricted `packbits` bit range does not encode whole elements
    let mut builder = ArrayBuilder::new(vec![8], vec![8], data_type::int4(), 0i8);
    builder.array_to_bytes_codec(Arc::new(PackBitsCodec::new(
        PackBitsPaddingEncoding::None,
        Some(1),
        None,
    )?));
    let array = builder.build(MemoryStore::new().into(), "/")?;
    array.store_chunk(&[0], INT4_ELEMENTS.to_vec())?;
    assert!(matches!(
        array.retrieve_chunk_stored_layout(&[0]),
        Err(ArrayError::NoStoredLayout)
    ));

    Ok(())
}
