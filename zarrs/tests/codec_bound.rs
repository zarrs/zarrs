#![allow(missing_docs)]

use std::sync::Arc;
use zarrs::array::Resources;

use zarrs::array::codec::BytesCodec;
use zarrs::array::{
    ArrayBuilder, ArrayBytes, ArrayCodecTraits, ArrayCreateError, ArrayToBytesCodecTraits,
    CodecOptions, FillValue, data_type,
};
use zarrs::storage::store::MemoryStore;

#[test]
fn codec_chain_bound_context_and_runtime() -> Result<(), Box<dyn std::error::Error>> {
    let array = ArrayBuilder::new(vec![4], vec![4], data_type::uint8(), 7u8)
        .build(Arc::new(MemoryStore::new()), "/array")?;
    let codecs = array.codecs_bound();

    assert_eq!(codecs.data_type(), array.data_type());
    assert_eq!(codecs.fill_value(), array.fill_value());

    let shape = array.chunk_shape(&[0])?;
    let decoded = ArrayBytes::from(vec![1, 2, 3, 4]);
    let encoded = codecs.encode(
        decoded.clone(),
        &shape,
        &CodecOptions::default(),
        &Resources::default(),
    )?;
    assert_eq!(
        codecs.decode(
            encoded,
            &shape,
            &CodecOptions::default(),
            &Resources::default()
        )?,
        decoded
    );
    codecs.encoded_representation(&shape)?;
    codecs.recommended_concurrency(&shape)?;

    Ok(())
}

#[test]
fn codec_binding_error_propagates_from_array_builder() {
    let mut builder = ArrayBuilder::new(
        vec![4, 4],
        vec![4, 4],
        data_type::uint8().to_optional(),
        FillValue::from(None::<u8>),
    );
    // The `bytes` codec does not support optional data types
    builder.array_to_bytes_codec(Arc::new(BytesCodec::default()));

    assert!(matches!(
        builder.build(Arc::new(MemoryStore::new()), "/array"),
        Err(ArrayCreateError::CodecsCreateError(_))
    ));
}
