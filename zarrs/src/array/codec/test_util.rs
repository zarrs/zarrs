//! Helpers for codec tests.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_codec::plan::partial_decode_memory;
use zarrs_codec::{
    ArrayToBytesCodecTraits, BytesToBytesCodecTraits, CodecError, CodecOptions,
    CodecSpecificOptions, CowBytes, Resources, UnboundArrayToArrayCodecTraits,
    UnboundArrayToBytesCodecTraits,
};
use zarrs_storage::byte_range::ByteRange;

use crate::array::codec::{BytesCodec, CodecChain};
use crate::array::{
    ArrayBytes, ArrayIndices, CodecChainBound, DataType, FillValue, Indexer, data_type,
};

/// Decode the byte `ranges` of the `decoded_len` decoded bytes of `encoded`, which is encoded with `codec`.
///
/// The bytes are decoded by a planned partial decode of a chain of `bytes` (`uint8`) and `codec`.
pub(crate) fn partial_decode_bytes(
    codec: Arc<dyn BytesToBytesCodecTraits>,
    encoded: &[u8],
    decoded_len: u64,
    ranges: &[ByteRange],
) -> Vec<Vec<u8>> {
    let chain = CodecChain::new(vec![], Arc::new(BytesCodec::new(None)), vec![codec])
        .with_context(
            data_type::uint8(),
            FillValue::from(0u8),
            &CodecSpecificOptions::default(),
        )
        .unwrap();
    let indices: Vec<ArrayIndices> = ranges
        .iter()
        .flat_map(|range| range.to_range(decoded_len).map(|i| vec![i]))
        .collect();
    let decoded = partial_decode_memory(
        chain,
        Some(encoded.into()),
        &[NonZeroU64::new(decoded_len).unwrap()],
        &indices,
        &CodecOptions::default(),
        &Resources::default(),
    )
    .unwrap()
    .into_fixed()
    .unwrap()
    .into_vec();
    let mut decoded = decoded.as_slice();
    ranges
        .iter()
        .map(|range| {
            let (head, tail) =
                decoded.split_at(usize::try_from(range.length(decoded_len)).unwrap());
            decoded = tail;
            head.to_vec()
        })
        .collect()
}

/// Bind a codec chain of `array_to_array`, `array_to_bytes` and `bytes_to_bytes` codecs.
pub(crate) fn chain(
    array_to_array: Vec<Arc<dyn UnboundArrayToArrayCodecTraits>>,
    array_to_bytes: Arc<dyn UnboundArrayToBytesCodecTraits>,
    bytes_to_bytes: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    data_type: DataType,
    fill_value: FillValue,
) -> Arc<CodecChainBound> {
    CodecChain::new(array_to_array, array_to_bytes, bytes_to_bytes)
        .with_context(data_type, fill_value, &CodecSpecificOptions::default())
        .unwrap()
}

/// Planned partial decodes of an encoded chunk in memory.
pub(crate) struct MemoryPartialDecoder {
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    encoded: Option<Vec<u8>>,
    shape: Vec<NonZeroU64>,
}

impl MemoryPartialDecoder {
    /// A partial decoder of the chunk of `shape` `encoded` with `codec`, or a missing chunk.
    pub(crate) fn new(
        codec: Arc<dyn ArrayToBytesCodecTraits>,
        encoded: Option<Vec<u8>>,
        shape: &[NonZeroU64],
    ) -> Self {
        Self {
            codec,
            encoded,
            shape: shape.to_vec(),
        }
    }

    /// A partial decoder of the chunk `bytes` of `shape` encoded with `codec`.
    pub(crate) fn encode(
        codec: Arc<dyn ArrayToBytesCodecTraits>,
        bytes: ArrayBytes<'_>,
        shape: &[NonZeroU64],
    ) -> Self {
        let encoded = codec
            .encode(
                bytes,
                shape,
                &CodecOptions::default(),
                &Resources::default(),
            )
            .unwrap()
            .into_vec();
        Self::new(codec, Some(encoded), shape)
    }

    /// Decode the elements `indexer`.
    pub(crate) fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'static>, CodecError> {
        partial_decode_memory(
            self.codec.clone(),
            self.encoded.as_deref().map(CowBytes::from),
            &self.shape,
            indexer,
            options,
            resources,
        )
    }
}
