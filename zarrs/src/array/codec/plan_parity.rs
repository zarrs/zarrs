//! Parity tests of planned decoding and encoding against whole-chunk decoding and encoding of codec chains.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_codec::plan::{
    ByteSource, DecodeRequest, PlanIndexer, ReadBytes, SharedOutput, SourceKey, SyncIo, TaskTarget,
    TaskValue, drive_serial, drive_sync, encoded_value,
};
use zarrs_codec::{
    ArrayCodecTraits, ArrayToBytesCodecTraits, CodecError, CodecOptions, ComputeResources,
    Resources, update_array_bytes,
};
use zarrs_storage::byte_range::{ByteRange, extract_byte_ranges};
use zarrs_storage::{Bytes, StorageError};

use crate::array::{
    ArrayBytes, ArrayIndices, ArraySubset, CodecChainBound, CowBytes, FillValue, Indexer,
};

/// A backend with one value at [`SourceKey`] 0, and no value at [`SourceKey`] 1.
struct OneValueIo(Bytes, std::sync::atomic::AtomicU64);

impl OneValueIo {
    fn new(bytes: Bytes) -> Self {
        Self(bytes, std::sync::atomic::AtomicU64::new(0))
    }
}

impl SyncIo for OneValueIo {
    fn read(
        &self,
        key: SourceKey,
        ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        if key != SourceKey(0) {
            return Ok(None);
        }
        let bytes: ReadBytes = extract_byte_ranges(&self.0, ranges.iter().copied())?
            .into_iter()
            .map(Bytes::from)
            .collect();
        self.1.fetch_add(
            bytes.iter().map(|bytes| bytes.len() as u64).sum(),
            std::sync::atomic::Ordering::Relaxed,
        );
        Ok(Some(bytes))
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl zarrs_codec::plan::AsyncIo for OneValueIo {
    async fn read(
        &self,
        key: SourceKey,
        ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        SyncIo::read(self, key, ranges)
    }
}

type Driver = for<'a> fn(
    Vec<zarrs_codec::plan::Task<'a>>,
    &dyn SyncIo,
    &Resources,
) -> Result<Vec<TaskValue<'a>>, CodecError>;

fn drivers() -> Vec<(&'static str, Driver)> {
    vec![("serial", drive_serial), ("sync", drive_sync)]
}

/// Decode `indexer` of a chunk of `shape` through a plan, with each driver and target.
#[allow(clippy::too_many_lines)]
fn planned_decodes(
    codecs: &Arc<CodecChainBound>,
    shape: &[NonZeroU64],
    io: &OneValueIo,
    key: SourceKey,
    indexer: &dyn Indexer,
) -> Vec<(String, ArrayBytes<'static>)> {
    let options = CodecOptions::default();
    let resources = Resources::default();
    let data_type = codecs.data_type();
    let mut results = vec![];

    for (name, drive) in drivers() {
        // Return the decoded elements as a value
        let task = codecs
            .clone()
            .plan_decode(
                ByteSource::external(key),
                shape,
                DecodeRequest {
                    indexer: PlanIndexer::Borrowed(indexer),
                    target: TaskTarget::Value,
                },
                &options,
            )
            .unwrap();
        let mut values = drive(vec![task], io, &resources).unwrap();
        let TaskValue::Array(bytes) = values.remove(0) else {
            panic!("expected array bytes")
        };
        results.push((format!("{name}/value"), bytes.into_owned()));

        // Decode into an output
        if let Some((output, views)) =
            SharedOutput::for_data_type(data_type, &indexer.output_shape())
        {
            let task = codecs
                .clone()
                .plan_decode(
                    ByteSource::external(key),
                    shape,
                    DecodeRequest {
                        indexer: PlanIndexer::Borrowed(indexer),
                        target: TaskTarget::Views(views),
                    },
                    &options,
                )
                .unwrap();
            drive(vec![task], io, &resources).unwrap();
            let bytes = unsafe { output.into_array_bytes() }.unwrap();
            results.push((format!("{name}/views"), bytes));
        }
    }

    // A source in memory needs no I/O
    if key == SourceKey(0) {
        let task = codecs
            .clone()
            .plan_decode(
                ByteSource::memory(CowBytes::Borrowed(&io.0)),
                shape,
                DecodeRequest {
                    indexer: PlanIndexer::Borrowed(indexer),
                    target: TaskTarget::Value,
                },
                &options,
            )
            .unwrap();
        let outcome = drive_serial(vec![task], io, &resources).unwrap();
        let [TaskValue::Array(bytes)] = <[_; 1]>::try_from(outcome).unwrap() else {
            panic!("expected array bytes")
        };
        results.push(("memory/value".to_string(), bytes.into_owned()));
    }

    #[cfg(feature = "async")]
    {
        let task = codecs
            .clone()
            .plan_decode(
                ByteSource::external(key),
                shape,
                DecodeRequest {
                    indexer: PlanIndexer::Borrowed(indexer),
                    target: TaskTarget::Value,
                },
                &options,
            )
            .unwrap();
        let mut values =
            futures::executor::block_on(zarrs_codec::plan::drive_async(vec![task], io, &resources))
                .unwrap();
        let TaskValue::Array(bytes) = values.remove(0) else {
            panic!("expected array bytes")
        };
        results.push(("async/value".to_string(), bytes.into_owned()));
    }

    results
}

/// The indexers checked for a chunk of `shape`: the whole chunk, a partial subset, and scattered elements.
fn indexers(shape: &[NonZeroU64]) -> Vec<Box<dyn Indexer>> {
    let shape: Vec<u64> = shape.iter().map(|s| s.get()).collect();
    let mut indexers: Vec<Box<dyn Indexer>> =
        vec![Box::new(ArraySubset::new_with_shape(shape.clone()))];
    let partial: Vec<_> = shape
        .iter()
        .map(|&s| (s / 4)..(s - s / 3).max(s / 4 + 1))
        .collect();
    indexers.push(Box::new(ArraySubset::new_with_ranges(&partial)));
    let last: ArrayIndices = shape.iter().map(|s| s - 1).collect();
    let first: ArrayIndices = std::iter::repeat_n(0, shape.len()).collect();
    let middle: ArrayIndices = shape.iter().map(|s| s / 2).collect();
    indexers.push(Box::new(vec![last.clone(), first, middle, last]));
    indexers
}

/// The number of bytes read by a planned decode of `indexer` from the encoding of chunk `bytes` of `shape`.
pub(crate) fn planned_bytes_read(
    codecs: &Arc<CodecChainBound>,
    shape: &[NonZeroU64],
    bytes: ArrayBytes<'_>,
    indexer: &dyn Indexer,
) -> (u64, u64) {
    let encoded = codecs
        .encode(
            bytes,
            shape,
            &CodecOptions::default(),
            &ComputeResources::default(),
        )
        .unwrap()
        .into_vec();
    let encoded_len = encoded.len() as u64;
    let io = OneValueIo::new(Bytes::from(encoded));
    let task = codecs
        .clone()
        .plan_decode(
            ByteSource::external(SourceKey(0)),
            shape,
            DecodeRequest {
                indexer: PlanIndexer::Borrowed(indexer),
                target: TaskTarget::Value,
            },
            &CodecOptions::default(),
        )
        .unwrap();
    drive_serial(vec![task], &io, &Resources::default()).unwrap();
    (io.1.load(std::sync::atomic::Ordering::Relaxed), encoded_len)
}

/// Assert that planned encoding of chunk `bytes` of `shape` with each driver decodes as `encoded` does.
///
/// The encoded bytes are not compared, since the order of encoded subchunks of a shard may differ between encodes.
fn assert_encode_parity(
    codecs: &Arc<CodecChainBound>,
    shape: &[NonZeroU64],
    bytes: &ArrayBytes<'_>,
    encoded: &[u8],
) {
    let options = CodecOptions::default();
    let resources = Resources::default();
    let expected = codecs
        .decode(
            CowBytes::Borrowed(encoded),
            shape,
            &options,
            resources.compute(),
        )
        .unwrap();
    let io = OneValueIo::new(Bytes::new());
    for (name, drive) in drivers() {
        let task = codecs
            .clone()
            .plan_encode(bytes.clone(), shape, &options)
            .unwrap();
        let mut values = drive(vec![task], &io, &resources).unwrap();
        let planned = encoded_value(values.remove(0)).unwrap();
        let decoded = codecs
            .decode(planned, shape, &options, resources.compute())
            .unwrap();
        assert_eq!(decoded, expected, "{name}: encode");
    }
}

/// Assert that planned decoding of `codecs` matches decoding the whole chunk `bytes` of `shape` and extracting the
/// requested elements.
pub(crate) fn assert_plan_parity(
    codecs: &Arc<CodecChainBound>,
    shape: &[NonZeroU64],
    bytes: ArrayBytes<'_>,
) {
    let options = CodecOptions::default();
    let resources = Resources::default();
    let encoded = codecs
        .encode(bytes.clone(), shape, &options, resources.compute())
        .unwrap()
        .into_vec();
    assert_encode_parity(codecs, shape, &bytes, &encoded);
    let io = OneValueIo::new(Bytes::from(encoded.clone()));
    let decoded = codecs
        .decode(
            CowBytes::from(encoded),
            shape,
            &options,
            resources.compute(),
        )
        .unwrap();
    for indexer in indexers(shape) {
        let expected = decoded
            .extract_array_subset(
                &*indexer,
                bytemuck::must_cast_slice(shape),
                codecs.data_type(),
            )
            .unwrap()
            .into_owned();
        for (name, planned) in planned_decodes(codecs, shape, &io, SourceKey(0), &*indexer) {
            assert_eq!(planned, expected, "{name}: {:?}", indexer.output_shape());
        }

        // A missing chunk is the fill value
        let fill_value: &FillValue = codecs.fill_value();
        let expected =
            ArrayBytes::new_fill_value(codecs.data_type(), indexer.len(), fill_value).unwrap();
        for (name, planned) in planned_decodes(codecs, shape, &io, SourceKey(1), &*indexer) {
            assert_eq!(
                planned,
                expected,
                "missing {name}: {:?}",
                indexer.output_shape()
            );
        }
    }
}

/// A backend with one value at [`SourceKey`] 0 that can be written.
#[derive(Default)]
struct OneWritableValueIo(std::sync::Mutex<Option<Vec<u8>>>);

impl SyncIo for OneWritableValueIo {
    fn read(
        &self,
        key: SourceKey,
        ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        assert_eq!(key, SourceKey(0));
        let value = self.0.lock().unwrap();
        let Some(value) = value.as_ref() else {
            return Ok(None);
        };
        Ok(Some(
            extract_byte_ranges(value, ranges.iter().copied())?
                .into_iter()
                .map(Bytes::from)
                .collect(),
        ))
    }

    fn write(&self, op: zarrs_codec::plan::WriteOp<'_>) -> Result<(), StorageError> {
        use zarrs_codec::plan::WriteOp;
        assert_eq!(op.key(), SourceKey(0));
        let mut value = self.0.lock().unwrap();
        match op {
            WriteOp::Set(_, bytes) => *value = Some(bytes.to_vec()),
            WriteOp::SetPartial(_, offset_values) => {
                let value = value.get_or_insert_default();
                for (offset, bytes) in offset_values {
                    let offset = usize::try_from(offset).unwrap();
                    if value.len() < offset + bytes.len() {
                        value.resize(offset + bytes.len(), 0);
                    }
                    value[offset..offset + bytes.len()].copy_from_slice(&bytes);
                }
            }
            WriteOp::Erase(_) => *value = None,
        }
        Ok(())
    }
}

/// Assert that planned partial encodes of `updates` to an initially missing chunk of `shape` write an encoded chunk
/// that decodes to the chunk updated in full after each update, with each driver, with and without the state of the
/// previous partial encode.
pub(crate) fn assert_partial_encode_parity(
    codecs: &Arc<CodecChainBound>,
    shape: &[NonZeroU64],
    updates: &[(Box<dyn Indexer>, ArrayBytes<'static>)],
) {
    let options = CodecOptions::default();
    let resources = Resources::default();
    let shape_u64: &[u64] = bytemuck::must_cast_slice(shape);
    for (name, drive) in drivers() {
        for keep_state in [false, true] {
            let mut expected = ArrayBytes::new_fill_value(
                codecs.data_type(),
                shape_u64.iter().product(),
                codecs.fill_value(),
            )
            .unwrap();
            let io = OneWritableValueIo::default();
            let mut state = None;
            for (i, (indexer, bytes)) in updates.iter().enumerate() {
                // Round trip the updated chunk, since codecs may be lossy (e.g. bitround)
                let updated = update_array_bytes(
                    expected,
                    shape_u64,
                    indexer.as_ref(),
                    bytes,
                    codecs.data_type().size(),
                )
                .unwrap();
                let encoded = codecs
                    .encode(updated, shape, &options, resources.compute())
                    .unwrap();
                expected = codecs
                    .decode(encoded, shape, &options, resources.compute())
                    .unwrap()
                    .into_owned();
                let task = codecs
                    .clone()
                    .plan_partial_encode(
                        SourceKey(0),
                        shape,
                        PlanIndexer::Borrowed(indexer.as_ref()),
                        bytes.clone(),
                        state.take(),
                        true,
                        &options,
                    )
                    .unwrap();
                let mut values = drive(vec![task], &io, &resources).unwrap();
                let new_state = zarrs_codec::plan::partial_encode_state(values.remove(0));
                assert!(new_state.is_some(), "{name}: a partial encode has a state");
                if keep_state {
                    state = new_state;
                }
                // A chunk that is entirely the fill value may be erased
                let decoded = match io.0.lock().unwrap().clone() {
                    Some(encoded) => codecs
                        .decode(
                            CowBytes::from(encoded),
                            shape,
                            &options,
                            resources.compute(),
                        )
                        .unwrap(),
                    None => ArrayBytes::new_fill_value(
                        codecs.data_type(),
                        shape_u64.iter().product(),
                        codecs.fill_value(),
                    )
                    .unwrap(),
                };
                assert_eq!(
                    decoded, expected,
                    "{name} (keep state: {keep_state}): update {i}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::codec::{BytesCodec, CodecChain};
    use crate::array::{ChunkShape, data_type};
    use zarrs_codec::{
        BytesToBytesCodecTraits, CodecSpecificOptions, UnboundArrayToArrayCodecTraits,
        UnboundArrayToBytesCodecTraits,
    };

    pub(crate) fn chain(
        array_to_array: Vec<Arc<dyn UnboundArrayToArrayCodecTraits>>,
        array_to_bytes: Arc<dyn UnboundArrayToBytesCodecTraits>,
        bytes_to_bytes: Vec<Arc<dyn BytesToBytesCodecTraits>>,
        data_type: crate::array::DataType,
        fill_value: FillValue,
    ) -> Arc<CodecChainBound> {
        CodecChain::new(array_to_array, array_to_bytes, bytes_to_bytes)
            .with_context(data_type, fill_value, &CodecSpecificOptions::default())
            .unwrap()
    }

    fn u16_bytes(shape: &[NonZeroU64]) -> ArrayBytes<'static> {
        let n = shape.iter().map(|s| s.get()).product::<u64>();
        let data: Vec<u16> = (0..n).map(|i| u16::try_from(i % 1000).unwrap()).collect();
        ArrayBytes::new_flen(crate::array::transmute_to_bytes_vec(data))
    }

    fn shape(shape: &[u64]) -> ChunkShape {
        shape.iter().map(|&s| NonZeroU64::new(s).unwrap()).collect()
    }

    #[test]
    fn plan_parity_bytes() {
        let shape = shape(&[8, 6]);
        let codecs = chain(
            vec![],
            Arc::new(BytesCodec::new(Some(crate::array::Endianness::Big))),
            vec![],
            data_type::uint16(),
            FillValue::from(7u16),
        );
        assert_plan_parity(&codecs, &shape, u16_bytes(&shape));
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn plan_parity_bytes_gzip() {
        let shape = shape(&[8, 6]);
        let codecs = chain(
            vec![],
            Arc::new(BytesCodec::default()),
            vec![Arc::new(crate::array::codec::GzipCodec::new(5).unwrap())],
            data_type::uint16(),
            FillValue::from(7u16),
        );
        assert_plan_parity(&codecs, &shape, u16_bytes(&shape));
    }

    #[test]
    fn plan_parity_sharding() {
        let shape = shape(&[8, 6]);
        let sharding = crate::array::codec::ShardingCodecBuilder::new(
            vec![NonZeroU64::new(3).unwrap(), NonZeroU64::new(4).unwrap()],
            &data_type::uint16(),
        )
        .build_arc();
        let codecs = chain(
            vec![],
            sharding,
            vec![],
            data_type::uint16(),
            FillValue::from(7u16),
        );
        assert_plan_parity(&codecs, &shape, u16_bytes(&shape));
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn plan_parity_bytes_zstd_crc32c() {
        let shape = shape(&[8, 6]);
        let codecs = chain(
            vec![],
            Arc::new(BytesCodec::new(Some(crate::array::Endianness::Big))),
            vec![
                Arc::new(crate::array::codec::ZstdCodec::new(3, false)),
                #[cfg(feature = "crc32c")]
                Arc::new(crate::array::codec::Crc32cCodec::new()),
            ],
            data_type::uint16(),
            FillValue::from(7u16),
        );
        assert_plan_parity(&codecs, &shape, u16_bytes(&shape));
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn plan_parity_sharding_in_gzip() {
        let shape = shape(&[8, 6]);
        let sharding = crate::array::codec::ShardingCodecBuilder::new(
            vec![NonZeroU64::new(3).unwrap(), NonZeroU64::new(4).unwrap()],
            &data_type::uint16(),
        )
        .build_arc();
        let codecs = chain(
            vec![],
            sharding,
            vec![Arc::new(crate::array::codec::GzipCodec::new(5).unwrap())],
            data_type::uint16(),
            FillValue::from(7u16),
        );
        assert_plan_parity(&codecs, &shape, u16_bytes(&shape));
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn plan_parity_gzip_in_sharding() {
        let shape = shape(&[8, 6]);
        let sharding = crate::array::codec::ShardingCodecBuilder::new(
            vec![NonZeroU64::new(3).unwrap(), NonZeroU64::new(4).unwrap()],
            &data_type::uint16(),
        )
        .bytes_to_bytes_codecs(vec![Arc::new(
            crate::array::codec::GzipCodec::new(5).unwrap(),
        )])
        .build_arc();
        let codecs = chain(
            vec![],
            sharding,
            vec![],
            data_type::uint16(),
            FillValue::from(7u16),
        );
        assert_plan_parity(&codecs, &shape, u16_bytes(&shape));
    }

    #[cfg(all(feature = "crc32c", feature = "fletcher32"))]
    #[test]
    fn plan_parity_checksum_windows() {
        use crate::array::codec::{Crc32cCodec, Fletcher32Codec};
        let shape = shape(&[8, 6]);
        let bytes = || Arc::new(BytesCodec::new(Some(crate::array::Endianness::Big)));
        let sharding = || {
            crate::array::codec::ShardingCodecBuilder::new(
                vec![NonZeroU64::new(3).unwrap(), NonZeroU64::new(4).unwrap()],
                &data_type::uint16(),
            )
            .build_arc()
        };
        type Chain = (
            Arc<dyn UnboundArrayToBytesCodecTraits>,
            Vec<Arc<dyn BytesToBytesCodecTraits>>,
        );
        let chains: Vec<Chain> = vec![
            (bytes(), vec![Arc::new(Crc32cCodec::new())]),
            (
                bytes(),
                vec![
                    Arc::new(Crc32cCodec::new()),
                    Arc::new(Fletcher32Codec::new()),
                ],
            ),
            (sharding(), vec![Arc::new(Crc32cCodec::new())]),
            #[cfg(feature = "gzip")]
            (
                bytes(),
                vec![
                    Arc::new(crate::array::codec::GzipCodec::new(5).unwrap()),
                    Arc::new(Crc32cCodec::new()),
                ],
            ),
            #[cfg(feature = "adler32")]
            (
                sharding(),
                vec![Arc::new(
                    crate::array::codec::Adler32Codec::new_with_configuration(
                        &serde_json::from_str(r#"{"location":"start"}"#).unwrap(),
                    )
                    .unwrap(),
                )],
            ),
        ];
        for (array_to_bytes, bytes_to_bytes) in chains {
            let codecs = chain(
                vec![],
                array_to_bytes,
                bytes_to_bytes,
                data_type::uint16(),
                FillValue::from(7u16),
            );
            assert_plan_parity(&codecs, &shape, u16_bytes(&shape));
        }
    }

    #[test]
    fn plan_parity_bytes_and_packbits_partial_reads() {
        use crate::array::codec::PackBitsCodec;
        use crate::metadata_ext::codec::packbits::PackBitsPaddingEncoding;
        let shape = shape(&[16, 12]);
        let codecs: Vec<Arc<dyn UnboundArrayToBytesCodecTraits>> = vec![
            Arc::new(BytesCodec::new(Some(crate::array::Endianness::Little))),
            Arc::new(BytesCodec::new(Some(crate::array::Endianness::Big))),
            Arc::new(PackBitsCodec::new(PackBitsPaddingEncoding::None, None, None).unwrap()),
            Arc::new(
                PackBitsCodec::new(PackBitsPaddingEncoding::FirstByte, Some(2), Some(11)).unwrap(),
            ),
            Arc::new(
                PackBitsCodec::new(PackBitsPaddingEncoding::LastByte, Some(0), Some(12)).unwrap(),
            ),
        ];
        for array_to_bytes in codecs {
            let codecs = chain(
                vec![],
                array_to_bytes,
                vec![],
                data_type::uint16(),
                FillValue::from(7u16),
            );
            assert_plan_parity(&codecs, &shape, u16_bytes(&shape));
            // A partial read reads only the bytes of the requested elements
            let (read, total) = planned_bytes_read(
                &codecs,
                &shape,
                u16_bytes(&shape),
                &ArraySubset::new_with_ranges(&[2..4, 0..12]),
            );
            assert!(read * 4 <= total, "read {read} of {total} bytes");
        }
    }

    #[cfg(feature = "blosc")]
    #[test]
    fn plan_parity_blosc() {
        use crate::array::codec::BloscCodec;
        use crate::metadata_ext::codec::blosc::{
            BloscCompressionLevel, BloscCompressor, BloscShuffleMode,
        };
        let shape = shape(&[16, 12]);
        let blosc = || {
            Arc::new(
                BloscCodec::new(
                    BloscCompressor::LZ4,
                    BloscCompressionLevel::try_from(5).unwrap(),
                    Some(64),
                    BloscShuffleMode::Shuffle,
                    Some(2),
                )
                .unwrap(),
            )
        };
        let codecs = chain(
            vec![],
            Arc::new(BytesCodec::default()),
            vec![blosc()],
            data_type::uint16(),
            FillValue::from(7u16),
        );
        assert_plan_parity(&codecs, &shape, u16_bytes(&shape));
        #[cfg(feature = "crc32c")]
        {
            let codecs = chain(
                vec![],
                Arc::new(BytesCodec::default()),
                vec![blosc(), Arc::new(crate::array::codec::Crc32cCodec::new())],
                data_type::uint16(),
                FillValue::from(7u16),
            );
            assert_plan_parity(&codecs, &shape, u16_bytes(&shape));
        }
    }

    /// An array-to-array codec from its metadata.
    fn array_to_array(json: &str) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        let metadata: zarrs_metadata::v3::MetadataV3 = serde_json::from_str(json).unwrap();
        match zarrs_codec::Codec::from_metadata(&metadata).unwrap() {
            zarrs_codec::Codec::ArrayToArray(codec) => codec,
            zarrs_codec::Codec::ArrayToBytes(_) | zarrs_codec::Codec::BytesToBytes(_) => {
                panic!("expected an array-to-array codec")
            }
        }
    }

    fn sharding_u16(subchunk_shape: &[u64]) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        crate::array::codec::ShardingCodecBuilder::new(
            subchunk_shape
                .iter()
                .map(|&s| NonZeroU64::new(s).unwrap())
                .collect(),
            &data_type::uint16(),
        )
        .build_arc()
    }

    #[test]
    fn plan_parity_array_to_array() {
        let transpose =
            || array_to_array(r#"{"name":"transpose","configuration":{"order":[1,0]}}"#);
        type Case = (
            &'static [u64],
            Vec<Arc<dyn UnboundArrayToArrayCodecTraits>>,
            Arc<dyn UnboundArrayToBytesCodecTraits>,
        );
        let cases: Vec<Case> = vec![
            (&[8, 6], vec![transpose()], Arc::new(BytesCodec::default())),
            (&[8, 6], vec![transpose()], sharding_u16(&[3, 4])),
            (
                &[8, 1, 6],
                vec![array_to_array(
                    r#"{"name":"zarrs.squeeze","configuration":{}}"#,
                )],
                sharding_u16(&[3, 4]),
            ),
            (
                &[8, 6],
                vec![array_to_array(
                    r#"{"name":"reshape","configuration":{"shape":[[0, 1]]}}"#,
                )],
                Arc::new(BytesCodec::default()),
            ),
            (
                &[8, 6],
                vec![
                    transpose(),
                    array_to_array(
                        r#"{"name":"cast_value","configuration":{"data_type":"uint32"}}"#,
                    ),
                ],
                Arc::new(BytesCodec::default()),
            ),
        ];
        for (shape_u64, array_to_array_codecs, array_to_bytes) in cases {
            let shape = shape(shape_u64);
            let codecs = chain(
                array_to_array_codecs,
                array_to_bytes,
                vec![],
                data_type::uint16(),
                FillValue::from(7u16),
            );
            assert_plan_parity(&codecs, &shape, u16_bytes(&shape));
        }
    }

    #[cfg(all(feature = "bitround", feature = "gzip"))]
    #[test]
    fn plan_parity_array_to_array_float() {
        let shape = shape(&[8, 6]);
        let n = 48;
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 1.37).collect();
        let bytes = ArrayBytes::new_flen(crate::array::transmute_to_bytes_vec(data));
        let codecs = chain(
            vec![
                array_to_array(r#"{"name":"bitround","configuration":{"keepbits":10}}"#),
                array_to_array(r#"{"name":"transpose","configuration":{"order":[1,0]}}"#),
                array_to_array(
                    r#"{"name":"numcodecs.fixedscaleoffset","configuration":{"offset":1.5,"scale":4,"dtype":"<f4","astype":"<f4"}}"#,
                ),
            ],
            crate::array::codec::ShardingCodecBuilder::new(
                vec![NonZeroU64::new(3).unwrap(), NonZeroU64::new(4).unwrap()],
                &data_type::float32(),
            )
            .bytes_to_bytes_codecs(vec![Arc::new(
                crate::array::codec::GzipCodec::new(5).unwrap(),
            )])
            .build_arc(),
            vec![],
            data_type::float32(),
            FillValue::from(0.5f32),
        );
        assert_plan_parity(&codecs, &shape, bytes);
    }

    fn sharding_builder(
        subchunk_shape: &[u64],
        data_type: &crate::array::DataType,
    ) -> crate::array::codec::ShardingCodecBuilder {
        let mut builder = crate::array::codec::ShardingCodecBuilder::new(
            subchunk_shape
                .iter()
                .map(|&s| NonZeroU64::new(s).unwrap())
                .collect(),
            data_type,
        );
        builder.array_to_bytes_codec(crate::array::codec::default_array_to_bytes_codec(data_type));
        builder
    }

    #[test]
    fn plan_parity_sharding_variants() {
        use crate::array::codec::array_to_bytes::sharding::ShardingIndexLocation;
        let shape = shape(&[12, 10]);
        // Some subchunks are entirely the fill value, so they are absent from the shard
        let n = 120u16;
        let data: Vec<u16> = (0..n).map(|i| if i % 40 < 15 { 7 } else { i }).collect();
        let bytes = || ArrayBytes::new_flen(crate::array::transmute_to_bytes_vec(data.clone()));

        // Index at the start
        let mut builder = sharding_builder(&[3, 4], &data_type::uint16());
        builder.index_location(ShardingIndexLocation::Start);
        let codecs = chain(
            vec![],
            builder.build_arc(),
            vec![],
            data_type::uint16(),
            FillValue::from(7u16),
        );
        assert_plan_parity(&codecs, &shape, bytes());

        // Nested sharding
        let inner = sharding_builder(&[2, 2], &data_type::uint16()).build_arc();
        let mut outer = sharding_builder(&[6, 5], &data_type::uint16());
        outer.array_to_bytes_codec(inner);
        let codecs = chain(
            vec![],
            outer.build_arc(),
            vec![],
            data_type::uint16(),
            FillValue::from(7u16),
        );
        assert_plan_parity(&codecs, &shape, bytes());

        // A shard in memory (decompressed) with subchunks whose partial decodes plan tasks
        #[cfg(all(feature = "gzip", feature = "transpose"))]
        {
            let mut builder = sharding_builder(&[3, 4], &data_type::uint16());
            builder.array_to_array_codecs(vec![array_to_array(
                r#"{"name":"transpose","configuration":{"order":[1,0]}}"#,
            )]);
            let codecs = chain(
                vec![],
                builder.build_arc(),
                vec![Arc::new(crate::array::codec::GzipCodec::new(5).unwrap())],
                data_type::uint16(),
                FillValue::from(7u16),
            );
            assert_plan_parity(&codecs, &shape, bytes());
        }

        // A partial read of one subchunk reads only its bytes and the shard index
        let codecs = chain(
            vec![],
            sharding_builder(&[3, 4], &data_type::uint16()).build_arc(),
            vec![],
            data_type::uint16(),
            FillValue::from(7u16),
        );
        let (read, total) = planned_bytes_read(
            &codecs,
            &shape,
            bytes(),
            &ArraySubset::new_with_ranges(&[9..11, 8..10]),
        );
        // The index of 4 x 3 subchunks (16 bytes each, and a 4 byte checksum) and one subchunk of 2 x 2 elements
        assert_eq!(read, 4 * 3 * 16 + 4 + 8, "read {read} of {total} bytes");
    }

    #[test]
    fn plan_parity_sharding_variable_length() {
        let shape = shape(&[6, 5]);
        let strings: Vec<String> = (0..30)
            .map(|i| {
                if i % 7 == 0 {
                    String::new()
                } else {
                    "x".repeat(i % 5) + &i.to_string()
                }
            })
            .collect();
        let mut offsets = vec![0usize];
        let mut bytes = Vec::new();
        for string in &strings {
            bytes.extend_from_slice(string.as_bytes());
            offsets.push(bytes.len());
        }
        let data = ArrayBytes::new_vlen(
            bytes,
            crate::array::array_bytes_internal::offsets_from_usize(offsets).unwrap(),
        )
        .unwrap();
        let codecs = chain(
            vec![],
            sharding_builder(&[2, 2], &data_type::string()).build_arc(),
            vec![],
            data_type::string(),
            FillValue::from(""),
        );
        assert_plan_parity(&codecs, &shape, data);
    }

    #[test]
    fn plan_parity_sharding_optional() {
        let shape = shape(&[6, 5]);
        let data_type = data_type::optional(data_type::uint16());
        let values: Vec<u16> = (0..30).collect();
        let mask: Vec<u8> = (0..30).map(|i| u8::from(i % 3 != 0)).collect();
        let data = ArrayBytes::new_flen(crate::array::transmute_to_bytes_vec(values))
            .with_optional_mask(mask);
        let codecs = chain(
            vec![],
            sharding_builder(&[2, 2], &data_type).build_arc(),
            vec![],
            data_type,
            FillValue::new_optional_null(),
        );
        assert_plan_parity(&codecs, &shape, data);
    }

    #[test]
    fn partial_encode_parity_sharding() {
        use crate::array::codec::ShardingIndexLocation;
        let shape = shape(&[8, 6]);
        let subset = |ranges: &[std::ops::Range<u64>]| -> Box<dyn Indexer> {
            Box::new(ArraySubset::new_with_ranges(ranges))
        };
        let values = |n: u64, value: u16| -> ArrayBytes<'static> {
            ArrayBytes::new_flen(crate::array::transmute_to_bytes_vec(vec![
                value;
                usize::try_from(n)
                    .unwrap()
            ]))
        };
        let updates: Vec<(Box<dyn Indexer>, ArrayBytes<'static>)> = vec![
            // Straddles subchunks of a missing shard
            (subset(&[1..5, 2..6]), values(16, 1)),
            // Straddles existing subchunks
            (subset(&[2..7, 0..3]), values(15, 2)),
            // Replaces a subchunk entirely
            (subset(&[0..3, 0..4]), values(12, 3)),
            // Elements by index, including a duplicate
            (
                Box::new(vec![vec![7u64, 5], vec![0, 0], vec![7, 5]]),
                values(3, 4),
            ),
            // The fill value empties subchunks
            (subset(&[0..3, 0..6]), values(18, 7)),
            // Writes nothing
            (subset(&[0..0, 0..0]), values(0, 5)),
            // Rewrites every subchunk, then empties the shard
            (subset(&[0..8, 0..6]), values(48, 6)),
            (subset(&[0..8, 0..6]), values(48, 7)),
            (subset(&[3..4, 3..4]), values(1, 8)),
        ];
        for index_location in [ShardingIndexLocation::Start, ShardingIndexLocation::End] {
            for gzip in [false, true] {
                let mut builder = crate::array::codec::ShardingCodecBuilder::new(
                    vec![NonZeroU64::new(3).unwrap(), NonZeroU64::new(4).unwrap()],
                    &data_type::uint16(),
                );
                builder.index_location(index_location);
                #[cfg(feature = "gzip")]
                if gzip {
                    builder.bytes_to_bytes_codecs(vec![Arc::new(
                        crate::array::codec::GzipCodec::new(5).unwrap(),
                    )]);
                }
                #[cfg(not(feature = "gzip"))]
                let _ = gzip;
                let codecs = chain(
                    vec![],
                    builder.build_arc(),
                    vec![],
                    data_type::uint16(),
                    FillValue::from(7u16),
                );
                assert_partial_encode_parity(&codecs, &shape, &updates);
            }
        }

        // A chain with other codecs is not partially encoded
        #[cfg(feature = "gzip")]
        {
            let codecs = chain(
                vec![],
                sharding_u16(&[3, 4]),
                vec![Arc::new(crate::array::codec::GzipCodec::new(5).unwrap())],
                data_type::uint16(),
                FillValue::from(7u16),
            );
            let task = codecs
                .clone()
                .plan_partial_encode(
                    SourceKey(0),
                    &shape,
                    PlanIndexer::Subset(ArraySubset::new_with_ranges(&[0..1, 0..1])),
                    values(1, 1),
                    None,
                    true,
                    &CodecOptions::default(),
                )
                .unwrap();
            // The whole chunk is updated, which has no state
            let io = OneWritableValueIo::default();
            let value = drive_serial(vec![task], &io, &Resources::default())
                .unwrap()
                .remove(0);
            assert!(zarrs_codec::plan::partial_encode_state(value).is_none());
            let encoded = io.0.lock().unwrap().clone().expect("the chunk is written");
            let decoded = codecs
                .decode(
                    CowBytes::from(encoded),
                    &shape,
                    &CodecOptions::default(),
                    &ComputeResources::default(),
                )
                .unwrap();
            let mut expected = ArrayBytes::new_fill_value(
                codecs.data_type(),
                shape.iter().map(|s| s.get()).product(),
                codecs.fill_value(),
            )
            .unwrap();
            expected = update_array_bytes(
                expected,
                bytemuck::must_cast_slice(&shape),
                &ArraySubset::new_with_ranges(&[0..1, 0..1]),
                &values(1, 1),
                codecs.data_type().size(),
            )
            .unwrap();
            assert_eq!(decoded, expected);
        }
    }

    #[test]
    fn partial_encode_parity_bytes_and_array_to_array() {
        #[cfg(feature = "bitround")]
        use crate::array::codec::BitroundCodec;
        #[cfg(feature = "transpose")]
        use crate::array::codec::TransposeCodec;
        use crate::array::codec::array_to_bytes::bytes::Endianness;
        use crate::array::codec::{FixedScaleOffsetCodec, ReshapeCodec, SqueezeCodec};
        fn updates<T: bytemuck::Pod>(
            value: impl Fn(u16) -> T,
            fill_value: T,
            dimensionality: usize,
        ) -> Vec<(Box<dyn Indexer>, ArrayBytes<'static>)> {
            let values = |n: usize, start: u16| -> ArrayBytes<'static> {
                let values: Vec<T> = (0..n)
                    .map(|i| value(start + u16::try_from(i).unwrap()))
                    .collect();
                ArrayBytes::new_flen(bytemuck::cast_slice::<T, u8>(&values).to_vec())
            };
            let subset = |ranges: &[std::ops::Range<u64>]| -> Box<dyn Indexer> {
                // Insert a dimension of length 1 after the first dimension
                let mut ranges = ranges.to_vec();
                if dimensionality == 3 {
                    ranges.insert(1, 0..1);
                }
                Box::new(ArraySubset::new_with_ranges(&ranges))
            };
            let index = |indices: &[[u64; 2]]| -> Box<dyn Indexer> {
                Box::new(
                    indices
                        .iter()
                        .map(|&[i, j]| {
                            if dimensionality == 3 {
                                vec![i, 0, j]
                            } else {
                                vec![i, j]
                            }
                        })
                        .collect::<Vec<_>>(),
                )
            };
            let fill =
                ArrayBytes::new_flen(bytemuck::cast_slice::<T, u8>(&[fill_value; 4]).to_vec());
            vec![
                // A missing chunk
                (subset(&[1..5, 2..6]), values(16, 100)),
                // An existing chunk
                (subset(&[2..7, 0..3]), values(15, 200)),
                // Elements by index, including a duplicate
                (index(&[[7, 5], [0, 0], [7, 5]]), values(3, 300)),
                // The fill value
                (subset(&[0..2, 0..2]), fill),
                // Writes nothing
                (subset(&[0..0, 0..0]), values(0, 400)),
                // Every element
                (subset(&[0..8, 0..6]), values(48, 500)),
            ]
        }
        fn config<T: serde::de::DeserializeOwned>(json: &str) -> T {
            serde_json::from_str(json).unwrap()
        }
        let u16_updates = updates(|i| i, 7u16, 2);
        let codecs: Vec<(Vec<Arc<dyn UnboundArrayToArrayCodecTraits>>, Endianness)> = vec![
            (vec![], Endianness::Little),
            (vec![], Endianness::Big),
            #[cfg(feature = "transpose")]
            (
                vec![Arc::new(
                    TransposeCodec::new_with_configuration(&config(r#"{"order":[1,0]}"#)).unwrap(),
                )],
                Endianness::Little,
            ),
            (
                vec![Arc::new(
                    ReshapeCodec::new_with_configuration(&config(r#"{"shape":[[0,1]]}"#)).unwrap(),
                )],
                Endianness::Big,
            ),
            #[cfg(feature = "bitround")]
            (vec![Arc::new(BitroundCodec::new(3))], Endianness::Little),
            #[cfg(all(feature = "bitround", feature = "transpose"))]
            (
                vec![
                    Arc::new(BitroundCodec::new(3)),
                    Arc::new(
                        TransposeCodec::new_with_configuration(&config(r#"{"order":[1,0]}"#))
                            .unwrap(),
                    ),
                ],
                Endianness::Little,
            ),
        ];
        let shape_2d = shape(&[8, 6]);
        for (array_to_array, endianness) in codecs {
            let codecs = chain(
                array_to_array,
                Arc::new(BytesCodec::new(Some(endianness))),
                vec![],
                data_type::uint16(),
                FillValue::from(7u16),
            );
            assert_partial_encode_parity(&codecs, &shape_2d, &u16_updates);
        }

        // Squeeze a dimension of length 1
        let codecs = chain(
            vec![Arc::new(SqueezeCodec::new())],
            Arc::new(BytesCodec::new(Some(Endianness::Little))),
            vec![],
            data_type::uint16(),
            FillValue::from(7u16),
        );
        assert_partial_encode_parity(&codecs, &shape(&[8, 1, 6]), &updates(|i| i, 7u16, 3));

        // A type-changing elementwise codec
        let codecs = chain(
            vec![Arc::new(
                FixedScaleOffsetCodec::new_with_configuration(&config(
                    r#"{"offset":1000,"scale":10,"dtype":"f8","astype":"u2"}"#,
                ))
                .unwrap(),
            )],
            Arc::new(BytesCodec::new(Some(Endianness::Little))),
            vec![],
            data_type::float64(),
            FillValue::from(1000.0f64),
        );
        assert_partial_encode_parity(
            &codecs,
            &shape_2d,
            &updates(|i| 1000.0 + f64::from(i) / 10.0, 1000.0f64, 2),
        );

        // Transposed shards
        #[cfg(feature = "transpose")]
        {
            let codecs = chain(
                vec![Arc::new(
                    TransposeCodec::new_with_configuration(&config(r#"{"order":[1,0]}"#)).unwrap(),
                )],
                sharding_u16(&[3, 4]),
                vec![],
                data_type::uint16(),
                FillValue::from(7u16),
            );
            assert_partial_encode_parity(&codecs, &shape_2d, &u16_updates);
        }
    }

    #[test]
    fn partial_encode_corrupt_shard_index() {
        let shape = shape(&[8, 6]);
        let codecs = chain(
            vec![],
            sharding_u16(&[3, 4]),
            vec![],
            data_type::uint16(),
            FillValue::from(7u16),
        );
        // A subchunk ends beyond the maximum shard size
        let mut index = vec![u64::MAX; 12];
        index[10] = u64::MAX - 1;
        index[11] = 5;
        let state: zarrs_codec::plan::PartialEncodeState = Arc::new(index);
        let task = codecs
            .plan_partial_encode(
                SourceKey(0),
                &shape,
                PlanIndexer::Subset(ArraySubset::new_with_ranges(&[0..1, 0..1])),
                ArrayBytes::new_flen(vec![1, 0]),
                Some(state),
                true,
                &CodecOptions::default(),
            )
            .unwrap();
        let io = OneWritableValueIo::default();
        assert!(drive_serial(vec![task], &io, &Resources::default()).is_err());
        assert!(io.0.lock().unwrap().is_none());
    }
}
