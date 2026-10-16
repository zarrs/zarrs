//! Partial decoding and encoding of an encoded chunk in memory.

use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};

use zarrs_chunk_grid::Indexer;
use zarrs_storage::byte_range::{ByteRange, InvalidByteRangeError};
use zarrs_storage::{Bytes, StorageError};

use super::{
    ByteSource, DecodeRequest, PlanIndexer, ReadBytes, SourceKey, SyncIo, Task, TaskTarget,
    TaskValue, WriteOp,
};
use crate::{
    ArrayBytes, ArrayToBytesCodecTraits, CodecError, CodecOptions, CowBytes, Resources,
    update_array_bytes,
};

/// Drive `task` to completion with the backend `io`.
fn drive_one<'a>(
    task: Task<'a>,
    io: &dyn SyncIo,
    resources: &Resources,
) -> Result<TaskValue<'a>, CodecError> {
    #[cfg(not(target_family = "wasm"))]
    return super::drive_sync_one(task, io, resources);
    #[cfg(target_family = "wasm")]
    return super::drive_serial_one(task, io, resources);
}

/// Decode the elements `indexer` of the encoded chunk `encoded` of `shape` with `codec`.
///
/// If `encoded` is [`None`], the chunk does not exist and the elements are the fill value.
///
/// # Errors
/// Returns a [`CodecError`] if `indexer` is invalid for the chunk or the codec fails.
pub fn partial_decode_memory(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    encoded: Option<CowBytes<'_>>,
    shape: &[NonZeroU64],
    indexer: &dyn Indexer,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<ArrayBytes<'static>, CodecError> {
    let data_type = codec.data_type().clone();
    let source = match &encoded {
        Some(encoded) => ByteSource::memory(CowBytes::Borrowed(encoded)),
        None => ByteSource::external(SourceKey(0)),
    };
    let task = codec.plan_decode(
        source,
        shape,
        DecodeRequest {
            indexer: PlanIndexer::Borrowed(indexer),
            target: TaskTarget::Value,
        },
        options,
    )?;
    let bytes = match drive_one(task, &MemoryValueIo::default(), resources)? {
        TaskValue::Array(bytes) => bytes.into_owned(),
        TaskValue::None | TaskValue::Bytes(_) | TaskValue::Prepared(_) | TaskValue::Any(_) => {
            return Err(CodecError::Other(
                "a decode task did not produce array bytes".to_string(),
            ));
        }
    };
    bytes.validate(indexer.len(), &data_type)?;
    Ok(bytes)
}

/// Update the elements `indexer` of the encoded chunk `encoded` of `shape` with `codec` to `bytes`.
///
/// If `encoded` is [`None`], the chunk does not exist and its other elements are the fill value. After the update,
/// `encoded` is [`None`] if the chunk was erased (it is the fill value, and empty chunks are not stored).
///
/// The chunk is partially encoded if the codec plans partial encodes
/// ([`ArrayToBytesCodecTraits::plans_partial_encode`]), and otherwise decoded, updated and encoded.
///
/// # Errors
/// Returns a [`CodecError`] if `indexer` or `bytes` are invalid for the chunk or the codec fails.
pub fn partial_encode_memory(
    codec: Arc<dyn ArrayToBytesCodecTraits>,
    encoded: &mut Option<Vec<u8>>,
    shape: &[NonZeroU64],
    indexer: &dyn Indexer,
    bytes: &ArrayBytes<'_>,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<(), CodecError> {
    bytes.validate(indexer.len(), codec.data_type())?;
    if codec.plans_partial_encode() {
        let io = MemoryValueIo(Mutex::new(encoded.take().map(Bytes::from)));
        let task = codec
            .plan_partial_encode(
                SourceKey(0),
                shape,
                PlanIndexer::Borrowed(indexer),
                bytes.clone(),
                None,
                options,
            )?
            .ok_or_else(|| {
                CodecError::Other("a codec did not plan a partial encode".to_string())
            })?;
        let result = drive_one(task, &io, resources);
        *encoded = io.0.into_inner().unwrap().map(|value| value.to_vec());
        result?;
        return Ok(());
    }

    let chunk = match encoded.take() {
        Some(value) => codec.decode(CowBytes::from(value), shape, options, resources)?,
        None => ArrayBytes::new_fill_value(
            codec.data_type(),
            shape.iter().map(|s| s.get()).product(),
            codec.fill_value(),
        )?,
    };
    let chunk = update_array_bytes(
        chunk,
        bytemuck::must_cast_slice(shape),
        indexer,
        bytes,
        codec.data_type().size(),
    )?;
    if !options.store_empty_chunks() && chunk.is_fill_value(codec.fill_value()) {
        return Ok(());
    }
    *encoded = Some(codec.encode(chunk, shape, options, resources)?.into_vec());
    Ok(())
}

/// The backend of a drive with one value at [`SourceKey`] 0 in memory, if it exists.
#[derive(Default)]
struct MemoryValueIo(Mutex<Option<Bytes>>);

impl SyncIo for MemoryValueIo {
    fn read(
        &self,
        _key: SourceKey,
        ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        let value = self.0.lock().unwrap();
        let Some(value) = value.as_ref() else {
            return Ok(None);
        };
        let len = value.len() as u64;
        let read: ReadBytes = ranges
            .iter()
            .map(|range| {
                let valid = match *range {
                    ByteRange::FromStart(offset, length) => offset
                        .checked_add(length.unwrap_or(0))
                        .is_some_and(|end| offset <= len && end <= len),
                    ByteRange::Suffix(length) => length <= len,
                };
                if valid {
                    Ok(value.slice(range.to_range_usize(len)))
                } else {
                    Err(InvalidByteRangeError::new(*range, len))
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(Some(read))
    }

    fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
        let mut value = self.0.lock().unwrap();
        match op {
            WriteOp::Set(_, bytes) => *value = Some(Bytes::copy_from_slice(&bytes)),
            WriteOp::SetPartial(_, offset_values) => {
                let mut updated = value
                    .as_ref()
                    .map(|value| value.to_vec())
                    .unwrap_or_default();
                for (offset, bytes) in offset_values {
                    // As in a memory store, a write beyond the end of the value zero fills the gap.
                    let too_large = || {
                        StorageError::Other(format!(
                            "a partial write of {} bytes at offset {offset} is too large",
                            bytes.len()
                        ))
                    };
                    let start = usize::try_from(offset).map_err(|_| too_large())?;
                    let end = start.checked_add(bytes.len()).ok_or_else(too_large)?;
                    if updated.len() < end {
                        updated
                            .try_reserve_exact(end - updated.len())
                            .map_err(|_| too_large())?;
                        updated.resize(end, 0);
                    }
                    updated[start..end].copy_from_slice(&bytes);
                }
                *value = Some(Bytes::from(updated));
            }
            WriteOp::Erase(_) => *value = None,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn io(value: &[u8]) -> MemoryValueIo {
        MemoryValueIo(Mutex::new(Some(Bytes::copy_from_slice(value))))
    }

    #[test]
    fn memory_value_io_read_out_of_range() {
        let io = io(&[1, 2, 3]);
        let read = io
            .read(SourceKey(0), &[ByteRange::FromStart(1, Some(2))])
            .unwrap()
            .unwrap();
        assert_eq!(read[0].as_ref(), &[2, 3]);
        for range in [
            ByteRange::FromStart(2, Some(2)),
            ByteRange::FromStart(4, None),
            ByteRange::FromStart(u64::MAX, Some(2)),
            ByteRange::Suffix(4),
        ] {
            assert!(io.read(SourceKey(0), &[range]).is_err());
        }
    }

    #[test]
    fn memory_value_io_write_partial() {
        let io = io(&[1, 2, 3]);
        io.write(WriteOp::SetPartial(
            SourceKey(0),
            vec![(1, CowBytes::from(&[4][..])), (5, CowBytes::from(&[5][..]))],
        ))
        .unwrap();
        assert_eq!(
            io.0.lock().unwrap().as_deref(),
            Some(&[1, 4, 3, 0, 0, 5][..])
        );
    }

    #[test]
    fn memory_value_io_write_partial_too_large() {
        let io = io(&[1, 2, 3]);
        for offset in [u64::MAX, u64::MAX / 2] {
            assert!(
                io.write(WriteOp::SetPartial(
                    SourceKey(0),
                    vec![(offset, CowBytes::from(&[4][..]))],
                ))
                .is_err()
            );
        }
        assert_eq!(io.0.lock().unwrap().as_deref(), Some(&[1, 2, 3][..]));
    }
}
