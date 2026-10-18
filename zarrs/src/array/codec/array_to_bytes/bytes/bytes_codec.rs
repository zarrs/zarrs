// Note: No validation that this codec is created *without* a specified endianness for multi-byte data types.

use std::sync::Arc;

use zarrs_plugin::{ExtensionAliasesV3, PluginCreateError, ZarrVersion};

use super::{BytesCodecConfiguration, BytesCodecConfigurationV1, BytesDataTypeExt, Endianness};
use crate::array::{ArrayBytes, BytesRepresentation, CowBytes, DataType, FillValue};
use smallvec::SmallVec;
use std::num::NonZeroU64;
use zarrs_chunk_grid::IndexerError;
use zarrs_codec::plan::{
    ByteSource, ComputeCtx, DecodeRequest, Outcome, PartialEncodeState, PlanIndexer, PrepareDecode,
    SourceDecoder, SourceKey, Task, TaskHint, TaskTarget, TaskValue, WriteOp, decode_and_update,
    decode_whole, encode_hint, partial_encode_value, plan_decode_whole, plan_partial_encode_whole,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoInput, ArrayBytesDecodeIntoTarget, ArrayCodecTraits,
    ArrayToBytesCodecTraits, CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions,
    CodecSpecificOptions, CodecTraits, InvalidBytesLengthError, Resources,
    UnboundArrayToBytesCodecTraits, decode_into_array_bytes_target,
};
use zarrs_metadata::Configuration;
use zarrs_storage::byte_range::ByteRange;

/// A `bytes` codec implementation.
#[derive(Debug, Clone)]
pub struct BytesCodec {
    endian: Option<Endianness>,
}

/// The state of a chunk partially encoded by the `bytes` codec: it exists.
struct ChunkExists;

/// A `bytes` codec implementation bound to a data type and fill value.
#[derive(Debug, Clone)]
struct BytesCodecBound {
    endian: Option<Endianness>,
    /// Whether decoding is the identity on the encoded bytes.
    passthrough: bool,
    data_type: DataType,
    fill_value: FillValue,
}

impl Default for BytesCodec {
    fn default() -> Self {
        Self::new(Some(Endianness::native()))
    }
}

impl BytesCodec {
    /// Create a new `bytes` codec.
    ///
    /// `endian` is optional because an 8-bit type has no endianness.
    #[must_use]
    pub const fn new(endian: Option<Endianness>) -> Self {
        Self { endian }
    }

    /// Create a new `bytes` codec for little endian data.
    #[must_use]
    pub const fn little() -> Self {
        Self::new(Some(Endianness::Little))
    }

    /// Create a new `bytes` codec for big endian data.
    #[must_use]
    pub const fn big() -> Self {
        Self::new(Some(Endianness::Big))
    }

    /// Create a new `bytes` codec from configuration.
    ///
    /// # Errors
    /// Returns an error if the configuration is not supported.
    pub fn new_with_configuration(
        configuration: &BytesCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            BytesCodecConfiguration::V1(configuration) => Ok(Self::new(configuration.endian)),
            _ => Err(PluginCreateError::Other(
                "this bytes codec configuration variant is unsupported".to_string(),
            )),
        }
    }
}

impl CodecTraits for BytesCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        let configuration = BytesCodecConfiguration::V1(BytesCodecConfigurationV1 {
            endian: self.endian,
        });
        Some(configuration.into())
    }
}

impl UnboundArrayToBytesCodecTraits for BytesCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self as Arc<dyn UnboundArrayToBytesCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        if data_type.is_optional() {
            return Err(CodecCreateError::UnsupportedDataType(
                data_type,
                Self::aliases_v3().default_name.to_string(),
            ));
        }
        let passthrough = data_type.codec_bytes()?.is_decode_passthrough(self.endian);
        Ok(Arc::new(BytesCodecBound {
            endian: self.endian,
            passthrough,
            data_type,
            fill_value,
        }))
    }
}

impl BytesCodecBound {
    fn decoded_size(&self, shape: &[NonZeroU64]) -> Result<(u64, u64), CodecError> {
        let data_type_size = self.data_type.fixed_size().ok_or_else(|| {
            CodecError::UnsupportedDataType(
                self.data_type.clone(),
                BytesCodec::aliases_v3().default_name.to_string(),
            )
        })?;
        let num_elements = shape
            .iter()
            .try_fold(1u64, |count, d| count.checked_mul(d.get()))
            .ok_or("the decoded chunk element count overflows u64")?;
        let num_bytes = num_elements
            .checked_mul(data_type_size as u64)
            .ok_or("the decoded chunk size in bytes overflows u64")?;
        Ok((num_elements, num_bytes))
    }

    fn decoded_len(&self, shape: &[NonZeroU64]) -> Result<(u64, usize), CodecError> {
        let (num_elements, num_bytes) = self.decoded_size(shape)?;
        let len = usize::try_from(num_bytes)
            .map_err(|_| CodecError::from("the decoded chunk size in bytes overflows usize"))?;
        Ok((num_elements, len))
    }
}

impl ArrayCodecTraits for BytesCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }
}

impl zarrs_codec::ArrayToBytesCodecNoSubchunkingTraits for BytesCodecBound {}

impl ArrayToBytesCodecTraits for BytesCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let (num_elements, _) = self.decoded_len(shape)?;
        bytes.validate(num_elements, &self.data_type)?;
        let bytes = bytes.into_fixed()?;

        Ok(self.data_type.codec_bytes()?.encode(bytes, self.endian)?)
    }

    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let (num_elements, _) = self.decoded_len(shape)?;
        let bytes = if self.passthrough {
            bytes
        } else {
            self.data_type.codec_bytes()?.decode(bytes, self.endian)?
        };
        let bytes_decoded = ArrayBytes::Fixed(bytes);

        bytes_decoded.validate(num_elements, &self.data_type)?;

        Ok(bytes_decoded)
    }

    fn decode_into(
        &self,
        input: ArrayBytesDecodeIntoInput<'_>,
        shape: &[NonZeroU64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        match output_target {
            ArrayBytesDecodeIntoTarget::Fixed(output) if self.data_type.is_fixed() => {
                let (num_elements, expected_len) = self.decoded_len(shape)?;
                if output.num_elements() != num_elements {
                    return Err("the decoded element count does not match the output target".into());
                }
                let codec = self.data_type.codec_bytes()?;
                let passthrough = self.passthrough;
                let input = match input {
                    ArrayBytesDecodeIntoInput::Deferred(source)
                        if source.is_decode_into_efficient()
                            && (passthrough || codec.is_decode_in_place_efficient())
                            && source.decoded_representation()
                                == &BytesRepresentation::FixedSize(expected_len as u64) =>
                    {
                        if let Some(bytes) = output.as_mut_slice()
                            && bytes.len() == expected_len
                        {
                            if !passthrough {
                                // Validate endianness before the producer writes anything.
                                // `decode_in_place` must check it even for empty bytes.
                                codec.decode_in_place(&mut [], self.endian)?;
                            }
                            source.decode_into(bytes, options, resources)?;
                            return if passthrough {
                                Ok(())
                            } else {
                                Ok(codec.decode_in_place(bytes, self.endian)?)
                            };
                        }
                        ArrayBytesDecodeIntoInput::Deferred(source)
                    }
                    input @ (ArrayBytesDecodeIntoInput::Bytes(_)
                    | ArrayBytesDecodeIntoInput::Deferred(_)) => input,
                };

                // Keep owned intermediates intact when direct output is unsuitable.
                let bytes = input.into_bytes(options, resources)?;
                if bytes.len() != expected_len {
                    return Err(InvalidBytesLengthError::new(bytes.len(), expected_len).into());
                }
                if passthrough {
                    Ok(output.copy_from_slice(&bytes)?)
                } else if codec.is_decode_in_place_efficient() {
                    // Validate endianness before anything is written to the output.
                    codec.decode_in_place(&mut [], self.endian)?;
                    output.try_copy_from_slice_with(&bytes, |source, destination| {
                        destination.copy_from_slice(source);
                        Ok::<_, CodecError>(codec.decode_in_place(destination, self.endian)?)
                    })
                } else {
                    let decoded = codec.decode(bytes, self.endian)?;
                    Ok(output.copy_from_slice(&decoded)?)
                }
            }
            target @ (ArrayBytesDecodeIntoTarget::Fixed(_)
            | ArrayBytesDecodeIntoTarget::Optional(..)) => {
                let bytes = input.into_bytes(options, resources)?;
                let decoded = self.decode(bytes, shape, options, resources)?;
                decode_into_array_bytes_target(&decoded, target)
            }
        }
    }

    fn prepare_decode(
        self: Arc<Self>,
        source: ByteSource<'static>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<PrepareDecode, CodecError> {
        // Requests are planned from the source, as only their bytes are read and decoded
        Ok(PrepareDecode::Ready(Arc::new(SourceDecoder::new(
            self, source, shape,
        ))))
    }

    fn plan_decode<'a>(
        self: Arc<Self>,
        source: ByteSource<'a>,
        shape: &[NonZeroU64],
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        let Some(data_type_size) = self.data_type.fixed_size() else {
            return plan_decode_whole(self, source, shape, request, options);
        };
        if request.indexer.is_whole(shape) {
            return plan_decode_whole(self, source, shape, request, options);
        }
        let indexer = request.indexer.as_indexer();
        if indexer.dimensionality() != shape.len() {
            return Err(IndexerError::new_incompatible_dimensionality(
                indexer.dimensionality(),
                shape.len(),
            )
            .into());
        }

        // A subset of a chunk in memory is decoded directly into the target views as it is copied
        if indexer.as_array_subset().is_some()
            && matches!(request.target, TaskTarget::Views(_))
            && matches!(source.memory_slice(), Some(Ok(_)))
            && (self.passthrough || self.data_type.codec_bytes()?.is_decode_in_place_efficient())
        {
            indexer.validate(bytemuck::must_cast_slice(shape))?;
            if !self.passthrough {
                // Validate endianness before anything is written to the output.
                self.data_type
                    .codec_bytes()?
                    .decode_in_place(&mut [], self.endian)?;
            }
            let shape = shape.to_vec();
            return Ok(Task::compute(move |_| {
                let DecodeRequest { indexer, target } = request;
                let TaskTarget::Views(views) = target else {
                    unreachable!("the target is views")
                };
                let bytes = source.memory_slice().expect("the source is in memory")?;
                let subset = indexer
                    .as_indexer()
                    .as_array_subset()
                    .expect("an array subset");
                let shape = bytemuck::must_cast_slice(&shape);
                views.with_target(|target| match target {
                    ArrayBytesDecodeIntoTarget::Fixed(view) if self.passthrough => {
                        view.copy_from_subset(bytes, shape, subset)
                    }
                    ArrayBytesDecodeIntoTarget::Fixed(view) => {
                        let codec = self.data_type.codec_bytes()?;
                        view.try_copy_from_subset_with(
                            bytes,
                            shape,
                            subset,
                            |source, destination| {
                                destination.copy_from_slice(source);
                                Ok(codec.decode_in_place(destination, self.endian)?)
                            },
                        )
                    }
                    ArrayBytesDecodeIntoTarget::Optional(..) => {
                        Err(zarrs_codec::ExpectedFixedLengthBytesError.into())
                    }
                })?;
                Ok(Outcome::Done)
            }));
        }

        // Read only the bytes of the requested elements
        let ranges: SmallVec<[ByteRange; 1]> = indexer
            .iter_contiguous_byte_ranges(bytemuck::must_cast_slice(shape), data_type_size)?
            .map(ByteRange::new)
            .collect();
        let num_bytes = indexer.len() * data_type_size as u64;
        let shape = shape.to_vec();
        let options = *options;
        Ok(Task::read(source, ranges, move |reads, ctx| {
            if reads.iter().any(Option::is_none) {
                return decode_whole(&*self, None, &shape, request, &options, ctx.resources());
            }
            let num_reads = reads.len();
            let mut reads = reads.into_iter().flatten();
            let bytes = if num_reads == 1 {
                reads.next().expect("one read")
            } else {
                let mut bytes = Vec::with_capacity(usize::try_from(num_bytes).unwrap());
                reads.for_each(|read| bytes.extend_from_slice(&read));
                CowBytes::from(bytes)
            };
            let num_elements = request.indexer.as_indexer().len();
            let Some(num_elements) = NonZeroU64::new(num_elements) else {
                return Ok(match request.target {
                    TaskTarget::Views(_) => Outcome::Done,
                    TaskTarget::Value => {
                        Outcome::Value(TaskValue::Array(ArrayBytes::new_flen(Vec::new())))
                    }
                });
            };
            // The bytes codec decodes elementwise, so the requested elements decode as a 1D chunk
            match request.target {
                TaskTarget::Views(views) => {
                    views.with_target(|target| {
                        self.decode_into(
                            ArrayBytesDecodeIntoInput::Bytes(bytes),
                            &[num_elements],
                            target,
                            &options,
                            ctx.resources(),
                        )
                    })?;
                    Ok(Outcome::Done)
                }
                TaskTarget::Value => Ok(Outcome::Value(TaskValue::Array(self.decode(
                    bytes,
                    &[num_elements],
                    &options,
                    ctx.resources(),
                )?))),
            }
        })
        .with_hint(TaskHint {
            read_bytes: Some(num_bytes),
            compute_bytes: num_bytes,
        }))
    }

    fn plan_partial_encode<'a>(
        self: Arc<Self>,
        key: SourceKey,
        shape: &[NonZeroU64],
        indexer: PlanIndexer<'a>,
        bytes: ArrayBytes<'a>,
        state: Option<PartialEncodeState>,
        _set_partial: bool,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        // A partial write is cheaper than a whole read-modify-write even if the storage emulates it, as the bytes
        // codec writes elements in place
        let Some(data_type_size) = self.data_type.fixed_size() else {
            return Ok(plan_partial_encode_whole(
                self, key, shape, indexer, bytes, options,
            ));
        };
        indexer.validate(shape)?;
        bytes.validate(indexer.as_indexer().len(), &self.data_type)?;
        let (_, chunk_size) = self.decoded_len(shape)?;
        let hint = encode_hint(&bytes);
        let shape: SmallVec<[NonZeroU64; 4]> = shape.into();
        let options = *options;

        // The encoded elements are written in place if the chunk exists, otherwise the chunk is written in full
        let update = move |exists: bool, ctx: &ComputeCtx| {
            let op = if exists {
                let bytes = self
                    .data_type
                    .codec_bytes()?
                    .encode(bytes.into_fixed()?, self.endian)?;
                let mut offset_in = 0;
                let offset_values = indexer
                    .as_indexer()
                    .iter_contiguous_byte_ranges(bytemuck::must_cast_slice(&shape), data_type_size)?
                    .map(|range| {
                        let len = usize::try_from(range.end - range.start).unwrap();
                        let value = bytes.slice(offset_in..offset_in + len);
                        offset_in += len;
                        (range.start, value)
                    })
                    .collect();
                WriteOp::SetPartial(key, offset_values)
            } else {
                let chunk = decode_and_update(
                    &*self,
                    None,
                    &shape,
                    indexer.as_indexer(),
                    &bytes,
                    &options,
                    ctx.resources(),
                )?;
                if !options.store_empty_chunks() && chunk.is_fill_value(&self.fill_value) {
                    // The chunk remains missing
                    return Ok(Outcome::Value(TaskValue::None));
                }
                WriteOp::Set(
                    key,
                    self.encode(chunk, &shape, &options, ctx.resources())?
                        .into_static(),
                )
            };
            Ok(Outcome::Write {
                ops: smallvec::smallvec![op],
                then: partial_encode_value(Arc::new(ChunkExists)),
            })
        };
        let known_to_exist = state.is_some_and(|state| state.is::<ChunkExists>());
        let task = if known_to_exist {
            Task::compute(move |ctx| update(true, ctx))
        } else if chunk_size == 0 {
            // An empty chunk is written in full
            Task::compute(move |ctx| update(false, ctx))
        } else {
            // Read the first byte of the chunk to check if it exists
            Task::read(
                ByteSource::external(key),
                [ByteRange::FromStart(0, Some(1))],
                move |mut reads, ctx| update(reads.pop().flatten().is_some(), ctx),
            )
        };
        Ok(task.with_hint(hint))
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        let (_, size) = self.decoded_size(shape)?;
        Ok(BytesRepresentation::FixedSize(size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::{ArrayBytesFixedDisjointView, ArraySubset, data_type};
    use unsafe_cell_slice::UnsafeCellSlice;
    use zarrs_codec::plan::{TargetViews, drive_memory_one};

    fn decode_into_view(
        codec: &Arc<dyn ArrayToBytesCodecTraits>,
        input: ArrayBytesDecodeIntoInput<'_>,
        chunk_shape: &[NonZeroU64],
        array_shape: &[u64],
    ) -> (Vec<u16>, Result<(), CodecError>) {
        let num_elements = usize::try_from(array_shape.iter().product::<u64>()).unwrap();
        let mut output = vec![0u16; num_elements];
        let result = {
            let mut view = unsafe {
                // SAFETY: the only view of output, used synchronously.
                ArrayBytesFixedDisjointView::new(
                    UnsafeCellSlice::new(bytemuck::cast_slice_mut(&mut output)),
                    size_of::<u16>(),
                    array_shape,
                    ArraySubset::new_with_shape(bytemuck::must_cast_slice::<_, u64>(chunk_shape)),
                )
            }
            .unwrap();
            codec.decode_into(
                input,
                chunk_shape,
                (&mut view).into(),
                &CodecOptions::default(),
                &Resources::default(),
            )
        };
        (output, result)
    }

    #[test]
    fn decode_into_endianness_and_noncontiguous_views() {
        let values = [1u16, 2, 3, 4];
        let shape = [NonZeroU64::new(2).unwrap(); 2];
        for (endian, encoded) in [
            (Endianness::Little, values.map(u16::to_le_bytes).concat()),
            (Endianness::Big, values.map(u16::to_be_bytes).concat()),
        ] {
            let codec = BytesCodec::new(Some(endian))
                .with_context(
                    data_type::uint16(),
                    FillValue::from(0u16),
                    &CodecSpecificOptions::default(),
                )
                .unwrap();
            for (array_shape, expected) in [
                ([2, 2], values.to_vec()),
                ([2, 4], vec![1, 2, 0, 0, 3, 4, 0, 0]),
            ] {
                let (output, result) = decode_into_view(
                    &codec,
                    CowBytes::from(encoded.as_slice()).into(),
                    &shape,
                    &array_shape,
                );
                result.unwrap();
                assert_eq!(output, expected);

                let (output, result) = decode_into_view(
                    &codec,
                    CowBytes::from(&encoded[1..]).into(),
                    &shape,
                    &array_shape,
                );
                assert!(result.is_err());
                assert!(output.iter().all(|&value| value == 0));
            }
        }
        let codec = BytesCodec::new(None)
            .with_context(
                data_type::uint16(),
                FillValue::from(0u16),
                &CodecSpecificOptions::default(),
            )
            .unwrap();
        for array_shape in [[2, 2], [2, 4]] {
            let (output, result) = decode_into_view(
                &codec,
                CowBytes::from(&[1; 8][..]).into(),
                &shape,
                &array_shape,
            );
            assert!(result.is_err());
            assert!(output.iter().all(|&value| value == 0));
        }
    }

    /// A subset of a chunk in memory is decoded as it is copied into views, in either endianness.
    #[test]
    fn plan_decode_memory_subset_into_views() {
        let values: Vec<u16> = (0..16).collect();
        let shape = [NonZeroU64::new(4).unwrap(); 2];
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..4]);
        let decode = |endian, encoded: &[u8]| {
            let codec = BytesCodec::new(endian)
                .with_context(
                    data_type::uint16(),
                    FillValue::from(0u16),
                    &CodecSpecificOptions::default(),
                )
                .unwrap();
            // Decode into the right of a 2x4 output, so the view is not contiguous
            let mut output = vec![0u16; 8];
            let result = {
                let mut view = unsafe {
                    // SAFETY: the only view of output, used synchronously.
                    ArrayBytesFixedDisjointView::new(
                        UnsafeCellSlice::new(bytemuck::cast_slice_mut(&mut output)),
                        size_of::<u16>(),
                        &[2, 4],
                        ArraySubset::new_with_ranges(&[0..2, 1..4]),
                    )
                }
                .unwrap();
                let target = ArrayBytesDecodeIntoTarget::from(&mut view);
                codec
                    .plan_decode(
                        ByteSource::memory(CowBytes::from(encoded)),
                        &shape,
                        DecodeRequest {
                            indexer: PlanIndexer::Borrowed(&subset),
                            target: TaskTarget::Views(
                                unsafe { TargetViews::from_target(&target) }.unwrap(),
                            ),
                        },
                        &CodecOptions::default(),
                    )
                    .and_then(|task| drive_memory_one(task, &Resources::default()))
                    .map(|_| ())
            };
            (output, result)
        };
        for (endian, encoded) in [
            (
                Endianness::Little,
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                Endianness::Big,
                values.iter().flat_map(|v| v.to_be_bytes()).collect(),
            ),
        ] {
            let (output, result) = decode(Some(endian), &encoded);
            result.unwrap();
            assert_eq!(output, [0, 5, 6, 7, 0, 9, 10, 11]);
        }
        // A multi-byte data type needs an endianness, which is checked before the output is written
        let (output, result) = decode(None, &[1; 32]);
        assert!(result.is_err());
        assert!(output.iter().all(|&value| value == 0));
    }

    #[cfg(feature = "zstd")]
    #[test]
    fn decode_deferred_zstd_input() {
        use zarrs_codec::{BytesDecodeSource, BytesToBytesCodecTraits};

        let values = [1u16, 2, 3, 4];
        let shape = [NonZeroU64::new(2).unwrap(); 2];
        let producer = crate::array::codec::ZstdCodec::new(1, false);
        let options = CodecOptions::default();
        for endian in [Endianness::Little, Endianness::Big] {
            let codec = BytesCodec::new(Some(endian))
                .with_context(
                    data_type::uint16(),
                    FillValue::from(0u16),
                    &CodecSpecificOptions::default(),
                )
                .unwrap();
            let encoded = if endian == Endianness::Little {
                values.map(u16::to_le_bytes).concat()
            } else {
                values.map(u16::to_be_bytes).concat()
            };
            let compressed = producer
                .encode(encoded.into(), &options, &Resources::default())
                .unwrap();
            for representation in [
                BytesRepresentation::FixedSize(8),
                BytesRepresentation::BoundedSize(8),
            ] {
                let input = ArrayBytesDecodeIntoInput::Deferred(BytesDecodeSource::new(
                    &producer,
                    compressed.clone(),
                    representation,
                ));
                let (output, result) = decode_into_view(&codec, input, &shape, &[2, 2]);
                result.unwrap();
                assert_eq!(output, values);
            }
        }
    }

    #[test]
    fn decoded_size_overflow() {
        let codec = BytesCodec::default()
            .with_context(
                data_type::uint16(),
                FillValue::from(0u16),
                &CodecSpecificOptions::default(),
            )
            .unwrap();
        let options = CodecOptions::default();
        for shape in [
            vec![NonZeroU64::new(u64::MAX).unwrap()],
            vec![NonZeroU64::new(1 << 32).unwrap(); 2],
        ] {
            assert!(codec.encoded_representation(&shape).is_err());
            assert!(
                codec
                    .encode(
                        ArrayBytes::new_flen(vec![0; 2]),
                        &shape,
                        &options,
                        &Resources::default()
                    )
                    .is_err()
            );
            assert!(
                codec
                    .decode(
                        CowBytes::from(&[0; 2][..]),
                        &shape,
                        &options,
                        &Resources::default()
                    )
                    .is_err()
            );
        }
    }

    #[test]
    fn unsupported_data_types() {
        let options = CodecSpecificOptions::default();
        assert!(
            BytesCodec::new(None)
                .with_context(
                    data_type::uint8().to_optional(),
                    FillValue::from(0u8).into_optional(),
                    &options
                )
                .is_err()
        );
        assert!(
            BytesCodec::new(None)
                .with_context(
                    data_type::bytes(),
                    FillValue::from(Vec::<u8>::new()),
                    &options
                )
                .is_err()
        );
    }
}
