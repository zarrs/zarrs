// Note: No validation that this codec is created *without* a specified endianness for multi-byte data types.

use std::sync::Arc;

use zarrs_plugin::{ExtensionAliasesV3, PluginCreateError, ZarrVersion};

use super::{
    BytesCodecConfiguration, BytesCodecConfigurationV1, BytesDataTypeSupport, BytesDataTypeTraits,
    Endianness,
};
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
    ArrayBytesDecodeIntoInput, ArrayBytesDecodeIntoTarget, ArrayBytesFixedDisjointView,
    ArrayCodecTraits, ArrayToBytesCodecTraits, CodecCreateError, CodecError, CodecMetadataOptions,
    CodecOptions, CodecSpecificOptions, CodecTraits, ComputeResources, InvalidBytesLengthError,
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
    /// Whether elements are decoded efficiently in place, and the endianness is valid for decoding in place.
    decode_in_place: bool,
    support: BytesDataTypeSupport,
    fill_value: Option<FillValue>,
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
        fill_value: Option<FillValue>,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        if data_type.is_optional() {
            return Err(CodecCreateError::UnsupportedDataType(
                data_type,
                Self::aliases_v3().default_name.to_string(),
            ));
        }
        let support = BytesDataTypeSupport::new(&data_type)?;
        let codec = support.get();
        let passthrough = codec.is_decode_passthrough(self.endian);
        // `decode_in_place` checks the endianness even for empty bytes
        let decode_in_place = !passthrough
            && codec.is_decode_in_place_efficient()
            && codec.decode_in_place(&mut [], self.endian).is_ok();
        Ok(Arc::new(BytesCodecBound {
            endian: self.endian,
            passthrough,
            decode_in_place,
            support,
            fill_value,
        }))
    }
}

impl BytesCodecBound {
    fn decoded_size(&self, shape: &[NonZeroU64]) -> Result<(u64, u64), CodecError> {
        let data_type_size = self.support.data_type().fixed_size().ok_or_else(|| {
            CodecError::UnsupportedDataType(
                self.support.data_type().clone(),
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

    /// How encoded elements are decoded as they are copied into an output.
    ///
    /// Returns [`None`] if the elements are not decoded efficiently as they are copied, or the endianness is invalid
    /// (so decoding fails before anything is written to the output).
    fn in_place(&self) -> Option<InPlace<'_>> {
        if self.passthrough {
            Some(InPlace::Copy)
        } else if self.decode_in_place {
            Some(InPlace::Decode(self.support.get(), self.endian))
        } else {
            None
        }
    }

    /// Plan the decoding of a subset of a chunk in memory into target views, which decodes it as it is copied.
    ///
    /// The request must be for an array subset into views, `source` must be in memory, and the elements must be
    /// [copied in place](Self::copy_in_place).
    fn plan_decode_memory_subset<'a>(
        self: Arc<Self>,
        source: ByteSource<'a>,
        shape: &[NonZeroU64],
        request: DecodeRequest<'a>,
    ) -> Result<Task<'a>, CodecError> {
        let indexer = request.indexer.as_indexer();
        indexer.validate(bytemuck::must_cast_slice(shape))?;
        let shape: SmallVec<[NonZeroU64; 4]> = shape.into();
        Ok(Task::compute(move |_| {
            let DecodeRequest { indexer, target } = request;
            let TaskTarget::Views(views) = target else {
                unreachable!("the target is views")
            };
            let bytes = source.memory_slice().expect("the source is in memory")?;
            let subset = indexer
                .as_indexer()
                .as_array_subset()
                .expect("an array subset");
            let in_place = self.in_place().expect("decoded in place");
            let shape = bytemuck::must_cast_slice(&shape);
            views.with_target(|target| {
                let view = fixed_view(target)?;
                match in_place {
                    InPlace::Copy => view.copy_from_subset(bytes, shape, subset),
                    InPlace::Decode(codec, endian) => view.try_copy_from_subset_with(
                        bytes,
                        shape,
                        subset,
                        decode_copy(codec, endian),
                    ),
                }
            })?;
            Ok(Outcome::Done)
        }))
    }
}

/// How encoded elements are decoded as they are copied into an output.
#[derive(Clone, Copy)]
enum InPlace<'a> {
    /// The encoded elements are the decoded elements.
    Copy,
    /// The elements are decoded in place after they are copied.
    Decode(&'a dyn BytesDataTypeTraits, Option<Endianness>),
}

/// Copy a contiguous region of encoded elements into an output and decode them in place.
fn decode_copy(
    codec: &dyn BytesDataTypeTraits,
    endian: Option<Endianness>,
) -> impl Fn(&[u8], &mut [u8]) -> Result<(), CodecError> {
    move |source, destination| {
        destination.copy_from_slice(source);
        Ok(codec.decode_in_place(destination, endian)?)
    }
}

/// The view of a fixed-length `target`.
fn fixed_view(
    target: ArrayBytesDecodeIntoTarget<'_>,
) -> Result<&mut ArrayBytesFixedDisjointView<'_>, CodecError> {
    match target {
        ArrayBytesDecodeIntoTarget::Fixed(view) => Ok(view),
        ArrayBytesDecodeIntoTarget::Optional(..) => {
            Err(zarrs_codec::ExpectedFixedLengthBytesError.into())
        }
    }
}

impl ArrayCodecTraits for BytesCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        self.support.data_type()
    }

    fn fill_value(&self) -> Option<&FillValue> {
        self.fill_value.as_ref()
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
        _resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        let (num_elements, _) = self.decoded_len(shape)?;
        bytes.validate(num_elements, self.support.data_type())?;
        let bytes = bytes.into_fixed()?;

        Ok(self.support.get().encode(bytes, self.endian)?)
    }

    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let (num_elements, _) = self.decoded_len(shape)?;
        let bytes = if self.passthrough {
            bytes
        } else {
            self.support.get().decode(bytes, self.endian)?
        };
        let bytes_decoded = ArrayBytes::Fixed(bytes);

        bytes_decoded.validate(num_elements, self.support.data_type())?;

        Ok(bytes_decoded)
    }

    fn decode_into(
        &self,
        input: ArrayBytesDecodeIntoInput<'_>,
        shape: &[NonZeroU64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        match output_target {
            ArrayBytesDecodeIntoTarget::Fixed(output) if self.support.data_type().is_fixed() => {
                let (num_elements, expected_len) = self.decoded_len(shape)?;
                if output.num_elements() != num_elements {
                    return Err("the decoded element count does not match the output target".into());
                }
                let in_place = self.in_place();
                let input = match input {
                    ArrayBytesDecodeIntoInput::Deferred(source)
                        if source.is_decode_into_efficient()
                            && in_place.is_some()
                            && source.decoded_representation()
                                == &BytesRepresentation::FixedSize(expected_len as u64) =>
                    {
                        if let Some(bytes) = output.as_mut_slice()
                            && bytes.len() == expected_len
                        {
                            source.decode_into(bytes, options, resources)?;
                            return match in_place {
                                Some(InPlace::Decode(codec, endian)) => {
                                    Ok(codec.decode_in_place(bytes, endian)?)
                                }
                                _ => Ok(()),
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
                match in_place {
                    Some(InPlace::Copy) => Ok(output.copy_from_slice(&bytes)?),
                    Some(InPlace::Decode(codec, endian)) => {
                        output.try_copy_from_slice_with(&bytes, decode_copy(codec, endian))
                    }
                    None => {
                        let decoded = self.support.get().decode(bytes, self.endian)?;
                        Ok(output.copy_from_slice(&decoded)?)
                    }
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
        let Some(data_type_size) = self.support.data_type().fixed_size() else {
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

        if indexer.as_array_subset().is_some()
            && matches!(request.target, TaskTarget::Views(_))
            && matches!(source.memory_slice(), Some(Ok(_)))
            && self.in_place().is_some()
        {
            return self.plan_decode_memory_subset(source, shape, request);
        }

        // Read only the bytes of the requested elements
        let ranges: SmallVec<[ByteRange; 1]> = indexer
            .iter_contiguous_byte_ranges(bytemuck::must_cast_slice(shape), data_type_size)?
            .map(ByteRange::new)
            .collect();
        let num_bytes = indexer.len() * data_type_size as u64;
        let shape: SmallVec<[NonZeroU64; 4]> = shape.into();
        let options = *options;
        Ok(Task::read(source, ranges, move |reads, ctx| {
            if reads.iter().any(Option::is_none) {
                return decode_whole(&*self, None, &shape, request, &options, ctx.resources());
            }
            // Several reads are decoded directly into the target views as they are copied
            if reads.len() > 1
                && let Some(in_place) = self.in_place()
                && let TaskTarget::Views(views) = request.target
            {
                let reads = reads.iter().flatten().map(|read| &**read);
                views.with_target(|target| {
                    let view = fixed_view(target)?;
                    match in_place {
                        InPlace::Copy => Ok(view.copy_from_slices(reads)?),
                        InPlace::Decode(codec, endian) => {
                            view.try_copy_from_slices_with(reads, decode_copy(codec, endian))
                        }
                    }
                })?;
                return Ok(Outcome::Done);
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
        if indexer.is_whole(shape) {
            return Ok(plan_partial_encode_whole(
                self, key, shape, indexer, bytes, options,
            ));
        }

        // A partial write is cheaper than a whole read-modify-write even if the storage emulates it, as the bytes
        // codec writes elements in place
        let Some(data_type_size) = self.support.data_type().fixed_size() else {
            return Ok(plan_partial_encode_whole(
                self, key, shape, indexer, bytes, options,
            ));
        };
        indexer.validate(shape)?;
        bytes.validate(indexer.as_indexer().len(), self.support.data_type())?;
        let (_, chunk_size) = self.decoded_len(shape)?;
        let hint = encode_hint(&bytes);
        let shape: SmallVec<[NonZeroU64; 4]> = shape.into();
        let options = *options;

        // The encoded elements are written in place if the chunk exists, otherwise the chunk is written in full
        let update = move |exists: bool, ctx: &ComputeCtx| {
            let op = if exists {
                let bytes = self
                    .support
                    .get()
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
                if !options.store_empty_chunks()
                    && self
                        .fill_value
                        .as_ref()
                        .is_some_and(|fill| chunk.is_fill_value(fill))
                {
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
            // Request an empty range to check whether the chunk exists
            Task::read(
                ByteSource::external(key),
                [ByteRange::FromStart(0, Some(0))],
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
    use crate::array::{ArraySubset, data_type};
    use unsafe_cell_slice::UnsafeCellSlice;
    use zarrs_codec::Resources;
    use zarrs_codec::plan::{ReadBytes, SyncIo, TargetViews, drive_blocking_one};
    use zarrs_storage::{Bytes, StorageError};

    /// The backend of a drive with one value.
    struct ValueIo(Bytes);

    impl SyncIo for ValueIo {
        fn read(
            &self,
            _key: SourceKey,
            ranges: &[ByteRange],
        ) -> Result<Option<ReadBytes>, StorageError> {
            let len = self.0.len() as u64;
            Ok(Some(
                ranges
                    .iter()
                    .map(|range| self.0.slice(range.to_range_usize(len)))
                    .collect(),
            ))
        }

        fn write(&self, _op: WriteOp<'_>) -> Result<(), StorageError> {
            unimplemented!("the value is only read")
        }
    }

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
                &ComputeResources::default(),
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
                    Some(FillValue::from(0u16)),
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
                Some(FillValue::from(0u16)),
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
    fn plan_decode_subset_into_views() {
        let values: Vec<u16> = (0..16).collect();
        let shape = [NonZeroU64::new(4).unwrap(); 2];
        let subset = ArraySubset::new_with_ranges(&[1..3, 1..4]);
        let decode = |endian, encoded: &[u8], in_memory: bool| {
            let codec = BytesCodec::new(endian)
                .with_context(
                    data_type::uint16(),
                    Some(FillValue::from(0u16)),
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
                // A subset of a chunk in storage is read as a byte range per row
                let io = ValueIo(Bytes::copy_from_slice(encoded));
                let source = if in_memory {
                    ByteSource::memory(CowBytes::from(encoded))
                } else {
                    ByteSource::external(SourceKey(0))
                };
                codec
                    .plan_decode(
                        source,
                        &shape,
                        DecodeRequest {
                            indexer: PlanIndexer::Borrowed(&subset),
                            target: TaskTarget::Views(
                                unsafe { TargetViews::from_target(&target) }.unwrap(),
                            ),
                        },
                        &CodecOptions::default(),
                    )
                    .and_then(|task| drive_blocking_one(task, &io, &Resources::default()))
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
            for in_memory in [true, false] {
                let (output, result) = decode(Some(endian), &encoded, in_memory);
                result.unwrap();
                assert_eq!(output, [0, 5, 6, 7, 0, 9, 10, 11]);
            }
        }
        // A multi-byte data type needs an endianness, which is checked before the output is written
        for in_memory in [true, false] {
            let (output, result) = decode(None, &[1; 32], in_memory);
            assert!(result.is_err());
            assert!(output.iter().all(|&value| value == 0));
        }
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
                    Some(FillValue::from(0u16)),
                    &CodecSpecificOptions::default(),
                )
                .unwrap();
            let encoded = if endian == Endianness::Little {
                values.map(u16::to_le_bytes).concat()
            } else {
                values.map(u16::to_be_bytes).concat()
            };
            let compressed = producer
                .encode(encoded.into(), &options, &ComputeResources::default())
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
                Some(FillValue::from(0u16)),
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
                        &ComputeResources::default()
                    )
                    .is_err()
            );
            assert!(
                codec
                    .decode(
                        CowBytes::from(&[0; 2][..]),
                        &shape,
                        &options,
                        &ComputeResources::default()
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
                    Some(FillValue::from(0u8).into_optional()),
                    &options
                )
                .is_err()
        );
        assert!(
            BytesCodec::new(None)
                .with_context(
                    data_type::bytes(),
                    Some(FillValue::from(Vec::<u8>::new())),
                    &options
                )
                .is_err()
        );
    }
}
