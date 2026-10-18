//! An array to bytes codec formed by joining an array to array sequence, array to bytes, and bytes to bytes sequence of codecs.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_plugin::{ExtensionName, MaybeSend, ZarrVersion};

use crate::array::{
    ArrayBytes, BytesRepresentation, ChunkGrid, ChunkShape, CowBytes, DataType, FillValue,
};
use zarrs_codec::plan::{
    ArrayToArrayDecodePlan, ByteSource, ComputeCtx, DecodeRequest, Join, MissingDecoder, Outcome,
    PartialEncodeState, PlanIndexer, PrepareDecode, PreparedDecoder, SourceKey, SourceStep, Task,
    TaskTarget, TaskValue, decode_whole, encode_hint, encoded_value, local_subchunk_grids,
    local_subchunk_grids_value, place_decoded, plan_decode_whole, plan_encode_whole,
    plan_partial_encode_whole, subchunk_location_value, whole_chunk_hint,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoInput, ArrayBytesDecodeIntoTarget, ArrayCodecTraits,
    ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesDecodeSource, BytesToBytesCodecTraits,
    ChunkGridDecoded, ChunkGridDecodedRef, ChunkGridEncoded, Codec, CodecCreateError, CodecError,
    CodecMetadataOptions, CodecOptions, CodecSpecificOptions, CodecTraits, Resources,
    UnboundArrayToArrayCodecTraits, UnboundArrayToBytesCodecTraits, decode_into_array_bytes_target,
};
use zarrs_metadata::Configuration;
use zarrs_metadata::v3::MetadataV3;
use zarrs_storage::byte_range::ByteRange;

/// The decoded shape of each array-to-array codec of a chain, outermost first, then the shape they encode.
type EncodedShapes = smallvec::SmallVec<[ChunkShape; 4]>;
/// The encoded representation of the array-to-bytes codec of a chain, then of each bytes-to-bytes codec.
type BytesRepresentations = smallvec::SmallVec<[BytesRepresentation; 4]>;

/// The error of an array-to-array codec that plans partial encodes not mapping a partial encode.
fn not_partially_encoded() -> CodecError {
    CodecError::Other(
        "an array-to-array codec that plans partial encodes did not map a partial encode"
            .to_string(),
    )
}

fn cumulative_boundaries(edge_lengths: &[NonZeroU64]) -> Vec<u64> {
    let mut offset = 0;
    let mut boundaries = Vec::with_capacity(edge_lengths.len() + 1);
    boundaries.push(offset);
    for edge_length in edge_lengths {
        offset += edge_length.get();
        boundaries.push(offset);
    }
    boundaries
}

fn validate_subchunk_grid_refines_chunk_grid(
    chunk_grid: &ChunkGrid,
    subchunk_grid: &ChunkGrid,
) -> Result<(), ChunkGridCreateError> {
    if chunk_grid.dimensionality() != subchunk_grid.dimensionality() {
        return Err(ChunkGridCreateError::new(format!(
            "subchunk grid dimensionality {} does not match chunk grid dimensionality {}",
            subchunk_grid.dimensionality(),
            chunk_grid.dimensionality()
        )));
    }

    for dim in 0..chunk_grid.dimensionality() {
        let chunk_boundaries = cumulative_boundaries(&chunk_grid.chunk_edge_lengths(dim)?);
        let subchunk_boundaries = cumulative_boundaries(&subchunk_grid.chunk_edge_lengths(dim)?);
        if chunk_boundaries.last() != subchunk_boundaries.last() {
            return Err(ChunkGridCreateError::Other(format!(
                "subchunk grid edge lengths in dimension {dim} do not sum to chunk grid edge lengths"
            )));
        }
        if !chunk_boundaries
            .iter()
            .all(|boundary| subchunk_boundaries.binary_search(boundary).is_ok())
        {
            return Err(ChunkGridCreateError::Other(format!(
                "subchunk grid boundaries in dimension {dim} do not align with chunk grid boundaries"
            )));
        }
    }

    Ok(())
}

/// A codec chain is a sequence of array to array, a bytes to bytes, and a sequence of array to bytes codecs.
///
/// A codec chain partial decoder may insert a cache.
/// For example, the output of the `blosc`/`gzip` codecs should be cached since they read and decode an entire chunk.
/// If decoding (i.e. going backwards through a codec chain), then a cache may be inserted
///    - following the last codec with `partial_decode` false, otherwise
///    - preceding the first codec with `partial_decode` true and `partial_read` false.
#[derive(Debug, Clone)]
pub struct CodecChain {
    array_to_array: Vec<Arc<dyn UnboundArrayToArrayCodecTraits>>,
    array_to_bytes: Arc<dyn UnboundArrayToBytesCodecTraits>,
    bytes_to_bytes: Vec<Arc<dyn BytesToBytesCodecTraits>>,
}

/// A codec chain bound to an array data type and fill value.
#[derive(Debug, Clone)]
pub struct CodecChainBound {
    array_to_array: Vec<Arc<dyn ArrayToArrayCodecTraits>>,
    array_to_bytes: Arc<dyn ArrayToBytesCodecTraits>,
    bytes_to_bytes: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    /// The array-to-bytes and bytes-to-bytes codecs of the chain, if it has array-to-array codecs (for planning).
    bytes_chain: Option<Arc<CodecChainBound>>,
}

impl CodecChain {
    /// Bind this codec chain to a decoded data type, fill value, and codec-specific options.
    ///
    /// Each codec in the chain may read its own options type from `codec_specific_options`.
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if any codec cannot be bound to the derived context.
    pub fn with_context(
        &self,
        mut data_type: DataType,
        mut fill_value: FillValue,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<CodecChainBound>, CodecCreateError> {
        let mut array_to_array = Vec::with_capacity(self.array_to_array.len());
        for codec in &self.array_to_array {
            let bound = codec.with_context(
                data_type.clone(),
                fill_value.clone(),
                codec_specific_options,
            )?;
            data_type = bound.encoded_data_type().clone();
            fill_value = bound.encoded_fill_value().clone();
            array_to_array.push(bound);
        }
        let array_to_bytes =
            self.array_to_bytes
                .with_context(data_type, fill_value, codec_specific_options)?;
        let bytes_to_bytes = self
            .bytes_to_bytes
            .iter()
            .map(|codec| {
                codec
                    .clone()
                    .with_codec_specific_options(codec_specific_options)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let bytes_chain = (!array_to_array.is_empty()).then(|| {
            Arc::new(CodecChainBound {
                array_to_array: vec![],
                array_to_bytes: array_to_bytes.clone(),
                bytes_to_bytes: bytes_to_bytes.clone(),
                bytes_chain: None,
            })
        });

        Ok(Arc::new(CodecChainBound {
            array_to_array,
            array_to_bytes,
            bytes_to_bytes,
            bytes_chain,
        }))
    }

    /// Create a new codec chain.
    #[must_use]
    pub fn new(
        array_to_array: Vec<Arc<dyn UnboundArrayToArrayCodecTraits>>,
        array_to_bytes: Arc<dyn UnboundArrayToBytesCodecTraits>,
        bytes_to_bytes: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    ) -> Self {
        Self {
            array_to_array,
            array_to_bytes,
            bytes_to_bytes,
        }
    }

    /// Create a new codec chain from a list of metadata.
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if:
    ///  - a codec could not be created,
    ///  - no array to bytes codec is supplied, or
    ///  - more than one array to bytes codec is supplied.
    pub fn from_metadata(metadatas: &[MetadataV3]) -> Result<Self, CodecCreateError> {
        let mut array_to_array: Vec<Arc<dyn UnboundArrayToArrayCodecTraits>> = vec![];
        let mut array_to_bytes: Option<Arc<dyn UnboundArrayToBytesCodecTraits>> = None;
        let mut bytes_to_bytes: Vec<Arc<dyn BytesToBytesCodecTraits>> = vec![];
        for metadata in metadatas {
            let codec = match Codec::from_metadata(metadata) {
                Ok(codec) => Ok(codec),
                Err(err) => {
                    if metadata.must_understand() {
                        Err(err)
                    } else {
                        continue;
                    }
                }
            }?;

            match codec {
                Codec::ArrayToArray(codec) => {
                    array_to_array.push(codec);
                }
                Codec::ArrayToBytes(codec) => {
                    if array_to_bytes.is_none() {
                        array_to_bytes = Some(codec);
                    } else {
                        return Err(CodecCreateError::from("multiple array to bytes codecs"));
                    }
                }
                Codec::BytesToBytes(codec) => {
                    bytes_to_bytes.push(codec);
                }
            }
        }

        array_to_bytes.map_or_else(
            || Err(CodecCreateError::from("missing array to bytes codec")),
            |array_to_bytes| Ok(Self::new(array_to_array, array_to_bytes, bytes_to_bytes)),
        )
    }

    /// Create codec chain metadata.
    ///
    /// # Panics
    /// Panics if any codec does not have a V3 name.
    #[must_use]
    pub fn create_metadatas(&self, options: &CodecMetadataOptions) -> Vec<MetadataV3> {
        let mut metadatas =
            Vec::with_capacity(self.array_to_array.len() + 1 + self.bytes_to_bytes.len());
        for codec in &self.array_to_array {
            if let Some(configuration) = codec.configuration_v3(options) {
                let name = codec
                    .name_v3()
                    .expect("codec must have a V3 name")
                    .into_owned();
                metadatas.push(MetadataV3::new_with_configuration(name, configuration));
            }
        }
        {
            let codec = &self.array_to_bytes;
            if let Some(configuration) = codec.configuration_v3(options) {
                let name = codec
                    .name_v3()
                    .expect("codec must have a V3 name")
                    .into_owned();
                metadatas.push(MetadataV3::new_with_configuration(name, configuration));
            }
        }
        for codec in &self.bytes_to_bytes {
            if let Some(configuration) = codec.configuration_v3(options) {
                let name = codec
                    .name_v3()
                    .expect("codec must have a V3 name")
                    .into_owned();
                metadatas.push(MetadataV3::new_with_configuration(name, configuration));
            }
        }
        metadatas
    }

    /// Get the array to array codecs
    #[must_use]
    pub fn array_to_array_codecs(&self) -> &[Arc<dyn UnboundArrayToArrayCodecTraits>] {
        &self.array_to_array
    }

    /// Get the array to bytes codec
    #[must_use]
    pub fn array_to_bytes_codec(&self) -> &Arc<dyn UnboundArrayToBytesCodecTraits> {
        &self.array_to_bytes
    }

    /// Get the bytes to bytes codecs
    #[must_use]
    pub fn bytes_to_bytes_codecs(&self) -> &[Arc<dyn BytesToBytesCodecTraits>] {
        &self.bytes_to_bytes
    }
}

/// A prepared decoder of a chain with array-to-array codecs, which maps each request through the array-to-array
/// codecs to the prepared decoder of the rest of the chain.
#[derive(Debug)]
struct ArrayToArrayDecoder {
    chain: Arc<CodecChainBound>,
    shapes: EncodedShapes,
    inner: Arc<dyn PreparedDecoder>,
}

impl PreparedDecoder for ArrayToArrayDecoder {
    fn plan_decode<'a>(
        self: Arc<Self>,
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        self.chain
            .plan_decode_array_to_array(&self.shapes, request, options, |_, request| {
                self.inner.clone().plan_decode(request, options)
            })
    }

    fn exists(&self) -> Option<bool> {
        self.inner.exists()
    }

    fn size_held(&self) -> usize {
        self.inner.size_held()
    }
}

impl CodecChainBound {
    /// Plan decoding through the array-to-array codecs, then the rest of the chain with `plan_inner`, which plans a
    /// request of the encoded shape of the array-to-array codecs.
    ///
    /// `shapes` are the [`encoded_shapes`](Self::encoded_shapes) of the chunk.
    fn plan_decode_array_to_array<'a>(
        &self,
        shapes: &[ChunkShape],
        request: DecodeRequest<'a>,
        options: &CodecOptions,
        plan_inner: impl FnOnce(&[NonZeroU64], DecodeRequest<'a>) -> Result<Task<'a>, CodecError>,
    ) -> Result<Task<'a>, CodecError> {
        // Map the request through each array-to-array codec, outermost first
        let mut request = request;
        let mut finishes = Vec::new();
        for (codec, decoded_shape) in std::iter::zip(&self.array_to_array, shapes) {
            match codec.plan_decode_request(decoded_shape, request)? {
                ArrayToArrayDecodePlan::Forward(inner) => request = inner,
                ArrayToArrayDecodePlan::Decode {
                    indexer,
                    shape,
                    target,
                } => {
                    finishes.push((codec.clone(), shape, target));
                    request = DecodeRequest {
                        indexer,
                        target: TaskTarget::Value,
                    };
                }
            }
        }
        let inner = plan_inner(shapes.last().expect("nonempty"), request)?;
        if finishes.is_empty() {
            return Ok(inner);
        }

        // Decode the decoded value with each codec that decodes, innermost first
        let options = *options;
        Ok(Task::compute(move |_| {
            Ok(Outcome::Fork {
                children: vec![inner],
                join: Join::new(move |mut values, ctx| {
                    let Some(TaskValue::Array(mut bytes)) = values.pop() else {
                        return Err(CodecError::Other(
                            "expected decoded array bytes".to_string(),
                        ));
                    };
                    let mut finishes = finishes.into_iter().rev().peekable();
                    while let Some((codec, shape, target)) = finishes.next() {
                        bytes = codec.decode(bytes, &shape, &options, ctx.resources())?;
                        // Only the outermost codec that decodes has the target of the request
                        if finishes.peek().is_none() {
                            return place_decoded(bytes, target);
                        }
                    }
                    unreachable!("at least one codec decodes")
                }),
            })
        }))
    }

    /// Strip the windows of the outermost bytes-to-bytes codecs whose decoded bytes are a window of their encoded bytes
    /// from `source`.
    ///
    /// Returns the source of the encoded bytes of the remaining codecs, and the number of remaining codecs.
    fn strip_windows<'a>(
        &self,
        mut source: ByteSource<'a>,
    ) -> Result<(ByteSource<'a>, usize), CodecError> {
        let mut num_materialized = self.bytes_to_bytes.len();
        for codec in self.bytes_to_bytes.iter().rev() {
            let SourceStep::Window { prefix, suffix } = codec.source_step() else {
                break;
            };
            source = source.strip(prefix, suffix)?;
            num_materialized -= 1;
        }
        Ok((source, num_materialized))
    }

    /// Plan reading the entire encoded chunk `source` of `shape`, decoding the innermost `num_materialized`
    /// bytes-to-bytes codecs (see [`strip_windows`](Self::strip_windows)), then continuing with `then` on a source of
    /// the decoded bytes, or [`None`] if the chunk does not exist.
    ///
    /// The decoded source is in memory, so a task planned on it runs immediately ([`Task::run_without_io`]).
    fn plan_materialized<'a>(
        self: Arc<Self>,
        source: ByteSource<'a>,
        num_materialized: usize,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        then: impl FnOnce(
            Arc<Self>,
            Option<ByteSource<'a>>,
            &ComputeCtx,
        ) -> Result<Outcome<'a>, CodecError>
        + MaybeSend
        + 'a,
    ) -> Task<'a> {
        let hint = whole_chunk_hint(&*self, &source, shape);
        let shape: smallvec::SmallVec<[NonZeroU64; 4]> = shape.into();
        let options = *options;
        Task::read(
            source,
            [ByteRange::FromStart(0, None)],
            move |mut reads, ctx| {
                let decoded_source = reads
                    .pop()
                    .flatten()
                    .map(|encoded| {
                        self.decoded_source(
                            encoded,
                            &shape,
                            num_materialized,
                            &options,
                            ctx.resources(),
                        )
                    })
                    .transpose()?;
                then(self, decoded_source, ctx)
            },
        )
        .with_hint(hint)
    }

    /// The source of the encoded bytes of the array-to-bytes codec, given the `encoded` bytes of the innermost
    /// `num_materialized` bytes-to-bytes codecs of a chain without array-to-array codecs.
    ///
    /// The innermost codec may decode only the requested ranges of its decoded bytes, otherwise the codecs are decoded.
    fn decoded_source<'a>(
        &self,
        encoded: CowBytes<'a>,
        shape: &[NonZeroU64],
        num_materialized: usize,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ByteSource<'a>, CodecError> {
        let representations = self.bytes_representations(shape)?;
        if let SourceStep::RangeDecode(decoder) = self.bytes_to_bytes[0].source_step() {
            let bytes = self.decode_bytes_to_bytes(
                encoded,
                &representations,
                1..num_materialized,
                options,
                resources,
            )?;
            Ok(ByteSource::range_decoded(
                bytes,
                decoder,
                representations[0],
            )?)
        } else {
            Ok(ByteSource::memory(self.decode_bytes_to_bytes(
                encoded,
                &representations,
                0..num_materialized,
                options,
                resources,
            )?))
        }
    }

    /// Decode `bytes` through the bytes-to-bytes `codecs` (indices, innermost last) given their
    /// [`bytes_representations`](Self::bytes_representations).
    fn decode_bytes_to_bytes<'a>(
        &self,
        mut bytes: CowBytes<'a>,
        representations: &[BytesRepresentation],
        codecs: std::ops::Range<usize>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<CowBytes<'a>, CodecError> {
        for (codec, representation) in std::iter::zip(
            self.bytes_to_bytes[codecs.clone()].iter().rev(),
            representations[codecs].iter().rev(),
        ) {
            bytes = codec.decode(bytes, representation, options, resources)?;
        }
        Ok(bytes)
    }

    /// The decoded shape of each array-to-array codec for a chunk of `shape`, outermost first, then the shape they
    /// encode.
    fn encoded_shapes(&self, shape: &[NonZeroU64]) -> Result<EncodedShapes, CodecError> {
        let mut shapes = EncodedShapes::with_capacity(self.array_to_array.len() + 1);
        shapes.push(shape.to_vec());
        for codec in &self.array_to_array {
            let encoded_shape = codec.encoded_shape(shapes.last().expect("nonempty"))?;
            shapes.push(encoded_shape);
        }
        Ok(shapes)
    }

    /// The shape encoded by the array-to-array codecs of a chunk of `shape`.
    fn array_encoded_shape<'s>(
        &self,
        shape: &'s [NonZeroU64],
    ) -> Result<std::borrow::Cow<'s, [NonZeroU64]>, CodecError> {
        let mut encoded_shape = std::borrow::Cow::Borrowed(shape);
        for codec in &self.array_to_array {
            encoded_shape = std::borrow::Cow::Owned(codec.encoded_shape(&encoded_shape)?);
        }
        Ok(encoded_shape)
    }

    /// The bytes representations of a chunk of `shape` encoded by the array-to-array codecs.
    fn bytes_representations(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentations, CodecError> {
        let mut representations = BytesRepresentations::new();
        representations.push(self.array_to_bytes.encoded_representation(shape)?);
        for codec in &self.bytes_to_bytes {
            let representation = codec.encoded_representation(representations.last().unwrap());
            representations.push(representation);
        }
        Ok(representations)
    }

    /// Get the array to array codecs
    #[must_use]
    pub fn array_to_array_codecs(&self) -> &[Arc<dyn ArrayToArrayCodecTraits>] {
        &self.array_to_array
    }

    /// Get the array to bytes codec
    #[must_use]
    pub fn array_to_bytes_codec(&self) -> &Arc<dyn ArrayToBytesCodecTraits> {
        &self.array_to_bytes
    }

    /// Get the bytes to bytes codecs
    #[must_use]
    pub fn bytes_to_bytes_codecs(&self) -> &[Arc<dyn BytesToBytesCodecTraits>] {
        &self.bytes_to_bytes
    }
}

impl zarrs_plugin::ExtensionName for CodecChain {
    fn name(&self, _version: zarrs_plugin::ZarrVersion) -> Option<std::borrow::Cow<'static, str>> {
        // CodecChain is an internal type and does not have a serialization name
        None
    }
}

impl CodecTraits for CodecChain {
    /// Returns [`None`] since a codec chain does not have standard codec metadata.
    ///
    /// Note that usage of the codec chain is explicit in [`Array`](crate::array::Array) and [`CodecChain::create_metadatas()`] will call [`CodecTraits::configuration()`] from for each codec.
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        None
    }
}

impl UnboundArrayToBytesCodecTraits for CodecChain {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self as Arc<dyn UnboundArrayToBytesCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        Ok(CodecChain::with_context(
            self,
            data_type,
            fill_value,
            codec_specific_options,
        )?)
    }
}

impl zarrs_codec::ArrayToBytesCodecSubchunkingTraits for CodecChainBound {
    fn decoded_subchunk_grids(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<Vec<ChunkGridDecoded>, ChunkGridCreateError> {
        let mut chunk_grids = Vec::with_capacity(self.array_to_array.len() + 1);
        let mut mapped_chunk_grid: ChunkGridDecoded = decoded_chunk_grid.into();
        chunk_grids.push(mapped_chunk_grid.clone());
        for codec in &self.array_to_array {
            let decoded_grid_is_global = matches!(mapped_chunk_grid, ChunkGridDecoded::Array(_));
            let encoded_chunk_grid = codec.encoded_chunk_grid((&mapped_chunk_grid).into())?;
            if matches!(encoded_chunk_grid, ChunkGridEncoded::Array(_)) && !decoded_grid_is_global {
                return Err(ChunkGridCreateError::new(
                    "a codec mapped a chunk-local grid to a global chunk grid",
                ));
            }
            mapped_chunk_grid = encoded_chunk_grid;
            chunk_grids.push(mapped_chunk_grid.clone());
        }

        let encoded_subchunk_grids = self
            .array_to_bytes
            .decoded_subchunk_grids((&mapped_chunk_grid).into())?;
        let mut decoded_subchunk_grids = Vec::with_capacity(encoded_subchunk_grids.len());

        for mut subchunk_grid in encoded_subchunk_grids {
            if matches!(subchunk_grid, ChunkGridDecoded::Array(_))
                && !matches!(mapped_chunk_grid, ChunkGridDecoded::Array(_))
            {
                return Err(ChunkGridCreateError::new(
                    "a codec returned a global subchunk grid for a non-global chunk grid",
                ));
            }

            for (codec, decoded_chunk_grid) in self
                .array_to_array
                .iter()
                .rev()
                .zip(chunk_grids.iter().rev().skip(1))
            {
                subchunk_grid = codec
                    .decoded_subchunk_grid(decoded_chunk_grid.into(), (&subchunk_grid).into())?;
            }
            decoded_subchunk_grids.push(subchunk_grid);
        }

        if let ChunkGridDecodedRef::Array(decoded_chunk_grid) = decoded_chunk_grid {
            let mut parent_grid = decoded_chunk_grid;
            for subchunk_grid in &decoded_subchunk_grids {
                if let ChunkGridDecoded::Array(subchunk_grid) = subchunk_grid {
                    validate_subchunk_grid_refines_chunk_grid(parent_grid, subchunk_grid)?;
                    parent_grid = subchunk_grid;
                }
            }
        }

        Ok(decoded_subchunk_grids)
    }

    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
        self.array_to_bytes.subchunk_codecs()
    }
}

impl ArrayToBytesCodecTraits for CodecChainBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self
    }

    fn encode<'a>(
        &self,
        mut bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<CowBytes<'a>, CodecError> {
        bytes.validate(shape.iter().map(|v| v.get()).product(), self.data_type())?;

        let mut shape = ChunkShape::from(shape.to_vec());

        // array->array
        for codec in &self.array_to_array {
            bytes = codec.encode(bytes, &shape, options, resources)?;
            shape = codec.encoded_shape(&shape)?;
        }

        // array->bytes
        let mut bytes = self
            .array_to_bytes
            .encode(bytes, &shape, options, resources)?;

        // bytes->bytes
        for codec in &self.bytes_to_bytes {
            bytes = codec.encode(bytes, options, resources)?;
        }

        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        mut bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let shapes = (!self.array_to_array.is_empty())
            .then(|| self.encoded_shapes(shape))
            .transpose()?;
        let encoded_shape = shapes
            .as_ref()
            .map_or(shape, |shapes| shapes.last().expect("nonempty"));

        // bytes->bytes
        if !self.bytes_to_bytes.is_empty() {
            let representations = self.bytes_representations(encoded_shape)?;
            bytes = self.decode_bytes_to_bytes(
                bytes,
                &representations,
                0..self.bytes_to_bytes.len(),
                options,
                resources,
            )?;
        }

        // bytes->array
        let mut bytes = self
            .array_to_bytes
            .decode(bytes, encoded_shape, options, resources)?;

        // array->array
        if let Some(shapes) = &shapes {
            for (codec, shape) in std::iter::zip(
                self.array_to_array.iter().rev(),
                shapes.iter().rev().skip(1),
            ) {
                bytes = codec.decode(bytes, shape, options, resources)?;
            }
        }

        bytes.validate(shape.iter().map(|v| v.get()).product(), self.data_type())?;

        Ok(bytes)
    }

    fn decode_into(
        &self,
        input: ArrayBytesDecodeIntoInput<'_>,
        shape: &[NonZeroU64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        if self.array_to_array.is_empty() {
            let input = if let Some(last) = self.bytes_to_bytes.first() {
                let representations = self.bytes_representations(shape)?;
                let bytes = self.decode_bytes_to_bytes(
                    input.into_bytes(options, resources)?,
                    &representations,
                    1..self.bytes_to_bytes.len(),
                    options,
                    resources,
                )?;
                // Leave placement of the final decode to the array-to-bytes receiver.
                ArrayBytesDecodeIntoInput::Deferred(BytesDecodeSource::new(
                    last.as_ref(),
                    bytes,
                    representations[0],
                ))
            } else {
                // Transparent nested chains must not materialise a deferred input.
                input
            };
            return self.array_to_bytes.decode_into(
                input,
                shape,
                output_target,
                options,
                resources,
            );
        }

        let bytes = self.decode(
            input.into_bytes(options, resources)?,
            shape,
            options,
            resources,
        )?;
        decode_into_array_bytes_target(&bytes, output_target)
    }

    fn plan_decode<'a>(
        self: Arc<Self>,
        source: ByteSource<'a>,
        shape: &[NonZeroU64],
        request: DecodeRequest<'a>,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        if let Some(bytes_chain) = self.bytes_chain.clone() {
            if request.indexer.is_whole(shape) {
                return plan_decode_whole(self, source, shape, request, options);
            }
            return self.plan_decode_array_to_array(
                &self.encoded_shapes(shape)?,
                request,
                options,
                |encoded_shape, request| {
                    bytes_chain.plan_decode(source, encoded_shape, request, options)
                },
            );
        }
        if self.bytes_to_bytes.is_empty() {
            return self
                .array_to_bytes
                .clone()
                .plan_decode(source, shape, request, options);
        }

        // Whole chunks are read and decoded directly into the target (or the fill value)
        if request.indexer.is_whole(shape) {
            return plan_decode_whole(self, source, shape, request, options);
        }

        request.indexer.validate(shape)?;

        // Partial reads pass through codecs whose decoded bytes are a window of their encoded bytes
        let (source, num_materialized) = self.strip_windows(source)?;
        if num_materialized == 0 {
            return self
                .array_to_bytes
                .clone()
                .plan_decode(source, shape, request, options);
        }

        // Otherwise, decode the remaining bytes-to-bytes codecs, and plan the array-to-bytes codec on the decoded bytes
        let shape: smallvec::SmallVec<[NonZeroU64; 4]> = shape.into();
        let options = *options;
        Ok(self.plan_materialized(
            source,
            num_materialized,
            &shape.clone(),
            &options,
            move |chain, decoded_source, ctx| match decoded_source {
                Some(source) => chain
                    .array_to_bytes
                    .clone()
                    .plan_decode(source, &shape, request, &options)?
                    .run_without_io(ctx),
                None => decode_whole(&*chain, None, &shape, request, &options, ctx.resources()),
            },
        ))
    }

    fn plans_subtasks(&self) -> bool {
        self.array_to_bytes.plans_subtasks()
    }

    fn plan_local_subchunk_grids<'a>(
        self: Arc<Self>,
        source: ByteSource<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        if let Some(bytes_chain) = self.bytes_chain.clone() {
            // Map the grids of the encoded chunk through each array-to-array codec, innermost first
            let decoded_shapes = self.encoded_shapes(shape)?;
            let encoded_shape = decoded_shapes.last().expect("nonempty");
            let inner = bytes_chain.plan_local_subchunk_grids(source, encoded_shape, options)?;
            return Ok(Task::compute(move |_| {
                Ok(Outcome::Fork {
                    children: vec![inner],
                    join: Join::new(move |mut values, _ctx| {
                        let mut grids = local_subchunk_grids(values.pop().unwrap_or_default())?;
                        for (codec, shape) in
                            std::iter::zip(&self.array_to_array, &decoded_shapes).rev()
                        {
                            grids = grids
                                .into_iter()
                                .map(|grid| {
                                    grid.map_or(Ok(None), |grid| {
                                        codec.decoded_local_subchunk_grid(shape, &grid)
                                    })
                                })
                                .collect::<Result<_, _>>()?;
                        }
                        Ok(Outcome::Value(local_subchunk_grids_value(grids)))
                    }),
                })
            }));
        }

        // Grids are resolved through codecs whose decoded bytes are a window of their encoded bytes
        let (source, num_materialized) = self.strip_windows(source)?;
        if num_materialized == 0 {
            return self
                .array_to_bytes
                .clone()
                .plan_local_subchunk_grids(source, shape, options);
        }

        // Otherwise, decode the remaining bytes-to-bytes codecs
        let missing_source = source.clone();
        let shape: smallvec::SmallVec<[NonZeroU64; 4]> = shape.into();
        let options = *options;
        Ok(self.plan_materialized(
            source,
            num_materialized,
            &shape.clone(),
            &options,
            move |chain, decoded_source, ctx| {
                let array_to_bytes = chain.array_to_bytes.clone();
                match decoded_source {
                    Some(source) => array_to_bytes
                        .plan_local_subchunk_grids(source, &shape, &options)?
                        .run_without_io(ctx),
                    None => {
                        // The chunk does not exist, so the array-to-bytes codec resolves the grids of a missing chunk
                        let task = array_to_bytes.plan_local_subchunk_grids(
                            missing_source,
                            &shape,
                            &options,
                        )?;
                        Ok(Outcome::Fork {
                            children: vec![task],
                            join: Join::new(|mut values, _ctx| {
                                Ok(Outcome::Value(values.pop().unwrap_or_default()))
                            }),
                        })
                    }
                }
            },
        ))
    }

    fn plan_locate_subchunk(
        self: Arc<Self>,
        source: ByteSource<'static>,
        shape: &[NonZeroU64],
        coordinates: &[u64],
        options: &CodecOptions,
    ) -> Result<Task<'static>, CodecError> {
        if coordinates.len() != shape.len()
            || std::iter::zip(coordinates, shape)
                .any(|(coordinate, length)| *coordinate >= length.get())
        {
            return Err(CodecError::Other(format!(
                "coordinates {coordinates:?} are out of bounds of a chunk with shape {shape:?}"
            )));
        }
        if let Some(bytes_chain) = self.bytes_chain.clone() {
            // Map the element through the array-to-array codecs, outermost first
            let mut shape = ChunkShape::from(shape.to_vec());
            let mut coordinates = crate::array::ArrayIndices::from(coordinates);
            for codec in &self.array_to_array {
                coordinates = codec.encoded_coordinates(&shape, &coordinates)?;
                shape = codec.encoded_shape(&shape)?;
            }
            return bytes_chain.plan_locate_subchunk(source, &shape, &coordinates, options);
        }

        // Subchunks are located through codecs whose decoded bytes are a window of their encoded bytes
        let (source, num_materialized) = self.strip_windows(source)?;
        if num_materialized == 0 {
            return self.array_to_bytes.clone().plan_locate_subchunk(
                source,
                shape,
                coordinates,
                options,
            );
        }

        // Otherwise, decode the remaining bytes-to-bytes codecs, so the subchunk is located in memory
        let shape = ChunkShape::from(shape.to_vec());
        let coordinates = coordinates.to_vec();
        let options = *options;
        Ok(self.plan_materialized(
            source,
            num_materialized,
            &shape.clone(),
            &options,
            move |chain, decoded_source, ctx| match decoded_source {
                Some(source) => chain
                    .array_to_bytes
                    .clone()
                    .plan_locate_subchunk(source, &shape, &coordinates, &options)?
                    .run_without_io(ctx),
                None => Ok(Outcome::Value(subchunk_location_value(None))),
            },
        ))
    }

    fn plan_encode<'a>(
        self: Arc<Self>,
        bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        if !self.array_to_bytes.plans_encode_subtasks() {
            return plan_encode_whole(self, bytes, shape, options);
        }
        bytes.validate(shape.iter().map(|v| v.get()).product(), self.data_type())?;
        if self.array_to_array.is_empty() && self.bytes_to_bytes.is_empty() {
            return self
                .array_to_bytes
                .clone()
                .plan_encode(bytes, shape, options);
        }

        // Encode the array-to-array codecs, then plan the array-to-bytes codec, and encode its value with the
        // bytes-to-bytes codecs
        let hint = encode_hint(&bytes);
        let shape = ChunkShape::from(shape.to_vec());
        let options = *options;
        Ok(Task::compute(move |ctx| {
            let (mut bytes, mut shape) = (bytes, shape);
            for codec in &self.array_to_array {
                bytes = codec.encode(bytes, &shape, &options, ctx.resources())?;
                shape = codec.encoded_shape(&shape)?;
            }
            let task = self
                .array_to_bytes
                .clone()
                .plan_encode(bytes, &shape, &options)?;
            Ok(Outcome::Fork {
                children: vec![task],
                join: Join::new(move |mut values, ctx| {
                    let mut bytes = encoded_value(values.pop().unwrap_or_default())?;
                    for codec in &self.bytes_to_bytes {
                        bytes = codec.encode(bytes, &options, ctx.resources())?;
                    }
                    Ok(Outcome::Value(TaskValue::Bytes(Some(bytes))))
                }),
            })
        })
        .with_hint(hint))
    }

    fn plans_encode_subtasks(&self) -> bool {
        self.array_to_bytes.plans_encode_subtasks()
    }

    fn plan_partial_encode<'a>(
        self: Arc<Self>,
        key: SourceKey,
        shape: &[NonZeroU64],
        indexer: PlanIndexer<'a>,
        bytes: ArrayBytes<'a>,
        state: Option<PartialEncodeState>,
        set_partial: bool,
        options: &CodecOptions,
    ) -> Result<Task<'a>, CodecError> {
        bytes.validate(indexer.as_indexer().len(), self.data_type())?;
        // A partial encode cannot be mapped through bytes-to-bytes codecs, or through array-to-array codecs that do not
        // map partial encodes, so the whole chunk is updated
        if !self.bytes_to_bytes.is_empty()
            || !self
                .array_to_array
                .iter()
                .all(|codec| codec.plans_partial_encode())
        {
            return Ok(plan_partial_encode_whole(
                self, key, shape, indexer, bytes, options,
            ));
        }
        if self.array_to_array.is_empty() {
            return self.array_to_bytes.clone().plan_partial_encode(
                key,
                shape,
                indexer,
                bytes,
                state,
                set_partial,
                options,
            );
        }

        // Map the partial encode through the array-to-array codecs, then plan it with the array-to-bytes codec
        let hint = encode_hint(&bytes);
        let shape = ChunkShape::from(shape.to_vec());
        let options = *options;
        Ok(Task::compute(move |ctx| {
            let (mut indexer, mut bytes, mut shape) = (indexer, bytes, shape);
            for codec in &self.array_to_array {
                (indexer, bytes) = codec
                    .partial_encode_request(&shape, indexer, bytes, &options, ctx.resources())?
                    .ok_or_else(not_partially_encoded)?;
                shape = codec.encoded_shape(&shape)?;
            }
            let task = self.array_to_bytes.clone().plan_partial_encode(
                key,
                &shape,
                indexer,
                bytes,
                state,
                set_partial,
                &options,
            )?;
            Ok(Outcome::Fork {
                children: vec![task],
                join: Join::new(|mut values, _ctx| {
                    Ok(Outcome::Value(values.pop().unwrap_or_default()))
                }),
            })
        })
        .with_hint(hint))
    }

    fn prepare_decode(
        self: Arc<Self>,
        source: ByteSource<'static>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<PrepareDecode, CodecError> {
        if let Some(bytes_chain) = self.bytes_chain.clone() {
            // Requests are mapped through the array-to-array codecs as they are planned
            let shapes = self.encoded_shapes(shape)?;
            return Ok(bytes_chain
                .prepare_decode(source, shapes.last().expect("nonempty"), options)?
                .map(move |inner| {
                    Arc::new(ArrayToArrayDecoder {
                        chain: self,
                        shapes,
                        inner,
                    })
                }));
        }

        let (source, num_materialized) = self.strip_windows(source)?;
        if num_materialized == 0 {
            return self
                .array_to_bytes
                .clone()
                .prepare_decode(source, shape, options);
        }

        // Otherwise, decode the remaining bytes-to-bytes codecs once, then prepare the array-to-bytes codec on the
        // decoded bytes
        let shape: smallvec::SmallVec<[NonZeroU64; 4]> = shape.into();
        let options = *options;
        Ok(PrepareDecode::Plan(self.plan_materialized(
            source,
            num_materialized,
            &shape.clone(),
            &options,
            move |chain, decoded_source, ctx| {
                let Some(source) = decoded_source else {
                    let decoder = MissingDecoder::new(chain.into_dyn(), &shape);
                    return Ok(Outcome::Value(TaskValue::Prepared(Arc::new(decoder))));
                };
                match chain
                    .array_to_bytes
                    .clone()
                    .prepare_decode(source, &shape, &options)?
                {
                    PrepareDecode::Ready(decoder) => {
                        Ok(Outcome::Value(TaskValue::Prepared(decoder)))
                    }
                    PrepareDecode::Plan(task) => task.run_without_io(ctx),
                }
            },
        )))
    }

    fn compact<'a>(
        &self,
        mut bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Option<CowBytes<'a>>, CodecError> {
        let encoded_shape = self.array_encoded_shape(shape)?;
        let encoded_shape = encoded_shape.as_ref();

        // Decode through bytes_to_bytes codecs (in reverse) to get to array_to_bytes level
        if !self.bytes_to_bytes.is_empty() {
            let representations = self.bytes_representations(encoded_shape)?;
            bytes = self.decode_bytes_to_bytes(
                bytes,
                &representations,
                0..self.bytes_to_bytes.len(),
                options,
                resources,
            )?;
        }

        // Compact at the array_to_bytes level (e.g., ShardingCodec compact)
        let compacted = self
            .array_to_bytes
            .compact(bytes, encoded_shape, options, resources)?;

        // If compaction occurred, re-encode through bytes_to_bytes codecs
        if let Some(mut compacted_bytes) = compacted {
            for codec in &self.bytes_to_bytes {
                compacted_bytes = codec.encode(compacted_bytes, options, resources)?;
            }
            Ok(Some(compacted_bytes))
        } else {
            Ok(None)
        }
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        let mut bytes_representation = self
            .array_to_bytes
            .encoded_representation(&self.array_encoded_shape(shape)?)?;
        for codec in &self.bytes_to_bytes {
            bytes_representation = codec.encoded_representation(&bytes_representation);
        }
        Ok(bytes_representation)
    }
}

impl ArrayCodecTraits for CodecChainBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        self.array_to_array.first().map_or_else(
            || self.array_to_bytes.data_type(),
            |codec| codec.data_type(),
        )
    }

    fn fill_value(&self) -> &FillValue {
        self.array_to_array.first().map_or_else(
            || self.array_to_bytes.fill_value(),
            |codec| codec.fill_value(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;
    use crate::array::codec::BytesCodec;
    use crate::array::{ArraySubset, ArraySubsetTraits, ChunkShapeTraits, data_type};

    #[cfg(feature = "transpose")]
    const JSON_TRANSPOSE1: &str = r#"{
    "name": "transpose",
    "configuration": {
      "order": [0, 2, 1]
    }
}"#;

    #[cfg(feature = "transpose")]
    const JSON_TRANSPOSE2: &str = r#"{
    "name": "transpose",
    "configuration": {
        "order": [2, 0, 1]
    }
}"#;

    #[cfg(feature = "blosc")]
    const JSON_BLOSC: &str = r#"{
    "name": "blosc",
    "configuration": {
        "cname": "lz4",
        "clevel": 5,
        "shuffle": "shuffle",
        "typesize": 2,
        "blocksize": 0
    }
}"#;

    #[cfg(feature = "gzip")]
    const JSON_GZIP: &str = r#"{
    "name": "gzip",
    "configuration": {
        "level": 1
    }
}"#;

    #[cfg(feature = "zstd")]
    const JSON_ZSTD: &str = r#"{
    "name": "zstd",
    "configuration": {
        "level": 1,
        "checksum": false
    }
}"#;

    #[cfg(feature = "bz2")]
    const JSON_BZ2: &str = r#"{ 
    "name": "numcodecs.bz2",
    "configuration": {
        "level": 5
    }
}"#;

    const JSON_BYTES: &str = r#"{
    "name": "bytes",
    "configuration": {
        "endian": "big"
    }
}"#;

    #[derive(Debug)]
    struct RepresentationCheckingBytesCodec {
        expected_decoded_representation: BytesRepresentation,
    }

    impl zarrs_plugin::ExtensionName for RepresentationCheckingBytesCodec {
        fn name(
            &self,
            _version: zarrs_plugin::ZarrVersion,
        ) -> Option<std::borrow::Cow<'static, str>> {
            None
        }
    }

    impl CodecTraits for RepresentationCheckingBytesCodec {
        fn configuration(
            &self,
            _version: zarrs_plugin::ZarrVersion,
            _options: &CodecMetadataOptions,
        ) -> Option<Configuration> {
            None
        }
    }

    impl BytesToBytesCodecTraits for RepresentationCheckingBytesCodec {
        fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
            self
        }

        fn encoded_representation(
            &self,
            decoded_representation: &BytesRepresentation,
        ) -> BytesRepresentation {
            match decoded_representation {
                BytesRepresentation::FixedSize(size) => BytesRepresentation::FixedSize(size + 1),
                BytesRepresentation::BoundedSize(size) => {
                    BytesRepresentation::BoundedSize(size + 1)
                }
                BytesRepresentation::UnboundedSize => BytesRepresentation::UnboundedSize,
            }
        }

        fn encode<'a>(
            &self,
            decoded_value: CowBytes<'a>,
            _options: &CodecOptions,
            _resources: &Resources,
        ) -> Result<CowBytes<'a>, CodecError> {
            let mut encoded = decoded_value.into_vec();
            encoded.push(0);
            Ok(CowBytes::from(encoded))
        }

        fn decode<'a>(
            &self,
            encoded_value: CowBytes<'a>,
            decoded_representation: &BytesRepresentation,
            _options: &CodecOptions,
            _resources: &Resources,
        ) -> Result<CowBytes<'a>, CodecError> {
            assert_eq!(
                decoded_representation,
                &self.expected_decoded_representation
            );
            Ok(CowBytes::from(
                encoded_value[..encoded_value.len() - 1].to_vec(),
            ))
        }
    }

    #[cfg(feature = "crc32c")]
    const JSON_CRC32C: &str = r#"{ 
    "name": "crc32c"
}"#;

    #[cfg(feature = "pcodec")]
    const JSON_PCODEC: &str = r#"{ 
    "name": "numcodecs.pcodec"
}"#;

    #[cfg(feature = "gdeflate")]
    const JSON_GDEFLATE: &str = r#"{ 
    "name": "zarrs.gdeflate",
    "configuration": {
        "level": 5
    }
}"#;

    fn codec_chain_round_trip_impl(
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        elements: Vec<f32>,
        json_array_to_bytes: &str,
        decoded_region: &dyn ArraySubsetTraits,
        decoded_partial_chunk_true: Vec<f32>,
    ) {
        let bytes: ArrayBytes = crate::array::transmute_to_bytes_vec(elements).into();

        let codec_configurations: Vec<MetadataV3> = vec![
            #[cfg(feature = "transpose")]
            serde_json::from_str(JSON_TRANSPOSE1).unwrap(),
            #[cfg(feature = "transpose")]
            serde_json::from_str(JSON_TRANSPOSE2).unwrap(),
            serde_json::from_str(json_array_to_bytes).unwrap(),
            #[cfg(feature = "blosc")]
            serde_json::from_str(JSON_BLOSC).unwrap(),
            #[cfg(feature = "gzip")]
            serde_json::from_str(JSON_GZIP).unwrap(),
            #[cfg(feature = "zstd")]
            serde_json::from_str(JSON_ZSTD).unwrap(),
            #[cfg(feature = "bz2")]
            serde_json::from_str(JSON_BZ2).unwrap(),
            #[cfg(feature = "gdeflate")]
            serde_json::from_str(JSON_GDEFLATE).unwrap(),
            #[cfg(feature = "crc32c")]
            serde_json::from_str(JSON_CRC32C).unwrap(),
        ];
        println!("{codec_configurations:?}");
        let not_just_bytes = codec_configurations.len() > 1;
        let codec = CodecChain::from_metadata(&codec_configurations)
            .unwrap()
            .with_context(
                data_type.clone(),
                fill_value.clone(),
                &CodecSpecificOptions::default(),
            )
            .unwrap();

        let encoded = codec
            .encode(
                bytes.clone(),
                shape,
                &CodecOptions::default(),
                &Resources::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(
                encoded.clone(),
                shape,
                &CodecOptions::default(),
                &Resources::default(),
            )
            .unwrap();
        if not_just_bytes {
            assert_ne!(encoded, decoded.clone().into_fixed().unwrap());
        }
        assert_eq!(bytes, decoded);

        // let encoded = codec
        //     .par_encode(bytes.clone(), &chunk_representation)
        //     .unwrap();
        // let decoded = codec
        //     .par_decode(encoded.clone(), &chunk_representation)
        //     .unwrap();
        // if not_just_bytes {
        //     assert_ne!(encoded, decoded);
        // }
        // assert_eq!(bytes, decoded);

        let decoded_partial_chunk = zarrs_codec::plan::partial_decode_memory(
            codec.clone(),
            Some(encoded),
            shape,
            decoded_region,
            &CodecOptions::default(),
            &Resources::default(),
        )
        .unwrap();

        let decoded_partial_chunk: Vec<f32> = decoded_partial_chunk
            .into_fixed()
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_ne_bytes(*b))
            .collect();
        println!("decoded_partial_chunk {decoded_partial_chunk:?}");
        assert_eq!(decoded_partial_chunk_true, decoded_partial_chunk);

        // println!("{} {}", encoded_chunk.len(), decoded_chunk.len());
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_chain_round_trip_bytes() {
        let chunk_shape = ChunkShape::from(vec![
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(2).unwrap(),
        ]);
        let elements: Vec<f32> = (0..chunk_shape.num_elements_usize())
            .map(|i| i as f32)
            .collect();
        let decoded_region = ArraySubset::new_with_ranges(&[0..2, 1..2, 0..1]);
        let decoded_partial_chunk_true = vec![2.0, 6.0];
        codec_chain_round_trip_impl(
            &chunk_shape,
            &data_type::float32(),
            &FillValue::from(0f32),
            elements,
            JSON_BYTES,
            &decoded_region,
            decoded_partial_chunk_true,
        );
    }

    #[test]
    fn codec_chain_partial_decode_uses_decoded_byte_representation() {
        let shape = ChunkShape::from(vec![NonZeroU64::new(4).unwrap()]);
        let data_type = data_type::uint16();
        let fill_value = FillValue::from(0u16);
        let expected_decoded_representation = BytesRepresentation::FixedSize(8);
        let codec = CodecChain::new(
            vec![],
            Arc::new(BytesCodec::new(Some(crate::array::Endianness::native()))),
            vec![Arc::new(RepresentationCheckingBytesCodec {
                expected_decoded_representation,
            })],
        )
        .with_context(data_type, fill_value, &CodecSpecificOptions::default())
        .unwrap();

        zarrs_codec::plan::partial_decode_memory(
            codec,
            Some(CowBytes::from(vec![0; 9])),
            &shape,
            &ArraySubset::new_with_ranges(&[1..3]),
            &CodecOptions::default(),
            &Resources::default(),
        )
        .unwrap();
    }

    #[cfg(feature = "pcodec")]
    #[test]
    #[cfg_attr(miri, ignore)]
    fn codec_chain_round_trip_pcodec() {
        let chunk_shape = ChunkShape::from(vec![
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(2).unwrap(),
        ]);
        let elements: Vec<f32> = (0..chunk_shape.num_elements_usize())
            .map(|i| i as f32)
            .collect();
        let decoded_region = ArraySubset::new_with_ranges(&[0..2, 1..2, 0..1]);
        let decoded_partial_chunk_true = vec![2.0, 6.0];
        codec_chain_round_trip_impl(
            &chunk_shape,
            &data_type::float32(),
            &FillValue::from(0f32),
            elements,
            JSON_PCODEC,
            &decoded_region,
            decoded_partial_chunk_true,
        );
    }
}
