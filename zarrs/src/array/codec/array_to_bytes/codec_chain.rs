//! An array to bytes codec formed by joining an array to array sequence, array to bytes, and bytes to bytes sequence of codecs.

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs_plugin::{ExtensionName, ZarrVersion};

use crate::array::codec::{ArrayPartialDecoderCache, BytesPartialDecoderCache};
use crate::array::{
    ArrayBytes, ArrayBytesRaw, BytesRepresentation, ChunkShape, DataType, FillValue,
};
use zarrs_codec::{
    ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderTraits,
    ArrayPartialEncoderTraits, ArrayToArrayCodecTraits, ArrayToBytesCodecTraits,
    BytesPartialDecoderTraits, BytesPartialEncoderTraits, BytesToBytesCodecTraits, Codec,
    CodecCreateError, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency,
    UnboundArrayToArrayCodecTraits, UnboundArrayToBytesCodecTraits, decode_into_array_bytes_target,
};
#[cfg(feature = "async")]
use zarrs_codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};
use zarrs_metadata::Configuration;
use zarrs_metadata::v3::MetadataV3;

type ArrayRepresentations = Vec<(ChunkShape, DataType, FillValue)>;
type BytesRepresentations = Vec<BytesRepresentation>;

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
#[derive(Debug)]
pub struct CodecChainBound {
    array_to_array: Vec<Arc<dyn ArrayToArrayCodecTraits>>,
    array_to_bytes: Arc<dyn ArrayToBytesCodecTraits>,
    bytes_to_bytes: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    cache_index: Option<usize>,
}

impl CodecChain {
    /// Bind this codec chain to a decoded data type and fill value.
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if any codec cannot be bound to the derived context.
    pub fn with_context(
        &self,
        mut data_type: DataType,
        mut fill_value: FillValue,
    ) -> Result<Arc<CodecChainBound>, CodecCreateError> {
        let mut array_to_array = Vec::with_capacity(self.array_to_array.len());
        for codec in &self.array_to_array {
            let bound = codec
                .clone()
                .with_context(data_type.clone(), fill_value.clone())?;
            data_type = bound.encoded_data_type().clone();
            fill_value = bound.encoded_fill_value().clone();
            array_to_array.push(bound);
        }
        let array_to_bytes = self
            .array_to_bytes
            .clone()
            .with_context(data_type, fill_value)?;

        let mut cache_index_must = None;
        let mut cache_index_should = None;
        let mut codec_index = 0;
        for codec in self.bytes_to_bytes.iter().rev() {
            let cap = codec.partial_decoder_capability();
            if !cap.partial_decode {
                cache_index_must = Some(codec_index + 1);
            }
            if !cap.partial_read {
                cache_index_should = Some(codec_index);
            }
            codec_index += 1;
        }
        {
            let cap = self.array_to_bytes.partial_decoder_capability();
            if !cap.partial_decode {
                cache_index_must = Some(codec_index + 1);
            }
            if !cap.partial_read {
                cache_index_should = Some(codec_index);
            }
            codec_index += 1;
        }
        for codec in self.array_to_array.iter().rev() {
            let cap = codec.partial_decoder_capability();
            if !cap.partial_decode {
                cache_index_must = Some(codec_index + 1);
            }
            if !cap.partial_read {
                cache_index_should = Some(codec_index);
            }
            codec_index += 1;
        }
        let cache_index = match (cache_index_must, cache_index_should) {
            (Some(m), Some(s)) => Some(m.max(s)),
            (Some(m), None) => Some(m),
            (None, Some(s)) => Some(s),
            (None, None) => None,
        };

        Ok(Arc::new(CodecChainBound {
            array_to_array,
            array_to_bytes,
            bytes_to_bytes: self.bytes_to_bytes.clone(),
            cache_index,
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

    /// Return a new codec chain with each codec reconfigured using `opts`.
    ///
    /// Each codec in the chain is given the opportunity to read its own options type
    /// from `opts` and return a new instance. Codecs that do not recognise any option
    /// in `opts` are returned unchanged (default behaviour).
    pub fn with_codec_specific_options(
        self,
        opts: &zarrs_codec::CodecSpecificOptions,
    ) -> Result<Self, CodecCreateError> {
        Ok(Self::new(
            self.array_to_array
                .into_iter()
                .map(|c| c.with_codec_specific_options(opts))
                .collect::<Result<_, _>>()?,
            self.array_to_bytes.with_codec_specific_options(opts)?,
            self.bytes_to_bytes
                .into_iter()
                .map(|c| c.with_codec_specific_options(opts))
                .collect::<Result<_, _>>()?,
        ))
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

    fn recommended_concurrency(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        let mut concurrency_min = usize::MAX;
        let mut concurrency_max = 0;

        let array_representations = self.get_array_representations(shape)?;
        let bytes_representations = self.get_bytes_representations(
            &array_representations
                .last()
                .expect("array representations is non-empty")
                .0,
        )?;

        // bytes->bytes
        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            let recommended_concurrency = &codec.recommended_concurrency(bytes_representation)?;
            concurrency_min = std::cmp::min(concurrency_min, recommended_concurrency.min());
            concurrency_max = std::cmp::max(concurrency_max, recommended_concurrency.max());
        }

        let recommended_concurrency = self
            .array_to_bytes
            .recommended_concurrency(&array_representations.last().unwrap().0)?;
        concurrency_min = std::cmp::min(concurrency_min, recommended_concurrency.min());
        concurrency_max = std::cmp::max(concurrency_max, recommended_concurrency.max());

        // array->array
        for (codec, (shape, _data_type, _fill_value)) in std::iter::zip(
            self.array_to_array.iter().rev(),
            array_representations.iter().rev().skip(1),
        ) {
            let recommended_concurrency = codec.recommended_concurrency(shape)?;
            concurrency_min = std::cmp::min(concurrency_min, recommended_concurrency.min());
            concurrency_max = std::cmp::max(concurrency_max, recommended_concurrency.max());
        }

        let recommended_concurrency = RecommendedConcurrency::new(
            std::cmp::min(concurrency_min, concurrency_max)..concurrency_max,
        );

        Ok(recommended_concurrency)
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToBytesCodecTraits for CodecChainBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        let array_representations = self.get_array_representations(shape)?;
        let mut bytes_representation = self
            .array_to_bytes
            .encoded_representation(&array_representations.last().unwrap().0)?;
        for codec in &self.bytes_to_bytes {
            bytes_representation = codec.encoded_representation(&bytes_representation);
        }
        Ok(bytes_representation)
    }

    fn partial_decode_granularity(
        &self,
        decoded_shape: &[NonZeroU64],
    ) -> Result<ChunkShape, CodecError> {
        let array_representations = self.get_array_representations(decoded_shape)?;
        let mut granularity = self
            .array_to_bytes
            .partial_decode_granularity(&array_representations.last().unwrap().0)?;

        for (codec, (decoded_shape, _decoded_data_type, _decoded_fill_value)) in self
            .array_to_array
            .iter()
            .rev()
            .zip(array_representations.iter().rev().skip(1))
        {
            granularity = codec.partial_decode_granularity(decoded_shape, &granularity)?;
        }

        Ok(granularity)
    }

    fn encode<'a>(
        &self,
        mut bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        bytes.validate(shape.iter().map(|v| v.get()).product(), self.data_type())?;

        let mut shape = ChunkShape::from(shape.to_vec());

        // array->array
        for codec in &self.array_to_array {
            bytes = codec.encode(bytes, &shape, options)?;
            shape = codec.encoded_shape(&shape)?;
        }

        // array->bytes
        let mut bytes = self.array_to_bytes.encode(bytes, &shape, options)?;
        let mut decoded_representation = self.array_to_bytes.encoded_representation(&shape)?;

        // bytes->bytes
        for codec in &self.bytes_to_bytes {
            bytes = codec.encode(bytes, options)?;
            decoded_representation = codec.encoded_representation(&decoded_representation);
        }

        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        mut bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let (array_representations, bytes_representations) = self.get_representations(shape)?;

        // bytes->bytes
        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            bytes = codec.decode(bytes, bytes_representation, options)?;
        }

        // bytes->array
        let mut bytes =
            self.array_to_bytes
                .decode(bytes, &array_representations.last().unwrap().0, options)?;

        // array->array
        for (codec, (shape, _data_type, _fill_value)) in std::iter::zip(
            self.array_to_array.iter().rev(),
            array_representations.iter().rev().skip(1),
        ) {
            bytes = codec.decode(bytes, shape, options)?;
        }

        let (shape, data_type, _) = array_representations.first().unwrap();
        bytes.validate(shape.iter().map(|v| v.get()).product(), data_type)?;

        Ok(bytes)
    }

    fn compact<'a>(
        &self,
        mut bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytesRaw<'a>>, CodecError> {
        let (array_representations, bytes_representations) = self.get_representations(shape)?;

        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            bytes = codec.decode(bytes, bytes_representation, options)?;
        }

        let compacted = self.array_to_bytes.compact(
            bytes,
            &array_representations.last().unwrap().0,
            options,
        )?;

        if let Some(mut compacted_bytes) = compacted {
            let mut bytes_representation = *bytes_representations.first().unwrap();
            for codec in &self.bytes_to_bytes {
                compacted_bytes = codec.encode(compacted_bytes, options)?;
                bytes_representation = codec.encoded_representation(&bytes_representation);
            }
            Ok(Some(compacted_bytes))
        } else {
            Ok(None)
        }
    }

    fn decode_into(
        &self,
        mut bytes: ArrayBytesRaw<'_>,
        shape: &[NonZeroU64],
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let (array_representations, bytes_representations) = self.get_representations(shape)?;

        if self.bytes_to_bytes.is_empty() && self.array_to_array.is_empty() {
            return self.array_to_bytes.decode_into(
                bytes,
                &array_representations.last().unwrap().0,
                output_target,
                options,
            );
        }

        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            bytes = codec.decode(bytes, bytes_representation, options)?;
        }

        if self.array_to_array.is_empty() {
            return self.array_to_bytes.decode_into(
                bytes,
                &array_representations.last().unwrap().0,
                output_target,
                options,
            );
        }

        let mut bytes =
            self.array_to_bytes
                .decode(bytes, &array_representations.last().unwrap().0, options)?;

        for (codec, (shape, _data_type, _fill_value)) in std::iter::zip(
            self.array_to_array.iter().rev(),
            array_representations.iter().rev().skip(1),
        ) {
            bytes = codec.decode(bytes, shape, options)?;
        }

        let (shape, data_type, _) = array_representations.first().unwrap();
        bytes.validate(shape.iter().map(|v| v.get()).product(), data_type)?;

        decode_into_array_bytes_target(&bytes, output_target)
    }

    fn partial_decoder(
        self: Arc<Self>,
        mut input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        let (array_representations, bytes_representations) = self.get_representations(shape)?;
        let mut codec_index = 0;
        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            if Some(codec_index) == self.cache_index {
                input_handle = Arc::new(BytesPartialDecoderCache::new(&*input_handle, options)?);
            }
            codec_index += 1;
            input_handle =
                Arc::clone(codec).partial_decoder(input_handle, bytes_representation, options)?;
        }

        if Some(codec_index) == self.cache_index {
            input_handle = Arc::new(BytesPartialDecoderCache::new(&*input_handle, options)?);
        }

        let mut input_handle = {
            let (shape, _data_type, _fill_value) = array_representations.last().unwrap();
            codec_index += 1;
            self.array_to_bytes
                .clone()
                .partial_decoder(input_handle, shape, options)?
        };

        for (codec, (shape, data_type, _fill_value)) in std::iter::zip(
            self.array_to_array.iter().rev(),
            array_representations.iter().rev().skip(1),
        ) {
            if Some(codec_index) == self.cache_index {
                input_handle = Arc::new(ArrayPartialDecoderCache::new(
                    &*input_handle,
                    shape.clone(),
                    data_type.clone(),
                    options,
                )?);
            }
            codec_index += 1;
            input_handle = codec
                .clone()
                .partial_decoder(input_handle, shape, options)?;
        }

        if Some(codec_index) == self.cache_index {
            let (shape, data_type, _fill_value) = array_representations.first().unwrap();
            input_handle = Arc::new(ArrayPartialDecoderCache::new(
                &*input_handle,
                shape.clone(),
                data_type.clone(),
                options,
            )?);
        }

        Ok(input_handle)
    }

    fn partial_encoder(
        self: Arc<Self>,
        mut input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        let (array_representations, bytes_representations) = self.get_representations(shape)?;

        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            input_output_handle = Arc::clone(codec).partial_encoder(
                input_output_handle,
                bytes_representation,
                options,
            )?;
        }

        let mut input_output_handle = self.array_to_bytes.clone().partial_encoder(
            input_output_handle,
            &array_representations.last().unwrap().0,
            options,
        )?;

        for (codec, (shape, _data_type, _fill_value)) in std::iter::zip(
            self.array_to_array.iter().rev(),
            array_representations.iter().rev().skip(1),
        ) {
            input_output_handle =
                codec
                    .clone()
                    .partial_encoder(input_output_handle, shape, options)?;
        }

        Ok(input_output_handle)
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        mut input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        let (array_representations, bytes_representations) = self.get_representations(shape)?;
        let mut codec_index = 0;
        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            if Some(codec_index) == self.cache_index {
                input_handle =
                    Arc::new(BytesPartialDecoderCache::async_new(&*input_handle, options).await?);
            }
            codec_index += 1;
            input_handle = codec
                .clone()
                .async_partial_decoder(input_handle, bytes_representation, options)
                .await?;
        }

        if Some(codec_index) == self.cache_index {
            input_handle =
                Arc::new(BytesPartialDecoderCache::async_new(&*input_handle, options).await?);
        }

        let mut input_handle = {
            let (shape, _data_type, _fill_value) = array_representations.last().unwrap();
            codec_index += 1;
            self.array_to_bytes
                .clone()
                .async_partial_decoder(input_handle, shape, options)
                .await?
        };

        for (codec, (shape, data_type, _fill_value)) in std::iter::zip(
            self.array_to_array.iter().rev(),
            array_representations.iter().rev().skip(1),
        ) {
            if Some(codec_index) == self.cache_index {
                input_handle = Arc::new(
                    ArrayPartialDecoderCache::async_new(
                        &*input_handle,
                        shape.clone(),
                        data_type.clone(),
                        options,
                    )
                    .await?,
                );
            }
            codec_index += 1;
            input_handle = codec
                .clone()
                .async_partial_decoder(input_handle, shape, options)
                .await?;
        }

        if Some(codec_index) == self.cache_index {
            let (shape, data_type, _fill_value) = array_representations.first().unwrap();
            input_handle = Arc::new(
                ArrayPartialDecoderCache::async_new(
                    &*input_handle,
                    shape.clone(),
                    data_type.clone(),
                    options,
                )
                .await?,
            );
        }

        Ok(input_handle)
    }

    #[cfg(feature = "async")]
    async fn async_partial_encoder(
        self: Arc<Self>,
        mut input_output_handle: Arc<dyn AsyncBytesPartialEncoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        let (array_representations, bytes_representations) = self.get_representations(shape)?;

        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            input_output_handle = codec
                .clone()
                .async_partial_encoder(input_output_handle, bytes_representation, options)
                .await?;
        }

        let mut input_output_handle = self
            .array_to_bytes
            .clone()
            .async_partial_encoder(
                input_output_handle,
                &array_representations.last().unwrap().0,
                options,
            )
            .await?;

        for (codec, (shape, _data_type, _fill_value)) in std::iter::zip(
            self.array_to_array.iter().rev(),
            array_representations.iter().rev().skip(1),
        ) {
            input_output_handle = codec
                .clone()
                .async_partial_encoder(input_output_handle, shape, options)
                .await?;
        }

        Ok(input_output_handle)
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

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        // All codecs in the chain must support partial decoding capabilities
        itertools::chain!(
            self.array_to_array
                .iter()
                .map(|codec| codec.partial_decoder_capability()),
            std::iter::once(&self.array_to_bytes).map(|codec| codec.partial_decoder_capability()),
            self.bytes_to_bytes
                .iter()
                .map(|codec| codec.partial_decoder_capability())
        )
        .fold(
            PartialDecoderCapability {
                partial_read: true,
                partial_decode: true,
            },
            |acc, capability| PartialDecoderCapability {
                partial_read: acc.partial_read && capability.partial_read,
                partial_decode: acc.partial_decode && capability.partial_decode,
            },
        )
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        // All codecs in the chain must support partial encoding capabilities
        itertools::chain!(
            self.array_to_array
                .iter()
                .map(|codec| codec.partial_encoder_capability()),
            std::iter::once(&self.array_to_bytes).map(|codec| codec.partial_encoder_capability()),
            self.bytes_to_bytes
                .iter()
                .map(|codec| codec.partial_encoder_capability())
        )
        .fold(
            PartialEncoderCapability {
                partial_encode: true,
            },
            |acc, capability| PartialEncoderCapability {
                partial_encode: acc.partial_encode && capability.partial_encode,
            },
        )
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl UnboundArrayToBytesCodecTraits for CodecChain {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self as Arc<dyn UnboundArrayToBytesCodecTraits>
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: FillValue,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        Ok(CodecChain::with_context(self, data_type, fill_value)?)
    }
}

impl CodecChainBound {
    fn get_array_representations(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<ArrayRepresentations, CodecError> {
        let mut array_representations = Vec::with_capacity(self.array_to_array.len() + 1);
        array_representations.push((
            shape.to_vec(),
            self.data_type().clone(),
            self.fill_value().clone(),
        ));
        for codec in &self.array_to_array {
            let (shape, _data_type, _fill_value) = array_representations.last().unwrap();
            array_representations.push((
                codec.encoded_shape(shape)?,
                codec.encoded_data_type().clone(),
                codec.encoded_fill_value().clone(),
            ));
        }
        Ok(array_representations)
    }

    fn get_bytes_representations(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<BytesRepresentations, CodecError> {
        let mut bytes_representations = Vec::with_capacity(self.bytes_to_bytes.len() + 1);
        bytes_representations.push(self.array_to_bytes.encoded_representation(shape)?);
        for codec in &self.bytes_to_bytes {
            bytes_representations
                .push(codec.encoded_representation(bytes_representations.last().unwrap()));
        }
        Ok(bytes_representations)
    }

    fn get_representations(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<(ArrayRepresentations, BytesRepresentations), CodecError> {
        let array_representations = self.get_array_representations(shape)?;
        let bytes_representations =
            self.get_bytes_representations(&array_representations.last().unwrap().0)?;
        Ok((array_representations, bytes_representations))
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
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

        fn partial_decoder_capability(&self) -> PartialDecoderCapability {
            PartialDecoderCapability {
                partial_read: false,
                partial_decode: false,
            }
        }

        fn partial_encoder_capability(&self) -> PartialEncoderCapability {
            PartialEncoderCapability {
                partial_encode: false,
            }
        }
    }

    impl BytesToBytesCodecTraits for RepresentationCheckingBytesCodec {
        fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
            self
        }

        fn recommended_concurrency(
            &self,
            _decoded_representation: &BytesRepresentation,
        ) -> Result<RecommendedConcurrency, CodecError> {
            Ok(RecommendedConcurrency::new_maximum(1))
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
            decoded_value: ArrayBytesRaw<'a>,
            _options: &CodecOptions,
        ) -> Result<ArrayBytesRaw<'a>, CodecError> {
            let mut encoded = decoded_value.into_owned();
            encoded.push(0);
            Ok(Cow::Owned(encoded))
        }

        fn decode<'a>(
            &self,
            encoded_value: ArrayBytesRaw<'a>,
            _decoded_representation: &BytesRepresentation,
            _options: &CodecOptions,
        ) -> Result<ArrayBytesRaw<'a>, CodecError> {
            Ok(Cow::Owned(
                encoded_value[..encoded_value.len() - 1].to_vec(),
            ))
        }

        fn partial_decoder(
            self: Arc<Self>,
            input_handle: Arc<dyn BytesPartialDecoderTraits>,
            decoded_representation: &BytesRepresentation,
            _options: &CodecOptions,
        ) -> Result<Arc<dyn BytesPartialDecoderTraits>, CodecError> {
            if decoded_representation != &self.expected_decoded_representation {
                return Err(CodecError::Other(format!(
                    "wrong decoded representation: expected {:?}, got {:?}",
                    self.expected_decoded_representation, decoded_representation
                )));
            }
            Ok(input_handle)
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
            .with_context(data_type.clone(), fill_value.clone())
            .unwrap();

        let encoded = codec
            .encode(bytes.clone(), shape, &CodecOptions::default())
            .unwrap();
        let decoded = codec
            .decode(encoded.clone(), shape, &CodecOptions::default())
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

        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .clone()
            .partial_decoder(input_handle.clone(), shape, &CodecOptions::default())
            .unwrap();
        assert_eq!(partial_decoder.size_held(), decoded.size()); // codec chain caches with most decompression codecs
        let decoded_partial_chunk = partial_decoder
            .partial_decode(decoded_region, &CodecOptions::default())
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
    fn codec_chain_partial_decoder_uses_decoded_byte_representation_at_cache_boundary() {
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
        .with_context(data_type, fill_value)
        .unwrap();

        let encoded_input: Arc<dyn BytesPartialDecoderTraits> = Arc::new(Cow::Owned(vec![0; 9]));
        codec
            .partial_decoder(encoded_input, &shape, &CodecOptions::default())
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
