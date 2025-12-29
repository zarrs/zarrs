//! An array to bytes codec formed by joining an array to array sequence, array to bytes, and bytes to bytes sequence of codecs.

use std::{num::NonZeroU64, sync::Arc};

#[cfg(feature = "async")]
use crate::array::codec::{
    AsyncArrayPartialDecoderTraits, AsyncArrayPartialEncoderTraits, AsyncBytesPartialDecoderTraits,
    AsyncBytesPartialEncoderTraits,
};
use crate::{
    array::{
        ArrayBytes, ArrayBytesRaw, BytesRepresentation, ChunkShape, DataType, FillValue,
        codec::{
            ArrayBytesDecodeIntoTarget, ArrayCodecTraits, ArrayPartialDecoderCache,
            ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToArrayCodecTraits,
            ArrayToBytesCodecTraits, BytesPartialDecoderCache, BytesPartialDecoderTraits,
            BytesPartialEncoderTraits, BytesToBytesCodecTraits, Codec, CodecError,
            CodecMetadataOptions, CodecOptions, CodecTraits, NamedArrayToArrayCodec,
            NamedArrayToBytesCodec, NamedBytesToBytesCodec, PartialDecoderCapability,
            PartialEncoderCapability,
        },
        concurrency::RecommendedConcurrency,
        data_type::DataTypeExt,
    },
    metadata::{Configuration, v3::MetadataV3},
    plugin::PluginCreateError,
};

/// A codec chain is a sequence of array to array, a bytes to bytes, and a sequence of array to bytes codecs.
///
/// A codec chain partial decoder may insert a cache.
/// For example, the output of the `blosc`/`gzip` codecs should be cached since they read and decode an entire chunk.
/// If decoding (i.e. going backwards through a codec chain), then a cache may be inserted
///    - following the last codec with `partial_decode` false, otherwise
///    - preceding the first codec with `partial_decode` true and `partial_read` false.
#[derive(Debug, Clone)]
pub struct CodecChain {
    array_to_array: Vec<NamedArrayToArrayCodec>,
    array_to_bytes: NamedArrayToBytesCodec,
    bytes_to_bytes: Vec<NamedBytesToBytesCodec>,
    cache_index: Option<usize>, // for partial decoders
}

impl CodecChain {
    /// Create a new codec chain.
    #[must_use]
    pub fn new(
        array_to_array: Vec<Arc<dyn ArrayToArrayCodecTraits>>,
        array_to_bytes: Arc<dyn ArrayToBytesCodecTraits>,
        bytes_to_bytes: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    ) -> Self {
        let array_to_array = array_to_array
            .into_iter()
            .map(NamedArrayToArrayCodec::new_default_name)
            .collect();
        let array_to_bytes = NamedArrayToBytesCodec::new_default_name(array_to_bytes);
        let bytes_to_bytes = bytes_to_bytes
            .into_iter()
            .map(NamedBytesToBytesCodec::new_default_name)
            .collect();
        Self::new_named(array_to_array, array_to_bytes, bytes_to_bytes)
    }

    /// Create a new codec chain from named codecs.
    #[must_use]
    pub fn new_named(
        array_to_array: Vec<NamedArrayToArrayCodec>,
        array_to_bytes: NamedArrayToBytesCodec,
        bytes_to_bytes: Vec<NamedBytesToBytesCodec>,
    ) -> Self {
        let mut cache_index_must = None;
        let mut cache_index_should = None;
        let mut codec_index = 0;
        for codec in bytes_to_bytes.iter().rev() {
            let capability = codec.partial_decoder_capability();
            if !capability.partial_read {
                cache_index_should = Some(codec_index);
            }
            if !capability.partial_decode {
                cache_index_must = Some(codec_index + 1);
            }
            codec_index += 1;
        }

        {
            let codec = &array_to_bytes;
            let capability = codec.partial_decoder_capability();
            if !capability.partial_read {
                cache_index_should = Some(codec_index);
            }
            if !capability.partial_decode {
                cache_index_must = Some(codec_index + 1);
            }
            codec_index += 1;
        }

        for codec in array_to_array.iter().rev() {
            let capability = codec.partial_decoder_capability();
            if !capability.partial_read {
                cache_index_should = Some(codec_index);
            }
            if !capability.partial_decode {
                cache_index_must = Some(codec_index + 1);
            }
            codec_index += 1;
        }

        let cache_index = if let (Some(cache_index_must), Some(cache_index_should)) =
            (cache_index_must, cache_index_should)
        {
            Some(std::cmp::max(cache_index_must, cache_index_should))
        } else if cache_index_must.is_some() {
            cache_index_must
        } else if cache_index_should.is_some() {
            cache_index_should
        } else {
            None
        };

        Self {
            array_to_array,
            array_to_bytes,
            bytes_to_bytes,
            cache_index,
        }
    }

    /// Create a new codec chain from a list of metadata.
    ///
    /// # Errors
    /// Returns a [`PluginCreateError`] if:
    ///  - a codec could not be created,
    ///  - no array to bytes codec is supplied, or
    ///  - more than one array to bytes codec is supplied.
    pub fn from_metadata(metadatas: &[MetadataV3]) -> Result<Self, PluginCreateError> {
        let mut array_to_array: Vec<NamedArrayToArrayCodec> = vec![];
        let mut array_to_bytes: Option<NamedArrayToBytesCodec> = None;
        let mut bytes_to_bytes: Vec<NamedBytesToBytesCodec> = vec![];
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
                    array_to_array.push(NamedArrayToArrayCodec::new(
                        metadata.name().to_string(),
                        codec,
                    ));
                }
                Codec::ArrayToBytes(codec) => {
                    if array_to_bytes.is_none() {
                        array_to_bytes = Some(NamedArrayToBytesCodec::new(
                            metadata.name().to_string(),
                            codec,
                        ));
                    } else {
                        return Err(PluginCreateError::from("multiple array to bytes codecs"));
                    }
                }
                Codec::BytesToBytes(codec) => {
                    bytes_to_bytes.push(NamedBytesToBytesCodec::new(
                        metadata.name().to_string(),
                        codec,
                    ));
                }
            }
        }

        array_to_bytes.map_or_else(
            || Err(PluginCreateError::from("missing array to bytes codec")),
            |array_to_bytes| {
                Ok(Self::new_named(
                    array_to_array,
                    array_to_bytes,
                    bytes_to_bytes,
                ))
            },
        )
    }

    /// Create codec chain metadata.
    #[must_use]
    pub fn create_metadatas(&self, options: &CodecMetadataOptions) -> Vec<MetadataV3> {
        let mut metadatas =
            Vec::with_capacity(self.array_to_array.len() + 1 + self.bytes_to_bytes.len());
        for codec in &self.array_to_array {
            if let Some(configuration) = codec.configuration(options) {
                metadatas.push(MetadataV3::new_with_configuration(
                    codec.name().to_string(),
                    configuration,
                ));
            }
        }
        {
            let codec = &self.array_to_bytes;
            if let Some(configuration) = codec.configuration(options) {
                metadatas.push(MetadataV3::new_with_configuration(
                    codec.name().to_string(),
                    configuration,
                ));
            }
        }
        for codec in &self.bytes_to_bytes {
            if let Some(configuration) = codec.configuration(options) {
                metadatas.push(MetadataV3::new_with_configuration(
                    codec.name().to_string(),
                    configuration,
                ));
            }
        }
        metadatas
    }

    /// Get the array to array codecs
    #[must_use]
    pub fn array_to_array_codecs(&self) -> &[NamedArrayToArrayCodec] {
        &self.array_to_array
    }

    /// Get the array to bytes codec
    #[must_use]
    pub fn array_to_bytes_codec(&self) -> &NamedArrayToBytesCodec {
        &self.array_to_bytes
    }

    /// Get the bytes to bytes codecs
    #[must_use]
    pub fn bytes_to_bytes_codecs(&self) -> &[NamedBytesToBytesCodec] {
        &self.bytes_to_bytes
    }

    fn get_array_representations(
        &self,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
    ) -> Result<Vec<(ChunkShape, DataType, FillValue)>, CodecError> {
        let mut array_representations = Vec::with_capacity(self.array_to_array.len() + 1);
        array_representations.push((shape.to_vec(), data_type.clone(), fill_value.clone()));
        for codec in &self.array_to_array {
            let (shape, data_type, fill_value) = array_representations.last().unwrap();
            array_representations.push(codec.encoded_representation(shape, data_type, fill_value)?);
        }
        Ok(array_representations)
    }

    fn get_bytes_representations(
        &self,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
    ) -> Result<Vec<BytesRepresentation>, CodecError> {
        let mut bytes_representations = Vec::with_capacity(self.bytes_to_bytes.len() + 1);
        bytes_representations.push(
            self.array_to_bytes
                .codec()
                .encoded_representation(shape, data_type, fill_value)?,
        );
        for codec in &self.bytes_to_bytes {
            bytes_representations
                .push(codec.encoded_representation(bytes_representations.last().unwrap()));
        }
        Ok(bytes_representations)
    }
}

impl CodecTraits for CodecChain {
    fn identifier(&self) -> &'static str {
        "_zarrs_codec_chain"
    }

    /// Returns [`None`] since a codec chain does not have standard codec metadata.
    ///
    /// Note that usage of the codec chain is explicit in [`Array`](crate::array::Array) and [`CodecChain::create_metadatas()`] will call [`CodecTraits::configuration()`] from for each codec.
    fn configuration(&self, _name: &str, _options: &CodecMetadataOptions) -> Option<Configuration> {
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
impl ArrayToBytesCodecTraits for CodecChain {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self as Arc<dyn ArrayToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        mut bytes: ArrayBytes<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<ArrayBytesRaw<'a>, CodecError> {
        bytes.validate(shape.iter().map(|v| v.get()).product(), data_type)?;

        let mut shape = ChunkShape::from(shape.to_vec());
        let mut data_type = data_type.clone();
        let mut fill_value = fill_value.clone();

        // array->array
        for codec in &self.array_to_array {
            bytes = codec.encode(bytes, &shape, &data_type, &fill_value, options)?;
            shape = codec.encoded_shape(&shape)?;
            fill_value = codec.encoded_fill_value(&data_type, &fill_value)?;
            data_type = codec.encoded_data_type(&data_type)?;
        }

        // array->bytes
        let mut bytes =
            self.array_to_bytes
                .codec()
                .encode(bytes, &shape, &data_type, &fill_value, options)?;
        let mut decoded_representation =
            self.array_to_bytes
                .codec()
                .encoded_representation(&shape, &data_type, &fill_value)?;

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
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let array_representations = self.get_array_representations(shape, data_type, fill_value)?;
        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let bytes_representations = self.get_bytes_representations(shape, data_type, fill_value)?;

        // bytes->bytes
        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            bytes = codec.decode(bytes, bytes_representation, options)?;
        }

        // bytes->array
        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let mut bytes = self
            .array_to_bytes
            .codec()
            .decode(bytes, shape, data_type, fill_value, options)?;

        // array->array
        for (codec, (shape, data_type, fill_value)) in std::iter::zip(
            self.array_to_array.iter().rev(),
            array_representations.iter().rev().skip(1),
        ) {
            bytes = codec.decode(bytes, shape, data_type, fill_value, options)?;
        }

        bytes.validate(shape.iter().map(|v| v.get()).product(), data_type)?;
        Ok(bytes)
    }

    fn decode_into(
        &self,
        mut bytes: ArrayBytesRaw<'_>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        output_target: ArrayBytesDecodeIntoTarget<'_>,
        options: &CodecOptions,
    ) -> Result<(), CodecError> {
        let array_representations = self.get_array_representations(shape, data_type, fill_value)?;
        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let bytes_representations = self.get_bytes_representations(shape, data_type, fill_value)?;

        if self.bytes_to_bytes.is_empty() && self.array_to_array.is_empty() {
            // Fast path if no bytes to bytes or array to array codecs
            let (shape, data_type, fill_value) = array_representations.last().unwrap();
            return self.array_to_bytes.codec().decode_into(
                bytes,
                shape,
                data_type,
                fill_value,
                output_target,
                options,
            );
        }

        // bytes->bytes
        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            bytes = codec.decode(bytes, bytes_representation, options)?;
        }

        if self.array_to_array.is_empty() {
            // Fast path if no array to array codecs
            let (shape, data_type, fill_value) = array_representations.last().unwrap();
            return self.array_to_bytes.codec().decode_into(
                bytes,
                shape,
                data_type,
                fill_value,
                output_target,
                options,
            );
        }

        // bytes->array
        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let mut bytes = self
            .array_to_bytes
            .codec()
            .decode(bytes, shape, data_type, fill_value, options)?;

        // array->array
        for (codec, (shape, data_type, fill_value)) in std::iter::zip(
            self.array_to_array.iter().rev(),
            array_representations.iter().rev().skip(1),
        ) {
            bytes = codec.decode(bytes, shape, data_type, fill_value, options)?;
        }
        bytes.validate(shape.iter().map(|v| v.get()).product(), data_type)?;

        crate::array::array_bytes::decode_into_array_bytes_target(&bytes, output_target)
    }

    fn compact<'a>(
        &self,
        mut bytes: ArrayBytesRaw<'a>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Option<ArrayBytesRaw<'a>>, CodecError> {
        let array_representations = self.get_array_representations(shape, data_type, fill_value)?;
        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let bytes_representations = self.get_bytes_representations(shape, data_type, fill_value)?;

        // Decode through bytes_to_bytes codecs (in reverse) to get to array_to_bytes level
        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            bytes = codec.decode(bytes, bytes_representation, options)?;
        }

        // Compact at the array_to_bytes level (e.g., ShardingCodec compact)
        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let compacted = self
            .array_to_bytes
            .codec()
            .compact(bytes, shape, data_type, fill_value, options)?;

        // If compaction occurred, re-encode through bytes_to_bytes codecs
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

    fn partial_decoder(
        self: Arc<Self>,
        mut input_handle: Arc<dyn BytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        let array_representations = self.get_array_representations(shape, data_type, fill_value)?;
        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let bytes_representations = self.get_bytes_representations(shape, data_type, fill_value)?;
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
            let (shape, data_type, fill_value) = array_representations.last().unwrap();
            let codec = &self.array_to_bytes;
            codec_index += 1;
            codec.codec().clone().partial_decoder(
                input_handle,
                shape,
                data_type,
                fill_value,
                options,
            )?
        };

        for (codec, (shape, data_type, fill_value)) in std::iter::zip(
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
            input_handle = codec.codec().clone().partial_decoder(
                input_handle,
                shape,
                data_type,
                fill_value,
                options,
            )?;
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
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        let array_representations = self.get_array_representations(shape, data_type, fill_value)?;
        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let bytes_representations = self.get_bytes_representations(shape, data_type, fill_value)?;

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

        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let mut input_output_handle = self.array_to_bytes.codec().clone().partial_encoder(
            input_output_handle,
            shape,
            data_type,
            fill_value,
            options,
        )?;

        if self.array_to_array.is_empty() {
            return Ok(input_output_handle);
        }

        for (codec, (shape, data_type, fill_value)) in std::iter::zip(
            self.array_to_array.iter().rev(),
            array_representations.iter().rev().skip(1),
        ) {
            input_output_handle = Arc::clone(codec).partial_encoder(
                input_output_handle,
                shape,
                data_type,
                fill_value,
                options,
            )?;
        }

        Ok(input_output_handle)
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        mut input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        let array_representations = self.get_array_representations(shape, data_type, fill_value)?;
        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let bytes_representations = self.get_bytes_representations(shape, data_type, fill_value)?;

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
                .codec()
                .clone()
                .async_partial_decoder(input_handle, bytes_representation, options)
                .await?;
        }

        if Some(codec_index) == self.cache_index {
            input_handle =
                Arc::new(BytesPartialDecoderCache::async_new(&*input_handle, options).await?);
        }

        let mut input_handle = {
            let (shape, data_type, fill_value) = array_representations.last().unwrap();
            let codec = &self.array_to_bytes;
            codec_index += 1;
            codec
                .codec()
                .clone()
                .async_partial_decoder(input_handle, shape, data_type, fill_value, options)
                .await?
        };

        for (codec, (shape, data_type, fill_value)) in std::iter::zip(
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
                .codec()
                .clone()
                .async_partial_decoder(input_handle, shape, data_type, fill_value, options)
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
        data_type: &DataType,
        fill_value: &FillValue,
        options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialEncoderTraits>, CodecError> {
        let array_representations = self.get_array_representations(shape, data_type, fill_value)?;
        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let bytes_representations = self.get_bytes_representations(shape, data_type, fill_value)?;

        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            input_output_handle = Arc::clone(codec)
                .async_partial_encoder(input_output_handle, bytes_representation, options)
                .await?;
        }

        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let mut input_output_handle = self
            .array_to_bytes
            .codec()
            .clone()
            .async_partial_encoder(input_output_handle, shape, data_type, fill_value, options)
            .await?;

        if self.array_to_array.is_empty() {
            return Ok(input_output_handle);
        }

        for (codec, (shape, data_type, fill_value)) in std::iter::zip(
            self.array_to_array.iter().rev(),
            array_representations.iter().rev().skip(1),
        ) {
            input_output_handle = Arc::clone(codec)
                .async_partial_encoder(input_output_handle, shape, data_type, fill_value, options)
                .await?;
        }

        Ok(input_output_handle)
    }

    fn encoded_representation(
        &self,
        shape: &[NonZeroU64],
        data_type: &DataType,
        fill_value: &FillValue,
    ) -> Result<BytesRepresentation, CodecError> {
        let mut shape = ChunkShape::from(shape.to_vec());
        let mut data_type = data_type.clone();
        let mut fill_value = fill_value.clone();
        for codec in &self.array_to_array {
            shape = codec.encoded_shape(&shape)?;
            fill_value = codec.encoded_fill_value(&data_type, &fill_value)?;
            data_type = codec.encoded_data_type(&data_type)?;
        }

        let mut bytes_representation =
            self.array_to_bytes
                .codec()
                .encoded_representation(&shape, &data_type, &fill_value)?;

        for codec in &self.bytes_to_bytes {
            bytes_representation = codec.encoded_representation(&bytes_representation);
        }

        Ok(bytes_representation)
    }
}

impl ArrayCodecTraits for CodecChain {
    fn recommended_concurrency(
        &self,
        shape: &[NonZeroU64],
        data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        let mut concurrency_min = usize::MAX;
        let mut concurrency_max = 0;

        // Create a dummy fill value for computing array representations.
        // The fill value doesn't affect concurrency computation.
        let fill_value: FillValue = vec![0u8; data_type.fixed_size().unwrap_or(0)].into();
        let array_representations =
            self.get_array_representations(shape, data_type, &fill_value)?;
        let (shape, data_type, fill_value) = array_representations.last().unwrap();
        let bytes_representations = self.get_bytes_representations(shape, data_type, fill_value)?;

        // bytes->bytes
        for (codec, bytes_representation) in std::iter::zip(
            self.bytes_to_bytes.iter().rev(),
            bytes_representations.iter().rev().skip(1),
        ) {
            let recommended_concurrency = &codec.recommended_concurrency(bytes_representation)?;
            concurrency_min = std::cmp::min(concurrency_min, recommended_concurrency.min());
            concurrency_max = std::cmp::max(concurrency_max, recommended_concurrency.max());
        }

        let (shape, data_type, _fill_value) = array_representations.last().unwrap();
        let recommended_concurrency = &self
            .array_to_bytes
            .codec()
            .recommended_concurrency(shape, data_type)?;
        concurrency_min = std::cmp::min(concurrency_min, recommended_concurrency.min());
        concurrency_max = std::cmp::max(concurrency_max, recommended_concurrency.max());

        // array->array
        for (codec, (shape, data_type, _fill_value)) in std::iter::zip(
            self.array_to_array.iter().rev(),
            array_representations.iter().rev().skip(1),
        ) {
            let recommended_concurrency = codec.recommended_concurrency(shape, data_type)?;
            concurrency_min = std::cmp::min(concurrency_min, recommended_concurrency.min());
            concurrency_max = std::cmp::max(concurrency_max, recommended_concurrency.max());
        }

        let recommended_concurrency = RecommendedConcurrency::new(
            std::cmp::min(concurrency_min, concurrency_max)
                ..std::cmp::max(concurrency_max, concurrency_max),
        );

        Ok(recommended_concurrency)
    }

    fn partial_decode_granularity(&self, shape: &[NonZeroU64]) -> ChunkShape {
        if let Some(array_to_array) = self.array_to_array.first() {
            array_to_array.partial_decode_granularity(shape)
        } else {
            self.array_to_bytes
                .codec()
                .partial_decode_granularity(shape)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;
    use crate::{
        array::{ChunkShapeTraits, data_types},
        array_subset::ArraySubset,
    };

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
        decoded_region: &ArraySubset,
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
        let codec = Arc::new(CodecChain::from_metadata(&codec_configurations).unwrap());

        let encoded = codec
            .encode(
                bytes.clone(),
                shape,
                data_type,
                fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        let decoded = codec
            .decode(
                encoded.clone(),
                shape,
                data_type,
                fill_value,
                &CodecOptions::default(),
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

        let input_handle = Arc::new(encoded);
        let partial_decoder = codec
            .clone()
            .partial_decoder(
                input_handle.clone(),
                shape,
                data_type,
                fill_value,
                &CodecOptions::default(),
            )
            .unwrap();
        assert_eq!(partial_decoder.size_held(), decoded.size()); // codec chain caches with most decompression codecs
        let decoded_partial_chunk = partial_decoder
            .partial_decode(decoded_region, &CodecOptions::default())
            .unwrap();

        let decoded_partial_chunk: Vec<f32> = decoded_partial_chunk
            .into_fixed()
            .unwrap()
            .chunks(size_of::<f32>())
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
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
            &data_types::float32(),
            &FillValue::from(0f32),
            elements,
            JSON_BYTES,
            &decoded_region,
            decoded_partial_chunk_true,
        );
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
            &data_types::float32(),
            &FillValue::from(0f32),
            elements,
            JSON_PCODEC,
            &decoded_region,
            decoded_partial_chunk_true,
        );
    }
}
