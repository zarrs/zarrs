use std::sync::Arc;

use codec::CodecChain;

use super::{ShardingCodec, ShardingIndexLocation};
use crate::{
    array::{
        ChunkShape, DataType,
        codec::{
            self, ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesToBytesCodecTraits,
            NamedArrayToArrayCodec, NamedArrayToBytesCodec, NamedBytesToBytesCodec, NamedCodec,
            default_array_to_bytes_codec,
        },
    },
    config::global_config,
};

/// A [`ShardingCodec`] builder.
///
/// By default, the inner chunks are encoded with the default codec for the data type (see [`default_array_to_bytes_codec`]).
/// The index is encoded with the `bytes` codec with native endian encoding, additionally with the `crc32c checksum` codec (if the `crc32c` feature is enabled).
///
/// Use the methods in the `sharding` codec builder to change the configuration away from these defaults, and then build the `sharding` codec with [`build`](ShardingCodecBuilder::build).
#[derive(Debug)]
pub struct ShardingCodecBuilder {
    subchunk_shape: ChunkShape,
    index_array_to_bytes_codec: NamedArrayToBytesCodec,
    index_bytes_to_bytes_codecs: Vec<NamedBytesToBytesCodec>,
    array_to_array_codecs: Vec<NamedArrayToArrayCodec>,
    array_to_bytes_codec: NamedArrayToBytesCodec,
    bytes_to_bytes_codecs: Vec<NamedBytesToBytesCodec>,
    index_location: ShardingIndexLocation,
}

impl ShardingCodecBuilder {
    /// Create a new `sharding` codec builder.
    ///
    /// The default inner chunk array-to-bytes codec is chosen based on the data type
    /// (see [`default_array_to_bytes_codec`]).
    #[must_use]
    pub fn new(subchunk_shape: ChunkShape, data_type: &DataType) -> Self {
        let config = global_config();
        let aliases = config.codec_aliases_v3();
        Self {
            subchunk_shape,
            index_array_to_bytes_codec: NamedCodec::new_default_name(
                Arc::<codec::BytesCodec>::default(),
                aliases,
            ),
            index_bytes_to_bytes_codecs: vec![
                #[cfg(feature = "crc32c")]
                NamedCodec::new_default_name(Arc::new(codec::Crc32cCodec::new()), aliases),
            ],
            array_to_array_codecs: Vec::default(),
            array_to_bytes_codec: default_array_to_bytes_codec(data_type, aliases),
            bytes_to_bytes_codecs: Vec::default(),
            index_location: ShardingIndexLocation::default(),
        }
    }

    /// Set the index array to bytes codec.
    ///
    /// If left unmodified, the index will be encoded with the `bytes` codec with native endian encoding.
    pub fn index_array_to_bytes_codec(
        &mut self,
        index_array_to_bytes_codec: Arc<dyn ArrayToBytesCodecTraits>,
    ) -> &mut Self {
        let config = global_config();
        let aliases = config.codec_aliases_v3();
        self.index_array_to_bytes_codec =
            NamedCodec::new_default_name(index_array_to_bytes_codec, aliases);
        self
    }

    /// Set the index bytes to bytes codecs.
    ///
    /// If left unmodified, the index will be encoded with the `crc32c checksum` codec (if supported).
    pub fn index_bytes_to_bytes_codecs(
        &mut self,
        index_bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    ) -> &mut Self {
        let config = global_config();
        let aliases = config.codec_aliases_v3();
        self.index_bytes_to_bytes_codecs = index_bytes_to_bytes_codecs
            .into_iter()
            .map(|codec| NamedCodec::new_default_name(codec, aliases))
            .collect();
        self
    }

    /// Set the index array to bytes codec with non-default names.
    ///
    /// If left unmodified, the index will be encoded with the `bytes` codec with native endian encoding.
    pub fn index_array_to_bytes_codec_named(
        &mut self,
        index_array_to_bytes_codec: impl Into<NamedArrayToBytesCodec>,
    ) -> &mut Self {
        self.index_array_to_bytes_codec = index_array_to_bytes_codec.into();
        self
    }

    /// Set the inner chunk array to array codecs.
    ///
    /// If left unmodified, no array to array codecs will be applied for the inner chunks.
    pub fn array_to_array_codecs(
        &mut self,
        array_to_array_codecs: Vec<Arc<dyn ArrayToArrayCodecTraits>>,
    ) -> &mut Self {
        let config = global_config();
        let aliases = config.codec_aliases_v3();
        self.array_to_array_codecs = array_to_array_codecs
            .into_iter()
            .map(|codec| NamedCodec::new_default_name(codec, aliases))
            .collect();
        self
    }

    /// Set the index bytes to bytes codecs with non-default names.
    ///
    /// If left unmodified, the index will be encoded with the `crc32c checksum` codec (if supported).
    pub fn index_bytes_to_bytes_codecs_named(
        &mut self,
        index_bytes_to_bytes_codecs: Vec<impl Into<NamedBytesToBytesCodec>>,
    ) -> &mut Self {
        self.index_bytes_to_bytes_codecs = index_bytes_to_bytes_codecs
            .into_iter()
            .map(Into::into)
            .collect();
        self
    }

    /// Set the inner chunk array to bytes codec.
    ///
    /// If left unmodified, the inner chunks will be encoded with the default codec for the data type
    /// (see [`default_array_to_bytes_codec`]).
    pub fn array_to_bytes_codec(
        &mut self,
        array_to_bytes_codec: Arc<dyn ArrayToBytesCodecTraits>,
    ) -> &mut Self {
        let config = global_config();
        let aliases = config.codec_aliases_v3();
        self.array_to_bytes_codec = NamedCodec::new_default_name(array_to_bytes_codec, aliases);
        self
    }

    /// Set the inner chunk array to array codecs with non-default names.
    ///
    /// If left unmodified, no array to array codecs will be applied for the inner chunks.
    pub fn array_to_array_codecs_named(
        &mut self,
        array_to_array_codecs: Vec<impl Into<NamedArrayToArrayCodec>>,
    ) -> &mut Self {
        self.array_to_array_codecs = array_to_array_codecs.into_iter().map(Into::into).collect();
        self
    }

    /// Set the inner chunk array to bytes codec with a non-default name.
    ///
    /// If left unmodified, the inner chunks will be encoded with the default codec for the data type
    /// (see [`default_array_to_bytes_codec`]).
    pub fn array_to_bytes_codec_named(
        &mut self,
        array_to_bytes_codec: impl Into<NamedArrayToBytesCodec>,
    ) -> &mut Self {
        self.array_to_bytes_codec = array_to_bytes_codec.into();
        self
    }

    /// Set the inner chunk bytes to bytes codecs.
    ///
    /// If left unmodified, no bytes to bytes codecs will be applied for the inner chunks.
    pub fn bytes_to_bytes_codecs(
        &mut self,
        bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    ) -> &mut Self {
        let config = global_config();
        let aliases = config.codec_aliases_v3();
        self.bytes_to_bytes_codecs = bytes_to_bytes_codecs
            .into_iter()
            .map(|codec| NamedCodec::new_default_name(codec, aliases))
            .collect();
        self
    }

    /// Set the inner chunk bytes to bytes codecs.
    ///
    /// If left unmodified, no bytes to bytes codecs will be applied for the inner chunks.
    pub fn bytes_to_bytes_codecs_named(
        &mut self,
        bytes_to_bytes_codecs: Vec<impl Into<NamedBytesToBytesCodec>>,
    ) -> &mut Self {
        self.bytes_to_bytes_codecs = bytes_to_bytes_codecs.into_iter().map(Into::into).collect();
        self
    }

    /// Set the index location.
    ///
    /// If left unmodified, defaults to the end of the shard.
    pub fn index_location(&mut self, index_location: ShardingIndexLocation) -> &mut Self {
        self.index_location = index_location;
        self
    }

    /// Build into a [`ShardingCodec`].
    #[must_use]
    pub fn build(&self) -> ShardingCodec {
        let inner_codecs = Arc::new(CodecChain::new_named(
            self.array_to_array_codecs.clone(),
            self.array_to_bytes_codec.clone(),
            self.bytes_to_bytes_codecs.clone(),
        ));
        let index_codecs = Arc::new(CodecChain::new_named(
            vec![],
            self.index_array_to_bytes_codec.clone(),
            self.index_bytes_to_bytes_codecs.clone(),
        ));
        ShardingCodec::new(
            self.subchunk_shape.clone(),
            inner_codecs,
            index_codecs,
            self.index_location,
        )
    }

    /// Build into an [`Arc<ShardingCodec>`].
    #[must_use]
    pub fn build_arc(&self) -> Arc<ShardingCodec> {
        Arc::new(self.build())
    }
}
