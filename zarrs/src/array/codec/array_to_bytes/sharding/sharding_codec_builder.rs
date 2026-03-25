use std::sync::Arc;

use super::{ShardingCodec, ShardingIndexLocation};
use crate::array::codec::{BytesCodec, CodecChain, Crc32cCodec, default_array_to_bytes_codec};
use crate::array::{ChunkShape, DataType};
use zarrs_codec::{ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesToBytesCodecTraits};

/// A [`ShardingCodec`] builder.
///
/// By default, the subchunks are encoded with the default codec for the data type (see [`default_array_to_bytes_codec`]).
/// The index is encoded with the `bytes` codec with native endian encoding, additionally with the `crc32c checksum` codec (if the `crc32c` feature is enabled).
///
/// Use the methods in the `sharding` codec builder to change the configuration away from these defaults, and then build the `sharding` codec with [`build`](ShardingCodecBuilder::build).
#[derive(Debug)]
pub struct ShardingCodecBuilder {
    subchunk_shape: ChunkShape,
    index_array_to_bytes_codec: Arc<dyn ArrayToBytesCodecTraits>,
    index_bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    array_to_array_codecs: Vec<Arc<dyn ArrayToArrayCodecTraits>>,
    array_to_bytes_codec: Arc<dyn ArrayToBytesCodecTraits>,
    bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    index_location: ShardingIndexLocation,
}

impl ShardingCodecBuilder {
    /// Create a new `sharding` codec builder.
    ///
    /// The default subchunk array-to-bytes codec is chosen based on the data type
    /// (see [`default_array_to_bytes_codec`]).
    #[must_use]
    pub fn new(subchunk_shape: ChunkShape, data_type: &DataType) -> Self {
        Self {
            subchunk_shape,
            index_array_to_bytes_codec: Arc::<BytesCodec>::default(),
            index_bytes_to_bytes_codecs: vec![
                #[cfg(feature = "crc32c")]
                Arc::new(Crc32cCodec::new()),
            ],
            array_to_array_codecs: Vec::default(),
            array_to_bytes_codec: default_array_to_bytes_codec(data_type),
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
        self.index_array_to_bytes_codec = index_array_to_bytes_codec;
        self
    }

    /// Set the index bytes to bytes codecs.
    ///
    /// If left unmodified, the index will be encoded with the `crc32c checksum` codec (if supported).
    pub fn index_bytes_to_bytes_codecs(
        &mut self,
        index_bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    ) -> &mut Self {
        self.index_bytes_to_bytes_codecs = index_bytes_to_bytes_codecs;
        self
    }

    /// Set the subchunk array to array codecs.
    ///
    /// If left unmodified, no array to array codecs will be applied for the subchunks.
    pub fn array_to_array_codecs(
        &mut self,
        array_to_array_codecs: Vec<Arc<dyn ArrayToArrayCodecTraits>>,
    ) -> &mut Self {
        self.array_to_array_codecs = array_to_array_codecs;
        self
    }

    /// Set the subchunk array to bytes codec.
    ///
    /// If left unmodified, the subchunks will be encoded with the default codec for the data type
    /// (see [`default_array_to_bytes_codec`]).
    pub fn array_to_bytes_codec(
        &mut self,
        array_to_bytes_codec: Arc<dyn ArrayToBytesCodecTraits>,
    ) -> &mut Self {
        self.array_to_bytes_codec = array_to_bytes_codec;
        self
    }

    /// Set the subchunk bytes to bytes codecs.
    ///
    /// If left unmodified, no bytes to bytes codecs will be applied for the subchunks.
    pub fn bytes_to_bytes_codecs(
        &mut self,
        bytes_to_bytes_codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
    ) -> &mut Self {
        self.bytes_to_bytes_codecs = bytes_to_bytes_codecs;
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
        let inner_codecs = Arc::new(CodecChain::new(
            self.array_to_array_codecs.clone(),
            self.array_to_bytes_codec.clone(),
            self.bytes_to_bytes_codecs.clone(),
        ));
        let index_codecs = Arc::new(CodecChain::new(
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
