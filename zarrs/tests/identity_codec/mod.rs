//! An identity array-to-array test codec, shared by the encoded subchunk tests.

// Each test crate that includes this module uses only some of it
#![allow(dead_code)]

use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use zarrs::array::{
    ArrayBytes, ArrayIndices, ChunkGrid, ChunkShape, CodecCreateError, CodecError,
    CodecMetadataOptions, CodecOptions, CodecTraits, DataType, FillValue, RecommendedConcurrency,
};
use zarrs::metadata::Configuration;
use zarrs_chunk_grid::ChunkGridCreateError;
use zarrs_codec::{
    ArrayCodecTraits, ArrayToArrayCodecTraits, ChunkGridDecoded, ChunkGridDecodedRef,
    ChunkGridEncoded, ChunkGridEncodedRef, CodecSpecificOptions, PartialDecoderCapability,
    PartialEncoderCapability, UnboundArrayToArrayCodecTraits,
};
use zarrs_plugin::{ExtensionName, ZarrVersion};

/// Counts of the calls to the subchunk mapping methods of an [`IdentityCodec`].
#[derive(Debug, Default)]
pub(crate) struct IdentityCodecCalls {
    /// Calls to `decoded_local_subchunk_grid`, made when resolving a subchunk grid hierarchy.
    pub(crate) decoded_local_subchunk_grid: AtomicUsize,
    /// Calls to `encoded_coordinates`, made when mapping a subchunk subset.
    pub(crate) encoded_coordinates: AtomicUsize,
}

impl IdentityCodecCalls {
    /// Return and reset the counts of calls to `decoded_local_subchunk_grid` and `encoded_coordinates`.
    pub(crate) fn take(&self) -> (usize, usize) {
        (
            self.decoded_local_subchunk_grid.swap(0, Ordering::Relaxed),
            self.encoded_coordinates.swap(0, Ordering::Relaxed),
        )
    }
}

/// An identity array-to-array codec that maps subchunks through its subchunking traits, like a third-party codec
/// with the default partial decoder and encoder.
///
/// Decoding panics while `forbid_decode` is set.
#[derive(Debug)]
pub(crate) struct IdentityCodec {
    partial_decode: bool,
    pub(crate) forbid_decode: Arc<AtomicBool>,
    pub(crate) calls: Arc<IdentityCodecCalls>,
}

impl IdentityCodec {
    /// Create an identity codec. If not `partial_decode`, partial decoders of its codec chain cache the decoded chunk.
    pub(crate) fn new(partial_decode: bool) -> Self {
        Self {
            partial_decode,
            forbid_decode: Arc::default(),
            calls: Arc::default(),
        }
    }
}

#[derive(Debug)]
struct IdentityCodecBound {
    forbid_decode: Arc<AtomicBool>,
    calls: Arc<IdentityCodecCalls>,
    data_type: DataType,
    fill_value: Option<FillValue>,
}

impl ExtensionName for IdentityCodec {
    fn name(&self, _version: ZarrVersion) -> Option<Cow<'static, str>> {
        None
    }
}

impl CodecTraits for IdentityCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        None
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: self.partial_decode,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl UnboundArrayToArrayCodecTraits for IdentityCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToArrayCodecTraits> {
        self
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: Option<FillValue>,
        _codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        Ok(Arc::new(IdentityCodecBound {
            forbid_decode: self.forbid_decode.clone(),
            calls: self.calls.clone(),
            data_type,
            fill_value,
        }))
    }
}

impl ArrayCodecTraits for IdentityCodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> Option<&FillValue> {
        self.fill_value.as_ref()
    }

    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl zarrs_codec::ArrayToArrayCodecSubchunkingTraits for IdentityCodecBound {
    fn encoded_chunk_grid(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<ChunkGridEncoded, ChunkGridCreateError> {
        Ok(match decoded_chunk_grid {
            ChunkGridDecodedRef::Array(grid) => ChunkGridEncoded::Array(grid.clone()),
            ChunkGridDecodedRef::ChunkLocal => ChunkGridEncoded::ChunkLocal,
            ChunkGridDecodedRef::None => ChunkGridEncoded::None,
        })
    }

    fn decoded_subchunk_grid(
        &self,
        _decoded_chunk_grid: ChunkGridDecodedRef<'_>,
        encoded_subchunk_grid: ChunkGridEncodedRef<'_>,
    ) -> Result<ChunkGridDecoded, ChunkGridCreateError> {
        Ok(encoded_subchunk_grid.into())
    }

    fn decoded_local_subchunk_grid(
        &self,
        _decoded_shape: &[NonZeroU64],
        encoded_subchunk_grid: &ChunkGrid,
    ) -> Result<Option<ChunkGrid>, CodecError> {
        self.calls
            .decoded_local_subchunk_grid
            .fetch_add(1, Ordering::Relaxed);
        Ok(Some(encoded_subchunk_grid.clone()))
    }

    fn encoded_coordinates(
        &self,
        _decoded_shape: &[NonZeroU64],
        decoded_coordinates: &[u64],
    ) -> Result<ArrayIndices, CodecError> {
        self.calls
            .encoded_coordinates
            .fetch_add(1, Ordering::Relaxed);
        Ok(decoded_coordinates.into())
    }
}

impl ArrayToArrayCodecTraits for IdentityCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.data_type
    }

    fn encoded_fill_value(&self) -> Option<&FillValue> {
        self.fill_value.as_ref()
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        Ok(decoded_shape.to_vec())
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        assert!(
            !self.forbid_decode.load(Ordering::Relaxed),
            "a chunk was decoded"
        );
        Ok(bytes)
    }
}
