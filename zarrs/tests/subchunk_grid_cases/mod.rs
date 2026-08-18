//! Shared scenarios for the chunk-local subchunk grid tests.
//!
//! Every case wraps a `sharding_indexed` array-to-bytes codec (the source of subchunks) in an
//! optional array-to-array codec, so that the chunk-local subchunk grid has to be forwarded or
//! remapped by that codec's partial decoder.
#![allow(dead_code)]

use std::num::NonZeroU64;
use std::sync::Arc;

use zarrs::array::builder::ArrayBuilderFillValue;
use zarrs::array::codec::array_to_array::reshape::ReshapeShape;
use zarrs::array::codec::array_to_bytes::sharding::ShardingCodecBuilder;
use zarrs::array::codec::{
    BitroundCodec, CastValueCodec, ReshapeCodec, SqueezeCodec, TransposeCodec, TransposeOrder,
};
use zarrs::array::{
    ArrayBuilder, ArrayBytes, ArrayToArrayCodecTraits, ChunkGrid, CodecCreateError,
    CodecMetadataOptions, CodecOptions, CodecTraits, DataType, DataTypeSize, FillValue,
    RecommendedConcurrency, UnboundArrayToArrayCodecTraits, data_type,
};
use zarrs::metadata::Configuration;
use zarrs_codec::{
    ArrayCodecTraits, ArrayToArrayCodecSubchunkingIdentityTraits, CodecError,
    PartialDecoderCapability, PartialEncoderCapability,
};
use zarrs_plugin::ZarrVersion;

pub(crate) const fn nz(value: u64) -> NonZeroU64 {
    NonZeroU64::new(value).unwrap()
}

/// An identity array-to-array codec that deliberately does not implement `partial_decoder`.
///
/// It therefore always resolves to [`CodecPartialDefault`](zarrs_codec::CodecPartialDefault),
/// unlike any real codec, which may gain a bespoke partial codec at any time.
#[derive(Debug)]
pub(crate) struct IdentityCodec;

#[derive(Debug)]
struct IdentityCodecBound {
    data_type: DataType,
    fill_value: FillValue,
}

zarrs_plugin::impl_extension_aliases!(IdentityCodec, v3: "zarrs.test.identity");

impl CodecTraits for IdentityCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(Configuration::default())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        // The default array-to-array partial decoder supports partial read/decode.
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
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
        fill_value: FillValue,
    ) -> Result<Arc<dyn ArrayToArrayCodecTraits>, CodecCreateError> {
        Ok(Arc::new(IdentityCodecBound {
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

    fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl ArrayToArrayCodecSubchunkingIdentityTraits for IdentityCodecBound {}

impl ArrayToArrayCodecTraits for IdentityCodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self
    }

    fn encoded_data_type(&self) -> &DataType {
        &self.data_type
    }

    fn encoded_fill_value(&self) -> &FillValue {
        &self.fill_value
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
        Ok(bytes)
    }
}

pub(crate) struct Case {
    /// Identifies the case in assertion failures.
    pub(crate) name: &'static str,
    /// The array-to-array codec under test, applied before the sharding codec.
    pub(crate) codec: Option<Arc<dyn UnboundArrayToArrayCodecTraits>>,
    pub(crate) data_type: DataType,
    pub(crate) fill_value: ArrayBuilderFillValue,
    /// The data type seen by the sharding codec, i.e. after `codec`.
    pub(crate) encoded_data_type: DataType,
    pub(crate) array_shape: Vec<u64>,
    pub(crate) chunk_shape: Vec<u64>,
    /// The sharding inner chunk shape, in the encoded representation.
    pub(crate) encoded_subchunk_shape: Vec<NonZeroU64>,
    /// The chunk to resolve the local subchunk grid for. It is written by the tests.
    pub(crate) chunk_indices: Vec<u64>,
    /// A chunk that the tests leave unwritten, for the absent/`exists` assertions.
    pub(crate) absent_chunk_indices: Vec<u64>,
    /// The expected local subchunk shape, in the decoded chunk representation.
    pub(crate) local_subchunk_shape: Vec<NonZeroU64>,
}

impl Case {
    /// An [`ArrayBuilder`] for this case, ready to `build` against a sync or async store.
    pub(crate) fn builder(&self) -> ArrayBuilder {
        let sharding =
            ShardingCodecBuilder::new(self.encoded_subchunk_shape.clone(), &self.encoded_data_type)
                .build_arc();
        let mut builder = ArrayBuilder::new(
            self.array_shape.clone(),
            self.chunk_shape.clone(),
            self.data_type.clone(),
            self.fill_value.clone(),
        );
        if let Some(codec) = &self.codec {
            builder.array_to_array_codecs(vec![codec.clone()]);
        }
        builder.array_to_bytes_codec(sharding);
        builder
    }

    /// Zeroed chunk bytes, whatever the data type.
    ///
    /// Every case uses a non-zero fill value, so a zeroed chunk is never elided as empty.
    pub(crate) fn zero_chunk_bytes(&self) -> ArrayBytes<'static> {
        let DataTypeSize::Fixed(element_size) = self.data_type.size() else {
            panic!("{}: expected a fixed-size data type", self.name);
        };
        let num_elements: u64 = self.chunk_shape.iter().product();
        ArrayBytes::new_flen(vec![
            0u8;
            usize::try_from(num_elements).unwrap() * element_size
        ])
    }

    /// Assert that `grid` is the expected chunk-local subchunk grid.
    pub(crate) fn assert_local_grid(&self, grid: Option<&ChunkGrid>) {
        let grid = grid.unwrap_or_else(|| panic!("{}: expected a local subchunk grid", self.name));
        assert_eq!(
            grid.array_shape(),
            self.chunk_shape,
            "{}: local grid covers the decoded chunk",
            self.name
        );
        assert_eq!(
            grid.chunk_shape(&vec![0; self.chunk_shape.len()]).unwrap(),
            Some(self.local_subchunk_shape.clone()),
            "{}: local subchunk shape",
            self.name
        );
    }
}

/// A sharded array with no array-to-array codec, used by the partial encoder and cache tests.
pub(crate) fn plain_sharding() -> Case {
    Case {
        name: "sharding",
        codec: None,
        data_type: data_type::uint16(),
        fill_value: 1u16.into(),
        encoded_data_type: data_type::uint16(),
        array_shape: vec![8, 4],
        chunk_shape: vec![4, 4],
        encoded_subchunk_shape: vec![nz(2), nz(2)],
        chunk_indices: vec![0, 0],
        absent_chunk_indices: vec![1, 0],
        local_subchunk_shape: vec![nz(2), nz(2)],
    }
}

/// One case per array-to-array partial decoder that has to propagate a subchunk grid.
pub(crate) fn cases() -> Vec<Case> {
    vec![
        plain_sharding(),
        Case {
            name: "bitround",
            codec: Some(Arc::new(
                BitroundCodec::new_with_configuration(
                    &serde_json::from_str(r#"{"keepbits": 8}"#).unwrap(),
                )
                .unwrap(),
            )),
            data_type: data_type::float32(),
            fill_value: 1.0f32.into(),
            encoded_data_type: data_type::float32(),
            array_shape: vec![8, 4],
            chunk_shape: vec![4, 4],
            encoded_subchunk_shape: vec![nz(2), nz(2)],
            chunk_indices: vec![0, 0],
            absent_chunk_indices: vec![1, 0],
            local_subchunk_shape: vec![nz(2), nz(2)],
        },
        Case {
            name: "cast_value",
            codec: Some(Arc::new(
                CastValueCodec::new_with_configuration(
                    &serde_json::from_str(r#"{"data_type": "uint16"}"#).unwrap(),
                )
                .unwrap(),
            )),
            data_type: data_type::uint8(),
            fill_value: 1u8.into(),
            encoded_data_type: data_type::uint16(),
            array_shape: vec![8, 4],
            chunk_shape: vec![4, 4],
            encoded_subchunk_shape: vec![nz(2), nz(2)],
            chunk_indices: vec![0, 0],
            absent_chunk_indices: vec![1, 0],
            local_subchunk_shape: vec![nz(2), nz(2)],
        },
        // `IdentityCodec` has no bespoke partial decoder, so it exercises `CodecPartialDefault`.
        Case {
            name: "identity",
            codec: Some(Arc::new(IdentityCodec)),
            data_type: data_type::uint16(),
            fill_value: 1u16.into(),
            encoded_data_type: data_type::uint16(),
            array_shape: vec![8, 4],
            chunk_shape: vec![4, 4],
            encoded_subchunk_shape: vec![nz(2), nz(2)],
            chunk_indices: vec![0, 0],
            absent_chunk_indices: vec![1, 0],
            local_subchunk_shape: vec![nz(2), nz(2)],
        },
        Case {
            name: "reshape",
            codec: Some(Arc::new(ReshapeCodec::new(
                ReshapeShape::new([nz(12).into()]).unwrap(),
            ))),
            data_type: data_type::uint16(),
            fill_value: 1u16.into(),
            encoded_data_type: data_type::uint16(),
            array_shape: vec![4, 6],
            chunk_shape: vec![2, 6],
            encoded_subchunk_shape: vec![nz(3)],
            chunk_indices: vec![1, 0],
            absent_chunk_indices: vec![0, 0],
            local_subchunk_shape: vec![nz(1), nz(3)],
        },
        Case {
            name: "squeeze",
            codec: Some(Arc::new(SqueezeCodec::new())),
            data_type: data_type::uint16(),
            fill_value: 1u16.into(),
            encoded_data_type: data_type::uint16(),
            array_shape: vec![2, 4],
            chunk_shape: vec![1, 4],
            encoded_subchunk_shape: vec![nz(2)],
            chunk_indices: vec![1, 0],
            absent_chunk_indices: vec![0, 0],
            local_subchunk_shape: vec![nz(1), nz(2)],
        },
        Case {
            name: "transpose",
            codec: Some(Arc::new(TransposeCodec::new(
                TransposeOrder::new(&[1, 0]).unwrap(),
            ))),
            data_type: data_type::uint16(),
            fill_value: 1u16.into(),
            encoded_data_type: data_type::uint16(),
            array_shape: vec![8, 6],
            chunk_shape: vec![4, 6],
            encoded_subchunk_shape: vec![nz(3), nz(2)],
            chunk_indices: vec![0, 0],
            absent_chunk_indices: vec![1, 0],
            local_subchunk_shape: vec![nz(2), nz(3)],
        },
    ]
}
