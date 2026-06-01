#![allow(missing_docs)]
#![cfg(all(feature = "sharding", feature = "transpose"))]

use std::num::NonZeroU64;
use std::sync::Arc;
use zarrs::array::codec::{
    BytesCodec, FixedScaleOffsetCodec, ReshapeCodec, ShardingCodecBuilder, SqueezeCodec,
    TransposeCodec,
};
use zarrs::array::{
    ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecTraits, ChunkShape, CodecChain, CodecError,
    CodecMetadataOptions, CodecOptions, CodecTraits, DataType, FillValue, data_type,
};
use zarrs_codec::{PartialDecoderCapability, PartialEncoderCapability, RecommendedConcurrency};
use zarrs_metadata::Configuration;
use zarrs_metadata_ext::codec::reshape::ReshapeShape;
use zarrs_metadata_ext::codec::transpose::TransposeOrder;
use zarrs_plugin::ZarrVersion;

fn shape(dims: &[u64]) -> ChunkShape {
    dims.iter().map(|&n| NonZeroU64::new(n).unwrap()).collect()
}
fn transpose(order: &[usize]) -> Arc<dyn ArrayToArrayCodecTraits> {
    Arc::new(TransposeCodec::new(TransposeOrder::new(order).unwrap()))
}
fn chain(codecs: Vec<Arc<dyn ArrayToArrayCodecTraits>>, subchunk: &[u64]) -> CodecChain {
    CodecChain::new(
        codecs,
        ShardingCodecBuilder::new(shape(subchunk), &data_type::uint8()).build_arc(),
        vec![],
    )
}

#[test]
fn codec_chain_partial_decode_granularity_composes_shapes() {
    let cases = [
        (
            chain(
                vec![
                    Arc::new(
                        FixedScaleOffsetCodec::new_with_configuration(
                            &serde_json::from_value(
                                serde_json::json!({"offset": 0.0, "scale": 1.0, "dtype": "|u1"}),
                            )
                            .unwrap(),
                        )
                        .unwrap(),
                    ),
                    transpose(&[1, 0]),
                ],
                &[3, 2],
            ),
            shape(&[4, 6]),
            shape(&[2, 3]),
        ),
        (
            chain(vec![transpose(&[1, 0]), transpose(&[1, 0])], &[2, 3]),
            shape(&[8, 12]),
            shape(&[2, 3]),
        ),
        (
            chain(vec![transpose(&[1, 0])], &[3, 2]),
            shape(&[8, 12]),
            shape(&[2, 3]),
        ),
        (
            chain(
                vec![transpose(&[1, 2, 0]), transpose(&[1, 2, 0])],
                &[4, 2, 3],
            ),
            shape(&[4, 6, 8]),
            shape(&[2, 3, 4]),
        ),
        (
            chain(
                vec![Arc::new(SqueezeCodec::new()), transpose(&[1, 0])],
                &[3, 2],
            ),
            shape(&[1, 4, 6]),
            shape(&[1, 2, 3]),
        ),
        (
            chain(
                vec![Arc::new(ReshapeCodec::new(
                    serde_json::from_str::<ReshapeShape>("[40]").unwrap(),
                ))],
                &[5],
            ),
            shape(&[2, 20]),
            shape(&[1, 5]),
        ),
    ];
    for (chain, decoded, expected) in cases {
        assert_eq!(chain.partial_decode_granularity(&decoded), expected);
    }
    let no_sharding = CodecChain::new(
        vec![transpose(&[1, 0])],
        Arc::new(BytesCodec::new(None)),
        vec![],
    );
    assert_eq!(
        no_sharding.partial_decode_granularity(&shape(&[4, 6])),
        shape(&[4, 6])
    );
    // An invalid dimensionality remains conservative because the old query is infallible.
    assert_eq!(
        chain(vec![transpose(&[1, 0])], &[2, 3]).partial_decode_granularity(&shape(&[4])),
        shape(&[4])
    );
}

// An external codec written against 0.23 implements neither the new helper nor any
// new required method. Its existing whole-chunk recommendation remains conservative.
#[derive(Debug)]
struct ConservativeCodec;
zarrs_plugin::impl_extension_aliases!(ConservativeCodec, v3: "zarrs.test.conservative");
impl CodecTraits for ConservativeCodec {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn configuration(&self, _: ZarrVersion, _: &CodecMetadataOptions) -> Option<Configuration> {
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
impl ArrayCodecTraits for ConservativeCodec {
    fn recommended_concurrency(
        &self,
        _: &[NonZeroU64],
        _: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new(1..1))
    }
}
impl ArrayToArrayCodecTraits for ConservativeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self
    }
    fn encoded_data_type(&self, data_type: &DataType) -> Result<DataType, CodecError> {
        Ok(data_type.clone())
    }
    fn encoded_shape(&self, shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        Ok(shape.to_vec())
    }
    fn decoded_shape(&self, shape: &[NonZeroU64]) -> Result<Option<ChunkShape>, CodecError> {
        Ok(Some(shape.to_vec()))
    }
    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _: &[NonZeroU64],
        _: &DataType,
        _: &FillValue,
        _: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }
    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _: &[NonZeroU64],
        _: &DataType,
        _: &FillValue,
        _: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }
}
#[test]
fn codec_chain_partial_decode_granularity_keeps_custom_defaults() {
    assert_eq!(
        chain(
            vec![transpose(&[1, 0]), Arc::new(ConservativeCodec)],
            &[3, 2]
        )
        .partial_decode_granularity(&shape(&[4, 6])),
        shape(&[4, 6])
    );
}
