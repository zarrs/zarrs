//! The `reshape` array to array codec (Experimental).
//!
//! Performs a reshaping operation.
//!
//! <div class="warning">
//! This codec is experimental and may be incompatible with other Zarr V3 implementations.
//! </div>
//!
//! ### Compatible Implementations
//! None
//!
//! ### Specification
//! - <https://github.com/zarr-developers/zarr-extensions/blob/7295bf1ec15c978f1a63b90d55891712b950c797/codecs/reshape/README.md>
//!
//! ### Codec `name` Aliases (Zarr V3)
//! - `reshape`
//!
//! ### Codec `id` Aliases (Zarr V2)
//! - `reshape`
//!
//! ### Codec `configuration` Example - [`ReshapeCodecConfiguration`]:
//! ```rust
//! # let JSON = r#"
//! {
//!     "shape": [[0, 1], -1, [3], 10]
//! }
//! # "#;
//! # use zarrs::metadata_ext::codec::reshape::ReshapeCodecConfiguration;
//! # let configuration: ReshapeCodecConfiguration = serde_json::from_str(JSON).unwrap();
//! ```

mod reshape_codec;

use std::{num::NonZeroU64, sync::Arc};

use num::Integer;
pub use reshape_codec::ReshapeCodec;

// use itertools::Itertools;
use crate::metadata::ChunkShape;
pub use crate::metadata_ext::codec::reshape::{
    ReshapeCodecConfiguration, ReshapeCodecConfigurationV1, ReshapeDim, ReshapeShape,
};
use crate::registry::codec::RESHAPE;
use crate::{
    array::codec::{Codec, CodecError, CodecPlugin},
    metadata::v3::MetadataV3,
    plugin::{PluginCreateError, PluginMetadataInvalidError},
};

fn get_encoded_shape(
    reshape_shape: &ReshapeShape,
    decoded_shape: &[NonZeroU64],
) -> Result<ChunkShape, CodecError> {
    let mut encoded_shape = Vec::with_capacity(reshape_shape.0.len());
    let mut fill_index = None;
    for output_dim in &reshape_shape.0 {
        match output_dim {
            ReshapeDim::Size(size) => encoded_shape.push(*size),
            ReshapeDim::InputDims(input_dims) => {
                let mut product = NonZeroU64::new(1).unwrap();
                for input_dim in input_dims {
                    let input_shape = *decoded_shape
                        .get(usize::try_from(*input_dim).unwrap())
                        .ok_or_else(|| {
                            CodecError::Other(
                                format!("reshape codec shape references a dimension ({input_dim}) larger than the chunk dimensionality ({})", decoded_shape.len()),
                            )
                        })?;
                    product = product.checked_mul(input_shape).unwrap();
                }
                encoded_shape.push(product);
            }
            ReshapeDim::Auto(_) => {
                fill_index = Some(encoded_shape.len());
                encoded_shape.push(NonZeroU64::new(1).unwrap());
            }
        }
    }

    let num_elements_input = decoded_shape.iter().map(|u| u.get()).product::<u64>();
    let num_elements_output = encoded_shape.iter().map(|u| u.get()).product::<u64>();
    if let Some(fill_index) = fill_index {
        let (quot, rem) = num_elements_input.div_rem(&num_elements_output);
        if rem == 0 {
            encoded_shape[fill_index] = NonZeroU64::new(quot).unwrap();
        } else {
            return Err(CodecError::Other(format!(
                "reshape codec no substitution for dim {fill_index} can satisfy decoded_shape {decoded_shape:?} == encoded_shape {encoded_shape:?}."
            )));
        }
    } else if num_elements_input != num_elements_output {
        return Err(CodecError::Other(format!(
            "reshape codec encoded/decoded number of elements differ: decoded_shape {decoded_shape:?} ({num_elements_input}) encoded_shape {encoded_shape:?} ({num_elements_output})."
        )));
    }

    Ok(encoded_shape.into())
}

// Register the codec.
inventory::submit! {
    CodecPlugin::new(RESHAPE, is_identifier_reshape, create_codec_reshape)
}

fn is_identifier_reshape(identifier: &str) -> bool {
    identifier == RESHAPE
}

pub(crate) fn create_codec_reshape(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
    let configuration: ReshapeCodecConfiguration = metadata
        .to_configuration()
        .map_err(|_| PluginMetadataInvalidError::new(RESHAPE, "codec", metadata.to_string()))?;
    let codec = Arc::new(ReshapeCodec::new_with_configuration(&configuration)?);
    Ok(Codec::ArrayToArray(codec))
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;
    use crate::array::{
        ArrayBytes, ChunkRepresentation, DataType, FillValue,
        codec::{ArrayToArrayCodecTraits, CodecOptions},
    };

    fn codec_reshape_round_trip_impl(
        json: &str,
        data_type: DataType,
        fill_value: FillValue,
        output_shape: Vec<NonZeroU64>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let chunk_representation = ChunkRepresentation::new(
            vec![
                NonZeroU64::new(5).unwrap(),
                NonZeroU64::new(4).unwrap(),
                NonZeroU64::new(4).unwrap(),
                NonZeroU64::new(3).unwrap(),
            ],
            data_type,
            fill_value,
        )?;
        let size = chunk_representation.num_elements_usize()
            * chunk_representation.data_type().fixed_size().unwrap();
        let bytes: Vec<u8> = (0..size).map(|s| s as u8).collect();
        let bytes: ArrayBytes = bytes.into();

        let configuration: ReshapeCodecConfiguration = serde_json::from_str(json)?;
        let codec = ReshapeCodec::new_with_configuration(&configuration)?;
        assert_eq!(
            codec.encoded_shape(chunk_representation.shape())?,
            output_shape.into()
        );

        let encoded = codec.encode(
            bytes.clone(),
            &chunk_representation,
            &CodecOptions::default(),
        )?;
        let decoded = codec.decode(encoded, &chunk_representation, &CodecOptions::default())?;
        assert_eq!(bytes, decoded);
        Ok(())
    }

    #[test]
    fn codec_reshape_round_trip_array1() {
        const JSON: &str = r#"{
            "shape": [[0, 1], [2], 3]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(20).unwrap(),
            NonZeroU64::new(4).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                DataType::UInt32,
                FillValue::from(0u32),
                output_shape
            )
            .is_ok()
        );
    }

    #[test]
    fn codec_reshape_round_trip_array2() {
        const JSON: &str = r#"{
            "shape": [[0, 1], [2], -1]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(20).unwrap(),
            NonZeroU64::new(4).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                DataType::UInt32,
                FillValue::from(0u32),
                output_shape
            )
            .is_ok()
        );
    }

    #[test]
    fn codec_reshape_round_trip_array3() {
        const JSON: &str = r#"{
            "shape": [[0, 1, 2], 3]
        }"#;
        let output_shape = vec![NonZeroU64::new(80).unwrap(), NonZeroU64::new(3).unwrap()];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                DataType::UInt32,
                FillValue::from(0u32),
                output_shape
            )
            .is_ok()
        );
    }

    #[test]
    fn codec_reshape_round_trip_array4() {
        const JSON: &str = r#"{
            "shape": [[0], -1, [2, 3]]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(5).unwrap(),
            NonZeroU64::new(4).unwrap(),
            NonZeroU64::new(12).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                DataType::UInt32,
                FillValue::from(0u32),
                output_shape
            )
            .is_ok()
        );
    }

    #[test]
    fn codec_reshape_round_trip_array5() {
        const JSON: &str = r#"{
            "shape": [[0], -1, [3]]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(5).unwrap(),
            NonZeroU64::new(16).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                DataType::UInt32,
                FillValue::from(0u32),
                output_shape
            )
            .is_ok()
        );
    }

    #[test]
    fn codec_reshape_round_trip_array6() {
        const JSON: &str = r#"{
            "shape": [-1, 2, 2, [3]]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(20).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                DataType::UInt32,
                FillValue::from(0u32),
                output_shape
            )
            .is_ok()
        );
    }

    #[test]
    fn codec_reshape_invalid1() {
        const JSON: &str = r#"{
            "shape": [-1, 2, 2, [4]]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(20).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                DataType::UInt32,
                FillValue::from(0u32),
                output_shape
            )
            .is_err()
        );
    }

    #[test]
    fn codec_reshape_invalid2() {
        const JSON: &str = r#"{
            "shape": [2, 2, 2]
        }"#;
        let output_shape = vec![
            NonZeroU64::new(20).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(2).unwrap(),
            NonZeroU64::new(3).unwrap(),
        ];
        assert!(
            codec_reshape_round_trip_impl(
                JSON,
                DataType::UInt32,
                FillValue::from(0u32),
                output_shape
            )
            .is_err()
        );
    }
}
