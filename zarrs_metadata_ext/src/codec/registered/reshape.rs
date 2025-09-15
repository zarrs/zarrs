use std::num::NonZeroU64;

use derive_more::{Display, From};
use monostate::MustBe;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use zarrs_metadata::ConfigurationSerialize;

/// A wrapper to handle various versions of Reshape codec configuration parameters.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[non_exhaustive]
#[serde(untagged)]
pub enum ReshapeCodecConfiguration {
    /// Version 1.0.
    V1(ReshapeCodecConfigurationV1),
}

impl ConfigurationSerialize for ReshapeCodecConfiguration {}

/// `reshape` codec configuration parameters (version 1.0).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct ReshapeCodecConfigurationV1 {
    /// The reshape order defining how to permute the array.
    pub shape: ReshapeShape,
}

impl ReshapeCodecConfigurationV1 {
    /// Create a new Reshape codec configuration given a [`ReshapeShape`].
    #[must_use]
    pub const fn new(shape: ReshapeShape) -> Self {
        Self { shape }
    }
}

/// An element of `ReshapeShape`.
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, From)]
#[serde(untagged)]
pub enum ReshapeDim {
    /// A positive integer `size`, specifying that `B_shape[i] := size`.
    Size(NonZeroU64),
    /// An array of integers `input_dims`, specifying that: `B_shape[i] := prod(A_shape[input_dims]]`.
    ///
    /// Specifying the corresponding `input_dims` rather than an explicit `size` is
    /// particularly useful when using variable-size chunking.
    InputDims(Vec<u64>),
    /// A special value indicating the shape is determined automatically.
    ///
    /// The shape is chosen to satisfy the invariant that `prod(B_shape) == prod(A_shape)`.
    Auto(MustBe!(-1i64)),
}

impl ReshapeDim {
    /// Create a new `ReshapeDim` with the corresponding `auto` value.
    pub fn auto() -> Self {
        Self::Auto(MustBe!(-1i64))
    }
}

impl<const N: usize> From<[u64; N]> for ReshapeDim {
    fn from(value: [u64; N]) -> Self {
        ReshapeDim::InputDims(value.to_vec())
    }
}

/// An array specifying the size `B_shape[i]` of each dimension `i` of the output array `B` as a function of the shape `A_shape` of the input array `A`.
#[derive(Serialize, Clone, Eq, PartialEq, Debug)]
pub struct ReshapeShape(pub Vec<ReshapeDim>);

/// A `ReshapeShape` error.
///
/// For example
/// - more than one element is -1, or
/// - input dims are not increasing.
#[derive(Clone, Debug, Error, From)]
#[error("reshape shape {0:?} is invalid")]
pub struct ReshapeShapeError(Vec<ReshapeDim>);

impl ReshapeShape {
    /// Create a new [`ReshapeShape`].
    ///
    /// # Errors
    /// Returns [`ReshapeShapeError`] if the elements are not valid according to the constraints of the `reshape` codec.
    pub fn new(shape: impl IntoIterator<Item = ReshapeDim>) -> Result<Self, ReshapeShapeError> {
        let shape: Vec<ReshapeDim> = shape.into_iter().collect();
        if validate_shape(&shape) {
            Ok(Self(shape))
        } else {
            Err(ReshapeShapeError(shape))
        }
    }
}

impl<'de> serde::Deserialize<'de> for ReshapeShape {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let shape = Vec::<ReshapeDim>::deserialize(d)?;
        if validate_shape(&shape) {
            Ok(Self(shape))
        } else {
            Err(serde::de::Error::custom(
                "reshape shape {shape:?} is invalid",
            ))
        }
    }
}

fn validate_shape(shape: &[ReshapeDim]) -> bool {
    let mut dim_idx = 0;
    let mut has_auto = false;
    for dim in shape {
        match dim {
            ReshapeDim::Size(_size) => {
                // always valid
                dim_idx += 1;
            }
            ReshapeDim::InputDims(dims) => {
                for dim in dims {
                    // dims must be increasing
                    if *dim < dim_idx {
                        return false;
                    }
                    dim_idx = dim_idx.max(*dim) + 1;
                }
            }
            ReshapeDim::Auto(_) => {
                if has_auto {
                    // At most one auto
                    return false;
                }
                has_auto = true;
                dim_idx += 1;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_reshape_array_valid1() {
        let json = r#"{
            "shape": [[0, 1], 10, [3, 4]]
        }"#;
        serde_json::from_str::<ReshapeCodecConfiguration>(json).unwrap();
    }

    #[test]
    fn codec_reshape_array_valid2() {
        let json = r#"{
            "shape": [[0, 1], [2], 3]
        }"#;
        serde_json::from_str::<ReshapeCodecConfiguration>(json).unwrap();
    }

    #[test]
    fn codec_reshape_array_valid3() {
        let json = r#"{
            "shape": [[0, 1], -1, [3], 4]
        }"#;
        serde_json::from_str::<ReshapeCodecConfiguration>(json).unwrap();
    }

    #[test]
    fn codec_reshape_array_valid4() {
        let json = r#"{
            "shape": [-1]
        }"#;
        serde_json::from_str::<ReshapeCodecConfiguration>(json).unwrap();
    }

    #[test]
    fn codec_reshape_array_valid5() {
        let json = r#"{
            "shape": [[0], -1, [2, 3]]
        }"#;
        serde_json::from_str::<ReshapeCodecConfiguration>(json).unwrap();
    }

    #[test]
    fn codec_reshape_array_valid6() {
        let json = r#"{
            "shape": [[0], -1, [3]]
        }"#;
        serde_json::from_str::<ReshapeCodecConfiguration>(json).unwrap();
    }

    #[test]
    fn codec_reshape_invalid1() {
        let json = r#"{
            "shape": [[1], [0]]
        }"#;
        assert!(serde_json::from_str::<ReshapeCodecConfiguration>(json).is_err());
    }

    #[test]
    fn codec_reshape_invalid2() {
        let json = r#"{
            "shape": [[1, 0], 10, [3, 4]]
        }"#;
        assert!(serde_json::from_str::<ReshapeCodecConfiguration>(json).is_err());
    }

    #[test]
    fn codec_reshape_invalid3() {
        let json = r#"{
            "shape": [[3, 4], 10, [0, 1]]
        }"#;
        assert!(serde_json::from_str::<ReshapeCodecConfiguration>(json).is_err());
    }

    #[test]
    fn codec_reshape_array_invalid4() {
        let json = r#"{
            "shape": [[0, 1], -1, [2], 3]
        }"#;
        assert!(serde_json::from_str::<ReshapeCodecConfiguration>(json).is_err());
    }

    #[test]
    fn codec_reshape_array_invalid5() {
        let json = r#"{
            "shape": [-1, -1]
        }"#;
        assert!(serde_json::from_str::<ReshapeCodecConfiguration>(json).is_err());
    }
}
