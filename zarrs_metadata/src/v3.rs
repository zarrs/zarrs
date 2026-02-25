use derive_more::{Display, From};

/// Zarr V3 group metadata.
mod group;
pub use group::GroupMetadataV3;

/// Zarr V3 array metadata.
mod array;
pub use array::{
    ArrayMetadataV3, FillValueMetadataV3, ZARR_NAN_BF16, ZARR_NAN_F16, ZARR_NAN_F32, ZARR_NAN_F64,
};

mod metadata;
pub use metadata::{AdditionalFieldV3, AdditionalFieldsV3, MetadataV3};

/// Zarr V3 node metadata ([`ArrayMetadataV3`] or [`GroupMetadataV3`]).
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Display, From)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum NodeMetadataV3 {
    /// Array metadata.
    Array(ArrayMetadataV3),
    /// Group metadata.
    Group(GroupMetadataV3),
}
