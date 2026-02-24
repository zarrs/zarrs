/// Zarr V2 group metadata.
mod group;
use derive_more::{Display, From};
pub use group::GroupMetadataV2;

/// Zarr V2 array metadata.
mod array;

pub use array::{
    data_type_metadata_v2_to_endianness, ArrayMetadataV2, ArrayMetadataV2Order, DataTypeMetadataV2,
    DataTypeMetadataV2EndiannessError, DataTypeMetadataV2Structured, FillValueMetadataV2,
};

mod metadata;
pub use metadata::MetadataV2;

/// Zarr V2 node metadata ([`ArrayMetadataV2`] or [`GroupMetadataV2`]).
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Display, From)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum NodeMetadataV2 {
    /// Array metadata.
    Array(ArrayMetadataV2),
    /// Group metadata.
    Group(GroupMetadataV2),
}
