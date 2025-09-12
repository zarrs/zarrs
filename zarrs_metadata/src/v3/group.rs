use derive_more::Display;
use serde::{Deserialize, Serialize};

use super::AdditionalFieldsV3;

/// Zarr V3 group metadata.
///
/// See <https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#group-metadata>.
///
/// An example `JSON` document for an explicit Zarr V3 group:
/// ```json
/// {
///     "zarr_format": 3,
///     "node_type": "group",
///     "attributes": {
///         "spam": "ham",
///         "eggs": 42,
///     }
/// }
#[non_exhaustive]
#[derive(Serialize, Deserialize, Clone, Debug, Display)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct GroupMetadataV3 {
    /// An integer defining the version of the storage specification to which the group adheres. Must be `3`.
    pub zarr_format: monostate::MustBe!(3u64),
    /// A string defining the type of hierarchy node element, must be `group` here.
    pub node_type: monostate::MustBe!("group"),
    /// Optional user metadata.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub attributes: serde_json::Map<String, serde_json::Value>,
    // /// Extension definitions (Zarr 3.2, [ZEP0010](https://github.com/zarr-developers/zeps/pull/67)).
    // #[serde(default, skip_serializing_if = "Vec::is_empty")]
    // pub extensions: Vec<MetadataV3>,
    /// Additional fields.
    #[serde(flatten)]
    pub additional_fields: AdditionalFieldsV3,
}

impl std::cmp::PartialEq for GroupMetadataV3 {
    fn eq(&self, other: &Self) -> bool {
        self.attributes == other.attributes
            // && self.consolidated_metadata == other.consolidated_metadata
            && self.additional_fields == other.additional_fields
    }
}

impl Eq for GroupMetadataV3 {}

impl Default for GroupMetadataV3 {
    fn default() -> Self {
        Self::new()
    }
}

impl GroupMetadataV3 {
    /// Create Zarr V3 group metadata.
    #[must_use]
    pub fn new() -> Self {
        Self {
            zarr_format: monostate::MustBe!(3u64),
            node_type: monostate::MustBe!("group"),
            attributes: serde_json::Map::new(),
            // extensions: Vec::default(),
            additional_fields: AdditionalFieldsV3::default(),
        }
    }

    /// Serialize the metadata as a pretty-printed String of JSON.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn to_string_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("group metadata is valid JSON")
    }

    /// Set the user attributes.
    #[must_use]
    pub fn with_attributes(
        mut self,
        attributes: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        self.attributes = attributes;
        self
    }

    /// Set the additional fields.
    #[must_use]
    pub fn with_additional_fields(mut self, additional_fields: AdditionalFieldsV3) -> Self {
        self.additional_fields = additional_fields;
        self
    }

    // /// Set the extension definitions.
    // #[must_use]
    // pub fn with_extensions(mut self, extensions: Vec<MetadataV3>) -> Self {
    //     self.extensions = extensions;
    //     self
    // }
}
