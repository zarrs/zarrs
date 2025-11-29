//! Consolidated metadata.

use std::collections::HashMap;

use derive_more::Display;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use zarrs_metadata::NodeMetadata;

/// Consolidated metadata of a Zarr hierarchy.
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct ConsolidatedMetadata {
    /// A mapping from node path to Group or Array [`NodeMetadata`] object.
    pub metadata: ConsolidatedMetadataMetadata,
    /// The kind of the consolidated metadata. Must be `'inline'`. Reserved for future use.
    pub kind: ConsolidatedMetadataKind,
}

impl From<ConsolidatedMetadata> for Value {
    fn from(value: ConsolidatedMetadata) -> Self {
        serde_json::to_value(value).expect("consolidated metadata is serializable to value")
    }
}

/// The `metadata` field of `consolidated_metadata`.
pub type ConsolidatedMetadataMetadata = HashMap<String, NodeMetadata>;

impl Default for ConsolidatedMetadata {
    fn default() -> Self {
        Self {
            metadata: HashMap::default(),
            kind: ConsolidatedMetadataKind::Inline,
        }
    }
}

/// The "kind" of consolidated metadata.
#[non_exhaustive]
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
pub enum ConsolidatedMetadataKind {
    /// Indicates that consolidated metadata is stored inline in the root `zarr.json` object.
    #[serde(rename = "inline")]
    Inline,
}

#[cfg(test)]
mod tests {
    use zarrs_metadata::{v3::GroupMetadataV3, NodeMetadata};

    use super::*;

    #[test]
    fn group_metadata_consolidated() {
        let group_metadata = serde_json::from_str::<GroupMetadataV3>(
            r#"{
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {
                "spam": "ham",
                "eggs": 42
            },
            "consolidated_metadata": {
                "metadata": {
                    "/subgroup": {
                        "zarr_format": 3,
                        "node_type": "group",
                        "attributes": {
                            "consolidated": "attributes"
                        }
                    }
                },
                "kind": "inline",
                "must_understand": false
            }
        }"#,
        )
        .unwrap();

        let consolidated_metadata = group_metadata
            .additional_fields
            .get("consolidated_metadata")
            .unwrap();
        assert!(!consolidated_metadata.must_understand());
        let consolidated_metadata: ConsolidatedMetadata =
            serde_json::from_value(consolidated_metadata.as_value().clone()).unwrap();

        assert_eq!(
            consolidated_metadata.metadata.get("/subgroup").unwrap(),
            &serde_json::from_str::<NodeMetadata>(
                r#"{
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {
                        "consolidated": "attributes"
                    }
                }"#
            )
            .unwrap()
        );
    }
}
