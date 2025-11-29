use std::sync::Arc;

use super::{Group, GroupCreateError};
use crate::metadata::v3::{AdditionalFieldsV3, GroupMetadataV3};

/// A [`Group`] builder.
pub struct GroupBuilder {
    metadata: GroupMetadataV3,
}

impl Default for GroupBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GroupBuilder {
    /// Create a new group builder for a group at `path`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: GroupMetadataV3::default(),
        }
    }

    /// Set the attributes.
    pub fn attributes(
        &mut self,
        attributes: serde_json::Map<String, serde_json::Value>,
    ) -> &mut Self {
        self.metadata.attributes = attributes;
        self
    }

    /// Set the additional fields.
    ///
    /// Set additional fields not defined in the Zarr specification.
    /// Use this cautiously. In general, store user defined attributes using [`GroupBuilder::attributes`].
    ///
    /// `zarrs` and other implementations are expected to error when opening a group with unsupported additional fields, unless they are a JSON object containing `"must_understand": false`.
    pub fn additional_fields(&mut self, additional_fields: AdditionalFieldsV3) -> &mut Self {
        self.metadata.additional_fields = additional_fields;
        self
    }

    /// Build into a [`Group`].
    ///
    /// # Errors
    ///
    /// Returns [`GroupCreateError`] if the group could not be created.
    pub fn build<TStorage: ?Sized>(
        &self,
        storage: Arc<TStorage>,
        path: &str,
    ) -> Result<Group<TStorage>, GroupCreateError> {
        Group::new_with_metadata(storage, path, self.metadata.clone().into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::GroupMetadata;
    use crate::storage::store::MemoryStore;

    #[test]
    fn group_builder() {
        let mut builder = GroupBuilder::default();

        let mut attributes = serde_json::Map::new();
        attributes.insert("key".to_string(), "value".into());
        builder.attributes(attributes.clone());

        let mut additional_fields = AdditionalFieldsV3::new();
        let additional_field = serde_json::Map::new();
        additional_fields.insert("key".to_string(), additional_field.into());
        builder.additional_fields(additional_fields.clone());

        let storage = Arc::new(MemoryStore::new());
        println!("{:?}", builder.build(storage.clone(), "/"));
        let mut group = builder.build(storage, "/").unwrap();

        assert_eq!(group.attributes(), &attributes);
        assert_eq!(group.attributes_mut(), &attributes);
        if let GroupMetadata::V3(metadata) = group.metadata() {
            assert_eq!(metadata.additional_fields, additional_fields);
        }
    }
}
