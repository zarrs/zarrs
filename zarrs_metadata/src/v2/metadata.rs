use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::Configuration;

/// Zarr V2 generic metadata with an `id` and optional flattened `configuration`.
///
/// For example:
/// ```json
/// {
///     "id": "blosc",
///     "cname": "lz4",
///     "clevel": 5,
///     "shuffle": 1
/// }
/// ```
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug)]
pub struct MetadataV2 {
    id: String,
    #[serde(flatten)]
    configuration: Configuration,
}

impl MetadataV2 {
    /// Return the value of the `id` field.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Mutate the value of the `id` field.
    pub fn set_id(&mut self, id: String) -> &mut Self {
        self.id = id;
        self
    }

    /// Return the configuration, which includes all fields excluding the `id`.
    #[must_use]
    pub fn configuration(&self) -> &Configuration {
        &self.configuration
    }

    /// Try and convert [`Configuration`] to a specific serializable configuration.
    ///
    /// # Errors
    /// Returns a [`serde_json`] error if the metadata cannot be converted.
    pub fn to_typed_configuration<TConfiguration: DeserializeOwned>(
        &self,
    ) -> Result<TConfiguration, std::sync::Arc<serde_json::Error>> {
        self.configuration.to_typed()
    }
}
