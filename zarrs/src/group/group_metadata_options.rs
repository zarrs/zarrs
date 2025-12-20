use crate::config::{MetadataConvertVersion, global_config};

/// Options for writing group metadata.
#[derive(Debug, Clone)]
pub struct GroupMetadataOptions {
    convert_version: MetadataConvertVersion,
}

impl Default for GroupMetadataOptions {
    fn default() -> Self {
        Self {
            convert_version: global_config().metadata_convert_version(),
        }
    }
}

impl GroupMetadataOptions {
    /// Get the [metadata convert version](crate::config::Config#metadata-convert-version) configuration.
    #[must_use]
    pub fn metadata_convert_version(&self) -> MetadataConvertVersion {
        self.convert_version
    }

    /// Set the [metadata convert version](crate::config::Config#metadata-convert-version) configuration.
    #[must_use]
    pub fn with_metadata_convert_version(
        mut self,
        convert_version: MetadataConvertVersion,
    ) -> Self {
        self.convert_version = convert_version;
        self
    }

    /// Set the [metadata convert version](crate::config::Config#metadata-convert-version) configuration.
    pub fn set_metadata_convert_version(
        &mut self,
        convert_version: MetadataConvertVersion,
    ) -> &mut Self {
        self.convert_version = convert_version;
        self
    }
}
