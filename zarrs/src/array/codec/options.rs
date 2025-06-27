//! Codec options for encoding and decoding.

use crate::config::global_config;

/// Codec options for encoding/decoding.
///
/// Default values for these options are set by the global [`Config`](crate::config::Config).
#[derive(Debug, Clone)]
pub struct CodecOptions {
    validate_checksums: bool,
    store_empty_chunks: bool,
    concurrent_target: usize,
    experimental_partial_encoding: bool,
}

impl Default for CodecOptions {
    fn default() -> Self {
        Self {
            validate_checksums: global_config().validate_checksums(),
            store_empty_chunks: global_config().store_empty_chunks(),
            concurrent_target: global_config().codec_concurrent_target(),
            experimental_partial_encoding: global_config().experimental_partial_encoding(),
        }
    }
}

impl CodecOptions {
    /// Create a new default codec options builder.
    #[must_use]
    pub fn builder() -> CodecOptionsBuilder {
        CodecOptionsBuilder::new()
    }

    /// Copy codec options into a new [`CodecOptionsBuilder`].
    #[must_use]
    pub fn into_builder(&self) -> CodecOptionsBuilder {
        CodecOptionsBuilder {
            validate_checksums: self.validate_checksums,
            store_empty_chunks: self.store_empty_chunks,
            concurrent_target: self.concurrent_target,
            experimental_partial_encoding: self.experimental_partial_encoding,
        }
    }

    /// Return the validate checksums setting.
    #[must_use]
    pub fn validate_checksums(&self) -> bool {
        self.validate_checksums
    }

    /// Set whether or not to validate checksums.
    pub fn set_validate_checksums(&mut self, validate_checksums: bool) -> &mut Self {
        self.validate_checksums = validate_checksums;
        self
    }

    /// Return the store empty chunks setting.
    #[must_use]
    pub fn store_empty_chunks(&self) -> bool {
        self.store_empty_chunks
    }

    /// Set whether or not to store empty chunks.
    pub fn set_store_empty_chunks(&mut self, store_empty_chunks: bool) -> &mut Self {
        self.store_empty_chunks = store_empty_chunks;
        self
    }

    /// Return the concurrent target.
    #[must_use]
    pub fn concurrent_target(&self) -> usize {
        self.concurrent_target
    }

    /// Set the concurrent target.
    pub fn set_concurrent_target(&mut self, concurrent_target: usize) -> &mut Self {
        self.concurrent_target = concurrent_target;
        self
    }

    /// Return the experimental partial encoding setting.
    #[must_use]
    pub fn experimental_partial_encoding(&self) -> bool {
        self.experimental_partial_encoding
    }

    /// Set whether or not to use experimental partial encoding.
    pub fn set_experimental_partial_encoding(
        &mut self,
        experimental_partial_encoding: bool,
    ) -> &mut Self {
        self.experimental_partial_encoding = experimental_partial_encoding;
        self
    }
}

/// Builder for [`CodecOptions`].
///
/// Default values for these options are set by the global [`Config`](crate::config::Config).
#[derive(Debug, Clone)]
pub struct CodecOptionsBuilder {
    validate_checksums: bool,
    store_empty_chunks: bool,
    concurrent_target: usize,
    experimental_partial_encoding: bool,
}

impl Default for CodecOptionsBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl CodecOptionsBuilder {
    /// Create a new encode options builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            validate_checksums: global_config().validate_checksums(),
            store_empty_chunks: global_config().store_empty_chunks(),
            concurrent_target: global_config().codec_concurrent_target(),
            experimental_partial_encoding: global_config().experimental_partial_encoding(),
        }
    }

    /// Build into encode options.
    #[must_use]
    pub fn build(&self) -> CodecOptions {
        CodecOptions {
            validate_checksums: self.validate_checksums,
            store_empty_chunks: self.store_empty_chunks,
            concurrent_target: self.concurrent_target,
            experimental_partial_encoding: self.experimental_partial_encoding,
        }
    }

    /// Set whether or not to validate checksums.
    #[must_use]
    pub fn validate_checksums(mut self, validate_checksums: bool) -> Self {
        self.validate_checksums = validate_checksums;
        self
    }

    /// Set whether or not to store empty chunks.
    #[must_use]
    pub fn store_empty_chunks(mut self, store_empty_chunks: bool) -> Self {
        self.store_empty_chunks = store_empty_chunks;
        self
    }

    /// Set the concurrent target for parallel operations.
    #[must_use]
    pub fn concurrent_target(mut self, concurrent_target: usize) -> Self {
        self.concurrent_target = concurrent_target;
        self
    }

    /// Set whether or not to use experimental partial encoding.
    #[must_use]
    pub fn experimental_partial_encoding(mut self, experimental_partial_encoding: bool) -> Self {
        self.experimental_partial_encoding = experimental_partial_encoding;
        self
    }
}

/// Options for codec metadata.
#[derive(Debug, Clone)]
pub struct CodecMetadataOptions {
    codec_store_metadata_if_encode_only: bool,
}

impl Default for CodecMetadataOptions {
    fn default() -> Self {
        Self {
            codec_store_metadata_if_encode_only: global_config()
                .codec_store_metadata_if_encode_only(),
        }
    }
}

impl CodecMetadataOptions {
    /// Return the [codec store metadata if encode only](crate::config::Config#codec-store-metadata-if-encode-only) setting.
    #[must_use]
    pub fn codec_store_metadata_if_encode_only(&self) -> bool {
        self.codec_store_metadata_if_encode_only
    }

    /// Set the [codec store metadata if encode only](crate::config::Config#codec-store-metadata-if-encode-only) setting.
    #[must_use]
    pub fn with_codec_store_metadata_if_encode_only(mut self, enabled: bool) -> Self {
        self.codec_store_metadata_if_encode_only = enabled;
        self
    }

    /// Set the [codec store metadata if encode only](crate::config::Config#codec-store-metadata-if-encode-only) setting.
    pub fn set_codec_store_metadata_if_encode_only(&mut self, enabled: bool) -> &mut Self {
        self.codec_store_metadata_if_encode_only = enabled;
        self
    }
}
