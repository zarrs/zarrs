//! Codec options for encoding and decoding.

/// Per-operation codec options for encoding and decoding.
///
/// These are passed at each encode/decode call and control runtime behaviour such as checksum validation.
/// They are distinct from [`CodecSpecificOptions`](super::CodecSpecificOptions), which carry codec-specific configuration,
/// and [`Resources`](super::Resources), which set the resources (e.g. concurrency) an operation may use.
///
/// The default values are:
/// - `validate_checksums`: `true`
/// - `store_empty_chunks`: `false`
/// - `experimental_partial_encoding`: `false`
#[derive(Debug, Clone, Copy)]
pub struct CodecOptions {
    validate_checksums: bool,
    store_empty_chunks: bool,
    experimental_partial_encoding: bool,
}

impl Default for CodecOptions {
    fn default() -> Self {
        Self {
            validate_checksums: true,
            store_empty_chunks: false,
            experimental_partial_encoding: false,
        }
    }
}

impl CodecOptions {
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

    /// Set whether or not to validate checksums.
    #[must_use]
    pub fn with_validate_checksums(mut self, validate_checksums: bool) -> Self {
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

    /// Set whether or not to store empty chunks.
    #[must_use]
    pub fn with_store_empty_chunks(mut self, store_empty_chunks: bool) -> Self {
        self.store_empty_chunks = store_empty_chunks;
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

    /// Set whether or not to use experimental partial encoding.
    #[must_use]
    pub fn with_experimental_partial_encoding(
        mut self,
        experimental_partial_encoding: bool,
    ) -> Self {
        self.experimental_partial_encoding = experimental_partial_encoding;
        self
    }
}

/// Options for codec metadata.
#[derive(Debug, Clone, Copy)]
pub struct CodecMetadataOptions {
    codec_store_metadata_if_encode_only: bool,
}

impl Default for CodecMetadataOptions {
    fn default() -> Self {
        Self {
            codec_store_metadata_if_encode_only: true,
        }
    }
}

impl CodecMetadataOptions {
    /// Return the store metadata if encode only setting.
    #[must_use]
    pub fn codec_store_metadata_if_encode_only(&self) -> bool {
        self.codec_store_metadata_if_encode_only
    }

    /// Set the store metadata if encode only setting.
    #[must_use]
    pub fn with_codec_store_metadata_if_encode_only(mut self, enabled: bool) -> Self {
        self.codec_store_metadata_if_encode_only = enabled;
        self
    }

    /// Set the codec store metadata if encode only setting.
    pub fn set_codec_store_metadata_if_encode_only(&mut self, enabled: bool) -> &mut Self {
        self.codec_store_metadata_if_encode_only = enabled;
        self
    }
}
