//! Codec options for encoding and decoding.

/// Codec options for encoding/decoding.
///
/// The default values are:
/// - `validate_checksums`: `true`
/// - `store_empty_chunks`: `false`
/// - `concurrent_target`: number of threads available to Rayon
/// - `chunk_concurrent_minimum`: `4`
/// - `experimental_partial_encoding`: `false`
#[derive(Debug, Clone, Copy)]
pub struct CodecOptions {
    validate_checksums: bool,
    store_empty_chunks: bool,
    concurrent_target: usize,
    chunk_concurrent_minimum: usize,
    experimental_partial_encoding: bool,
}

impl Default for CodecOptions {
    fn default() -> Self {
        Self {
            validate_checksums: true,
            store_empty_chunks: false,
            concurrent_target: rayon::current_num_threads(),
            chunk_concurrent_minimum: 4,
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

    /// Set the concurrent target.
    #[must_use]
    pub fn with_concurrent_target(mut self, concurrent_target: usize) -> Self {
        self.concurrent_target = concurrent_target;
        self
    }

    /// Return the chunk concurrent minimum.
    ///
    /// Array operations involving multiple chunks can tune the chunk and codec concurrency to improve performance/reduce memory usage.
    /// This option sets the preferred minimum chunk concurrency.
    /// The concurrency of internal codecs is adjusted to accomodate for the chunk concurrency in accordance with the concurrent target.
    #[must_use]
    pub fn chunk_concurrent_minimum(&self) -> usize {
        self.chunk_concurrent_minimum
    }

    /// Set the chunk concurrent minimum.
    pub fn set_chunk_concurrent_minimum(&mut self, chunk_concurrent_minimum: usize) -> &mut Self {
        self.chunk_concurrent_minimum = chunk_concurrent_minimum;
        self
    }

    /// Set the chunk concurrent minimum.
    #[must_use]
    pub fn with_chunk_concurrent_minimum(mut self, chunk_concurrent_minimum: usize) -> Self {
        self.chunk_concurrent_minimum = chunk_concurrent_minimum;
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
