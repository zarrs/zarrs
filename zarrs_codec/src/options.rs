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
/// - `multipart_write`: [`MultipartWrite::Auto`]
/// - `multipart_part_size`: 1 MiB
#[derive(Debug, Clone, Copy)]
pub struct CodecOptions {
    validate_checksums: bool,
    store_empty_chunks: bool,
    multipart_write: MultipartWrite,
    multipart_part_size: usize,
    multipart_patch: bool,
}

impl Default for CodecOptions {
    fn default() -> Self {
        Self {
            validate_checksums: true,
            store_empty_chunks: false,
            multipart_write: MultipartWrite::Auto,
            multipart_part_size: 1024 * 1024,
            multipart_patch: false,
        }
    }
}

/// Whether a codec that can write a chunk as it is encoded does so, with a multipart write.
///
/// For example, the `sharding_indexed` codec can write the encoded subchunks of a shard as they are encoded, rather
/// than assembling the whole shard first, so a write holds a window of encoded subchunks rather than the whole shard.
///
/// An array resolves [`Auto`](MultipartWrite::Auto) for a write: it becomes [`Always`](MultipartWrite::Always) if its
/// storage writes values in parts
/// ([`WritableStorageTraits::supports_multipart`](zarrs_storage::WritableStorageTraits::supports_multipart)), and
/// [`Never`](MultipartWrite::Never) otherwise, or if the array has storage transformers.
/// A codec only writes a chunk as it is encoded with [`Always`](MultipartWrite::Always).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum MultipartWrite {
    /// Write chunks as they are encoded if the storage writes values in parts.
    #[default]
    Auto,
    /// Write chunks as they are encoded, even if the storage collects the parts and sets the value whole (e.g. for
    /// testing).
    Always,
    /// Encode chunks whole before writing them.
    Never,
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

    /// Return the multipart write setting.
    #[must_use]
    pub fn multipart_write(&self) -> MultipartWrite {
        self.multipart_write
    }

    /// Set whether chunks are written as they are encoded, with multipart writes.
    pub fn set_multipart_write(&mut self, multipart_write: MultipartWrite) -> &mut Self {
        self.multipart_write = multipart_write;
        self
    }

    /// Set whether chunks are written as they are encoded, with multipart writes.
    #[must_use]
    pub fn with_multipart_write(mut self, multipart_write: MultipartWrite) -> Self {
        self.multipart_write = multipart_write;
        self
    }

    /// Return the part size of multipart writes.
    ///
    /// A codec that writes a chunk as it is encoded collects encoded bytes until they reach this size, then writes
    /// them as a part, so a write is not a request per small piece. A chunk whose decoded size is at most this size is
    /// encoded whole, since it would be written in one part.
    #[must_use]
    pub fn multipart_part_size(&self) -> usize {
        self.multipart_part_size
    }

    /// Set the part size of multipart writes (see [`multipart_part_size`](Self::multipart_part_size)).
    pub fn set_multipart_part_size(&mut self, multipart_part_size: usize) -> &mut Self {
        self.multipart_part_size = multipart_part_size;
        self
    }

    /// Set the part size of multipart writes (see [`multipart_part_size`](Self::multipart_part_size)).
    #[must_use]
    pub fn with_multipart_part_size(mut self, multipart_part_size: usize) -> Self {
        self.multipart_part_size = multipart_part_size;
        self
    }

    /// Returns true if multipart writes can overwrite bytes already written (e.g. a header written last).
    ///
    /// An array sets this for a write from its storage
    /// ([`WritableStorageTraits::supports_multipart_patch`](zarrs_storage::WritableStorageTraits::supports_multipart_patch)).
    /// For example, the `sharding_indexed` codec only writes a shard with its index at the start as it is encoded if
    /// so.
    #[must_use]
    pub fn multipart_patch(&self) -> bool {
        self.multipart_patch
    }

    /// Set whether multipart writes can overwrite bytes already written (see
    /// [`multipart_patch`](Self::multipart_patch)).
    pub fn set_multipart_patch(&mut self, multipart_patch: bool) -> &mut Self {
        self.multipart_patch = multipart_patch;
        self
    }

    /// Set whether multipart writes can overwrite bytes already written (see
    /// [`multipart_patch`](Self::multipart_patch)).
    #[must_use]
    pub fn with_multipart_patch(mut self, multipart_patch: bool) -> Self {
        self.multipart_patch = multipart_patch;
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
