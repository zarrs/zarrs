//! Runtime options for the Zstd codec.

/// Runtime options for the [`ZstdCodec`](super::ZstdCodec).
///
/// Set these directly with [`ZstdCodec::with_options`](super::ZstdCodec::with_options),
/// or attach them to [`CodecSpecificOptions`](zarrs_codec::CodecSpecificOptions)
/// when creating or opening an array. They are not stored in codec metadata.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct ZstdCodecOptions {
    compression_context_reuse_threshold: usize,
}

impl Default for ZstdCodecOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl ZstdCodecOptions {
    pub(super) const DEFAULT: Self = Self {
        compression_context_reuse_threshold: 1024 * 1024,
    };

    /// Return the maximum uncompressed input size, in bytes, eligible for compression context reuse.
    ///
    /// The default is 1 MiB. Inputs up to and including the threshold use a
    /// thread-local context; larger inputs use a temporary context.
    /// Set this to `0` to disable reuse, including for empty inputs, or to
    /// [`usize::MAX`] to allow reuse for any input size.
    ///
    /// Contexts are shared by Zstd codecs on the same thread. This threshold
    /// controls eligibility for reuse, not the context's allocated memory size,
    /// and does not release memory retained by earlier encodes.
    #[must_use]
    pub const fn compression_context_reuse_threshold(&self) -> usize {
        self.compression_context_reuse_threshold
    }

    /// Set the [`compression_context_reuse_threshold`](Self::compression_context_reuse_threshold).
    pub fn set_compression_context_reuse_threshold(&mut self, threshold: usize) -> &mut Self {
        self.compression_context_reuse_threshold = threshold;
        self
    }

    /// Set the [`compression_context_reuse_threshold`](Self::compression_context_reuse_threshold).
    #[must_use]
    pub fn with_compression_context_reuse_threshold(mut self, threshold: usize) -> Self {
        self.set_compression_context_reuse_threshold(threshold);
        self
    }
}
