//! Runtime options for the sharding codec.

/// Write order for subchunks within a shard
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum SubchunkWriteOrder {
    /// An alias for `Unordered`. Soft deprecated.
    ///
    /// `Random` is a misnomer and this variant will be removed in a future release.
    // TODO: Remove in 0.24
    Random,
    /// C order i.e., row-major
    C,
    /// No order guarantee.
    ///
    /// Because subchunk writing is parallelized, it will often appear that subchunks are written at random with this setting although this is dependent on the parallelizable workload.
    /// For example in the degenerate case of one thread, you may observe (mostly) ordered chunks.
    Unordered,
    // TODO: Morton order - depend on https://docs.rs/morton-encoding/latest/morton_encoding/?
}

/// Runtime options for the [`ShardingCodec`](super::ShardingCodec).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ShardingCodecOptions {
    subchunk_write_order: SubchunkWriteOrder,
    allow_nondivisible_subchunks: Option<bool>,
}

impl Default for ShardingCodecOptions {
    fn default() -> Self {
        Self {
            subchunk_write_order: SubchunkWriteOrder::Unordered,
            allow_nondivisible_subchunks: None,
        }
    }
}

impl ShardingCodecOptions {
    /// Set the subchunk ordering.
    #[must_use]
    pub fn with_subchunk_write_order(mut self, subchunk_write_order: SubchunkWriteOrder) -> Self {
        self.subchunk_write_order = subchunk_write_order;
        self
    }

    /// Set the subchunk ordering.
    pub fn set_subchunk_write_order(
        &mut self,
        subchunk_write_order: SubchunkWriteOrder,
    ) -> &mut Self {
        self.subchunk_write_order = subchunk_write_order;
        self
    }

    /// Return the subchunk ordering.
    #[must_use]
    pub fn subchunk_write_order(&self) -> SubchunkWriteOrder {
        self.subchunk_write_order
    }

    /// Permit subchunk shapes that do not evenly divide the shard shape.
    ///
    /// See [non-divisible subchunk shapes](crate::array::codec::array_to_bytes::sharding#non-divisible-subchunk-shapes).
    /// If unset, a codec keeps its existing setting when these options are applied.
    /// Codecs created from metadata permit non-divisible subchunks, otherwise they are rejected by default.
    #[must_use]
    pub fn with_allow_nondivisible_subchunks(mut self, allow_nondivisible_subchunks: bool) -> Self {
        self.allow_nondivisible_subchunks = Some(allow_nondivisible_subchunks);
        self
    }

    /// Permit subchunk shapes that do not evenly divide the shard shape.
    ///
    /// See [`with_allow_nondivisible_subchunks`](Self::with_allow_nondivisible_subchunks).
    pub fn set_allow_nondivisible_subchunks(
        &mut self,
        allow_nondivisible_subchunks: bool,
    ) -> &mut Self {
        self.allow_nondivisible_subchunks = Some(allow_nondivisible_subchunks);
        self
    }

    /// Return whether subchunk shapes that do not evenly divide the shard shape are permitted, if set.
    #[must_use]
    pub fn allow_nondivisible_subchunks(&self) -> Option<bool> {
        self.allow_nondivisible_subchunks
    }

    /// Return `options`, inheriting unset options from `self`.
    pub(crate) fn merged(&self, options: &Self) -> Self {
        Self {
            allow_nondivisible_subchunks: options
                .allow_nondivisible_subchunks
                .or(self.allow_nondivisible_subchunks),
            ..options.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::array::codec::array_to_bytes::sharding::sharding_options::SubchunkWriteOrder;

    use super::ShardingCodecOptions;
    use zarrs_codec::CodecSpecificOptions;

    #[test]
    fn sharding_options_not_set_by_default() {
        let opts = CodecSpecificOptions::default();
        assert!(opts.get_option::<ShardingCodecOptions>().is_none());
    }

    #[test]
    fn sharding_options_present_after_set() {
        let opts = CodecSpecificOptions::default().with_option(ShardingCodecOptions::default());
        assert!(opts.get_option::<ShardingCodecOptions>().is_some());
    }

    #[test]
    fn sharding_has_option() {
        let opts = CodecSpecificOptions::default().with_option(
            ShardingCodecOptions::default().with_subchunk_write_order(SubchunkWriteOrder::C),
        );
        assert!(matches!(
            opts.get_option::<ShardingCodecOptions>()
                .unwrap()
                .subchunk_write_order(),
            SubchunkWriteOrder::C
        ));
    }
}
