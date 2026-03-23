//! Runtime options for the sharding codec.

/// Write order for subchunks within a shard
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum SubchunkWriteOrder {
    /// Random order.
    /// This setting means that there is no order guarantee, and not that the order will be guranteed to be random.
    /// Because subchunk writing is parallelized, it will often appear that subchunks are written at random with this setting although this is dependent on the parallelizable workload.
    Random,
    /// C order i.e., row-major
    C,
    // TODO: Morton order - depend on https://docs.rs/morton-encoding/latest/morton_encoding/?
}

/// Runtime options for the [`ShardingCodec`](super::ShardingCodec).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ShardingCodecOptions {
    subchunk_write_order: SubchunkWriteOrder,
}

impl Default for ShardingCodecOptions {
    fn default() -> Self {
        Self {
            subchunk_write_order: SubchunkWriteOrder::Random,
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
