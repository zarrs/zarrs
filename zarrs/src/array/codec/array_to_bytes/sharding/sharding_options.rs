//! Runtime options for the sharding codec.

/// Runtime options for the [`ShardingCodec`](super::ShardingCodec).
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ShardingCodecOptions {
    // Placeholder for future sharding-specific runtime options.
}

#[cfg(test)]
mod tests {
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
}
