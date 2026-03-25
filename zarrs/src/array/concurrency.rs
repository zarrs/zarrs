//! Concurrency utilities for arrays and codecs.

// /// The preferred concurrency in a [`RecommendedConcurrency`].
// #[derive(Debug, Copy, Clone)]
// pub enum PreferredConcurrency {
//     /// Prefer the minimum concurrency.
//     ///
//     /// This is suggested in situations where memory scales with concurrency.
//     Minimum,
//     /// Prefer the maximum concurrency.
//     ///
//     /// This is suggested in situations where memory does not scale with concurrency (or does not scale much).
//     Maximum,
// }

use zarrs_codec::CodecOptions;
pub use zarrs_codec::RecommendedConcurrency;

/// Calculate the outer and inner concurrent limits given a concurrency target and their recommended concurrency.
///
/// Return is (outer, inner).
#[must_use]
pub fn calc_concurrency_outer_inner(
    concurrency_target: usize,
    recommended_concurrency_outer: &RecommendedConcurrency,
    recommended_concurrency_inner: &RecommendedConcurrency,
) -> (usize, usize) {
    let mut concurrency_inner = recommended_concurrency_inner.min();
    let mut concurrency_outer = recommended_concurrency_outer.min();

    if concurrency_inner * concurrency_outer < concurrency_target {
        // Try increasing inner
        concurrency_inner = std::cmp::min(
            concurrency_target.div_ceil(concurrency_outer),
            recommended_concurrency_inner.max(),
        );
    }

    if concurrency_inner * concurrency_outer < concurrency_target {
        // Try increasing outer
        concurrency_outer = std::cmp::min(
            concurrency_target.div_ceil(concurrency_inner),
            recommended_concurrency_outer.max(),
        );
    }

    (concurrency_outer, concurrency_inner)
}

/// Calculate the outer concurrency and inner options for a codec.
#[must_use]
pub fn concurrency_chunks_and_codec(
    concurrency_target: usize,
    num_chunks: usize,
    codec_options: &CodecOptions,
    codec_concurrency: &RecommendedConcurrency,
) -> (usize, CodecOptions) {
    // core::cmp::minmax https://github.com/rust-lang/rust/issues/115939
    let chunk_concurrent_minimum = codec_options.chunk_concurrent_minimum();
    let min_concurrent_chunks = std::cmp::min(chunk_concurrent_minimum, num_chunks);
    let max_concurrent_chunks = std::cmp::max(chunk_concurrent_minimum, num_chunks);
    let (self_concurrent_limit, codec_concurrent_limit) = calc_concurrency_outer_inner(
        concurrency_target,
        &RecommendedConcurrency::new(min_concurrent_chunks..max_concurrent_chunks),
        codec_concurrency,
    );
    let codec_options = codec_options.with_concurrent_target(codec_concurrent_limit);
    (self_concurrent_limit, codec_options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concurrent_limits() {
        let target = 32;

        let (self_limit, inner_limit) = calc_concurrency_outer_inner(
            target,
            &RecommendedConcurrency::new_minimum(24),
            &RecommendedConcurrency::new_maximum(1),
        );
        assert_eq!((self_limit, inner_limit), (32, 1));

        let (self_limit, inner_limit) = calc_concurrency_outer_inner(
            target,
            &RecommendedConcurrency::new_minimum(24),
            &RecommendedConcurrency::new(4..8),
        );
        assert_eq!((self_limit, inner_limit), (24, 4));

        let (self_limit, inner_limit) = calc_concurrency_outer_inner(
            target,
            &RecommendedConcurrency::new_maximum(5),
            &RecommendedConcurrency::new(7..12),
        );
        assert_eq!((self_limit, inner_limit), (3, 12));

        let (self_limit, inner_limit) = calc_concurrency_outer_inner(
            target,
            &RecommendedConcurrency::new_maximum(2),
            &RecommendedConcurrency::new(7..14),
        );
        assert_eq!((self_limit, inner_limit), (2, 14));
    }
}
