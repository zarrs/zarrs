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

use super::codec::CodecOptions;
use crate::config::global_config;

/// The recommended concurrency of a codec includes the most efficient and maximum recommended concurrency.
///
/// Consider a chain that does slow decoding first on a single thread, but subsequent codecs can run on multiple threads.
/// In this case, recommended concurrency is best expressed by two numbers:
///    - the efficient concurrency, equal to the minimum of codecs
///    - the maximum concurrency, equal to the maximum of codecs
// TODO: Compression codec example in docs?
#[derive(Debug, Clone)]
pub struct RecommendedConcurrency {
    /// The range is just used for its constructor and start/end, no iteration
    range: std::ops::Range<usize>,
    // preferred_concurrency: PreferredConcurrency,
}

impl RecommendedConcurrency {
    /// Create a new recommended concurrency struct with an explicit concurrency range and preferred concurrency.
    ///
    /// A minimum concurrency of zero is interpreted as a minimum concurrency of one.
    #[must_use]
    pub fn new(range: impl std::ops::RangeBounds<usize>) -> Self {
        // , preferred_concurrency: PreferredConcurrency
        let start = match range.start_bound() {
            std::ops::Bound::Included(start) => *start,
            std::ops::Bound::Excluded(start) => start.saturating_add(1),
            std::ops::Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            std::ops::Bound::Excluded(end) => *end,
            std::ops::Bound::Included(end) => end.saturating_add(1),
            std::ops::Bound::Unbounded => usize::MAX,
        };
        Self {
            range: start.max(1)..end.max(1),
            // preferred_concurrency,
        }
    }

    /// Create a new recommended concurrency struct with a specified minimum concurrency and unbounded maximum concurrency.
    #[must_use]
    pub fn new_minimum(minimum: usize) -> Self {
        Self::new(minimum..)
    }

    /// Create a new recommended concurrency struct with a specified maximum concurrency.
    #[must_use]
    pub fn new_maximum(maximum: usize) -> Self {
        Self::new(..maximum)
    }

    /// Return the minimum concurrency.
    #[must_use]
    pub fn min(&self) -> usize {
        self.range.start
    }

    /// Return the maximum concurrency.
    #[must_use]
    pub fn max(&self) -> usize {
        self.range.end
    }

    // /// Return the preferred concurrency.
    // #[must_use]
    // pub fn preferred(&self) -> usize {
    //     match self.preferred_concurrency {
    //         PreferredConcurrency::Minimum => self.range.start,
    //         PreferredConcurrency::Maximum => self.range.end,
    //     }
    // }
}

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
    let chunk_concurrent_minimum = global_config().chunk_concurrent_minimum();
    let min_concurrent_chunks = std::cmp::min(chunk_concurrent_minimum, num_chunks);
    let max_concurrent_chunks = std::cmp::max(chunk_concurrent_minimum, num_chunks);
    let (self_concurrent_limit, codec_concurrent_limit) = calc_concurrency_outer_inner(
        concurrency_target,
        &RecommendedConcurrency::new(min_concurrent_chunks..max_concurrent_chunks),
        codec_concurrency,
    );
    let codec_options = codec_options
        .into_builder()
        .concurrent_target(codec_concurrent_limit)
        .build();
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
