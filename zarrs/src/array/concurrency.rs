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

pub use zarrs_codec::RecommendedConcurrency;
use zarrs_codec::{CodecOptions, ComputeResources, Resources};

/// Calculate the outer and inner concurrent limits given a concurrency target and their recommended concurrency.
///
/// Return is (outer, inner).
///
/// A concurrency target of zero is unconstrained: the outer concurrency is its
/// recommended maximum and the inner concurrency target remains zero.
#[must_use]
pub fn calc_concurrency_outer_inner(
    concurrency_target: usize,
    recommended_concurrency_outer: &RecommendedConcurrency,
    recommended_concurrency_inner: &RecommendedConcurrency,
) -> (usize, usize) {
    if concurrency_target == 0 {
        return (recommended_concurrency_outer.max(), 0);
    }

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

/// Split compute resources between an outer operation and its concurrent inner codecs.
#[must_use]
pub fn split_compute_resources(
    resources: &ComputeResources,
    recommended_concurrency_outer: &RecommendedConcurrency,
    recommended_concurrency_inner: &RecommendedConcurrency,
) -> (usize, ComputeResources) {
    let (outer, inner) = calc_concurrency_outer_inner(
        resources.compute_target(),
        recommended_concurrency_outer,
        recommended_concurrency_inner,
    );
    (outer, resources.clone().with_compute_target(inner))
}

/// Split resources between an outer operation and its concurrent inner operations.
///
/// Only the compute resources are split.
/// Return is (outer concurrency, resources of each inner operation).
#[must_use]
pub fn split_resources(
    resources: &Resources,
    recommended_concurrency_outer: &RecommendedConcurrency,
    recommended_concurrency_inner: &RecommendedConcurrency,
) -> (usize, Resources) {
    let (outer, compute) = split_compute_resources(
        resources.compute(),
        recommended_concurrency_outer,
        recommended_concurrency_inner,
    );
    (outer, resources.clone().with_compute(compute))
}

/// Calculate chunk concurrency and the resources passed to each chunk's codecs.
#[must_use]
pub fn concurrency_chunks_and_codec(
    num_chunks: usize,
    codec_options: &CodecOptions,
    resources: &Resources,
    codec_concurrency: &RecommendedConcurrency,
) -> (usize, Resources) {
    let chunk_concurrent_minimum = codec_options.chunk_concurrent_minimum();
    let min_concurrent_chunks = std::cmp::min(chunk_concurrent_minimum, num_chunks);
    let max_concurrent_chunks = std::cmp::max(chunk_concurrent_minimum, num_chunks);
    split_resources(
        resources,
        &RecommendedConcurrency::new(min_concurrent_chunks..max_concurrent_chunks),
        codec_concurrency,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharing_compute_splits_the_compute_target() {
        let resources = Resources::default().with_compute_target(12);
        let options = CodecOptions::default().with_chunk_concurrent_minimum(3);
        let (outer, inner) = concurrency_chunks_and_codec(
            9,
            &options,
            &resources,
            &RecommendedConcurrency::new(1..4),
        );
        assert_eq!(outer, 3);
        assert_eq!(inner.compute_target(), 4);
        assert_eq!(resources.compute_target(), 12);
        let (outer, inner) = concurrency_chunks_and_codec(
            9,
            &options,
            &resources.with_compute_target(0),
            &RecommendedConcurrency::new(1..4),
        );
        assert_eq!(outer, 9);
        assert_eq!(inner.compute_target(), 0);
    }

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

        let (self_limit, inner_limit) = calc_concurrency_outer_inner(
            0,
            &RecommendedConcurrency::new_maximum(5),
            &RecommendedConcurrency::new(7..14),
        );
        assert_eq!((self_limit, inner_limit), (5, 0));
    }
}
