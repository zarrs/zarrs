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
