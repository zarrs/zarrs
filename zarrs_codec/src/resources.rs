//! Resources available to an operation.

/// Resources available to an encode/decode or array operation.
///
/// These are passed at each call and control how much of the system an operation may use.
/// They are distinct from [`CodecOptions`](super::CodecOptions), which control the behaviour of an operation.
///
/// The default values are:
/// - `concurrent_target`: number of threads available to Rayon
/// - `chunk_concurrent_minimum`: `4`
#[derive(Debug, Clone)]
pub struct Resources {
    concurrent_target: usize,
    chunk_concurrent_minimum: usize,
}

impl Default for Resources {
    fn default() -> Self {
        Self {
            concurrent_target: rayon::current_num_threads(),
            chunk_concurrent_minimum: 4,
        }
    }
}

impl Resources {
    /// Return the concurrent target.
    ///
    /// This is the number of concurrent operations to target for an operation.
    /// Limiting concurrent operations is needed to reduce memory usage and improve performance.
    /// Concurrency is unconstrained if the concurrent target is set to zero.
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
    /// This sets the preferred minimum chunk concurrency.
    /// The concurrency of internal codecs is adjusted to accommodate for the chunk concurrency in accordance with the concurrent target.
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
}
