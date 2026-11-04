//! Resources available to an operation.

/// Compute resources available to an encode or decode.
///
/// These are the part of [`Resources`] that codecs use. They are passed to the encode and decode methods of codecs,
/// which never read storage.
///
/// The default compute target is the number of threads available to Rayon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeResources {
    compute_target: usize,
}

impl Default for ComputeResources {
    fn default() -> Self {
        Self {
            compute_target: rayon::current_num_threads(),
        }
    }
}

impl ComputeResources {
    /// Return the compute target.
    ///
    /// This is the number of concurrent compute (encoding/decoding) operations to target.
    /// Concurrency is unconstrained if the compute target is set to zero.
    #[must_use]
    pub fn compute_target(&self) -> usize {
        self.compute_target
    }

    /// Set the compute target.
    pub fn set_compute_target(&mut self, compute_target: usize) -> &mut Self {
        self.compute_target = compute_target;
        self
    }

    /// Set the compute target.
    #[must_use]
    pub fn with_compute_target(mut self, compute_target: usize) -> Self {
        self.compute_target = compute_target;
        self
    }
}

/// Resources available to an array operation.
///
/// These are passed at each call and control how much of the system an operation may use.
/// They are distinct from [`CodecOptions`](super::CodecOptions), which control the behaviour of an operation.
///
/// Codecs only receive the [`ComputeResources`].
///
/// The default values are:
/// - `compute_target`: number of threads available to Rayon
// TODO: I/O bounds and more
#[derive(Debug, Clone)]
pub struct Resources {
    compute: ComputeResources,
}

impl Default for Resources {
    fn default() -> Self {
        ComputeResources::default().into()
    }
}

impl From<ComputeResources> for Resources {
    /// Resources with `compute`.
    fn from(compute: ComputeResources) -> Self {
        Self { compute }
    }
}

impl Resources {
    /// Return the compute resources, which are passed to codecs.
    #[must_use]
    pub fn compute(&self) -> &ComputeResources {
        &self.compute
    }

    /// Set the compute resources.
    pub fn set_compute(&mut self, compute: ComputeResources) -> &mut Self {
        self.compute = compute;
        self
    }

    /// Set the compute resources.
    #[must_use]
    pub fn with_compute(mut self, compute: ComputeResources) -> Self {
        self.compute = compute;
        self
    }

    /// Return the compute target.
    ///
    /// This is the number of concurrent compute (encoding/decoding) operations to target for an operation.
    /// Limiting concurrent operations is needed to reduce memory usage and improve performance.
    /// Concurrency is unconstrained if the compute target is set to zero.
    #[must_use]
    pub fn compute_target(&self) -> usize {
        self.compute.compute_target
    }

    /// Set the compute target.
    pub fn set_compute_target(&mut self, compute_target: usize) -> &mut Self {
        self.compute.compute_target = compute_target;
        self
    }

    /// Set the compute target.
    #[must_use]
    pub fn with_compute_target(mut self, compute_target: usize) -> Self {
        self.compute.compute_target = compute_target;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_unconstrained_targets() {
        let resources = Resources::default();
        assert_eq!(resources.compute_target(), rayon::current_num_threads());
        assert_eq!(resources.compute(), &ComputeResources::default());

        let resources = resources.with_compute_target(0);
        assert_eq!(resources.compute_target(), 0);
        assert_eq!(resources.compute().compute_target(), 0);
    }

    #[test]
    fn compute_setters_and_overrides() {
        let compute = ComputeResources::default().with_compute_target(12);
        let mut resources = Resources::from(compute.clone());
        assert_eq!(resources.compute(), &compute);
        let clone = resources.clone();
        resources
            .set_compute(ComputeResources::default().with_compute_target(3))
            .set_compute_target(2);
        assert_eq!(resources.compute_target(), 2);
        assert_eq!(clone.compute_target(), 12);
        let mut compute = compute;
        compute.set_compute_target(5);
        assert_eq!(resources.with_compute(compute).compute_target(), 5);
    }
}
