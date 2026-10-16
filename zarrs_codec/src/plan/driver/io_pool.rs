//! A thread pool for blocking I/O.

use std::any::Any;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// The default number of threads of the global I/O pool.
const DEFAULT_IO_THREADS: usize = 32;

type Job = Box<dyn FnOnce() + Send + 'static>;

/// A pool of threads for blocking I/O.
///
/// Synchronous drives perform storage requests on an I/O pool so that the number of concurrent
/// requests is not limited by the number of compute (Rayon) threads, and so that compute threads
/// are not blocked waiting for storage.
///
/// I/O threads never run compute work or wait on Rayon, so they cannot deadlock with compute.
#[derive(Debug)]
pub struct IoPool {
    sender: crossbeam_channel::Sender<Job>,
    num_threads: usize,
}

impl IoPool {
    /// Create an I/O pool with `num_threads` threads named `zarrs-io-N`.
    ///
    /// The threads exit when the pool is dropped.
    ///
    /// # Panics
    /// Panics if `num_threads` is zero or a thread cannot be spawned.
    #[must_use]
    pub fn new(num_threads: usize) -> Self {
        assert!(num_threads > 0, "an I/O pool must have at least one thread");
        let (sender, receiver) = crossbeam_channel::unbounded::<Job>();
        for i in 0..num_threads {
            let receiver = receiver.clone();
            std::thread::Builder::new()
                .name(format!("zarrs-io-{i}"))
                .spawn(move || {
                    while let Ok(job) = receiver.recv() {
                        job();
                    }
                })
                .expect("failed to spawn an I/O thread");
        }
        Self {
            sender,
            num_threads,
        }
    }

    /// The global I/O pool, with 32 threads.
    #[must_use]
    pub fn global() -> &'static Self {
        static GLOBAL: OnceLock<IoPool> = OnceLock::new();
        GLOBAL.get_or_init(|| Self::new(DEFAULT_IO_THREADS))
    }

    /// The number of threads in the pool.
    #[must_use]
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Run `f` with a scope in which jobs that borrow from outside the scope can be spawned on the pool.
    ///
    /// All jobs spawned in the scope have completed when this function returns, even if `f` or a job panics.
    /// A panic of `f` or of a job is resumed once all jobs have completed.
    pub fn scope<'env, F, R>(&self, f: F) -> R
    where
        F: for<'scope> FnOnce(&'scope IoScope<'scope, 'env>) -> R,
    {
        let scope = IoScope {
            pool: self,
            state: Arc::new(ScopeState::default()),
            _scope: PhantomData,
            _env: PhantomData,
        };
        let result = catch_unwind(AssertUnwindSafe(|| f(&scope)));
        scope.state.wait();
        if let Some(panic) = scope.state.panic.lock().unwrap().take() {
            resume_unwind(panic);
        }
        match result {
            Ok(result) => result,
            Err(panic) => resume_unwind(panic),
        }
    }
}

#[derive(Default)]
struct ScopeState {
    pending: Mutex<usize>,
    completed: Condvar,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl ScopeState {
    fn wait(&self) {
        let mut pending = self.pending.lock().unwrap();
        while *pending > 0 {
            pending = self.completed.wait(pending).unwrap();
        }
    }

    fn finish(&self) {
        let mut pending = self.pending.lock().unwrap();
        *pending -= 1;
        if *pending == 0 {
            self.completed.notify_all();
        }
    }
}

/// A scope for spawning jobs on an [`IoPool`]. See [`IoPool::scope`].
pub struct IoScope<'scope, 'env: 'scope> {
    pool: &'scope IoPool,
    state: Arc<ScopeState>,
    // Invariant lifetimes, as for `std::thread::Scope`.
    _scope: PhantomData<&'scope mut &'scope ()>,
    _env: PhantomData<&'env mut &'env ()>,
}

impl<'scope> IoScope<'scope, '_> {
    /// Spawn a job on the pool.
    ///
    /// The job may borrow anything that outlives the scope.
    pub fn spawn<F>(&'scope self, job: F)
    where
        F: FnOnce() + Send + 'scope,
    {
        *self.state.pending.lock().unwrap() += 1;
        let state = self.state.clone();
        let job: Box<dyn FnOnce() + Send + 'scope> = Box::new(move || {
            if let Err(panic) = catch_unwind(AssertUnwindSafe(job)) {
                state.panic.lock().unwrap().get_or_insert(panic);
            }
            state.finish();
        });
        // SAFETY: `IoPool::scope` does not return until every job spawned in the scope has
        // completed (`ScopeState::wait`), including when the scope closure or a job panics,
        // so nothing borrowed by the job for `'scope` is used after it is freed.
        let job: Job =
            unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + 'scope>, Job>(job) };
        self.pool
            .sender
            .send(job)
            .expect("the I/O pool threads are running");
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    #[test]
    fn scoped_jobs_borrow_and_complete() {
        let pool = IoPool::new(4);
        let mut data = vec![0u32; 64];
        let concurrent = AtomicUsize::new(0);
        let peak = AtomicUsize::new(0);
        pool.scope(|scope| {
            for chunk in data.chunks_mut(8) {
                let concurrent = &concurrent;
                let peak = &peak;
                scope.spawn(move || {
                    let now = concurrent.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(if cfg!(miri) { 1 } else { 10 }));
                    for value in chunk.iter_mut() {
                        *value += 1;
                    }
                    concurrent.fetch_sub(1, Ordering::SeqCst);
                });
            }
        });
        assert_eq!(data, vec![1; 64]);
        assert!(peak.load(Ordering::SeqCst) > 1);
        assert!(peak.load(Ordering::SeqCst) <= 4);
    }

    #[test]
    fn job_panics_are_resumed_after_all_jobs_complete() {
        let pool = IoPool::new(2);
        let completed = AtomicUsize::new(0);
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            pool.scope(|scope| {
                scope.spawn(|| panic!("job panicked"));
                for _ in 0..4 {
                    scope.spawn(|| {
                        std::thread::sleep(Duration::from_millis(1));
                        completed.fetch_add(1, Ordering::SeqCst);
                    });
                }
            });
        }));
        assert!(result.is_err());
        assert_eq!(completed.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn scope_panics_wait_for_jobs() {
        let pool = IoPool::new(2);
        let completed = AtomicUsize::new(0);
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            pool.scope(|scope| {
                scope.spawn(|| {
                    std::thread::sleep(Duration::from_millis(5));
                    completed.fetch_add(1, Ordering::SeqCst);
                });
                panic!("scope panicked");
            });
        }));
        assert!(result.is_err());
        assert_eq!(completed.load(Ordering::SeqCst), 1);
    }
}
