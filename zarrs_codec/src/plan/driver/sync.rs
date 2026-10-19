//! The synchronous parallel driver.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::time::Duration;

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use zarrs_storage::StorageError;

use super::super::{ComputeCtx, SyncIo, Task, TaskValue};
use smallvec::SmallVec;
use zarrs_storage::ReadCoalescing;

use super::core::{
    ByteBudget, Completed, InlineIo, IoRequest, IoSlot, IoWork, NodeId, Report, Scheduler, Work,
    WriteWork, generation_window, read_coalescing,
};
use super::serial::drive_single;
use super::{IoPool, IoScope};
use crate::{CodecError, ReadDispatch, Resources};

/// Execute `tasks` with up to [`Resources::compute_target`] concurrent compute steps on Rayon,
/// and up to [`Resources::io_target`] concurrent storage requests.
///
/// I/O and compute are interleaved: a task computes as soon as its reads complete.
/// Returns the values of `tasks`.
///
/// A drive starts on the calling thread, and only starts parallel work once more than one
/// task is ready, so a drive of a single task that does not fan out runs entirely on the calling thread.
///
/// Reads are performed on Rayon threads while storage requests are fast, and otherwise on the [`IoPool`]
/// (so more reads than Rayon threads can be in flight), unless set by [`Resources::read_dispatch`].
/// Every storage request holds a permit of the [`Resources::io_limiter`], if set.
/// Reads are not dispatched while the bytes read and not yet computed on would exceed the [`Resources::fetched_bytes_limit`].
///
/// Reads on the I/O pool of nearby byte ranges of the same key are coalesced into one request as set by the
/// [`SyncIo::read_coalescing`] hint (or [`Resources::read_coalescing`]), and the bytes of each read are slices of the
/// bytes of the request.
///
/// # Errors
/// Returns the first error of a read, write, or compute step.
/// Once an error occurs, no further work is started, and the drive returns once its work in flight completes.
///
/// # Panics
/// Resumes the first panic of a compute step or storage request once the work in flight completes.
pub fn drive_sync<'a>(
    mut tasks: Vec<Task<'a>>,
    io: &dyn SyncIo,
    resources: &Resources,
) -> Result<Vec<TaskValue<'a>>, CodecError> {
    let ctx = ComputeCtx::new(resources.compute().clone());
    let scheduler = if let [_] = tasks.as_slice() {
        let task = tasks.pop().expect("one task");
        match drive_single(task, io, resources, &ctx, drive_shape(resources))? {
            Ok(value) => return Ok(vec![value]),
            Err(scheduler) => scheduler,
        }
    } else {
        Scheduler::new(tasks)
    };
    drive_scheduler(scheduler, io, resources, &ctx)
}

/// Execute `task` as [`drive_sync`], returning its value.
///
/// # Errors
/// Returns the first error of a read, write, or compute step.
///
/// # Panics
/// Resumes the first panic of a compute step or storage request once the work in flight completes.
pub fn drive_sync_one<'a>(
    task: Task<'a>,
    io: &dyn SyncIo,
    resources: &Resources,
) -> Result<TaskValue<'a>, CodecError> {
    let ctx = ComputeCtx::new(resources.compute().clone());
    match drive_single(task, io, resources, &ctx, drive_shape(resources))? {
        Ok(value) => Ok(value),
        Err(scheduler) => Ok(drive_scheduler(scheduler, io, resources, &ctx)?
            .pop()
            .expect("one root value")),
    }
}

/// The generation window and parallelism of a drive with `resources`.
fn drive_shape(resources: &Resources) -> (usize, usize) {
    (generation_window(resources), parallelism(resources))
}

/// The number of compute steps of a drive with `resources` that run in parallel.
fn parallelism(resources: &Resources) -> usize {
    resources.compute_limit().min(rayon::current_num_threads())
}

fn drive_scheduler<'a>(
    mut scheduler: Scheduler<'a>,
    io: &dyn SyncIo,
    resources: &Resources,
    ctx: &ComputeCtx,
) -> Result<Vec<TaskValue<'a>>, CodecError> {
    let reads = DriveIo::new(io, resources);
    let parallelism = parallelism(resources);
    scheduler.set_parallelism(parallelism);
    scheduler.set_stream_window(generation_window(resources));
    scheduler.set_max_deferred(resources.io_limit().min(1024));

    // Run on the calling thread while there is at most one thing to do.
    // Inline reads occupy a compute thread, so with one compute thread everything runs one at a time on the calling thread.
    loop {
        scheduler.refill(2);
        match scheduler.queued() {
            0 => return scheduler.finish(),
            1 => {
                scheduler.run_one(&reads, ctx);
            }
            _ if parallelism == 1 && reads.enabled() => {
                scheduler.run_one(&reads, ctx);
            }
            _ => break,
        }
    }

    let mut driver = Driver {
        window: generation_window(resources),
        reads: &reads,
        scheduler,
        resources,
        compute_limit: resources.compute_limit(),
        coalescing: read_coalescing(resources, io.read_coalescing()),
        planning_in_flight: 0,
        compute_in_flight: 0,
        io_in_flight: 0,
        budget: ByteBudget::new(resources.fetched_bytes_limit()),
        panic: None,
    };
    if reads.forced == Some(ReadDispatch::Inline) {
        rayon::in_place_scope(|rayon_scope| driver.run(rayon_scope, None));
    } else {
        let pool = resources.io_pool().unwrap_or_else(|| IoPool::global());
        pool.scope(|io_scope| {
            rayon::in_place_scope(|rayon_scope| driver.run(rayon_scope, Some(io_scope)));
        });
    }
    if let Some(panic) = driver.panic {
        resume_unwind(panic);
    }
    driver.scheduler.finish()
}

/// The mean storage request time below which a drive reads on compute threads.
///
/// Slower reads go to the I/O pool, where the cost of handing them over is small in comparison,
/// and where more reads than compute threads can be in flight.
const INLINE_READ_THRESHOLD: Duration = Duration::from_micros(100);

/// A value on its own cache line (of up to 128 bytes), so that threads updating it do not contend with threads using
/// neighbouring values.
#[repr(align(128))]
struct CacheAligned<T>(T);

impl<T> std::ops::Deref for CacheAligned<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

/// The reads of a sync drive, which are inline while storage requests are fast (or as set by [`Resources::read_dispatch`]).
struct DriveIo<'d> {
    io: &'d dyn SyncIo,
    resources: &'d Resources,
    forced: Option<ReadDispatch>,
    /// The maximum number of storage requests in flight.
    io_limit: usize,
    /// An exponentially weighted mean of the duration of storage requests, in nanoseconds (0 until a request completes).
    mean_ns: CacheAligned<AtomicU64>,
    /// The number of storage requests in flight, dispatched by the driver or performed inline by compute work.
    io_in_flight: CacheAligned<AtomicUsize>,
}

impl<'d> DriveIo<'d> {
    fn new(io: &'d dyn SyncIo, resources: &'d Resources) -> Self {
        Self {
            io,
            resources,
            forced: resources.read_dispatch(),
            io_limit: resources.io_limit(),
            mean_ns: CacheAligned(AtomicU64::new(0)),
            io_in_flight: CacheAligned(AtomicUsize::new(0)),
        }
    }

    /// Returns true if storage requests should currently be performed inline.
    fn enabled(&self) -> bool {
        match self.forced {
            Some(ReadDispatch::Inline) => true,
            Some(ReadDispatch::Pooled) => false,
            None => {
                self.mean_ns.load(Ordering::Relaxed)
                    < u64::try_from(INLINE_READ_THRESHOLD.as_nanos()).unwrap()
            }
        }
    }

    /// Count a storage request towards the I/O target if it admits another, returning true if so.
    fn try_acquire(&self) -> bool {
        let mut n = self.io_in_flight.load(Ordering::Acquire);
        while n < self.io_limit {
            match self.io_in_flight.compare_exchange_weak(
                n,
                n + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(current) => n = current,
            }
        }
        false
    }

    /// Release a storage request counted by [`try_acquire`](Self::try_acquire).
    fn release(&self) {
        self.io_in_flight.fetch_sub(1, Ordering::AcqRel);
    }

    /// Record the duration of a storage request.
    fn record(&self, elapsed: Duration) {
        if self.forced.is_some() {
            return;
        }
        let elapsed = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        let mean = self.mean_ns.load(Ordering::Relaxed);
        // Concurrent updates may be lost, which only delays the mean
        let updated = if mean == 0 {
            elapsed.max(1)
        } else {
            mean - mean / 8 + elapsed / 8
        };
        // Small changes are not stored, so that threads reading at a steady rate do not contend on the mean. The mean then
        // lags durations within a quarter of it, which is precise enough to compare with the threshold.
        if updated.abs_diff(mean) > mean / 32 {
            self.mean_ns.store(updated, Ordering::Relaxed);
        }
    }
}

impl InlineIo for DriveIo<'_> {
    fn admit(&self) -> Option<IoSlot<'_>> {
        (self.enabled() && self.try_acquire()).then(|| IoSlot(Some(&*self.io_in_flight)))
    }

    fn enabled(&self) -> bool {
        DriveIo::enabled(self)
    }

    fn read(&self, request: &IoRequest) -> Result<Option<super::super::ReadBytes>, StorageError> {
        let _permit = self.resources.io_permit_blocking();
        let start = Instant::now();
        let result = self.io.read(request.key, &request.ranges);
        self.record(start.elapsed());
        result
    }

    fn write(&self, op: super::super::WriteOp<'_>) -> Result<(), StorageError> {
        let _permit = self.resources.io_permit_blocking();
        let start = Instant::now();
        let result = self.io.write(op);
        self.record(start.elapsed());
        result
    }
}

/// The maximum number of compute steps run by one Rayon job.
const MAX_BATCH: usize = 64;

/// A completed compute step.
struct Computed<'a> {
    completed: Completed<'a>,
    /// The bytes read for the compute step.
    fetched: u64,
    /// The bytes reserved for the read of an inline read.
    reserved: Option<u64>,
}

/// An event from work in flight.
enum Event<'a> {
    /// A batch of compute steps completed (possibly after their reads, for inline reads).
    Computed {
        results: SmallVec<[Computed<'a>; 1]>,
        /// Whether the batch performed inline reads.
        read: bool,
        /// Whether the batch was counted as planning reads.
        planning: bool,
    },
    /// A pooled read (of one or more coalesced reads) completed.
    Read {
        reserved: u64,
        results: SmallVec<[Result<Work<'a>, CodecError>; 1]>,
    },
    /// A running job created and joined an indexed generator.
    Report(Report<'a>),
    /// A write completed.
    Written(NodeId, Result<(), StorageError>),
    /// Work panicked.
    Panicked {
        payload: Box<dyn Any + Send>,
        compute: bool,
        io: bool,
    },
}

/// The number of queued items to run in one job, so that each of `free` threads gets several jobs.
///
/// Batching amortises the cost of a Rayon job and an event over small compute steps.
fn batch_size(queued: usize, free: usize) -> usize {
    (queued / free.saturating_mul(4).max(1)).clamp(1, MAX_BATCH)
}

struct Driver<'d, 'a> {
    window: usize,
    reads: &'d DriveIo<'d>,
    scheduler: Scheduler<'a>,
    resources: &'d Resources,
    compute_limit: usize,
    compute_in_flight: usize,
    /// How queued reads of the same key are coalesced, if at all.
    coalescing: Option<ReadCoalescing>,
    /// The compute jobs in flight that are likely to only plan reads (see [`Work::is_planning_range`]).
    planning_in_flight: usize,
    /// The storage requests dispatched by the driver in flight, each counted by [`DriveIo::try_acquire`].
    io_in_flight: usize,
    budget: ByteBudget,
    panic: Option<Box<dyn Any + Send>>,
}

impl<'d, 'a: 'd> Driver<'d, 'a> {
    fn run<'scope>(
        &mut self,
        rayon_scope: &rayon::Scope<'scope>,
        io_scope: Option<&'scope IoScope<'scope, '_>>,
    ) where
        'd: 'scope,
    {
        let (sender, receiver) = crossbeam_channel::unbounded::<Event<'a>>();
        loop {
            if !self.scheduler.failed() {
                self.dispatch(rayon_scope, io_scope, &sender);
            }
            if self.compute_in_flight == 0 && self.io_in_flight == 0 {
                debug_assert!(self.scheduler.failed() || self.scheduler.queued() == 0);
                break;
            }
            let event = wait(&receiver);
            self.handle(event);
        }
    }

    /// The resources of a compute step: a share of the compute target.
    fn compute_ctx(&self) -> ComputeCtx {
        let busy = self.compute_in_flight + self.scheduler.compute_queued() + 1;
        ComputeCtx::new(self.resources.compute().with_compute_share(busy))
    }

    #[allow(clippy::too_many_lines)]
    fn dispatch<'scope>(
        &mut self,
        rayon_scope: &rayon::Scope<'scope>,
        io_scope: Option<&'scope IoScope<'scope, '_>>,
        sender: &crossbeam_channel::Sender<Event<'a>>,
    ) where
        'd: 'scope,
    {
        let ctx = self.compute_ctx();
        self.scheduler.run_folds(&ctx);
        self.scheduler.refill(self.window);

        // Compute first, so the bytes of completed reads are consumed before more are read
        while self.compute_in_flight < self.compute_limit {
            let queued = self.scheduler.compute_queued();
            if queued == 0 {
                break;
            }
            let batch: SmallVec<[Work<'a>; 1]> =
                (0..batch_size(queued, self.compute_limit - self.compute_in_flight))
                    .map_while(|_| self.scheduler.pop_compute())
                    .collect();
            let ctx = self.compute_ctx();
            self.compute_in_flight += 1;
            let planning = batch.iter().any(Work::is_planning_range);
            if planning {
                self.planning_in_flight += 1;
            }
            let sender = sender.clone();
            let reads = self.reads;
            rayon_scope.spawn(move |_| {
                let event = match catch_unwind(AssertUnwindSafe(|| {
                    batch
                        .into_iter()
                        .map(|work| {
                            let fetched = work.fetched();
                            let on_report = |report| {
                                let _ = sender.send(Event::Report(report));
                            };
                            let completed = work.run(&ctx, Some(reads), Some(&on_report));
                            Computed {
                                completed,
                                fetched,
                                reserved: None,
                            }
                        })
                        .collect()
                })) {
                    Ok(results) => Event::Computed {
                        results,
                        read: false,
                        planning,
                    },
                    Err(payload) => Event::Panicked {
                        payload,
                        compute: true,
                        io: false,
                    },
                };
                let _ = sender.send(event);
            });
        }

        // Then writes and reads, while the I/O target admits them (including storage requests performed inline by compute work)
        while self.scheduler.io_queued() + self.scheduler.writes_queued() > 0
            && self.reads.try_acquire()
        {
            if let Some(write) = self.scheduler.pop_write() {
                self.io_in_flight += 1;
                let job = write_job(self.reads, write, sender.clone());
                match io_scope {
                    Some(io_scope) => io_scope.spawn(job),
                    None => rayon_scope.spawn(move |_| job()),
                }
                continue;
            }
            match io_scope {
                Some(io_scope) if !self.reads.enabled() => {
                    // Queued reads of the same key are coalesced into one request.
                    // Pooled reads are slow, so they wait for jobs planning more reads (which finish quickly) to
                    // queue them, rather than being dispatched before reads they could coalesce with.
                    if self.coalescing.is_some() && self.planning_in_flight > 0 {
                        self.reads.release();
                        break;
                    }
                    let Some((group, reserved)) = self
                        .scheduler
                        .pop_read_group(self.coalescing, &mut self.budget)
                    else {
                        self.reads.release();
                        break;
                    };
                    let group = match group {
                        Ok(group) => group,
                        Err(err) => {
                            self.reads.release();
                            self.budget.cancel(reserved);
                            self.scheduler.fail(err);
                            break;
                        }
                    };
                    self.io_in_flight += 1;
                    let (reads, sender) = (self.reads, sender.clone());
                    io_scope.spawn(move || {
                        let event = match catch_unwind(AssertUnwindSafe(|| {
                            let result = reads.read(group.request());
                            group.complete(result)
                        })) {
                            Ok(results) => Event::Read { reserved, results },
                            Err(payload) => Event::Panicked {
                                payload,
                                compute: false,
                                io: true,
                            },
                        };
                        let _ = sender.send(event);
                    });
                }
                Some(_) | None => {
                    // An inline read occupies a compute thread for its read and compute step
                    if self.compute_in_flight >= self.compute_limit {
                        self.reads.release();
                        break;
                    }
                    let size = batch_size(
                        self.scheduler.io_queued(),
                        self.compute_limit - self.compute_in_flight,
                    );
                    let mut batch: SmallVec<[(IoWork<'a>, IoRequest, u64); 1]> = SmallVec::new();
                    while batch.len() < size {
                        let Some(reserved) = self
                            .scheduler
                            .peek_io_hint()
                            .and_then(|hint| self.budget.try_reserve(hint))
                        else {
                            break;
                        };
                        let mut read = self.scheduler.pop_io().expect("a read was admitted");
                        match read.request() {
                            Ok(request) => batch.push((read, request, reserved)),
                            Err(err) => {
                                self.budget.cancel(reserved);
                                self.scheduler.fail(err);
                                break;
                            }
                        }
                    }
                    if batch.is_empty() {
                        self.reads.release();
                        break;
                    }
                    let ctx = self.compute_ctx();
                    self.io_in_flight += 1;
                    self.compute_in_flight += 1;
                    let (reads, sender) = (self.reads, sender.clone());
                    rayon_scope.spawn(move |_| {
                        let event = match catch_unwind(AssertUnwindSafe(|| {
                            batch
                                .into_iter()
                                .map(|(read, request, reserved)| {
                                    let (fetched, completed) =
                                        inline_read_compute(reads, read, &request, &ctx);
                                    Computed {
                                        completed,
                                        fetched,
                                        reserved: Some(reserved),
                                    }
                                })
                                .collect()
                        })) {
                            Ok(results) => Event::Computed {
                                results,
                                read: true,
                                planning: false,
                            },
                            Err(payload) => Event::Panicked {
                                payload,
                                compute: true,
                                io: true,
                            },
                        };
                        let _ = sender.send(event);
                    });
                    if self.scheduler.failed() {
                        break;
                    }
                }
            }
        }
    }

    fn handle(&mut self, event: Event<'a>) {
        match event {
            Event::Computed {
                results,
                read,
                planning,
            } => {
                self.compute_in_flight -= 1;
                if planning {
                    self.planning_in_flight -= 1;
                }
                if read {
                    self.io_in_flight -= 1;
                    self.reads.release();
                }
                for Computed {
                    completed,
                    fetched,
                    reserved,
                } in results
                {
                    if let Some(reserved) = reserved {
                        self.budget.arrived(reserved, fetched);
                    }
                    self.budget.release(fetched);
                    self.scheduler.complete_work(completed);
                }
            }
            Event::Read { reserved, results } => {
                self.io_in_flight -= 1;
                self.reads.release();
                self.scheduler
                    .complete_read_group(&mut self.budget, reserved, results);
            }
            Event::Report(report) => self.scheduler.report(report),
            Event::Written(node, result) => {
                self.io_in_flight -= 1;
                self.reads.release();
                self.scheduler.complete_write(node, result);
            }
            Event::Panicked {
                payload,
                compute,
                io,
            } => {
                if compute {
                    self.compute_in_flight -= 1;
                }
                if io {
                    self.io_in_flight -= 1;
                    self.reads.release();
                }
                self.panic.get_or_insert(payload);
                self.scheduler
                    .fail(CodecError::Other("a task panicked".to_string()));
            }
        }
    }
}

/// Read, then run the compute step on the bytes read.
///
/// Returns the bytes read, and the result of the compute step.
fn inline_read_compute<'a>(
    reads: &DriveIo<'_>,
    read: IoWork<'a>,
    request: &IoRequest,
    ctx: &ComputeCtx,
) -> (u64, Completed<'a>) {
    let node = read.node;
    match read.into_work(request, reads.read(request)) {
        Ok(work) => {
            let fetched = work.fetched();
            (fetched, work.run(ctx, None, None))
        }
        Err(err) => (0, Completed::Step(node, Err(err))),
    }
}

fn write_job<'s, 'a: 's>(
    io: &'s DriveIo<'s>,
    write: WriteWork<'a>,
    sender: crossbeam_channel::Sender<Event<'a>>,
) -> impl FnOnce() + Send + 's {
    move || {
        let WriteWork { node, op } = write;
        let event = match catch_unwind(AssertUnwindSafe(|| io.write(op))) {
            Ok(result) => Event::Written(node, result),
            Err(payload) => Event::Panicked {
                payload,
                compute: false,
                io: true,
            },
        };
        let _ = sender.send(event);
    }
}

/// Wait for the next event.
///
/// If the calling thread is a Rayon thread, it runs other Rayon work while waiting,
/// so that a drive on a Rayon thread cannot deadlock a pool with few threads.
fn wait<'a>(receiver: &crossbeam_channel::Receiver<Event<'a>>) -> Event<'a> {
    if rayon::current_thread_index().is_none() {
        return receiver.recv().expect("the driver holds a sender");
    }
    loop {
        if let Ok(event) = receiver.try_recv() {
            return event;
        }
        match rayon::yield_now() {
            Some(rayon::Yield::Executed) => {}
            _ => {
                if let Ok(event) = receiver.recv_timeout(Duration::from_micros(50)) {
                    return event;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;

    use crate::plan::Outcome;
    use zarrs_storage::byte_range::ByteRange;

    use super::super::super::{ByteSource, Join, ReadBytes, SourceKey, WriteOp};
    use super::super::tests::MemoryIo;
    use super::*;

    fn drive_sync_inline<'a>(
        tasks: Vec<Task<'a>>,
        io: &dyn SyncIo,
        resources: &Resources,
    ) -> Result<Vec<TaskValue<'a>>, CodecError> {
        drive_sync(
            tasks,
            io,
            &resources
                .clone()
                .with_read_dispatch(Some(ReadDispatch::Inline)),
        )
    }

    fn drive_sync_pooled<'a>(
        tasks: Vec<Task<'a>>,
        io: &dyn SyncIo,
        resources: &Resources,
    ) -> Result<Vec<TaskValue<'a>>, CodecError> {
        drive_sync(
            tasks,
            io,
            &resources
                .clone()
                .with_read_dispatch(Some(ReadDispatch::Pooled)),
        )
    }

    mod adaptive {
        crate::plan::driver::tests::driver_tests!(super::super::drive_sync);
    }

    mod pooled {
        crate::plan::driver::tests::driver_tests!(super::drive_sync_pooled);
    }

    mod inline {
        crate::plan::driver::tests::driver_tests!(super::drive_sync_inline);
    }

    /// Tracks the number of concurrent operations and the peak.
    #[derive(Default)]
    struct Gauge {
        now: AtomicUsize,
        peak: AtomicUsize,
    }

    impl Gauge {
        fn enter(&self) {
            let now = self.now.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
        }

        fn exit(&self) {
            self.now.fetch_sub(1, Ordering::SeqCst);
        }

        fn peak(&self) -> usize {
            self.peak.load(Ordering::SeqCst)
        }
    }

    /// An I/O backend with read latency that records concurrency and timing.
    struct LatencyIo {
        inner: MemoryIo,
        latency: Duration,
        reads: Gauge,
        /// Reads and writes.
        requests: Gauge,
        /// The number of reads on I/O pool threads.
        pooled_reads: AtomicUsize,
        /// The number of writes on I/O pool threads.
        pooled_writes: AtomicUsize,
        read_threads: Mutex<Vec<std::thread::ThreadId>>,
        last_read_end: Mutex<Option<Instant>>,
    }

    impl LatencyIo {
        fn new(num_keys: usize, latency: Duration) -> Self {
            Self {
                inner: MemoryIo::with_values((0..num_keys).map(|i| (i, vec![1, 2, 3]))),
                latency,
                reads: Gauge::default(),
                requests: Gauge::default(),
                pooled_reads: AtomicUsize::default(),
                pooled_writes: AtomicUsize::default(),
                read_threads: Mutex::default(),
                last_read_end: Mutex::default(),
            }
        }
    }

    impl SyncIo for LatencyIo {
        fn read(
            &self,
            key: SourceKey,
            ranges: &[ByteRange],
        ) -> Result<Option<ReadBytes>, StorageError> {
            self.reads.enter();
            self.requests.enter();
            if std::thread::current()
                .name()
                .is_some_and(|name| name.starts_with("zarrs-io"))
            {
                self.pooled_reads.fetch_add(1, Ordering::SeqCst);
            }
            self.read_threads
                .lock()
                .unwrap()
                .push(std::thread::current().id());
            std::thread::sleep(self.latency);
            let result = self.inner.read(key, ranges);
            *self.last_read_end.lock().unwrap() = Some(Instant::now());
            self.requests.exit();
            self.reads.exit();
            result
        }

        fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
            if std::thread::current()
                .name()
                .is_some_and(|name| name.starts_with("zarrs-io"))
            {
                self.pooled_writes.fetch_add(1, Ordering::SeqCst);
            }
            self.requests.enter();
            std::thread::sleep(self.latency);
            let result = self.inner.write(op);
            self.requests.exit();
            result
        }
    }

    /// Tasks that read keys and then compute for `compute` time.
    fn read_compute_tasks<'a>(
        num_keys: usize,
        compute: Duration,
        gauge: &'a Gauge,
        first_compute: &'a Mutex<Option<Instant>>,
    ) -> Vec<Task<'a>> {
        (0..num_keys)
            .map(|key| {
                Task::read(
                    ByteSource::external(SourceKey(key)),
                    [ByteRange::FromStart(0, None)],
                    move |reads, _| {
                        assert_eq!(reads[0].as_ref().unwrap().as_slice(), &[1, 2, 3]);
                        gauge.enter();
                        first_compute
                            .lock()
                            .unwrap()
                            .get_or_insert_with(Instant::now);
                        std::thread::sleep(compute);
                        gauge.exit();
                        Ok(Outcome::Done)
                    },
                )
            })
            .collect()
    }

    #[test]
    fn io_and_compute_limits() {
        for dispatch in [ReadDispatch::Pooled, ReadDispatch::Inline] {
            let io = LatencyIo::new(48, Duration::from_millis(2));
            let compute = Gauge::default();
            let first_compute = Mutex::default();
            let resources = Resources::default()
                .with_compute_target(3)
                .with_io_target(6)
                .with_read_dispatch(Some(dispatch));
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(8)
                .build()
                .unwrap();
            pool.install(|| {
                drive_sync(
                    read_compute_tasks(48, Duration::from_millis(2), &compute, &first_compute),
                    &io,
                    &resources,
                )
            })
            .unwrap();
            assert!(
                compute.peak() <= 3,
                "{dispatch:?}: compute peak {}",
                compute.peak()
            );
            assert!(
                compute.peak() > 1,
                "{dispatch:?}: compute peak {}",
                compute.peak()
            );
            match dispatch {
                // Reads are not limited by compute
                ReadDispatch::Pooled => {
                    assert!(io.reads.peak() <= 6, "reads peak {}", io.reads.peak());
                    assert!(io.reads.peak() > 3, "reads peak {}", io.reads.peak());
                }
                // A read occupies a compute thread
                ReadDispatch::Inline => {
                    assert!(io.reads.peak() <= 3, "reads peak {}", io.reads.peak());
                }
            }
            // I/O and compute interleave
            assert!(
                first_compute.lock().unwrap().unwrap() < io.last_read_end.lock().unwrap().unwrap()
            );
        }
    }

    #[test]
    fn io_concurrency_exceeds_rayon_threads() {
        // Reads move to the I/O pool once they are found to be slow
        let io = LatencyIo::new(32, Duration::from_millis(20));
        let compute = Gauge::default();
        let first_compute = Mutex::default();
        let resources = Resources::default()
            .with_compute_target(2)
            .with_io_target(16);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let start = Instant::now();
        pool.install(|| {
            drive_sync(
                read_compute_tasks(32, Duration::ZERO, &compute, &first_compute),
                &io,
                &resources,
            )
        })
        .unwrap();
        assert!(io.reads.peak() > 2, "reads peak {}", io.reads.peak());
        assert!(io.reads.peak() <= 16, "reads peak {}", io.reads.peak());
        // 32 reads of 20 ms on 2 threads would take at least 320 ms
        assert!(start.elapsed() < Duration::from_millis(300));
        assert!(io.pooled_reads.load(Ordering::SeqCst) > 16);
    }

    #[test]
    fn fast_reads_are_inline() {
        let io = LatencyIo::new(256, Duration::ZERO);
        let compute = Gauge::default();
        let first_compute = Mutex::default();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            drive_sync(
                read_compute_tasks(256, Duration::ZERO, &compute, &first_compute),
                &io,
                &Resources::default(),
            )
        })
        .unwrap();
        assert_eq!(io.reads.peak().max(1), io.reads.peak());
        assert_eq!(io.pooled_reads.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn writes_are_inline_while_fast() {
        let data: Vec<u8> = (0..=255).collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        let io = LatencyIo::new(0, Duration::ZERO);
        pool.install(|| {
            drive_sync(
                vec![super::super::tests::index_writes(256, &data)],
                &io,
                &Resources::default(),
            )
        })
        .unwrap();
        assert_eq!(io.inner.writes.load(Ordering::SeqCst), 256);
        // Most writes are inline (a write delayed by a busy machine can move a few to the pool)
        assert!(io.pooled_writes.load(Ordering::SeqCst) < 64);

        // Slow writes move to the I/O pool
        let io = LatencyIo::new(0, Duration::from_millis(1));
        pool.install(|| {
            drive_sync(
                vec![super::super::tests::index_writes(64, &data)],
                &io,
                &Resources::default(),
            )
        })
        .unwrap();
        assert_eq!(io.inner.writes.load(Ordering::SeqCst), 64);
        assert!(io.pooled_writes.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn single_task_runs_on_the_calling_thread() {
        let io = LatencyIo::new(1, Duration::ZERO);
        let caller = std::thread::current().id();
        let task = Task::read(
            ByteSource::external(SourceKey(0)),
            [ByteRange::FromStart(0, None)],
            move |_, _| {
                assert_eq!(std::thread::current().id(), caller);
                // A single child continues on the calling thread
                Ok(Outcome::Spawn(vec![Task::compute(move |_| {
                    assert_eq!(std::thread::current().id(), caller);
                    Ok(Outcome::Done)
                })]))
            },
        );
        drive_sync(vec![task], &io, &Resources::default()).unwrap();
        assert_eq!(*io.read_threads.lock().unwrap(), vec![caller]);
    }

    #[test]
    fn one_thread_pool_with_nested_drives_does_not_deadlock() {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        for dispatch in [ReadDispatch::Pooled, ReadDispatch::Inline] {
            let io = LatencyIo::new(8, Duration::from_millis(1));
            let resources = Resources::default().with_read_dispatch(Some(dispatch));
            let completed = AtomicUsize::new(0);
            let (io, completed, resources_ref) = (&io, &completed, &resources);
            let tasks = (0..8)
                .map(|key| {
                    Task::read(
                        ByteSource::external(SourceKey(key)),
                        [ByteRange::FromStart(0, None)],
                        move |_, ctx| {
                            // A nested drive inside a compute step
                            let nested = (0..4)
                                .map(|key| {
                                    Task::read(
                                        ByteSource::external(SourceKey(key)),
                                        [ByteRange::Suffix(1)],
                                        |_, _| {
                                            completed.fetch_add(1, Ordering::SeqCst);
                                            Ok(Outcome::Done)
                                        },
                                    )
                                })
                                .collect();
                            let nested_resources =
                                resources_ref.clone().with_compute(ctx.resources().clone());
                            drive_sync(nested, io, &nested_resources)?;
                            Ok(Outcome::Done)
                        },
                    )
                })
                .collect();
            pool.install(|| drive_sync(tasks, io, &resources)).unwrap();
            assert_eq!(completed.load(Ordering::SeqCst), 32);
        }
    }

    #[test]
    fn panics_resume_after_work_in_flight() {
        let io = LatencyIo::new(16, Duration::from_millis(1));
        let completed = AtomicUsize::new(0);
        let completed = &completed;
        let tasks = (0..16)
            .map(|key| {
                Task::read(
                    ByteSource::external(SourceKey(key)),
                    [ByteRange::FromStart(0, None)],
                    move |_, _| {
                        if key == 3 {
                            panic!("compute panicked");
                        }
                        completed.fetch_add(1, Ordering::SeqCst);
                        Ok(Outcome::Done)
                    },
                )
            })
            .collect();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            drive_sync(tasks, &io, &Resources::default())
        }));
        assert!(result.is_err());
        assert_eq!(io.reads.now.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn pooled_reads_are_coalesced() {
        crate::plan::driver::tests::coalesced_reads(
            drive_sync,
            &Resources::default()
                .with_compute_target(2)
                .with_io_target(1)
                .with_read_dispatch(Some(ReadDispatch::Pooled)),
        );
    }

    #[test]
    fn io_target_bounds_inline_requests_of_generated_tasks() {
        // Generated tasks read and write on compute threads
        let io = LatencyIo::new(16, Duration::from_millis(1));
        let generated = Task::compute(|_| {
            Ok(Outcome::Generate {
                children: crate::plan::Generator::indexed(64, |i| {
                    Ok(Some(Task::read(
                        ByteSource::external(SourceKey(i % 16)),
                        [ByteRange::FromStart(0, None)],
                        move |_, _| {
                            Ok(Outcome::Write {
                                ops: smallvec::smallvec![WriteOp::Set(
                                    SourceKey(16 + i),
                                    vec![1].into()
                                )],
                                then: TaskValue::None,
                            })
                        },
                    )))
                }),
                join: None,
            })
        });
        let resources = Resources::default()
            .with_compute_target(8)
            .with_io_target(2)
            .with_read_dispatch(Some(ReadDispatch::Inline));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();
        pool.install(|| drive_sync(vec![generated], &io, &resources))
            .unwrap();
        assert!(
            io.requests.peak() <= 2,
            "requests peak {}",
            io.requests.peak()
        );
        assert!(io.inner.read(SourceKey(16 + 63), &[]).unwrap().is_some());
    }

    #[test]
    fn io_limiter_is_shared_by_concurrent_drives() {
        let io = LatencyIo::new(24, Duration::from_millis(2));
        let resources = Resources::default()
            .with_io_target(16)
            .with_io_limiter(Some(crate::IoLimiter::new(3)));
        std::thread::scope(|scope| {
            for _ in 0..3 {
                scope.spawn(|| {
                    let compute = Gauge::default();
                    let first_compute = Mutex::default();
                    drive_sync(
                        read_compute_tasks(24, Duration::ZERO, &compute, &first_compute),
                        &io,
                        &resources,
                    )
                    .unwrap();
                });
            }
        });
        assert!(io.reads.peak() <= 3, "reads peak {}", io.reads.peak());
        assert!(io.reads.peak() > 1, "reads peak {}", io.reads.peak());
    }

    #[test]
    fn subchunk_style_fanout() {
        // A chunk task forks into many child reads of windows of the same key
        let io = MemoryIo::with_values([(0, (0..=255).collect())]);
        let task = Task::read(
            ByteSource::external(SourceKey(0)),
            [ByteRange::FromStart(0, Some(4))],
            |_, _| {
                let source = ByteSource::external(SourceKey(0));
                Ok(Outcome::Fork {
                    children: (0..64)
                        .map(|i| {
                            Task::read(
                                source.interval(i * 4, 4).unwrap(),
                                [ByteRange::Suffix(1)],
                                |mut reads, _| {
                                    Ok(Outcome::Value(TaskValue::Bytes(reads.pop().unwrap())))
                                },
                            )
                        })
                        .collect(),
                    join: Join::new(|values, _| {
                        let last: Vec<u8> = values
                            .into_iter()
                            .map(|value| match value {
                                TaskValue::Bytes(Some(bytes)) => bytes[0],
                                _ => panic!(),
                            })
                            .collect();
                        assert_eq!(last, (0..64).map(|i| i * 4 + 3).collect::<Vec<u8>>());
                        Ok(Outcome::Done)
                    }),
                })
            },
        );
        drive_sync(vec![task], &io, &Resources::default()).unwrap();
        assert_eq!(io.reads.load(Ordering::SeqCst), 65);
    }

    /// Counts reads that completed and whose compute step has not completed.
    struct BufferedIo {
        inner: MemoryIo,
        latency: Duration,
        buffered: Gauge,
    }

    impl SyncIo for BufferedIo {
        fn read(
            &self,
            key: SourceKey,
            ranges: &[ByteRange],
        ) -> Result<Option<ReadBytes>, StorageError> {
            std::thread::sleep(self.latency);
            let result = self.inner.read(key, ranges);
            self.buffered.enter();
            result
        }
    }

    fn buffered_tasks(io: &BufferedIo, num_keys: usize, compute: Duration) -> Vec<Task<'_>> {
        (0..num_keys)
            .map(|key| {
                Task::read(
                    ByteSource::external(SourceKey(key)),
                    [ByteRange::FromStart(0, None)],
                    move |_, _| {
                        std::thread::sleep(compute);
                        io.buffered.exit();
                        Ok(Outcome::Done)
                    },
                )
            })
            .collect()
    }

    #[test]
    fn fetched_bytes_limit_bounds_buffered_reads() {
        for dispatch in [ReadDispatch::Pooled, ReadDispatch::Inline] {
            let io = BufferedIo {
                inner: MemoryIo::with_values((0..32).map(|i| (i, vec![0; 1024]))),
                latency: Duration::from_millis(1),
                buffered: Gauge::default(),
            };
            let resources = Resources::default()
                .with_compute_target(2)
                .with_io_target(16)
                .with_fetched_bytes_limit(4 * 1024)
                .with_read_dispatch(Some(dispatch));
            drive_sync(
                buffered_tasks(&io, 32, Duration::from_millis(3)),
                &io,
                &resources,
            )
            .unwrap();
            let peak = io.buffered.peak();
            assert!(peak <= 4 && peak > 1, "{dispatch:?}: buffered peak {peak}");
        }
    }

    #[test]
    fn reads_larger_than_the_fetched_bytes_limit_progress() {
        let io = BufferedIo {
            inner: MemoryIo::with_values((0..8).map(|i| (i, vec![0; 1024]))),
            latency: Duration::from_millis(1),
            buffered: Gauge::default(),
        };
        let resources = Resources::default()
            .with_io_target(16)
            .with_fetched_bytes_limit(100)
            .with_read_dispatch(Some(ReadDispatch::Pooled));
        drive_sync(buffered_tasks(&io, 8, Duration::ZERO), &io, &resources).unwrap();
        assert_eq!(io.buffered.peak(), 1);
    }
}
