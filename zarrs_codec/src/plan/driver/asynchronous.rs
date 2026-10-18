//! The asynchronous driver.

// Work is moved by value rather than boxed, to avoid an allocation per task.
#![allow(clippy::result_large_err)]

use std::any::Any;
use std::panic::resume_unwind;

use futures::future::Either;
use futures::stream::FuturesUnordered;
use futures::{FutureExt, StreamExt};
use zarrs_storage::StorageError;

use super::super::{AsyncIo, ComputeCtx, Outcome, Task, TaskValue};
use super::coalesce::ReadGroup;
use super::core::{
    ByteBudget, Completed, NodeId, Scheduler, Work, WriteWork, external_key, finish_reads,
    generation_window, read_coalescing, single_read,
};
use crate::{CodecError, Resources};
use smallvec::SmallVec;

/// Compute steps that process at least this many bytes are offloaded to Rayon by [`drive_async_offload`].
#[cfg(not(target_arch = "wasm32"))]
pub const OFFLOAD_THRESHOLD: u64 = 8 * 1024;

#[cfg(not(target_arch = "wasm32"))]
type EventFuture<'f, 'a> = futures::future::BoxFuture<'f, Event<'a>>;
#[cfg(target_arch = "wasm32")]
type EventFuture<'f, 'a> = futures::future::LocalBoxFuture<'f, Event<'a>>;

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
enum Event<'a> {
    Computed(u64, Completed<'a>),
    Read(u64, SmallVec<[Result<Work<'a>, CodecError>; 1]>),
    Written(NodeId, Result<(), StorageError>),
    Panicked(Box<dyn Any + Send>),
}

/// Execute `tasks` asynchronously with up to [`Resources::io_target`] concurrent storage requests.
///
/// I/O and compute are interleaved: a task computes as soon as its reads complete.
/// Compute steps run on the polling task, so they may borrow from the caller (e.g. an output buffer).
/// Use [`drive_async_offload`] to run compute steps on Rayon.
/// Returns the values of `tasks`.
///
/// Every storage request holds a permit of the [`Resources::io_limiter`], if set.
/// Reads are not dispatched while the bytes read and not yet computed on would exceed the [`Resources::fetched_bytes_limit`].
///
/// Queued reads of nearby byte ranges of the same key are coalesced into one request as set by the
/// [`AsyncIo::read_coalescing`] hint (or [`Resources::read_coalescing`]), and the bytes of each read are slices of the
/// bytes of the request.
///
/// # Errors
/// Returns the first error of a read, write, or compute step.
/// Once an error occurs, no further work is started.
pub async fn drive_async<'a>(
    tasks: Vec<Task<'a>>,
    io: &dyn AsyncIo,
    resources: &Resources,
) -> Result<Vec<TaskValue<'a>>, CodecError> {
    drive(tasks, io, resources, |_| true, |work, _| Err(work)).await
}

/// Execute `task` as [`drive_async`], returning its value.
///
/// # Errors
/// Returns the first error of a read, write, or compute step.
pub async fn drive_async_one<'a>(
    task: Task<'a>,
    io: &dyn AsyncIo,
    resources: &Resources,
) -> Result<TaskValue<'a>, CodecError> {
    drive_one(task, io, resources, |_| true, |work, _| Err(work)).await
}

/// Execute `tasks` asynchronously, offloading compute steps to Rayon.
///
/// As [`drive_async`], except that compute steps that process at least [`OFFLOAD_THRESHOLD`] bytes
/// ([`TaskHint::compute_bytes`](super::super::TaskHint::compute_bytes)) run on Rayon with up to
/// [`Resources::compute_target`] concurrent compute steps, so they do not block the polling task.
/// Smaller compute steps run on the polling task, avoiding a thread hop.
///
/// Tasks must be `'static` because the drive can be cancelled (dropped) while compute steps are running.
///
/// On `wasm32`, where there is no Rayon, this is [`drive_async`].
///
/// # Errors
/// Returns the first error of a read, write, or compute step.
///
/// # Panics
/// Resumes the first panic of an offloaded compute step once the work in flight completes.
pub async fn drive_async_offload(
    tasks: Vec<Task<'static>>,
    io: &dyn AsyncIo,
    resources: &Resources,
) -> Result<Vec<TaskValue<'static>>, CodecError> {
    #[cfg(not(target_arch = "wasm32"))]
    return drive(tasks, io, resources, runs_inline, offload).await;
    #[cfg(target_arch = "wasm32")]
    return drive_async(tasks, io, resources).await;
}

/// Execute `task` as [`drive_async_offload`], returning its value.
///
/// # Errors
/// Returns the first error of a read, write, or compute step.
///
/// # Panics
/// Resumes the first panic of an offloaded compute step once the work in flight completes.
pub async fn drive_async_offload_one(
    task: Task<'static>,
    io: &dyn AsyncIo,
    resources: &Resources,
) -> Result<TaskValue<'static>, CodecError> {
    #[cfg(not(target_arch = "wasm32"))]
    return drive_one(task, io, resources, runs_inline, offload).await;
    #[cfg(target_arch = "wasm32")]
    return drive_async_one(task, io, resources).await;
}

/// Returns true if the compute step of `task` is small enough to run on the polling task (see [`offload`]).
#[cfg(not(target_arch = "wasm32"))]
fn runs_inline(task: &Task<'_>) -> bool {
    task.hint().compute_bytes < OFFLOAD_THRESHOLD || polled_on_rayon()
}

/// Returns true if the drive is polled on a Rayon thread (e.g. `block_on` in a Rayon job).
///
/// Its compute steps then run on the polling thread: offloading them to the same pool could deadlock if every thread
/// of the pool is polling, and running them inline does not block an async executor.
#[cfg(not(target_arch = "wasm32"))]
fn polled_on_rayon() -> bool {
    rayon::current_thread_index().is_some()
}

/// Run `work` on Rayon if it is large enough.
#[cfg(not(target_arch = "wasm32"))]
fn offload(
    work: Work<'static>,
    ctx: &dyn Fn() -> ComputeCtx,
) -> Result<EventFuture<'static, 'static>, Work<'static>> {
    // Jobs of generators whose tasks compute (rather than read) plan and run several tasks
    if (!work.is_computing_range() && work.hint().compute_bytes < OFFLOAD_THRESHOLD)
        || polled_on_rayon()
    {
        return Err(work);
    }
    let ctx = ctx();
    let (sender, receiver) = futures::channel::oneshot::channel();
    let fetched = work.fetched();
    rayon::spawn(move || {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work.run(&ctx, None, None)));
        // The drive may have been cancelled
        let _ = sender.send(result);
    });
    Ok(async move {
        match receiver.await {
            Ok(Ok(completed)) => Event::Computed(fetched, completed),
            Ok(Err(payload)) => Event::Panicked(payload),
            Err(futures::channel::oneshot::Canceled) => {
                unreachable!("the compute step always sends its result")
            }
        }
    }
    .boxed())
}

/// Drive `tasks`, running work on the polling task unless `offload` takes it.
///
/// A single root task runs on the polling task without a scheduler if `runs_inline` admits it.
async fn drive<'a, 'f>(
    mut tasks: Vec<Task<'a>>,
    io: &'f dyn AsyncIo,
    resources: &'f Resources,
    runs_inline: impl Fn(&Task<'a>) -> bool,
    offload: impl Fn(Work<'a>, &dyn Fn() -> ComputeCtx) -> Result<EventFuture<'a, 'a>, Work<'a>>,
) -> Result<Vec<TaskValue<'a>>, CodecError>
where
    'a: 'f,
{
    let scheduler = if let [task] = tasks.as_slice()
        && runs_inline(task)
    {
        let task = tasks.pop().expect("one task");
        match drive_single(task, io, resources).await? {
            Ok(value) => return Ok(vec![value]),
            Err(scheduler) => scheduler,
        }
    } else {
        Scheduler::new(tasks)
    };
    drive_scheduler(scheduler, io, resources, offload).await
}

/// Drive `task` as [`drive`], returning its value.
async fn drive_one<'a, 'f>(
    task: Task<'a>,
    io: &'f dyn AsyncIo,
    resources: &'f Resources,
    runs_inline: impl Fn(&Task<'a>) -> bool,
    offload: impl Fn(Work<'a>, &dyn Fn() -> ComputeCtx) -> Result<EventFuture<'a, 'a>, Work<'a>>,
) -> Result<TaskValue<'a>, CodecError>
where
    'a: 'f,
{
    let scheduler = if runs_inline(&task) {
        match drive_single(task, io, resources).await? {
            Ok(value) => return Ok(value),
            Err(scheduler) => scheduler,
        }
    } else {
        Scheduler::new(vec![task])
    };
    Ok(drive_scheduler(scheduler, io, resources, offload)
        .await?
        .pop()
        .expect("one root value"))
}

/// Run a single root task on the polling task, returning its value, or a scheduler to continue it if it fans out.
async fn drive_single<'a>(
    task: Task<'a>,
    io: &dyn AsyncIo,
    resources: &Resources,
) -> Result<Result<TaskValue<'a>, Scheduler<'a>>, CodecError> {
    let ctx = ComputeCtx::new_inline(resources.compute().clone());
    let outcome = if task.needs_io() {
        let (source, ranges, step) = single_read(task)?;
        let result = {
            let _permit = resources.io_permit().await;
            io.read(external_key(&source), &ranges).await
        };
        step.run(finish_reads(&source, &ranges, result)?, &ctx)?
    } else {
        task.run_without_io(&ctx)?
    };
    match outcome {
        Outcome::Done => Ok(Ok(TaskValue::None)),
        Outcome::Value(value) => Ok(Ok(value)),
        Outcome::Write { mut ops, then } if ops.len() == 1 => {
            let _permit = resources.io_permit().await;
            io.write(ops.pop().expect("one write")).await?;
            Ok(Ok(then))
        }
        outcome => Ok(Err(Scheduler::from_root_outcome(outcome))),
    }
}

#[allow(clippy::too_many_lines)]
async fn drive_scheduler<'a, 'f>(
    mut scheduler: Scheduler<'a>,
    io: &'f dyn AsyncIo,
    resources: &'f Resources,
    offload: impl Fn(Work<'a>, &dyn Fn() -> ComputeCtx) -> Result<EventFuture<'a, 'a>, Work<'a>>,
) -> Result<Vec<TaskValue<'a>>, CodecError>
where
    'a: 'f,
{
    let compute_limit = resources.compute_limit();
    let io_limit = resources.io_limit();
    scheduler.set_parallelism(compute_limit.min(1024));
    scheduler.set_max_deferred(io_limit.min(1024));
    // Storage requests are not boxed: their futures are a concrete type
    let mut io_futures = FuturesUnordered::new();
    let mut compute_futures: FuturesUnordered<EventFuture<'a, 'a>> = FuturesUnordered::new();
    let mut budget = ByteBudget::new(resources.fetched_bytes_limit());
    let mut panic = None;
    let window = generation_window(resources);
    let coalescing = read_coalescing(resources, io.read_coalescing());
    loop {
        let mut inline = None;
        scheduler.refill(window);
        if !scheduler.failed() {
            while io_futures.len() < io_limit {
                if let Some(write) = scheduler.pop_write() {
                    io_futures.push(Either::Right(write_future(io, resources, write)));
                    continue;
                }
                // Queued reads of the same key are coalesced into one request
                let Some((group, reserved)) = scheduler.pop_read_group(coalescing, &mut budget)
                else {
                    break;
                };
                match group {
                    Ok(group) => {
                        io_futures.push(Either::Left(read_future(io, resources, group, reserved)));
                    }
                    Err(err) => {
                        budget.cancel(reserved);
                        scheduler.fail(err);
                        break;
                    }
                }
            }
            while compute_futures.len() < compute_limit {
                let Some(work) = scheduler.pop_compute() else {
                    break;
                };
                let running = compute_futures.len() + 1;
                let queued = scheduler.compute_queued();
                let offload_ctx =
                    || ComputeCtx::new(resources.compute().with_compute_share(running + queued));
                match offload(work, &offload_ctx) {
                    Ok(future) => compute_futures.push(future),
                    Err(work) => {
                        // Queued work does not run alongside work on the polling task
                        inline = Some((
                            work,
                            ComputeCtx::new_inline(resources.compute().with_compute_share(running)),
                        ));
                        break;
                    }
                }
            }
        }

        // Handle events that are ready without waiting
        // (Polling an empty stream would register a waker for nothing)
        let mut handled = false;
        while !io_futures.is_empty()
            && let Some(Some(event)) = io_futures.next().now_or_never()
        {
            handle(&mut scheduler, &mut budget, &mut panic, event);
            handled = true;
        }
        while !compute_futures.is_empty()
            && let Some(Some(event)) = compute_futures.next().now_or_never()
        {
            handle(&mut scheduler, &mut budget, &mut panic, event);
            handled = true;
        }

        // Run a compute step on the polling task
        if let Some((mut work, ctx)) = inline {
            // A job of a generator runs on the polling task until it computes, then later jobs can be offloaded
            work.run_briefly();
            let fetched = work.fetched();
            let completed = work.run(&ctx, None, None);
            budget.release(fetched);
            scheduler.complete_work(completed);
            continue;
        }
        if handled {
            // Dispatch the work the events made ready (and into the requests and compute steps they freed)
            continue;
        }

        let event = match (io_futures.is_empty(), compute_futures.is_empty()) {
            (true, true) => {
                // Release generators whose jobs have all been recorded, which may queue join steps
                scheduler.refill(window);
                if scheduler.failed() || scheduler.queued() == 0 {
                    break;
                }
                continue;
            }
            (false, true) => io_futures.next().await,
            (true, false) => compute_futures.next().await,
            (false, false) => {
                match futures::future::select(io_futures.next(), compute_futures.next()).await {
                    Either::Left((event, _)) | Either::Right((event, _)) => event,
                }
            }
        };
        handle(
            &mut scheduler,
            &mut budget,
            &mut panic,
            event.expect("a nonempty stream yields an event"),
        );
    }
    if let Some(panic) = panic {
        resume_unwind(panic);
    }
    scheduler.finish()
}

fn handle<'a>(
    scheduler: &mut Scheduler<'a>,
    budget: &mut ByteBudget,
    panic: &mut Option<Box<dyn Any + Send>>,
    event: Event<'a>,
) {
    match event {
        Event::Computed(fetched, completed) => {
            budget.release(fetched);
            scheduler.complete_work(completed);
        }
        Event::Read(reserved, results) => {
            scheduler.complete_read_group(budget, reserved, results);
        }
        Event::Written(node, result) => scheduler.complete_write(node, result),
        Event::Panicked(payload) => {
            panic.get_or_insert(payload);
            scheduler.fail(CodecError::Other("a task panicked".to_string()));
        }
    }
}

async fn read_future<'f, 'a: 'f>(
    io: &'f dyn AsyncIo,
    resources: &'f Resources,
    group: ReadGroup<'a>,
    reserved: u64,
) -> Event<'a> {
    let result = {
        let _permit = resources.io_permit().await;
        let request = group.request();
        io.read(request.key, &request.ranges).await
    };
    Event::Read(reserved, group.complete(result))
}

async fn write_future<'f, 'a: 'f>(
    io: &'f dyn AsyncIo,
    resources: &'f Resources,
    write: WriteWork<'a>,
) -> Event<'a> {
    let WriteWork { node, op } = write;
    let _permit = resources.io_permit().await;
    Event::Written(node, io.write(op).await)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crate::plan::Outcome;
    use zarrs_storage::byte_range::ByteRange;

    use super::super::super::{ByteSource, ReadBytes, SourceKey, SyncIo, TaskHint, WriteOp};
    use super::super::tests::MemoryIo;
    use super::*;

    /// An asynchronous view of a synchronous backend, with optional latency.
    struct AsyncAdapter<'i> {
        io: &'i dyn SyncIo,
        latency: Duration,
        now: AtomicUsize,
        peak: AtomicUsize,
    }

    impl<'i> AsyncAdapter<'i> {
        fn new(io: &'i dyn SyncIo, latency: Duration) -> Self {
            Self {
                io,
                latency,
                now: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl AsyncIo for AsyncAdapter<'_> {
        async fn read(
            &self,
            key: SourceKey,
            ranges: &[ByteRange],
        ) -> Result<Option<ReadBytes>, StorageError> {
            let now = self.now.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(now, Ordering::SeqCst);
            if !self.latency.is_zero() {
                tokio::time::sleep(self.latency).await;
            }
            self.now.fetch_sub(1, Ordering::SeqCst);
            self.io.read(key, ranges)
        }

        async fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
            self.io.write(op)
        }

        fn read_coalescing(&self) -> Option<zarrs_storage::ReadCoalescing> {
            self.io.read_coalescing()
        }
    }

    fn drive_async_blocking<'a>(
        tasks: Vec<Task<'a>>,
        io: &dyn SyncIo,
        resources: &Resources,
    ) -> Result<Vec<TaskValue<'a>>, CodecError> {
        futures::executor::block_on(drive_async(
            tasks,
            &AsyncAdapter::new(io, Duration::ZERO),
            resources,
        ))
    }

    mod semantics {
        crate::plan::driver::tests::driver_tests!(super::drive_async_blocking);
    }

    fn read_tasks<'a>(
        num_keys: usize,
        compute_bytes: u64,
        f: impl Fn() + Copy + Send + 'a,
    ) -> Vec<Task<'a>> {
        (0..num_keys)
            .map(|key| {
                Task::read(
                    ByteSource::external(SourceKey(key)),
                    [ByteRange::FromStart(0, None)],
                    move |reads, _| {
                        assert!(reads[0].is_some());
                        f();
                        Ok(Outcome::Done)
                    },
                )
                .with_hint(TaskHint {
                    read_bytes: None,
                    compute_bytes,
                })
            })
            .collect()
    }

    #[test]
    fn reads_are_coalesced() {
        crate::plan::driver::tests::coalesced_reads(
            drive_async_blocking,
            &Resources::default().with_io_target(1),
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn io_target_limits_concurrent_reads() {
        let io = MemoryIo::with_values((0..32).map(|i| (i, vec![1])));
        let io = AsyncAdapter::new(&io, Duration::from_millis(2));
        let resources = Resources::default().with_io_target(5);
        drive_async(read_tasks(32, 0, || {}), &io, &resources)
            .await
            .unwrap();
        let peak = io.peak.load(Ordering::SeqCst);
        assert!(peak <= 5 && peak > 1, "reads peak {peak}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn large_compute_steps_are_offloaded() {
        let io = MemoryIo::with_values((0..4).map(|i| (i, vec![1])));
        let io = AsyncAdapter::new(&io, Duration::ZERO);
        let on_rayon = || assert!(rayon::current_thread_index().is_some());
        let off_rayon = || assert!(rayon::current_thread_index().is_none());
        drive_async_offload(
            read_tasks(4, OFFLOAD_THRESHOLD, on_rayon),
            &io,
            &Resources::default(),
        )
        .await
        .unwrap();
        drive_async_offload(read_tasks(4, 0, off_rayon), &io, &Resources::default())
            .await
            .unwrap();
        // Including a single task
        for offloaded in [true, false] {
            let compute_bytes = if offloaded { OFFLOAD_THRESHOLD } else { 0 };
            let check = move || assert_eq!(rayon::current_thread_index().is_some(), offloaded);
            let read = read_tasks(1, compute_bytes, check).pop().unwrap();
            drive_async_offload_one(read, &io, &Resources::default())
                .await
                .unwrap();
            let compute = Task::compute(move |_| {
                check();
                Ok(Outcome::Done)
            })
            .with_hint(TaskHint {
                read_bytes: None,
                compute_bytes,
            });
            drive_async_offload_one(compute, &io, &Resources::default())
                .await
                .unwrap();
        }
        // Borrowing drives never offload
        drive_async(
            read_tasks(4, OFFLOAD_THRESHOLD, off_rayon),
            &io,
            &Resources::default(),
        )
        .await
        .unwrap();
    }

    #[test]
    fn drives_polled_on_rayon_do_not_offload() {
        // Offloading to a pool whose only thread is polling the drive would deadlock
        let io = MemoryIo::with_values((0..4).map(|i| (i, vec![1])));
        let io = AsyncAdapter::new(&io, Duration::ZERO);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        pool.install(|| {
            futures::executor::block_on(async {
                let read = read_tasks(1, OFFLOAD_THRESHOLD, || {}).pop().unwrap();
                drive_async_offload_one(read, &io, &Resources::default())
                    .await
                    .unwrap();
                drive_async_offload(
                    read_tasks(4, OFFLOAD_THRESHOLD, || {}),
                    &io,
                    &Resources::default(),
                )
                .await
                .unwrap();
            });
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_a_drive_is_safe() {
        static COMPLETED: AtomicUsize = AtomicUsize::new(0);
        let io = MemoryIo::with_values((0..64).map(|i| (i, vec![1])));
        let io = AsyncAdapter::new(&io, Duration::from_millis(5));
        let tasks = read_tasks(64, OFFLOAD_THRESHOLD, || {
            std::thread::sleep(Duration::from_millis(5));
            COMPLETED.fetch_add(1, Ordering::SeqCst);
        });
        let resources = Resources::default();
        let drive = drive_async_offload(tasks, &io, &resources);
        assert!(
            tokio::time::timeout(Duration::from_millis(15), drive)
                .await
                .is_err()
        );
        // Offloaded compute steps that were running complete harmlessly
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(COMPLETED.load(Ordering::SeqCst) < 64);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn offloaded_panics_resume() {
        let tasks = read_tasks(1, OFFLOAD_THRESHOLD, || panic!("compute panicked"));
        let drive = tokio::spawn(async move {
            let io = MemoryIo::with_values([(0, vec![1])]);
            let io = AsyncAdapter::new(&io, Duration::ZERO);
            drive_async_offload(tasks, &io, &Resources::default()).await
        });
        assert!(drive.await.unwrap_err().is_panic());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn io_limiter_is_shared_by_sync_and_async_drives() {
        let memory = MemoryIo::with_values((0..32).map(|i| (i, vec![1])));
        let limiter = crate::IoLimiter::new(2);
        let resources = Resources::default()
            .with_io_target(16)
            .with_io_limiter(Some(limiter));
        let io = AsyncAdapter::new(&memory, Duration::from_millis(2));
        // The synchronous drive reads through the same counters
        struct SyncView<'v, 'i>(&'v AsyncAdapter<'i>);
        impl SyncIo for SyncView<'_, '_> {
            fn read(
                &self,
                key: SourceKey,
                ranges: &[ByteRange],
            ) -> Result<Option<ReadBytes>, StorageError> {
                let now = self.0.now.fetch_add(1, Ordering::SeqCst) + 1;
                self.0.peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(self.0.latency);
                self.0.now.fetch_sub(1, Ordering::SeqCst);
                self.0.io.read(key, ranges)
            }
        }
        let sync_io = SyncView(&io);
        let sync_resources = resources.clone();
        let (sync_io, sync_resources) = (&sync_io, &sync_resources);
        std::thread::scope(|scope| {
            scope.spawn(move || {
                super::super::drive_sync(read_tasks(32, 0, || {}), sync_io, sync_resources)
                    .unwrap();
            });
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(drive_async(
                    read_tasks(32, 0, || {}),
                    &io,
                    &resources,
                ))
            })
            .unwrap();
        });
        let peak = io.peak.load(Ordering::SeqCst);
        assert!(peak <= 2, "reads peak {peak}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fetched_bytes_limit_bounds_buffered_reads() {
        static BUFFERED: AtomicUsize = AtomicUsize::new(0);
        static PEAK: AtomicUsize = AtomicUsize::new(0);
        struct BufferedIo(MemoryIo);
        #[async_trait::async_trait]
        impl AsyncIo for BufferedIo {
            async fn read(
                &self,
                key: SourceKey,
                ranges: &[ByteRange],
            ) -> Result<Option<ReadBytes>, StorageError> {
                tokio::time::sleep(Duration::from_millis(1)).await;
                let now = BUFFERED.fetch_add(1, Ordering::SeqCst) + 1;
                PEAK.fetch_max(now, Ordering::SeqCst);
                self.0.read(key, ranges)
            }
        }
        let io = BufferedIo(MemoryIo::with_values((0..32).map(|i| (i, vec![0; 1024]))));
        let resources = Resources::default()
            .with_io_target(16)
            .with_fetched_bytes_limit(4 * 1024);
        let tasks = read_tasks(32, OFFLOAD_THRESHOLD, || {
            std::thread::sleep(Duration::from_millis(2));
            BUFFERED.fetch_sub(1, Ordering::SeqCst);
        });
        drive_async_offload(tasks, &io, &resources).await.unwrap();
        let peak = PEAK.load(Ordering::SeqCst);
        assert!(peak <= 4 && peak > 1, "buffered peak {peak}");
    }
}
