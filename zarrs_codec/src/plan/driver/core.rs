//! The scheduling state shared by the drivers.
//!
//! A drive is a forest of task nodes. A node runs its compute step (after its reads, if any), and
//! the [`Outcome`] may add child nodes or writes. A node completes once its compute step and all
//! of its children and writes have completed. A forked node then runs its join step, which
//! produces a further outcome for the node.
//!
//! The scheduler only tracks state. Drivers decide when and where work runs.

use std::collections::VecDeque;

use smallvec::SmallVec;
use zarrs_storage::byte_range::ByteRange;
use zarrs_storage::{ReadCoalescing, StorageError};

use super::super::io::ReadBytes;
use super::coalesce::{self, ReadGroup};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::super::task::{GeneratorKind, PlanFn};
use super::super::{
    ByteSource, ComputeCtx, ComputeStep, Join, Outcome, Reads, SourceKey, SourceRoot, Task,
    TaskHint, TaskValue, WriteOp, read_memory,
};
use crate::{CodecError, CowBytes};

pub(crate) type NodeId = usize;

/// Where a node delivers its value.
#[derive(Debug, Clone, Copy)]
enum Parent {
    /// A root task, at the given position of the drive's tasks.
    Root(usize),
    /// A child of a fork, at the given position of the fork's children.
    Join(NodeId, usize),
    /// A spawned child, whose value is discarded.
    Spawn(NodeId),
}

struct Node<'a> {
    parent: Parent,
    depth: usize,
    /// Whether the compute or join step of the node is queued or running.
    computing: bool,
    /// The number of incomplete children and writes.
    pending: usize,
    join: Option<Join<'a>>,
    values: Vec<TaskValue<'a>>,
    value: TaskValue<'a>,
}

/// A compute step ready to run.
pub(crate) enum Work<'a> {
    /// The compute step of a task.
    Compute {
        node: NodeId,
        source: Option<ByteSource<'a>>,
        ranges: SmallVec<[ByteRange; 1]>,
        /// The bytes read, or [`None`] if the reads are from memory (and resolved when the work runs).
        reads: Option<Reads<'a>>,
        /// The number of bytes read from storage.
        #[cfg_attr(all(target_family = "wasm", not(feature = "async")), allow(dead_code))]
        fetched: u64,
        step: ComputeStep<'a>,
        #[cfg_attr(any(not(feature = "async"), target_arch = "wasm32"), allow(dead_code))]
        hint: TaskHint,
    },
    /// The join step of a fork.
    Join {
        node: NodeId,
        join: Join<'a>,
        values: Vec<TaskValue<'a>>,
    },
    /// Plan and run the tasks of a range of indices of an indexed generator.
    Range(RangeWork<'a>),
}

/// A job that plans and runs tasks of indexed generators.
///
/// Jobs claim blocks of indices from a shared cursor, so they balance load without returning to
/// the driver for each task, and take tasks from other generators when theirs are exhausted.
pub(crate) struct RangeWork<'a> {
    state: Arc<IndexedState<'a>>,
    registry: Arc<Registry<'a>>,
    /// The maximum number of tasks returned to the driver before the job completes.
    max_deferred: usize,
    /// The number of compute steps that run in parallel, which sets the block size of generators.
    parallelism: usize,
    /// Whether the job stops after a block of indices in which it runs a task (see [`Work::run_briefly`]).
    brief: bool,
}

/// The number of indices a job claims from the cursor of a generator of `len` tasks at a time:
/// small enough that each of `parallelism` jobs claims several, and large enough that jobs rarely contend.
fn block_size(len: usize, parallelism: usize) -> usize {
    (len / (16 * parallelism)).clamp(1, 64)
}

/// An indexed generator, shared by the scheduler and the jobs that run its tasks.
pub(crate) struct IndexedState<'a> {
    /// The node of the generator, set by the scheduler.
    node: AtomicUsize,
    /// The next index to plan.
    cursor: AtomicUsize,
    /// The number of jobs that have worked on the generator and whose results the scheduler has not recorded.
    ///
    /// The generator completes once it is exhausted and this is zero.
    unreported: AtomicUsize,
    /// The number of jobs queued for the generator that have not started.
    queued: AtomicUsize,
    len: usize,
    plan: PlanFn<'a>,
    /// Whether the values of the tasks are collected by a join step of the node.
    collect: bool,
    /// The number of indices claimed from the cursor at a time.
    block: usize,
    /// Whether a job has run a task of the generator (rather than returning it for a read).
    runs_tasks: std::sync::atomic::AtomicBool,
}

impl<'a> IndexedState<'a> {
    fn new(len: usize, plan: PlanFn<'a>, collect: bool, block: usize, unreported: usize) -> Self {
        Self {
            node: AtomicUsize::new(usize::MAX),
            cursor: AtomicUsize::new(0),
            unreported: AtomicUsize::new(unreported),
            queued: AtomicUsize::new(0),
            len,
            plan,
            collect,
            block,
            runs_tasks: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn node(&self) -> NodeId {
        let node = self.node.load(Ordering::Acquire);
        debug_assert_ne!(node, usize::MAX, "the scheduler sets the node first");
        node
    }

    /// Register a job with the generator if it has tasks left.
    fn try_join(&self) -> bool {
        // Registering before checking the cursor ensures the scheduler cannot complete the generator in between
        self.unreported.fetch_add(1, Ordering::SeqCst);
        if self.cursor.load(Ordering::SeqCst) < self.len {
            true
        } else {
            self.unreported.fetch_sub(1, Ordering::SeqCst);
            false
        }
    }

    /// Returns true if the generator has no tasks left, and all jobs that worked on it have been recorded.
    fn is_complete(&self) -> bool {
        self.cursor.load(Ordering::SeqCst) >= self.len
            && self.unreported.load(Ordering::SeqCst) == 0
    }
}

/// The indexed generators of a drive, which jobs take tasks from when theirs are exhausted.
pub(crate) type Registry<'a> = Mutex<Vec<Arc<IndexedState<'a>>>>;

/// An indexed generator that a job created (as the outcome of a task) and continues to run.
///
/// The job reports it to the scheduler while it runs, so the scheduler can add jobs to it straight away.
#[cfg_attr(target_family = "wasm", allow(dead_code))]
pub(crate) struct Joined<'a> {
    /// The generator of the task.
    parent: Arc<IndexedState<'a>>,
    /// The index of the task in its generator.
    index: usize,
    join: Option<Join<'a>>,
    /// The generator, with the job that created it registered.
    state: Arc<IndexedState<'a>>,
}

/// A function that reports a [`Joined`] generator to the scheduler.
#[cfg(not(target_family = "wasm"))]
pub(crate) type OnJoin<'r, 'a> = dyn Fn(Joined<'a>) + Sync + 'r;
/// A function that reports a [`Joined`] generator to the scheduler.
#[cfg(target_family = "wasm")]
pub(crate) type OnJoin<'r, 'a> = dyn Fn(Joined<'a>) + 'r;

/// Storage requests performed on the thread running compute work.
pub(crate) trait InlineIo: zarrs_plugin::MaybeSync {
    /// Admit a storage request performed inline, if storage requests should currently be performed inline
    /// and the I/O target of the drive admits another.
    ///
    /// The request is counted towards the I/O target until the slot is dropped.
    fn admit(&self) -> Option<IoSlot<'_>>;

    /// Read the byte ranges of `request`.
    fn read(&self, request: &IoRequest) -> Result<Option<ReadBytes>, StorageError>;

    /// Perform a write operation.
    fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError>;
}

/// A storage request admitted by [`InlineIo::admit`], counted in `in_flight` (if any) until dropped.
pub(crate) struct IoSlot<'s>(pub(crate) Option<&'s AtomicUsize>);

impl Drop for IoSlot<'_> {
    fn drop(&mut self) {
        if let Some(in_flight) = self.0 {
            in_flight.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Perform the writes of a [`Outcome::Write`] with `inline` if it admits an inline storage request,
/// completing with its value.
fn write_inline<'a>(
    outcome: Outcome<'a>,
    inline: Option<&dyn InlineIo>,
) -> Result<Outcome<'a>, CodecError> {
    match outcome {
        Outcome::Write { ops, then } => {
            let Some(_slot) = inline.and_then(InlineIo::admit) else {
                return Ok(Outcome::Write { ops, then });
            };
            for op in ops {
                inline.expect("admitted").write(op)?;
            }
            Ok(Outcome::Value(then))
        }
        outcome => Ok(outcome),
    }
}

/// The result of running [`Work`].
pub(crate) enum Completed<'a> {
    /// A compute or join step of a node completed.
    Step(NodeId, Result<Outcome<'a>, CodecError>),
    /// A job of indexed generators completed, with the tasks it ran of each generator.
    Range(Result<RangeResults<'a>, CodecError>),
}

/// The tasks a job ran of each generator it worked on.
pub(crate) type RangeResults<'a> = SmallVec<[(Arc<IndexedState<'a>>, RangeOutput<'a>); 1]>;

/// The tasks run by a [`RangeWork`].
pub(crate) struct RangeOutput<'a> {
    collect: bool,
    /// The values of tasks that completed, if collected.
    values: Vec<(usize, TaskValue<'a>)>,
    /// Tasks whose compute step produced more work (e.g. children or writes).
    outcomes: Vec<(usize, Outcome<'a>)>,
    /// Tasks that need a read that could not be performed while running the range.
    tasks: Vec<(usize, Task<'a>)>,
}

/// The generator a job is working on, and the tasks it ran.
struct Current<'a> {
    state: Arc<IndexedState<'a>>,
    output: RangeOutput<'a>,
}

impl<'a> Current<'a> {
    fn new(state: Arc<IndexedState<'a>>) -> Self {
        let collect = state.collect;
        Self {
            state,
            output: RangeOutput {
                collect,
                values: Vec::new(),
                outcomes: Vec::new(),
                tasks: Vec::new(),
            },
        }
    }

    fn finish(self) -> (Arc<IndexedState<'a>>, RangeOutput<'a>) {
        (self.state, self.output)
    }
}

impl<'a> RangeWork<'a> {
    /// Claim, plan and run tasks of indexed generators, reading with `inline` if given.
    ///
    /// Tasks that need a read are returned to the scheduler if there is no `inline` read.
    /// A task that produces an indexed generator is reported with `on_join` (if given), and the
    /// job works on that generator once it finishes its block of indices, so its tasks start
    /// without waiting for the scheduler.
    /// When its generator is exhausted, the job takes tasks from other generators of the drive,
    /// the most recently created first.
    /// The job completes when no generator has tasks left, or after the block of indices in which a
    /// task produces other work (so it is scheduled depth-first) or `max_deferred` tasks need a read.
    fn run(
        self,
        ctx: &ComputeCtx,
        inline: Option<&dyn InlineIo>,
        on_join: Option<&OnJoin<'_, 'a>>,
    ) -> Result<RangeResults<'a>, CodecError> {
        self.state.queued.fetch_sub(1, Ordering::SeqCst);
        let mut results = RangeResults::new();
        let mut current = Current::new(self.state);
        let mut next: Option<Current<'a>> = None;
        let mut indices = 0..0;
        let mut ran = false;
        loop {
            let index = if let Some(index) = indices.next() {
                index
            } else {
                // Finish claimed blocks, then move to a joined generator, or stop once there is work for the driver
                if let Some(joined) = next.take() {
                    results.push(std::mem::replace(&mut current, joined).finish());
                }
                if !current.output.outcomes.is_empty()
                    || current.output.tasks.len() >= self.max_deferred
                    || (self.brief && ran)
                {
                    break;
                }
                // Claim indices in blocks, so jobs rarely contend for the cursor
                let state = &current.state;
                let start = state.cursor.fetch_add(state.block, Ordering::SeqCst);
                if start >= state.len {
                    let Some(other) = steal(&self.registry, state) else {
                        break;
                    };
                    results.push(std::mem::replace(&mut current, Current::new(other)).finish());
                    continue;
                }
                indices = start..(start + state.block).min(state.len);
                indices.next().expect("a nonempty block")
            };
            let outcome = match run_indexed(&current.state.plan, index, ctx, inline) {
                Ok(Some(IndexedRun::Outcome(outcome))) => {
                    if !ran {
                        ran = true;
                        current.state.runs_tasks.store(true, Ordering::Relaxed);
                    }
                    outcome
                }
                Ok(Some(IndexedRun::Deferred(task))) => {
                    current.output.tasks.push((index, task));
                    continue;
                }
                Ok(None) => continue,
                Err(err) => return Err(err),
            };
            match outcome {
                Outcome::Done => {}
                Outcome::Value(value) => {
                    if current.state.collect {
                        current.output.values.push((index, value));
                    }
                }
                Outcome::Generate {
                    children: super::super::Generator(GeneratorKind::Indexed { len, plan }),
                    join,
                } if next.is_none() && on_join.is_some() => {
                    let on_join = on_join.expect("checked");
                    let collect = join.as_ref().is_some_and(Join::collects);
                    let block = block_size(len, self.parallelism);
                    let state = Arc::new(IndexedState::new(len, plan, collect, block, 1));
                    on_join(Joined {
                        parent: current.state.clone(),
                        index,
                        join,
                        state: state.clone(),
                    });
                    // Other jobs can take its tasks straight away. The scheduler receives the
                    // generator before the results of any job that takes its tasks.
                    self.registry
                        .lock()
                        .expect("the registry lock is not poisoned")
                        .push(state.clone());
                    next = Some(Current::new(state));
                }
                outcome @ (Outcome::Spawn(_)
                | Outcome::Fork { .. }
                | Outcome::Generate { .. }
                | Outcome::Write { .. }) => current.output.outcomes.push((index, outcome)),
            }
        }
        results.push(current.finish());
        Ok(results)
    }
}

/// Register a job with a generator of `registry` other than `exhausted` that has tasks left, the most recently created first.
fn steal<'a>(
    registry: &Registry<'a>,
    exhausted: &Arc<IndexedState<'a>>,
) -> Option<Arc<IndexedState<'a>>> {
    let registry = registry.lock().expect("the registry lock is not poisoned");
    registry
        .iter()
        .rev()
        .find(|state| !Arc::ptr_eq(state, exhausted) && state.try_join())
        .cloned()
}

/// The result of planning and running the task at an index of a generator.
enum IndexedRun<'a> {
    /// The task ran, with this outcome.
    Outcome(Outcome<'a>),
    /// The task needs a read that cannot be performed inline.
    Deferred(Task<'a>),
}

/// Plan the task at `index` with `plan`, and run it if possible.
fn run_indexed<'a>(
    plan: &PlanFn<'a>,
    index: usize,
    ctx: &ComputeCtx,
    inline: Option<&dyn InlineIo>,
) -> Result<Option<IndexedRun<'a>>, CodecError> {
    let Some(task) = plan(index)? else {
        return Ok(None);
    };
    let outcome = if !task.needs_io() {
        task.run_without_io(ctx)?
    } else if let Some((inline, slot)) = inline.and_then(|inline| Some((inline, inline.admit()?))) {
        match SingleTask::new(task)? {
            SingleTask::Read(task, request) => {
                let result = inline.read(&request);
                drop(slot);
                task.run(&request, result, ctx)?
            }
            SingleTask::Compute(task) => task.run_without_io(ctx)?,
        }
    } else {
        return Ok(Some(IndexedRun::Deferred(task)));
    };
    Ok(Some(IndexedRun::Outcome(write_inline(outcome, inline)?)))
}

impl<'a> Work<'a> {
    /// The number of bytes read from storage for the work, released once it completes.
    #[cfg_attr(all(target_family = "wasm", not(feature = "async")), allow(dead_code))]
    pub(crate) fn fetched(&self) -> u64 {
        match self {
            Self::Compute { fetched, .. } => *fetched,
            Self::Join { .. } | Self::Range(_) => 0,
        }
    }

    /// Returns true if the work is a job of an indexed generator whose jobs have run tasks
    /// (rather than only returning them for reads), so the job is likely to compute.
    #[cfg_attr(any(not(feature = "async"), target_arch = "wasm32"), allow(dead_code))]
    pub(crate) fn is_computing_range(&self) -> bool {
        match self {
            Self::Range(range) => range.state.runs_tasks.load(Ordering::Relaxed),
            Self::Compute { .. } | Self::Join { .. } => false,
        }
    }

    /// Returns true if the work is a job of an indexed generator whose jobs have not run tasks (so far only returning
    /// them for reads), so the job is likely to only plan reads.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) fn is_planning_range(&self) -> bool {
        match self {
            Self::Range(range) => !range.state.runs_tasks.load(Ordering::Relaxed),
            Self::Compute { .. } | Self::Join { .. } => false,
        }
    }

    /// Make a job of an indexed generator stop after the block of indices in which it runs a task,
    /// so that a driver running it on a thread that should not compute for long can run later jobs elsewhere.
    #[cfg_attr(not(feature = "async"), allow(dead_code))]
    pub(crate) fn run_briefly(&mut self) {
        if let Self::Range(range) = self {
            range.brief = true;
        }
    }

    /// Run the work, performing the reads of tasks planned by a range and the writes of outcomes with `inline` if given.
    pub(crate) fn run(
        self,
        ctx: &ComputeCtx,
        inline: Option<&dyn InlineIo>,
        on_join: Option<&OnJoin<'_, 'a>>,
    ) -> Completed<'a> {
        let (node, result) = match self {
            Self::Range(range) => return Completed::Range(range.run(ctx, inline, on_join)),
            Self::Compute {
                node,
                source,
                ranges,
                reads,
                step,
                fetched: _,
                hint: _,
            } => {
                let reads = match (reads, source) {
                    (Some(reads), _) => Ok(reads),
                    (None, Some(source)) if !ranges.is_empty() => read_memory(&source, &ranges),
                    (None, _) => Ok(Reads::new()),
                };
                (node, reads.and_then(|reads| step.run(reads, ctx)))
            }
            Self::Join { node, join, values } => (node, join.run(values, ctx)),
        };
        Completed::Step(
            node,
            result.and_then(|outcome| write_inline(outcome, inline)),
        )
    }

    /// The scheduling hints of the work.
    #[cfg_attr(any(not(feature = "async"), target_arch = "wasm32"), allow(dead_code))]
    pub(crate) fn hint(&self) -> TaskHint {
        match self {
            Self::Compute { hint, .. } => *hint,
            Self::Join { .. } | Self::Range(_) => TaskHint::default(),
        }
    }
}

/// A read ready to be performed.
pub(crate) struct IoWork<'a> {
    pub(crate) node: NodeId,
    source: ByteSource<'a>,
    ranges: SmallVec<[ByteRange; 1]>,
    step: ComputeStep<'a>,
    hint: TaskHint,
}

impl<'a> IoWork<'a> {
    /// The compute step of the read, given the result of the read of `request`.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if the read failed or returned unexpected bytes.
    pub(crate) fn into_work(
        self,
        request: &IoRequest,
        result: Result<Option<ReadBytes>, StorageError>,
    ) -> Result<Work<'a>, CodecError> {
        let Self {
            node,
            source,
            ranges: _,
            step,
            hint,
        } = self;
        let reads = finish_reads(&source, &request.ranges, result)?;
        let fetched = reads
            .iter()
            .map(|read| read.as_ref().map_or(0, |read| read.len() as u64))
            .sum();
        Ok(Work::Compute {
            node,
            source: Some(source),
            ranges: SmallVec::new(),
            reads: Some(reads),
            fetched,
            step,
            hint,
        })
    }

    /// The key and byte ranges to read, without taking them from the work.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if a byte range is invalid for the source.
    pub(crate) fn peek_request(&self) -> Result<IoRequest, CodecError> {
        let SourceRoot::External(key) = self.source.root() else {
            unreachable!("I/O work has an external source")
        };
        let mut ranges = self.ranges.clone();
        self.source.resolve_in_place(&mut ranges)?;
        Ok(IoRequest { key: *key, ranges })
    }

    /// The key of the source of the read.
    fn key(&self) -> SourceKey {
        let SourceRoot::External(key) = self.source.root() else {
            unreachable!("I/O work has an external source")
        };
        *key
    }

    /// The key and byte ranges to read, taken from the work.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if a byte range is invalid for the source.
    pub(crate) fn request(&mut self) -> Result<IoRequest, CodecError> {
        let SourceRoot::External(key) = self.source.root() else {
            unreachable!("I/O work has an external source")
        };
        let mut ranges = std::mem::take(&mut self.ranges);
        self.source.resolve_in_place(&mut ranges)?;
        Ok(IoRequest { key: *key, ranges })
    }
}

/// The bytes of the `result` of a read of `ranges` (resolved) of `source`.
fn finish_reads<'a>(
    source: &ByteSource<'a>,
    ranges: &[ByteRange],
    result: Result<Option<ReadBytes>, StorageError>,
) -> Result<Reads<'a>, CodecError> {
    match result? {
        None => Ok(ranges.iter().map(|_| None).collect()),
        Some(mut bytes) if bytes.len() == 1 && ranges.len() == 1 => {
            let bytes = bytes.pop().expect("one range");
            Ok(smallvec::smallvec![Some(
                source.finish_read(CowBytes::Shared(bytes), ranges[0])?
            )])
        }
        Some(bytes) => {
            if bytes.len() != ranges.len() {
                return Err(CodecError::Other(format!(
                    "expected {} byte ranges from storage, got {}",
                    ranges.len(),
                    bytes.len()
                )));
            }
            let mut reads = Reads::with_capacity(ranges.len());
            for (bytes, range) in std::iter::zip(bytes, ranges) {
                reads.push(Some(source.finish_read(CowBytes::Shared(bytes), *range)?));
            }
            Ok(reads)
        }
    }
}

/// A single task, split into its read request (if it needs I/O) and the rest of the task.
pub(crate) enum SingleTask<'a> {
    /// The task needs a read.
    Read(SingleRead<'a>, IoRequest),
    /// The task needs no I/O.
    Compute(Task<'a>),
}

/// The part of a task that runs after its read.
pub(crate) struct SingleRead<'a> {
    source: ByteSource<'a>,
    step: ComputeStep<'a>,
}

impl<'a> SingleTask<'a> {
    /// Split `task` into its read request and the rest.
    pub(crate) fn new(task: Task<'a>) -> Result<Self, CodecError> {
        if !task.needs_io() {
            return Ok(Self::Compute(task));
        }
        let (source, mut ranges, step) = task.into_parts();
        let source = source.expect("a task with I/O has a source");
        let SourceRoot::External(key) = source.root() else {
            unreachable!("a task with I/O has an external source")
        };
        let key = *key;
        source.resolve_in_place(&mut ranges)?;
        Ok(Self::Read(
            SingleRead { source, step },
            IoRequest { key, ranges },
        ))
    }
}

impl<'a> SingleRead<'a> {
    /// Run the compute step on the `result` of the read of `request`.
    pub(crate) fn run(
        self,
        request: &IoRequest,
        result: Result<Option<ReadBytes>, StorageError>,
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        let reads = finish_reads(&self.source, &request.ranges, result)?;
        self.step.run(reads, ctx)
    }
}

/// The value of a single root task that completed with `outcome`, or a scheduler to continue it.
// The scheduler is moved once per drive, which is cheaper than boxing it
#[allow(clippy::result_large_err)]
pub(crate) fn single_outcome(outcome: Outcome<'_>) -> Result<TaskValue<'_>, Scheduler<'_>> {
    match outcome {
        Outcome::Done => Ok(TaskValue::None),
        Outcome::Value(value) => Ok(value),
        outcome @ (Outcome::Spawn(_)
        | Outcome::Fork { .. }
        | Outcome::Generate { .. }
        | Outcome::Write { .. }) => Err(Scheduler::from_root_outcome(outcome)),
    }
}

/// The key and byte ranges of a read, resolved to the root of the source.
pub(crate) struct IoRequest {
    pub(crate) key: SourceKey,
    pub(crate) ranges: SmallVec<[ByteRange; 1]>,
}

/// A soft limit on the bytes read from storage and not yet consumed by a compute step.
///
/// A read reserves an estimate of its size when it is dispatched. The reservation is replaced by
/// the actual size when the read completes, which is released when its compute step completes.
/// A read is always admitted if no bytes are held or reserved, so a read larger than the limit
/// does not stall a drive.
#[cfg_attr(target_family = "wasm", allow(dead_code))]
pub(crate) struct ByteBudget {
    limit: u64,
    held: u64,
    reserved: u64,
    observed_bytes: u64,
    observed_reads: u64,
}

#[cfg_attr(target_family = "wasm", allow(dead_code))]
impl ByteBudget {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            limit: limit as u64,
            held: 0,
            reserved: 0,
            observed_bytes: 0,
            observed_reads: 0,
        }
    }

    /// Estimate the bytes of a read: the hint if available, otherwise the mean of the reads of the drive so far,
    /// otherwise the bytes processed by its compute step.
    ///
    /// Without any of these, the estimate is the entire limit, so the read is dispatched alone.
    pub(crate) fn estimate(&self, hint: TaskHint) -> u64 {
        hint.read_bytes
            .or_else(|| self.observed_bytes.checked_div(self.observed_reads))
            .unwrap_or(if hint.compute_bytes > 0 {
                hint.compute_bytes
            } else {
                self.limit
            })
    }

    /// Returns true if a read of `estimate` bytes can be dispatched.
    pub(crate) fn admits(&self, estimate: u64) -> bool {
        (self.held == 0 && self.reserved == 0)
            || self
                .held
                .saturating_add(self.reserved)
                .saturating_add(estimate)
                <= self.limit
    }

    /// Reserve `estimate` bytes for a dispatched read.
    pub(crate) fn reserve(&mut self, estimate: u64) {
        self.reserved = self.reserved.saturating_add(estimate);
    }

    /// A read reserved with `estimate` completed with `actual` bytes, held until released.
    pub(crate) fn arrived(&mut self, estimate: u64, actual: u64) {
        self.reserved = self.reserved.saturating_sub(estimate);
        self.held = self.held.saturating_add(actual);
        self.observed_bytes = self.observed_bytes.saturating_add(actual);
        self.observed_reads += 1;
    }

    /// A read reserved with `estimate` failed.
    pub(crate) fn cancel(&mut self, estimate: u64) {
        self.reserved = self.reserved.saturating_sub(estimate);
    }

    /// Release `bytes` held for a completed compute step.
    pub(crate) fn release(&mut self, bytes: u64) {
        self.held = self.held.saturating_sub(bytes);
    }

    /// The bytes held and reserved.
    #[cfg(test)]
    pub(crate) fn outstanding(&self) -> u64 {
        self.held.saturating_add(self.reserved)
    }
}

/// A write ready to be performed.
pub(crate) struct WriteWork<'a> {
    pub(crate) node: NodeId,
    pub(crate) op: WriteOp<'a>,
}

/// A generator of the children of a node.
struct ActiveGenerator<'a> {
    node: NodeId,
    generator: Generating<'a>,
    /// Whether the children deliver values to a join of the node.
    join: bool,
    next_slot: usize,
    depth: usize,
}

/// The generator of an [`ActiveGenerator`].
enum Generating<'a> {
    /// An iterator of tasks, pulled by the scheduler.
    Iter(super::super::task::GeneratorIter<'a>),
    /// An indexed generator, run by jobs.
    Indexed(Arc<IndexedState<'a>>),
}

/// How a drive with `resources` coalesces reads of a backend with the `hint`, if at all.
#[cfg_attr(target_family = "wasm", allow(dead_code))]
pub(crate) fn read_coalescing(
    resources: &crate::Resources,
    hint: Option<ReadCoalescing>,
) -> Option<ReadCoalescing> {
    resources
        .read_coalescing()
        .or(hint)
        .filter(|coalescing| coalescing.max_coalesced_bytes() > 0)
}

/// The number of queued items that generators keep available to drivers with `resources`.
#[cfg_attr(all(target_family = "wasm", not(feature = "async")), allow(dead_code))]
pub(crate) fn generation_window(resources: &crate::Resources) -> usize {
    let target = |target: usize| if target == 0 { 64 } else { target.min(1024) };
    (2 * (target(resources.compute_target()) + target(resources.io_target()))).max(4)
}

/// A queue that yields deeper items first, then earlier items.
struct DepthQueue<T> {
    levels: Vec<VecDeque<T>>,
    len: usize,
}

impl<T> DepthQueue<T> {
    fn new() -> Self {
        Self {
            levels: Vec::new(),
            len: 0,
        }
    }

    fn push(&mut self, depth: usize, item: T) {
        if self.levels.len() <= depth {
            self.levels.resize_with(depth + 1, VecDeque::new);
        }
        self.levels[depth].push_back(item);
        self.len += 1;
    }

    fn pop(&mut self) -> Option<T> {
        let item = self.levels.iter_mut().rev().find_map(VecDeque::pop_front)?;
        self.len -= 1;
        Some(item)
    }

    fn peek(&self) -> Option<&T> {
        self.levels.iter().rev().find_map(VecDeque::front)
    }

    fn len(&self) -> usize {
        self.len
    }

    /// The positions (level, index) and items of up to `limit` items, in the order they would be popped.
    fn scan(&self, limit: usize) -> impl Iterator<Item = ((usize, usize), &T)> {
        self.levels
            .iter()
            .enumerate()
            .rev()
            .flat_map(|(level, items)| {
                items
                    .iter()
                    .enumerate()
                    .map(move |(index, item)| ((level, index), item))
            })
            .take(limit)
    }

    /// Remove the items at `positions` (from [`scan`](Self::scan)), returning them in the order of `positions`.
    fn remove_positions(&mut self, positions: &[(usize, usize)]) -> Vec<T> {
        let mut order: Vec<usize> = (0..positions.len()).collect();
        // Remove later items of a level first, so earlier indices are unchanged
        order.sort_unstable_by_key(|&i| std::cmp::Reverse(positions[i]));
        let mut removed: Vec<Option<T>> = (0..positions.len()).map(|_| None).collect();
        for i in order {
            let (level, index) = positions[i];
            removed[i] = self.levels[level].remove(index);
            self.len -= 1;
        }
        removed
            .into_iter()
            .map(|item| item.expect("a queued item"))
            .collect()
    }

    fn clear(&mut self) {
        self.levels.iter_mut().for_each(VecDeque::clear);
        self.len = 0;
    }
}

/// The scheduling state of a drive.
///
/// Work is prioritised depth-first, so a drive finishes what it started (e.g. the subchunks of a
/// shard) before starting more, which bounds the bytes it holds.
pub(crate) struct Scheduler<'a> {
    nodes: Vec<Option<Node<'a>>>,
    free: Vec<NodeId>,
    roots: Vec<TaskValue<'a>>,
    /// Root tasks that have not been submitted, with their index.
    pending_roots: std::iter::Enumerate<std::vec::IntoIter<Task<'a>>>,
    compute: DepthQueue<Work<'a>>,
    io: DepthQueue<IoWork<'a>>,
    writes: Vec<WriteWork<'a>>,
    generators: Vec<ActiveGenerator<'a>>,
    live: usize,
    error: Option<CodecError>,
    /// The number of compute steps that run in parallel, the maximum number of jobs of an indexed generator.
    parallelism: usize,
    /// The maximum number of tasks a job of an indexed generator returns for reads.
    max_deferred: usize,
    /// The indexed generators of the drive.
    registry: Arc<Registry<'a>>,
    /// The number of jobs of indexed generators queued or running.
    indexed_jobs: usize,
}

impl<'a> Scheduler<'a> {
    /// Create a scheduler for the root `tasks`.
    ///
    /// Root tasks are submitted as the queues are refilled ([`Scheduler::refill`]), after the children of submitted tasks.
    pub(crate) fn new(tasks: Vec<Task<'a>>) -> Self {
        let mut roots = Vec::new();
        roots.resize_with(tasks.len(), || TaskValue::None);
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
            roots,
            pending_roots: tasks.into_iter().enumerate(),
            compute: DepthQueue::new(),
            io: DepthQueue::new(),
            writes: Vec::new(),
            generators: Vec::new(),
            live: 0,
            error: None,
            parallelism: 1,
            max_deferred: 64,
            registry: Arc::default(),
            indexed_jobs: 0,
        }
    }

    /// Set the number of compute steps that run in parallel (default 1), the maximum number of jobs of an indexed generator.
    #[cfg_attr(all(target_family = "wasm", not(feature = "async")), allow(dead_code))]
    pub(crate) fn set_parallelism(&mut self, parallelism: usize) {
        self.parallelism = parallelism.max(1);
    }

    /// Set the maximum number of tasks a job of an indexed generator returns for reads (default 64).
    #[cfg_attr(all(target_family = "wasm", not(feature = "async")), allow(dead_code))]
    pub(crate) fn set_max_deferred(&mut self, max_deferred: usize) {
        self.max_deferred = max_deferred.max(1);
    }

    /// Create a scheduler for a single root task whose compute step produced `outcome`.
    pub(crate) fn from_root_outcome(outcome: Outcome<'a>) -> Self {
        let mut scheduler = Self::new(Vec::new());
        scheduler.roots.push(TaskValue::None);
        let root = scheduler.insert(Node {
            parent: Parent::Root(0),
            depth: 0,
            computing: true,
            pending: 0,
            join: None,
            values: Vec::new(),
            value: TaskValue::None,
        });
        scheduler.complete(root, Ok(outcome));
        scheduler
    }

    fn insert(&mut self, node: Node<'a>) -> NodeId {
        let id = if let Some(id) = self.free.pop() {
            self.nodes[id] = Some(node);
            id
        } else {
            self.nodes.push(Some(node));
            self.nodes.len() - 1
        };
        self.live += 1;
        id
    }

    fn submit(&mut self, parent: Parent, depth: usize, task: Task<'a>) {
        let id = self.insert(Node {
            parent,
            depth,
            computing: true,
            pending: 0,
            join: None,
            values: Vec::new(),
            value: TaskValue::None,
        });
        let needs_io = task.needs_io();
        let hint = *task.hint();
        let (source, ranges, step) = task.into_parts();
        if needs_io {
            self.io.push(
                depth,
                IoWork {
                    node: id,
                    source: source.expect("a task with I/O has a source"),
                    ranges,
                    step,
                    hint,
                },
            );
        } else {
            self.push_compute(
                depth,
                Work::Compute {
                    node: id,
                    source,
                    ranges,
                    reads: None,
                    fetched: 0,
                    step,
                    hint,
                },
            );
        }
    }

    fn push_compute(&mut self, depth: usize, work: Work<'a>) {
        self.compute.push(depth, work);
    }

    fn node(&mut self, id: NodeId) -> &mut Node<'a> {
        self.nodes[id].as_mut().expect("the node is live")
    }

    /// Returns true if the drive has failed.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) fn failed(&self) -> bool {
        self.error.is_some()
    }

    /// The number of queued compute steps, reads, and writes.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) fn queued(&self) -> usize {
        self.compute.len() + self.io.len() + self.writes.len()
    }

    /// The number of queued compute steps.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) fn compute_queued(&self) -> usize {
        self.compute.len()
    }

    /// The number of queued reads.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) fn io_queued(&self) -> usize {
        self.io.len()
    }

    /// Take the next compute step to run.
    pub(crate) fn pop_compute(&mut self) -> Option<Work<'a>> {
        self.compute.pop()
    }

    /// Take the next read to perform.
    pub(crate) fn pop_io(&mut self) -> Option<IoWork<'a>> {
        self.io.pop()
    }

    /// Take the next read to perform if `budget` admits it, with the queued reads of the same key that coalesce with it
    /// under `coalescing` and that `budget` admits.
    ///
    /// Returns [`None`] if there is no read or `budget` does not admit the next read.
    /// Returns the bytes reserved for the group, also on error.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) fn pop_read_group(
        &mut self,
        coalescing: Option<ReadCoalescing>,
        budget: &mut ByteBudget,
    ) -> Option<(Result<ReadGroup<'a>, CodecError>, u64)> {
        let mut admit = |hint: TaskHint| {
            let estimate = budget.estimate(hint);
            budget.admits(estimate).then(|| {
                budget.reserve(estimate);
                estimate
            })
        };
        let hint = self.io.peek()?.hint;
        let mut reserved = admit(hint)?;
        let mut read = self.io.pop().expect("peeked");
        let request = match read.request() {
            Ok(request) => request,
            Err(err) => return Some((Err(err), reserved)),
        };
        let Some(coalescing) = coalescing.filter(|_| coalesce::coalescable(&request)) else {
            return Some((Ok(ReadGroup::single(read, request)), reserved));
        };
        if hint.read_bytes.is_none() {
            // The bytes of a coalescable read are known, so others can be admitted with it
            budget.cancel(reserved);
            reserved = coalesce::known_len(&request);
            budget.reserve(reserved);
        }
        let mut admit = |hint: TaskHint| {
            let estimate = budget.estimate(hint);
            budget.admits(estimate).then(|| {
                budget.reserve(estimate);
                estimate
            })
        };

        // Queued reads of the same key, and their requests
        let key = request.key;
        let mut candidates: Vec<((usize, usize), IoRequest, TaskHint)> = self
            .io
            .scan(coalesce::SCAN_LIMIT)
            .filter(|(_, candidate)| candidate.key() == key)
            .filter_map(|(position, candidate)| {
                let request = candidate.peek_request().ok()?;
                // The bytes of a coalescable read are known
                let hint = TaskHint {
                    read_bytes: Some(coalesce::known_len(&request)),
                    ..candidate.hint
                };
                coalesce::coalescable(&request).then_some((position, request, hint))
            })
            .collect();
        // Grow the group with the candidates that coalesce with its ranges, while admitted
        let mut group = IoRequest {
            key,
            ranges: request.ranges.clone(),
        };
        let mut members = Vec::new();
        let mut grew = true;
        while grew && !candidates.is_empty() {
            grew = false;
            let mut i = 0;
            while i < candidates.len() {
                if coalesce::coalesces_with(&group, &candidates[i].1.ranges, coalescing) {
                    let Some(bytes) = admit(candidates[i].2) else {
                        grew = false;
                        candidates.clear();
                        break;
                    };
                    reserved = reserved.saturating_add(bytes);
                    let (position, candidate, _) = candidates.swap_remove(i);
                    group.ranges.extend_from_slice(&candidate.ranges);
                    members.push((position, candidate));
                    grew = true;
                } else {
                    i += 1;
                }
            }
        }
        if members.is_empty() {
            return Some((Ok(ReadGroup::single(read, request)), reserved));
        }
        let parts_len = members
            .iter()
            .fold(coalesce::known_len(&request), |len, (_, request)| {
                len.saturating_add(coalesce::known_len(request))
            });
        let positions: Vec<(usize, usize)> =
            members.iter().map(|(position, _)| *position).collect();
        let mut reads: SmallVec<[(IoWork<'a>, IoRequest); 1]> =
            smallvec::smallvec![(read, request)];
        for (mut member, (_, request)) in
            std::iter::zip(self.io.remove_positions(&positions), members)
        {
            // The ranges of the work are taken, as by `request`
            member.ranges = SmallVec::new();
            reads.push((member, request));
        }
        let group = ReadGroup::coalesced(reads, coalescing);
        // The bytes read in gaps between the ranges are also held until the read completes
        let gaps = coalesce::known_len(group.request()).saturating_sub(parts_len);
        budget.reserve(gaps);
        reserved = reserved.saturating_add(gaps);
        Some((Ok(group), reserved))
    }

    /// The scheduling hints of the next read, if any.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) fn peek_io_hint(&self) -> Option<TaskHint> {
        self.io.peek().map(|read| read.hint)
    }

    /// The number of queued writes.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) fn writes_queued(&self) -> usize {
        self.writes.len()
    }

    /// Take the next write to perform.
    pub(crate) fn pop_write(&mut self) -> Option<WriteWork<'a>> {
        self.writes.pop()
    }

    /// Record the result of a read.
    pub(crate) fn complete_io(
        &mut self,
        work: IoWork<'a>,
        request: &IoRequest,
        result: Result<Option<ReadBytes>, StorageError>,
    ) {
        match work.into_work(request, result) {
            Ok(work) => self.ready(work),
            Err(err) => self.fail(err),
        }
    }

    /// Queue compute work whose reads have completed.
    pub(crate) fn ready(&mut self, work: Work<'a>) {
        if self.error.is_none() {
            let node = match &work {
                Work::Compute { node, .. } | Work::Join { node, .. } => *node,
                Work::Range(range) => range.state.node(),
            };
            let depth = self.node(node).depth;
            self.push_compute(depth, work);
        }
    }

    /// Record the result of a write.
    pub(crate) fn complete_write(&mut self, node: NodeId, result: Result<(), StorageError>) {
        match result {
            Ok(()) => {
                if self.error.is_none() {
                    self.node(node).pending -= 1;
                    self.try_finish(node);
                }
            }
            Err(err) => self.fail(err.into()),
        }
    }

    /// Record the result of running work.
    pub(crate) fn complete_work(&mut self, completed: Completed<'a>) {
        match completed {
            Completed::Step(node, result) => self.complete(node, result),
            Completed::Range(result) => self.complete_range(result),
        }
    }

    /// Record the results of a job of indexed generators.
    fn complete_range(&mut self, result: Result<RangeResults<'a>, CodecError>) {
        self.indexed_jobs -= 1;
        if self.error.is_some() {
            return;
        }
        let results = match result {
            Ok(results) => results,
            Err(err) => {
                self.fail(err);
                return;
            }
        };
        for (state, output) in results {
            let id = state.node();
            self.complete_range_output(id, output);
            // Once recorded, the generator can complete (see `refill`)
            state.unreported.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// Record the tasks a job ran of the indexed generator of the node `id`.
    fn complete_range_output(&mut self, id: NodeId, output: RangeOutput<'a>) {
        let RangeOutput {
            collect,
            values,
            outcomes,
            tasks,
        } = output;
        let parent = |index: usize| {
            if collect {
                Parent::Join(id, index)
            } else {
                Parent::Spawn(id)
            }
        };
        let node = self.node(id);
        let depth = node.depth + 1;
        // Tasks with more work remain pending until it completes
        node.pending += outcomes.len() + tasks.len();
        for (index, value) in values {
            node.values[index] = value;
        }
        for (index, outcome) in outcomes {
            let child = self.insert(Node {
                parent: parent(index),
                depth,
                computing: true,
                pending: 0,
                join: None,
                values: Vec::new(),
                value: TaskValue::None,
            });
            self.complete(child, Ok(outcome));
        }
        for (index, task) in tasks {
            self.submit(parent(index), depth, task);
        }
    }

    /// Record an indexed generator created and joined by a running job.
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    pub(crate) fn joined(&mut self, joined: Joined<'a>) {
        if self.error.is_some() {
            return;
        }
        let Joined {
            parent,
            index,
            join,
            state,
        } = joined;
        let parent_id = parent.node();
        let parent_node = self.node(parent_id);
        // The task of the generator remains pending until the generator completes
        parent_node.pending += 1;
        let depth = parent_node.depth + 1;
        let has_join = join.as_ref().is_some_and(Join::collects);
        let mut values = Vec::new();
        if has_join {
            values.resize_with(state.len, || TaskValue::None);
        }
        let id = self.insert(Node {
            parent: if parent.collect {
                Parent::Join(parent_id, index)
            } else {
                Parent::Spawn(parent_id)
            },
            depth,
            computing: false,
            // The generator is pending until it completes
            pending: 1,
            join,
            values,
            value: TaskValue::None,
        });
        self.add_indexed(id, depth + 1, has_join, state, false);
    }

    /// Add the indexed generator `state` of the node `id`, whose tasks have `depth`.
    ///
    /// The generator is added to the registry if `register` (a joined generator is registered by its job).
    fn add_indexed(
        &mut self,
        id: NodeId,
        depth: usize,
        collect: bool,
        state: Arc<IndexedState<'a>>,
        register: bool,
    ) {
        state.node.store(id, Ordering::Release);
        if register {
            self.registry
                .lock()
                .expect("the registry lock is not poisoned")
                .push(state.clone());
        }
        self.generators.push(ActiveGenerator {
            node: id,
            generator: Generating::Indexed(state),
            join: collect,
            next_slot: 0,
            depth,
        });
    }

    /// Record the outcome of a compute or join step.
    pub(crate) fn complete(&mut self, id: NodeId, result: Result<Outcome<'a>, CodecError>) {
        if self.error.is_some() {
            return;
        }
        let outcome = match result {
            Ok(outcome) => outcome,
            Err(err) => {
                self.fail(err);
                return;
            }
        };
        let node = self.node(id);
        node.computing = false;
        let depth = node.depth;
        match outcome {
            Outcome::Done => node.value = TaskValue::None,
            Outcome::Value(value) => node.value = value,
            Outcome::Spawn(children) => {
                node.pending += children.len();
                for child in children {
                    self.submit(Parent::Spawn(id), depth + 1, child);
                }
            }
            Outcome::Fork { children, join } => {
                let collect = join.collects();
                node.join = Some(join);
                node.pending += children.len();
                if collect {
                    node.values = (0..children.len()).map(|_| TaskValue::None).collect();
                    for (i, child) in children.into_iter().enumerate() {
                        self.submit(Parent::Join(id, i), depth + 1, child);
                    }
                } else {
                    for child in children {
                        self.submit(Parent::Spawn(id), depth + 1, child);
                    }
                }
            }
            Outcome::Generate { children, join } => {
                // The generator is pending until it completes
                node.pending += 1;
                let has_join = join.as_ref().is_some_and(Join::collects);
                node.join = join;
                match children.0 {
                    GeneratorKind::Indexed { len, plan } => {
                        if has_join {
                            node.values.resize_with(len, || TaskValue::None);
                        }
                        let block = block_size(len, self.parallelism);
                        let state = Arc::new(IndexedState::new(len, plan, has_join, block, 0));
                        self.add_indexed(id, depth + 1, has_join, state, true);
                    }
                    GeneratorKind::Iter(iter) => self.generators.push(ActiveGenerator {
                        node: id,
                        generator: Generating::Iter(iter),
                        join: has_join,
                        next_slot: 0,
                        depth: depth + 1,
                    }),
                }
            }
            Outcome::Write { ops, then } => {
                node.value = then;
                node.pending += ops.len();
                self.writes
                    .extend(ops.into_iter().map(|op| WriteWork { node: id, op }));
            }
        }
        self.try_finish(id);
    }

    /// Generate tasks from active generators, deepest first, then submit root tasks, until `window` items are queued or there are no more tasks.
    pub(crate) fn refill(&mut self, window: usize) {
        // Indexed generators with enough jobs, skipped so that other generators can use free compute threads
        let mut saturated = Vec::new();
        while self.error.is_none() && self.queued() < window {
            let Some(mut active) = self.generators.pop() else {
                let Some((i, task)) = self.pending_roots.next() else {
                    break;
                };
                self.submit(Parent::Root(i), 0, task);
                continue;
            };
            let generator = match &mut active.generator {
                Generating::Indexed(state) => {
                    if state.is_complete() {
                        let state = state.clone();
                        self.registry
                            .lock()
                            .expect("the registry lock is not poisoned")
                            .retain(|other| !Arc::ptr_eq(other, &state));
                        self.node(active.node).pending -= 1;
                        self.try_finish(active.node);
                        continue;
                    }
                    if !state.try_join() {
                        // Exhausted, with jobs whose results are not recorded
                        saturated.push(active);
                        continue;
                    }
                    let unclaimed = state
                        .len
                        .saturating_sub(state.cursor.load(Ordering::SeqCst));
                    if self.indexed_jobs >= self.parallelism
                        || state.queued.load(Ordering::SeqCst) >= unclaimed
                    {
                        // Enough jobs (which take tasks from any generator), or enough queued for the tasks left
                        state.unreported.fetch_sub(1, Ordering::SeqCst);
                        saturated.push(active);
                        continue;
                    }
                    self.indexed_jobs += 1;
                    state.queued.fetch_add(1, Ordering::SeqCst);
                    let work = Work::Range(RangeWork {
                        state: state.clone(),
                        registry: self.registry.clone(),
                        max_deferred: self.max_deferred,
                        parallelism: self.parallelism,
                        brief: false,
                    });
                    let depth = active.depth;
                    self.generators.push(active);
                    self.push_compute(depth, work);
                    continue;
                }
                Generating::Iter(generator) => generator,
            };
            match generator.next() {
                Some(Ok(task)) => {
                    let node = self.node(active.node);
                    node.pending += 1;
                    let parent = if active.join {
                        node.values.push(TaskValue::None);
                        Parent::Join(active.node, active.next_slot)
                    } else {
                        Parent::Spawn(active.node)
                    };
                    active.next_slot += 1;
                    let depth = active.depth;
                    self.generators.push(active);
                    self.submit(parent, depth, task);
                }
                Some(Err(err)) => self.fail(err),
                None => {
                    self.node(active.node).pending -= 1;
                    self.try_finish(active.node);
                }
            }
        }
        if self.error.is_none() {
            // Restore the order of the generators, deepest last
            self.generators.extend(saturated.into_iter().rev());
        }
    }

    fn try_finish(&mut self, mut id: NodeId) {
        loop {
            let node = self.node(id);
            if node.computing || node.pending > 0 {
                return;
            }
            if let Some(join) = node.join.take() {
                node.computing = true;
                let values = std::mem::take(&mut node.values);
                let depth = node.depth;
                self.push_compute(
                    depth,
                    Work::Join {
                        node: id,
                        join,
                        values,
                    },
                );
                return;
            }
            let node = self.nodes[id].take().expect("the node is live");
            self.free.push(id);
            self.live -= 1;
            match node.parent {
                Parent::Root(i) => {
                    self.roots[i] = node.value;
                    return;
                }
                Parent::Join(parent, slot) => {
                    let parent_node = self.node(parent);
                    parent_node.values[slot] = node.value;
                    parent_node.pending -= 1;
                    id = parent;
                }
                Parent::Spawn(parent) => {
                    self.node(parent).pending -= 1;
                    id = parent;
                }
            }
        }
    }

    /// Fail the drive with `err`, unless it has already failed, and drop all queued work.
    pub(crate) fn fail(&mut self, err: CodecError) {
        if self.error.is_none() {
            self.error = Some(err);
        }
        self.compute.clear();
        self.io.clear();
        self.writes.clear();
        self.generators.clear();
        self.pending_roots = Vec::new().into_iter().enumerate();
    }

    /// The values of the root tasks, or the first error.
    ///
    /// Must only be called once the scheduler is idle and the driver has no work in flight.
    pub(crate) fn finish(self) -> Result<Vec<TaskValue<'a>>, CodecError> {
        if let Some(err) = self.error {
            return Err(err);
        }
        debug_assert_eq!(self.live, 0, "all tasks completed");
        Ok(self.roots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A task that reads `length` bytes at `offset` of key 0, with its size as a hint.
    fn read_at<'a>(offset: u64, length: u64) -> Task<'a> {
        Task::read(
            ByteSource::external(SourceKey(0)),
            [ByteRange::FromStart(offset, Some(length))],
            |_, _| Ok(Outcome::Done),
        )
        .with_hint(TaskHint {
            read_bytes: Some(length),
            compute_bytes: 0,
        })
    }

    #[test]
    fn byte_budget_saturates() {
        let mut budget = ByteBudget::new(100);
        // A read is admitted alone, whatever its size
        assert!(budget.admits(u64::MAX));
        budget.reserve(u64::MAX);
        budget.reserve(10);
        // Nothing more is admitted, without overflowing
        assert!(!budget.admits(u64::MAX));
        assert!(!budget.admits(1));
        budget.arrived(u64::MAX, u64::MAX);
        assert!(!budget.admits(1));
        budget.release(u64::MAX);
        budget.cancel(10);
        assert_eq!(budget.outstanding(), 0);
        assert!(budget.admits(100));
    }

    #[test]
    fn coalesced_gaps_are_charged_to_the_budget() {
        let mut scheduler = Scheduler::new(vec![read_at(0, 10), read_at(20, 10), read_at(60, 10)]);
        scheduler.refill(16);
        let mut budget = ByteBudget::new(1000);
        let (group, reserved) = scheduler
            .pop_read_group(Some(ReadCoalescing::new(16, 1024)), &mut budget)
            .unwrap();
        let group = group.unwrap();
        // The first two reads and the gap between them, but not the third read beyond the maximum gap
        assert_eq!(
            group.request().ranges.as_slice(),
            &[ByteRange::FromStart(0, Some(30))]
        );
        assert_eq!(reserved, 30);
        assert_eq!(budget.outstanding(), 30);
        assert_eq!(scheduler.io_queued(), 1);
    }

    #[test]
    fn ranges_whose_end_overflows_are_not_coalesced() {
        let mut scheduler = Scheduler::new(vec![read_at(0, 10), read_at(u64::MAX - 5, 10)]);
        scheduler.refill(16);
        let mut budget = ByteBudget::new(1000);
        let coalescing = Some(ReadCoalescing::new(u64::MAX, u64::MAX));
        let (group, _) = scheduler.pop_read_group(coalescing, &mut budget).unwrap();
        assert_eq!(
            group.unwrap().request().ranges.as_slice(),
            &[ByteRange::FromStart(0, Some(10))]
        );
        // The other read is performed alone
        let (group, _) = scheduler.pop_read_group(coalescing, &mut budget).unwrap();
        assert_eq!(
            group.unwrap().request().ranges.as_slice(),
            &[ByteRange::FromStart(u64::MAX - 5, Some(10))]
        );
    }

    #[test]
    fn byte_budget() {
        let mut budget = ByteBudget::new(100);
        // Without information, a read reserves the entire budget
        assert_eq!(budget.estimate(TaskHint::default()), 100);
        // The first read is admitted whatever its size
        let estimate = budget.estimate(TaskHint {
            read_bytes: None,
            compute_bytes: 1000,
        });
        assert_eq!(estimate, 1000);
        assert!(budget.admits(estimate));
        budget.reserve(estimate);
        assert!(!budget.admits(1));
        budget.arrived(estimate, 40);
        assert_eq!(budget.outstanding(), 40);
        // Later estimates use the mean observed size
        let estimate = budget.estimate(TaskHint::default());
        assert_eq!(estimate, 40);
        let estimate = budget.estimate(TaskHint {
            read_bytes: None,
            compute_bytes: 1000,
        });
        assert_eq!(estimate, 40);
        assert!(budget.admits(estimate));
        budget.reserve(estimate);
        assert!(!budget.admits(estimate));
        budget.cancel(estimate);
        budget.release(40);
        assert_eq!(budget.outstanding(), 0);
        // Hinted sizes are exact
        assert_eq!(
            budget.estimate(TaskHint {
                read_bytes: Some(7),
                compute_bytes: 0
            }),
            7
        );
    }
}
