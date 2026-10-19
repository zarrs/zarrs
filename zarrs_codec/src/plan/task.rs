//! Tasks of a planned operation.

use std::any::Any;

use smallvec::SmallVec;
use zarrs_storage::byte_range::ByteRange;

use super::{ByteSource, SourceKey};
use crate::{ArrayBytes, CodecError, ComputeResources, CowBytes};

/// The bytes read by a task, one per requested byte range.
///
/// An entry is [`None`] if the source does not exist (e.g. a missing chunk).
pub type Reads<'a> = SmallVec<[Option<CowBytes<'a>>; 1]>;

/// The context of a compute step.
#[derive(Debug, Clone)]
pub struct ComputeCtx {
    resources: ComputeResources,
    inline: bool,
}

impl ComputeCtx {
    /// Create a compute context.
    #[must_use]
    pub fn new(resources: ComputeResources) -> Self {
        Self {
            resources,
            inline: false,
        }
    }

    /// Create the context of a compute step that runs on the polling task of an asynchronous drive.
    #[cfg(feature = "async")]
    #[must_use]
    pub(crate) fn new_inline(resources: ComputeResources) -> Self {
        Self {
            resources,
            inline: true,
        }
    }

    /// Returns true if the compute step runs on the polling task of an asynchronous drive.
    ///
    /// The tasks it forks then also run serially on the polling task, so a step that can either fork tasks or compute
    /// in one step that parallelises internally (e.g. encoding a shard) should compute in one step.
    #[must_use]
    pub fn runs_inline(&self) -> bool {
        self.inline
    }

    /// The compute resources available to the compute step.
    ///
    /// The compute target is the share of the compute target of the drive available to this step.
    /// A compute step that runs an operation with its own concurrency (e.g. a nested drive) should use these resources.
    #[must_use]
    pub fn resources(&self) -> &ComputeResources {
        &self.resources
    }
}

#[cfg(not(target_arch = "wasm32"))]
type ComputeFn<'a> =
    Box<dyn FnOnce(Reads<'a>, &ComputeCtx) -> Result<Outcome<'a>, CodecError> + Send + 'a>;
#[cfg(target_arch = "wasm32")]
type ComputeFn<'a> =
    Box<dyn FnOnce(Reads<'a>, &ComputeCtx) -> Result<Outcome<'a>, CodecError> + 'a>;

#[cfg(not(target_arch = "wasm32"))]
type JoinFn<'a> =
    Box<dyn FnOnce(Vec<TaskValue<'a>>, &ComputeCtx) -> Result<Outcome<'a>, CodecError> + Send + 'a>;
#[cfg(target_arch = "wasm32")]
type JoinFn<'a> =
    Box<dyn FnOnce(Vec<TaskValue<'a>>, &ComputeCtx) -> Result<Outcome<'a>, CodecError> + 'a>;

/// Scheduling hints of a task.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TaskHint {
    /// The expected number of bytes read by the task, if known.
    pub read_bytes: Option<u64>,
    /// The approximate number of bytes processed by the compute step of the task.
    ///
    /// An asynchronous driver only moves compute steps above a threshold off the polling task.
    pub compute_bytes: u64,
}

/// A unit of work in a planned operation: read byte ranges of a source, then compute.
pub struct Task<'a> {
    source: Option<ByteSource<'a>>,
    reads: SmallVec<[ByteRange; 1]>,
    compute: ComputeFn<'a>,
    hint: TaskHint,
}

impl std::fmt::Debug for Task<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Task")
            .field("source", &self.source)
            .field("reads", &self.reads)
            .field("hint", &self.hint)
            .finish_non_exhaustive()
    }
}

impl<'a> Task<'a> {
    /// A task that reads `reads` from `source`, then runs `compute` on the bytes read.
    pub fn read(
        source: ByteSource<'a>,
        reads: impl Into<SmallVec<[ByteRange; 1]>>,
        compute: impl FnOnce(Reads<'a>, &ComputeCtx) -> Result<Outcome<'a>, CodecError>
        + zarrs_plugin::MaybeSend
        + 'a,
    ) -> Self {
        Self {
            source: Some(source),
            reads: reads.into(),
            compute: Box::new(compute),
            hint: TaskHint::default(),
        }
    }

    /// A task that only computes.
    pub fn compute(
        compute: impl FnOnce(&ComputeCtx) -> Result<Outcome<'a>, CodecError>
        + zarrs_plugin::MaybeSend
        + 'a,
    ) -> Self {
        Self {
            source: None,
            reads: SmallVec::new(),
            compute: Box::new(move |_reads, ctx| compute(ctx)),
            hint: TaskHint::default(),
        }
    }

    /// Set the scheduling hints of the task.
    #[must_use]
    pub fn with_hint(mut self, hint: TaskHint) -> Self {
        self.hint = hint;
        self
    }

    /// The scheduling hints of the task.
    #[must_use]
    pub fn hint(&self) -> &TaskHint {
        &self.hint
    }

    /// The source the task reads from, if any.
    #[must_use]
    pub fn source(&self) -> Option<&ByteSource<'a>> {
        self.source.as_ref()
    }

    /// The byte ranges the task reads.
    #[must_use]
    pub fn reads(&self) -> &[ByteRange] {
        &self.reads
    }

    /// Returns true if the task reads from an external source and so needs I/O.
    #[must_use]
    pub fn needs_io(&self) -> bool {
        !self.reads.is_empty() && self.source.as_ref().is_some_and(ByteSource::is_external)
    }

    /// Split the task into its source, byte ranges, and compute step.
    pub(crate) fn into_parts(
        self,
    ) -> (
        Option<ByteSource<'a>>,
        SmallVec<[ByteRange; 1]>,
        ComputeStep<'a>,
    ) {
        (self.source, self.reads, ComputeStep(self.compute))
    }

    /// Read the byte ranges of a task with a memory source (or no reads) and run its compute step.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if a read is invalid or the compute step fails.
    ///
    /// # Panics
    /// Panics if the task needs I/O.
    pub fn run_without_io(self, ctx: &ComputeCtx) -> Result<Outcome<'a>, CodecError> {
        assert!(!self.needs_io(), "the task needs I/O");
        let (source, ranges, compute) = self.into_parts();
        let reads = match source {
            Some(source) if !ranges.is_empty() => read_memory(&source, &ranges)?,
            _ => Reads::new(),
        };
        compute.run(reads, ctx)
    }
}

/// Read `ranges` of a memory `source`.
pub(crate) fn read_memory<'a>(
    source: &ByteSource<'a>,
    ranges: &[ByteRange],
) -> Result<Reads<'a>, CodecError> {
    if let super::SourceRoot::Memory(_) = source.root() {
        // Slices of the root, without collecting them first
        return ranges
            .iter()
            .map(|range| {
                Ok(Some(
                    source
                        .read_memory(*range)
                        .expect("the source is in memory")?,
                ))
            })
            .collect();
    }
    Ok(source
        .read_memory_many(ranges)
        .expect("the source is in memory")?
        .into_iter()
        .map(Some)
        .collect())
}

/// The compute step of a task.
pub(crate) struct ComputeStep<'a>(ComputeFn<'a>);

impl<'a> ComputeStep<'a> {
    pub(crate) fn run(self, reads: Reads<'a>, ctx: &ComputeCtx) -> Result<Outcome<'a>, CodecError> {
        (self.0)(reads, ctx)
    }
}

/// The step that combines the values of the children of a [`Outcome::Fork`].
pub struct Join<'a> {
    join: JoinFn<'a>,
    collect: bool,
}

impl<'a> Join<'a> {
    /// Create a join step.
    ///
    /// The values are ordered as the children of the fork.
    pub fn new(
        join: impl FnOnce(Vec<TaskValue<'a>>, &ComputeCtx) -> Result<Outcome<'a>, CodecError>
        + zarrs_plugin::MaybeSend
        + 'a,
    ) -> Self {
        Self {
            join: Box::new(join),
            collect: true,
        }
    }

    /// Create a join step that runs once the children complete, discarding their values.
    pub fn after(
        join: impl FnOnce(&ComputeCtx) -> Result<Outcome<'a>, CodecError> + zarrs_plugin::MaybeSend + 'a,
    ) -> Self {
        Self {
            join: Box::new(move |_values, ctx| join(ctx)),
            collect: false,
        }
    }

    /// Returns true if the join step receives the values of the children.
    pub(crate) fn collects(&self) -> bool {
        self.collect
    }

    pub(crate) fn run(
        self,
        values: Vec<TaskValue<'a>>,
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        (self.join)(values, ctx)
    }
}

impl std::fmt::Debug for Join<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Join").finish_non_exhaustive()
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) type GeneratorIter<'a> =
    Box<dyn Iterator<Item = Result<Task<'a>, CodecError>> + Send + 'a>;
#[cfg(target_arch = "wasm32")]
pub(crate) type GeneratorIter<'a> = Box<dyn Iterator<Item = Result<Task<'a>, CodecError>> + 'a>;

#[cfg(not(target_arch = "wasm32"))]
pub(crate) type PlanFn<'a> =
    std::sync::Arc<dyn Fn(usize) -> Result<Option<Task<'a>>, CodecError> + Send + Sync + 'a>;
#[cfg(target_arch = "wasm32")]
pub(crate) type PlanFn<'a> =
    std::sync::Arc<dyn Fn(usize) -> Result<Option<Task<'a>>, CodecError> + 'a>;

/// A generator of child tasks. See [`Outcome::Generate`].
///
/// A generator is either
/// - an iterator of tasks ([`Generator::new`]), pulled by the driver as it has capacity for them, or
/// - a number of tasks planned by index ([`Generator::indexed`]), which drivers plan and run in ranges on their compute threads.
pub struct Generator<'a>(pub(crate) GeneratorKind<'a>);

pub(crate) enum GeneratorKind<'a> {
    Iter(GeneratorIter<'a>),
    Indexed { len: usize, plan: PlanFn<'a> },
}

impl<'a> Generator<'a> {
    /// Create a generator from an iterator of tasks.
    ///
    /// The driver pulls tasks from the iterator on the thread driving the tasks.
    /// An error from the iterator fails the drive.
    pub fn new(
        tasks: impl Iterator<Item = Result<Task<'a>, CodecError>> + zarrs_plugin::MaybeSend + 'a,
    ) -> Self {
        Self(GeneratorKind::Iter(Box::new(tasks)))
    }

    /// Create a generator of `len` tasks, where `plan` plans the task with an index.
    ///
    /// `plan` may return [`None`] if the index needs no task (e.g. it has already been handled).
    /// Tasks are planned (and, where possible, run) in ranges of indices on compute threads, so
    /// planning is parallel and the driver does not handle each task.
    /// The values of the tasks are delivered to a join step in index order.
    /// An error from `plan` fails the drive.
    pub fn indexed(
        len: usize,
        plan: impl Fn(usize) -> Result<Option<Task<'a>>, CodecError>
        + zarrs_plugin::MaybeSend
        + zarrs_plugin::MaybeSync
        + 'a,
    ) -> Self {
        Self(GeneratorKind::Indexed {
            len,
            plan: std::sync::Arc::new(plan),
        })
    }

    /// Plan the task at `index` of an indexed generator.
    ///
    /// # Errors
    /// Returns a [`CodecError`] if planning fails.
    ///
    /// # Panics
    /// Panics if the generator is not indexed or `index` is out of bounds.
    pub fn plan_index(&self, index: usize) -> Result<Option<Task<'a>>, CodecError> {
        let GeneratorKind::Indexed { len, plan } = &self.0 else {
            panic!("the generator is not indexed")
        };
        assert!(index < *len, "the index is out of bounds");
        plan(index)
    }
}

impl std::fmt::Debug for Generator<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Generator").finish_non_exhaustive()
    }
}

#[cfg(not(target_arch = "wasm32"))]
type FoldBox<'a> = Box<dyn FoldState<'a> + Send + 'a>;
#[cfg(target_arch = "wasm32")]
type FoldBox<'a> = Box<dyn FoldState<'a> + 'a>;

/// The state and steps of a [`Fold`].
trait FoldState<'a> {
    fn step(&mut self, value: TaskValue<'a>, ctx: &ComputeCtx) -> Result<Outcome<'a>, CodecError>;

    fn finish(self: Box<Self>, ctx: &ComputeCtx) -> Result<Outcome<'a>, CodecError>;
}

struct FoldFns<S, F, G> {
    state: S,
    fold: F,
    finish: G,
}

impl<'a, S, F, G> FoldState<'a> for FoldFns<S, F, G>
where
    F: FnMut(&mut S, TaskValue<'a>, &ComputeCtx) -> Result<Outcome<'a>, CodecError>,
    G: FnOnce(S, &ComputeCtx) -> Result<Outcome<'a>, CodecError>,
{
    fn step(&mut self, value: TaskValue<'a>, ctx: &ComputeCtx) -> Result<Outcome<'a>, CodecError> {
        (self.fold)(&mut self.state, value, ctx)
    }

    fn finish(self: Box<Self>, ctx: &ComputeCtx) -> Result<Outcome<'a>, CodecError> {
        (self.finish)(self.state, ctx)
    }
}

/// The step that folds the values of the children of an [`Outcome::Stream`] into a state, one value at a time.
pub struct Fold<'a> {
    state: FoldBox<'a>,
    ordered: bool,
}

impl<'a> Fold<'a> {
    /// Create a fold of the values of the children of a stream into `state`.
    ///
    /// `fold` is called with each value, in the order the children were generated (see
    /// [`unordered`](Fold::unordered)). The outcome of a call is run as a task (e.g. a write of the value) before
    /// `fold` is called with the next value, and its value is discarded.
    /// Once every value is folded, `finish` produces the outcome of the stream.
    pub fn new<S: zarrs_plugin::MaybeSend + 'a>(
        state: S,
        fold: impl FnMut(&mut S, TaskValue<'a>, &ComputeCtx) -> Result<Outcome<'a>, CodecError>
        + zarrs_plugin::MaybeSend
        + 'a,
        finish: impl FnOnce(S, &ComputeCtx) -> Result<Outcome<'a>, CodecError>
        + zarrs_plugin::MaybeSend
        + 'a,
    ) -> Self {
        Self {
            state: Box::new(FoldFns {
                state,
                fold,
                finish,
            }),
            ordered: true,
        }
    }

    /// Fold the values in the order the children complete, rather than the order they were generated.
    #[must_use]
    pub fn unordered(mut self) -> Self {
        self.ordered = false;
        self
    }

    /// Returns true if the values are folded in the order the children were generated.
    pub(crate) fn is_ordered(&self) -> bool {
        self.ordered
    }

    pub(crate) fn step(
        &mut self,
        value: TaskValue<'a>,
        ctx: &ComputeCtx,
    ) -> Result<Outcome<'a>, CodecError> {
        self.state.step(value, ctx)
    }

    pub(crate) fn finish(self, ctx: &ComputeCtx) -> Result<Outcome<'a>, CodecError> {
        self.state.finish(ctx)
    }
}

impl std::fmt::Debug for Fold<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fold")
            .field("ordered", &self.ordered)
            .finish_non_exhaustive()
    }
}

/// The result of the compute step of a task.
///
/// A task completes once its outcome and all of its descendants have completed.
#[derive(Debug)]
pub enum Outcome<'a> {
    /// The task is complete (e.g. it wrote its output into its target).
    Done,
    /// The task is complete with a value for its parent [`Join`].
    Value(TaskValue<'a>),
    /// Run child tasks. The task completes when they complete, without a value.
    Spawn(Vec<Task<'a>>),
    /// Run child tasks, then run `join` on their values.
    /// The task completes with the outcome of the join.
    Fork {
        /// The child tasks.
        children: Vec<Task<'a>>,
        /// The join step.
        join: Join<'a>,
    },
    /// Run child tasks generated on demand, so that only a bounded number of them exist at a time.
    ///
    /// Without a `join`, the task completes without a value when the children complete, as [`Outcome::Spawn`].
    /// With a `join`, it runs on the values of the children in the order they were generated, as [`Outcome::Fork`].
    Generate {
        /// The generator of child tasks.
        children: Generator<'a>,
        /// The join step, if any.
        join: Option<Join<'a>>,
    },
    /// Run child tasks generated on demand, and fold their values one at a time as they complete.
    ///
    /// Unlike [`Outcome::Generate`] with a join, the values need not all be held at once: each is folded as soon as
    /// the values before it (in generation order, unless the fold is [unordered](Fold::unordered)) are folded, and
    /// the outcome of each fold step (e.g. a write) completes before the next. A value waiting for an earlier one is
    /// held, and the children generated but not yet folded are bounded by the generation window of the drive.
    /// The task completes with the outcome of [`Fold`]'s finish step.
    Stream {
        /// The generator of child tasks.
        children: Generator<'a>,
        /// The fold of the values of the children.
        fold: Fold<'a>,
    },
    /// Perform writes, then complete with the value `then`.
    Write {
        /// The write operations, performed concurrently, or in sequence if any is a step of a multipart write.
        ops: SmallVec<[WriteOp<'a>; 1]>,
        /// The value of the task once the writes complete.
        then: TaskValue<'a>,
    },
}

/// A value produced by a task for the [`Join`] of its parent.
#[derive(Debug, Default)]
pub enum TaskValue<'a> {
    /// No value.
    #[default]
    None,
    /// Array bytes (e.g. decoded data of a variable-length data type).
    Array(ArrayBytes<'a>),
    /// Bytes, or [`None`] if absent (e.g. an encoded chunk that is entirely the fill value).
    Bytes(Option<CowBytes<'a>>),
    /// A prepared decoder (see [`PrepareDecode`](super::PrepareDecode)).
    Prepared(std::sync::Arc<dyn super::PreparedDecoder>),
    /// Any other value.
    Any(Box<dyn AnyValue>),
}

impl<'a> TaskValue<'a> {
    /// The array bytes of a [`TaskValue::Array`].
    #[must_use]
    pub fn into_array(self) -> Option<ArrayBytes<'a>> {
        match self {
            Self::Array(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// The bytes of a [`TaskValue::Bytes`].
    #[must_use]
    pub fn into_bytes(self) -> Option<Option<CowBytes<'a>>> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// The prepared decoder of a [`TaskValue::Prepared`].
    #[must_use]
    pub fn into_prepared(self) -> Option<std::sync::Arc<dyn super::PreparedDecoder>> {
        match self {
            Self::Prepared(decoder) => Some(decoder),
            _ => None,
        }
    }

    /// The value of a [`TaskValue::Any`] if it is a `T`.
    #[must_use]
    pub fn downcast<T: Any>(self) -> Option<T> {
        match self {
            Self::Any(any) => any.downcast().ok().map(|value| *value),
            _ => None,
        }
    }
}

/// A value of any type, which is [`Send`] and [`Sync`] on targets with threads.
pub trait AnyValue: Any + zarrs_plugin::MaybeSend + zarrs_plugin::MaybeSync {}

impl<T: Any + zarrs_plugin::MaybeSend + zarrs_plugin::MaybeSync> AnyValue for T {}

impl dyn AnyValue {
    /// Downcast the value to a concrete type.
    ///
    /// # Errors
    /// Returns the value if it is not a `T`.
    pub fn downcast<T: Any>(self: Box<Self>) -> Result<Box<T>, Box<dyn Any>> {
        let any: Box<dyn Any> = self;
        any.downcast()
    }
}

impl std::fmt::Debug for dyn AnyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("AnyValue")
    }
}

/// A storage write operation.
#[derive(Debug)]
pub enum WriteOp<'a> {
    /// Store bytes.
    Set(SourceKey, CowBytes<'a>),
    /// Store bytes at offsets of an existing value.
    SetPartial(SourceKey, Vec<(u64, CowBytes<'a>)>),
    /// Erase the value.
    Erase(SourceKey),
    /// A step of the multipart write with an identifier.
    ///
    /// The steps of a multipart write must be performed in order. The multipart steps of an [`Outcome::Write`] are
    /// performed in sequence (with its other writes), and the task that produces them must not produce later steps of
    /// the same write until it completes (as the fold of an [`Outcome::Stream`] does not).
    Multipart(MultipartId, MultipartOp<'a>),
}

impl WriteOp<'_> {
    /// The key written, or [`None`] for a step of a multipart write other than [`MultipartOp::Begin`].
    #[must_use]
    pub fn key(&self) -> Option<SourceKey> {
        match self {
            Self::Set(key, _)
            | Self::SetPartial(key, _)
            | Self::Erase(key)
            | Self::Multipart(_, MultipartOp::Begin(key, _)) => Some(*key),
            Self::Multipart(..) => None,
        }
    }

    /// Returns true if the operation is a step of a multipart write.
    #[must_use]
    pub fn is_multipart(&self) -> bool {
        matches!(self, Self::Multipart(..))
    }
}

/// The identifier of a multipart write ([`WriteOp::Multipart`]).
///
/// Identifiers are unique within the process, so an I/O backend can hold the multipart writes of several drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MultipartId(u64);

impl MultipartId {
    /// A new identifier, distinct from all others.
    #[must_use]
    pub fn unique() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Self(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

/// A step of a multipart write ([`WriteOp::Multipart`]).
///
/// A multipart write begins, appends parts, and then finishes (making the value visible) or aborts.
#[derive(Debug)]
pub enum MultipartOp<'a> {
    /// Begin the write of the value of a key, with an upper bound on the size of the value if known.
    Begin(SourceKey, Option<u64>),
    /// Append parts of the value, in order.
    Parts(Vec<CowBytes<'a>>),
    /// Complete the value.
    Finish,
    /// Abandon the value, leaving any existing value in place.
    Abort,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_without_io_reads_memory() {
        let bytes: Vec<u8> = (0..8).collect();
        let source = ByteSource::memory(CowBytes::from(&bytes[..]))
            .interval(2, 4)
            .unwrap();
        let task = Task::read(
            source,
            smallvec::smallvec![ByteRange::FromStart(1, Some(2)), ByteRange::Suffix(1)],
            |reads, _ctx| {
                let reads: Vec<Vec<u8>> = reads
                    .into_iter()
                    .map(|read| read.unwrap().as_slice().to_vec())
                    .collect();
                assert_eq!(reads, vec![vec![3, 4], vec![5]]);
                Ok(Outcome::Done)
            },
        );
        assert!(!task.needs_io());
        let ctx = ComputeCtx::new(ComputeResources::default());
        assert!(matches!(task.run_without_io(&ctx), Ok(Outcome::Done)));

        let task = Task::read(
            ByteSource::external(SourceKey(0)),
            [ByteRange::Suffix(1)],
            |_, _| Ok(Outcome::Done),
        );
        assert!(task.needs_io());
        assert!(!Task::compute(|_| Ok(Outcome::Done)).needs_io());
    }

    #[test]
    fn join_receives_values() {
        let join = Join::new(|values, _ctx| {
            assert!(matches!(values[0], TaskValue::Bytes(None)));
            Ok(Outcome::Value(TaskValue::None))
        });
        let ctx = ComputeCtx::new(ComputeResources::default());
        assert!(matches!(
            join.run(vec![TaskValue::Bytes(None)], &ctx),
            Ok(Outcome::Value(TaskValue::None))
        ));
    }
}
