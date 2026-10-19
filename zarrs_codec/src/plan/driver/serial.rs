//! The serial driver.

use super::super::{ComputeCtx, Outcome, SyncIo, Task, TaskValue, WriteOp};
use zarrs_storage::StorageError;

use super::super::ReadBytes;
use super::core::{
    InlineIo, IoRequest, IoSlot, Scheduler, external_key, finish_reads, single_read,
};
use crate::{CodecError, Resources};

/// Execute `tasks` serially on the calling thread.
///
/// Reads, writes and compute steps run one at a time, depth-first.
/// Returns the values of `tasks`.
///
/// # Errors
/// Returns the first error of a read, write, or compute step.
/// Once an error occurs, no further work is started.
pub fn drive_serial<'a>(
    mut tasks: Vec<Task<'a>>,
    io: &dyn SyncIo,
    resources: &Resources,
) -> Result<Vec<TaskValue<'a>>, CodecError> {
    let ctx = ComputeCtx::new(resources.compute().clone());
    let scheduler = if let [_] = tasks.as_slice() {
        let task = tasks.pop().expect("one task");
        match drive_single(task, io, resources, &ctx, (1, 1))? {
            Ok(value) => return Ok(vec![value]),
            Err(scheduler) => scheduler,
        }
    } else {
        Scheduler::new(tasks)
    };
    drive_scheduler(scheduler, io, resources, &ctx)
}

/// Execute `task` as [`drive_serial`], returning its value.
///
/// # Errors
/// Returns the first error of a read, write, or compute step.
pub fn drive_serial_one<'a>(
    task: Task<'a>,
    io: &dyn SyncIo,
    resources: &Resources,
) -> Result<TaskValue<'a>, CodecError> {
    let ctx = ComputeCtx::new(resources.compute().clone());
    match drive_single(task, io, resources, &ctx, (1, 1))? {
        Ok(value) => Ok(value),
        Err(scheduler) => Ok(drive_scheduler(scheduler, io, resources, &ctx)?
            .pop()
            .expect("one root value")),
    }
}

/// Run a single root task on the calling thread, returning its value, or a scheduler with the generation `window` and
/// `parallelism` to continue it if it fans out.
// The scheduler is moved once per drive, which is cheaper than boxing it
#[allow(clippy::result_large_err)]
pub(super) fn drive_single<'a>(
    task: Task<'a>,
    io: &dyn SyncIo,
    resources: &Resources,
    ctx: &ComputeCtx,
    (window, parallelism): (usize, usize),
) -> Result<Result<TaskValue<'a>, Scheduler<'a>>, CodecError> {
    let outcome = if task.needs_io() {
        let (source, ranges, step) = single_read(task)?;
        let result = {
            let _permit = resources.io_permit_blocking();
            io.read(external_key(&source), &ranges)
        };
        step.run(finish_reads(&source, &ranges, result)?, ctx)?
    } else {
        task.run_without_io(ctx)?
    };
    match outcome {
        Outcome::Done => Ok(Ok(TaskValue::None)),
        Outcome::Value(value) => Ok(Ok(value)),
        Outcome::Write { mut ops, then } if ops.len() == 1 => {
            let _permit = resources.io_permit_blocking();
            io.write(ops.pop().expect("one write"))?;
            Ok(Ok(then))
        }
        outcome => Ok(Err(Scheduler::from_root_outcome(
            outcome,
            window,
            parallelism,
        ))),
    }
}

fn drive_scheduler<'a>(
    mut scheduler: Scheduler<'a>,
    io: &dyn SyncIo,
    resources: &Resources,
    ctx: &ComputeCtx,
) -> Result<Vec<TaskValue<'a>>, CodecError> {
    let io = SerialIo { io, resources };
    // One child of a stream at a time, as one task is queued at a time
    scheduler.set_stream_window(1);
    loop {
        scheduler.refill(1);
        if !scheduler.run_one(&io, ctx) {
            break;
        }
    }
    scheduler.finish()
}

/// The I/O of a serial drive, which is all inline.
struct SerialIo<'d> {
    io: &'d dyn SyncIo,
    resources: &'d Resources,
}

impl InlineIo for SerialIo<'_> {
    fn admit(&self) -> Option<IoSlot<'_>> {
        // One storage request is performed at a time
        Some(IoSlot(None))
    }

    fn enabled(&self) -> bool {
        true
    }

    fn read(&self, request: &IoRequest) -> Result<Option<ReadBytes>, StorageError> {
        let _permit = self.resources.io_permit_blocking();
        self.io.read(request.key, &request.ranges)
    }

    fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
        let _permit = self.resources.io_permit_blocking();
        self.io.write(op)
    }
}

#[cfg(test)]
mod serial_tests {
    super::super::tests::driver_tests!(super::drive_serial);
}
