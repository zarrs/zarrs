//! The serial driver.

use super::super::{ComputeCtx, SyncIo, Task, TaskValue, WriteOp};
use zarrs_storage::StorageError;

use super::super::ReadBytes;
use super::core::{InlineIo, IoRequest, IoSlot, Scheduler, SingleTask, single_outcome};
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
    let ctx = ComputeCtx::new(resources.clone());
    let scheduler = if let [_] = tasks.as_slice() {
        let task = tasks.pop().expect("one task");
        match drive_single(task, io, resources, &ctx)? {
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
    let ctx = ComputeCtx::new(resources.clone());
    match drive_single(task, io, resources, &ctx)? {
        Ok(value) => Ok(value),
        Err(scheduler) => Ok(drive_scheduler(scheduler, io, resources, &ctx)?
            .pop()
            .expect("one root value")),
    }
}

/// Run a single root task on the calling thread, returning its value, or a scheduler to continue it if it fans out.
// The scheduler is moved once per drive, which is cheaper than boxing it
#[allow(clippy::result_large_err)]
pub(super) fn drive_single<'a>(
    task: Task<'a>,
    io: &dyn SyncIo,
    resources: &Resources,
    ctx: &ComputeCtx,
) -> Result<Result<TaskValue<'a>, Scheduler<'a>>, CodecError> {
    let outcome = match SingleTask::new(task)? {
        SingleTask::Compute(task) => task.run_without_io(ctx)?,
        SingleTask::Read(read, request) => {
            let result = {
                let _permit = resources.io_permit_blocking();
                io.read(request.key, &request.ranges)
            };
            read.run(&request, result, ctx)?
        }
    };
    Ok(single_outcome(outcome))
}

fn drive_scheduler<'a>(
    mut scheduler: Scheduler<'a>,
    io: &dyn SyncIo,
    resources: &Resources,
    ctx: &ComputeCtx,
) -> Result<Vec<TaskValue<'a>>, CodecError> {
    let io = SerialIo { io, resources };
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
