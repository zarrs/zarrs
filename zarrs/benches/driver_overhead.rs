//! Benchmark the per-task overhead of the drivers of planned tasks.
//!
//! - `compute`: tasks that only compute (no-op).
//! - `read`: tasks that read a value from an in-memory backend.
//! - `fork`: a task that forks into children, joined by a parent.
//! - `offload`: asynchronous tasks that compute on the polling task or on Rayon.
#![allow(missing_docs)]

use std::cell::LazyCell;
use std::collections::BTreeMap;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use zarrs::storage::byte_range::ByteRange;
use zarrs::storage::{Bytes, StorageError};
use zarrs_codec::Resources;
use zarrs_codec::plan::{
    ByteSource, Join, Outcome, ReadBytes, SourceKey, SyncIo, Task, TaskValue, drive_serial,
    drive_sync,
};

const NUM_TASKS: usize = 10_000;

struct MemoryIo(BTreeMap<SourceKey, Bytes>);

impl MemoryIo {
    fn new() -> Self {
        Self(
            (0..NUM_TASKS)
                .map(|i| (SourceKey(i), Bytes::from(vec![0u8; 64])))
                .collect(),
        )
    }
}

impl SyncIo for MemoryIo {
    fn read(
        &self,
        key: SourceKey,
        ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        let Some(value) = self.0.get(&key) else {
            return Ok(None);
        };
        Ok(Some(
            ranges
                .iter()
                .map(|range| value.slice(range.to_range_usize(value.len() as u64)))
                .collect(),
        ))
    }
}

fn compute_tasks<'a>() -> Vec<Task<'a>> {
    (0..NUM_TASKS)
        .map(|i| {
            Task::compute(move |_| {
                black_box(i);
                Ok(Outcome::Done)
            })
        })
        .collect()
}

fn read_tasks<'a>() -> Vec<Task<'a>> {
    (0..NUM_TASKS)
        .map(|i| {
            Task::read(
                ByteSource::external(SourceKey(i)),
                [ByteRange::FromStart(0, None)],
                |reads, _| {
                    black_box(reads);
                    Ok(Outcome::Done)
                },
            )
        })
        .collect()
}

fn fork_task<'a>() -> Task<'a> {
    Task::compute(|_| {
        Ok(Outcome::Fork {
            children: (0..NUM_TASKS)
                .map(|_| Task::compute(|_| Ok(Outcome::Value(TaskValue::None))))
                .collect(),
            join: Join::new(|values, _| {
                black_box(values);
                Ok(Outcome::Done)
            }),
        })
    })
}

type Driver = for<'a> fn(
    Vec<Task<'a>>,
    &dyn SyncIo,
    &Resources,
) -> Result<Vec<TaskValue<'a>>, zarrs_codec::CodecError>;

fn drivers() -> [(&'static str, Driver); 2] {
    [("serial", drive_serial), ("sync", drive_sync)]
}

fn bench_drivers(c: &mut Criterion) {
    let io = LazyCell::new(MemoryIo::new);
    let resources = LazyCell::new(Resources::default);
    let mut group = c.benchmark_group("driver_overhead");
    group.throughput(Throughput::Elements(NUM_TASKS as u64));
    group.bench_function("baseline/closures", |b| {
        b.iter(|| {
            let closures: Vec<Box<dyn FnOnce() -> usize>> = (0..NUM_TASKS)
                .map(|i| Box::new(move || black_box(i)) as Box<dyn FnOnce() -> usize>)
                .collect();
            for closure in closures {
                black_box(closure());
            }
        });
    });
    for (name, drive) in drivers() {
        group.bench_function(BenchmarkId::new("compute", name), |b| {
            let io = &*io;
            let resources = &*resources;
            b.iter(|| drive(compute_tasks(), io, resources).unwrap());
        });
        group.bench_function(BenchmarkId::new("read", name), |b| {
            let io = &*io;
            let resources = &*resources;
            b.iter(|| drive(read_tasks(), io, resources).unwrap());
        });
        group.bench_function(BenchmarkId::new("fork", name), |b| {
            let io = &*io;
            let resources = &*resources;
            b.iter(|| drive(vec![fork_task()], io, resources).unwrap());
        });
    }
    group.finish();
}

#[cfg(feature = "async")]
fn bench_offload(c: &mut Criterion) {
    use zarrs_codec::plan::{AsyncIo, drive_async, drive_async_offload};

    struct AsyncMemoryIo(MemoryIo);

    #[async_trait::async_trait]
    impl AsyncIo for AsyncMemoryIo {
        async fn read(
            &self,
            key: SourceKey,
            ranges: &[ByteRange],
        ) -> Result<Option<ReadBytes>, StorageError> {
            self.0.read(key, ranges)
        }
    }

    let io = LazyCell::new(|| AsyncMemoryIo(MemoryIo::new()));
    let resources = LazyCell::new(Resources::default);
    let runtime = LazyCell::new(|| tokio::runtime::Builder::new_multi_thread().build().unwrap());
    let mut group = c.benchmark_group("driver_overhead_async");
    for size in [1024usize, 64 * 1024] {
        group.throughput(Throughput::Bytes((size * 64) as u64));
        let tasks = |compute_bytes: u64| -> Vec<Task<'static>> {
            (0..64)
                .map(move |i| {
                    Task::read(
                        ByteSource::external(SourceKey(i)),
                        [ByteRange::FromStart(0, None)],
                        move |_, _| {
                            // Stand-in for decoding `size` bytes
                            let mut buffer = vec![0u8; size];
                            for (i, byte) in buffer.iter_mut().enumerate() {
                                *byte = (i % 251) as u8;
                            }
                            black_box(buffer);
                            Ok(Outcome::Done)
                        },
                    )
                    .with_hint(zarrs_codec::plan::TaskHint {
                        read_bytes: Some(64),
                        compute_bytes,
                    })
                })
                .collect()
        };
        group.bench_function(BenchmarkId::new("inline", size), |b| {
            let runtime = &*runtime;
            let io = &*io;
            let resources = &*resources;
            b.to_async(runtime)
                .iter(|| async { drive_async(tasks(0), io, resources).await.unwrap() });
        });
        group.bench_function(BenchmarkId::new("offload", size), |b| {
            let runtime = &*runtime;
            let io = &*io;
            let resources = &*resources;
            b.to_async(runtime).iter(|| async {
                drive_async_offload(tasks(u64::MAX), io, resources)
                    .await
                    .unwrap()
            });
        });
    }
    group.finish();
}

#[cfg(feature = "async")]
criterion_group!(benches, bench_drivers, bench_offload);
#[cfg(not(feature = "async"))]
criterion_group!(benches, bench_drivers);
criterion_main!(benches);
