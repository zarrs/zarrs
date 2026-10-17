//! Semantics tests shared by all drivers.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use smallvec::smallvec;
use zarrs_storage::byte_range::{ByteRange, extract_byte_ranges};
use zarrs_storage::{Bytes, ReadCoalescing, StorageError};

use super::super::{
    ByteSource, Generator, Join, Outcome, ReadBytes, SourceKey, SyncIo, Task, TaskValue, WriteOp,
};
use crate::{CodecError, CowBytes, Resources};

/// An in-memory I/O backend that counts reads and writes.
#[derive(Default)]
pub(super) struct MemoryIo {
    pub(super) values: Mutex<BTreeMap<SourceKey, Vec<u8>>>,
    pub(super) reads: AtomicUsize,
    pub(super) writes: AtomicUsize,
}

impl MemoryIo {
    pub(super) fn with_values(values: impl IntoIterator<Item = (usize, Vec<u8>)>) -> Self {
        Self {
            values: Mutex::new(
                values
                    .into_iter()
                    .map(|(key, value)| (SourceKey(key), value))
                    .collect(),
            ),
            ..Default::default()
        }
    }

    pub(super) fn get(&self, key: usize) -> Option<Vec<u8>> {
        self.values.lock().unwrap().get(&SourceKey(key)).cloned()
    }
}

impl SyncIo for MemoryIo {
    fn read(
        &self,
        key: SourceKey,
        ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        let values = self.values.lock().unwrap();
        let Some(value) = values.get(&key) else {
            return Ok(None);
        };
        Ok(Some(
            extract_byte_ranges(value, ranges.iter().copied())?
                .into_iter()
                .map(Bytes::from)
                .collect(),
        ))
    }

    fn write(&self, op: WriteOp<'_>) -> Result<(), StorageError> {
        self.writes.fetch_add(1, Ordering::SeqCst);
        let mut values = self.values.lock().unwrap();
        match op {
            WriteOp::Set(key, value) => {
                values.insert(key, value.as_slice().to_vec());
            }
            WriteOp::SetPartial(key, offset_values) => {
                let value = values.entry(key).or_default();
                for (offset, bytes) in offset_values {
                    let offset = usize::try_from(offset).unwrap();
                    if value.len() < offset + bytes.len() {
                        value.resize(offset + bytes.len(), 0);
                    }
                    value[offset..offset + bytes.len()].copy_from_slice(&bytes);
                }
            }
            WriteOp::Erase(key) => {
                values.remove(&key);
            }
        }
        Ok(())
    }
}

/// A driver under test.
pub(super) type Driver =
    for<'a> fn(Vec<Task<'a>>, &dyn SyncIo, &Resources) -> Result<Vec<TaskValue<'a>>, CodecError>;

fn bytes_value(value: &TaskValue<'_>) -> Option<Vec<u8>> {
    match value {
        TaskValue::Bytes(bytes) => bytes.as_ref().map(|bytes| bytes.as_slice().to_vec()),
        _ => panic!("expected bytes, got {value:?}"),
    }
}

/// A task that reads all of `key` and returns the bytes as its value.
fn read_task<'a>(key: usize) -> Task<'a> {
    Task::read(
        ByteSource::external(SourceKey(key)),
        [ByteRange::FromStart(0, None)],
        |mut reads, _| Ok(Outcome::Value(TaskValue::Bytes(reads.pop().unwrap()))),
    )
}

pub(super) fn reads_and_missing_values(drive: Driver) {
    let io = MemoryIo::with_values([(0, vec![1, 2, 3]), (2, vec![4, 5])]);
    let values = drive(
        vec![read_task(0), read_task(1), read_task(2)],
        &io,
        &Resources::default(),
    )
    .unwrap();
    let values: Vec<_> = values.iter().map(bytes_value).collect();
    assert_eq!(values, vec![Some(vec![1, 2, 3]), None, Some(vec![4, 5])]);
    assert_eq!(io.reads.load(Ordering::SeqCst), 3);
}

pub(super) fn windows_resolve_to_root_reads(drive: Driver) {
    let io = MemoryIo::with_values([(0, (0..16).collect())]);
    let source = ByteSource::external(SourceKey(0))
        .strip(2, 2)
        .unwrap()
        .interval(4, 6)
        .unwrap();
    let task = Task::read(
        source,
        smallvec::smallvec![ByteRange::FromStart(1, Some(2)), ByteRange::Suffix(2)],
        |reads, _| {
            let reads: Vec<_> = reads
                .into_iter()
                .map(|read| read.unwrap().as_slice().to_vec())
                .collect();
            assert_eq!(reads, vec![vec![7, 8], vec![10, 11]]);
            Ok(Outcome::Done)
        },
    );
    drive(vec![task], &io, &Resources::default()).unwrap();
    assert_eq!(io.reads.load(Ordering::SeqCst), 1);
}

pub(super) fn fork_join_values_are_ordered(drive: Driver) {
    let io = MemoryIo::with_values((0..8).map(|i| (i, vec![u8::try_from(i).unwrap()])));
    let task = Task::compute(|_| {
        Ok(Outcome::Fork {
            children: (0..8).rev().map(read_task).collect(),
            join: Join::new(|values, _| {
                let values: Vec<u8> = values
                    .iter()
                    .map(|value| bytes_value(value).unwrap()[0])
                    .collect();
                Ok(Outcome::Value(TaskValue::Bytes(Some(CowBytes::from(
                    values,
                )))))
            }),
        })
    });
    let values = drive(vec![task], &io, &Resources::default()).unwrap();
    assert_eq!(bytes_value(&values[0]), Some(vec![7, 6, 5, 4, 3, 2, 1, 0]));
}

pub(super) fn spawned_descendants_complete_before_join(drive: Driver) {
    let io = MemoryIo::default();
    let completed = AtomicUsize::new(0);
    let completed = &completed;
    let leaf = move || {
        Task::compute(move |_| {
            completed.fetch_add(1, Ordering::SeqCst);
            Ok(Outcome::Done)
        })
    };
    // A fork whose children spawn grandchildren, which spawn great-grandchildren
    let task = Task::compute(move |_| {
        Ok(Outcome::Fork {
            children: (0..4)
                .map(|_| {
                    Task::compute(move |_| {
                        Ok(Outcome::Spawn(
                            (0..4)
                                .map(|_| {
                                    Task::compute(move |_| {
                                        Ok(Outcome::Spawn((0..4).map(|_| leaf()).collect()))
                                    })
                                })
                                .collect(),
                        ))
                    })
                })
                .collect(),
            join: Join::new(move |values, _| {
                assert_eq!(values.len(), 4);
                assert_eq!(completed.load(Ordering::SeqCst), 64);
                Ok(Outcome::Value(TaskValue::Bytes(None)))
            }),
        })
    });
    let values = drive(vec![task], &io, &Resources::default()).unwrap();
    assert!(matches!(values[0], TaskValue::Bytes(None)));
}

pub(super) fn nested_forks(drive: Driver) {
    let io = MemoryIo::default();
    fn tree<'a>(depth: u32) -> Task<'a> {
        Task::compute(move |_| {
            if depth == 0 {
                return Ok(Outcome::Value(TaskValue::Bytes(Some(CowBytes::from(
                    vec![1u8],
                )))));
            }
            Ok(Outcome::Fork {
                children: vec![tree(depth - 1), tree(depth - 1)],
                join: Join::new(|values, _| {
                    let sum: u8 = values
                        .iter()
                        .map(|value| bytes_value(value).unwrap()[0])
                        .sum();
                    Ok(Outcome::Value(TaskValue::Bytes(Some(CowBytes::from(
                        vec![sum],
                    )))))
                }),
            })
        })
    }
    let values = drive(vec![tree(6), tree(3)], &io, &Resources::default()).unwrap();
    assert_eq!(bytes_value(&values[0]), Some(vec![64]));
    assert_eq!(bytes_value(&values[1]), Some(vec![8]));
}

pub(super) fn writes_complete_before_value(drive: Driver) {
    let io = MemoryIo::with_values([(1, vec![9, 9, 9, 9])]);
    let data = vec![1u8, 2, 3];
    let task = Task::compute(|_| {
        Ok(Outcome::Fork {
            children: vec![Task::compute(|_| {
                Ok(Outcome::Write {
                    ops: smallvec![
                        WriteOp::Set(SourceKey(0), CowBytes::Borrowed(&data)),
                        WriteOp::SetPartial(SourceKey(1), vec![(1, CowBytes::from(vec![7u8]))]),
                        WriteOp::Erase(SourceKey(2)),
                    ],
                    then: TaskValue::Bytes(None),
                })
            })],
            join: Join::new(|_, _| Ok(Outcome::Done)),
        })
    });
    drive(vec![task, read_task(0)], &io, &Resources::default()).unwrap();
    assert_eq!(io.writes.load(Ordering::SeqCst), 3);
    assert_eq!(io.get(0), Some(vec![1, 2, 3]));
    assert_eq!(io.get(1), Some(vec![9, 7, 9, 9]));
}

/// A single root task that writes once, which a drive performs without a scheduler, then completes with its value.
pub(super) fn single_root_write(drive: Driver) {
    let write = |key: usize| {
        Task::compute(move |_| {
            Ok(Outcome::Write {
                ops: smallvec![WriteOp::Set(SourceKey(key), CowBytes::from(vec![4u8, 5]))],
                then: TaskValue::Bytes(Some(CowBytes::from(vec![6u8]))),
            })
        })
    };
    let io = MemoryIo::default();
    let values = drive(vec![write(3)], &io, &Resources::default()).unwrap();
    assert_eq!(io.writes.load(Ordering::SeqCst), 1);
    assert_eq!(io.get(3), Some(vec![4, 5]));
    assert_eq!(bytes_value(&values[0]), Some(vec![6]));

    // A failed write is an error of the drive
    assert!(drive(vec![write(3)], &ReadOnlyIo, &Resources::default()).is_err());
}

pub(super) fn memory_sources_need_no_io(drive: Driver) {
    let io = MemoryIo::default();
    let bytes: Vec<u8> = (0..32).collect();
    let root = ByteSource::memory(CowBytes::Borrowed(&bytes));
    let tasks = (0..4)
        .map(|i| {
            Task::read(
                root.interval(i * 8, 8).unwrap(),
                [ByteRange::FromStart(0, Some(1))],
                move |reads, _| {
                    let read = reads[0].as_ref().unwrap();
                    assert!(matches!(read, CowBytes::Borrowed(_)));
                    assert_eq!(read.as_slice(), &[u8::try_from(i * 8).unwrap()]);
                    Ok(Outcome::Done)
                },
            )
        })
        .collect();
    drive(tasks, &io, &Resources::default()).unwrap();
    assert_eq!(io.reads.load(Ordering::SeqCst), 0);
}

pub(super) fn errors_cancel_queued_work(drive: Driver) {
    let io = MemoryIo::default();
    let started = AtomicUsize::new(0);
    let started = &started;
    let task = Task::compute(move |_| {
        Ok(Outcome::Fork {
            children: (0..64)
                .map(|i| {
                    Task::compute(move |_| {
                        started.fetch_add(1, Ordering::SeqCst);
                        if i == 0 {
                            Err(CodecError::Other("failed".to_string()))
                        } else {
                            // Children of later tasks are never run once the error is recorded
                            Ok(Outcome::Spawn(
                                (0..64)
                                    .map(|_| {
                                        Task::compute(move |_| {
                                            started.fetch_add(1, Ordering::SeqCst);
                                            Ok(Outcome::Done)
                                        })
                                    })
                                    .collect(),
                            ))
                        }
                    })
                })
                .collect(),
            join: Join::new(|_, _| panic!("the join of a failed fork never runs")),
        })
    });
    let result = drive(vec![task], &io, &Resources::default());
    assert!(matches!(result, Err(CodecError::Other(message)) if message == "failed"));
    assert!(started.load(Ordering::SeqCst) < 64 * 65);
}

pub(super) fn invalid_reads_fail(drive: Driver) {
    let io = MemoryIo::with_values([(0, vec![1, 2, 3])]);
    let task = Task::read(
        ByteSource::external(SourceKey(0)),
        [ByteRange::FromStart(2, Some(4))],
        |_, _| Ok(Outcome::Done),
    );
    assert!(drive(vec![task], &io, &Resources::default()).is_err());
}

pub(super) fn generated_children(drive: Driver) {
    let io = MemoryIo::with_values((0..256).map(|i| (i, vec![u8::try_from(i % 256).unwrap()])));
    let created = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let max_outstanding = AtomicUsize::new(0);
    let (created, completed, max_outstanding) = (&created, &completed, &max_outstanding);
    // A generator with a join receives values in generation order, and children are created on demand
    let task = Task::compute(move |_| {
        Ok(Outcome::Generate {
            children: Generator::new((0..256).map(move |key| {
                let outstanding =
                    created.fetch_add(1, Ordering::SeqCst) + 1 - completed.load(Ordering::SeqCst);
                max_outstanding.fetch_max(outstanding, Ordering::SeqCst);
                Ok(Task::read(
                    ByteSource::external(SourceKey(key)),
                    [ByteRange::FromStart(0, None)],
                    move |mut reads, _| {
                        completed.fetch_add(1, Ordering::SeqCst);
                        Ok(Outcome::Value(TaskValue::Bytes(reads.pop().unwrap())))
                    },
                ))
            })),
            join: Some(Join::new(|values, _| {
                let values: Vec<u8> = values
                    .iter()
                    .map(|value| bytes_value(value).unwrap()[0])
                    .collect();
                assert_eq!(values, (0..=255).collect::<Vec<u8>>());
                Ok(Outcome::Value(TaskValue::Bytes(None)))
            })),
        })
    });
    let values = drive(
        vec![task],
        &io,
        &Resources::default()
            .with_compute_target(2)
            .with_io_target(2),
    )
    .unwrap();
    assert!(matches!(values[0], TaskValue::Bytes(None)));
    assert!(max_outstanding.load(Ordering::SeqCst) <= 64);

    // Nested generators without joins, alongside another root task
    let leaves = AtomicUsize::new(0);
    let leaves = &leaves;
    let task = Task::compute(move |_| {
        Ok(Outcome::Generate {
            children: Generator::new((0..8).map(move |_| {
                Ok(Task::compute(move |_| {
                    Ok(Outcome::Generate {
                        children: Generator::new((0..8).map(move |_| {
                            Ok(Task::compute(move |_| {
                                leaves.fetch_add(1, Ordering::SeqCst);
                                Ok(Outcome::Done)
                            }))
                        })),
                        join: None,
                    })
                }))
            })),
            join: None,
        })
    });
    drive(vec![task, read_task(3)], &io, &Resources::default()).unwrap();
    assert_eq!(leaves.load(Ordering::SeqCst), 64);

    // Errors from a generator fail the drive
    let task = Task::compute(|_| {
        Ok(Outcome::Generate {
            children: Generator::new(
                [
                    Ok(Task::compute(|_| Ok(Outcome::Done))),
                    Err(CodecError::Other("generator failed".to_string())),
                ]
                .into_iter(),
            ),
            join: None,
        })
    });
    assert!(drive(vec![task], &io, &Resources::default()).is_err());
}

/// An I/O backend that cannot write.
struct ReadOnlyIo;

impl SyncIo for ReadOnlyIo {
    fn read(
        &self,
        _key: SourceKey,
        _ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        Ok(None)
    }
}

/// An indexed generator of `len` tasks that write their index to the key of their index, with the index as their value.
pub(super) fn index_writes<'a>(len: usize, data: &'a [u8]) -> Task<'a> {
    Task::compute(move |_| {
        Ok(Outcome::Generate {
            children: Generator::indexed(len, move |i| {
                Ok(Some(Task::compute(move |_| {
                    Ok(Outcome::Write {
                        ops: smallvec![WriteOp::Set(
                            SourceKey(i),
                            CowBytes::Borrowed(&data[i..=i])
                        )],
                        then: TaskValue::Bytes(Some(CowBytes::Borrowed(&data[i..=i]))),
                    })
                })))
            }),
            join: Some(Join::new(move |values, _| {
                assert_eq!(values.len(), len);
                for (i, value) in values.iter().enumerate() {
                    assert_eq!(bytes_value(value), Some(vec![data[i]]));
                }
                Ok(Outcome::Done)
            })),
        })
    })
}

pub(super) fn indexed_writes(drive: Driver) {
    let io = MemoryIo::default();
    let resources = Resources::default()
        .with_compute_target(4)
        .with_io_target(4);
    let data: Vec<u8> = (0..=255).collect();
    drive(vec![index_writes(256, &data)], &io, &resources).unwrap();
    assert_eq!(io.writes.load(Ordering::SeqCst), 256);
    for (i, &byte) in data.iter().enumerate() {
        assert_eq!(io.get(i), Some(vec![byte]));
    }

    // A failed write fails the drive
    assert!(drive(vec![index_writes(256, &data)], &ReadOnlyIo, &resources).is_err());
}

pub(super) fn indexed_children(drive: Driver) {
    let io = MemoryIo::with_values((0..300).map(|i| (i, vec![u8::try_from(i % 256).unwrap()])));
    let resources = Resources::default()
        .with_compute_target(4)
        .with_io_target(4);
    // Values are joined in index order: reads, compute-only tasks, skipped indices and tasks that fork
    let task = Task::compute(|_| {
        Ok(Outcome::Generate {
            children: Generator::indexed(300, |i| {
                Ok(match i % 4 {
                    0 => Some(read_task(i)),
                    1 => Some(Task::compute(move |_| {
                        let byte = u8::try_from(i % 256).unwrap();
                        Ok(Outcome::Value(TaskValue::Bytes(Some(CowBytes::from(
                            vec![byte],
                        )))))
                    })),
                    2 => None,
                    _ => Some(Task::compute(move |_| {
                        Ok(Outcome::Fork {
                            children: vec![read_task(i)],
                            join: Join::new(|mut values, _| {
                                Ok(Outcome::Value(values.pop().unwrap()))
                            }),
                        })
                    })),
                })
            }),
            join: Some(Join::new(|values, _| {
                assert_eq!(values.len(), 300);
                for (i, value) in values.iter().enumerate() {
                    if i % 4 == 2 {
                        assert!(matches!(value, TaskValue::None));
                    } else {
                        assert_eq!(
                            bytes_value(value),
                            Some(vec![u8::try_from(i % 256).unwrap()])
                        );
                    }
                }
                Ok(Outcome::Value(TaskValue::Bytes(None)))
            })),
        })
    });
    let values = drive(vec![task], &io, &resources).unwrap();
    assert!(matches!(values[0], TaskValue::Bytes(None)));
    assert_eq!(io.reads.load(Ordering::SeqCst), 150);

    // Without a join, alongside another root task, and with an empty generator
    let completed = AtomicUsize::new(0);
    let completed = &completed;
    let tasks = vec![
        Task::compute(move |_| {
            Ok(Outcome::Generate {
                children: Generator::indexed(1000, move |_| {
                    Ok(Some(Task::compute(move |_| {
                        completed.fetch_add(1, Ordering::SeqCst);
                        Ok(Outcome::Done)
                    })))
                }),
                join: None,
            })
        }),
        Task::compute(|_| {
            Ok(Outcome::Generate {
                children: Generator::indexed(0, |_| unreachable!()),
                join: Some(Join::new(|values, _| {
                    assert!(values.is_empty());
                    Ok(Outcome::Done)
                })),
            })
        }),
        read_task(3),
    ];
    drive(tasks, &io, &resources).unwrap();
    assert_eq!(completed.load(Ordering::SeqCst), 1000);

    // Errors from planning fail the drive
    let task = Task::compute(|_| {
        Ok(Outcome::Generate {
            children: Generator::indexed(100, |i| {
                if i == 50 {
                    Err(CodecError::Other("planning failed".to_string()))
                } else {
                    Ok(Some(Task::compute(|_| Ok(Outcome::Done))))
                }
            }),
            join: None,
        })
    });
    assert!(drive(vec![task], &io, &resources).is_err());
}

/// An in-memory I/O backend with a read coalescing hint that records the byte ranges of each request.
pub(super) struct CoalescingIo {
    inner: MemoryIo,
    coalescing: Option<ReadCoalescing>,
    pub(super) requests: Mutex<Vec<Vec<ByteRange>>>,
}

impl SyncIo for CoalescingIo {
    fn read(
        &self,
        key: SourceKey,
        ranges: &[ByteRange],
    ) -> Result<Option<ReadBytes>, StorageError> {
        self.requests.lock().unwrap().push(ranges.to_vec());
        self.inner.read(key, ranges)
    }

    fn read_coalescing(&self) -> Option<ReadCoalescing> {
        self.coalescing
    }
}

/// Queued reads of nearby byte ranges of the same key are read with one request, as slices of its bytes.
///
/// For drivers that coalesce queued reads.
pub(super) fn coalesced_reads(drive: Driver, resources: &Resources) {
    let value: Vec<u8> = (0..=255).collect();
    // One read per stride of 10 bytes, and a duplicate of the first
    let cases: [(u64, Option<ReadCoalescing>, &[usize]); 5] = [
        // Adjacent ranges
        (10, Some(ReadCoalescing::new(0, 1024)), &[1]),
        // Gaps of 2 bytes, within the maximum gap
        (12, Some(ReadCoalescing::new(4, 1024)), &[1]),
        // Gaps of 2 bytes, beyond the maximum gap
        (12, Some(ReadCoalescing::new(1, 1024)), &[1; 16]),
        // Up to 40 bytes per coalesced range, in one request
        (10, Some(ReadCoalescing::new(0, 40)), &[4]),
        // No coalescing
        (10, None, &[1; 16]),
    ];
    for (stride, coalescing, expected_ranges) in cases {
        let io = CoalescingIo {
            inner: MemoryIo::with_values([(0, value.clone())]),
            coalescing,
            requests: Mutex::default(),
        };
        let offsets: Vec<u64> = (0..16).map(|i| i * stride).chain([0]).collect();
        let children = offsets
            .iter()
            .map(|&offset| {
                Task::read(
                    ByteSource::external(SourceKey(0)),
                    [ByteRange::FromStart(offset, Some(10))],
                    |mut reads, _| Ok(Outcome::Value(TaskValue::Bytes(reads.pop().unwrap()))),
                )
            })
            .collect();
        let value = &value;
        let offsets = &offsets;
        let task = Task::compute(move |_| {
            Ok(Outcome::Fork {
                children,
                join: Join::new(move |values, _| {
                    let bytes: Vec<CowBytes<'_>> = values
                        .into_iter()
                        .map(|value| match value {
                            TaskValue::Bytes(Some(bytes)) => bytes,
                            value => panic!("expected bytes, got {value:?}"),
                        })
                        .collect();
                    for (bytes, &offset) in std::iter::zip(&bytes, offsets) {
                        let offset = usize::try_from(offset).unwrap();
                        assert_eq!(bytes.as_slice(), &value[offset..offset + 10]);
                    }
                    if stride == 10 && coalescing.is_some_and(|c| c.max_coalesced_bytes() > 40) {
                        // Coalesced reads are slices of one buffer
                        assert!(bytes.windows(2).any(|pair| {
                            pair[1].as_slice().as_ptr()
                                == pair[0].as_slice().as_ptr().wrapping_add(10)
                        }));
                    }
                    Ok(Outcome::Done)
                }),
            })
        });
        drive(vec![task], &io, resources).unwrap();
        let requests = io.requests.into_inner().unwrap();
        let ranges: Vec<usize> = requests.iter().map(Vec::len).collect();
        if expected_ranges.len() == 1 {
            assert_eq!(
                ranges, expected_ranges,
                "{stride} {coalescing:?}: {requests:?}"
            );
        } else {
            // Not coalesced: one request per read, except the duplicate if it was read with the first
            assert!(
                ranges.iter().all(|&ranges| ranges == 1) && ranges.len() >= 16,
                "{stride} {coalescing:?}: {requests:?}"
            );
        }
    }
}

macro_rules! driver_tests {
    ($drive:expr) => {
        #[test]
        fn reads_and_missing_values() {
            crate::plan::driver::tests::reads_and_missing_values($drive);
        }
        #[test]
        fn windows_resolve_to_root_reads() {
            crate::plan::driver::tests::windows_resolve_to_root_reads($drive);
        }
        #[test]
        fn fork_join_values_are_ordered() {
            crate::plan::driver::tests::fork_join_values_are_ordered($drive);
        }
        #[test]
        fn spawned_descendants_complete_before_join() {
            crate::plan::driver::tests::spawned_descendants_complete_before_join($drive);
        }
        #[test]
        fn nested_forks() {
            crate::plan::driver::tests::nested_forks($drive);
        }
        #[test]
        fn writes_complete_before_value() {
            crate::plan::driver::tests::writes_complete_before_value($drive);
        }
        #[test]
        fn single_root_write() {
            crate::plan::driver::tests::single_root_write($drive);
        }
        #[test]
        fn memory_sources_need_no_io() {
            crate::plan::driver::tests::memory_sources_need_no_io($drive);
        }
        #[test]
        fn errors_cancel_queued_work() {
            crate::plan::driver::tests::errors_cancel_queued_work($drive);
        }
        #[test]
        fn indexed_writes() {
            crate::plan::driver::tests::indexed_writes($drive);
        }
        #[test]
        fn indexed_children() {
            crate::plan::driver::tests::indexed_children($drive);
        }
        #[test]
        fn generated_children() {
            crate::plan::driver::tests::generated_children($drive);
        }
        #[test]
        fn invalid_reads_fail() {
            crate::plan::driver::tests::invalid_reads_fail($drive);
        }
    };
}
pub(super) use driver_tests;
