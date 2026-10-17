# Benchmarks

The everyday suites exercise distinct execution paths with a small scenario
catalogue. They use one representative size unless another size explores an
important threshold or small-operation overhead. Correctness tests cover broader
combinations and failure cases; benchmarks do not claim exhaustive branch coverage.

## Run and select scenarios

Run these commands from the workspace root. The `just` recipes enable default
features plus `async`, run benchmark executables sequentially, and default to all
everyday targets. The large filesystem experiment is excluded from `all`.

```bash
# Execute every everyday scenario once, without statistical sampling
just bench-smoke

# Discover scenarios, or list one target
just bench-list
just bench-list deferred_decode_into

# Short measurements of one target or a matching family
just bench-quick deferred_decode_into
just bench-quick array_ops single_chunk

# Normal Criterion measurements of a selected family
just bench array_ops chunk_cache

# Select one exact benchmark; extra arguments go to Criterion
just bench-smoke deferred_decode_into \
  'deferred_decode_into/deferred/u16/native/zstd/contiguous/1048576' --exact

# Explicitly select the large filesystem experiment
just bench sharded_partial_read
```

The second argument filters full benchmark IDs. Use `all` as the first argument to
filter across everyday targets, e.g. `just bench-smoke all sharded`. Supply `''` as
the filter when forwarding only options: `just bench array_ops '' --noplot`.
`--list` and unmatched filters never populate fixtures or execute diagnostic reads.

The modes operate on the same scenario catalogue:

| Mode | Purpose | Settings |
| --- | --- | --- |
| `bench-smoke` | Explore paths and check that scenarios execute | Criterion `--test`: execute each once, no sampling or baseline |
| `bench-quick` | Development performance feedback | 0.5 s warm-up, 1 s measurement, 10 samples, 1,000 bootstrap resamples, no plots, `quick` baseline |
| `bench` | Investigate performance changes | Criterion defaults: 3 s warm-up, 5 s measurement, 100 samples, 100,000 resamples; existing per-group sample counts still apply |
| `bench-list` | Discover/filter scenarios | IDs only; no fixture setup |

Measurement durations are targets, not hard limits: a slow operation must still
run enough times to collect the requested samples. Quick results are less precise;
confirm small changes with normal settings. Quick runs save to `quick` rather than
overwriting the normal `base` baseline. Do not compare measurements from different
modes, machines, feature sets, or thread configurations as if they were equivalent.

Compilation is separate from benchmark execution. The first run builds optimized
executables; changing features can trigger another build. Cargo reuses build
artifacts on subsequent runs. To select features directly:

```bash
# Default features, without the async cases added by the recipes
cargo bench -p zarrs --bench deferred_decode_into -- --test

# Minimal codec/buffer paths, without optional compressors
cargo bench -p zarrs --no-default-features --bench deferred_decode_into -- --test

# Async scheduling cases
cargo bench -p zarrs --features async --bench driver_overhead -- --test
```

## Scenario groups

| Target | Distinct behavior explored | Required optional features |
| --- | --- | --- |
| `array_subset` | Multidimensional index iteration | None |
| `fill_value` | Fill comparison for each element width | None |
| `codecs` | Bytes conversion, blosc encode/decode, compressed/uncompressed sharding | `blosc`, `zstd` |
| `deferred_decode_into` | Immediate/direct/fallback decoding and destination placement | None; enabled `zstd`, `gzip`, `blosc` add cases |
| `zero_copy_paths` | Byte versus typed reads, borrowed versus shared writes, subchunk slicing | None |
| `array_uncompressed` | Whole-array reads/writes, unsharded/sharded | None |
| `array_blosc` | Whole-array compressed reads/writes | `blosc` |
| `array_ops` | Tiny chunks, whole/subset/into reads, partial encoding versus read-modify-write, partial writes, nested sharding, partial decoders, empty/filled caches | `zstd`, `crc32c`; `filesystem` adds partial-write comparisons |
| `driver_overhead` | Closure baseline, serial/sync compute/read/fork, async inline/offload | `async` adds executor cases |
| `io_latency` | Zero/delayed reads, subsets, sharding, coalescing, sync/async and owned/borrowed writes | `zstd`; `async` adds executor cases |
| `sharded_partial_read_memory` | Full, shard-aligned and unaligned reads from a shared 32 MiB fixture | None |
| `sharded_partial_read` | Large filesystem full/aligned/unaligned reads | `filesystem`; explicit target only |

The filesystem experiment is also excluded from plain `cargo bench` by its
`bench = false` target setting. Explicit `--bench sharded_partial_read` selects it;
Cargo's `--benches` or `--all-targets` also overrides that exclusion. It generates
4 GiB of data at `zarrs/benches/data/sharded_partial_read/` on first execution and
reuses it afterwards. Its measurements reflect filesystem caching as well as array
operations. The memory fixture is smaller and does not reproduce filesystem costs.

### Deferred decoding

There are **16 benchmarks with default features**, replacing the former 192-case
Cartesian product. IDs of retained cases are unchanged. All use `u16` and 1 MiB
unless specified below.

| Scenarios | Count | Reason |
| --- | --- | --- |
| Bytes: native/swapped × contiguous/strided | 4 | Immediate copy, swap and destination traversal |
| Zstd: native contiguous, swapped contiguous, swapped strided | 3 | Direct placement, direct placement followed by swap, strided fallback |
| Gzip and blosc: native contiguous | 2 | Different compressor implementations |
| Generic receiver fallback: swapped contiguous; bytes receiver fallback: swapped strided | 2 | Distinct ordinary decode routes |
| Zstd: `u32`, swapped contiguous | 1 | Another element-width implementation |
| Zstd: 1 KiB, native contiguous | 1 | Small-operation overhead |
| Owned-intermediate references: the two `u16` contiguous 1 MiB zstd cases and the 1 KiB case | 3 | Compare routes where direct placement changes the work |

Without optional compressors there are six cases. Zstd adds eight; gzip and blosc
add one each. Binding, encoding, fixture allocation, and validation happen only for
selected cases, outside timing. Per-iteration view construction stays outside
timing using `BatchSize::PerIteration`, so overlapping borrowed views never coexist.

## Repeatable comparisons

Inputs are deterministic. To compare changes, keep the machine, compiler,
features, thread counts and command identical. Run on an idle machine with
consistent power settings; on Linux, optional CPU affinity should cover enough
cores for the selected concurrency. The recipes run executables sequentially.

```bash
RAYON_NUM_THREADS=4 TOKIO_WORKER_THREADS=4 just bench-quick array_ops single_chunk
RAYON_NUM_THREADS=4 TOKIO_WORKER_THREADS=4 just bench io_latency
```

These variables constrain the thread pools, not every source of concurrency:
`Resources` also controls compute and I/O targets. The current-thread Tokio cases
remain single-threaded. Default read dispatch adapts to observed latency; preserve
it when measuring normal behavior, or explicitly select inline/pooled dispatch in
a focused experiment. Record such changes with results.

Fixtures and populated caches are initialized outside timing. Cases named `empty`
intentionally construct a cache inside timing. Whole-array write benchmarks also
intentionally include array construction/allocation; changing that would measure a
different operation. Filesystem runs are warm-cache experiments after warm-up,
not controlled cold-disk measurements. Memory stores have their own allocation,
copying, synchronization and buffer-ownership costs.

For a suspected regression, repeat the same selected benchmark in separate runs
and inspect the confidence intervals and variability. Spend measurement time on
those cases instead of expanding an unrelated size/type matrix. Add a scenario
only when it exercises a distinct branch or answers a concrete performance question.
