# Deferred input for array-to-bytes `decode_into`

## Status and goal

Implemented on the new `perf/deferred-decode-into` branch from `main`.
The previous five-branch performance stack is preserved unchanged.
This document records the selected design, implementation boundaries, and
verification evidence.

Keep one array-to-bytes `decode_into` operation, and let its implementation choose
where an upstream bytes-to-bytes decoder produces the input.

The main optimisation is decoding an encoding pipeline such as
`array -> bytes -> zstd` directly into a caller's preallocated array output:

1. Decompress into the final output.
2. Do nothing for passthrough data, or transform the output in place for
   non-native endianness.

Do not select this path before knowing that the array-to-bytes codec can consume
the target efficiently. Otherwise a generic fallback can copy the target into a
temporary buffer and then copy back, defeating the optimisation.

## Required outcomes

- Remove `ArrayToBytesCodecTraits::decode_in_place`; do not replace it with an
  array-to-bytes support or efficiency query.
- Keep `ArrayToBytesCodecTraits::decode` for ownership-preserving decoding.
- Keep `ArrayToBytesCodecTraits::decode_into` for preallocated output, with
  immediate and deferred input sources.
- Preserve unique `CowBytes` ownership on the ordinary path.
- Unknown array-to-bytes codecs use the ordinary path, not a target-based
  decode-and-copy fallback.
- Preserve the existing specialised `decode_into` implementations for
  noncontiguous views and optional output.
- Keep data-type-level `BytesDataTypeTraits::decode_in_place` and its efficiency
  hint: these are the primitive used by the specialised `bytes` implementation.
- Require no changes to `CowBytes` or `UnsafeCellSlice`.

## Current code touchpoints

- [Array-to-bytes default and API](../zarrs_codec/src/codec_traits/array_to_bytes.rs#L154).
- [Immediate/deferred input and consuming producer](../zarrs_codec/src/array_bytes_decode_into_input.rs#L3).
- [Bytes-to-bytes default](../zarrs_codec/src/codec_traits/bytes_to_bytes.rs#L70-L86).
- [Output target](../zarrs_codec/src/lib.rs#L103-L114) and
  [contiguous mutable view](../zarrs_codec/src/array_bytes_fixed_disjoint_view.rs#L142-L160).
- [Specialised bytes decoder](../zarrs/src/array/codec/array_to_bytes/bytes/bytes_codec.rs#L242).
- [Codec-chain placement](../zarrs/src/array/codec/array_to_bytes/codec_chain.rs#L593).
- [Data-type efficiency hint](../zarrs_data_type/src/codec_traits/bytes.rs#L67-L98).
- [Placement, ownership, nested-chain, and array workflow regressions](../zarrs/tests/deferred_decode_into.rs#L1)
  and [copy-sensitive benchmarks](../zarrs/benches/deferred_decode_into.rs#L1).

## API design

### An immediate or deferred input

Add a concrete, object-safe input type in `zarrs_codec`:

```rust
pub enum ArrayBytesDecodeIntoInput<'a> {
    Bytes(CowBytes<'a>),
    Deferred(BytesDecodeSource<'a>),
}
```

`Bytes` contains the encoded input of the array-to-bytes codec, as today.
`Deferred` describes one pending bytes-to-bytes decode whose result will be that
encoded input.

Change the array-to-bytes method to:

```rust
fn decode_into(
    &self,
    input: ArrayBytesDecodeIntoInput<'_>,
    shape: &[NonZeroU64],
    output_target: ArrayBytesDecodeIntoTarget<'_>,
    options: &CodecOptions,
) -> Result<(), CodecError>;
```

Implement `From<CowBytes<'a>>` so ordinary call sites use `bytes.into()`.
Do not make the trait method generic over `Into`, which would compromise trait
object use.

There is no public `InPlace` sentinel in this design. The receiver must be able
to choose placement *before* the input has been produced. In-place transformation
of the target is an internal step of the specialised `bytes` decoder.

### A single-use producer

`BytesDecodeSource<'a>` contains:

- A borrowed `&'a dyn BytesToBytesCodecTraits`.
- The upstream encoded `CowBytes<'a>`, moved into the source without cloning.
- The upstream decoder's expected `BytesRepresentation`, stored by value.

Give it a constructor and consuming operations equivalent to:

```rust
fn decode(self, options: &CodecOptions) -> Result<CowBytes<'a>, CodecError>;

fn decode_into(
    self,
    output: &mut [u8],
    options: &CodecOptions,
) -> Result<(), CodecError>;
```

These delegate to the existing bytes-to-bytes operations. Consuming the source
ensures only one route executes, preserves unique ownership, and avoids mutable
state for tracking whether a producer has already run.
Do not implement `Clone` for the producer.

Expose the decoded representation for placement checks. Give
`ArrayBytesDecodeIntoInput` a consuming `into_bytes(options)` helper:

- `Bytes` returns its contained value unchanged.
- `Deferred` invokes the producer's ordinary `decode`.

Use a concrete source value, not boxed callbacks, a general lazy-computation
framework, or a trait containing one-shot closures. This producer only needs to
represent the final bytes-to-bytes stage.

### Approved producer cost hint

The receiver also needs to distinguish a specialised upstream direct-output
implementation from the bytes-to-bytes default that decodes and copies.
Rust cannot infer whether a trait method was overridden.

The approved implementation adds a conservative
`BytesToBytesCodecTraits::is_decode_into_efficient() -> bool`, defaulting to
`false`, and expose that information through `BytesDecodeSource`.

Its contract is about cost, not availability: a codec opting in has a direct
output implementation that avoids a full-size decoded intermediate and its
separate copy. It need not avoid decoder state or small workspace allocations.
Both decoding operations remain usable when the hint is `false`.

Audit and opt in the existing direct-output bytes-to-bytes implementations in
their own implementation commits. Do not infer efficiency from codec names.

This hint prevents choosing a copy-then-endianness-swap path when the ordinary
upstream decode followed by the existing fused copy-and-transform would be
better. Without it, the deferred design still fixes the array-to-bytes fallback
copy problem, but cannot promise to avoid that upstream fallback penalty.

## Placement rules

| Receiver and producer | Chosen route |
| --- | --- |
| Ordinary array-to-bytes implementation | Resolve input normally; run its existing decoding into the target |
| `bytes`, efficient producer, suitable target, passthrough or efficient data-type transformation | Producer writes into final target; transform there if necessary |
| `bytes`, producer without the direct-output hint | Resolve input normally; preserve the existing fused copy-and-transform path |
| `bytes`, data type without an efficient target transformation | Resolve input normally; preserve normal data-type decoding |
| Noncontiguous or optional target | Existing ordinary decode-into path |
| Array-to-array stages before the array-to-bytes stage | Existing owned-intermediate chain path |

Do not retry another route after a producer or transform fails. Placement is
chosen before consuming the source or writing the target.

## Default and existing codec implementations

The default array-to-bytes implementation becomes:

```rust
let bytes = input.into_bytes(options)?;
let decoded = self.decode(bytes, shape, options)?;
decode_into_array_bytes_target(&decoded, output_target)
```

This does not unconditionally clone the encoded input or make it static. A
decoder can reclaim a unique input allocation through `CowBytes::into_vec`, and
an identity decoder can retain borrowed or shared input.

Update existing overrides to accept the input type:

- Implementations needing ordinary bytes first resolve the input with
  `into_bytes(options)`, then keep their current specialised output logic.
- Transparent wrappers can forward the input unchanged when their shape,
  representation, and target semantics permit it.
- Preserve the packbits, pcodec, zfp, and zfpy direct-output work; do not replace
  these overrides with the generic decode-and-copy default.

## Specialised `bytes` implementation

Inspect the input and target before consuming the producer.

For a deferred input, select direct output only when all of the following hold:

1. The bound data type and target are fixed-length and non-optional.
2. The target view exposes one contiguous mutable region.
3. The producer's decoded representation is exactly `FixedSize(output.len())`.
4. Checked shape, element-count, and byte-length validation succeeds.
5. The producer advertises an efficient direct-output implementation.
6. The data-type bytes decoder is passthrough for this endianness, or advertises
   an efficient in-place transformation.

Reuse the current validation and endianness preflight where applicable; do not
skip validation to enter the fast path.

Then:

1. Consume the producer with `decode_into(output, options)`.
2. For passthrough, finish without a copy or transformation.
3. Otherwise, invoke the data-type in-place transformation on that same output.

For every other case, resolve the input normally and retain the current
`bytes` copy-and-transform implementation, including its handling of
noncontiguous views.

## Codec-chain integration

Replace the current unconditional direct-output-then-array-in-place branch.

For a chain without array-to-array stages and with bytes-to-bytes stages:

1. Resolve the chain's incoming input if necessary.
2. Decode the outer bytes-to-bytes stages in the existing reverse order.
3. Leave the innermost bytes-to-bytes stage pending: this is `split_first()` in
   encoding order, not the last element of the codec list.
4. Construct a source with that stage, the remaining encoded bytes, and
   `bytes_representations[0]`.
5. Pass `Deferred(source)` to the array-to-bytes codec's `decode_into`.

The chain no longer writes the final target before the receiving codec has
selected placement.

For a chain without either array-to-array or bytes-to-bytes stages, forward the
input unchanged to its array-to-bytes codec. This preserves deferred placement
through transparent nested chains.

Keep the existing owned-intermediate route when array-to-array stages are
present. Do not attempt general fusion across representation-changing stages in
this change.

## Safety and error contracts

- Never manufacture a borrowed `CowBytes` overlapping the writable target.
- Unique reference count does not establish that cell-backed views or ordinary
  slice references cannot alias.
- The direct route obtains mutable bytes from the target and passes that same
  mutable borrow sequentially through production and transformation.
- Preserve the existing contract that the target may be partially written on
  error. Do not add a rollback guarantee.
- Propagate producer and codec errors; do not run the producer a second time.
- Keep variable-length and optional decoding semantics unchanged.
- Pass the original shape, codec options, and correct intermediate bytes
  representation through both routes.

## Regression tests

Use counting and pointer-recording mock codecs, not only round-trip tests.

### Producer and ownership

- Immediate input resolves without cloning.
- Deferred ordinary decode runs once and direct decode runs zero times.
- Deferred direct decode runs once and ordinary decode runs zero times.
- The ordinary receiver sees the same unique allocation returned by the
  producer, including when its `decode` reclaims that allocation for mutation.
- Producer errors propagate without a second invocation.
- Options and intermediate representation are forwarded correctly.

### Placement and output

- Efficient producer plus native-endian `bytes`: producer receives the final
  output pointer; no subsequent copy or transform is needed.
- Efficient producer plus non-native-endian `bytes`: output pointer is the
  same and values match normal decoding.
- Unknown/default array-to-bytes decoder: ordinary producer path is selected;
  no target-to-temporary materialisation is introduced.
- Producer using the default `decode_into`: ordinary path is selected,
  including for non-native endianness.
- Data type without an efficient target transformation: ordinary path.
- Noncontiguous and optional targets: ordinary path with correct placement,
  masks, and untouched elements outside the view.
- Size mismatch, overflow, missing endianness, and malformed encoded bytes:
  errors rather than panics, with existing mutation guarantees.
- Multiple bytes-to-bytes stages: reverse order and representations are correct;
  only the stage adjacent to the array-to-bytes codec is deferred.
- Transparent nested chains forward the deferred source; chains with additional
  decoding stages resolve it exactly once.

Replace the array-to-bytes in-place fallback tests with deferred-source and
placement tests. Keep the data-type in-place tests.

Run focused producer/view tests under Miri with pure-Rust mock codecs.

## Performance evidence

Add copy-sensitive codec and array-read benchmarks, reusing the existing
Criterion setup.

Compare against the ordinary owned-intermediate route, not only the current
unconditionally selected in-place route:

- Native and non-native `u16`/`u32` with `bytes` plus zstd, gzip, and blosc.
- Small and multi-megabyte chunks.
- Contiguous and noncontiguous targets.
- A producer with direct output and one using the default fallback.
- A receiver that reuses unique ownership and one that allocates its result.
- A representative preallocated array read and sharded/subchunk read.

Record throughput and routing/pointer evidence. If allocation measurements are
added, keep them scoped so concurrent tests and decoder-state allocations do not
obscure full-size intermediate buffers.

Acceptance criteria:

- The common direct-output compression plus `bytes` case avoids the final
  decompression intermediate and separate copy.
- Fallback receivers do not gain an encoded-input clone or target-to-temporary
  round trip relative to ordinary decoding.
- Fallback producers retain the ordinary fused copy-and-transform route.
- Correctness and error behaviour match ordinary decoding.

## Atomic implementation and branch history

Work locally and validate before rewriting; keep recovery refs and do not
publish rewritten branches without a separate request.

Keep the new branch's commits small and fold corrections into the commit that
introduces the affected concept:

1. **Correctness prefix:** gdeflate multi-page/malformed-input handling,
   gdeflate allocation bounds, and checked `bytes` shape arithmetic.
2. **Output primitives:** mutable view access, custom copying, bytes-to-bytes
   `decode_into`, the data-type in-place primitive, and streaming read helpers.
3. **Direct-output producers:** separate codec-specific implementations. The
   zstd decoder returns a bounded direct error instead of retrying with a full
   allocating decode when its output is too small or its frame is invalid.
4. **Deferred API:** the input enum, consuming producer, approved cost hint,
   default receiver, and mechanical caller adaptations land together.
5. **Placement:** opt in efficient producers, specialise `bytes`, and integrate
   deferred chain routing in separate reviewable commits.
6. **Other receivers:** preserve direct packbits/pcodec/zfp/zfpy output. Full-width
   packbits forwards deferred input to its equivalent `bytes` receiver before
   materialisation; zfpy forwards to zfp.
7. **Evidence:** benchmarks and this design/verification record.

Do not introduce the redundant array-to-bytes in-place method at any point.
Preserve the previous performance branches rather than rewriting them. Keep
shared test helpers with their first consumers.

Validate:

- All-feature/all-target checks at every resulting atomic commit.
- Workspace all-feature library tests.
- `zarrs_codec` integration tests with and without its async feature.
- Default and no-default-feature `zarrs` library tests.
- Optional compression codec tests and focused Miri tests.
- Formatting, Clippy diagnostics against the baseline, and the benchmark matrix.

Finally verify that the previous branch refs are unchanged, the new history is
reviewable and buildable, the worktree is clean, and the rewritten tip contains
only the intended implementation plus `main`.

## Verification record

- Workspace all-feature library tests pass, including 474 `zarrs` tests
  (two existing tests are ignored).
- All 14 placement integration tests pass, including owned-buffer reuse,
  final-output pointers, false hints, nested chains, full-width packbits,
  optional and transpose paths, errors without retry, and compressed/sharded
  non-native-endian array reads.
- Core producer tests and compile-fail single-consumption/non-`Clone` doctests
  pass. The four core integration tests also pass under Miri with
  `MIRIFLAGS=-Zmiri-ignore-leaks` for Rayon's retained test registry.
- Default/no-default configurations pass the focused tests. Async core and the
  supported wasm codec configuration also compile.
- All 192 default-feature Criterion matrix cases pass correctness smoke testing.
  The ordinary baseline wraps only efficient innermost producers to suppress
  their hint; it preserves the same specialised receiver, including fused
  copy-and-endianness conversion.

Short local measurements for contiguous 1 MiB `u16` plus zstd, with 20 samples,
100 ms warm-up, and 300 ms measurement per case:

| Endianness | Deferred output | Owned-intermediate baseline |
| --- | --- | --- |
| Native | 33.1 microseconds | 54.3 microseconds |
| Non-native | 122.7 microseconds | 147.6 microseconds |

These are smoke measurements on one machine, not portable throughput guarantees.
Fallback controls take the same routing paths and show ordinary measurement
variation; pointer and invocation tests establish the no-extra-materialisation
property rather than timing alone. View construction is excluded from timing;
both routes include the same view-metadata destruction cost.

Reproduce the matrix and the measured subset:

```sh
cargo bench -p zarrs --bench deferred_decode_into -- --test
cargo bench -p zarrs --bench deferred_decode_into -- \
  'u16/(native|non_native)/(zstd|decode_only_owned_fallback|decode_only_bytes_fallback)/contiguous/1048576' \
  --sample-size 20 --warm-up-time 0.1 --measurement-time 0.3 --noplot
```
