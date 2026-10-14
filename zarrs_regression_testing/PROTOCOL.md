# The `zarrs_regression_testing` interface

`zarrs_regression_testing` tests data compatibility between the current `zarrs` and previous releases.
It is also the zarrs side of [`zarr_compatibility`](https://github.com/zarrs/zarr_compatibility), which compares Zarr implementations and generates HTML reports, including reports of `--json` results (e.g. for zarrs.dev). That tool uses the commands and JSON described here.

The Rust types are in [`src/schema.rs`](src/schema.rs). Depend on them without building `zarrs`:

```toml
zarrs_regression_testing = { git = "https://github.com/zarrs/zarrs", tag = "zarrs_regression_testing-v0.1.0", default-features = false }
```

The crate follows semantic versioning and is released as `zarrs_regression_testing-vX.Y.Z` tags; it is not published. Breaking changes to the commands or JSON are breaking changes of the crate, and incompatible JSON changes increment `schema`.

## Commands

| Command | Output |
| --- | --- |
| `zarrs_regression_testing [--all] [--seed N] [--samples N] [--filter S] --json PATH` | Results of current zarrs vs the latest release (or all releases with `--all`) |
| `zarrs_regression_testing cases [--seed N] [--samples N] [--filter S] [--out PATH]` | The case manifest, without running it |
| `zarrs_regression_testing serve --zarrs current\|0.NN` | The adapter protocol on stdin/stdout |
| `zarrs_regression_testing build [0.NN ...]` | Builds release helpers (run it before serving a release concurrently) |

Helpers and work directories go in `$ZARRS_REGRESSION_TESTING_DIR` if it is set, and otherwise under the cargo target directory.

## Array data

```json
{"bytes": [1, 0, 255, 255], "offsets": null, "masks": []}
```

- `bytes`: the element bytes in C order. Multi-byte elements are **little-endian**.
- `offsets`: for variable-length data types (`string`, `bytes`), the start offset of each element plus the end offset; otherwise `null`.
- `masks`: for `optional` data types, one validity mask per level of nesting, outermost first. `0` means null. The bytes of a null element are unspecified.

## Adapter protocol

An adapter reads a JSON array of requests from stdin. It writes a JSON array to stdout with one response per request, in order. Requests are handled in order, because a read can depend on an earlier write. The array is `/array` in the filesystem store at `path`.

```json
[
  {"op": "info"},
  {"op": "write", "path": "/work/zarr-python/0", "metadata": {"zarr_format": 3, "...": "..."}, "shape": [2, 3], "data": {"bytes": [], "offsets": null, "masks": []}},
  {"op": "read", "path": "/work/zarrs/0", "shape": [2, 3]}
]
```

Responses:

| Response | Meaning |
| --- | --- |
| `{"Ok": null}` | Write succeeded |
| `{"Ok": <data>}` | Read succeeded |
| `{"Err": "message"}` | The implementation failed, e.g. it rejected the metadata or failed to decode |
| `{"Unsupported": "message"}` | The adapter cannot express the request, e.g. a data type with no numpy equivalent. This does not count against the implementation. |
| `{"Info": {"implementation": "zarr-python", "version": "3.1.3", "notes": "..."}}` | The response to `info` |

A write creates the array with exactly `metadata`, where the implementation's API allows that, and then writes `data` to the whole array.

## Manifest and results

A manifest (from `cases`) contains:
- `schema`;
- `meta`: the generator, date, seed, samples, filter and a reproduce command;
- `codecs` and `data_types`: labels, with the zarrs release that introduced each one;
- `combinations`: indices of a codec and a data type;
- `cases`: each has a combination, shape, chunk shape, metadata, data, formatted `elements`, and `lossy`.

Results (from `--json`) are a manifest plus:
- `subjects`: `{implementation, version, label, reference}`. The reference comes first, then the others, newest first.
- `outcomes`: one entry per case:
  - `writes`: one per subject, each `"ok"`, `{"error": msg}` or `{"unsupported": msg}`;
  - `reads`: `{writer, reader, status}`, where `status` is `"ok"`, `"not_written"`, `{"fail": msg}` or `{"unsupported": msg}`;
  - `non_conformances`: one list per subject.

Lossy cases (`lossy: true`) are compared against the writer's own decoding of the data, not the input data.
