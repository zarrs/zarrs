# `explicit` Chunk Key Encoding

- **Type**: Chunk key encoding
- **Status**: Draft / Experimental
- **Name**: `explicit` (experimental implementations MAY use a namespaced alias such as `zarrs.experimental.explicit`)

## Motivation

Standard chunk key encodings (`default`, `v2`, and `template`) compute chunk store keys algorithmically from chunk grid coordinates. When virtualising pre-existing multi-file archives whose filenames do not follow a uniform naming formula (for example, arbitrary prefixes, irregular directory layouts, or date/identifier-stamped filenames), an **explicit** coordinate-to-key mapping allows a Zarr V3 array to address each chunk directly by its exact relative store key.

## Configuration

The configuration is a JSON object with the following field (see [`schema.json`](./schema.json)):

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `chunks` | `array<[array<integer>, string]>` | **Yes** | List of `[chunk_grid_indices, relative_store_key]` pairs. Each `chunk_grid_indices` is an array of non-negative integers matching the array's dimensionality; each `relative_store_key` is a valid Zarr V3 relative store key. |

### Validation Rules

1. Every `chunk_grid_indices` entry MUST have the same dimensionality (matching the array's dimensionality).
2. Duplicate `chunk_grid_indices` entries MUST be rejected.
3. Every `relative_store_key` MUST be non-empty, MUST NOT begin or end with `/`, and MUST NOT contain `//` or `..`.
4. Implementations SHOULD serialize `chunks` sorted lexicographically by `chunk_grid_indices` for deterministic metadata output.

### Example Configuration

```json
{
  "name": "explicit",
  "configuration": {
    "chunks": [
      [[0, 0, 0], "part_a.h5"],
      [[1, 0, 0], "nested/part_b.h5"]
    ]
  }
}
```

## Encoding Chunk Keys

Given `chunk_grid_indices`:

1. If `chunk_grid_indices` does not match the dimensionality of the array / mapping, encoding MUST fail with an incompatible-dimensionality error.
2. Look up `chunk_grid_indices` in `chunks` and return the corresponding `relative_store_key` (joined to the array's store prefix as defined by the Zarr V3 Core Specification).
3. If `chunk_grid_indices` is not present in `chunks`, encoding MUST fail with an error (or, for sparse read-only virtual arrays, callers MAY treat unmapped chunks as missing if explicitly specified by a future optional flag).

## References

- [Zarr V3 Core Specification — Chunk Key Encodings](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-key-encoding)
- [Template Chunk Key Encoding (`template`)](https://github.com/zarr-developers/zarr-extensions/tree/main/chunk-key-encodings/template)
