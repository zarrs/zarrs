# `hdf5` Codec

- **Type**: Array-to-bytes codec (read-only, subchunking)
- **Status**: Draft / Experimental
- **Name**: `hdf5` (experimental implementations MAY use a namespaced alias such as `zarrs.experimental.hdf5`)

## Motivation

Many scientific archives store multidimensional datasets across one or more HDF5 (or HDF5-backed NetCDF-4) files. The `hdf5` codec enables Zarr V3 arrays to **virtualise** existing HDF5 files directly as Zarr chunks without duplicating or transcoding the underlying data, while exposing each HDF5 file's internal chunking layout as Zarr subchunks for efficient byte-range partial reads and subchunk caching.

Because HDF5 files are multi-object containers (containing sibling datasets, groups, links, attributes, and NetCDF-4 dimension scales), this codec is **exclusively read-only**.

## Configuration

The codec configuration is a JSON object with the following fields (see [`schema.json`](./schema.json)):

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dataset` | `string` | **Yes** | Path to the HDF5 dataset within each HDF5 file (e.g. `"data"` or `"/group/data"`). |
| `chunk_shape` | `array<integer>` | No | Consistent internal HDF5 chunk shape across all outer chunks of the Zarr array (matching `sharding_indexed`'s `chunk_shape` field). Each element MUST be a positive integer, and the length MUST match the dimensionality of the array. |
| `codecs` | `array<CodecMetadata>` | No | Optional Zarr V3 codec pipeline describing the encoding of each internal HDF5 chunk (e.g. `[bytes, numcodecs.shuffle, numcodecs.zlib, numcodecs.fletcher32]`). Allows readers and caches to inspect subchunk codecs statically before opening any HDF5 file. |

### Rationale for Declaring `codecs` (and `chunk_shape`) in Metadata

Although each HDF5 file embeds its own dataset chunk dimensions and Filter Pipeline message (`0x000B`), declaring `codecs` (and `chunk_shape`) up front in Zarr array metadata provides important operational benefits over pure runtime discovery:

1. **Up-front support verification without chunk I/O**: A reader, validator, or catalog can determine solely from `zarr.json`—before opening or fetching any HDF5 chunk—whether it implements every codec required to decode the array (for example, failing fast when an unsupported filter such as `hdf5.szip` is required rather than mid-way through a distributed read).
2. **Pre-built fixed filter pipeline**: Implementations can construct, validate, and bind the subchunk byte/filter codecs once at array-open time—honouring per-chunk `filter_mask` bits by selectively skipping masked filters within that single pre-built pipeline—avoiding per-chunk codec instantiation.
3. **Static planning and subchunk caching**: Knowing `chunk_shape` and `codecs` ahead of chunk loading allows implementations to resolve the global subchunk grid, size memory/concurrency budgets, and configure subchunk caches prior to issuing any storage requests.

Both fields remain optional so the codec can still virtualise heterogeneous archives whose internal chunking or filters vary across files. Because HDF5 chunks carry an out-of-band `filter_mask` in the chunk index and store edge chunks padded to the full `chunk_shape` rather than the clipped logical subchunk shape, raw on-disk HDF5 chunk bytes MUST NOT be exposed for direct decoding with a standalone Zarr subchunk codec chain.

### Example Configuration

```json
{
  "name": "hdf5",
  "configuration": {
    "dataset": "data",
    "chunk_shape": [64, 64, 64],
    "codecs": [
      {
        "name": "bytes",
        "configuration": {
          "endian": "little"
        }
      },
      {
        "name": "numcodecs.shuffle",
        "configuration": {
          "elementsize": 2
        }
      },
      {
        "name": "numcodecs.zlib",
        "configuration": {
          "level": 1
        }
      },
      {
        "name": "numcodecs.fletcher32",
        "configuration": {}
      }
    ]
  }
}
```

## Supported Data Types

Implementations MUST support HDF5 fixed-point integer and IEEE 754 floating-point datatypes corresponding to fixed-size Zarr V3 numeric data types:

- Signed integers: `int8`, `int16`, `int32`, `int64`
- Unsigned integers: `uint8`, `uint16`, `uint32`, `uint64`
- Floating-point: `float32`, `float64`

When decoding, implementations MUST verify that the HDF5 dataset's datatype class, bit width, signedness, and dimensions match the Zarr array's `data_type` and outer chunk shape, and return an error on mismatch.

## Decoding

Each Zarr outer chunk stored in the underlying store is a complete HDF5 file. Implementations SHOULD perform partial decoding via byte-range reads against the store rather than reading the entire HDF5 file into memory.

### 1. Container & Dataset Resolution

1. If the outer chunk key does not exist in storage, every element in the requested region decodes to the Zarr array's `fill_value`.
2. Parse the HDF5 superblock and traverse the group hierarchy to locate the object header for `dataset`.
3. Validate that the HDF5 dataset's dataspace shape equals the outer Zarr chunk shape and that its HDF5 datatype matches the Zarr `data_type` (extracting its byte order: `little` or `big`).
4. Inspect the dataset's **Data Layout** message (`0x0008`) and optional **Filter Pipeline** message (`0x000B`).

### 2. Storage Layouts

- **Chunked (`layout_class = 2`)**:
  - Implementations SHOULD support all HDF5 chunk indexing structures: B-tree v1 (layout v3), Single Chunk, Implicit, Fixed Array, Extensible Array, and B-tree v2 (layout v4).
  - For a requested subset within the outer chunk, identify only the internal HDF5 chunks overlapping the subset and resolve their `(byte_offset, byte_length, filter_mask)` entries from the chunk index.
  - Fetch ONLY the overlapping chunk byte ranges from storage (e.g. via batched byte-range reads).
  - **Unallocated (sparse) chunks**: Any overlapping HDF5 chunk absent from the chunk index decodes to the Zarr `fill_value`.
  - **Edge-chunk padding**: In HDF5, chunks straddling the dataset boundary are always stored padded to the full HDF5 `chunk_shape`. Implementations MUST decode each fetched chunk at its full HDF5 `chunk_shape` through the subchunk codec pipeline and then extract only the in-bounds overlap with the requested region.
  - **Per-chunk `filter_mask`**: Bit `i` of `filter_mask` set to `1` indicates that filter `i` in the HDF5 filter pipeline was skipped when encoding that chunk. Implementations MUST omit masked filters when decoding that chunk.
- **Contiguous (`layout_class = 1`)**:
  - Compute the coalesced byte ranges overlapping the requested subset relative to the contiguous data address, fetch those byte ranges, and decode via `bytes` with the HDF5 datatype's endianness.
- **Compact (`layout_class = 0`)**:
  - Read the inline bytes from the Data Layout message, decode via `bytes` with the HDF5 datatype's endianness, and extract the requested subset.

### 3. Subchunk Grid Exposure

- When `chunk_shape` is specified in the codec configuration, the outer chunk grid can be refined globally into a subchunk grid by tiling each outer chunk with `chunk_shape` (clipped at each outer chunk boundary, analogous to non-divisible `sharding_indexed`). If every outer chunk is evenly divisible by `chunk_shape`, the resulting global subchunk grid is `regular`; otherwise it is `rectilinear`.
- When `chunk_shape` is omitted, the subchunk grid is chunk-local and MUST be discovered from each HDF5 file's Data Layout message upon opening that chunk.
- If `chunk_shape` is specified in metadata and an opened HDF5 file's actual chunk shape differs, implementations SHOULD either return an error or expose the file's actual chunk-local grid depending on strictness.

### 4. HDF5 Filter Pipeline to Zarr Codec Mapping

An HDF5 chunk's on-disk representation is produced by serializing elements in the dataset's byte order (`bytes` codec) followed by the ordered list of filters in the dataset's Filter Pipeline message (`0x000B`). Decoding runs the mapped Zarr codec chain in reverse order.

#### Filters Mapped to Existing Zarr Extensions

| HDF5 Filter ID | HDF5 Filter Name | Zarr V3 Codec Name | Specification Reference |
| --- | --- | --- | --- |
| *(datatype)* | Little / Big endian | `bytes` | [zarr-extensions `codecs/bytes`](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/bytes) |
| `1` | `H5Z_FILTER_DEFLATE` | `numcodecs.zlib` | [zarr-extensions `codecs/numcodecs.zlib`](https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.zlib) |
| `2` | `H5Z_FILTER_SHUFFLE` | `numcodecs.shuffle` | [zarr-extensions `codecs/numcodecs.shuffle`](https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.shuffle) |
| `3` | `H5Z_FILTER_FLETCHER32` | `numcodecs.fletcher32` | [zarr-extensions `codecs/numcodecs.fletcher32`](https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.fletcher32) |
| `307` | BZIP2 (dynamic) | `numcodecs.bz2` | [zarr-extensions `codecs/numcodecs.bz2`](https://github.com/zarr-developers/zarr-extensions/tree/numcodecs/codecs/numcodecs.bz2) |
| `32001` | Blosc (dynamic) | `blosc` | [zarr-extensions `codecs/blosc`](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/blosc) |
| `32015` | Zstd (dynamic) | `zstd` | [zarr-extensions `codecs/zstd`](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/zstd) |

#### Missing Zarr Codec Specifications (Future Extensions)

The following HDF5 built-in and commonly used registered dynamic filters do not currently have registered Zarr V3 extension specifications (or differ in wire framing from existing `numcodecs` codecs). Implementations encountering these filters without a corresponding codec plugin MUST return an error:

| HDF5 Filter ID | HDF5 Filter Name | Proposed Future Codec | Notes / Gap |
| --- | --- | --- | --- |
| `4` | `H5Z_FILTER_SZIP` | `hdf5.szip` | Adaptive Entropy Coding (CCSDS 121.0-B / `libaec`); no Zarr V3 codec spec exists. |
| `5` | `H5Z_FILTER_NBIT` | `hdf5.nbit` | HDF5 bit-packed datatype compression; no Zarr V3 codec spec exists. |
| `6` | `H5Z_FILTER_SCALEOFFSET` | `hdf5.scaleoffset` | HDF5 per-chunk dynamic min-offset + bit-packing; distinct wire format from `numcodecs.fixedscaleoffset`. |
| `32004` | LZ4 (dynamic) | `hdf5.lz4` | HDF5 LZ4 filter wraps blocks with an HDF5-specific 12-byte header + per-block sizes; differs from `numcodecs.lz4`. |
| `32008` | Bitshuffle (dynamic) | `hdf5.bitshuffle` | Bitshuffle + optional LZ4/Zstd with HDF5 filter framing; no Zarr V3 codec spec exists. |
| `32013` | ZFP (dynamic) | `hdf5.zfp` | HDF5 ZFP filter (`H5Z-ZFP`) embeds HDF5 filter metadata/cd_values framing that differs from `zfp` / `numcodecs.zfpy`. |
| `32024` | SZ3 (dynamic) | `hdf5.sz3` | Error-bounded lossy scientific compressor; no Zarr V3 codec spec exists. |

## Encoding (Read-Only)

This codec is **read-only**. Any attempt to encode or partially encode a chunk with the `hdf5` codec MUST fail with an unsupported-operation error. Applications SHOULD also expose virtualised arrays over read-only storage handles, since Zarr V3 write paths may elide all-fill-value chunks by erasing the underlying store key before invoking the codec.

### Rationale

1. **Multi-dataset container contention**: A single HDF5 file routinely stores multiple datasets (e.g. data variables, coordinate arrays, NetCDF-4 dimension scales, auxiliary tables). In Zarr V3, an array-to-bytes codec maps a single array's chunk to an opaque byte object at one `StoreKey`. Overwriting that `StoreKey` from one Zarr array would destroy every sibling dataset, group, link, and attribute in the HDF5 file, and concurrent writes from two Zarr arrays referencing datasets in the same HDF5 file would race.
2. **Rich container metadata**: HDF5 carries file/group/dataset attributes, superblock/object-header versions, dimension scale references, unlimited max dimensions, and allocation/fill policies that are outside the scope of a Zarr codec.
3. **Container-level mutation**: Creating or mutating HDF5 files is better handled by dedicated HDF5/NetCDF writers or container-level converters that own the entire file lifecycle.

## Complete Array Metadata Example

Virtualising two unequally sized HDF5 files (`part_a.h5` of shape `[14, 10, 10]` and `nested/part_b.h5` of shape `[8, 10, 10]`) stacked along `dim_0`:

```json
{
  "zarr_format": 3,
  "node_type": "array",
  "shape": [22, 10, 10],
  "data_type": "int16",
  "chunk_grid": {
    "name": "rectilinear",
    "configuration": {
      "kind": "inline",
      "chunk_shapes": [[14, 8], 10, 10]
    }
  },
  "chunk_key_encoding": {
    "name": "explicit",
    "configuration": {
      "chunks": [
        [[0, 0, 0], "part_a.h5"],
        [[1, 0, 0], "nested/part_b.h5"]
      ]
    }
  },
  "fill_value": 0,
  "codecs": [
    {
      "name": "hdf5",
      "configuration": {
        "dataset": "data",
        "chunk_shape": [4, 6, 6],
        "codecs": [
          { "name": "bytes", "configuration": { "endian": "little" } },
          { "name": "numcodecs.shuffle", "configuration": { "elementsize": 2 } },
          { "name": "numcodecs.zlib", "configuration": { "level": 1 } },
          { "name": "numcodecs.fletcher32", "configuration": {} }
        ]
      }
    }
  ],
  "dimension_names": ["dim_0", "dim_1", "dim_2"]
}
```

## References

- [Zarr V3 Core Specification](https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html)
- [Zarr Extensions Registry (`zarr-developers/zarr-extensions`)](https://github.com/zarr-developers/zarr-extensions)
- [Sharding Indexed Codec (`sharding_indexed`)](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/sharding_indexed)
- [Rectilinear Chunk Grid (`rectilinear`)](https://github.com/zarr-developers/zarr-extensions/tree/main/chunk-grids/rectilinear)
- [Template Chunk Key Encoding (`template`)](https://github.com/zarr-developers/zarr-extensions/tree/main/chunk-key-encodings/template)
- [HDF5 File Format Specification](https://docs.hdfgroup.org/hdf5/latest/_f_m_t3.html)
- [HDF5 Registered Filter Plugins](https://github.com/HDFGroup/hdf5_plugins/blob/master/docs/RegisteredFilterPlugins.md)
