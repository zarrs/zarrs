## Correctness Issues with Past Versions
- `zarrs: <0.24` `must_understand: false` was dropped from metadata (e.g. codecs) with an empty or no `configuration`
  - Implementations that do not support the extension fail to open these arrays, rather than ignoring it
  - `zarrs` 0.24+ preserves it
- `zarrs: 0.19-0.23` `numcodecs.fletcher32` checksums of data with an odd length omitted the last byte
  - These chunks fail checksum validation in other Zarr implementations (e.g. `numcodecs`)
  - `zarrs` 0.24+ accepts these checksums for backwards compatibility
- `zarrs: 0.20-0.23` the `numcodecs.fixedscaleoffset` codec was not computed as in `numcodecs`, so encoded data could differ
  - Values out of the range of the data type but in the range of `astype` (or vice versa) were saturated before the transform
  - Encoded values were rounded with ties away from zero rather than to even (`numpy.around`)
  - 8 and 16-bit integer data was computed in `f32`, losing precision
- `zarrs: 0.18-0.23` it was possible to create non-conformant arrays with the `vlen-bytes` codec and a data type other than `bytes`, or the `vlen-utf8` codec and a data type other than `string`
  - These arrays fail to be opened by other Zarr implementations (e.g. `zarr-python`)
  - `zarrs` 0.24+ `ArrayBuilder` returns an error for these combinations, but they are still read
- `zarrs: 0.10-0.22` non-conformant metadata was written, which `zarrs` 0.24+ reads for backwards compatibility:
  - `zarrs: 0.20-0.22` `numcodecs.zfpy` codec with a string `mode` rather than an integer
  - `zarrs: 0.19.x` `numcodecs.fletcher32` codec with the unregistered `fletcher32` name
  - `zarrs: 0.19.x` `zarrs.vlen_v2` codec with the unregistered `vlen_v2` name
  - `zarrs: 0.16-0.19` `zarrs.gdeflate` codec with the unregistered `gdeflate` name
  - `zarrs: 0.16-0.18` `bytes` data type with the unregistered `binary` name
  - `zarrs: 0.11-0.12` `numcodecs.pcodec` codec with the unregistered `pcodec` name (legacy configurations written by `zarrs` 0.11-0.15 are also read)
  - `zarrs: 0.11-0.12` `numcodecs.bz2` codec with the unregistered `bz2` name
  - `zarrs: 0.10-0.12` `zfp` codec with the `fixedrate`, `fixedprecision`, and `fixedaccuracy` modes
- `zarrs: 0.20.x` Data encoded with `packbits` with a non-zero `first_bit` is incorrectly encoded
- † `zarrs: 0.19.x` and `zarrs_metadata: <0.3.5`: it was possible for a user to create non-conformant Zarr V2 metadata with `filters: []`
  - Empty filters now always correctly serialise to `null`
  - `zarrs` will indefinitely support reading Zarr V2 data with `filters: []`
  - `zarr-python` shared this bug (see https://github.com/zarr-developers/zarr-python/issues/2842)
- † `zarrs: <0.11.5`: arrays that used the `crc32c` codec have invalid chunk checksums
  - These arrays will fail to be read by Zarr implementations if they validate checksums
  - These arrays can be read by zarrs if the [validate checksums](crate::config::Config#validate-checksums) global configuration option is disabled or the relevant codec option is set explicitly
- † `zarrs: 0.11.2-0.11.3`: the codec configuration of the `crc32c` codec or `bytes` codec (with unspecified endianness) does not conform to the Zarr specification
  - These arrays will fail to be read by other Zarr implementations
  - zarrs still supports reading these arrays, but this may become an error in a future release
  - Fixing these arrays only requires a simple metadata correction, e.g.
    - `sed -i -E "s/(^([ tab]+)\"(crc32c|bytes)\"(,?)$)/\2{ \"name\": \"\3\" }\4/" zarr.json`

## Fixing Erroneous Arrays
Issues marked with † above can be fixed automatically with `zarrs_reencode` in [zarrs_tools](https://github.com/zarrs/zarrs_tools). Example:
```bash
zarrs_reencode --ignore-checksums array.zarr array_fixed.zarr
```
