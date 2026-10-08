## Correctness Issues with Past Versions
- `zarrs: 0.20-0.22` `numcodecs.zfpy` codec metadata incorrectly serialised a string representation of the `zfp` mode rather than integer
  - `zarrs` 0.24+ supports deserialising the erroneous metadata
- `zarrs: 0.20.x` Data encoded with `packbits` with a non-zero `first_bit` is incorrectly encoded
- `zarrs: 0.16-0.19` `gdeflate` codec metadata was written with the non-conformant (unregistered) `gdeflate` name
  - `zarrs` 0.24+ reads it for backwards compatibility
- `zarrs: 0.19.x` `numcodecs.fletcher32` codec metadata was written with the non-conformant (unregistered) `fletcher32` name
  - `zarrs` 0.24+ reads it for backwards compatibility
- † `zarrs: 0.19.x` and `zarrs_metadata: <0.3.5`: it was possible for a user to create non-conformant Zarr V2 metadata with `filters: []`
  - Empty filters now always correctly serialise to `null`
  - `zarrs` will indefinitely support reading Zarr V2 data with `filters: []`
  - `zarr-python` shared this bug (see https://github.com/zarr-developers/zarr-python/issues/2842)
- `zarrs: 0.11-0.12` `bz2` codec metadata was written with the non-conformant (unregistered) `bz2` name
  - `zarrs` 0.24+ reads it for backwards compatibility
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
