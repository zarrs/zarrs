## TODO

### Features
- Add array methods supporting advanced indexing <https://github.com/zarrs/zarrs/issues/52>
- Stabilise the async `Array` API <https://github.com/zarrs/zarrs/issues/127>
- Stabilise the chunk grid API
- Stabilise the data type API
- Stabilise the codec API and move into the `zarrs_codec` crate
- Optional data types
  - Partial encoding and decoding

### Performance
- More codec parallelism (where efficient) <https://github.com/zarrs/zarrs/issues/128>
- Optimise the async `Array` API and async partial decoders
  - Test an `io_uring` filesystem store

### Maintenance/Code Quality
- Increase test coverage
- Use the `async_generic` crate to reduce `async` code duplication (pending https://github.com/scouten/async-generic/pull/17) or wait for keyword generics

### Zarr Extensions at [zarr-developers/zarr-extensions]
- Register the following:
  - `vlen`/`vlen_v2`: ZEP0007
  - `gdeflate`
  - `squeeze`

[zarr-developers/zarr-extensions]: https://github.com/zarr-developers/zarr-extensions
