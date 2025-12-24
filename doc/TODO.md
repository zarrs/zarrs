## TODO

- Optional data types
  - Support optional variable length data in retrieve_multi_chunk_variable and retrieve_multi_chunk_variable_impl
  - Example optional data
  - Partial encoding and decoding
  - Proper fill value handling in encode/decode?

### Features
- Add array methods supporting advanced indexing <https://github.com/zarrs/zarrs/issues/52>
- Stabilise the async `Array` API <https://github.com/zarrs/zarrs/issues/127>
- Stabilise the partial encoding API
- Stabilise the data type API
  - Support data type fallbacks
- Stabilise the codec API and move into the `zarrs_codec` crate
- Stabilise the chunk grid API and move into the `zarrs_chunk_grid` crate

### Ergonomics
- Remove most/all `_opt` methods when Rust [`import-trait-associated-functions`](https://github.com/rust-lang/rust/issues/134691) stabilises
- Move array `store_`/`retrieve_` variants into `Array` `Ext` traits, and remove `async_` prefix?

### Performance
- More codec parallelism (where efficient) <https://github.com/zarrs/zarrs/issues/128>
- Optimise the async `Array` API and async partial decoders
  - Test an `io_uring` filesystem store

### Maintenance/Code Quality
- Increase test coverage
- Use the `async_generic` crate to reduce `async` code duplication (pending https://github.com/scouten/async-generic/pull/17) or wait for keyword generics

### Zarr Extensions at [zarr-developers/zarr-extensions]
- Support newly registered extension points <https://github.com/zarrs/zarrs/issues/191>
- Register the following:
  - `bitround` (integer data types)
  - `vlen`/`vlen_v2`: ZEP0007
  - `gdeflate`
  - `squeeze`

[zarr-developers/zarr-extensions]: https://github.com/zarr-developers/zarr-extensions
