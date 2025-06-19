# zar<ins>rs</ins>

[![Latest Version](https://img.shields.io/crates/v/zarrs.svg)](https://crates.io/crates/zarrs)
[![zarrs documentation](https://docs.rs/zarrs/badge.svg)][documentation]
![msrv](https://img.shields.io/crates/msrv/zarrs)
[![downloads](https://img.shields.io/crates/d/zarrs)](https://crates.io/crates/zarrs)
[![build](https://github.com/zarrs/zarrs/actions/workflows/ci.yml/badge.svg)](https://github.com/zarrs/zarrs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/zarrs/zarrs/graph/badge.svg?component=zarrs)](https://codecov.io/gh/zarrs/zarrs)
[![DOI](https://zenodo.org/badge/695021547.svg)](https://zenodo.org/badge/latestdoi/695021547)

`zarrs` is a Rust library for the [Zarr] storage format for multidimensional arrays and metadata.

> [!TIP]
> If you are a Python user, check out [`zarrs-python`].
> It includes a high-performance codec pipeline for the reference [`zarr-python`] implementation.

`zarrs` supports [Zarr V3] and a V3 compatible subset of [Zarr V2].
It is fully up-to-date and conformant with the Zarr 3.1 specification with support for:
- all *core extensions* (data types, codecs, chunk grids, chunk key encodings, storage transformers),
- all accepted [Zarr Enhancement Proposals (ZEPs)](https://zarr.dev/zeps/) and several draft ZEPs:
  - ZEP 0003: Variable chunking
  - ZEP 0007: Strings
  - ZEP 0009: Zarr Extension Naming
- various registered extensions from [`zarr-extensions`],
- experimental extensions intended for future registration, and
- user-defined custom extensions and stores.

A changelog can be found [here][CHANGELOG].
Correctness issues with past versions are [detailed here](https://github.com/zarrs/zarrs/blob/main/doc/correctness_issues.md).

Developed at the [Department of Materials Physics, Australian National University, Canberra, Australia].

## Getting Started
- Read the [documentation (docs.rs)](https://docs.rs/zarrs/latest/zarrs/), which details:
  - Zarr version support,
  - array extension support (codecs, data types, chunk grids, etc.),
  - storage support,
  - examples of how to use `zarrs`, and
  - an overview of the `zarrs` ecosystem including supporting crates and Python and C/C++ bindings.
- Read [The `zarrs` Book].
- Review [benchmarks] of `zarrs` and `zarrs-python` compared to [`zarr-python`] and [`tensorstore`].
- Try the command line tools in [`zarrs_tools`]:
  - `zarrs_reencode`: a reencoder that can change codecs, chunk shape, convert Zarr V2 to V3, etc.
  - `zarrs_ome`: create an [OME-Zarr] hierarchy from a Zarr array.
  - `zarrs_filter`: transform arrays: crop, rescale, downsample, gradient magnitude, gaussian, noise filtering, etc.

## Licence
`zarrs` is licensed under either of
 - the Apache License, Version 2.0 [LICENSE-APACHE](./LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
 - the MIT license [LICENSE-MIT](./LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

[CHANGELOG]: https://github.com/zarrs/zarrs/blob/main/CHANGELOG.md
[documentation]: https://docs.rs/zarrs/latest/zarrs/
[benchmarks]: https://github.com/zarrs/zarr_benchmarks
[The `zarrs` Book]: https://book.zarrs.dev

[`zarr-extensions`]: https://github.com/zarr-developers/zarr-extensions/
[`zarrs`]: https://github.com/zarrs/zarrs/tree/main/zarrs
[`zarrs-python`]: https://github.com/zarrs/zarrs-python
[`zarr-python`]: https://github.com/zarr-developers/zarr-python
[`tensorstore`]: https://github.com/google/tensorstore
[`zarrs_tools`]: https://github.com/zarrs/zarrs_tools

[Zarr]: https://zarr.dev
[Zarr V3]: https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
[Zarr V2]: https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
[OME-Zarr]: https://ngff.openmicroscopy.org/latest/

[Department of Materials Physics, Australian National University, Canberra, Australia]: https://physics.anu.edu.au/research/mp/
