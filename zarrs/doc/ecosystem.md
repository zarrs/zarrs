The Zarr specification is inherently unstable.
It is under active development and new extensions are continually being introduced.

The `zarrs` crate has been split into multiple crates to:
- allow external implementations of stores and extensions points to target a relatively stable API compatible with a range of `zarrs` versions,
- enable automatic backporting of metadata compatibility fixes and changes due to standardisation,
- stay up-to-date with unstable public dependencies (e.g. `opendal`, `object_store`, `icechunk`, etc) without impacting the release cycle of `zarrs`, and
- improve compilation times.

A hierarchical overview of these crates can be found in the [The `zarrs` Book].

#### Core
- [`zarrs`]: The core library for manipulating Zarr hierarchies.
- [`zarrs_metadata`]: Zarr metadata support.
  - Re-exports [`zarrs::metadata`](crate::metadata).
- [`zarrs_metadata_ext`]: Zarr extensions metadata support.
  - Re-exports [`zarrs::metadata_ext`](crate::metadata_ext).
  - [`zarrs_metadata_numpy_time`]: `NumPy` temporal data type metadata support.
- [`zarrs_plugin`]: The plugin API.
  - Re-exports [`zarrs::plugin`](crate::plugin).
- [`zarrs_storage`]: The storage API.
  - Re-exports [`zarrs::storage`](crate::storage).
- [`zarrs_chunk_grid`]: The chunk grid extension API.
  - Re-exports items under [`zarrs::array`](crate::array) and [`zarrs::array::chunk_grid`](crate::array::chunk_grid).
- [`zarrs_chunk_key_encoding`]: The chunk key encoding extension API.
  - Re-exports items under [`zarrs::array`](crate::array) and [`zarrs::array::chunk_key_encoding`](crate::array::chunk_key_encoding).
- [`zarrs_codec`]: The codec extension API.
  - Re-exports items under [`zarrs::array`](crate::array) and [`zarrs::array::codec`](crate::array::codec).
- [`zarrs_data_type`]: The data type extension API.
  - Re-exports items under [`zarrs::array`](crate::array) and [`zarrs::array::data_type`](crate::array::data_type).

#### Stores
- [`zarrs_filesystem`]: A filesystem store.
  - Re-exports [`zarrs::filesystem`](crate::filesystem).
- [`zarrs_object_store`]: [`object_store`] store support.
- [`zarrs_opendal`]: [`opendal`] store support.
- [`zarrs_http`]: A synchronous http store.
- [`zarrs_zip`]: A storage adapter for zip files.
- [`zarrs_icechunk`]: [`icechunk`] store support.
  - `git`-like version control for Zarr hierachies.
  - Read "virtual Zarr datacubes" of archival formats (e.g., [`netCDF4`](https://www.unidata.ucar.edu/software/netcdf/), [`HDF5`](https://www.hdfgroup.org/solutions/hdf5/), etc.) created by [`VirtualiZarr`](https://github.com/zarr-developers/VirtualiZarr) and backed by [`icechunk`].

#### Bindings
- [`zarrs-python`]: A high-performance codec pipeline for [`zarr-python`].
- [`zarrs_ffi`]: A subset of `zarrs` exposed as a C/C++ API.

#### Zarr Metadata Conventions
- [`ome_zarr_metadata`]: A library for OME-Zarr (previously OME-NGFF) metadata.

#### Tools
- [`zarrs_tools`]: Various tools for creating and manipulating Zarr V3 data with the `zarrs` rust crate
  - A reencoder that can change codecs, chunk shape, convert Zarr V2 to V3, etc.
  - Create an [OME-Zarr] hierarchy from a Zarr array.
  - Transform arrays: crop, rescale, downsample, gradient magnitude, gaussian, noise filtering, etc.

### Benchmarks
- [`zarr_benchmarks`]: Benchmarks of various Zarr V3 implementations: `zarrs`, `zarr-python`, `tensorstore`

[The `zarrs` Book]: https://book.zarrs.dev

[`zarrs`]: https://docs.rs/zarrs/latest/zarrs/
[`zarrs_chunk_grid`]: https://docs.rs/zarrs_data_type/latest/zarrs_chunk_grid/
[`zarrs_chunk_key_encoding`]: https://docs.rs/zarrs_chunk_key_encoding/latest/zarrs_chunk_key_encoding/
[`zarrs_codec`]: https://docs.rs/zarrs_data_type/latest/zarrs_codec/
[`zarrs_data_type`]: https://docs.rs/zarrs_data_type/latest/zarrs_data_type/
[`zarrs_metadata`]: https://docs.rs/zarrs_metadata/latest/zarrs_metadata/
[`zarrs_metadata_ext`]: https://docs.rs/zarrs_metadata_ext/latest/zarrs_metadata_ext/
[`zarrs_metadata_numpy_time`]: https://docs.rs/zarrs_metadata_numpy_time/latest/zarrs_metadata_numpy_time/
[`zarrs_plugin`]: https://docs.rs/zarrs_plugin/latest/zarrs_plugin/
[`zarrs_registry`]: https://docs.rs/zarrs_registry/latest/zarrs_registry/
[`zarrs_storage`]: https://docs.rs/zarrs_storage/latest/zarrs_storage/
[`zarrs_filesystem`]: https://docs.rs/zarrs_filesystem/latest/zarrs_filesystem/
[`zarrs_http`]: https://docs.rs/zarrs_http/latest/zarrs_http/
[`zarrs_object_store`]: https://docs.rs/zarrs_object_store/latest/zarrs_object_store/
[`zarrs_opendal`]: https://docs.rs/zarrs_opendal/latest/zarrs_opendal/
[`zarrs_zip`]: https://docs.rs/zarrs_zip/latest/zarrs_zip/
[`zarrs_icechunk`]: https://docs.rs/zarrs_icechunk/latest/zarrs_icechunk/
[`zarrs_ffi`]: https://docs.rs/zarrs_ffi/latest/zarrs_ffi/
[`zarrs-python`]: https://github.com/zarrs/zarrs-python
[`zarr-python`]: https://github.com/zarr-developers/zarr-python
[`zarrs_tools`]: https://docs.rs/zarrs_tools/latest/zarrs_tools/
[`zarr_benchmarks`]: https://github.com/zarrs/zarr_benchmarks
[`ome_zarr_metadata`]: https://docs.rs/ome_zarr_metadata/latest/ome_zarr_metadata/
[`object_store`]: https://docs.rs/object_store/latest/object_store/
[`opendal`]: https://docs.rs/opendal/latest/opendal/
[`icechunk`]: https://github.com/earth-mover/icechunk

[OME-Zarr]: https://ngff.openmicroscopy.org/latest/
