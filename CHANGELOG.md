# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `index_location` support to `vlen` codec
  - Add `VlenCodec::with_index_location()`
- Add `numcodecs.adler32` codec
- Add `ChunkCacheTypePartialDecoder`, `ChunkCachePartialDecoderLru{Chunk,Size}Limit[ThreadLocal]`
- Add `Array::storage()` and `Array::with_storage()`
- Add `Array<T>::[async_]readable()` where `T: [Async]ReadableWritableStorageTraits`
- Implement `Clone` for `Error` structs
- Add `{Indices,Chunks,LinerisedIndices,ContiguousIndices,ContiguousLinearisedIndices}IntoIterator` and `Par{Indices,Chunks}IntoIterator`
  - Implement `Into[Parallel]Iterator` for `Indices` and `IntoParallelRefIterator` for `&Indices`
  - Implement `Into[Parallel]Iterator` for `Chunks` and `IntoParallelRefIterator` for `&Chunks`
  - Implement `IntoIterator` for `{Linearised,Contiguous,ContiguousLinearised}Indices`

### Changed
- **Breaking**: Refactor `ArrayBuilder`
  - All fields are now private
  - Add `ArrayBuilder::{new_with_chunk_grid,chunk_grid_metadata,build_metadata,attributes_mut}()`
  - Add `ArrayBuilder{ChunkGrid,DataType,FillValue}`
  - Change `ArrayBuilder::new()` to take a broader range of types for each parameter, and swap order of `chunk_grid`/`data_type`. See below
  ```diff
  ArrayBuilder::new(
      // array shape
      vec![8, 8], // or [8, 8], &[8, 8], etc.
  -   DataType::Float32,
  -   vec![4, 4].try_into()?, // no longer valid
  -   f32::NAN.into(), // no longer valid
  +   // regular chunk shape or chunk grid metadata
  +   vec![4, 4], // or [4, 4], &[4, 4], "{"name":"regular",...}", MetadataV3::new_with_configuration(...)
  +   // data type or data type metadata
  +   DataType::Float32, // or "float32", "{"name":"float32"}", MetadataV3::new("float32").
  +   // fill value or fill value metadata
  +   f32::NAN, // or "NaN", FillValue, FillValueMetadataV3
  )
  .build()
  ```
- **Breaking**: change the `{Array,Chunk}Representation::new[_unchecked]` `fill_value` parameter to take `impl Into<FillValue>` instead of `FillValue`
  ```diff
  -  ChunkRepresentation::new(chunk_shape(), DataType::Float32, 0.0f32.into())?,
  +  ChunkRepresentation::new(chunk_shape(), DataType::Float32, 0.0f32)?,
  ```
- **Breaking**: `Array::set_shape()` now returns a `Result`
  - Previously it was possible to resize an array to a shape incompatible with a `rectangular` chunk grid
- **Breaking**: Refactor `ChunkGridTraits` and `ChunkGridPlugin`, chunk grids are initialised with the array shape
  - `ChunkGrid::from_metadata()` and `{Regular,Rectangular}ChunkGrid::new()` now have an `ArrayShape` parameter
  - Implementations must now implement `ChunkGridTraits::grid_shape()` as `grid_shape_unchecked()` has been removed
  - Add `ChunkGridTraits::array_shape()`
- **Breaking**: `VlenCodec::new()` gains an `index_location` parameter
- **Breaking**: `ArrayShardedExt::inner_chunk_grid_shape()` no longer returns an `Option`
- **Breaking**: change `array::codecs()` to return an `Arc`d instead of borrowed `CodecChain` 
- **Breaking**: Add `size()` method to `{Array,Bytes}PartialDecoderTraits`
- **Breaking**: Refactor the `ChunkCache` trait
  - The previous API supported misuse (e.g. using a chunk cache with different arrays)
  - **Breaking**: Add `retrieve_chunk_subset()` and `array()` methods (required)
  - Add `retrieve_{array_subset,chunks}()` methods with `_elements()` and `_ndarray()` variants (provided)
  - Remove `array` from method parameters, the `ChunkCache` must own an `Arc<Array>` instead. See below
  ```diff
  -  let cache = ChunkCacheEncodedLruChunkLimit::new(50);
  -  array.retrieve_chunk_opt_cached(&cache, &[0, 1], &CodecOptions::default()),
  +  let cache = ChunkCacheEncodedLruChunkLimit::new(array, 50);
  +  cache.retrieve_chunk(&[0, 1], &CodecOptions::default()),
  ```
- **Breaking**: Change `byte_range::extract_byte_ranges_read_seek` to take `T` instead of `&mut T`
- **Breaking**: Auto implement `[Async]BytesPartialDecoderTraits` for `T: AsRef<[u8]> + ...`
- **Breaking**: `Arc` the `ChunkCache` types
- **Breaking**: Change `Contiguous[Linearised]Indices` iterators to include the number of contiguous indices in their `Item`
- Bump `zarrs_metadata_ext` to 0.2.0
- Bump `zarrs_storage` to 0.4.0
- Bump `blosc-src` to 0.3.6
- Move `zarrs_opendal` to a new repository: `zarrs/zarrs_opendal`

### Removed
- **Breaking**: Remove `ArrayChunkCacheExt`. Use the `ChunkCache` methods instead
- **Breaking**: Remove `Par{Chunks,Indices}IteratorProducer`, which were unneeded
- **Breaking**: Remove `[Async]BytesPartialDecoderTraits` implementations for `std::io::Cursor` variants

### Fixed
- Permit data types with empty configurations that do not require one

## [0.21.2] - 2025-06-19

### Added
- Add complex variants of `float4_e2m1fn`, `float6_{e2m3fn,e3m2fn}`, `float8_{e3m4,e4m3,e4m3b11fnuz,e4m3fnuz,e5m2,e5m2fnuz,e8m0fnu}` data types
- Add `float8` feature enabling the use of `float8::F8E4M3` and `float8::F8E5M2` in `Array::*_elements` methods

### Changed
- Cleanup `zarrs` root docs and README
- Bump `bzip2` to 0.6.0
- Bump `criterion` (dev) to 0.6.0
- Bump `zip` (dev) to 4.0.0
- Bump `zarrs_registry` to 0.1.4
- Bump `zarrs_data_type` to 0.3.2
- Bump `zarrs_zip` to 0.2.3

## [0.21.1] - 2025-06-16

### Added
- Add `numpy.datetime64` and `numpy.timedelta64` data type support
  - `chrono` and `jiff` native element types are supported

### Changed
- Add `Other` to `ArrayError` enum
- Bump `zarrs_registry` to 0.1.3
- Bump `zarrs_metadata_ext` to 0.1.1

## [0.21.0] - 2025-06-08

### Added
- Implement `Element` for `&[u8; N]` for `r*` data type
- Add `complex_{bfloat16,float16,float32,float64}` data types
- Add `float4_e2m1fn`, `float6_{e2m3fn,e3m2fn}`, `float8_{e3m4,e4m3,e4m3b11fnuz,e4m3fnuz,e5m2,e5m2fnuz,e8m0fnu}` data types
  - These have no associated `Element`/`ElementOwned` type and cannot be used with `Array::*_elements()` methods
  - Only hex string fill values are supported
- Add `[u]int{2,4}` data types

### Changed
- Sign extend to the nearest byte when decoding in the `packbits` codec

### Fixed
- **Breaking behaviour**: The `bytes` data type now uses the `vlen-bytes` codec rather than `vlen` by default
  - This was intended for 0.20.0 but incorrectly implemented (the changelog has been amended)
- **Breaking**: Resolve bugs in `Group::child_*` methods and remove no-op `recursive` boolean ([#200] by [@jder])
- Fix missing support for `bitround` `[u]int8` partial decoding
- Fix missing support for `pcodec` `{int,uint,float}16` partial decoding
- Fixed `packbits` codec with a non-zero `first_bit`

[#200]: https://github.com/zarrs/zarrs/pull/200

## [0.20.1] - 2025-06-01

### Changed
- Improve `zstd` codec performance

## [0.20.0] - 2025-05-17

### Highlights
- Added support for ZEP0009 — Zarr Extension Naming
- Added support for ZEP0010 — Generic Extensions (Draft as at 2025/05/17)
  - array/group `extensions` metadata, broader `must_understand` support
- Added support for data type extensions
- Add various new codecs

### Added
- Add codecs: `numcodecs.zlib`, `numcodecs.shuffle`, `numcodecs.fixedscaleoffset`, `packbits`, `squeeze`
- Add support for data type extensions
  - The data type extension API is defined in the `zarrs_data_type` crate
  - **Breaking**: `DataType::metadata_fill_value()` is now fallible
  - **Breaking**: `DataType::from_metadata()` now returns a `PluginCreateError` on error instead of `UnsupportedDataTypeError`
  - **Breaking**: `DataType::from_metadata()` has an additional `ExtensionAliasesDataTypeV3` parameter
  - **Breaking**: `DataType::[fixed_]size()` are no longer `const`
  - **Breaking**: Remove `TryFrom<DataTypeMetadataV3>` for `DataType`
  - **Breaking**: Remove `DataType::identifier()`
  - **Breaking**: move the `zarrs::array::{data_type,fill_value}` modules into the `zarrs_data_type` crate
  - **Breaking**: Rename `IncompatibleFillValueError` to `DataTypeFillValueError`
  - **Breaking**: Rename `IncompatibleFillValueMetadataError` to `DataTypeFillValueMetadataError`
  - Add `Extension` variant to `DataType`
  - Add `custom_data_type_{fixed_size,variable_size,uint4,uint12,float8_e3m4}` examples
- Add `zarrs_registry` crate that tracks extension `name`/`id` aliases
  - Add `config::{{codec,data_type}_aliases_{v3,v2}}[_mut]`
- Add `array:codec::{InvalidBytesLengthError,InvalidArrayShapeError,InvalidNumberOfElementsError,SubsetOutOfBoundsError}`
- Add `ArraySubset::inbounds_shape()` (matches the old `ArraySubset::inbounds` behaviour)
- Add `ArrayBytesFixedDisjointView[CreateError]`
- Add `[Async]ArrayDlPackExt` traits that add methods to `Array` for `DLPack` tensor interop (needs `dlpack` feature)
- Add missing `Group::async_child_*` methods
- Add `[Async]{Array,Bytes}PartialDecoderDefault`
- Add `Config::{codec,data_type}_aliases_{v2,v3}[_mut]`
- Add `Async{Array,Bytes}PartialEncoderTraits` and `*CodecTraits::async_partial_encoder()`
- Add `array_subset::ArraySubsetError` [#156] by [@ilan-gold]
- Add `array_subset::IncompatibleOffsetError`
- Add `AsyncArrayShardedReadableExt` and `AsyncArrayShardedReadableExtCache`
- Add `ArrayBytesFixedDisjointViewCreateError::IncompatibleArraySubsetAndShapeError` [#156] by [@ilan-gold]
- Add `CodecError::IncompatibleDimensionalityError` variant [#156] by [@ilan-gold]
- Add `CodecError::{DataTypeExtension,IncompatibleFillValueError,InvalidArrayShape,InvalidNumberOfElements,SubsetOutOfBounds,RawBytesOffsetsCreate,RawBytesOffsetsOutOfBounds}` variants
- Add `ArrayError::{ArrayBytesFixedDisjointViewCreateError,IncompatibleStartEndIndicesError,IncompatibleOffset,DlPackError}` variants
- Add `CodecMetadataOptions` and `ArrayMetadataOptions::codec_options[_mut]`
- Implement `From<T: IntoIterator<Item = Range<u64>>>` for `ArraySubset`

### Changed
- **Breaking Behaviour**: Use the `vlen-utf8` codec by default for `string` data types
  - `zarrs` previously used `vlen`, an experimental codec not supported by other implementations
- **Breaking Behaviour**: Refactor `codec` name handling and `CodecTraits` in alignment with ZEP0009 and the [`zarr-extensions`] repository
  - All "experimental" codecs now use the `zarrs.` prefix (or `numcodecs.` if fully compatible)
  - Add support for aliased codec names
  - Enables pass-through of codecs from Zarr V2 to V3 without converting to a V3 equivalent (if supported)
- **Breaking**: Split the `zarrs_metadata` crate into `zarrs_metadata` (core) and `zarrs_metadata_ext` (extensions)
  - `zarrs_metadata_ext` is re-exported as `zarrs::metadata_ext`
- **Breaking**: Split the `plugin` module to the `zarrs_plugin` crate
  - `zarrs_plugin` is re-exported as `zarrs::plugin`
  - **Breaking**: `Plugin` is now generic over the creation arguments
  - **Breaking**: `StorageTransformerPlugin` now uses a `Plugin`
- **Breaking**: change `ArraySubset::inbounds` to take another subset rather than a shape
- **Breaking**: Change `CodecError::UnexpectedChunkDecodedSize` to an `InvalidBytesLengthError`
- **Breaking**: Make the following safe and change output args to `ArrayBytesFixedDisjointView`:
  - `Array::[async_]retrieve_chunk[_subset]_into`
  - `[Async]ArrayPartialDecoderTraits::partial_decode_into`
  - `ArrayToBytesCodecTraits::decode_into`
  - `zarrs::array::copy_fill_value_into`
  - `zarrs::array::update_array_bytes`
- **Breaking**: change `RawBytesOffsets` into a validated newtype
- **Breaking**: `ArrayBytes::new_vlen()` not returns a `Result` and validates bytes/offsets compatibility
- **Breaking**: `ArrayError` and `CodecErorr` are now marked as non-exhaustive
- **Breaking**: Add `CodecTraits::{identifier,default_name,configuration[_opt]}()`
- **Breaking**: Change the error type of `node::[async_]get_child_nodes()` and `Group::{children,child_*}()` to `NodeCreateError` instead of `StorageError`
- **Breaking**: Refactor `ArrayToArrayCodecTraits`:
  - Rename `compute_encoded_size()` to `encoded_representation()` and add a default implementation
  - Rename `compute_decoded_shape()` to `decoded_shape()`
  - Add `encoded_shape()` and `encoded_fill_value()`
- **Breaking**: Rename `{ArrayToArray,ArrayToBytes,BytesToBytes}CodecTraits::compute_encoded_size()` to `encoded_representation()`
- **Breaking**: Rename `{ArrayToArray,ArrayToBytes,BytesToBytes}CodecTraits::dynamic()` to `into_dyn()`
- **Breaking**: Rename `[Async]ArrayPartial{Encoder,Decoder}Default` to `[Async]ArrayToBytesPartial{Encoder,Decoder}Default`
- **Breaking**: Rename `[Async]BytesPartial{Encoder,Decoder}Default` to `[Async]BytesToBytesPartial{Encoder,Decoder}Default`
- **Breaking**: `ArraySubset::bound` error type changed to `ArraySubsetError` [#156] by [@ilan-gold]
- **Breaking**: `ArraySubset::relative_to` error type changed to `ArraySubsetError`
- **Breaking**: `Group::consolidated_metadata` now returns an owned `ConsolidatedMetadata` instead of a reference
- **Breaking**: Move `zarrs_metadata::v3::UnsupportedAdditionalFieldError` to `zarrs::array::AdditionalFieldUnsupportedError`
- **Breaking**: Move `ArrayMetadataOptions::*experimental_codec_store_metadata_if_encode_only` into `CodecMetadataOptions`
- `ArrayCreateError::DataTypeCreateError` now uses a `PluginCreateError` internally
- Add default implementations for `{ArrayToArray,ArrayToBytes,BytesToBytes}CodecTraits::[async_]partial_{encoder,decoder}`
- **Breaking**: Bump `zarrs_metadata` to 0.5.0
- Bump `thiserror` to 2.0.2
- Bump `lru` to 0.14.0
- Bump `half` to 2.3.1
- Bump `derive_more` to 2.0.0

### Removed
- **Breaking**: Remove `ArraySubset` unchecked methods [#156] by [@ilan-gold]
- **Breaking**: Remove `{Array,Group}::additional_fields[_mut]`
- **Breaking**: Remove `CodecTraits::create_metadata[_opt]()`
- **Breaking**: Remove `Config::experimental_codec_names[_mut]`

### Fixed
- Fixed reserving one more element than necessary when retrieving `string` or `bytes` array elements
- Check offset is valid in `ArraySubset::relative_to`
- Reenable broken compatibility tests since fixed in `zarr-python`/`numcodecs`
- Reject arrays and groups with unsupported `"must_understand": true` extensions
- Allow `{Array,Group}::new_with_metadata()` and `{Array,Group}Builder` to create arrays with `"must_understand": true` additional fields
  - `{Array,Group}::[async_]open[_opt]` continue to fail with additional fields with `"must_understand": true`

[#156]: https://github.com/zarrs/zarrs/pull/156

## [0.20.0-beta.2] - 2025-05-16

## [0.20.0-beta.1] - 2025-05-12

## [0.20.0-beta.0] - 2025-05-03

## [0.19.2] - 2025-02-13

### Changed
- Bump `zarrs_metadata` to 0.3.4 which includes a number of Zarr metadata fixes
  - See the [`zarrs_metadata` CHANGELOG.md](https://github.com/LDeakin/zarrs/blob/main/zarrs_metadata/CHANGELOG.md)

## [0.19.1] - 2025-01-19

### Added
- Document that elements in `ArrayBytes` must be in C-contiguous order

### Changed
- Use new language/library features added between Rust 1.78-1.82 (internal)
- Cleanup root docs and README removing ZEPs table and ecosystem table

### Fixed
- New clippy lints
- Mark `String` and `Bytes` data types as experimental in their docs
- Mark `rectangular` chunk grid as experimental since it is based on a draft ZEP
- Add missing invariant to `[partial_]decode_into` safety docs

## [0.19.0] - 2025-01-10

### Highlights
- `zarr-python` 3.0.0 has been released today!
- The focus of this release has been in maximising compatibility with unstandardised extensions in `zarr-python` (e.g. experimental codecs, consolidated metadata, etc.)

### Added
- Add `ArrayShardedReadableExt::retrieve_encoded_inner_chunk`
- Add `ArrayShardedReadableExt::inner_chunk_byte_range`
- Add `ArrayShardedExt::is_exclusively_sharded`
- Add `ArrayShardedReadableExtCache::array_is_exclusively_sharded`
- Add `Vlen{Array,Bytes,Utf8}Codec`, replacing `VlenV2Codec`
- Add `ZstdCodecConfigurationNumCodecs`
  - Adds support for Zarr V2 `zstd` encoded data created with `numcodecs` < 0.13
- Add support for pcodec `Auto`, `None`, and `TryLookback` delta specs
- Add `Group::[set_]consolidated_metadata`
- Add `Node::consolidate_metadata`
  - Consolidated metadata is not currently used to optimise node hierarchy requests
- Add experimental `fletcher32` checksum codec based on the numcodecs implementation
  - Adds `fletcher32` feature flag
- Add ecosystem compatibility notes in the experimental codecs docs

### Changed
- **Breaking**: Seal `Array` extension traits: `ArraySharded[Readable]Ext` and `ArrayChunkCacheExt`
- **Breaking**: Make `{Array,Bytes}PartialDecoderCache` private
- **Breaking**: Make `Any` a supertrait of partial encoder/decoder traits
- **Breaking**: Add `ArrayError::UnsupportedMethod`
- **Breaking**: Rename `DataType::Binary` to `Bytes` for compatibility with `zarr-python`
- **Breaking**: Make `array::codec::array_to_bytes::bytes::reverse_endianness` private
- **Breaking**: Make `VlenV2Codec` private
- Bump `itertools` to 0.14
- Indicate that `zfp` / `zfpy` codecs have different metadata in codec table
- Use `zarr-python` 3.0.0 in compatibility tests
- **Breaking**: Bump MSRV to 1.82 (17 October, 2024)

### Removed
- Remove support for pcodec `Try{FloatMult,FloatQuant,IntMult}` mode specs
  - These may be reimplemented when supported by `zarr-python`/`numcodecs`

### Fixed
- Cleanup unnecessary lifetime constraints in partial decoders
- Fix `clippy::useless_conversion` lint

## [0.18.3] - 2024-12-30

### Added
- impl `From<Node>` for `NodePath` ([#112] by [@niklasmueboe])
- Add `Group::child[_{group,array}]_paths` ([#112] by [@niklasmueboe])

[#112]: https://github.com/LDeakin/zarrs/pull/112

## [0.18.2] - 2024-12-25

### Added
- functions to get children of Group ([#104] by [@niklasmueboe])
  - adds `Group::[async_]children`, `Group::[async_]child_groups`, `Group::[async_]child_arrays`
- Impl `From<Node>` for `NodeMetadata`

### Changed
- Reduce metadata code duplication in the `Node` module

[#104]: https://github.com/LDeakin/zarrs/pull/104

## [0.18.1] - 2024-12-17

### Changed
- Bump `zfp-sys` to 0.3.0
- Bump `bzip2` to 0.5.0
- Minor readme/ecosystem updates

### Fixed
- Fix `unsafe_op_in_unsafe_fn` lint
- Clarify that the `zstd` codec is draft in docs
- Clarify that the `gdeflate` codec is experimental in docs

## [0.18.0] - 2024-11-23

### Announcements
- [`zarrs-python`](https://github.com/ilan-gold/zarrs-python) was recently released
  - It implements a high-performance codec pipeline for the reference [`zarr-python`](https://github.com/zarr-developers/zarr-python) implementation
  - It is supported by downstream libraries like `dask`
  - See [zarr_benchmarks](https://github.com/LDeakin/zarr_benchmarks) for benchmarks
- [The `zarrs` Book](https://book.zarrs.dev) has been created and is under construction

### Release Highlights
- Experimental partial encoding support (enabling writing shards incrementally)
- Improve Zarr V2 interoperability and conversion

### Added
- Add a `makefile` and simplify `BUILD.md`
- Add chunk-by-chunk update example in `Array` docs
- Add `array::copy_fill_value_into()`
- Add experimental partial encoding support (sync only):
  - Enables writing shards incrementally
  - With `{CodecOptions,Config}::experimental_partial_encoding` enabled, `Array::store_{array,chunk}_subset` will use partial encoding
  - This is an experimental feature for now until it has more comprehensively tested and support is added in the async API
  - Adds `ArrayPartialEncoderTraits`, `BytesPartialEncoderTraits`, `StoragePartialEncoder`, `ArrayPartialEncoderDefault`, `BytesPartialEncoderDefault`
  - **Breaking**: Add `{ArrayToArray,ArrayToBytes,BytesToBytes}CodecTraits::partial_encoder`
- Add `with_` methods to `{Array,Group}MetadataOptions`
- Add `zarr_v2_to_v3` example
- Add `{Array,Group}::to_v3()`
- Add `ShardingCodecBuilder::build_arc()`
- Add `zarrs::version::version_{str,pre}`
- Add "The zarrs Book" and `zarrs-python` to docs

### Changed
- Bump `unsafe_cell_slice` to 0.2.0
- **Breaking**: Change `output` parameter of `decode_into` codec trait methods to `&UnsafeCellSlice<u8>`
- **Breaking**: Add `dynamic()` to all `CodecTraits`
- **Breaking**: Add `options` parameter to `[Async]ArrayPartialDecoderTraits::partial_decode` and remove `partial_decode_opt`
- Make `array::update_array_bytes()` public
- **Breaking**: Bump MSRV to 1.77 (21 March, 2024)
- Bump `zfp-sys` to 0.2.0
- Display `ArraySubset` as a list of ranges
- Relax `output_subset` requirements on `ArrayToBytesCodecTraits::decode_into` and `ArrayPartialDecoderTraits::partial_decode_into`
  - The subset shape and dimensionality no longer has to match, only the number of elements
- Bump `pco` (pcodec) to 0.4
- **Breaking**: Change `experimental_codec_names` config hashmap to `HashMap<String, String>` from `HashMap<&'static str, String>`
- **Breaking**: Add `name` parameter to `vlen_v2` codec constructors
- Register `vlen-array`, `vlen-bytes`, and `vlen-utf8` codecs
- Bump `zarrs_metadata` to 0.2.0
- Bump `zarrs_storage` to 0.3.0
- Bump `zarrs_filesystem` to 0.2.0
- Make `zarrs::version::version_{,major,minor,patch}` const

### Removed
- Remove `async-recursion` dependency
- **Breaking**: Remove `Default` implementation for `VlenV2Codec`

### Fixed
- Fix panics that could occur with with empty byte ranges / empty array subsets in `Array`, `ByteRange` and codec methods

## [0.18.0-beta.0] - 2024-11-15

## [0.17.1] - 2024-10-18

### Added
 - Add `zarrs_icechunk` to ecosystem docs

### Fixed
 - Fix `data_key` encoding on windows (it contained '//')
 - Fix `clippy::needless_lifetimes` lint

## [0.17.0] - 2024-10-02

### Highlights / Major Changes
 - `zarrs` has been split into 3 core crates: `zarrs`, `zarrs_metadata`, and `zarrs_storage`
   - `zarrs_storage` and `zarrs_metadata` are re-exported as the `storage` and `metadata` modules
 - Store implementations have been moved into separate crates: `zarrs_{filesystem,http,object_store,opendal,zip}`
   - `zarrs_filesystem` is re-exported as the `filesystem` module with the `filesystem` feature (enabled by default)
 - Direct IO support for Linux in `FilesystemStore`
 - **Implicit groups are no longer supported**
   - It is the responsibility of the `zarrs` consumer to explicitly write group metadata when creating a hierarchy
 - Codecs must now be `Arc`'d instead of `Box`'d
 - Fixes a performance regression introduced in 0.16 when reading sharded arrays
 - Check the full release notes for all changes. This release has many breaking changes due to items being removed, moved, renamed, and deprecated.

### Added
 - Add `ChunkGridTraits::chunks_in_array_subset()`
 - Add chunk cache support
   - Add `ArrayChunkCacheExt` extension trait for `Array`
   - Add traits: `ChunkCache`, `ChunkCacheType` (implemented by `ChunkCacheType{Encoded,Decoded}`)
   - Add chunk cache implementations: `ChunkCache{En,De}codedLru{Size,Chunk}Limit[ThreadLocal]`
 - [#58](https://github.com/LDeakin/zarrs/pull/58) Add direct I/O support in `FilesystemStore` by [@sk1p]
   - Adds `FilesystemStoreOptions` and `FilesystemStore::new_with_options`
 - [#64](https://github.com/LDeakin/zarrs/pull/64) Add `Array::[async_]store_encoded_chunk` for writing already-encoded chunks by [@sk1p]
 - Add `key()`, `start()`, `value()` to `StoreKeyStartValue`
 - Add `StorageError::MissingMetadata` now that implicit groups are not supported
 - Add `ChunkShape::to_array_shape()`
 - Add `DataTypeMetadataV3`
 - Add `ArrayShardedExt::effective_inner_chunk_shape`
   - This is the effective inner chunk shape (i.e. read granularity) of a sharded array
   - It is equal to the inner chunk shape unless the `transpose` codec precedes `sharding_indexed`
 - **Breaking**: Add `ArrayToArrayCodecTraits::compute_decoded_shape()`
   - Needed for `ArrayShardedExt::effective_inner_chunk_shape`
 - Add `ArrayToBytesCodecTraits::decode_into` and `[Async]ArrayPartialDecoderTraits::partial_decode_into`
   - This revisits the array view API that was removed in 0.16 [#39](https://github.com/LDeakin/zarrs/pull/39), but simpler and less public
   - Resolves a performance regression introduced in 0.16 when decoding sharded arrays with `Array::[async_]retrieve_array_subset_opt`.
 - Add `Array::subset_all()`
 - Add `ArraySubset::to_ranges()`

### Changed
 - **Breaking**: `Arc` instead of `Box` partial decoders
 - Expand `set_partial_values` tests
 - Specialise `set_partial_values` for `MemoryStore`
 - Bump maximum supported `ndarray` version from 0.15 to 0.16
 - **Breaking**: Make `create_chunk_grid_{regular,rectangular}` `pub(crate)` in alignment with other internal create from metadata methods
 - **Breaking**: Bump MSRV to 1.76 (8 February, 2024)
 - [#59](https://github.com/LDeakin/zarrs/pull/59) Add `ReadableWritableStorageTraits` automatically for all implementations by [@dustinlagoy]
 - Remove implicit group support
   - This is a post-acceptance change of Zarr V3: https://github.com/zarr-developers/zarr-specs/pull/292
 - [#63](https://github.com/LDeakin/zarrs/pull/63) Make `StoreKeysPrefixes` constructible by [@sk1p]
 - **Breaking**: Use values of `Metadata{Convert,Erase}Version` instead of references in parameters/return values
 - Restructure `metadata` module
    - Move conversion to `metadata::v2_to_v3`
    - **Breaking**: Remove many re-exports to the root of `metadata`
    - Move various items to the `metadata` module, keeping re-exports in the `array` module
        - `{Array,Chunk}Shape`
        - `Endianness`
        - `ZARR_NAN_{F16, BF16, F32, F64}`
        - `ChunkKeySeparator`
        - `DimensionName`
    - Split `DataType` into `array::DataType` and `metadata::_::DataTypeMetadataV3`
 - **Breaking**: `data_key` in `zarrs::storage` now take a `chunk_key` instead of a `chunk_key_encoding` and `chunk_indices`
 - **Breaking**: Move `metadata::{v2,v3}::{codec,chunk_grid,chunk_key_encoding}` to `metadata::{v2,v3}::array::`
 - **Breaking**: Rename `ArrayMetadataV2DataType` to `DataTypeMetadataV2`
 - **Breaking**: Rename `FillValueMetadata` to `FillValueMetadataV3`
 - Move `crate::byte_range` into `crate::storage::byte_range` module, add re-export
 - Add `impl TryInto<StorePrefix> for &NodePath`
   - Removes `TryFrom<&NodePath> for StorePrefix`
 - **Breaking**: Move `extract_byte_ranges_read[_seek]` from `array::codec` to `storage::byte_range`
 - **Breaking**: Move `storage::[async_]{get_child_nodes,node_exists,node_exists,listable}` to `node` module
 - **Breaking**: Move `storage::{meta_key*,data_key}` into `node` module
 - Split the crate into multiple crates:
   - `zarrs`
   - `zarrs_storage` (re-exported as `storage` module)
   - `zarrs_metadata` (re-exported as `metadata` module)
   - `zarrs_filesystem` (re-exported as `filesystem` module with `filesystem` feature, enabled by default)
   - `zarrs_http`
   - `zarrs_object_store`
   - `zarrs_opendal`
   - `zarrs_zip`
 - **Breaking**: Codecs no longer need to implement `Clone` but must be in `Arc` instead of `Box`
 - **Breaking**: Change `Contiguous[Linearised]Indices` iterators to return indices only
 - **Breaking**: Remove lifetime constraints in partial decoder API by utilising `Arc`'d codecs
 - **Breaking**: `ShardingCodec::new` `CodecChain` parameters now must be in an `Arc`
 - **Breaking**: Change `UsageLogStorageTransformer` to `UsageLogStorageAdapter` and move to `zarrs_storage`
 - **Breaking**: Change `PerformanceMetricsStorageTransformer` to `PerformanceMetricsStorageAdapter` and move to `zarrs_storage`
 - **Breaking**: Storage transformer refactor:
   - Add `StorageTransformerPlugin` for storage transformer registration instead of generic `Plugin`
   - Remove several `StorageTransformerExtension` trait methods no longer needed
   - Change `StorageTransformerExtension::create_metadata` to return `MetadataV3` instead of an `Option`
   - `StorageTransformerExtension::[async_]create_*_transformer` methods are now fallible
   - `StorageTransformerExtension::async_create_*_transformer` methods are now async
   - `StorageTransformerChain::from_metadata` and `try_create_storage_transformer` now has a `path: &NodePath` parameter

### Removed
 - **Breaking**: Remove `array::NonZeroError`, use `std::num::TryFromIntError` instead
 - **Breaking**: Move storage transformers from the `storage` to the `array` module
 - **Breaking**: Remove many functions in the storage module:
    - `[async_]create_{array, group}`
    - `[async_]erase_{chunk,metadata}`, `[async_]{retrieve,store}_chunk`
    - `[async_]retrieve_partial_values`
    - `[async_]discover_nodes`
 - **Breaking**: Remove `Default` implementation for `Metadata{Convert,Erase}Version`
    - Explicitly use `global_config()` instead
 - **Breaking**: Remove `array::UnsafeCellSlice`
    - Replaced by `UnsafeCellSlice` in the `unsafe_cell_slice` crate
 - **Breaking**: Remove `NATIVE_ENDIAN`, use `Endianness::native()`
 - **Breaking**: Remove unused `DataTypeExtension`
 - Remove remnants of old synchronisation API

### Fixed
 - `[async_]store_set_partial_values` no longer truncates
   - this could corrupt values depending on the order of `set_partial_values` calls
 - Fix `FilesystemStore::fspath_to_key` on windows
 - Make `ArrayRepresentationBase` pub so that `{Array,Chunk}Representation` are not opaque
 - Fix `ArrayShardedExt::inner_chunk_grid` when applied on a sharded array with the `transpose` codec preceding `sharding_indexed`
 - Fix `ZipStorageAdapter` on windows
 - Fixed an unnecessary copy in `Array::[async_]retrieve_chunk_if_exists_opt`
 - Fixed `CodecOptions` not being forwarded in `Array::retrieve_chunk_subset_opt` on the fast path
 - Fixed missing fast path in `Array::[async_]retrieve_chunk_subset_opt`
 - Fixed `blosc` codec partial decoding with `noshuffle`

## [0.17.0-beta.3] - 2024-09-26

## [0.17.0-beta.2] - 2024-09-23

## [0.17.0-beta.1] - 2024-09-16

## [0.17.0-beta.0] - 2024-09-06

## [0.16.4] - 2024-08-22

### Fixed
 - Reduce maximum supported `opendal` version from 0.49 to 0.48
   - This reverts a change in 0.16.3 that was not semver compatible

## [0.16.3] - 2024-08-14

*This release was yanked.*

### Changed
 - Bump `derive_more` to 1.0.0
 - Bump maximum supported `opendal` version from 0.48 to 0.49
 - Box the metadata in `PluginMetadataInvalidError`

### Fixed
 - Fix `cargo test` with `--no-default-features`
 - Fix new clippy warnings in nightly

## [0.16.2] - 2024-08-05

### Added
 - Add the experimental `gdeflate` bytes-to-bytes codec
 - Add `Array::chunk_key()`
 - Add `ArrayShardedExt::inner_chunk_grid_shape()`

## [0.16.1] - 2024-07-30

### Changed
 - Bump maximum supported `opendal` version from 0.47 to 0.48

### Fixed
 - Fixed handling of empty shards with a variable length data type with sharding partial decoder

## [0.16.0] - 2024-07-28

### Highlights
 - Add experimental support for the `string` and `binary` data types and the `vlen` and `vlen_v2` codecs
 - Cleanup the `Array` API for retrieving elements (`Vec<T>`) and `ndarray`s
 - Support more Zarr V2 array configurations and make various experimental codecs `numcodecs` compatible

### Added
 - Add `ArrayBytes`, `RawBytes`, `RawBytesOffsets`, and `ArrayBytesError`
    - These can represent array data with fixed and variable length data types
 - Add `array::Element[Owned]` traits representing array elements
    - Supports conversion to and from `ArrayBytes`
 - Add `array::ElementFixedLength` marker trait
 - Add experimental `vlen` and `vlen_v2` codec for variable length data types
    - `vlen_v2` is for legacy support of Zarr V2 `vlen-utf8`/`vlen-bytes`/`vlen-array` codecs
 - Add `DataType::{String,Binary}` data types
    - These are likely to become standardised in the future and are not feature gated
 - Add `ArraySubset::contains()`
 - Add `FillValueMetadata::{String,Unsupported}`
   - `ArrayMetadata` can be serialised and deserialised with an unsupported `fill_value`, but `Array` creation will fail.
 - Implement `From<{[u8; N],&[u8; N],String,&str}>` for `FillValue`
 - Add `ArraySize` and `DataTypeSize`
 - Add `DataType::fixed_size()` that returns `Option<usize>`. Returns `None` for variable length data types.
 - Add `ArrayError::IncompatibleElementType` (replaces `ArrayError::IncompatibleElementSize`)
 - Add `ArrayError::InvalidElementValue`
 - Add `ChunkShape::num_elements_u64`
 - Add global `Config` option for manipulating experimental codec names
 - Add `metadata::v2::codec::ZfpyCodecConfigurationNumcodecs` and associated structures

### Changed
 - Use `[async_]retrieve_array_subset_opt` internally in `Array::[async_]retrieve_chunks_opt`
 - **Breaking**: Replace `[Async]ArrayPartialDecoderTraits::element_size()` with `data_type()`
 - Array `_store` methods now use `impl Into<ArrayBytes<'a>>` instead of `&[u8]` for the input bytes
 - **Breaking**: Array `_store_{elements,ndarray}` methods now use `T: Element` instead of `T: bytemuck::Pod`
 - **Breaking**: Array `_retrieve_{elements,ndarray}` methods now use `T: ElementOwned` instead of `T: bytemuck::Pod`
 - **Breaking**: Simplify array store `_ndarray` methods to 2 generic type parameters
 - Optimised `Array::[async_]store_array_subset_opt` when the subset is a subset of a single chunk
 - Make `transmute_to_bytes` public
 - Relax `ndarray_into_vec` from `T: bytemuck:Pod` to `T: Clone`
 - **Breaking**: `DataType::size()` now returns a `DataTypeSize` instead of `usize`
 - **Breaking**: `ArrayCodecTraits::{encode/decode}` have been specialised into `ArrayTo{Array,Bytes}CodecTraits::{encode/decode}`
 - Various changes to the experimental `zfp` codec
   - **Breaking**: Remove `Zfp{Expert,FixedAccuracy,FixedPrecision,FixedRate}Configuration` and just embed these structures directly in `ZfpMode`
   - **Breaking**: `ZfpCodec::new_expert` now takes `minbits`, `maxbits`, `maxprec`, and `minexp` instead of `ZfpExpertConfiguration`
   - **Breaking**: All `ZfpCodec::new_*` methods now take a `write_header: bool` parameter
 - **Breaking**: Add `ArrayMetadataV2ToV3ConversionError::Other`
 - Make all v2 metadata available even without experimental codec features
 - **Breaking**: Change pcodec `max_page_n` configuration to `equal_pages_up_to` to match numcodecs
 - Improve the `Array` docs

### Removed
 - **Breaking**: Remove `into_array_view` array and codec API
   - This was not fully utilised, not applicable to variable sized data types, and quite unsafe for a public API
 - **Breaking**: Remove internal `ChunksPerShardError` and just use `CodecError::Other`
 - **Breaking**: Remove `array_subset::{ArrayExtractBytesError,ArrayStoreBytesError}`
 - **Breaking**: Remove `ArraySubset::{extract,store}_bytes[_unchecked]`, they are replaced by methods in `ArrayBytes`
 - **Breaking**: Remove `array::validate_element_size` and `ArrayError::IncompatibleElementSize`
    - The internal validation in array `_element` methods is now more strict than just matching the element size
    - Example: `u16` must match `uint16` data type and will not match `int16` or `float16`

### Fixed
 - Fix an unnecessary copy in `async_store_set_partial_values`
 - Fix error when `bytes` metadata is encoded without a configuration, even if empty
 - Fix an error in `ChunkGrid` docs
 - Fixed `[async_]store_set_partial_values` and `MemoryStore::set` to correctly truncate the bytes of store value if they shrink

## [0.15.1] - 2024-07-11

### Added
 - Add `CITATION.cff`

### Changed
 - Implement `From<&String>` for `DimensionName`
 - Cleanup macro usage in array

### Fixed
 - Fix unnecessary allocations in `_elements` variants of array store methods

## [0.15.0] - 2024-07-07

### Highlights
 - Zarr V2 support (a Zarr V3 compatible subset)
 - Codec and array optimisations
    - Array store methods previously taking `Vec<u8>` now take `&[u8]`
    - Codec methods previously taking `Vec<u8>` now take `Cow<'_, [u8]>`
 - `AsyncToSyncStorageAdapter`: use an async store (e.g. HTTP, S3, etc.) in a sync context
 - Snappy codec support for the `blosc` codec

### Added
 - Add support for a V3 compatible subset of Zarr V2
   - Compatible subset: Zarr V2 data that is Zarr V3 compatible with only a metadata change
   - Zarr V2 metadata (`.zarray`/`.zgroup`/`.zattrs`) can be transformed to V3 (`zarr.json`)
 - Add `ArrayBuilder::build_arc` method
 - Add `Array::[async_]retrieve_encoded_chunk[s]` method
 - Add `Group::metadata_opt` method
 - Add `{Array,Group}::{store,erase}_metadata_opt` methods
 - Add `metadata::Metadata{Retrieve,Convert,Erase}Version` enums
 - Add `Config::[set_]metadata_{convert,erase}_version` methods
 - Add `{Array,Group}MetadataOptions::[set_]metadata_convert_version` methods
 - Add `{Config,ArrayMetadataOptions}::[set_]include_zarrs_metadata` methods
 - Add `Array::set_dimension_names`
 - Add `storage::[Maybe]AsyncBytes`
 - Add `array::{convert_from_bytes_slice,convert_to_bytes_vec}`
 - Add `AdditionalField`
 - Add `AsyncToSyncStorageAdapter` and `AsyncToSyncBlockOn`
 - Add internal `fill_array_view_with_fill_value` function

### Changed
 - **Breaking**: Deprecate `{Array,Group,Node}::[async_]new` for `[async_]open`, and add `open_opt`
 - **Breaking**: `Array` store methods now take slices instead of `Vec`s
 - **Breaking**: Change various store methods to take `&Arc<TStorage>` instead of `&TStorage`
 - **Breaking**: Sync and async stores now consume and return `bytes::Bytes` instead of `Vec<u8>`
 - **Breaking**: `{Array,Group}::metadata()` now return references instead of values
 - **Breaking**: `AdditionalFields` is now an alias for `BTreeMap<String, AdditionalField>` instead of an opaque struct
 - **Breaking**: Move `[Async]ReadableStorageTraits::{size[_prefix]}` to `[Async]ListableStorageTraits` and add default implementation for `size`
 - Use `monostate` for `zarr_format`, `node_type`, and `must_understand` in unknown fields in array and group metadata
   - These fields must be be valid on deserialisation rather than array/group initialisation
   - **Breaking**: Remove associated field validation functions
 - Support `object_store` 0.9-0.10
 - **Breaking**: Support `opendal` 0.46-0.47, drop support for 0.45
 - Bump `rayon` to 1.10.0
 - Bump `itertools` to 0.13
 - Bump `reqwest` to 0.12
 - Bump `zip` to 2.1
 - Bump `blosc-src` to 0.3.4
   - Adds `snappy` codec support
 - Bump minimum supported `flate2` to 1.0.30 and `thiserror` to 1.0.61
 - Add default implementations for `[Async]ReadableStorageTraits::{get,get_partial_values}`
 - Use `futures::TryStreamExt::try_for_each_concurrent` instead of `FuturesUnordered` where appropriate
 - Move all metadata/configuration structures into the metadata module (non breaking with re-exports)
 - Rename `Metadata` to `MetadataV3`, an alias is retained
 - Improve various docs
 - **Breaking**: Add `MissingMetadata` to `GroupCreateError` enum
 - Change internal structure of various iterators to use `std::ops::Range` and remove redundant `length`
 - **Breaking**: `Indices::new_with_start_end` now takes a `range` rather than a `start` and `end`
 - `RecommendedConcurrency::new` takes `impl std::ops::RangeBounds<usize>` instead of `std::ops::Range`
 - **Breaking**: Move `array::MaybeBytes` to `storage::MaybeBytes`
 - **Breaking**: Move `storage::storage_adapter::ZipStorageAdapter[CreateError]` to `storage::storage_adapter::zip::`
 - The `{async,sync}_http_array_read` examples now demonstrate usage of `opendal` and `object_store` storage backends
 - Bump `pcodec` to 0.3
   - Adds support for `uint16`, `int16` and `float16` data types to the experimental `pcodec` codec
   - **Breaking**: The schema for `PcodecCodecConfigurationV1` has changed
 - Exclude integration tests and data from published package

### Removed
 - **Breaking**: Remove re-exports of public dependencies
 - **Breaking**: Remove `Array::set_include_zarrs_metadata`. Use `{Config,ArrayMetadataOptions}::set_include_zarrs_metadata`
 - **Breaking**: Remove `ArrayMetadataV2ToV3ConversionError::InvalidZarrFormat`
 - **Breaking**: Remove `{Array,Group}CreateError::{InvalidZarrFormat,InvalidNodeType}`

### Fixed
 - **Breaking**: Change `ZfpExpertParams` to `ZfpExpertConfiguration` replacing existing `ZfpExpertConfiguration`
   - The previous `ZfpExpertConfiguration` was incorrect and was identical to `ZfpFixedRateConfiguration`
 - Fix `FilesystemStore::list_prefix` with an empty prefix
 - Fix `storage::[async_]discover_nodes` and add tests

## [0.14.0] - 2024-05-16

### Removed
 - **Breaking**: Remove `store_locks` module, `[Async]ReadableWritableStorageTraits::mutex()`, and `new_with_locks` constructors from stores
   - `DefaultStoreLocks` could result in a deadlock
   - It is now the responsibility of zarrs consumers to ensure that:
     - `Array::store_chunk_subset` is not called concurrently on the same chunk
     - `Array::store_array_subset` is not called concurrently on regions sharing chunks
   - Chunk locking may be revisited in a future release

## [0.13.3] - 2024-05-16

*This release was yanked and changes reverted.*

## [0.13.2] - 2024-05-08

### Changed
 - Make the `bz2` and `pcodec` codecs public
 - The `"name"` of experimental codecs in array metadata now points to a URI to avoid potential future incompatibilities with other implementations
 - Improve docs of several experimental codecs

## [0.13.1] - 2024-05-06

### Added
 - Added the `array_sharded_ext::{ArrayShardedExt,ArrayShardedReadableExt}` extension traits for `Array` to simplify working with sharded arrays
   - Abstracts the chunk grid to an "inner chunk grid" to simplify inner chunk retrieval.
   - Shard indexes are cached in a `ArrayShardedReadableExtCache`
   - Retrieval and chunk grid methods have fallbacks for unsharded arrays. For example, an inner chunk in an unsharded array is just a chunk
   - Sync API only, `AsyncArrayShardedReadableExt` and `AsyncArrayShardedReadableExtCache` are planned for a future release
 - Added `ChunkGridTraits::chunks_subset()` with default implementation

### Changed
 - Allow float fill values to be created from int fill value metadata
 - Make `chunk_grid::{regular,rectangular}` public
 - Support 8 and 16 bit integer data types with zfp codec by promoting to 32 bit

### Fixed
 - Fix `compute_encoded_size()` for `BitroundCodec` incorrectly indicating various data types were unsupported
 - Fix a link in chunk grid docs
 - Fix incorrect minimum dependency versions and add CI check
 - Fix clippy `unexpected_cfgs` warning with recent nightly

## [0.13.0] - 2024-04-20

### Added
 - Add "experimental codec store metadata if encode only" option to global config
 - Add "store empty chunks" option to global config and `CodecOptions`
 - Add `ArrayMetadataOptions`
 - Add small example to `README.md` and crate root documentation
 - Add `uint8`/`int8` support to the bitround codec
 - Add `len()` and `is_empty()` methods to array subset iterator producers
 - Add `Array::chunk_origin()`

### Changed
 - **Breaking**: Bump MSRV to 1.75 (28 December, 2023)
 - Bump `pco` (pcodec) to 0.2.1
 - **Breaking**: Add `CodecTraits::create_metadata_opt()`
 - **Breaking**: Rename `data_type::IncompatibleFillValueErrorMetadataError` to `IncompatibleFillValueMetadataError`
 - Add a useful warning if the shard index references out-of-bounds bytes in a chunk encoded with the sharding codec
 - **Breaking**: Require `ndarray::Array` parameters in various async array methods to implement `Send`

### Fixed
 - Fix implementation of `from_chunkgrid_regular_configuration` macro leading to recursion
 - Fix arithmetic overflow possible with the bitround codec with integer data types
 - Address `todo!()`s in `retrieve_chunks_into_array_view_opt` methods

## [0.12.5] - 2024-03-17

### Added
 - Implement `TryFrom<&str>` for `{Array,Group}Metadata`

### Changed
 - Make `array::codec::array_to_bytes::bytes::{Endianness::is_native(),NATIVE_ENDIAN,reverse_endianness}` public

## [0.12.4] - 2024-03-09

### Fixed
 - Remove unnecessary copy in `OpendalStore::set`
 - Fixed `zfp` codec encode not truncating compressed data
 - `zfp` codec `compute_encoded_size()` now correctly outputs a bounded size instead of an unbounded size

## [0.12.3] - 2024-03-07

### Added
 - Implement `Deserialize` for `DataType`
   - A convenience for `zarrs` consumers. `ArrayMetadata` continues to use `Metadata` to parse unknown data types.
 - Add `{Array,Group}::{async_}erase_metadata()` and `storage::{async_}erase_metadata()`

### Fixed
 - Fixed various errors in storage docs
 - Blosc codec config allow deserialization with missing `typesize`/`shuffle`
 - Blosc codec encoding with `typesize` of 0 with shuffling

## [0.12.2] - 2024-02-26

### Added
 - Added "Getting Started" in root documentation

### Changed
 - Disabled `blosc` codec partial decoding pending a faster implementation

### Fixed
 - Remove an unnecessary allocation in `IndicesIterator`

## [0.12.1] - 2024-02-24

### Added
 - Add `byte_range::extract_byte_ranges_concat()`
 - Add `{Async}BytesPartialDecoderTraits::partial_decode_concat()` with default implementation
 - Add `ArrayCodecTraits::partial_decode_granularity()` with default implementation

### Changed
 - Cleanup "Implementation Status" and "Crate Features" in root documentation
 - Minor changes to docs

## [0.12.0] - 2024-02-22

### Highlights
 - This release targeted:
   - Improving performance and reducing memory usage
   - Increasing test coverage (82% from 66% in v0.11.6)
 - There are a number of breaking changes, the most impactful user facing changes are
   - `Array` `par_` variants have been removed and the default variants are now parallel instead of serial
   - `Array` `_opt` variants use new `CodecOptions` instead of `parallel: bool`
   - `ArraySubset` `iter_` methods have had the `iter_` prefix removed. Returned iterators now implement `into_iter()` and some also implement `into_par_iter()`
   - `Array::{set_}parallel_codecs` and `ArrayBuilder::parallel_codecs` have been removed. New methods supporting `CodecOptions` offer far more concurrency control
   - `DimensionName`s can now be created less verbosely. E.g. `array_builder.dimension_names(["y", "x"].into())`. Type inference will fail on old syntax like `.dimension_names(vec!["y".into(), "x".into()].into())`
   - `Array` store `_ndarray` variants now take any type implementing `Into<ndarray::Array<T, D>>` instead of `&ndarray::ArrayViewD<T>`.

### Added
#### Arrays
 - Add `array::concurrency::RecommendedConcurrency`
 - Add `array::ArrayView`
 - Add array `into_array_view` methods
   - `Array::{async_}retrieve_chunk{_subset}_into_array_view{_opt}`
   - `Array::{async_}retrieve_chunks_into_array_view{_opt}`
   - `Array::{async_}retrieve_array_subset_into_array_view{_opt}`
 - Add `Array::{async_}retrieve_chunk_if_exists{_elements,_ndarray}_{opt}`
 - Add `_opt` variants to various array store/retrieve methods
 - Add `Array::dimensionality()`
 - Add `Array::chunk_shape_usize()`

#### Codecs
 - Add `codec::CodecOptions{Builder}`
 - Add `ArrayCodecTraits::decode_into_array_view` with default implementation
 - Add `{Async}ArrayPartialDecoderTraits::partial_decode_into_array_view{_opt}` with default implementation
 - Add `TestUnbounded` codec for internal testing

#### Array Subset Iterators
 - Add `Contiguous{Linearised}IndicesIterator::contiguous_elements{_usize}()`
 - Implement `DoubleEndedIterator` for `{Indices,LinearisedIndices,ContiguousIndices,ContiguousLinearisedIndicesIterator}Iterator`
 - Add `ParIndicesIterator` and `ParChunksIterator`

#### Miscellaneous
 - Add `Default Codec Concurrent Target` and `Default Chunk Concurrency Minimum` global configuration options to `Config`
 - Add `{Async}ReadableWritableListableStorageTraits` and `{Async}ReadableWritableListableStorage`
 - Add `{Chunk,Array}Representation::shape_u64`
 - Implement `AsyncBytesPartialDecoderTraits` for `std::io::Cursor<{&[u8],Vec<u8>}>`
 - Implement `From<ChunkShape>` for `Vec<NonZeroU64>`
 - Add `ChunkShape::num_elements()`
 - Implement `From<String>` for `DimensionName`
 - Add `array::unsafe_cell_slice::UnsafeCellSlice::len()`
 - Add `{Array,Chunk}Representation::dimensionality()`
 - Add `ArraySubset::new_empty()` and `ArraySubset::is_empty()`
 - Add missing `IncompatibleArraySubsetAndShapeError::new()`
 - Add `--usage-log` argument to examples to use `UsageLog` storage transformer
 - Add more tests for `Array`, codecs, store locks, and more
 - Add `array_write_read_ndarray` example
 - Add `array::bytes_to_ndarray()` and make `array::elements_to_ndarray()` public
 - [#13](https://github.com/LDeakin/zarrs/pull/13) Add `node::Node::path()`, `metadata()`, and `children()` by [@lorenzocerrone]
 - [#13](https://github.com/LDeakin/zarrs/pull/13) Derive `Clone` for `node::Node` and `node::NodeMetadata` by [@lorenzocerrone]
 - Add `bytemuck` feature to the `half` dependency (public)

### Changed
#### Arrays
 - **Breaking**: `Array` `_opt` methods now use a `codec::CodecOptions` parameter instead of `parallel: bool`
 - **Behaviour change**: default variants without `_opt` are no longer serial but parallel by default
 - **Breaking**: `Array` store `_ndarray` variants now take any type implementing `Into<ndarray::Array<T, D>>` instead of `&ndarray::ArrayViewD<T>`
   - This is to reflect that these methods consume the underlying `Vec` in the ndarray
   - It also removes the constraint that arrays have a dynamic dimension

#### Codecs
 - **Breaking**: remove `par_` variants and many `_opt` variants in favor of a single method with a `codec::CodecOptions` parameter
   - `partial_decode` and `partial_decode_opt` remain
   - **Behaviour change**: `partial_decode` is no longer serial but parallel by default
 - **Breaking**: add `{ArrayCodecTraits,BytesToBytesCodecTraits}::recommended_concurrency()`
 - **Breaking**: add `ArrayPartialDecoderTraits::element_size()`

#### Array Subset Iterators
 - **Breaking**: `ArraySubset::iter_` methods no longer have an `iter_` prefix and return structures implementing `IntoIterator` including
   - `Indices`, `LinearisedIndices`, `ContiguousIndices`, `ContiguousLinearisedIndices`, `Chunks`
   - `Indices` and `Chunks` also implement `IntoParallelIter`
 - Array subset iterators are moved into public `array_subset::iterators` and no longer in the `array_subset` namespace

### Storage
 - **Breaking**: Storage transformers must be `Arc` wrapped as `StorageTransformerExtension` trait methods now take `self: Arc<Self>`
 - **Breaking**: `Group` and `Array` methods generic on storage now require the storage have a `'static` lifetime
 - Removed lifetimes from `{Async}{Readable,Writable,ReadableWritable,Listable,ReadableListable}Storage`

#### Miscellaneous
 - **Breaking**: `ArrayBuilder::dimension_names()` generalised to accept `Option<I>` where `I: IntoIterator<Item = D>` and `D: Into<DimensionName>`
   - Can now write `builder.dimension_names(["y", "x"].into())` instead of `builder.dimension_names(vec!["y".into(), "x".into()].into())`
 - **Breaking**: Remove `Array::{set_}parallel_codecs` and `ArrayBuilder::parallel_codecs`
 - **Breaking**: Add `ChunkGridTraits::chunk_shape_u64{_unchecked}` to `ChunkGridTraits`
 - **Breaking**: Add `create{_async}_readable_writable_listable_transformer` to `StorageTransformerExtension` trait
 - **Breaking**: Rename `IncompatibleArrayShapeError` to `IncompatibleArraySubsetAndShapeError`
 - **Breaking**: Use `IncompatibleArraySubsetAndShapeError` in `ArrayStoreBytesError::InvalidArrayShape`
 - **Breaking**: Add `ArrayError::InvalidDataShape`
 - Add a fast path to `Array::retrieve_chunk_subset{_opt}` if the entire chunk is requested
 - `DimensionName::new()` generalised to accept a name implementing `Into<String>`
 - Cleanup uninitialised `Vec` handling
 - Dependency bumps
   - `crc32` (private) to [0.6.5](https://github.com/zowens/crc32c/releases/tag/v0.6.5) to fix nightly build
   - `opendal` (public) to [0.45](https://github.com/apache/opendal/releases/v0.45.0)
 - Make `UnsafeCellSlice` public

### Removed
 - **Breaking**: Remove `InvalidArraySubsetError` and `ArrayExtractElementsError`
 - **Breaking**: Remove non-default store lock constructors
 - **Breaking**: Remove unused `storage::store::{Readable,Writable,ReadableWritable,Listable}Store`

### Fixed
 - **Breaking**: `ArraySubset::end_inc` now returns an `Option`, which is `None` for an empty array subset
 - `Array::retrieve_array_subset` and variants now correctly return the fill value if the array subset references out-of-bounds elements
 - Add missing input validation to some `partial_decode` methods
 - Validate `ndarray` array shape in `{async_}store_{chunk,chunks}_ndarray{_opt}`
 - Fixed transpose partial decoder and its test, elements were not being correctly transposed
 - Minor docs fixes

## [0.11.6] - 2024-02-06

### Added
 - Add a global configuration `config::Config` accessible via `config::{get_config,get_config_mut}`
   - Currently it exposes a single configuration option: `validate_checksums` (default: `true`)
 - Document correctness issues with past versions and how to correct errant arrays in crate root

## [0.11.5] - 2024-02-05

### Fixed
 - **Major bug** Fixed the `crc32c` codec so it uses `CRC32C` rather than `CRC32`
   - All arrays written prior to this release that use the `crc32c` codec are not correct
 - Fixed the `crc32c` codec reserving more memory than necessary

## [0.11.4] - 2024-02-05

### Added
 - Add `codecov` support to CI

### Fixed
  - Fixed a regression introduced in v0.11.2 ([89fc63f](https://github.com/LDeakin/zarrs/commit/89fc63fa318cfd780e85fec6f9506ca65336a2c3)) where codecs with an empty configuration would serialise as a string rather than a struct with a `name` field, which goes against the zarr spec
    - Fixes the `codec_bytes_configuration_none` test and adds `codec_crc32c_configuration_none` test

## [0.11.3] - 2024-01-31

### Added
 - Added support for [miri](https://github.com/rust-lang/miri) testing and accompanying notes in `BUILD.md`

### Changed
 - Make `IDENTIFIER` public for codecs, chunk key encodings, and chunk grids

### Fixed
 - Fix formatting of `pcodec` feature in `lib.rs` docs
 - Remove `println!` in `PcodecCodec`
 - Fixed `FillValue::equals_all` with unaligned inputs

## [0.11.2] - 2024-01-30

### Added
 - Added experimental `bz2` (bzip2) codec behind `bz2` feature
 - Added experimental `pcodec` codec behind `pcodec` feature

### Changed
 - docs: clarify that `bitround` and `zfp` codec configurations are draft

### Fixed
 - Do not serialise `configuration` in `Metadata` if is empty
 - Do not serialise `endian` in `BytesCodecConfigurationV1` if it is none

## [0.11.1] - 2024-01-29

### Fixed
 - Fixed build with `bitround` or `zfp` features without `async` feature

## [0.11.0] - 2024-01-26

### Highlights
 - This release targeted
   - Improving documentation
   - Increasing coverage and correctness (line coverage increased from 70.66% to 78.46% since `0.10.0`)
   - Consolidating and improving errors and their messages
 - Major breaking changes
   - `Array` `retrieve_` methods now return `Vec<u8>`/`Vec<T>` instead of `Box<[u8]>`/`Box<[T]>`
   - Added `ChunkShape` (which wraps `Vec<NonZeroU64>`) and added `ChunkRepresentation`
     - Chunks can no longer have any zero dimensions
     - Creating an array now requires specifying a chunk shape like `vec![1, 2, 3].try_into()?` instead of `vec![1, 2, 3].into()`

### Added
 - Tests for `ByteRange`, `BytesRepresentation`, `StorePrefix`, `StoreKey`, `ArrayBuilder`, `ArraySubset`, `GroupBuilder`, `Group`, `NodeName`, `NodePath`, `Node`, `AdditionalFields`, `Metadata`, `FillValue`, `Group`, `Metadata`
 - `array_subset::IncompatibleStartEndIndicesError`
 - Add `array::transmute_from_bytes_vec`
 - Re-export public dependencies at the crate root: `bytes`, `bytemuck`, `dyn_clone`, `serde_json`, `ndarray`, `object_store`, and `opendal`
 - Implement `Display` for `ByteRange`, `StoreKeyRange`, `NodeName`
 - Add `HexString::new`
 - Add `PluginMetadataInvalidError`

### Changed
 - **Breaking**: `Array` `retrieve_` methods now return `Vec<u8>`/`Vec<T>` instead of `Box<[u8]>`/`Box<[T]>`
   - This avoids potential internal reallocations
 - **Breaking**: `StoreKey::parent` now returns `StorePrefix` instead of `Option<StorePrefix>`
 - **Breaking**: `ZipStorageAdapter::{new,new_with_path}` now take a `StoreKey`
 - **Breaking**: `ArraySubset::new_with_start_end_{inc,exc}` now return `IncompatibleStartEndIndicesError` instead of `IncompatibleDimensionalityError`
   - It is now an error if any element of `end` is less than `start`
 - Remove `#[must_use]` from `GroupBuilder::{attributes,additional_fields}`
 - **Breaking**: Rename `Node::new_with_store` to `Node::new`, and `Node::new` to `Node::new_with_metadata` for consistency with `Array`/`Group`
 - Use `serde_json` `float_roundtrip` feature
 - **Breaking**: Use `bytemuck` instead of `safe_transmute`
   - Array methods now have `<T: bytemuck::Pod + ..>` instead of `<T: safe_transmute::TriviallyTransmutable + ..>`
 - **Breaking**: Rename `array::safe_transmute_to_bytes_vec` to `array::transmute_to_bytes_vec`
 - **Breaking**: Make `zfp` a private dependency by changing `Zfp{Bitstream,Field,Stream}` from `pub` to `pub(super)`
 - **Breaking**: Make `zip` a private dependency by not exposing `ZipError` in `ZipStorageAdapterCreateError`
 - Refine `UsageLogStorageTransformer` outputs and add docs
 - Improve `PerformanceMetricsStorageTransformer` docs
 - **Breaking**: `InvalidByteRangeError` now holds a `ByteRange` and bytes length and returns a more informative error message
 - **Breaking**: Remove `StorageError::InvalidJSON` and add `StorageError::InvalidMetadata`
   - `InvalidMetadata` additionally holds a `StoreKey` for more informative error messages
 - More informative `Metadata` deserialisation error message with an invalid configuration
 - **Breaking**: `PluginCreateError::Other` changed to unit struct and added `PluginCreateError::from<{String,&str}>`
 - `PluginCreateError::Unsupported` now includes a `plugin_type` field for more informative error messages
 - Add `array::ChunkShape` wrapping `Vec<NonZeroU64>` and `array::ChunkRepresentation` which is essentially `ArrayRepresentation` with a `NonZeroU64` shape
   - **Breaking**: Relevant codec and partial decoder methods now use `ChunkRepresentation` instead of `ArrayRepresentation`
   - **Breaking**: Relevant chunk grid methods now use `ChunkShape` instead of `ArrayShape`
   - **Breaking**: Relevant array methods now use `ChunkShape` instead of `ArrayShape`

### Removed
 - **Breaking**: Remove `StorePrefixError::new`, deprecated since `v0.7.3`
 - **Breaking**: Remove `ArraySubset::{in_subset,in_subset_unchecked}`, deprecated since `v0.7.2`
 - **Breaking**: Remove `impl From<&StorePrefix> for NodeName`, unused and not useful
 - **Breaking**: Remove `NodeCreateError::Metadata`
   - `NodeCreateError::StorageError` with `StorageError::InvalidMetadata` is used instead
 - **Breaking**: Remove `{ArrayCreateError,GroupCreateError}::MetadataDeserializationError`
   - `{ArrayCreateError,GroupCreateError}::StorageError` with `StorageError::InvalidMetadata` is used instead
 - **Breaking**: Remove `GroupCreateError::Metadata` as it was unused
 - **Breaking**: Remove `PluginCreateError::ConfigurationInvalidError`

### Fixed
 - Disallow an empty string for a `StoreKey`
 - `ArrayBuilder` now validates additional fields
 - `FillValue::equals_all` incorrect behaviour with a `FillValue` with size not equal to 1, 2, 4, 8, or 16 bytes.
 - Fix `NodePath` display output
 - Fix handling of non-standard `NaN` values for `f16` and `bf16`
 - Fix potential missed error in `Metadata::to_configuration`

## [0.10.0] - 2024-01-17

### Changed
 - Bump `opendal` to 0.44
 - Bump `object_store` to 0.9
 - **Breaking** `async_store_chunk` and `AsyncWritableStorageTraits::set` now take `bytes::Bytes`
   - `bytes::Bytes` are used by both supported async stores (`object_store` and `opendal`), and this avoids a copy

### Fixed
 - Various clippy warnings

## [0.9.0] - 2024-01-03

### Highlights
 - New `Array` methods to store/retrieve/erase multiple chunks
 - Many `Array` internal revisions and removal of some unnecessary methods

### Added
 - Reexport `safe_transmute::TriviallyTransmutable` as `array::TriviallyTransmutable`
 - Add `Array::chunks_subset{_bounded}`
 - Add `store_chunks`, `retrieve_chunks`, `erase_chunks` and variants to `Array`

### Changed
 - Use macros to reduce common code patterns in `Array`
 - Separate `Array` methods into separate files for each storage trait
 - **Breaking**: Remove `_opt` and `par_` variants of `async_retrieve_array_subset` and `async_store_array_subset` (including `_elements` and `_ndarray` variants)
 - Revise `array_write_read` and `async_array_write_read` examples
 - **Breaking**: Storage `erase`/`erase_values`/`erase_prefix` methods and `Array::erase_chunk` now return `()` instead of `bool` and succeed irrespective of the whether the key/prefix exists

## [0.8.0] - 2023-12-26

### Highlights
 - Feature changes:
   - Added: `object_store` and `opendal` with generalised support for stores from these crates
   - Removed: `s3`, `gcp`, and `azure` (use `object_store` or `opendal` instead)
   - Changed: `http` and `zip` are no longer default features
 - `ReadableStorageTraits` is no longer a supertrait of `WritableStorageTraits`
 - Moved chunk locking from `Array` into stores
 - Improved documentation and code coverage

### Added
 - Added `ReadableWritableStorage` and `ReadableWritableStore` and async variants
 - Added `{async_}store_set_partial_values`
 - **Breaking** Added `create_readable_writable_transformer` to `StorageTransformerExtension` trait
 - Added `storage::store_lock` module
   - Adds generic `StoreLocks`, `StoreKeyMutex`, and `StoreKeyMutexGuard` with associated traits and async variants
   - Includes `DefaultStoreLocks` and `DisabledStoreLocks` implementations
 - Readable and writable stores include a `new_with_locks` method to choose the store lock implementation
 - Added `ArraySubset::new_with_ranges`
 - Added `ByteRange::offset`
 - Added `object_store` feature with `AsyncObjectStore` store (wraps `object_store::ObjectStore`)
   - **Breaking** Removes the explicit `object_store`-based stores (e.g. `AsyncAmazonS3Store`, `AsyncHTTPStore`)
   - **Breaking** Removes the `object_store_impl` macro
   - **Breaking** Removes the `s3`, `gcp`, and `azure` crate features
 - Added `opendal` feature
   - `OpendalStore` store (wraps `opendal::BlockingOperator`)
   - `AsyncOpendalStore` store (wraps `opendal::Operator`)

### Changed
 - **Breaking**  `ReadableStorageTraits` is no longer a supertrait of `WritableStorageTraits`
   - `WritableStorage` is no longer implicitly readable. Use `ReadableWritableStorage`
 - **Breaking**: `{Async}WritableStorageTraits::set_partial_values()` no longer include default implementations
   - Use new `{async_}store_set_partial_values` utility functions instead
 - Add `#[must_use]` to `Array::builder`, `Array::chunk_grid_shape`, and `ArrayBuilder::from_array`
 - **Breaking** Remove `http` and `zip` from default features
 - Locking functionality for arrays is moved into stores
 - Improved `Array` documentation
 - Add store testing utility functions for unified store testing
 - Make various `Array` methods with `parallel` parameter `pub`
 - Remove `#[doc(hidden)]` from various functions which are `unsafe` and primarily intended for internal use
 - **Breaking** Bump minimum supported rust version (MSRV) to `1.71` (13 July, 2023)

### Fixed
 - Fixed `MemoryStore::get_partial_values_key` if given an invalid byte range, now returns `InvalidByteRangeError` instead of panicking

## [0.7.3] - 2023-12-22

### Added
 - Add `From<ChunkKeyEncodingTraits>` for `ChunkKeyEncoding`
 - Add chunk key encoding tests

### Changed
 - Revise code coverage section in `BUILD.md` to use `cargo-llvm-cov`
 - Increased code coverage in some modules
 - Add `--all-features` to clippy usage in `BUILD.md` and `ci.yml`

### Fixed
 - Fixed chunk key encoding for 0 dimensional arrays with `default` and `v2` encoding
 - Fixed various clippy warnings

## [0.7.2] - 2023-12-17

### Added
 - `ArraySubset::{extract_elements/extract_elements_unchecked}` and `ArrayExtractElementsError`

### Changed
 - Add `ArraySubset::{overlap,overlap_unchecked}` and `ArraySubset::{relative_to,relative_to_unchecked}`
   - These replace `ArraySubset::{in_subset,in_subset_unchecked}`, which are now deprecated
 - Add `From<String>` for `StorePrefixError` and deprecate `StorePrefixError::new`

### Fixed
 - Fix `cargo test` with `async` crate feature disabled

## [0.7.1] - 2023-12-11

### Fixed
 - Fix use of `impl_trait_projections` in `{Array/Bytes}PartialDecoderCache`, which was only stabilised in Rust 1.74

## [0.7.0] - 2023-12-05

### Highlights
 - Experimental `async` feature: Support async stores and array/group operations
   - See `async_array_write_read` and `async_http_array_read` examples
 - Experimental `s3`/`gcp`/`azure` features for experimental async Amazon S3/Google Cloud Storage/Microsoft Azure Blob Storage stores

### Added
 - Add `storage_async` module with asynchronous storage traits: `Async{Readable,Writable,Listable,ReadableListable}StorageTraits`
  - Implemented for `StorageHandle`
 - Add async filesystem/http/memory stores in `storage::store::async` module and a wrapper for any store provided by the [`object_store`](https://docs.rs/object_store/latest/object_store/) crate
 - Add async support to zarr `Group` and `Array`
 - Add `Async{Readable,Writable,Listable,ReadableListable}Storage`
 - Add `StorageTransformerExtension::create_async_{readable,writable,listable,readable_listable}_transformer`
 - Add `Node::async_new_with_store` and implement `storage::async_get_child_nodes`
 - Add `async_array_write_read` and `async_http_array_read` examples
 - Add experimental `async` feature (disabled by default)
 - Add experimental async `Amazon S3`, `Google Cloud`, `Microsoft Azure` stores (untested)

### Changed
 - Bump itertools to `0.12` and zstd to `0.13`
 - Set minimum supported rust version (MSRV) to `1.70` (1 June, 2023)
   - Required by `half` since `2.3.1` (26 June, 2023)
 - Make `StoreKeyRange` and `StoreKeyStartValue` clonable
 - **Breaking**: Remove `ReadableWritableStorageTraits`, `ReadableWritableStorage`, `ReadableWritableStore`, and `StorageTransformerExtension::create_readable_writable_transformer`
   - These were redundant because `WritableStorageTraits` requires `ReadableStorageTraits` since 6e69a4d
 - Move sync stores to `storage::store::sync`
 - Move sync storage traits to `storage_sync.rs`
 - Move array sync storage trait impls into `array_sync.rs`
 - Use `required-features` for examples

## [0.6.0] - 2023-11-16

### Highlights
 - Revisions for recent updates to the Zarr V3 specification (e.g. sharding `index_location` and removal of `"order": "C"/"F"` from transpose codec)
 - API changes to improve usability
 - Performance improvements and a few bug fixes
 - Experimental `zfp` and `bitround` codecs

### Added
 - **Breaking**: Added `UnsupportedDataTypeError`
 - **Breaking**: Added `CodecError::UnsupportedDataType`
 - Added `array_subset::IncompatibleArrayShapeError`
 - Added `array_subset::iter_linearised_indices_unchecked`
 - Added parallel encoding/decoding to tests
 - Added `array_subset::ArrayStoreBytesError`, `store_bytes`, and `store_bytes_unchecked`
 - Added experimental `zfp` codec implementation behind `zfp` feature flag (disabled by default)
 - Added experimental `bitround` codec implementation behind `bitround` feature flag (disabled by default)
   - This is similar to [numcodecs BitRound](https://numcodecs.readthedocs.io/en/stable/bitround.html#numcodecs.bitround.BitRound), but it supports rounding integers from the most significant set bit
 - Added `ShardingCodecBuilder`
 - Added `ReadableListableStorage`, `ReadableListableStorageTraits`, `StorageTransformerExtension::create_readable_listable_transformer`
 - Added `ByteRange::to_range()` and `to_range_usize()`
 - Added `StoreKeyStartValue::end()`
 - Added default implementation for `WritableStorageTraits::set_partial_values`
    - `WritableStorageTraits` now requires `ReadableStorageTraits`
 - Added `From<&[u8]>` for `FillValue`
 - Added `FillValue::all_equal` and fill value benchmark
    - Implements a much faster fill value test
 - Added `Array::chunk_grid_shape`/`chunk_subset`
 - Added `par_` variants for the `store_array_subset`/`retrieve_array_subset` variants in `Array`, which can encode/decode multiple chunks in parallel
 - Added `ArraySubset::bound` and `Array::chunk_subset_bounded`
 - Added methods to `CodecChain` to retrieve underlying codecs
 - Added `Array::builder()` and `ArrayBuilder::from_array()`
 - Added more `ArrayBuilder` modifiers and made internals public
 - Added a fast path to `Array::retrieve_array_subset` methods if the array subset matches a chunk
 - Added `array::{ZARR_NAN_F64,ZARR_NAN_F32,ZARR_NAN_F16,ZARR_NAN_BF16}` which match the zarr nan bit representation on all implementations
 - Added `size_usize()` to `ArrayRepresentation`
 - Add generic implementations of storage supertraits (`ReadableWritableStorageTraits` and `ReadableListableStorageTraits`)
 - Add `size_prefix()` to stores

### Changed
 - **Breaking**: `array::data_type::DataType` is now marked `#[non_exhaustive]`
 - **Breaking**: Promote the `r*` (raw bits), `float16` and `bfloat16` data types to standard data types in `array::data_type::DataType`, rather than extension data types
   - **Breaking**: Remove the crate features: `raw_bits`, `float16`, `bfloat16`
   - **Breaking**: Removes `array::data_type::RawBitsDataType/Bfloat16DataType/Float16DataType`
   - **Breaking**: `half` is now a required dependency
 - **Breaking**: Rename `TransposeCodec::new` to `new_with_configuration`
 - **Breaking**: Array subset methods dependent on an `array_shape` now use the `IncompatibleArrayShapeError` error type instead of `IncompatibleDimensionalityError`
 - **Breaking**: Various array subset iterators now have validated `new` and unvalidated `new_unchecked` constructors
 - Blosc codec: disable parallel encoding/decoding for small inputs/outputs
 - Bytes codec: optimise implementation
 - **Breaking**: `ArrayToArrayCodecTraits::compute_encoded_size` and `ArrayToBytesCodecTraits::compute_encoded_size` can now return a `CodecError`
 - `ArrayBuilder::build()` and `GroupBuilder::build` now accept unsized storage
 - **Breaking**: `StoragePartialDecoder` now takes a `ReadableStorage` input
 - `sharded_array_write_read` example now prints storage operations and demonstrates retrieving inner chunks directly from a partial decoder
 - the zarrs version and a link to the source code is now written to the `_zarrs` attribute in array metadata, this can be disabled with `set_include_zarrs_metadata(false)`
 - **Breaking**: Rename `FillValueMetadata::Uint` to `UInt` for consistency with the `DataType::UInt` variants.
 - `StorePrefix`, `StoreKey`, `NodeName` `new` methods are now `: impl Into<String>`
 - **Breaking**: `ArrayBuilder::dimension_names` now takes an `Option` input, matching the output of `Array::dimension_names`
 - **Breaking**: `ArrayError::InvalidChunkGridIndicesError` now holds array indices directly, not a `chunk_grid::InvalidArrayIndicesError`
 - **Breaking**: `Array::chunk_array_representation` now returns `ArrayError` instead of `InvalidChunkGridIndicesError` and the `array_shape` parameter is removed
 - **Breaking**: `Array::chunks_in_array_subset` now returns `IncompatibleDimensionalityError` instead of `ArrayError`
 - **Breaking**: `ArraySubset::new_with_start_end_inc`/`new_with_start_end_exc` now take `end: ArrayIndices` instead of `end: &[u64]
 - **Breaking**: Move `array_subset::validate_array_subset` to `ArraySubset::inbounds`
 - Allow out-of-bounds `Array::store_array_subset` and `Array::retrieve_array_subset`
    - retrieved out-of-bounds elements are populated with the fill value
 - Derive `Clone` for `StorageTransformerChain`
 - **Breaking**: `ArrayBuilder::storage_transformers` use `StorageTransformerChain`
 - **Breaking**: change `ArrayError::InvalidFillValue` to `InvalidFillValueMetadata` and add a new `InvalidFillValue`
 - **Breaking**: The transpose codec order configuration parameter no longer supports the constants "C" or "F" and must instead always be specified as an explicit permutation [zarr-developers/zarr-specs #264](https://github.com/zarr-developers/zarr-specs/pull/264)
   - Removes `TransposeOrderImpl`
   - `TransposeOrder` is now validated on creation/deserialisation and `TransposeCodec::new` no longer returns a `Result`
 - **Breaking**: Change `HexString::as_bytes()` to `as_be_bytes()`
 - Support `index_location` for sharding codec
 - Optimise `unravel_index`
 - **Breaking**: Array subset iterator changes
   - Simplify the implementations
   - Remove `next` inputs
   - Make constructors consistent, remove `inner` in constructors
   - Add `size_hint` to all iterators, implement `ExactSizeIterator`/`FusedIterator`
 - **Major breaking**: Output boxed slices `Box<[..]>` from array retrieve methods instead of `Vec<..>`
 - **Major breaking**: Input `Vec<..>` instead of `&[..]` to array store methods
   - Supports some perf improvements
 - **Breaking**: Change `BytesRepresentation` enum from `KnownSize(u64)`/`VariableSize` to `FixedSize(u64)`/`BoundedSize(u64)`/`UnboundedSize`
 - Preallocate sharding codec encoded output when the encoded representation has a fixed or bounded size
 - Add `par_encode` for `zstd` codec
 - **Breaking**: Codecs now must implement `encode_opt`, `decode_opt`, and `partial_decoder_opt`
 - **Breaking**: Partial decoders now must implement `partial_decode_opt` and the `decoded_representation` is now supplied on creation, rather than when calling `partial_decode`/`partial_decode_par`/`partial_decode_opt`
 - Sharding partial decoder now decodes inner chunks in full rather than partially decoding them, this is much faster with some codecs (e.g. blosc)
    - In future, this will probably become configurable
 - Moved `storage/store/{key.rs,prefix.rs}` to `storage/{store_key.rs,store_prefix.rs}`

### Fixed
 - Bytes codec handling of complex and raw bits data types
 - Additional debug assertions to validate input in array subset `_unchecked` functions
 - **Breaking**: `array_subset::iter_linearised_indices` now returns a `Result<_, IncompatibleArrayShapeError>`, previously it could not fail even if the `array_shape` did not enclose the array subset
 - The `array_subset_iter_contiguous_indices3` test was incorrect as the array shape was invalid for the array subset
 - `ArraySubset::extract_bytes` now reserves the appropriate amount of memory
 - Sharding codec performance optimisations
 - `FilesystemStore::erase_prefix` now correctly removes non-empty directories
 - **Breaking**: `ArrayBuilder::storage_transformers` remove `#[must_use]`
 - Validate data type and fill value compatibility in `ArrayBuilder`
 - Handling of `"NaN"` fill values, they are now guaranteed to match the byte representation specified in the zarr v3 spec
 - Add a fast path to array retrieve methods which avoids a copy
 - Optimise sharding codec decode by removing initial population by fill value
 - Include checksum with `zstd` codec if enabled, previously this did nothing

### Removed
 - **Breaking**: Disabled data type extensions `array::data_type::DataType::Extension`.
 - **Breaking**: Remove `StoreExtension` traits
 - **Breaking**: Remove `chunk_grid::InvalidArrayIndicesError`/`ChunkGridShapeError`
 - **Breaking**: Remove `ArrayError::InvalidArrayIndicesError`

## [0.5.1] - 2023-10-10

### Added
 - Tests for `DataType::fill_value_from_metadata`
 - A paragraph on concurrency in the `Array` docs

### Changed
 - Fix some docs typos

### Fixed
 - `FillValueMetadata::try_as_uint/int` can both now handle `FillValueMetadata::Uint/Int`
 - `Array::store_chunk` now erases empty chunks
 - Fixed a race in `Array::store_chunk_subset` and add a fast path if the subset spans the whole chunk
 - Fix complex number handling in `DataType::fill_value_from_metadata`
 - Fix byte arrays being interpreted as complex number fill value metadata

## [0.5.0] - 2023-10-08

### Added
 - `MaybeBytes` an alias for `Option<Vec<u8>>`
   - When a value is read from the store but the key is not found, the returned value is now `None` instead of an error indicating a missing key
 - Add `storage::erase_chunk` and `Array::erase_chunk`
   - The `array_write_read` example is updated to demonstrate `erase_chunk`
 - Add `TryFrom::<&str>` for `Metadata` and `FillValueMetadata`
 - Add `chunk_key_encoding` and `chunk_key_encoding_default_separator` to `ArrayBuilder`
 - Add `TryFrom<char>` for `ChunkKeySeparator`
 - Add `ArraySubset::new_with_start_end_inc/new_with_start_end_exc`
 - Add `codec::extract_byte_ranges_read` utility function to read byte ranges from a `Read` source which does not support `Seek`

### Changed

 - **Breaking**: Remove `StorageError::KeyNotFound` in favour of returning `MaybeBytes`
 - **Breaking**: Add `StorageError::UnknownKeySize` if a method requires a known key size, but the key size cannot be determined
 - **Breaking**: Add `ArrayCreateError::MissingMetadata` if attempting to create an array in a store without specifying metadata, and the metadata does not exist
 - **Breaking**: `UsageLogStorageTransformer` now takes a prefix function rather than a string
   - The `http_array_read` example now logs store requests to stdout using `UsageLogStorageTransformer`
 - **Breaking**: `WritableStorageTraits::erase/erase_values/erase_prefix` return a boolean indicating if they actually deleted something
 - **Breaking**: Add `ReadableStorageTraits::get_partial_values_key` which reads byte ranges for a store key
 - **Breaking**: Changed `data_type::try_create_data_type` to `DataType::from_metadata`
 - **Breaking**: Changed `try_create_codec` to `Codec::from_metadata`
 - Make `ArrayBuilder` and `GroupBuilder` non-consuming
 - Add a fast-path to `Array::store_array_subset` if the array subset matches a chunk subset
 - **Breaking**: Make `ChunkKeyEncoding` a newtype.
   - **Breaking**: Changed `try_create_chunk_key_encoding` to `ChunkKeyEncoding::from_metadata`.
 - **Breaking**: Make `ChunkGrid` a newtype.
   - **Breaking**: Changed `try_create_chunk_grid` to `ChunkGrid::from_metadata`.
 - **Breaking**: Rename `StorageTransformerChain::new_with_metadatas` to `from_metadata`
 - **Breaking**: Rename `CodecChain::new_with_metadatas` to `from_metadata`
 - **Breaking**: Rename `DataTypeExtension::try_create_fill_value` to `fill_value_from_metadata`
 - **Breaking**: Rename `codec::extract_byte_ranges_rs` to `extract_byte_ranges_read_seek`

### Fixed

 - `BytesCodec` now defaults to native endian encoding as opposed to no encoding (only valid for 8 bit data types).
 - `storage::get_child_nodes` and `Node::new_with_store` now correctly propagate storage errors instead of treating all errors as missing metadata
 - `Group::new` now handles an implicit group (with a missing `zarr.json`)
 - `ZipStore` handle missing files
 - `ZipStore` no longer reads an internal file multiple times when extracting multiple byte ranges
 - `HTTPStore` improve error handling, check status codes

## [0.4.2] - 2023-10-06

### Added

 - Add `From<&str>` and `From<String>` for `Other` variants of `CodecError`, `StorageError`, `ChunkGridShapeError`, `StorePluginCreateError`

### Changed

 - Support parallel encoding, parallel decoding, and partial decoding with the `blosc` codec
   - Previously the blosc codec would read the whole chunk when partial decoding
 - `HTTPStore` range requests are now batched by default

### Fixed

 - Fixed `retrieve_chunk_subset` returning fill values on any `StorageError` instead of just `StorageError::KeyNotFound`

## [0.4.1] - 2023-10-04

### Added
 - Add `StorageHandle`, a clonable handle to borrowed unsized storage

### Changed
 - Support unsized storage (`TStorage: ?Sized` everywhere)

## [0.4.0] - 2023-10-04

### Added
 - Readable and listable `ZipStorageAdapter` behind `zip` feature
 - Readable `HTTPStore` behind `http` feature
 - Add `StorageValueIO`, a `std::io::Read` interface to a storage value.
 - Add `zip_array_write_read` and `http_array_read` examples

### Changed
 - Relax some dependency minimum versions
 - Add `size_hint()` to some array subset iterators
 - **Breaking**: Fix `LinearisedIndicesIterator` to use array shape instead of subset shape
   - `LinearisedIndicesIterator::new` and `ArraySubset::iter_linearised_indices` now require `array_shape` parameter
 - **Breaking**: Array subset shape no longer needs to be evenly divisible by chunk shape when creating a chunk iterator
   - Removes `ChunksIteratorError`
 - Add array subset iterator tests
 - **Breaking**: Remove unvalidated `from(String)` for store key/prefix
 - Validate that store prefixes and keys do not start with `/`
 - **Breaking**: Add `StorageError::Other` and `StorageError::Unsupported` variants
 - **Breaking** `FilesystemStore` now accepts a file and does not create directory on creation
   - adds `FilesystemStoreCreateError:InvalidBasePath` and removes `FilesystemStoreCreateError:InvalidBaseDirectory/ExistingFile`
 - **Breaking**: `ReadableStorageTraits::size()` to `u64` from `usize`
 - **Breaking**: Add `ReadableStorageTraits::size_key()`
 - Storage and store traits no longer require `Debug`
 - Add a default implementation for `WritableStorageTraits::erase_values`
 - Make array/group explicitly store `Arc<TStorage>`
 - `StorageTransformerChain` now only accepts `Arc` storage.
 - **Breaking**: Various group methods are now `#[must_use]`
 - Change `NodePath` internal representation to `PathBuf`
 - Remove unneeded '/' prefix strip in `StorePrefix`
 - **Breaking**: Remove storage trait implementations for `Arc<TStorage>`, now must explicitly deref
 - Implement `Eq` and `PartialEq` for `DataType`
 - **Breaking**: Use `u64` instead of `usize` for byte ranges/array index/shape etc. This makes it possible to index into larger arrays on 32-bit systems.

### Fixed
 - Fix store prefix to node path conversion and vice versa
 - Fix `StorePrefix::parent()` so it outputs a valid `StorePrefix`
 - Fix `StoreKey::parent()` for top level keys, e.g `"a"` now has parent prefix `"/"` instead of `None`
 - Print root node with `Node::hierarchy_tree`

## [0.3.0] - 2023-09-27

### Added
 - Add `CHANGELOG.md`

### Changed
 - Require the `ndarray` *feature* for examples
 - Remove the `ndarray` dev dependency
 - Remove the `ndarray` dependency for the `sharding` feature
 - Replace deprecated `tempdir` with `tempfile` for tests
 - **Breaking**: Change `ByteRange` enum to have `FromStart` and `FromEnd` variants
 - Substitute `blosc` dependency for `blosc-src` which builds blosc from source
 - **Breaking**: Rename `compression` field to `cname` in `BloscCodecConfigurationV1` for consistency with the zarr spec

## [0.2.0] - 2023-09-25

### Added
 - Initial public release

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs-v0.21.2...HEAD
[0.21.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.21.2
[0.21.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.21.1
[0.21.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.21.0
[0.20.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.20.1
[0.20.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.20.0
[0.20.0-beta.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.20.0-beta.2
[0.20.0-beta.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.20.0-beta.1
[0.20.0-beta.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.20.0-beta.0
[0.19.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.19.2
[0.19.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.19.1
[0.19.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.19.0
[0.18.3]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.18.3
[0.18.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.18.2
[0.18.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.18.1
[0.18.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.18.0
[0.18.0-beta.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.18.0-beta.0
[0.17.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.17.1
[0.17.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.17.0
[0.17.0-beta.3]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.17.0-beta.3
[0.17.0-beta.2]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.17.0-beta.2
[0.17.0-beta.1]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.17.0-beta.1
[0.17.0-beta.0]: https://github.com/LDeakin/zarrs/releases/tag/zarrs-v0.17.0-beta.0
[0.16.4]: https://github.com/LDeakin/zarrs/releases/tag/v0.16.4
[0.16.3]: https://github.com/LDeakin/zarrs/releases/tag/v0.16.3
[0.16.2]: https://github.com/LDeakin/zarrs/releases/tag/v0.16.2
[0.16.1]: https://github.com/LDeakin/zarrs/releases/tag/v0.16.1
[0.16.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.16.0
[0.15.1]: https://github.com/LDeakin/zarrs/releases/tag/v0.15.1
[0.15.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.15.0
[0.14.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.14.0
[0.13.3]: https://github.com/LDeakin/zarrs/releases/tag/v0.13.3
[0.13.2]: https://github.com/LDeakin/zarrs/releases/tag/v0.13.2
[0.13.1]: https://github.com/LDeakin/zarrs/releases/tag/v0.13.1
[0.13.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.13.0
[0.12.5]: https://github.com/LDeakin/zarrs/releases/tag/v0.12.5
[0.12.4]: https://github.com/LDeakin/zarrs/releases/tag/v0.12.4
[0.12.3]: https://github.com/LDeakin/zarrs/releases/tag/v0.12.3
[0.12.2]: https://github.com/LDeakin/zarrs/releases/tag/v0.12.2
[0.12.1]: https://github.com/LDeakin/zarrs/releases/tag/v0.12.1
[0.12.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.12.0
[0.11.6]: https://github.com/LDeakin/zarrs/releases/tag/v0.11.6
[0.11.5]: https://github.com/LDeakin/zarrs/releases/tag/v0.11.5
[0.11.4]: https://github.com/LDeakin/zarrs/releases/tag/v0.11.4
[0.11.3]: https://github.com/LDeakin/zarrs/releases/tag/v0.11.3
[0.11.2]: https://github.com/LDeakin/zarrs/releases/tag/v0.11.2
[0.11.1]: https://github.com/LDeakin/zarrs/releases/tag/v0.11.1
[0.11.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.11.0
[0.10.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.10.0
[0.9.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.9.0
[0.8.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.8.0
[0.7.3]: https://github.com/LDeakin/zarrs/releases/tag/v0.7.3
[0.7.2]: https://github.com/LDeakin/zarrs/releases/tag/v0.7.2
[0.7.1]: https://github.com/LDeakin/zarrs/releases/tag/v0.7.1
[0.7.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.7.0
[0.6.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.6.0
[0.5.1]: https://github.com/LDeakin/zarrs/releases/tag/v0.5.1
[0.5.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.5.0
[0.4.2]: https://github.com/LDeakin/zarrs/releases/tag/v0.4.2
[0.4.1]: https://github.com/LDeakin/zarrs/releases/tag/v0.4.1
[0.4.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.4.0
[0.3.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.3.0
[0.2.0]: https://github.com/LDeakin/zarrs/releases/tag/v0.2.0

[`zarr-extensions`]: https://github.com/zarr-developers/zarr-extensions

[@LDeakin]: https://github.com/LDeakin
[@lorenzocerrone]: https://github.com/lorenzocerrone
[@dustinlagoy]: https://github.com/dustinlagoy
[@sk1p]: https://github.com/sk1p
[@niklasmueboe]: https://github.com/niklasmueboe
[@ilan-gold]: https://github.com/ilan-gold
[@jder]: https://github.com/jder
