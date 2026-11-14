# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add `read_decoded_bytes_into` for exact-length streaming decoding, stopping after the first extra decoded byte
- Add the `unsafe` `BytesToBytesCodecDecodeInto` trait, `BytesToBytesCodecTraits::as_decode_into`, `BytesDecodeSource::decode_into`, `copy_decoded_bytes_into`, and the `unsafe` `ArrayBytesFixedDisjointView::as_mut_slice` for exact-length decoding into (possibly uninitialised) output views
- Add `ArrayBytesDecodeIntoInput` and `BytesDecodeSource` for single-use deferred decoding into array targets
- Add `ArrayBytesFixedDisjointView::copy_from_slice_with` to transform each region of a view once it is copied
- Add per-operation `Resources` and `ComputeResources`
- Add `UnboundArrayTo{Array,Bytes}CodecTraits`, the codecs held by `Codec::ArrayTo{Array,Bytes}`
  - `with_context` binds a codec to a data type, fill value, and `CodecSpecificOptions`, and codecs must bind any nested codecs with the same options
  - The fill value is `None` if it is undefined (e.g. in the `data_codecs` of an `optional` codec with a `null` fill value), and codecs whose encoded representation depends on the fill value must reject it
- Add `CodecCreateError` for codec creation, reconfiguration, and binding failures, and `CodecError::CodecCreateError`
- Add `CodecError::UndefinedFillValue`, returned by partial decoders and encoders for missing input if the fill value is undefined
- Add `ChunkGrid{Encoded,Decoded}[Ref]` and `[Async]ArrayPartialDecoderSubchunkingTraits::local_subchunk_grid[s]` for chunk-local subchunk grids
- Add `validate_chunk_{element_indices,subset}`
- Add `ArrayToBytesCodecTraits::is_decode_passthrough`
- Add `ArrayToBytesCodecTraits::as_subchunking`, returning the optional `ArrayToBytesCodecSubchunkingTraits` capability for codecs that expose subchunks
  - `decoded_subchunk_grids` resolves ordered subchunk-grid hierarchies, and `subchunk_codecs` returns the shape-independent codec for each level (outermost first)
- Add `ArrayBytesFixedDisjointView::{copy_elements_from_slice,fill_elements}` for scattering elements into a view
- Add `ArrayBytesDecodeIntoTarget::{with_optional_mask,with_optional_masks,shape,subdivide,permuted_axes,reshaped,reborrow}` to build, subdivide, transform and reborrow targets without depending on their structure
  - `with_optional_mask(s)` return the new `WithOptionalMaskError`, which holds the target and mask(s) and converts to the new `IncompatibleOptionalMaskError` (`CodecError::IncompatibleOptionalMask`), unless each mask has one byte per element and addresses the same elements of its bytes as the data (the same shape, offset and strides)
- Add `ArrayToArrayCodecTraits::decode_into`, which decodes into an output target (by default with `decode` and a copy), and which a codec chain calls for the outermost array-to-array codec that cannot remap the target
- Add the `unsafe` `ArrayBytesFixedDisjointView::for_each_region_mut` for writing the (possibly uninitialised) contiguous regions of a view in order
- Add `ArrayToArrayCodecTraits::decode_into_encoded_target` (default `None`), which lets a codec that only reorders or relabels elements on decode express its decode as a transform of the output target, so a codec chain decodes the encoded representation directly into it
- Add `ArrayBytesFixedDisjointView::{new_strided,offset,strides}` for views of strided grids of elements (e.g. column-major or transposed outputs), and the safe `ArrayBytesFixedDisjointView::{permuted_axes,reshaped}`, which return a view of the same elements with permuted dimensions or another shape
- Add `ArrayBytesFixedDisjointView::data_type_size`
- Add `ArrayBytes::with_optional_masks`, with masks ordered from outermost to innermost like `ArrayBytesDecodeIntoTarget::with_optional_masks`
- Add `ArrayBytesOffsets{Slice,Iter,RangesIter}` and `ArrayBytesOffsetsElement`, and implement `IntoIterator` for `&ArrayBytesOffsets` and `&ArrayBytesOffsetsSlice`
- Implement `[Async]BytesPartial{Encoder,Decoder}Traits` for `(TStorage: *StorageTraits, StoreKey)`
- Implement `From<CowBytes>` for `ArrayBytes`

### Changed
- **Breaking**: `ArrayToBytesCodecTraits::decode_into` accepts `ArrayBytesDecodeIntoInput`, resolving immediate or deferred encoded bytes without cloning them
- **Breaking**: Pass `ComputeResources` after codec options on in-memory operations and `Resources` on storage-capable partial interfaces and factories; remove compute-target control and `chunk_concurrent_minimum` (including its getter, setter, and `with_` method) from `CodecOptions`
- **Breaking**: Refactor `ArrayTo{Array,Bytes}CodecTraits` into codecs _bound_ to a data type and fill value and validated at array creation time
  - Add `ArrayCodecTraits::{as_any,data_type,fill_value}`, and `ArrayCodecTraits` no longer requires `CodecTraits`
  - `ArrayCodecTraits::fill_value()` and `ArrayToArrayCodecTraits::encoded_fill_value()` return `Option<&FillValue>`, which is `None` if the fill value is undefined; `encoded_fill_value` no longer has a default implementation
  - Remove the `data_type` and `fill_value` parameters from encoding, decoding, representation, and partial encoder/decoder methods
  - Replace `ArrayCodecTraits::partial_decode_granularity` and `ArrayToArrayCodecTraits::decoded_shape` with `ArrayToArrayCodecTraits::partial_decode_granularity`, which maps an encoded granularity to the decoded representation
- **Breaking**: Add the `ArrayToArrayCodecGeometryTraits` supertrait, which maps shapes, element indices, subsets, and chunk grids between the decoded and encoded representations of a codec
  - The `ArrayToArrayCodecGeometryIdentityTraits` marker trait implements it for codecs that preserve chunk shape and element ordering
  - `encoded_shape` moves here from `ArrayToArrayCodecTraits` and no longer has a default implementation
  - `encoded_chunk_grid` and `decoded_subchunk_grid` map whole-array chunk and subchunk grids
  - `decoded_local_subchunk_grid[s]` map chunk-local subchunk grids back to a decoded chunk
  - `encoded_element_indices` maps element indices to an encoded chunk
  - `encoded_subset` maps a decoded subset to an exact rectangular subset of an encoded chunk
- **Breaking**: Add the `[Async]ArrayPartialDecoderSubchunkingTraits` supertraits of `[Async]ArrayPartialDecoderTraits`, which hold the subchunking surface of a partial decoder
  - The `ArrayPartialDecoderNoSubchunkingTraits` marker trait default implements both for partial decoders without subchunks
- **Breaking**: Codec creation (`Codec::from_metadata`, `CodecTraitsV{2,3}::create`, `Codec[Runtime]PluginV{2,3}`) returns `CodecCreateError` rather than `PluginCreateError`, and `BytesToBytesCodecTraits::with_codec_specific_options` is fallible
- **Breaking**: Add the required `[Async]BytesPartialEncoderTraits::encode` method, which replaces the entire chunk
- **Breaking**: Use `CowBytes` for encoded and raw bytes throughout the crate to avoid copies in some circumstances
  - Replaces `ArrayBytesRaw = Cow<'a, [u8]>`, and is re-exported from `zarrs_storage` as `zarrs_codec::CowBytes`
  - `[Async]BytesPartialDecoderTraits` is implemented for `CowBytes<'static>` rather than `Cow<'static, [u8]>`
- **Breaking**: `ArrayBytesOffsets` stores native-endian `u32` or `u64` offsets in a `CowBytes<'static>` instead of a `Cow<'a, [usize]>` and no longer has a lifetime parameter
  - Cloning shares the offset allocation rather than copying its elements
  - Add `ArrayBytesOffsets::from_ne_bytes`, which retains shared and suitably aligned bytes without copying
  - Add `ArrayBytesOffsets::{as_ne_bytes,into_u32_ne_bytes,into_u64_ne_bytes}`, which do not copy if the width matches
  - `Deref<Target = [usize]>` is removed, use the new accessors instead (e.g. `element_range`, `element_ranges`, `iter`, `get`, `last`, `as_slice`, `as_u32`, `as_u64`, `into_u32_vec`, `into_u64_vec`)
  - `ArrayBytesOffsets::new[_unchecked]` and the `TryFrom` implementations accept `u32` or `u64` offsets (via the sealed `ArrayBytesOffsetsElement` trait) rather than `usize`
  - `ArrayBytesOffsets::into_owned` is removed and `ArrayBytesVariableLength::{offsets,into_parts}` return offsets without a lifetime, as the offsets are always owned
- **Breaking**: Rename `ArrayBytesRawOffsets{Create,OutOfBounds}Error` to `ArrayBytesOffsets{Create,OutOfBounds}Error` and `CodecError::RawBytesOffsets{Create,OutOfBounds}` to `CodecError::ArrayBytesOffsets{Create,OutOfBounds}`
  - Add `ArrayBytesOffsetsCreateError::{ExceedsUsizeMax,InvalidBytesLength}`
- **Breaking**: `ArrayBytesFixedDisjointView::new` takes `impl Into<UnsafeCellSlice<MaybeUninit<u8>>>` so views can cover uninitialised memory (e.g. `Vec::spare_capacity_mut()`), and `UnsafeCellSlice<u8>` arguments still work
  - Views never write uninitialised bytes
- **Breaking**: `ArrayBytesFixedDisjointView` is a strided view of elements in C order of its own shape
  - `shape()` returns the shape of the view rather than of the underlying array, and `subset()` is removed
  - `subdivide` takes a subset relative to the view rather than to the underlying array
  - `new` no longer borrows `shape` for the lifetime of the view
- **Breaking**: `ArrayBytesDecodeIntoTarget` variants own their `ArrayBytesFixedDisjointView`s rather than borrowing them, and `From<ArrayBytesFixedDisjointView>` replaces `From<&mut ArrayBytesFixedDisjointView>`
  - `ArrayBytesDecodeIntoTarget::Optional` holds the new `ArrayBytesDecodeIntoTargetOptional`, which can only be created with `with_optional_mask` and has read-only `data`/`mask` accessors and `into_parts`
  - `ArrayToBytesCodecTraits::decode_into`, `[Async]ArrayPartialDecoderTraits::partial_decode_into`, `copy_fill_value_into`, and `decode_into_array_bytes_target` take `&mut ArrayBytesDecodeIntoTarget`, so targets are not moved through each layer of decoding
- **Breaking**: Bump `zarrs_chunk_grid` to 0.6.0, `zarrs_data_type` to 0.10.0, and `zarrs_storage` to 0.5.0
- **Breaking**: Bump MSRV to 1.92 (11 December, 2025)
- The default array-to-array and array-to-bytes partial encoders do not read the existing chunk if the update replaces the entire chunk
- Make `async-trait` optional and enable it only with the `async` feature
- Document that a `CodecOptions` concurrent target of zero is unconstrained
- Improve performance
  - Copy small contiguous regions and elements directly when extracting array subsets and writing or filling output views
  - `ArrayBytesFixedDisjointView::{copy_elements_from_slice,fill_elements}` map ascending positions by stepping through the contiguous regions of the view, rather than unravelling each position
  - `ArrayBytes::extract_array_subset` copies the contiguous runs of a subset into an output of the subset size without boxed iterators, rather than first collecting the runs
  - Compute the contiguous regions of `ArrayBytesFixedDisjointView` once per view, without allocating for up to four dimensions
  - Reserve capacity for storage-backed partial decoding, mutex-backed decoding, and fallible output-shape conversions
  - Box the private array subsets in `SubsetOutOfBoundsError` to keep codec errors small
- Bump `zarrs_metadata` to 0.7.6, `itertools` to 0.15.0, and `unsafe_cell_slice` to 0.2.4

### Removed
- **Breaking**: Remove `[Async]StoragePartial{Encoder,Decoder}`, use `(storage, key)` instead
- **Breaking**: Remove `[Async]{Array,Bytes}PartialEncoderTraits::into_dyn_decoder()`
- **Breaking**: Remove `ArrayTo{Array,Bytes}CodecTraits::with_codec_specific_options`, codecs read their options in `UnboundArrayTo{Array,Bytes}CodecTraits::with_context` instead
- **Breaking**: Remove `CodecTraits::as_any`, superseded by `ArrayCodecTraits::as_any` for bound codecs
- **Breaking**: Remove `CodecError::UnsupportedDataTypeCodec`, unsupported data types are rejected when binding with `CodecCreateError::UnsupportedDataType`

### Fixed
- `[Async]ByteIntervalPartialDecoder` no longer reads beyond its interval for a `ByteRange::FromStart(offset, None)` with a nonzero `offset`
- `[Async]ByteIntervalPartialDecoder` returns `CodecError::InvalidByteRangeError` for a byte range outside of its interval, or an interval end that overflows, instead of panicking or reading bytes outside of its interval
- `BytesPartialDecoderTraits` for `Mutex<Option<Vec<u8>>>` returns `InvalidByteRangeError` instead of panicking on out-of-bounds suffix, offset, or overflowing byte ranges
- `ArrayBytes::validate` rejects optional masks of the wrong length or with values other than 0/1
- `ArrayBytes::is_fill_value` compares variable-length elements individually using their offsets, rather than comparing concatenated bytes
- `ArrayBytesOptional` equality ignores elements beneath nulls, is symmetric, and no longer panics on malformed masks
- The default bytes-to-bytes partial encoder replaces the entire chunk, rather than leaving trailing bytes of the existing chunk if the re-encoded chunk is shorter
  - The existing chunk is also read once rather than once per stacked bytes-to-bytes codec
- The default array-to-array and array-to-bytes partial encoders replace an existing chunk rather than erasing it first, so a failed encode or write leaves it intact

## [0.2.1] - 2026-03-21

### Added
- Add `CodecSpecificOptions` for codec-specific runtime configuration
- Add `with_codec_specific_options` default method to `ArrayToArrayCodecTraits`, `ArrayToBytesCodecTraits`, and `BytesToBytesCodecTraits`

## [0.2.0] - 2026-02-02

### Added
- Add `CodecTraitsV2` and  `CodecTraitsV3` traits
- Add `CodecError::UnsupportedDataTypeCodec` variant for data type codec support errors
- Add `ExpectedFixedLengthBytesError`, `ExpectedVariableLengthBytesError`, and `ExpectedOptionalBytesError` error types

### Changed
- **Breaking**: Remove `create_fn` parameter from `CodecPluginV2::create()` and add `T: CodecTraitsV2` bound
- **Breaking**: Remove `create_fn` parameter from `CodecPluginV3::create()` and add `T: CodecTraitsV3` bound
- **Breaking**: Rename `ArrayRawBytesOffsetsOutOfBoundsError` to `ArrayBytesRawOffsetsOutOfBoundsError`
- **Breaking**: Rename `ArrayRawBytesOffsetsCreateError` to `ArrayBytesRawOffsetsCreateError`
- **Breaking**: `ArrayBytes::into_fixed()` now returns `Result<_, ExpectedFixedLengthBytesError>` instead of `Result<_, CodecError>`
- **Breaking**: `ArrayBytes::into_variable()` now returns `Result<_, ExpectedVariableLengthBytesError>` instead of `Result<_, CodecError>`
- **Breaking**: `ArrayBytes::into_optional()` now returns `Result<_, ExpectedOptionalBytesError>` instead of `Result<_, CodecError>`
- **Breaking**: `CodecError::ExpectedFixedLengthBytes`, `CodecError::ExpectedVariableLengthBytes`, and `CodecError::ExpectedOptionalBytes` now wrap their respective dedicated error types

### Removed
- **Breaking**: Remove `CodecError::ExpectedNonOptionalBytes` (replaced with `CodecError::ExpectedOptionalBytes`)
- **Breaking**: Remove `ArrayBytes::into_optional_bytes()` method (use `into_optional()` instead)
- **Breaking**: Remove `optional_nesting_depth`, `build_nested_optional_target`, `merge_chunks_vlen`, `merge_chunks_vlen_optional`, and `extract_decoded_regions_vlen` (moved to `zarrs` as private functions)

## [0.1.0] - 2026-01-14

### Added
- Split from the `zarrs::array::codec` module of `zarrs` 0.23.0-beta.5

[unreleased]: https://github.com/zarrs/zarrs/compare/zarrs_codec-v0.2.1...HEAD
[0.2.1]: https://github.com/zarrs/zarrs/releases/tag/zarrs_codec-v0.2.1
[0.2.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_codec-v0.2.0
[0.1.0]: https://github.com/zarrs/zarrs/releases/tag/zarrs_codec-v0.1.0
