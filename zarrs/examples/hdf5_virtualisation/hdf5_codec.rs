//! A read-only HDF5 `ArrayToBytes` codec (`zarrs.experimental.hdf5`) with subchunking.
//!
//! - **Metadata & chunk offsets**: Uses `hdf5-reader` solely to parse HDF5 headers and resolve
//!   internal chunk byte offsets, encoded sizes, and filter masks directly over `zarrs`'s
//!   [`SyncBytesPartialDecoderTraits`]. Only overlapping HDF5 chunk byte ranges are fetched.
//! - **Subchunk decoding**: Decodes each raw HDF5 chunk byte range using `zarrs`'s own
//!   [`CodecChain`] (`BytesCodec`, `ShuffleCodec`, `ZlibCodec`, `Fletcher32Codec`, plus dynamic
//!   filters `Bz2Codec`, `BloscCodec`, `ZstdCodec` when feature-enabled).
//! - **Subchunking**: Implements [`ArrayToBytesCodecSubchunkingTraits`] and
//!   [`SyncArrayPartialDecoderSubchunkingTraits`] so `Array::subchunk_grid()`,
//!   `Array::local_subchunk_grid()`, `Array::retrieve_subchunk()`, and `SubchunkCache` all work.
//! - **Read-only**: HDF5 files are multi-object containers (multiple datasets, groups, links,
//!   attributes, dimension scales) that do not map safely to Zarr's per-chunk overwrite model.
//!   See `spec/codecs/hdf5/README.md` for the full specification and rationale.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use derive_more::Display;
use hdf5_reader::chunk_index::ChunkEntry;
use hdf5_reader::messages::HdfMessage;
use hdf5_reader::messages::datatype::Datatype as Hdf5Datatype;
use hdf5_reader::messages::filter_pipeline::{
    FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_NBIT, FILTER_SCALEOFFSET, FILTER_SHUFFLE,
    FILTER_SZIP, FilterDescription,
};
use hdf5_reader::messages::layout::{ChunkIndexing, DataLayout};
use hdf5_reader::{ByteOrder as Hdf5ByteOrder, Hdf5File, Storage, StorageBuffer};
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use zarrs::array::chunk_grid::{
    ChunkEdgeLengths, RectilinearChunkGrid, RegularBoundedChunkGrid, RegularChunkGrid,
};
use zarrs::array::codec::{BytesCodec, CodecChain, Fletcher32Codec, ShuffleCodec, ZlibCodec};
use zarrs::array::{
    ArrayBytes, ArrayBytesTarget, ArraySubset, ArraySubsetTraits, BytesRepresentation, ChunkGrid,
    ChunkShape, ChunkShapeTraits, CodecOptions, CowBytes, DataType, FillValue, Indexer, Resources,
    data_type,
};
use zarrs::metadata::v3::MetadataV3;
use zarrs::metadata::{Configuration, ConfigurationSerialize, Endianness};
use zarrs_chunk_grid::{ChunkGridCreateError, ChunkGridTraits};
use zarrs_codec::{
    ArrayCodecTraits, ArrayToBytesCodecSubchunkingTraits, ArrayToBytesCodecTraits,
    BytesDecodeInput, BytesToBytesCodecTraits, ChunkGridDecoded, ChunkGridDecodedRef, Codec,
    CodecCreateError, CodecError, CodecMetadataOptions, CodecPluginV3, CodecSpecificOptions,
    CodecTraits, CodecTraitsV3, ComputeResources, PartialDecoderCapability,
    PartialEncoderCapability, RecommendedConcurrency, SyncArrayPartialDecoderSubchunkingTraits,
    SyncArrayPartialDecoderTraits, SyncBytesPartialDecoderTraits, UnboundArrayToBytesCodecTraits,
    copy_fill_value_into, decode_into_array_bytes_target,
};
use zarrs_plugin::{PluginCreateError, ZarrVersion};
use zarrs_storage::StorageError;
use zarrs_storage::byte_range::ByteRange;

/// Dynamic HDF5 filter IDs registered with The HDF Group.
const FILTER_BZIP2: u16 = 307;
const FILTER_BLOSC: u16 = 32001;
const FILTER_LZ4: u16 = 32004;
const FILTER_BITSHUFFLE: u16 = 32008;
const FILTER_ZFP: u16 = 32013;
const FILTER_ZSTD: u16 = 32015;
const FILTER_SZ3: u16 = 32024;

/// Configuration for [`Hdf5Codec`].
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display)]
#[serde(deny_unknown_fields)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub(crate) struct Hdf5CodecConfiguration {
    /// HDF5 dataset path inside each file (e.g. `"data"` or `"/group/data"`).
    pub(crate) dataset: String,
    /// Optional consistent HDF5 internal chunk shape across files (matches `sharding_indexed`'s
    /// `chunk_shape` metadata key).
    ///
    /// When set, enables a globally resolvable subchunk grid (`Array::subchunk_grid`,
    /// `retrieve_subchunk`, `SubchunkCache`). When omitted, `decoded_subchunk_grids` reports
    /// `ChunkLocal` and the subchunk grid is resolved per file via `local_subchunk_grid`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) chunk_shape: Option<ChunkShape>,
    /// Optional Zarr codec chain declaring the per-subchunk encoding statically for
    /// [`ArrayToBytesCodecSubchunkingTraits::subchunk_codecs`]. On decode, the actual HDF5
    /// datatype endianness and filter pipeline are always read from each HDF5 file's object
    /// header.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) codecs: Option<Vec<MetadataV3>>,
}

impl ConfigurationSerialize for Hdf5CodecConfiguration {}

/// A read-only `ArrayToBytes` codec backed by HDF5 dataset files.
#[derive(Clone, Debug)]
pub(crate) struct Hdf5Codec {
    dataset: String,
    chunk_shape: Option<ChunkShape>,
    inner_codecs: Option<Arc<CodecChain>>,
}

zarrs_plugin::impl_extension_aliases!(Hdf5Codec, v3: "zarrs.experimental.hdf5");

inventory::submit! {
    CodecPluginV3::new::<Hdf5Codec>()
}

impl Hdf5Codec {
    /// Create a new read-only [`Hdf5Codec`].
    #[must_use]
    pub(crate) fn new(
        dataset: impl Into<String>,
        chunk_shape: Option<ChunkShape>,
        inner_codecs: Option<Arc<CodecChain>>,
    ) -> Self {
        Self {
            dataset: dataset.into(),
            chunk_shape,
            inner_codecs,
        }
    }

    /// Create a new [`Hdf5Codec`] from configuration.
    pub(crate) fn new_with_configuration(
        configuration: &Hdf5CodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        let inner_codecs = match &configuration.codecs {
            Some(metadatas) => Some(Arc::new(
                CodecChain::from_metadata(metadatas)
                    .map_err(|err| PluginCreateError::Other(err.to_string()))?,
            )),
            None => None,
        };
        Ok(Self::new(
            configuration.dataset.clone(),
            configuration.chunk_shape.clone(),
            inner_codecs,
        ))
    }
}

impl CodecTraitsV3 for Hdf5Codec {
    fn create(metadata: &MetadataV3) -> Result<Codec, CodecCreateError> {
        let configuration: Hdf5CodecConfiguration = metadata.to_typed_configuration()?;
        Ok(Codec::ArrayToBytes(Arc::new(
            Hdf5Codec::new_with_configuration(&configuration)?,
        )))
    }
}

impl CodecTraits for Hdf5Codec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        Some(
            Hdf5CodecConfiguration {
                dataset: self.dataset.clone(),
                chunk_shape: self.chunk_shape.clone(),
                codecs: self
                    .inner_codecs
                    .as_ref()
                    .map(|c| c.create_metadatas(options)),
            }
            .into(),
        )
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl UnboundArrayToBytesCodecTraits for Hdf5Codec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn UnboundArrayToBytesCodecTraits> {
        self
    }

    fn with_context(
        &self,
        data_type: DataType,
        fill_value: Option<FillValue>,
        codec_specific_options: &CodecSpecificOptions,
    ) -> Result<Arc<dyn ArrayToBytesCodecTraits>, CodecCreateError> {
        validate_supported_zarr_data_type(&data_type)?;
        let fill_value = fill_value.ok_or_else(|| {
            CodecCreateError::Other("the hdf5 codec requires a defined fill value".to_string())
        })?;

        // Validate any statically declared subchunk codec chain upfront against the array's
        // data type and fill value (before any HDF5 chunk is opened).
        if let Some(configured_chain) = &self.inner_codecs {
            let _ = configured_chain.with_context(
                data_type.clone(),
                Some(fill_value.clone()),
                codec_specific_options,
            )?;
        }

        Ok(Arc::new(Hdf5CodecBound {
            dataset: self.dataset.clone(),
            chunk_shape: self.chunk_shape.clone(),
            data_type,
            fill_value,
        }))
    }
}

/// [`Hdf5Codec`] bound to an array's data type and fill value.
#[derive(Clone, Debug)]
pub(crate) struct Hdf5CodecBound {
    dataset: String,
    chunk_shape: Option<ChunkShape>,
    data_type: DataType,
    fill_value: FillValue,
}

impl ArrayCodecTraits for Hdf5CodecBound {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn fill_value(&self) -> Option<&FillValue> {
        Some(&self.fill_value)
    }

    fn recommended_concurrency(
        &self,
        shape: &[NonZeroU64],
    ) -> Result<RecommendedConcurrency, CodecError> {
        if let Some(chunk_shape) = &self.chunk_shape
            && chunk_shape.len() == shape.len()
        {
            let num_subchunks: usize = std::iter::zip(shape, chunk_shape)
                .map(|(s, c)| usize::try_from(s.get().div_ceil(c.get())).unwrap())
                .product();
            return Ok(RecommendedConcurrency::new_maximum(num_subchunks.max(1)));
        }
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

impl ArrayToBytesCodecSubchunkingTraits for Hdf5CodecBound {
    fn decoded_subchunk_grids(
        &self,
        decoded_chunk_grid: ChunkGridDecodedRef<'_>,
    ) -> Result<Vec<ChunkGridDecoded>, ChunkGridCreateError> {
        let Some(chunk_shape) = &self.chunk_shape else {
            return Ok(vec![ChunkGridDecoded::ChunkLocal]);
        };
        let grid = match decoded_chunk_grid {
            ChunkGridDecodedRef::None => ChunkGridDecoded::None,
            ChunkGridDecodedRef::Array(chunk_grid) if chunk_grid.array_shape().contains(&0) => {
                ChunkGridDecoded::None
            }
            ChunkGridDecodedRef::Array(chunk_grid) => {
                ChunkGridDecoded::Array(build_global_subchunk_grid(chunk_grid, chunk_shape)?)
            }
            ChunkGridDecodedRef::ChunkLocal => ChunkGridDecoded::ChunkLocal,
        };
        Ok(vec![grid])
    }

    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
        // HDF5 subchunks cannot be decoded safely via a standalone Zarr subchunk codec because:
        // 1. Each chunk's `filter_mask` (skipped filters) lives out-of-band in the HDF5 chunk index.
        // 2. Edge chunks are stored padded to the full HDF5 `chunk_shape`, not the clipped logical
        //    subchunk shape reported by the subchunk grid.
        Vec::new()
    }
}

impl ArrayToBytesCodecTraits for Hdf5CodecBound {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
        self
    }

    fn as_subchunking(&self) -> Option<&dyn ArrayToBytesCodecSubchunkingTraits> {
        Some(self)
    }

    fn encoded_representation(
        &self,
        _shape: &[NonZeroU64],
    ) -> Result<BytesRepresentation, CodecError> {
        Ok(BytesRepresentation::UnboundedSize)
    }

    fn encode<'a>(
        &self,
        _bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _options: &CodecOptions,
        _resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        Err(CodecError::Other(
            "the hdf5 codec is read-only; writing HDF5 containers is unsupported".to_string(),
        ))
    }

    fn decode<'a>(
        &self,
        bytes: CowBytes<'a>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        let input_handle: Arc<dyn SyncBytesPartialDecoderTraits> = Arc::new(bytes.into_static());
        let io_resources = Resources::default().with_compute(resources.clone());
        let decoder = Hdf5PartialDecoder::new(
            input_handle,
            shape,
            self.dataset.clone(),
            self.data_type().clone(),
            self.fill_value.clone(),
            options,
            &io_resources,
        )?;
        let subset = ArraySubset::new_with_shape(bytemuck::must_cast_slice::<_, u64>(shape));
        Ok(decoder
            .partial_decode(&subset, options, &io_resources)?
            .into_owned())
    }

    fn decode_into(
        &self,
        input: BytesDecodeInput<'_>,
        shape: &[NonZeroU64],
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<(), CodecError> {
        let bytes = input.into_bytes(options, resources)?;
        let input_handle: Arc<dyn SyncBytesPartialDecoderTraits> = Arc::new(bytes.into_static());
        let io_resources = Resources::default().with_compute(resources.clone());
        let decoder = Hdf5PartialDecoder::new(
            input_handle,
            shape,
            self.dataset.clone(),
            self.data_type().clone(),
            self.fill_value.clone(),
            options,
            &io_resources,
        )?;
        let subset = ArraySubset::new_with_shape(bytemuck::must_cast_slice::<_, u64>(shape));
        decoder.partial_decode_into(&subset, output_target, options, &io_resources)
    }

    fn partial_decoder(
        self: Arc<Self>,
        input_handle: Arc<dyn SyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Arc<dyn SyncArrayPartialDecoderTraits>, CodecError> {
        Ok(Arc::new(Hdf5PartialDecoder::new(
            input_handle,
            shape,
            self.dataset.clone(),
            self.data_type().clone(),
            self.fill_value.clone(),
            options,
            resources,
        )?))
    }
}

// ---------------------------------------------------------------------------
// Partial Decoder & Byte-Range Storage Bridge
// ---------------------------------------------------------------------------

/// Adapts `zarrs`'s [`SyncBytesPartialDecoderTraits`] to `hdf5_reader::Storage` for metadata
/// and chunk-index byte-range reads (never fetching raw chunk payloads through `hdf5-reader`).
struct PartialDecoderStorage {
    input_handle: Arc<dyn SyncBytesPartialDecoderTraits>,
    options: CodecOptions,
    resources: Resources,
}

impl Storage for PartialDecoderStorage {
    fn len(&self) -> u64 {
        u64::MAX
    }

    fn read_range(&self, offset: u64, len: usize) -> hdf5_reader::error::Result<StorageBuffer> {
        let range = ByteRange::FromStart(offset, Some(len as u64));
        match self
            .input_handle
            .partial_decode(range, &self.options, &self.resources)
        {
            Ok(Some(bytes)) => Ok(StorageBuffer::from_vec(bytes.into_vec())),
            Ok(None) => Err(hdf5_reader::error::Error::UnexpectedEof {
                offset,
                needed: len as u64,
                available: 0,
            }),
            Err(CodecError::InvalidByteRangeError(_)) => {
                // Retry bounded to EOF (e.g. superblock probe reading 256 bytes on a small file).
                let tail = self
                    .input_handle
                    .partial_decode(
                        ByteRange::FromStart(offset, None),
                        &self.options,
                        &self.resources,
                    )
                    .map_err(|err| hdf5_reader::error::Error::InvalidData(err.to_string()))?;
                match tail {
                    Some(bytes) => {
                        let slice = &bytes[..bytes.len().min(len)];
                        Ok(StorageBuffer::from_vec(slice.to_vec()))
                    }
                    None => Err(hdf5_reader::error::Error::UnexpectedEof {
                        offset,
                        needed: len as u64,
                        available: 0,
                    }),
                }
            }
            Err(err) => Err(hdf5_reader::error::Error::InvalidData(err.to_string())),
        }
    }
}

/// Resolved on-disk storage layout for an HDF5 dataset.
enum ResolvedLayout {
    Chunked {
        subchunk_shape: ChunkShape,
        subchunk_grid: RegularBoundedChunkGrid,
        /// Exact `(address, byte_size, filter_mask)` keyed by subchunk grid indices.
        chunks: HashMap<Vec<u64>, (u64, u64, u32)>,
        /// Full uncompressed size of one HDF5 chunk (used when `entry.size == 0`).
        uncompressed_chunk_bytes: u64,
    },
    Contiguous {
        address: u64,
        #[allow(dead_code)]
        size: u64,
    },
    Compact {
        data: Vec<u8>,
    },
}

/// Internal bytes-to-bytes pipeline that applies a single pre-built sequence of HDF5 filters
/// while honouring per-chunk `filter_mask` bits (bit `i == 1` skips filter `i`) without
/// reconstructing a [`CodecChain`] per chunk.
struct FilterMaskPipeline {
    codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>,
}

impl FilterMaskPipeline {
    fn new(codecs: Vec<Arc<dyn BytesToBytesCodecTraits>>) -> Self {
        Self { codecs }
    }

    fn decode_with_mask<'a>(
        &self,
        mut bytes: CowBytes<'a>,
        filter_mask: u32,
        decoded_size: u64,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<CowBytes<'a>, CodecError> {
        if self.codecs.is_empty() {
            return Ok(bytes);
        }

        // Compute intermediate decoded byte representations for active filters in encode order.
        let mut active: Vec<(usize, BytesRepresentation)> = Vec::with_capacity(self.codecs.len());
        let mut repr = BytesRepresentation::FixedSize(decoded_size);
        for (i, codec) in self.codecs.iter().enumerate() {
            if (filter_mask & (1u32 << i)) != 0 {
                continue;
            }
            active.push((i, repr));
            repr = codec.encoded_representation(&repr);
        }

        // Decode active filters in reverse order (matching HDF5 filter pipeline reversal).
        for (i, decoded_repr) in active.into_iter().rev() {
            bytes = self.codecs[i].decode(bytes, &decoded_repr, options, resources)?;
        }
        Ok(bytes)
    }
}

/// Parsed HDF5 dataset metadata held by [`Hdf5PartialDecoder`].
struct ParsedDataset {
    layout: ResolvedLayout,
    /// Pre-bound `bytes` codec matching the HDF5 dataset's endianness.
    bytes_codec: Arc<dyn ArrayToBytesCodecTraits>,
    /// Single reusable filter pipeline honouring per-chunk `filter_mask`.
    filter_pipeline: FilterMaskPipeline,
}

impl ParsedDataset {
    fn decode_subchunk_bytes(
        &self,
        raw_bytes: CowBytes<'_>,
        subchunk_shape: &[NonZeroU64],
        uncompressed_chunk_bytes: u64,
        filter_mask: u32,
        options: &CodecOptions,
        resources: &ComputeResources,
    ) -> Result<ArrayBytes<'static>, CodecError> {
        let filtered = self.filter_pipeline.decode_with_mask(
            raw_bytes,
            filter_mask,
            uncompressed_chunk_bytes,
            options,
            resources,
        )?;
        Ok(self
            .bytes_codec
            .decode(filtered, subchunk_shape, options, resources)?
            .into_owned())
    }
}

/// Partial decoder for [`Hdf5Codec`].
pub(crate) struct Hdf5PartialDecoder {
    input_handle: Arc<dyn SyncBytesPartialDecoderTraits>,
    chunk_shape: ChunkShape,
    data_type: DataType,
    fill_value: FillValue,
    parsed: Option<ParsedDataset>,
}

impl Hdf5PartialDecoder {
    fn new(
        input_handle: Arc<dyn SyncBytesPartialDecoderTraits>,
        shape: &[NonZeroU64],
        dataset_path: String,
        data_type: DataType,
        fill_value: FillValue,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<Self, CodecError> {
        let parsed = if input_handle.exists()? {
            Some(parse_hdf5_dataset(
                input_handle.clone(),
                shape,
                &dataset_path,
                &data_type,
                &fill_value,
                options,
                resources,
            )?)
        } else {
            None
        };
        Ok(Self {
            input_handle,
            chunk_shape: shape.to_vec(),
            data_type,
            fill_value,
            parsed,
        })
    }

    fn chunk_shape_u64(&self) -> &[u64] {
        bytemuck::must_cast_slice(&self.chunk_shape)
    }

    fn partial_decode_subset_into(
        &self,
        subset: &dyn ArraySubsetTraits,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        let Some(parsed) = &self.parsed else {
            return copy_fill_value_into(&self.data_type, &self.fill_value, output_target);
        };

        let elem_size = fixed_elem_size(&self.data_type)?;
        let subset_start = subset.start();

        match &parsed.layout {
            ResolvedLayout::Chunked {
                subchunk_shape,
                subchunk_grid,
                chunks,
                uncompressed_chunk_bytes,
            } => {
                let overlapping =
                    subchunk_grid
                        .chunks_in_array_subset(subset)?
                        .ok_or_else(|| {
                            CodecError::Other("subset out of bounds of HDF5 subchunk grid".into())
                        })?;

                // Collect overlapping subchunks and batch byte-range reads in a single call.
                let mut missing_overlaps: Vec<ArraySubset> = Vec::new();
                let mut present_specs: Vec<(ArraySubset, ArraySubset, u32)> = Vec::new();
                let mut byte_ranges: Vec<ByteRange> = Vec::new();

                for subchunk_indices in overlapping.indices() {
                    let subchunk_subset = subchunk_grid
                        .subset(&subchunk_indices)?
                        .expect("in-bounds subchunk");
                    let overlap = subchunk_subset.overlap(subset)?;
                    if let Some(&(address, size, filter_mask)) =
                        chunks.get(subchunk_indices.as_slice())
                    {
                        let byte_len = if size == 0 {
                            *uncompressed_chunk_bytes
                        } else {
                            size
                        };
                        byte_ranges.push(ByteRange::FromStart(address, Some(byte_len)));
                        present_specs.push((subchunk_subset, overlap, filter_mask));
                    } else {
                        missing_overlaps.push(overlap);
                    }
                }

                // Fill unallocated (sparse) HDF5 subchunks with fill_value.
                for overlap in missing_overlaps {
                    let dst_subset = overlap.relative_to(&subset_start)?;
                    // SAFETY: subchunk overlaps partition `subset` into disjoint subsets.
                    let mut dst_target = unsafe { output_target.subdivide(dst_subset) }?;
                    copy_fill_value_into(&self.data_type, &self.fill_value, &mut dst_target)?;
                }

                if byte_ranges.is_empty() {
                    return Ok(());
                }

                let fetched = self
                    .input_handle
                    .partial_decode_many(Box::new(byte_ranges.into_iter()), options, resources)?
                    .ok_or_else(|| {
                        CodecError::Other("HDF5 chunk disappeared during read".into())
                    })?;

                let full_subchunk_shape_u64: &[u64] = bytemuck::must_cast_slice(subchunk_shape);

                // Decode each fetched HDF5 subchunk in parallel using zarrs's CodecChain.
                let decoded_pieces: Vec<(ArraySubset, ArrayBytes<'static>)> = present_specs
                    .into_par_iter()
                    .zip(fetched.into_par_iter())
                    .map(|((subchunk_subset, overlap, filter_mask), raw_bytes)| {
                        // HDF5 stores edge chunks padded to the full `subchunk_shape`.
                        let full_decoded = parsed.decode_subchunk_bytes(
                            raw_bytes,
                            subchunk_shape,
                            *uncompressed_chunk_bytes,
                            filter_mask,
                            options,
                            resources.compute(),
                        )?;
                        let local_overlap = overlap.relative_to(subchunk_subset.start())?;
                        let piece = if local_overlap.shape() == full_subchunk_shape_u64
                            && local_overlap.start().iter().all(|&s| s == 0)
                        {
                            full_decoded
                        } else {
                            full_decoded
                                .extract_array_subset(
                                    &local_overlap,
                                    full_subchunk_shape_u64,
                                    &self.data_type,
                                )?
                                .into_owned()
                        };
                        Ok::<_, CodecError>((overlap, piece))
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                for (overlap, piece) in decoded_pieces {
                    let dst_subset = overlap.relative_to(&subset_start)?;
                    // SAFETY: subchunk overlaps are pairwise disjoint.
                    let mut dst_target = unsafe { output_target.subdivide(dst_subset) }?;
                    decode_into_array_bytes_target(&piece, &mut dst_target)?;
                }
                Ok(())
            }
            ResolvedLayout::Contiguous { address, .. } => {
                // Fetch only the coalesced byte ranges overlapping `subset`.
                let elem_size_u64 = elem_size as u64;
                let shifted: Vec<ByteRange> = subset
                    .iter_contiguous_linearised_indices(self.chunk_shape_u64())?
                    .map(|(start, count)| {
                        ByteRange::FromStart(
                            address + start * elem_size_u64,
                            Some(count * elem_size_u64),
                        )
                    })
                    .collect();
                let fetched = self
                    .input_handle
                    .partial_decode_many(Box::new(shifted.into_iter()), options, resources)?
                    .ok_or_else(|| {
                        CodecError::Other("HDF5 contiguous data disappeared during read".into())
                    })?;
                let mut raw =
                    Vec::with_capacity(usize::try_from(subset.num_elements()).unwrap() * elem_size);
                for part in fetched {
                    raw.extend_from_slice(&part);
                }
                let decoded = parsed.bytes_codec.decode(
                    CowBytes::from(raw),
                    &subset
                        .shape()
                        .iter()
                        .map(|&d| NonZeroU64::new(d).unwrap())
                        .collect::<Vec<_>>(),
                    options,
                    resources.compute(),
                )?;
                decode_into_array_bytes_target(&decoded, output_target)
            }
            ResolvedLayout::Compact { data } => {
                let decoded = parsed.bytes_codec.decode(
                    CowBytes::Borrowed(data),
                    &self.chunk_shape,
                    options,
                    resources.compute(),
                )?;
                let piece = decoded.extract_array_subset(
                    subset,
                    self.chunk_shape_u64(),
                    &self.data_type,
                )?;
                decode_into_array_bytes_target(&piece, output_target)
            }
        }
    }
}

impl SyncArrayPartialDecoderSubchunkingTraits for Hdf5PartialDecoder {
    fn local_subchunk_grids(
        &self,
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<Vec<Option<ChunkGrid>>, CodecError> {
        let Some(parsed) = &self.parsed else {
            return Ok(Vec::new());
        };
        match &parsed.layout {
            ResolvedLayout::Chunked { subchunk_grid, .. } => {
                Ok(vec![Some(ChunkGrid::new(subchunk_grid.clone()))])
            }
            ResolvedLayout::Contiguous { .. } | ResolvedLayout::Compact { .. } => Ok(Vec::new()),
        }
    }
}

impl SyncArrayPartialDecoderTraits for Hdf5PartialDecoder {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn exists(&self) -> Result<bool, StorageError> {
        self.input_handle.exists()
    }

    fn size_held(&self) -> usize {
        self.input_handle.size_held()
            + self.parsed.as_ref().map_or(0, |p| match &p.layout {
                ResolvedLayout::Chunked { chunks, .. } => {
                    chunks.len() * (size_of::<(u64, u64, u32)>() + 32)
                }
                ResolvedLayout::Compact { data } => data.len(),
                ResolvedLayout::Contiguous { .. } => 0,
            })
    }

    fn partial_decode(
        &self,
        indexer: &dyn Indexer,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<ArrayBytes<'_>, CodecError> {
        indexer.validate(self.chunk_shape_u64())?;
        let elem_size = fixed_elem_size(&self.data_type)?;
        if indexer.is_empty() {
            return Ok(ArrayBytes::new_flen(Vec::new()));
        }

        if let Some(subset) = indexer.as_array_subset() {
            let subset_shape = subset.shape();
            let mut buf = vec![0u8; usize::try_from(subset.num_elements()).unwrap() * elem_size];
            // SAFETY: `buf` is freshly allocated and exclusively owned by this single view.
            let view = unsafe {
                zarrs_codec::ArrayBytesFixedDisjointView::new(
                    unsafe_cell_slice::UnsafeCellSlice::new(&mut buf),
                    elem_size,
                    &subset_shape,
                    ArraySubset::new_with_shape(&subset_shape),
                )
            }?;
            let mut target = ArrayBytesTarget::from(view);
            self.partial_decode_subset_into(subset, &mut target, options, resources)?;
            return Ok(ArrayBytes::new_flen(buf));
        }

        // Generic non-subset indexer fallback: decode full chunk and gather indexed elements.
        let full = ArraySubset::new_with_shape(self.chunk_shape_u64());
        let full_bytes = self.partial_decode(&full, options, resources)?;
        let raw = full_bytes.into_fixed()?;
        let mut out = Vec::with_capacity(usize::try_from(indexer.len()).unwrap() * elem_size);
        for linear in indexer.iter_linearised_indices(self.chunk_shape_u64())? {
            let start = usize::try_from(linear).unwrap() * elem_size;
            out.extend_from_slice(&raw[start..start + elem_size]);
        }
        Ok(ArrayBytes::new_flen(out))
    }

    fn partial_decode_into(
        &self,
        indexer: &dyn Indexer,
        output_target: &mut ArrayBytesTarget<'_>,
        options: &CodecOptions,
        resources: &Resources,
    ) -> Result<(), CodecError> {
        indexer.validate(self.chunk_shape_u64())?;
        if indexer.is_empty() {
            return Ok(());
        }
        if let Some(subset) = indexer.as_array_subset() {
            let subset_shape = subset.shape();
            if output_target.shape() == subset_shape.as_ref() {
                return self.partial_decode_subset_into(subset, output_target, options, resources);
            }
            if let Some(mut reshaped) = output_target.reshaped(&subset_shape) {
                return self.partial_decode_subset_into(subset, &mut reshaped, options, resources);
            }
        }
        let decoded = self.partial_decode(indexer, options, resources)?;
        decode_into_array_bytes_target(&decoded, output_target)
    }

    fn supports_partial_decode(&self) -> bool {
        self.input_handle.supports_partial_decode()
    }
}

// ---------------------------------------------------------------------------
// HDF5 Metadata Parsing & Codec/Type Mapping Helpers
// ---------------------------------------------------------------------------

fn parse_hdf5_dataset(
    input_handle: Arc<dyn SyncBytesPartialDecoderTraits>,
    expected_shape: &[NonZeroU64],
    dataset_path: &str,
    expected_data_type: &DataType,
    fill_value: &FillValue,
    options: &CodecOptions,
    resources: &Resources,
) -> Result<ParsedDataset, CodecError> {
    let storage: Arc<dyn Storage> = Arc::new(PartialDecoderStorage {
        input_handle,
        options: *options,
        resources: resources.clone(),
    });
    let file = Hdf5File::from_storage(storage.clone())
        .map_err(|err| CodecError::Other(format!("failed to open HDF5 file: {err}")))?;
    let dataset = file.dataset(dataset_path).map_err(|err| {
        CodecError::Other(format!(
            "failed to open HDF5 dataset {dataset_path:?}: {err}"
        ))
    })?;

    let expected_shape_u64: &[u64] = bytemuck::must_cast_slice(expected_shape);
    if dataset.shape() != expected_shape_u64 {
        return Err(CodecError::Other(format!(
            "HDF5 dataset {dataset_path:?} shape {:?} does not match Zarr chunk shape {expected_shape_u64:?}",
            dataset.shape()
        )));
    }

    let endianness = validate_hdf5_datatype(dataset.dtype(), expected_data_type)?;
    let elem_size = fixed_elem_size(expected_data_type)?;

    let header = file
        .get_or_parse_header(dataset.address())
        .map_err(|err| CodecError::Other(format!("failed to parse HDF5 dataset header: {err}")))?;

    let mut layout_msg: Option<&DataLayout> = None;
    let mut filter_descs: Vec<FilterDescription> = Vec::new();
    for msg in &header.messages {
        match msg {
            HdfMessage::DataLayout(m) => layout_msg = Some(&m.layout),
            HdfMessage::FilterPipeline(m) => filter_descs.clone_from(&m.filters),
            _ => {}
        }
    }
    let layout_msg = layout_msg
        .ok_or_else(|| CodecError::Other("HDF5 dataset missing DataLayout message".into()))?;

    let filter_pipeline =
        FilterMaskPipeline::new(hdf5_filters_to_zarr_codecs(&filter_descs, elem_size)?);
    let bytes_codec = BytesCodec::new(Some(endianness))
        .with_context(
            expected_data_type.clone(),
            Some(fill_value.clone()),
            &CodecSpecificOptions::default(),
        )
        .map_err(|err| CodecError::Other(err.to_string()))?;

    let sb = file.superblock();
    let layout = match layout_msg {
        DataLayout::Compact { data } => ResolvedLayout::Compact { data: data.clone() },
        DataLayout::Contiguous { address, size } => ResolvedLayout::Contiguous {
            address: *address,
            size: *size,
        },
        DataLayout::Chunked {
            address,
            dims,
            chunk_indexing,
            ..
        } => {
            let entries = collect_chunk_entries(
                storage.as_ref(),
                *address,
                sb.offset_size,
                sb.length_size,
                expected_shape_u64,
                dataset.max_dims(),
                dims,
                elem_size,
                chunk_indexing.as_ref(),
            )?;

            let subchunk_shape: ChunkShape = dims
                .iter()
                .map(|&d| {
                    NonZeroU64::new(u64::from(d))
                        .ok_or_else(|| CodecError::Other("HDF5 chunk dimension is zero".into()))
                })
                .collect::<Result<_, _>>()?;
            let subchunk_grid =
                RegularBoundedChunkGrid::new(expected_shape_u64.to_vec(), subchunk_shape.clone())?;
            let uncompressed_chunk_bytes = subchunk_shape.num_elements_u64() * (elem_size as u64);

            let mut chunks = HashMap::with_capacity(entries.len());
            for entry in entries {
                if entry.offsets.len() != dims.len() {
                    continue;
                }
                let indices: Vec<u64> = entry
                    .offsets
                    .iter()
                    .zip(dims)
                    .map(|(&off, &d)| off / u64::from(d))
                    .collect();
                chunks.insert(indices, (entry.address, entry.size, entry.filter_mask));
            }

            ResolvedLayout::Chunked {
                subchunk_shape,
                subchunk_grid,
                chunks,
                uncompressed_chunk_bytes,
            }
        }
    };

    Ok(ParsedDataset {
        layout,
        bytes_codec,
        filter_pipeline,
    })
}

#[allow(clippy::too_many_arguments)]
fn collect_chunk_entries(
    storage: &dyn Storage,
    index_address: u64,
    offset_size: u8,
    length_size: u8,
    shape: &[u64],
    max_dims: Option<&[u64]>,
    chunk_dims: &[u32],
    elem_size: usize,
    chunk_indexing: Option<&ChunkIndexing>,
) -> Result<Vec<ChunkEntry>, CodecError> {
    let ndim = shape.len();
    let map_err = |err: hdf5_reader::error::Error| {
        CodecError::Other(format!("failed to read HDF5 chunk index: {err}"))
    };
    match chunk_indexing {
        None => {
            let leaves = hdf5_reader::btree_v1::collect_btree_v1_leaves_storage(
                storage,
                index_address,
                offset_size,
                length_size,
                Some(ndim as u32),
                chunk_dims,
                None,
            )
            .map_err(map_err)?;
            let mut entries = Vec::with_capacity(leaves.len());
            for (key, addr) in leaves {
                if let hdf5_reader::btree_v1::BTreeV1Key::RawData {
                    chunk_size,
                    filter_mask,
                    offsets,
                } = key
                {
                    entries.push(ChunkEntry {
                        address: addr,
                        size: u64::from(chunk_size),
                        filter_mask,
                        offsets: offsets[..ndim].to_vec(),
                    });
                }
            }
            Ok(entries)
        }
        Some(ChunkIndexing::SingleChunk {
            filtered_size,
            filters,
        }) => Ok(vec![hdf5_reader::chunk_index::single_chunk_entry(
            index_address,
            *filtered_size,
            *filters,
            ndim,
        )]),
        Some(ChunkIndexing::Implicit) => hdf5_reader::chunk_index::collect_implicit_chunk_entries(
            index_address,
            shape,
            chunk_dims,
            elem_size,
            None,
            storage.len(),
        )
        .map_err(map_err),
        Some(ChunkIndexing::FixedArray { .. }) => {
            hdf5_reader::fixed_array::collect_fixed_array_chunk_entries_storage(
                storage,
                index_address,
                offset_size,
                length_size,
                shape,
                chunk_dims,
                None,
            )
            .map_err(map_err)
        }
        Some(ChunkIndexing::ExtensibleArray { .. }) => {
            hdf5_reader::extensible_array::collect_extensible_array_chunk_entries_storage(
                storage,
                index_address,
                offset_size,
                length_size,
                shape,
                max_dims,
                chunk_dims,
                None,
            )
            .map_err(map_err)
        }
        Some(ChunkIndexing::BTreeV2) => hdf5_reader::chunk_index::collect_v2_chunk_entries_storage(
            storage,
            index_address,
            offset_size,
            length_size,
            ndim as u32,
            chunk_dims,
            None,
        )
        .map_err(map_err),
    }
}

/// Map an HDF5 filter pipeline to `zarrs` bytes-to-bytes codecs.
///
/// Unsupported HDF5 filters return an actionable error referencing the missing Zarr extension
/// specification needed to support them (see `spec/codecs/hdf5/README.md`).
fn hdf5_filters_to_zarr_codecs(
    filters: &[FilterDescription],
    elem_size: usize,
) -> Result<Vec<Arc<dyn BytesToBytesCodecTraits>>, CodecError> {
    let mut codecs: Vec<Arc<dyn BytesToBytesCodecTraits>> = Vec::with_capacity(filters.len());
    for filter in filters {
        match filter.id {
            FILTER_DEFLATE => {
                let level = filter.client_data.first().copied().unwrap_or(6).min(9);
                let codec = ZlibCodec::new(level.try_into().map_err(|_| {
                    CodecError::Other(format!("invalid HDF5 deflate level {level}"))
                })?);
                codecs.push(Arc::new(codec));
            }
            FILTER_SHUFFLE => {
                let elementsize = filter
                    .client_data
                    .first()
                    .copied()
                    .map_or(elem_size, |v| v as usize);
                if elementsize == 0 {
                    return Err(CodecError::Other(
                        "HDF5 shuffle filter has zero element size".into(),
                    ));
                }
                codecs.push(Arc::new(ShuffleCodec::new(elementsize)));
            }
            FILTER_FLETCHER32 => {
                codecs.push(Arc::new(Fletcher32Codec::new()));
            }
            #[cfg(feature = "bz2")]
            FILTER_BZIP2 => {
                let level = filter.client_data.first().copied().unwrap_or(9).clamp(1, 9);
                let codec =
                    zarrs::array::codec::Bz2Codec::new(level.try_into().map_err(|_| {
                        CodecError::Other(format!("invalid HDF5 bzip2 level {level}"))
                    })?);
                codecs.push(Arc::new(codec));
            }
            #[cfg(feature = "blosc")]
            FILTER_BLOSC => {
                codecs.push(Arc::new(
                    zarrs::array::codec::BloscCodec::new(
                        zarrs::array::codec::BloscCompressor::LZ4,
                        5u8.try_into().unwrap(),
                        None,
                        zarrs::array::codec::BloscShuffleMode::NoShuffle,
                        None,
                    )
                    .map_err(|err| CodecError::Other(err.to_string()))?,
                ));
            }
            #[cfg(feature = "zstd")]
            FILTER_ZSTD => {
                let level = filter.client_data.first().copied().unwrap_or(3) as i32;
                codecs.push(Arc::new(zarrs::array::codec::ZstdCodec::new(level, false)));
            }
            FILTER_SZIP => {
                return Err(CodecError::Other(
                    "unsupported HDF5 filter: szip (id=4); requires a future `hdf5.szip` Zarr codec spec".into(),
                ));
            }
            FILTER_NBIT => {
                return Err(CodecError::Other(
                    "unsupported HDF5 filter: nbit (id=5); requires a future `hdf5.nbit` Zarr codec spec".into(),
                ));
            }
            FILTER_SCALEOFFSET => {
                return Err(CodecError::Other(
                    "unsupported HDF5 filter: scaleoffset (id=6); requires a future `hdf5.scaleoffset` Zarr codec spec".into(),
                ));
            }
            FILTER_LZ4 => {
                return Err(CodecError::Other(
                    "unsupported HDF5 filter: lz4 (id=32004); requires a future `hdf5.lz4` Zarr codec spec".into(),
                ));
            }
            FILTER_BITSHUFFLE => {
                return Err(CodecError::Other(
                    "unsupported HDF5 filter: bitshuffle (id=32008); requires a future `hdf5.bitshuffle` Zarr codec spec".into(),
                ));
            }
            FILTER_ZFP => {
                return Err(CodecError::Other(
                    "unsupported HDF5 filter: zfp (id=32013); requires a future `hdf5.zfp` Zarr codec spec".into(),
                ));
            }
            FILTER_SZ3 => {
                return Err(CodecError::Other(
                    "unsupported HDF5 filter: sz3 (id=32024); requires a future `hdf5.sz3` Zarr codec spec".into(),
                ));
            }
            other => {
                return Err(CodecError::Other(format!(
                    "unsupported HDF5 filter id={other} (name={:?})",
                    filter.name
                )));
            }
        }
    }
    Ok(codecs)
}

fn fixed_elem_size(dt: &DataType) -> Result<usize, CodecError> {
    match dt {
        DataType::Fixed(fixed) => Ok(fixed.size()),
        _ => Err(CodecError::Other(
            "hdf5 codec requires a fixed-size data type".into(),
        )),
    }
}

fn validate_supported_zarr_data_type(dt: &DataType) -> Result<(), CodecCreateError> {
    if *dt == data_type::int8()
        || *dt == data_type::uint8()
        || *dt == data_type::int16()
        || *dt == data_type::uint16()
        || *dt == data_type::int32()
        || *dt == data_type::uint32()
        || *dt == data_type::int64()
        || *dt == data_type::uint64()
        || *dt == data_type::float32()
        || *dt == data_type::float64()
    {
        Ok(())
    } else {
        Err(CodecCreateError::Other(format!(
            "unsupported data type for hdf5 codec: {dt}"
        )))
    }
}

fn validate_hdf5_datatype(
    hdf5_dt: &Hdf5Datatype,
    zarr_dt: &DataType,
) -> Result<Endianness, CodecError> {
    let (endianness, expected_zarr) = match hdf5_dt {
        Hdf5Datatype::FixedPoint {
            size,
            signed,
            byte_order,
        } => {
            let endianness = match byte_order {
                Hdf5ByteOrder::LittleEndian => Endianness::Little,
                Hdf5ByteOrder::BigEndian => Endianness::Big,
            };
            let dt = match (signed, size) {
                (true, 1) => data_type::int8(),
                (false, 1) => data_type::uint8(),
                (true, 2) => data_type::int16(),
                (false, 2) => data_type::uint16(),
                (true, 4) => data_type::int32(),
                (false, 4) => data_type::uint32(),
                (true, 8) => data_type::int64(),
                (false, 8) => data_type::uint64(),
                _ => {
                    return Err(CodecError::Other(format!(
                        "unsupported HDF5 integer datatype: {hdf5_dt:?}"
                    )));
                }
            };
            (endianness, dt)
        }
        Hdf5Datatype::FloatingPoint { size, byte_order } => {
            let endianness = match byte_order {
                Hdf5ByteOrder::LittleEndian => Endianness::Little,
                Hdf5ByteOrder::BigEndian => Endianness::Big,
            };
            let dt = match size {
                4 => data_type::float32(),
                8 => data_type::float64(),
                _ => {
                    return Err(CodecError::Other(format!(
                        "unsupported HDF5 floating-point datatype: {hdf5_dt:?}"
                    )));
                }
            };
            (endianness, dt)
        }
        other => {
            return Err(CodecError::Other(format!(
                "unsupported HDF5 datatype: {other:?}"
            )));
        }
    };

    if *zarr_dt != expected_zarr {
        return Err(CodecError::Other(format!(
            "HDF5 datatype {expected_zarr} does not match Zarr array data type {zarr_dt}"
        )));
    }
    Ok(endianness)
}

/// Build a globally resolvable subchunk grid refining `chunk_grid` by tiling each chunk with
/// `chunk_shape` (clipped at each outer chunk boundary).
fn build_global_subchunk_grid(
    chunk_grid: &ChunkGrid,
    chunk_shape: &ChunkShape,
) -> Result<ChunkGrid, ChunkGridCreateError> {
    if chunk_grid.dimensionality() != chunk_shape.len() {
        return Err(ChunkGridCreateError::new(format!(
            "hdf5 chunk shape dimensionality {} does not match chunk grid dimensionality {}",
            chunk_shape.len(),
            chunk_grid.dimensionality()
        )));
    }

    let mut grid_shape = Vec::with_capacity(chunk_grid.dimensionality());
    let mut edge_lengths_per_dim = Vec::with_capacity(chunk_grid.dimensionality());
    let mut needs_rectilinear = false;

    for (dim, &subchunk) in chunk_shape.iter().enumerate() {
        let outer_edges = chunk_grid.chunk_edge_lengths(dim)?;
        let mut subchunk_edges = Vec::new();
        for edge in outer_edges {
            subchunk_edges.extend(
                RegularBoundedChunkGrid::new(vec![edge.get()], vec![subchunk])?
                    .chunk_edge_lengths(0)?,
            );
        }
        let dim_total = subchunk_edges
            .iter()
            .try_fold(0u64, |sum, e| sum.checked_add(e.get()))
            .ok_or_else(|| ChunkGridCreateError::new("subchunk grid shape overflow"))?;
        let encoded = ChunkEdgeLengths::encode(&subchunk_edges);
        if !matches!(encoded, ChunkEdgeLengths::Scalar(_)) {
            needs_rectilinear = true;
        }
        grid_shape.push(dim_total);
        edge_lengths_per_dim.push(encoded);
    }

    if needs_rectilinear {
        Ok(ChunkGrid::new(RectilinearChunkGrid::new(
            grid_shape,
            &edge_lengths_per_dim,
        )?))
    } else {
        let regular_shape: ChunkShape = edge_lengths_per_dim
            .into_iter()
            .map(|e| match e {
                ChunkEdgeLengths::Scalar(s) => s,
                ChunkEdgeLengths::Varying(_) => unreachable!("checked scalar above"),
            })
            .collect();
        Ok(ChunkGrid::new(RegularChunkGrid::new(
            grid_shape,
            regular_shape,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_mask_skips_masked_filters_without_rebuilding_chain() {
        let pipeline = FilterMaskPipeline::new(vec![
            Arc::new(ShuffleCodec::new(2)),
            Arc::new(ZlibCodec::new(1u32.try_into().unwrap())),
            Arc::new(Fletcher32Codec::new()),
        ]);
        let options = CodecOptions::default();
        let compute = ComputeResources::default();
        let raw: Vec<u8> = (0i16..32).flat_map(i16::to_le_bytes).collect();

        // Simulate a chunk where bit 1 (zlib) was skipped, so only shuffle (bit 0) + fletcher32 (bit 2) ran.
        let shuffled = ShuffleCodec::new(2)
            .encode(CowBytes::Borrowed(&raw), &options, &compute)
            .unwrap();
        let checksummed = Fletcher32Codec::new()
            .encode(shuffled, &options, &compute)
            .unwrap();

        let decoded = pipeline
            .decode_with_mask(checksummed, 0b010, raw.len() as u64, &options, &compute)
            .unwrap();
        assert_eq!(decoded.as_ref(), raw.as_slice());

        // Simulate all filters skipped (bits 0, 1, 2 all set).
        let passthrough = pipeline
            .decode_with_mask(
                CowBytes::Borrowed(&raw),
                0b111,
                raw.len() as u64,
                &options,
                &compute,
            )
            .unwrap();
        assert_eq!(passthrough.as_ref(), raw.as_slice());
    }

    #[test]
    fn unsupported_hdf5_filter_diagnostics() {
        for (id, expected) in [
            (FILTER_SZIP, "hdf5.szip"),
            (FILTER_NBIT, "hdf5.nbit"),
            (FILTER_SCALEOFFSET, "hdf5.scaleoffset"),
            (FILTER_LZ4, "hdf5.lz4"),
            (FILTER_BITSHUFFLE, "hdf5.bitshuffle"),
            (FILTER_ZFP, "hdf5.zfp"),
            (FILTER_SZ3, "hdf5.sz3"),
        ] {
            let err = hdf5_filters_to_zarr_codecs(
                &[FilterDescription {
                    id,
                    name: None,
                    client_data: vec![],
                }],
                2,
            )
            .unwrap_err();
            assert!(
                err.to_string().contains(expected),
                "expected {expected:?} in error: {err}"
            );
        }
    }
}
