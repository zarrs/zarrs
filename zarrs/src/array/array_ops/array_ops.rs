use super::*;

/// Core array operations.
pub trait ArrayOps {
    /// The backing storage type.
    type Storage: ?Sized;

    /// Get the underlying storage backing the array.
    fn storage(&self) -> Arc<Self::Storage>;

    /// Get the node path.
    fn path(&self) -> &NodePath;

    /// Get the data type.
    fn data_type(&self) -> &DataType;

    /// Get the fill value.
    fn fill_value(&self) -> &FillValue;

    /// Get the array shape.
    fn shape(&self) -> &[u64];

    /// Get the array dimensionality.
    fn dimensionality(&self) -> usize;

    /// Get the codecs.
    fn codecs(&self) -> Arc<CodecChain>;

    /// Get the codecs bound to the array data type and fill value.
    fn codecs_bound(&self) -> Arc<CodecChainBound>;

    /// Get the chunk grid.
    fn chunk_grid(&self) -> &ChunkGrid;

    /// Get the chunk key encoding.
    fn chunk_key_encoding(&self) -> &ChunkKeyEncoding;

    /// Get the storage transformers.
    fn storage_transformers(&self) -> &StorageTransformerChain;

    /// Get the codec options used by the array operations.
    ///
    /// Override them with [`with_codec_options`](ArrayOps::with_codec_options), or
    /// [`ArrayMutOps::set_codec_options`] where the array is owned.
    fn codec_options(&self) -> &CodecOptions;

    /// Get the array metadata options used by the array operations.
    ///
    /// Override them with [`with_metadata_options`](ArrayOps::with_metadata_options), or
    /// [`ArrayMutOps::set_metadata_options`] where the array is owned.
    fn metadata_options(&self) -> &ArrayMetadataOptions;

    /// Get the metadata erase version used by the array operations.
    ///
    /// Override it with
    /// [`with_metadata_erase_version`](ArrayOps::with_metadata_erase_version), or
    /// [`ArrayMutOps::set_metadata_erase_version`] where the array is owned.
    fn metadata_erase_version(&self) -> MetadataEraseVersion;

    /// Return this array configured to use `codec_options` for its operations.
    ///
    /// Prefer deriving once and reusing the result over deriving per operation.
    #[must_use]
    fn with_codec_options(&self, codec_options: CodecOptions) -> Self
    where
        Self: Sized;

    /// Return this array configured to use `metadata_options` for its operations.
    #[must_use]
    fn with_metadata_options(&self, metadata_options: ArrayMetadataOptions) -> Self
    where
        Self: Sized;

    /// Return this array configured to use `metadata_erase_version` for its operations.
    #[must_use]
    fn with_metadata_erase_version(&self, metadata_erase_version: MetadataEraseVersion) -> Self
    where
        Self: Sized;

    /// Get the dimension names.
    fn dimension_names(&self) -> &Option<Vec<DimensionName>>;

    /// Get the attributes.
    fn attributes(&self) -> &serde_json::Map<String, serde_json::Value>;

    /// Return the underlying array metadata.
    fn metadata(&self) -> &ArrayMetadata;

    /// Return the array metadata in the form that [`Array::store_metadata`] writes.
    ///
    /// Unlike [`metadata`](ArrayOps::metadata), which borrows the metadata as it was stored or
    /// constructed, this applies [`metadata_options`](ArrayOps::metadata_options): it may inject
    /// `zarrs` provenance attributes, convert between Zarr versions, and rewrite aliased
    /// extension names.
    fn metadata_opt(&self) -> ArrayMetadata;

    /// Create an array builder matching the parameters of this array.
    fn builder(&self) -> ArrayBuilder;

    /// Return the shape of the chunk grid (i.e., the number of chunks).
    fn chunk_grid_shape(&self) -> &[u64];

    /// Return the level-zero subchunk shape if its grid has a regular chunk shape.
    ///
    /// Returns [`None`] if the array does not expose subchunks, or if the
    /// resolved subchunk grid has varying edge lengths.
    #[must_use]
    fn subchunk_shape(&self) -> Option<ChunkShape>;

    /// Return the subchunk shape at `level` if its grid has a regular chunk shape.
    ///
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    #[must_use]
    fn subchunk_shape_at_level(&self, level: usize) -> Option<ChunkShape>;

    /// Retrieve the level-zero subchunk grid.
    ///
    /// Returns [`ChunkGridDecodedRef::None`] if the array does not expose subchunks, or
    /// [`ChunkGridDecodedRef::ChunkLocal`] if a grid is only resolvable per chunk via
    /// [`local_subchunk_grid`](crate::array::ArrayReadOps::local_subchunk_grid).
    /// Use [`as_chunk_grid`](ChunkGridDecodedRef::as_chunk_grid) to get the grid if it is
    /// resolvable for the whole array.
    #[must_use]
    fn subchunk_grid(&self) -> ChunkGridDecodedRef<'_>;

    /// Return the subchunk grid levels exposed by the codec hierarchy, outermost first.
    ///
    /// This includes levels that cannot be resolved globally.
    #[must_use]
    fn subchunk_grids(&self) -> &[ChunkGridDecoded];

    /// Return the subchunk grid at `level`.
    ///
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    /// Returns [`ChunkGridDecodedRef::None`] if `level` is beyond the subchunk grid hierarchy.
    #[must_use]
    fn subchunk_grid_at_level(&self, level: usize) -> ChunkGridDecodedRef<'_>;

    /// Return the codecs that encode the subchunks at each level, outermost first.
    ///
    /// The codecs at each level match the grids returned by
    /// [`subchunk_grids`](ArrayOps::subchunk_grids). The level zero codecs decode the bytes
    /// returned by
    /// [`retrieve_encoded_subchunk`](crate::array::ArrayReadOps::retrieve_encoded_subchunk),
    /// and deeper levels decode the encoded subchunks of subchunks.
    ///
    /// An empty vector indicates that the array does not expose subchunks.
    ///
    /// The codecs operate in the encoded domain of the array-to-bytes codec. If the array has
    /// array-to-array codecs, then decoding an encoded subchunk yields array bytes that those
    /// codecs have yet to decode.
    #[must_use]
    fn subchunk_codecs(&self) -> Vec<Arc<dyn ArrayToBytesCodecTraits>> {
        ArrayToBytesCodecSubchunkingTraits::subchunk_codecs(self.codecs_bound().as_ref())
    }

    /// Return the codecs that encode the subchunks at `level`.
    ///
    /// Level zero is the outermost subchunk grid and increasing levels move inward.
    /// Returns [`None`] if `level` is beyond the subchunk grid hierarchy.
    #[must_use]
    fn subchunk_codecs_at_level(&self, level: usize) -> Option<Arc<dyn ArrayToBytesCodecTraits>> {
        self.subchunk_codecs().into_iter().nth(level)
    }

    /// Return the store key of the chunk at `chunk_indices`.
    fn chunk_key(&self, chunk_indices: &[u64]) -> StoreKey;

    /// Return the origin of the chunk at `chunk_indices`.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    #[allow(clippy::missing_errors_doc)]
    fn chunk_origin(&self, chunk_indices: &[u64]) -> Result<ArrayIndices, ArrayError>;

    /// Return the shape of the chunk at `chunk_indices`.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    #[allow(clippy::missing_errors_doc)]
    fn chunk_shape(&self, chunk_indices: &[u64]) -> Result<ChunkShape, ArrayError>;

    /// Return an array subset that spans the entire array.
    fn subset_all(&self) -> ArraySubset;

    /// Return the shape of the chunk at `chunk_indices` as `usize` values.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    ///
    /// # Panics
    /// Panics if any component of the chunk shape exceeds [`usize::MAX`].
    #[allow(clippy::missing_errors_doc)]
    fn chunk_shape_usize(&self, chunk_indices: &[u64]) -> Result<Vec<usize>, ArrayError>;

    /// Return the array subset of the chunk at `chunk_indices`.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    #[allow(clippy::missing_errors_doc)]
    fn chunk_subset(&self, chunk_indices: &[u64]) -> Result<ArraySubset, ArrayError>;

    /// Return the array subset of the chunk at `chunk_indices` bounded by the array shape.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    #[allow(clippy::missing_errors_doc)]
    fn chunk_subset_bounded(&self, chunk_indices: &[u64]) -> Result<ArraySubset, ArrayError>;

    /// Return the array subset of `chunks`.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if a chunk in `chunks` is incompatible with the chunk grid.
    #[allow(clippy::missing_errors_doc)]
    fn chunks_subset(&self, chunks: &dyn ArraySubsetTraits) -> Result<ArraySubset, ArrayError>;

    /// Return the array subset of `chunks` bounded by the array shape.
    ///
    /// # Errors
    /// Returns [`ArrayError::InvalidChunkGridIndicesError`] if the `chunk_indices` are incompatible with the chunk grid.
    #[allow(clippy::missing_errors_doc)]
    fn chunks_subset_bounded(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<ArraySubset, ArrayError>;

    /// Return an array subset indicating the chunks intersecting `array_subset`.
    ///
    /// Returns [`None`] if the intersecting chunks cannot be determined.
    ///
    /// # Errors
    /// Returns [`IncompatibleDimensionalityError`] if the array subset has an incorrect dimensionality.
    #[allow(clippy::missing_errors_doc)]
    fn chunks_in_array_subset(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError>;
}

pub(super) fn maybe_regular_chunk_grid_shape(chunk_grid: &ChunkGrid) -> Option<ChunkShape> {
    let mut chunk_shape = Vec::with_capacity(chunk_grid.dimensionality());
    for dimension in 0..chunk_grid.dimensionality() {
        let edge_lengths = chunk_grid.chunk_edge_lengths(dimension).ok()?;
        let (&edge_length, remaining_edge_lengths) = edge_lengths.split_first()?;
        if remaining_edge_lengths
            .iter()
            .any(|remaining_edge_length| *remaining_edge_length != edge_length)
        {
            return None;
        }
        chunk_shape.push(edge_length);
    }
    Some(chunk_shape)
}
