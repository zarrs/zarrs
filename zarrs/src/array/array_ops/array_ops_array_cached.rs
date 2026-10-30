use inherent::inherent;

use super::{
    Arc, ArrayBuilder, ArrayCached, ArrayError, ArrayIndices, ArrayMetadata, ArrayMetadataOptions,
    ArrayOps, ArraySubset, ArraySubsetTraits, ChunkGrid, ChunkGridDecoded, ChunkGridDecodedRef,
    ChunkKeyEncoding, ChunkShape, CodecChain, CodecChainBound, CodecOptions, CodecSpecificOptions,
    DataType, DimensionName, FillValue, IncompatibleDimensionalityError, MetadataEraseVersion,
    NodePath, StorageTransformerChain, StoreKey,
};

#[inherent]
impl<TStorage: ?Sized, C> ArrayOps for ArrayCached<TStorage, C> {
    type Storage = TStorage;

    #[allow(clippy::must_use_candidate)]
    pub fn storage(&self) -> Arc<TStorage> {
        self.array().storage()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn path(&self) -> &NodePath {
        self.array().path()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn data_type(&self) -> &DataType {
        self.array().data_type()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn fill_value(&self) -> &FillValue {
        self.array().fill_value()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn shape(&self) -> &[u64] {
        self.array().shape()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn dimensionality(&self) -> usize {
        self.array().dimensionality()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn codecs(&self) -> Arc<CodecChain> {
        self.array().codecs()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn codecs_bound(&self) -> Arc<CodecChainBound> {
        self.array().codecs_bound()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn chunk_grid(&self) -> &ChunkGrid {
        self.array().chunk_grid()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn chunk_key_encoding(&self) -> &ChunkKeyEncoding {
        self.array().chunk_key_encoding()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn storage_transformers(&self) -> &StorageTransformerChain {
        self.array().storage_transformers()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn codec_options(&self) -> &CodecOptions {
        self.array().codec_options()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn codec_specific_options(&self) -> &CodecSpecificOptions {
        self.array().codec_specific_options()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn metadata_options(&self) -> &ArrayMetadataOptions {
        self.array().metadata_options()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn metadata_erase_version(&self) -> MetadataEraseVersion {
        self.array().metadata_erase_version()
    }

    #[allow(clippy::must_use_candidate, clippy::return_self_not_must_use)]
    pub fn with_codec_options(&self, codec_options: CodecOptions) -> Self {
        self.map_array(|array| {
            array.set_codec_options(codec_options);
        })
    }

    #[allow(clippy::must_use_candidate, clippy::return_self_not_must_use)]
    pub fn with_metadata_options(&self, metadata_options: ArrayMetadataOptions) -> Self {
        self.map_array(|array| {
            array.set_metadata_options(metadata_options);
        })
    }

    #[allow(clippy::must_use_candidate, clippy::return_self_not_must_use)]
    pub fn with_metadata_erase_version(
        &self,
        metadata_erase_version: MetadataEraseVersion,
    ) -> Self {
        self.map_array(|array| {
            array.set_metadata_erase_version(metadata_erase_version);
        })
    }

    #[allow(clippy::must_use_candidate)]
    pub fn dimension_names(&self) -> &Option<Vec<DimensionName>> {
        self.array().dimension_names()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn attributes(&self) -> &serde_json::Map<String, serde_json::Value> {
        self.array().attributes()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn metadata(&self) -> &ArrayMetadata {
        self.array().metadata()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn metadata_opt(&self) -> ArrayMetadata {
        self.array().metadata_opt()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn builder(&self) -> ArrayBuilder {
        self.array().builder()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn chunk_grid_shape(&self) -> &[u64] {
        self.array().chunk_grid_shape()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn subchunk_shape(&self) -> Option<ChunkShape> {
        self.array().subchunk_shape()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn subchunk_shape_at_level(&self, level: usize) -> Option<ChunkShape> {
        self.array().subchunk_shape_at_level(level)
    }

    #[allow(clippy::must_use_candidate)]
    pub fn subchunk_grid(&self) -> ChunkGridDecodedRef<'_> {
        self.array().subchunk_grid()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn subchunk_grids(&self) -> &[ChunkGridDecoded] {
        self.array().subchunk_grids()
    }

    #[allow(clippy::must_use_candidate)]
    pub fn subchunk_grid_at_level(&self, level: usize) -> ChunkGridDecodedRef<'_> {
        self.array().subchunk_grid_at_level(level)
    }

    pub fn chunk_key(&self, chunk_indices: &[u64]) -> Result<StoreKey, ArrayError> {
        self.array().chunk_key(chunk_indices)
    }

    pub fn chunk_origin(&self, chunk_indices: &[u64]) -> Result<ArrayIndices, ArrayError> {
        self.array().chunk_origin(chunk_indices)
    }

    pub fn chunk_shape(&self, chunk_indices: &[u64]) -> Result<ChunkShape, ArrayError> {
        self.array().chunk_shape(chunk_indices)
    }

    #[allow(clippy::must_use_candidate)]
    pub fn subset_all(&self) -> ArraySubset {
        self.array().subset_all()
    }

    pub fn chunk_shape_usize(&self, chunk_indices: &[u64]) -> Result<Vec<usize>, ArrayError> {
        self.array().chunk_shape_usize(chunk_indices)
    }

    pub fn chunk_subset(&self, chunk_indices: &[u64]) -> Result<ArraySubset, ArrayError> {
        self.array().chunk_subset(chunk_indices)
    }

    pub fn chunk_subset_bounded(&self, chunk_indices: &[u64]) -> Result<ArraySubset, ArrayError> {
        self.array().chunk_subset_bounded(chunk_indices)
    }

    pub fn chunks_subset(&self, chunks: &dyn ArraySubsetTraits) -> Result<ArraySubset, ArrayError> {
        self.array().chunks_subset(chunks)
    }

    pub fn chunks_subset_bounded(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<ArraySubset, ArrayError> {
        self.array().chunks_subset_bounded(chunks)
    }

    pub fn chunks_in_array_subset(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        self.array().chunks_in_array_subset(array_subset)
    }
}
