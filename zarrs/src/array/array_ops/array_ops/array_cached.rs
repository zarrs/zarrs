use inherent::inherent;

use super::super::*;
use super::ArrayOps;

#[inherent]
impl<TStorage: ?Sized, C> ArrayOps for ArrayCached<TStorage, C> {
    type Storage = TStorage;

    pub fn storage(&self) -> Arc<TStorage> {
        self.array().storage()
    }

    pub fn path(&self) -> &NodePath {
        self.array().path()
    }

    pub fn data_type(&self) -> &DataType {
        self.array().data_type()
    }

    pub fn fill_value(&self) -> &FillValue {
        self.array().fill_value()
    }

    pub fn shape(&self) -> &[u64] {
        self.array().shape()
    }

    pub fn dimensionality(&self) -> usize {
        self.array().dimensionality()
    }

    pub fn codecs(&self) -> Arc<CodecChain> {
        self.array().codecs()
    }

    pub fn chunk_grid(&self) -> &ChunkGrid {
        self.array().chunk_grid()
    }

    pub fn chunk_key_encoding(&self) -> &ChunkKeyEncoding {
        self.array().chunk_key_encoding()
    }

    pub fn storage_transformers(&self) -> &StorageTransformerChain {
        self.array().storage_transformers()
    }

    pub fn codec_options(&self) -> &CodecOptions {
        self.array().codec_options()
    }

    pub fn metadata_options(&self) -> &ArrayMetadataOptions {
        self.array().metadata_options()
    }

    pub fn metadata_erase_version(&self) -> MetadataEraseVersion {
        self.array().metadata_erase_version()
    }

    pub fn dimension_names(&self) -> &Option<Vec<DimensionName>> {
        self.array().dimension_names()
    }

    pub fn attributes(&self) -> &serde_json::Map<String, serde_json::Value> {
        self.array().attributes()
    }

    pub fn metadata(&self) -> &ArrayMetadata {
        self.array().metadata()
    }

    pub fn metadata_opt(&self, options: &ArrayMetadataOptions) -> ArrayMetadata {
        self.array().metadata_opt(options)
    }

    pub fn builder(&self) -> ArrayBuilder {
        self.array().builder()
    }

    pub fn chunk_grid_shape(&self) -> &[u64] {
        self.array().chunk_grid_shape()
    }

    pub fn subchunk_shape(&self) -> Option<ChunkShape> {
        self.array().subchunk_shape()
    }

    pub fn subchunk_grid(&self) -> &ChunkGrid {
        self.array().subchunk_grid()
    }

    pub fn subchunk_grid_shape(&self) -> ArrayShape {
        self.array().subchunk_grid_shape()
    }

    pub fn chunk_key(&self, chunk_indices: &[u64]) -> StoreKey {
        self.array().chunk_key(chunk_indices)
    }

    pub fn chunk_origin(&self, chunk_indices: &[u64]) -> Result<ArrayIndices, ArrayError> {
        self.array().chunk_origin(chunk_indices)
    }

    pub fn chunk_shape(&self, chunk_indices: &[u64]) -> Result<ChunkShape, ArrayError> {
        self.array().chunk_shape(chunk_indices)
    }

    pub fn partial_decode_granularity(
        &self,
        chunk_indices: &[u64],
    ) -> Result<ChunkShape, ArrayError> {
        self.array().partial_decode_granularity(chunk_indices)
    }

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
