use inherent::inherent;

use super::super::super::{
    chunk_grid_default_name, chunk_key_encoding_default_name, codec_default_name, data_type,
    storage_transformer_default_name,
};
use super::super::*;
use super::{ArrayOps, maybe_regular_chunk_grid_shape};
use crate::config::MetadataConvertVersion;
use crate::convert::array_metadata_v2_to_v3;
use crate::node::data_key;
use zarrs_codec::ChunkGridDecoded;
use zarrs_metadata::v2::DataTypeMetadataV2;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::ZarrVersion;

#[inherent]
impl<TStorage: ?Sized> ArrayOps for Array<TStorage> {
    type Storage = TStorage;

    pub fn storage(&self) -> Arc<TStorage> {
        self.storage.clone()
    }

    pub fn path(&self) -> &NodePath {
        &self.path
    }

    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    pub fn fill_value(&self) -> &FillValue {
        &self.fill_value
    }

    pub fn shape(&self) -> &[u64] {
        self.chunk_grid().array_shape()
    }

    pub fn dimensionality(&self) -> usize {
        self.shape().len()
    }

    pub fn codecs(&self) -> Arc<CodecChain> {
        self.codecs.clone()
    }

    pub fn codecs_bound(&self) -> Arc<CodecChainBound> {
        self.codecs_bound.clone()
    }

    pub fn chunk_grid(&self) -> &ChunkGrid {
        &self.chunk_grid
    }

    pub fn chunk_key_encoding(&self) -> &ChunkKeyEncoding {
        &self.chunk_key_encoding
    }

    pub fn storage_transformers(&self) -> &StorageTransformerChain {
        &self.storage_transformers
    }

    pub fn codec_options(&self) -> &CodecOptions {
        &self.codec_options
    }

    pub fn metadata_options(&self) -> &ArrayMetadataOptions {
        &self.metadata_options
    }

    pub fn metadata_erase_version(&self) -> MetadataEraseVersion {
        self.metadata_erase_version
    }

    pub fn dimension_names(&self) -> &Option<Vec<DimensionName>> {
        &self.dimension_names
    }

    pub fn attributes(&self) -> &serde_json::Map<String, serde_json::Value> {
        match &self.metadata {
            ArrayMetadata::V3(metadata) => &metadata.attributes,
            ArrayMetadata::V2(metadata) => &metadata.attributes,
        }
    }

    pub fn metadata(&self) -> &ArrayMetadata {
        &self.metadata
    }

    #[allow(clippy::missing_panics_doc, clippy::too_many_lines)]
    pub fn metadata_opt(&self, options: &ArrayMetadataOptions) -> ArrayMetadata {
        use ArrayMetadata as AM;
        use MetadataConvertVersion as V;
        let mut metadata = self.metadata.clone();

        // Attribute manipulation
        if options.include_zarrs_metadata() {
            #[derive(serde::Serialize)]
            struct ZarrsMetadata {
                description: String,
                repository: String,
                version: String,
            }
            let zarrs_metadata = ZarrsMetadata {
                description: "This array was created with zarrs".to_string(),
                repository: env!("CARGO_PKG_REPOSITORY").to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            };
            let attributes = match &mut metadata {
                AM::V3(metadata) => &mut metadata.attributes,
                AM::V2(metadata) => &mut metadata.attributes,
            };
            attributes.insert("_zarrs".to_string(), unsafe {
                serde_json::to_value(zarrs_metadata).unwrap_unchecked()
            });
        }

        // Convert version
        let mut metadata = match (metadata, options.metadata_convert_version()) {
            (AM::V3(metadata), V::Default | V::V3) => ArrayMetadata::V3(metadata),
            (AM::V2(metadata), V::Default) => ArrayMetadata::V2(metadata),
            (AM::V2(metadata), V::V3) => {
                let metadata = array_metadata_v2_to_v3(&metadata)
                    .expect("conversion succeeded on array creation");
                AM::V3(metadata)
            }
        };

        // Convert aliased extension names
        if options.convert_aliased_extension_names() {
            match &mut metadata {
                AM::V3(metadata) => {
                    // Codecs
                    for codec in &mut metadata.codecs {
                        let name = codec_default_name(codec, ZarrVersion::V3).into_owned();
                        codec.set_name(name);
                    }
                    // Data type
                    {
                        let name =
                            data_type::data_type_v3_default_name(&metadata.data_type).into_owned();
                        metadata.data_type.set_name(name);
                    }
                    // Chunk grid
                    {
                        let array_shape: ArrayShape = metadata.shape.clone();
                        let name = chunk_grid_default_name(
                            &metadata.chunk_grid,
                            &array_shape,
                            ZarrVersion::V3,
                        )
                        .into_owned();
                        metadata.chunk_grid.set_name(name);
                    }
                    // Chunk key encoding
                    {
                        let name = chunk_key_encoding_default_name(
                            &metadata.chunk_key_encoding,
                            ZarrVersion::V3,
                        )
                        .into_owned();
                        metadata.chunk_key_encoding.set_name(name);
                    }
                    // Storage transformers
                    for transformer in &mut metadata.storage_transformers {
                        let name = storage_transformer_default_name(
                            transformer,
                            &self.path,
                            ZarrVersion::V3,
                        )
                        .into_owned();
                        transformer.set_name(name);
                    }
                }
                AM::V2(metadata) => {
                    if let Some(filters) = &mut metadata.filters {
                        for filter in filters {
                            let filter_metadata = MetadataV3::new_with_serializable_configuration(
                                filter.id().to_string(),
                                filter.configuration(),
                            )
                            .unwrap_or_else(|_| MetadataV3::new(filter.id()));
                            let name =
                                codec_default_name(&filter_metadata, ZarrVersion::V2).into_owned();
                            filter.set_id(name);
                        }
                    }
                    if let Some(compressor) = &mut metadata.compressor {
                        let compressor_metadata = MetadataV3::new_with_serializable_configuration(
                            compressor.id().to_string(),
                            compressor.configuration(),
                        )
                        .unwrap_or_else(|_| MetadataV3::new(compressor.id()));
                        let name =
                            codec_default_name(&compressor_metadata, ZarrVersion::V2).into_owned();
                        compressor.set_id(name);
                    }
                    match &mut metadata.dtype {
                        DataTypeMetadataV2::Simple(dtype) => {
                            *dtype = data_type::data_type_v2_default_name(dtype).into_owned();
                        }
                        DataTypeMetadataV2::Structured(_) => {
                            // FIXME: structured data type support
                        }
                    }
                }
            }
        }

        metadata
    }

    pub fn builder(&self) -> ArrayBuilder {
        ArrayBuilder::from_array(self)
    }

    pub fn chunk_grid_shape(&self) -> &[u64] {
        self.chunk_grid().grid_shape()
    }

    pub fn subchunk_shape(&self) -> Option<ChunkShape> {
        self.subchunk_grid()
            .and_then(maybe_regular_chunk_grid_shape)
    }

    pub fn subchunk_grid(&self) -> Option<&ChunkGrid> {
        match &self.subchunk_grid {
            ChunkGridDecoded::Array(subchunk_grid) => Some(subchunk_grid),
            ChunkGridDecoded::None | ChunkGridDecoded::ChunkLocal => None,
        }
    }

    pub fn chunk_key(&self, chunk_indices: &[u64]) -> StoreKey {
        data_key(self.path(), &self.chunk_key_encoding.encode(chunk_indices))
    }

    pub fn chunk_origin(&self, chunk_indices: &[u64]) -> Result<ArrayIndices, ArrayError> {
        self.chunk_grid()
            .chunk_origin(chunk_indices)
            .map_err(|_| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))
    }

    pub fn chunk_shape(&self, chunk_indices: &[u64]) -> Result<ChunkShape, ArrayError> {
        self.chunk_grid()
            .chunk_shape(chunk_indices)
            .map_err(|_| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))
    }

    pub fn subset_all(&self) -> ArraySubset {
        ArraySubset::new_with_shape(self.shape().to_vec())
    }

    pub fn chunk_shape_usize(&self, chunk_indices: &[u64]) -> Result<Vec<usize>, ArrayError> {
        Ok(self
            .chunk_shape(chunk_indices)?
            .iter()
            .map(|d| usize::try_from(d.get()).unwrap())
            .collect())
    }

    pub fn chunk_subset(&self, chunk_indices: &[u64]) -> Result<ArraySubset, ArrayError> {
        self.chunk_grid()
            .subset(chunk_indices)
            .map_err(|_| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))?
            .ok_or_else(|| ArrayError::InvalidChunkGridIndicesError(chunk_indices.to_vec()))
    }

    pub fn chunk_subset_bounded(&self, chunk_indices: &[u64]) -> Result<ArraySubset, ArrayError> {
        let chunk_subset = self.chunk_subset(chunk_indices)?;
        Ok(chunk_subset.bound(self.shape())?)
    }

    #[allow(clippy::similar_names)]
    pub fn chunks_subset(&self, chunks: &dyn ArraySubsetTraits) -> Result<ArraySubset, ArrayError> {
        match chunks.end_inc() {
            Some(end) => {
                let chunk0 = self.chunk_subset(&chunks.start())?;
                let chunk1 = self.chunk_subset(&end)?;
                let start = chunk0.start().to_vec();
                let end = chunk1.end_exc();
                ArraySubset::new_with_start_end_exc(start, end).map_err(std::convert::Into::into)
            }
            None => Ok(ArraySubset::new_empty(chunks.dimensionality())),
        }
    }

    pub fn chunks_subset_bounded(
        &self,
        chunks: &dyn ArraySubsetTraits,
    ) -> Result<ArraySubset, ArrayError> {
        let chunks_subset = self.chunks_subset(chunks)?;
        Ok(chunks_subset.bound(self.shape())?)
    }

    pub fn chunks_in_array_subset(
        &self,
        array_subset: &dyn ArraySubsetTraits,
    ) -> Result<Option<ArraySubset>, IncompatibleDimensionalityError> {
        self.chunk_grid.chunks_in_array_subset(array_subset)
    }
}
