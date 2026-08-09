use inherent::inherent;
use zarrs_codec::ArrayToBytesCodecSubchunkingTraits;

use super::{ArrayMutOps, *};

#[inherent]
impl<TStorage: ?Sized> ArrayMutOps for Array<TStorage> {
    pub fn set_codec_options(&mut self, codec_options: CodecOptions) -> &mut Self {
        self.codec_options = codec_options;
        self
    }

    /// Reconfigure and rebind the codec chain.
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if a codec cannot be reconfigured or rebound.
    pub fn set_codec_specific_options(
        &mut self,
        opts: &CodecSpecificOptions,
    ) -> Result<&mut Self, CodecCreateError> {
        let codecs = Arc::new((*self.codecs).clone().with_codec_specific_options(opts)?);
        let codecs_bound = codecs
            .clone()
            .with_context(self.data_type.clone(), self.fill_value.clone())?;
        let subchunk_grids = codecs_bound
            .decoded_subchunk_grids((&self.chunk_grid).into())
            .map_err(|err| CodecCreateError::from(err.to_string()))?;
        self.codecs = codecs;
        self.codecs_bound = codecs_bound;
        self.subchunk_grids = subchunk_grids;
        Ok(self)
    }

    pub fn set_metadata_options(&mut self, metadata_options: ArrayMetadataOptions) -> &mut Self {
        self.metadata_options = metadata_options;
        self
    }

    pub fn set_metadata_erase_version(
        &mut self,
        metadata_erase_version: MetadataEraseVersion,
    ) -> &mut Self {
        self.metadata_erase_version = metadata_erase_version;
        self
    }

    pub fn set_shape(&mut self, array_shape: ArrayShape) -> Result<&mut Self, ArrayCreateError> {
        // The dimensionality of an array is fixed once it exists. Most chunk grids would reject
        // this anyway, but not with a consistent error. Checked before any mutation.
        if array_shape.len() != self.dimensionality() {
            return Err(ArrayCreateError::ChangedDimensionality(
                array_shape.len(),
                self.dimensionality(),
            ));
        }

        self.chunk_grid = ChunkGrid::from_metadata(&self.chunk_grid.metadata(), &array_shape)
            .map_err(ArrayCreateError::ChunkGridCreateError)?;
        self.subchunk_grids = self
            .codecs_bound
            .decoded_subchunk_grids((&self.chunk_grid).into())?;
        match Arc::make_mut(&mut self.metadata) {
            ArrayMetadata::V3(metadata) => {
                metadata.shape = array_shape;
            }
            ArrayMetadata::V2(metadata) => {
                metadata.shape = array_shape;
            }
        }
        Ok(self)
    }

    /// Set the dimension names.
    ///
    /// # Errors
    /// Returns an [`ArrayCreateError`] if `dimension_names` is `Some` and its length does not
    /// match the array dimensionality.
    pub fn set_dimension_names(
        &mut self,
        dimension_names: Option<Vec<DimensionName>>,
    ) -> Result<&mut Self, ArrayCreateError> {
        // Matches the validation performed when an array is created
        if let Some(dimension_names) = &dimension_names
            && dimension_names.len() != self.dimensionality()
        {
            return Err(ArrayCreateError::InvalidDimensionNames(
                dimension_names.len(),
                self.dimensionality(),
            ));
        }

        // Write through to the metadata document, otherwise `store_metadata` would not see the
        // change. Zarr V2 metadata has no dimension names field; they are carried into the
        // converted metadata by `metadata_opt` when it converts to Zarr V3.
        if let ArrayMetadata::V3(metadata) = Arc::make_mut(&mut self.metadata) {
            metadata.dimension_names.clone_from(&dimension_names);
        }
        self.dimension_names = dimension_names;
        Ok(self)
    }

    pub fn attributes_mut(&mut self) -> &mut serde_json::Map<String, serde_json::Value> {
        match Arc::make_mut(&mut self.metadata) {
            ArrayMetadata::V3(metadata) => &mut metadata.attributes,
            ArrayMetadata::V2(metadata) => &mut metadata.attributes,
        }
    }
}
