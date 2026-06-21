use inherent::inherent;

use super::super::*;
use super::ArrayMutOps;

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
        self.codecs = codecs;
        self.codecs_bound = codecs_bound;
        Ok(self)
    }

    pub fn set_metadata_options(&mut self, metadata_options: ArrayMetadataOptions) -> &mut Self {
        self.metadata_options = metadata_options;
        self
    }

    pub fn set_shape(&mut self, array_shape: ArrayShape) -> Result<&mut Self, ArrayCreateError> {
        self.chunk_grid = ChunkGrid::from_metadata(&self.chunk_grid.metadata(), &array_shape)
            .map_err(ArrayCreateError::ChunkGridCreateError)?;
        self.subchunk_grid = crate::array::array_sharded_ext::create_subchunk_grid(
            &self.chunk_grid,
            self.codecs_bound.as_ref(),
        );
        match &mut self.metadata {
            ArrayMetadata::V3(metadata) => {
                metadata.shape = array_shape;
            }
            ArrayMetadata::V2(metadata) => {
                metadata.shape = array_shape;
            }
        }
        Ok(self)
    }

    pub fn set_dimension_names(
        &mut self,
        dimension_names: Option<Vec<DimensionName>>,
    ) -> &mut Self {
        self.dimension_names = dimension_names;
        self
    }

    pub fn attributes_mut(&mut self) -> &mut serde_json::Map<String, serde_json::Value> {
        match &mut self.metadata {
            ArrayMetadata::V3(metadata) => &mut metadata.attributes,
            ArrayMetadata::V2(metadata) => &mut metadata.attributes,
        }
    }
}
