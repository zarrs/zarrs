use super::*;

/// Mutable array metadata/configuration operations.
pub trait ArrayMutOps: ArrayOps {
    /// Set the codec options.
    fn set_codec_options(&mut self, codec_options: CodecOptions) -> &mut Self;

    /// Reconfigure the codec chain with codec-specific options.
    ///
    /// Refer to [`with_codec_specific_options`](Array::with_codec_specific_options) for details.
    ///
    /// # Errors
    /// Returns a [`CodecCreateError`] if a codec cannot be reconfigured or rebound.
    ///
    /// # Example
    /// ```rust,no_run
    /// # use std::sync::Arc;
    /// # use zarrs::array::Array;
    /// use zarrs::array::codec::ShardingCodecOptions;
    /// use zarrs_codec::CodecSpecificOptions;
    /// # let store = Arc::new(zarrs_filesystem::FilesystemStore::new("tests/data/array_write_read.zarr")?);
    /// # let mut array = Array::open(store, "/group/array")?;
    /// let opts = CodecSpecificOptions::default()
    ///     .with_option(ShardingCodecOptions::default());
    /// array.set_codec_specific_options(&opts);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn set_codec_specific_options(
        &mut self,
        opts: &CodecSpecificOptions,
    ) -> Result<&mut Self, CodecCreateError>;

    /// Set the metadata options.
    fn set_metadata_options(&mut self, metadata_options: ArrayMetadataOptions) -> &mut Self;

    /// Set the metadata erase version.
    fn set_metadata_erase_version(
        &mut self,
        metadata_erase_version: MetadataEraseVersion,
    ) -> &mut Self;

    /// Set the array shape.
    ///
    /// The dimensionality of an array is fixed once it has been created or opened, so
    /// `array_shape` must have the same length as the current [`shape`](super::ArrayOps::shape).
    ///
    /// # Errors
    /// Returns an [`ArrayCreateError`] if `array_shape` changes the array dimensionality, or the
    /// chunk grid is not compatible with `array_shape`.
    #[allow(clippy::missing_errors_doc)]
    fn set_shape(&mut self, array_shape: ArrayShape) -> Result<&mut Self, super::ArrayCreateError>;

    /// Set the dimension names.
    ///
    /// Zarr V2 metadata has no dimension names field, so for a Zarr V2 array these are only
    /// written on store if the metadata options convert the metadata to Zarr V3.
    ///
    /// # Errors
    /// Returns an [`ArrayCreateError`](super::ArrayCreateError) if `dimension_names` is `Some` and
    /// its length does not match the array dimensionality.
    #[allow(clippy::missing_errors_doc)]
    fn set_dimension_names(
        &mut self,
        dimension_names: Option<Vec<DimensionName>>,
    ) -> Result<&mut Self, super::ArrayCreateError>;

    /// Mutably borrow the array attributes.
    fn attributes_mut(&mut self) -> &mut serde_json::Map<String, serde_json::Value>;
}
