use zarrs_metadata::Configuration;
use zarrs_metadata::v2::MetadataV2;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::{ExtensionName, MaybeSend, MaybeSync, ZarrVersion};

use crate::{
    Codec, CodecCreateError, CodecMetadataOptions, PartialDecoderCapability,
    PartialEncoderCapability,
};

pub(crate) mod array;
pub(crate) mod array_partial;
pub(crate) mod array_to_array;
pub(crate) mod array_to_bytes;
pub(crate) mod bytes_partial;
pub(crate) mod bytes_to_bytes;

/// Trait for creating a codec from Zarr V2 metadata.
///
/// Types implementing this trait can be registered as V2 codec plugins via [`CodecPluginV2`](crate::CodecPluginV2).
pub trait CodecTraitsV2 {
    /// Create a codec from Zarr V2 metadata.
    ///
    /// # Errors
    /// Returns [`CodecCreateError`] if the codec cannot be created.
    fn create(metadata: &MetadataV2) -> Result<Codec, CodecCreateError>
    where
        Self: Sized;
}

/// Trait for creating a codec from Zarr V3 metadata.
///
/// Types implementing this trait can be registered as V3 codec plugins via [`CodecPluginV3`](crate::CodecPluginV3).
pub trait CodecTraitsV3 {
    /// Create a codec from Zarr V3 metadata.
    ///
    /// # Errors
    /// Returns [`CodecCreateError`] if the codec cannot be created.
    fn create(metadata: &MetadataV3) -> Result<Codec, CodecCreateError>
    where
        Self: Sized;
}

/// Codec traits.
pub trait CodecTraits: ExtensionName + MaybeSend + MaybeSync {
    /// Create the codec configuration.
    ///
    /// A hidden codec (e.g. a cache) will return [`None`], since it will not have any associated metadata.
    fn configuration(
        &self,
        version: ZarrVersion,
        options: &CodecMetadataOptions,
    ) -> Option<Configuration>;

    /// Create the Zarr V3 codec configuration.
    fn configuration_v3(&self, options: &CodecMetadataOptions) -> Option<Configuration> {
        self.configuration(ZarrVersion::V3, options)
    }

    /// Create the Zarr V2 codec configuration.
    fn configuration_v2(&self, options: &CodecMetadataOptions) -> Option<Configuration> {
        self.configuration(ZarrVersion::V2, options)
    }

    /// Returns the partial decoder capability of this codec.
    fn partial_decoder_capability(&self) -> PartialDecoderCapability;

    /// Returns the partial encoder capability of this codec.
    fn partial_encoder_capability(&self) -> PartialEncoderCapability;
}
