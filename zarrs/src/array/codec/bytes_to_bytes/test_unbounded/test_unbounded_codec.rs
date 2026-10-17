use std::sync::Arc;

use zarrs_plugin::ZarrVersion;

use crate::array::{BytesRepresentation, CowBytes};
use zarrs_codec::{
    BytesToBytesCodecTraits, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits, Resources,
};
use zarrs_metadata::Configuration;

zarrs_plugin::impl_extension_aliases!(TestUnboundedCodec, v3: "zarrs.test_unbounded");

/// A `test_unbounded` codec implementation.
#[derive(Clone, Debug)]
pub struct TestUnboundedCodec {}

impl TestUnboundedCodec {
    /// Create a new `test_unbounded` codec.
    ///
    /// # Errors
    /// Returns [`TestUnboundedCompressionLevelError`] if `compression_level` is not valid.
    #[must_use]
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for TestUnboundedCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl CodecTraits for TestUnboundedCodec {
    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        None
    }
}

impl BytesToBytesCodecTraits for TestUnboundedCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self as Arc<dyn BytesToBytesCodecTraits>
    }

    fn encode<'a>(
        &self,
        decoded_value: CowBytes<'a>,
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<CowBytes<'a>, CodecError> {
        Ok(decoded_value)
    }

    fn decode<'a>(
        &self,
        encoded_value: CowBytes<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
        _resources: &Resources,
    ) -> Result<CowBytes<'a>, CodecError> {
        Ok(encoded_value)
    }

    fn encoded_representation(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        BytesRepresentation::UnboundedSize
    }
}
