use std::ops::Deref;
use std::sync::Arc;

use zarrs_plugin::ZarrVersions;

use super::{
    ArrayToArrayCodecTraits, ArrayToBytesCodecTraits, BytesToBytesCodecTraits,
    CodecMetadataOptions, CodecPlugin, CodecTraits,
};
use crate::metadata::Configuration;

/// A named codec.
#[derive(Clone, Debug)]
pub struct NamedCodec<T: CodecTraits + ?Sized> {
    name: String,
    codec: Arc<T>,
}

impl<T: CodecTraits + ?Sized> NamedCodec<T> {
    /// Create a new [`NamedCodec`].
    #[must_use]
    pub fn new(name: String, codec: Arc<T>) -> Self {
        Self { name, codec }
    }

    /// Create a new [`NamedCodec`] with the default name for the codec.
    #[must_use]
    pub fn new_default_name(codec: Arc<T>) -> Self {
        let identifier = codec.identifier();
        let name = inventory::iter::<CodecPlugin>()
            .find(|plugin| plugin.identifier() == identifier)
            .map_or_else(
                || identifier.to_string(),
                |plugin| plugin.default_name(ZarrVersions::V3).into_owned(),
            );
        Self { name, codec }
    }

    /// The name of the codec.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create the codec configuration.
    ///
    /// See [`CodecTraits::configuration`].
    #[must_use]
    pub fn configuration(&self, options: &CodecMetadataOptions) -> Option<Configuration> {
        self.codec().configuration(self.name(), options)
    }

    /// The underlying codec.
    #[must_use]
    pub fn codec(&self) -> &Arc<T> {
        &self.codec
    }
}

impl<T: CodecTraits + ?Sized> Deref for NamedCodec<T> {
    type Target = Arc<T>;

    fn deref(&self) -> &Self::Target {
        &self.codec
    }
}

macro_rules! impl_named_codec {
    ($named_codec:ident, $codec_trait:ident) => {
        impl Clone for $named_codec {
            fn clone(&self) -> Self {
                Self {
                    name: self.name.clone(),
                    codec: self.codec.clone(),
                }
            }
        }
    };
}

/// A named array to array codec.
pub type NamedArrayToArrayCodec = NamedCodec<dyn ArrayToArrayCodecTraits>;
impl_named_codec!(NamedArrayToArrayCodec, ArrayToArrayCodecTraits);

/// A named array to bytes codec.
pub type NamedArrayToBytesCodec = NamedCodec<dyn ArrayToBytesCodecTraits>;
impl_named_codec!(NamedArrayToBytesCodec, ArrayToBytesCodecTraits);

/// A named bytes to bytes codec.
pub type NamedBytesToBytesCodec = NamedCodec<dyn BytesToBytesCodecTraits>;
impl_named_codec!(NamedBytesToBytesCodec, BytesToBytesCodecTraits);
