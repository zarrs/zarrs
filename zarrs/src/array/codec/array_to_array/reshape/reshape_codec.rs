// TODO: reshape partial decoder

use std::{num::NonZeroU64, sync::Arc};

#[cfg(feature = "async")]
use crate::array::codec::AsyncArrayPartialDecoderTraits;
use crate::array::{
    DataType, FillValue,
    codec::{
        ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, PartialDecoderCapability,
        PartialEncoderCapability,
    },
};
use crate::metadata::Configuration;
use crate::metadata_ext::codec::reshape::{
    ReshapeCodecConfiguration, ReshapeCodecConfigurationV1, ReshapeShape,
};
use crate::{
    array::{
        ChunkShape,
        codec::{
            ArrayBytes, ArrayCodecTraits, ArrayToArrayCodecTraits, CodecError,
            CodecMetadataOptions, CodecOptions, CodecTraits, RecommendedConcurrency,
        },
    },
    plugin::PluginCreateError,
};
use std::sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
use zarrs_plugin::{
    ExtensionAliases, ExtensionAliasesConfig, ExtensionIdentifier, ZarrVersion2, ZarrVersion3,
};

/// A `reshape` codec implementation.
#[derive(Clone, Debug)]
pub struct ReshapeCodec {
    shape: ReshapeShape,
}

impl ReshapeCodec {
    /// Create a new reshape codec from configuration.
    ///
    /// # Errors
    /// Returns [`PluginCreateError`] if there is a configuration issue.
    pub fn new_with_configuration(
        configuration: &ReshapeCodecConfiguration,
    ) -> Result<Self, PluginCreateError> {
        match configuration {
            ReshapeCodecConfiguration::V1(configuration) => {
                Ok(Self::new(configuration.shape.clone()))
            }
            _ => Err(PluginCreateError::Other(
                "this reshape codec configuration variant is unsupported".to_string(),
            )),
        }
    }

    /// Create a new reshape codec.
    #[must_use]
    pub const fn new(shape: ReshapeShape) -> Self {
        Self { shape }
    }
}

impl CodecTraits for ReshapeCodec {
    fn identifier(&self) -> &'static str {
        Self::IDENTIFIER
    }

    fn configuration(&self, _name: &str, _options: &CodecMetadataOptions) -> Option<Configuration> {
        let configuration = ReshapeCodecConfiguration::V1(ReshapeCodecConfigurationV1 {
            shape: self.shape.clone(),
        });
        Some(configuration.into())
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: true,
            partial_decode: true,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: true,
        }
    }
}

#[cfg_attr(
    all(feature = "async", not(target_arch = "wasm32")),
    async_trait::async_trait
)]
#[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
impl ArrayToArrayCodecTraits for ReshapeCodec {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToArrayCodecTraits> {
        self as Arc<dyn ArrayToArrayCodecTraits>
    }

    fn encoded_data_type(&self, decoded_data_type: &DataType) -> Result<DataType, CodecError> {
        Ok(decoded_data_type.clone())
    }

    fn encoded_fill_value(
        &self,
        _decoded_data_type: &DataType,
        decoded_fill_value: &FillValue,
    ) -> Result<FillValue, CodecError> {
        Ok(decoded_fill_value.clone())
    }

    fn encoded_shape(&self, decoded_shape: &[NonZeroU64]) -> Result<ChunkShape, CodecError> {
        super::get_encoded_shape(&self.shape, decoded_shape)
    }

    fn decoded_shape(
        &self,
        _encoded_shape: &[NonZeroU64],
    ) -> Result<Option<ChunkShape>, CodecError> {
        Ok(None)
    }

    fn encode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: ArrayBytes<'a>,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<ArrayBytes<'a>, CodecError> {
        Ok(bytes)
    }

    fn partial_decoder(
        self: Arc<Self>,
        _input_handle: Arc<dyn ArrayPartialDecoderTraits>,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
        // TODO: reshape partial decoding
        Err(CodecError::Other(
            "partial decoding with the reshape codec is not yet supported".to_string(),
        ))
    }

    fn partial_encoder(
        self: Arc<Self>,
        _input_output_handle: Arc<dyn ArrayPartialEncoderTraits>,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
        // TODO: reshape partial encoding
        Err(CodecError::Other(
            "partial encoding with the reshape codec is not yet supported".to_string(),
        ))
    }

    #[cfg(feature = "async")]
    async fn async_partial_decoder(
        self: Arc<Self>,
        _input_handle: Arc<dyn AsyncArrayPartialDecoderTraits>,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
        _fill_value: &FillValue,
        _options: &CodecOptions,
    ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
        // TODO: reshape partial decoding
        Err(CodecError::Other(
            "partial decoding with the reshape codec is not yet supported".to_string(),
        ))
    }
}

impl ArrayCodecTraits for ReshapeCodec {
    fn recommended_concurrency(
        &self,
        _shape: &[NonZeroU64],
        _data_type: &DataType,
    ) -> Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }
}

static RESHAPE_ALIASES_V3: LazyLock<RwLock<ExtensionAliasesConfig>> =
    LazyLock::new(|| RwLock::new(ExtensionAliasesConfig::new("reshape", vec![], vec![])));

static RESHAPE_ALIASES_V2: LazyLock<RwLock<ExtensionAliasesConfig>> =
    LazyLock::new(|| RwLock::new(ExtensionAliasesConfig::new("reshape", vec![], vec![])));

impl ExtensionAliases<ZarrVersion3> for ReshapeCodec {
    fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
        RESHAPE_ALIASES_V3.read().unwrap()
    }

    fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
        RESHAPE_ALIASES_V3.write().unwrap()
    }
}

impl ExtensionAliases<ZarrVersion2> for ReshapeCodec {
    fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
        RESHAPE_ALIASES_V2.read().unwrap()
    }

    fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
        RESHAPE_ALIASES_V2.write().unwrap()
    }
}

impl ExtensionIdentifier for ReshapeCodec {
    const IDENTIFIER: &'static str = "reshape";
}
