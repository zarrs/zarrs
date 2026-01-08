macro_rules! vlen_v2_module {
    ($module:ident, $module_codec:ident, $struct:ident, $identifier:literal) => {
        mod $module_codec;

        use std::sync::Arc;

        pub use $module_codec::$struct;

        use crate::array::codec::{Codec, CodecPlugin};
        use crate::metadata::v3::MetadataV3;
        use crate::plugin::{PluginCreateError, PluginMetadataInvalidError};
        use zarrs_plugin::ExtensionIdentifier;

        // Register the codec.
        inventory::submit! {
            CodecPlugin::new($struct::IDENTIFIER, matches_name, default_name, create_codec)
        }

        fn matches_name(name: &str, version: zarrs_plugin::ZarrVersions) -> bool {
            $struct::matches_name(name, version)
        }

        fn default_name(version: zarrs_plugin::ZarrVersions) -> std::borrow::Cow<'static, str> {
            $struct::default_name(version)
        }

        fn create_codec(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
            if metadata.configuration_is_none_or_empty() {
                let codec = Arc::new($struct::new());
                Ok(Codec::ArrayToBytes(codec))
            } else {
                Err(PluginMetadataInvalidError::new(
                    $struct::IDENTIFIER,
                    "codec",
                    metadata.to_string(),
                )
                .into())
            }
        }
    };
}

macro_rules! vlen_v2_codec {
    ($struct:ident, $identifier:literal) => {
        use std::sync::Arc;
        use std::sync::{LazyLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

        #[cfg(feature = "async")]
        use crate::array::codec::{AsyncArrayPartialDecoderTraits, AsyncBytesPartialDecoderTraits};
        use crate::array::{
            ArrayBytes, ArrayBytesRaw, BytesRepresentation,
            RecommendedConcurrency,
            codec::{
                ArrayCodecTraits,
                ArrayPartialDecoderTraits, ArrayPartialEncoderTraits, ArrayToBytesCodecTraits,
                BytesPartialDecoderTraits, BytesPartialEncoderTraits, CodecError,
                CodecMetadataOptions, CodecOptions, CodecTraits, PartialDecoderCapability,
                PartialEncoderCapability, array_to_bytes::vlen_v2::VlenV2Codec,
            },
        };
        use crate::metadata::Configuration;
        use zarrs_plugin::{
            ExtensionAliases, ExtensionAliasesConfig, ExtensionIdentifier, ZarrVersion2,
            ZarrVersion3,
        };

        #[doc = concat!("The `", $identifier, "` codec implementation.")]
        #[derive(Debug, Clone)]
        pub struct $struct {
            inner: Arc<VlenV2Codec>,
        }

        impl $struct {
            #[doc = concat!("Create a new `", $identifier, "` codec.")]
            #[must_use]
            pub fn new() -> Self {
                Self {
                    inner: Arc::new(VlenV2Codec::new()),
                }
            }
        }

        impl Default for $struct {
            fn default() -> Self {
                Self::new()
            }
        }

        impl CodecTraits for $struct {
            fn identifier(&self) -> &'static str {
                Self::IDENTIFIER
            }

            fn configuration(
                &self,
                name: &str,
                options: &CodecMetadataOptions,
            ) -> Option<Configuration> {
                self.inner.configuration(name, options)
            }

            fn partial_decoder_capability(&self) -> PartialDecoderCapability {
                self.inner.partial_decoder_capability()
            }

            fn partial_encoder_capability(&self) -> PartialEncoderCapability {
                self.inner.partial_encoder_capability()
            }
        }

        impl ArrayCodecTraits for $struct {
            fn recommended_concurrency(
                &self,
                shape: &[std::num::NonZeroU64],
                data_type: &crate::array::DataType,
            ) -> Result<RecommendedConcurrency, CodecError> {
                self.inner.recommended_concurrency(shape, data_type)
            }
        }

        #[cfg_attr(
            all(feature = "async", not(target_arch = "wasm32")),
            async_trait::async_trait
        )]
        #[cfg_attr(all(feature = "async", target_arch = "wasm32"), async_trait::async_trait(?Send))]
        impl ArrayToBytesCodecTraits for $struct {
            fn into_dyn(self: Arc<Self>) -> Arc<dyn ArrayToBytesCodecTraits> {
                self as Arc<dyn ArrayToBytesCodecTraits>
            }

            fn encode<'a>(
                &self,
                bytes: ArrayBytes<'a>,
                shape: &[std::num::NonZeroU64],
                data_type: &crate::array::DataType,
                fill_value: &crate::array::FillValue,
                options: &CodecOptions,
            ) -> Result<ArrayBytesRaw<'a>, CodecError> {
                self.inner
                    .encode(bytes, shape, data_type, fill_value, options)
            }

            fn decode<'a>(
                &self,
                bytes: ArrayBytesRaw<'a>,
                shape: &[std::num::NonZeroU64],
                data_type: &crate::array::DataType,
                fill_value: &crate::array::FillValue,
                options: &CodecOptions,
            ) -> Result<ArrayBytes<'a>, CodecError> {
                self.inner
                    .decode(bytes, shape, data_type, fill_value, options)
            }

            fn partial_decoder(
                self: Arc<Self>,
                input_handle: Arc<dyn BytesPartialDecoderTraits>,
                shape: &[std::num::NonZeroU64],
                data_type: &crate::array::DataType,
                fill_value: &crate::array::FillValue,
                options: &CodecOptions,
            ) -> Result<Arc<dyn ArrayPartialDecoderTraits>, CodecError> {
                self.inner.clone().partial_decoder(
                    input_handle,
                    shape,
                    data_type,
                    fill_value,
                    options,
                )
            }

            fn partial_encoder(
                self: Arc<Self>,
                input_output_handle: Arc<dyn BytesPartialEncoderTraits>,
                shape: &[std::num::NonZeroU64],
                data_type: &crate::array::DataType,
                fill_value: &crate::array::FillValue,
                options: &CodecOptions,
            ) -> Result<Arc<dyn ArrayPartialEncoderTraits>, CodecError> {
                self.inner.clone().partial_encoder(
                    input_output_handle,
                    shape,
                    data_type,
                    fill_value,
                    options,
                )
            }

            #[cfg(feature = "async")]
            async fn async_partial_decoder(
                self: Arc<Self>,
                input_handle: Arc<dyn AsyncBytesPartialDecoderTraits>,
                shape: &[std::num::NonZeroU64],
                data_type: &crate::array::DataType,
                fill_value: &crate::array::FillValue,
                options: &CodecOptions,
            ) -> Result<Arc<dyn AsyncArrayPartialDecoderTraits>, CodecError> {
                self.inner
                    .clone()
                    .async_partial_decoder(input_handle, shape, data_type, fill_value, options)
                    .await
            }

            fn encoded_representation(
                &self,
                shape: &[std::num::NonZeroU64],
                data_type: &crate::array::DataType,
                fill_value: &crate::array::FillValue,
            ) -> Result<BytesRepresentation, CodecError> {
                self.inner
                    .encoded_representation(shape, data_type, fill_value)
            }
        }

        paste::paste! {
            static [<$struct:upper _ALIASES_V3>]: LazyLock<RwLock<ExtensionAliasesConfig>> =
                LazyLock::new(|| RwLock::new(ExtensionAliasesConfig::new($identifier, vec![], vec![])));

            static [<$struct:upper _ALIASES_V2>]: LazyLock<RwLock<ExtensionAliasesConfig>> =
                LazyLock::new(|| RwLock::new(ExtensionAliasesConfig::new($identifier, vec![], vec![])));

            impl ExtensionAliases<ZarrVersion3> for $struct {
                fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
                    [<$struct:upper _ALIASES_V3>].read().unwrap()
                }

                fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
                    [<$struct:upper _ALIASES_V3>].write().unwrap()
                }
            }

            impl ExtensionAliases<ZarrVersion2> for $struct {
                fn aliases() -> RwLockReadGuard<'static, ExtensionAliasesConfig> {
                    [<$struct:upper _ALIASES_V2>].read().unwrap()
                }

                fn aliases_mut() -> RwLockWriteGuard<'static, ExtensionAliasesConfig> {
                    [<$struct:upper _ALIASES_V2>].write().unwrap()
                }
            }

            impl ExtensionIdentifier for $struct {
                const IDENTIFIER: &'static str = $identifier;
            }
        }
    };
}

pub(crate) use {vlen_v2_codec, vlen_v2_module};
