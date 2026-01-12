macro_rules! vlen_v2_module {
    ($module:ident, $module_codec:ident, $struct:ident) => {
        mod $module_codec;

        use std::sync::Arc;

        pub use $module_codec::$struct;

        use crate::array::codec::{Codec, CodecPluginV2, CodecPluginV3};
        use crate::plugin::{PluginConfigurationInvalidError, PluginCreateError};
        use zarrs_metadata::v2::MetadataV2;
        use zarrs_metadata::v3::MetadataV3;

        // Register the V3 codec.
        inventory::submit! {
            CodecPluginV3::new::<$struct>(create_codec_v3)
        }
        // Register the V2 codec.
        inventory::submit! {
            CodecPluginV2::new::<$struct>(create_codec_v2)
        }

        fn create_codec_v3(metadata: &MetadataV3) -> Result<Codec, PluginCreateError> {
            if metadata.configuration().is_none_or(|c| c.is_empty()) {
                let codec = Arc::new($struct::new());
                Ok(Codec::ArrayToBytes(codec))
            } else {
                Err(PluginConfigurationInvalidError::new(metadata.to_string()).into())
            }
        }

        fn create_codec_v2(metadata: &MetadataV2) -> Result<Codec, PluginCreateError> {
            if metadata.configuration().is_empty() {
                let codec = Arc::new($struct::new());
                Ok(Codec::ArrayToBytes(codec))
            } else {
                Err(PluginConfigurationInvalidError::new(format!("{metadata:?}")).into())
            }
        }
    };
}

macro_rules! vlen_v2_codec {
    ($struct:ident, $default_name:literal) => {
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
            ExtensionAliasesConfig, ExtensionAliases,
            ZarrVersion2, ZarrVersion3,
        };

        #[doc = concat!("The `", $default_name, "` codec implementation.")]
        #[derive(Debug, Clone)]
        pub struct $struct {
            inner: Arc<VlenV2Codec>,
        }

        impl $struct {
            #[doc = concat!("Create a new `", $default_name, "` codec.")]
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
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn configuration(
                &self,
                version: zarrs_plugin::ZarrVersions,
                options: &CodecMetadataOptions,
            ) -> Option<Configuration> {
                self.inner.configuration(version, options)
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
                LazyLock::new(|| RwLock::new(ExtensionAliasesConfig::new($default_name, vec![], vec![])));

            static [<$struct:upper _ALIASES_V2>]: LazyLock<RwLock<ExtensionAliasesConfig>> =
                LazyLock::new(|| RwLock::new(ExtensionAliasesConfig::new($default_name, vec![], vec![])));

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

            impl zarrs_plugin::ExtensionNameStatic for $struct {
                const DEFAULT_NAME_FN: fn(zarrs_plugin::ZarrVersions) -> ::core::option::Option<::std::borrow::Cow<'static, str>> = |version| {
                    match version {
                        zarrs_plugin::ZarrVersions::V2 => {
                            let aliases = [<$struct:upper _ALIASES_V2>].read().unwrap();
                            if aliases.default_name.is_empty() {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(aliases.default_name.clone())
                            }
                        }
                        zarrs_plugin::ZarrVersions::V3 => {
                            let aliases = [<$struct:upper _ALIASES_V3>].read().unwrap();
                            if aliases.default_name.is_empty() {
                                ::core::option::Option::None
                            } else {
                                ::core::option::Option::Some(aliases.default_name.clone())
                            }
                        }
                    }
                };
            }

        }
    };
}

pub(crate) use {vlen_v2_codec, vlen_v2_module};
