//! The `numpy.timedelta64` data type.

use std::num::NonZeroU32;

use zarrs_metadata::ConfigurationSerialize;
use zarrs_plugin::{ExtensionIdentifier, PluginCreateError, PluginMetadataInvalidError};

use super::macros::{impl_bitround_codec, impl_packbits_codec, impl_pcodec_codec, impl_zfp_codec};
use crate::metadata_ext::data_type::{
    NumpyTimeUnit, numpy_timedelta64::NumpyTimeDelta64DataTypeConfigurationV1,
};

/// The `numpy.timedelta64` data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumpyTimeDelta64DataType {
    /// The `NumPy` temporal unit.
    pub unit: NumpyTimeUnit,
    /// The `NumPy` temporal scale factor.
    pub scale_factor: NonZeroU32,
}
zarrs_plugin::impl_extension_aliases!(NumpyTimeDelta64DataType, "numpy.timedelta64");

impl NumpyTimeDelta64DataType {
    /// Static instance for use in trait implementations (bytes codec only).
    /// Uses `NumpyTimeUnit::Generic` and scale factor 1 as defaults.
    pub const STATIC: Self = Self {
        unit: NumpyTimeUnit::Generic,
        scale_factor: NonZeroU32::new(1).unwrap(),
    };

    /// Create a new `numpy.timedelta64` data type.
    #[must_use]
    pub const fn new(unit: NumpyTimeUnit, scale_factor: NonZeroU32) -> Self {
        Self { unit, scale_factor }
    }
}

impl zarrs_data_type::DataTypeExtension for NumpyTimeDelta64DataType {
    fn identifier(&self) -> &'static str {
        <Self as ExtensionIdentifier>::IDENTIFIER
    }

    fn configuration(&self) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::from(NumpyTimeDelta64DataTypeConfigurationV1 {
            unit: self.unit,
            scale_factor: self.scale_factor,
        })
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(8)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::v3::FillValueMetadataV3,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        let err = || {
            zarrs_data_type::DataTypeFillValueMetadataError::new(
                self.identifier().to_string(),
                fill_value_metadata.clone(),
            )
        };
        // Handle "NaT" (Not a Time) as i64::MIN
        if let Some("NaT") = fill_value_metadata.as_str() {
            return Ok(zarrs_data_type::FillValue::from(i64::MIN));
        }
        // Otherwise expect an integer
        let i = fill_value_metadata.as_i64().ok_or_else(err)?;
        Ok(zarrs_data_type::FillValue::from(i))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<zarrs_metadata::v3::FillValueMetadataV3, zarrs_data_type::DataTypeFillValueError>
    {
        let error = || {
            zarrs_data_type::DataTypeFillValueError::new(
                self.identifier().to_string(),
                fill_value.clone(),
            )
        };
        let bytes: [u8; 8] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let i = i64::from_ne_bytes(bytes);
        if i == i64::MIN {
            Ok(zarrs_metadata::v3::FillValueMetadataV3::from("NaT"))
        } else {
            Ok(zarrs_metadata::v3::FillValueMetadataV3::from(i))
        }
    }

    fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
        Some(self)
    }

    fn codec_bitround(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBitroundCodec> {
        Some(self)
    }

    fn codec_pcodec(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionPcodecCodec> {
        Some(self)
    }

    fn codec_zfp(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionZfpCodec> {
        Some(self)
    }

    fn codec_packbits(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionPackBitsCodec> {
        Some(self)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl zarrs_data_type::DataTypeExtensionBytesCodec for NumpyTimeDelta64DataType {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        let endianness = endianness
            .ok_or(zarrs_data_type::DataTypeExtensionBytesCodecError::EndiannessNotSpecified)?;
        if endianness == zarrs_metadata::Endianness::native() {
            Ok(bytes)
        } else {
            let mut result = bytes.into_owned();
            for chunk in result.chunks_exact_mut(8) {
                chunk.reverse();
            }
            Ok(std::borrow::Cow::Owned(result))
        }
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
        self.encode(bytes, endianness)
    }
}

impl_pcodec_codec!(NumpyTimeDelta64DataType, I64);
impl_bitround_codec!(NumpyTimeDelta64DataType, 8, int64);
impl_zfp_codec!(NumpyTimeDelta64DataType, Int64);
impl_packbits_codec!(NumpyTimeDelta64DataType, 64, signed, 1);

// Custom plugin registration for NumpyTimeDelta64DataType (has configuration)
inventory::submit! {
    zarrs_data_type::DataTypePlugin::new(
        <NumpyTimeDelta64DataType as ExtensionIdentifier>::IDENTIFIER,
        <NumpyTimeDelta64DataType as ExtensionIdentifier>::matches_name,
        <NumpyTimeDelta64DataType as ExtensionIdentifier>::default_name,
        |metadata: &zarrs_metadata::v3::MetadataV3| -> Result<std::sync::Arc<dyn zarrs_data_type::DataTypeExtension>, PluginCreateError> {
            let configuration = metadata.configuration().ok_or_else(|| {
                PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                    NumpyTimeDelta64DataType::IDENTIFIER,
                    "data_type",
                    "missing configuration".to_string(),
                ))
            })?;
            let config = NumpyTimeDelta64DataTypeConfigurationV1::try_from_configuration(configuration.clone())
                .map_err(|_| {
                    PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                        NumpyTimeDelta64DataType::IDENTIFIER,
                        "data_type",
                        metadata.to_string(),
                    ))
                })?;
            Ok(std::sync::Arc::new(NumpyTimeDelta64DataType::new(config.unit, config.scale_factor)))
        },
    )
}
