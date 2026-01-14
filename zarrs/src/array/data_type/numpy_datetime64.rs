//! The `numpy.datetime64` data type.

use std::num::NonZeroU32;

use zarrs_codec::CodecError;
use zarrs_data_type::DataType;
use zarrs_metadata::ConfigurationSerialize;
use zarrs_metadata::v3::MetadataV3;
use zarrs_plugin::{PluginConfigurationInvalidError, PluginCreateError};

use zarrs_metadata_ext::data_type::NumpyTimeUnit;
use zarrs_metadata_ext::data_type::numpy_datetime64::NumpyDateTime64DataTypeConfigurationV1;

/// The `numpy.datetime64` data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NumpyDateTime64DataType {
    /// The `NumPy` temporal unit.
    pub unit: NumpyTimeUnit,
    /// The `NumPy` temporal scale factor.
    pub scale_factor: NonZeroU32,
}

zarrs_plugin::impl_extension_aliases!(NumpyDateTime64DataType, v3: "numpy.datetime64");

impl zarrs_data_type::DataTypeTraitsV3 for NumpyDateTime64DataType {
    fn create(metadata: &MetadataV3) -> Result<DataType, PluginCreateError> {
        let configuration = metadata
            .configuration()
            .ok_or_else(|| PluginConfigurationInvalidError::new(metadata.to_string()))?;
        let config =
            NumpyDateTime64DataTypeConfigurationV1::try_from_configuration(configuration.clone())
                .map_err(|_| PluginConfigurationInvalidError::new(metadata.to_string()))?;
        Ok(std::sync::Arc::new(NumpyDateTime64DataType::new(
            config.unit,
            config.scale_factor,
        ))
        .into())
    }
}

// Register as V3-only data type.
inventory::submit! {
    zarrs_data_type::DataTypePluginV3::new::<NumpyDateTime64DataType>()
}

impl NumpyDateTime64DataType {
    /// Static instance for use in trait implementations (bytes codec only).
    /// Uses `NumpyTimeUnit::Generic` and scale factor 1 as defaults.
    pub const STATIC: Self = Self {
        unit: NumpyTimeUnit::Generic,
        scale_factor: NonZeroU32::new(1).unwrap(),
    };

    /// Create a new `numpy.datetime64` data type.
    #[must_use]
    pub const fn new(unit: NumpyTimeUnit, scale_factor: NonZeroU32) -> Self {
        Self { unit, scale_factor }
    }
}

impl zarrs_data_type::DataTypeTraits for NumpyDateTime64DataType {
    fn configuration(&self, _version: zarrs_plugin::ZarrVersion) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::from(NumpyDateTime64DataTypeConfigurationV1 {
            unit: self.unit,
            scale_factor: self.scale_factor,
        })
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(8)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::FillValueMetadata,
        _version: zarrs_plugin::ZarrVersion,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        // Handle "NaT" (Not a Time) as i64::MIN
        if let Some("NaT") = fill_value_metadata.as_str() {
            return Ok(zarrs_data_type::FillValue::from(i64::MIN));
        }
        // Otherwise expect an integer
        let i = fill_value_metadata
            .as_i64()
            .ok_or(zarrs_data_type::DataTypeFillValueMetadataError)?;
        Ok(zarrs_data_type::FillValue::from(i))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<zarrs_metadata::FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
        let bytes: [u8; 8] = fill_value
            .as_ne_bytes()
            .try_into()
            .map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let i = i64::from_ne_bytes(bytes);
        if i == i64::MIN {
            Ok(zarrs_metadata::FillValueMetadata::from("NaT"))
        } else {
            Ok(zarrs_metadata::FillValueMetadata::from(i))
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl crate::array::codec::BytesCodecDataTypeTraits for NumpyDateTime64DataType {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, CodecError> {
        let endianness = endianness.ok_or(CodecError::from(
            "endianness must be specified for multi-byte data types",
        ))?;
        if endianness == zarrs_metadata::Endianness::native() {
            Ok(bytes)
        } else {
            let mut result = bytes.into_owned();
            for chunk in result.as_chunks_mut::<8>().0 {
                chunk.reverse();
            }
            Ok(std::borrow::Cow::Owned(result))
        }
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, CodecError> {
        self.encode(bytes, endianness)
    }
}

zarrs_codec::register_data_type_extension_codec!(
    NumpyDateTime64DataType,
    crate::array::codec::BytesPlugin,
    crate::array::codec::BytesCodecDataTypeTraits
);
#[cfg(feature = "pcodec")]
crate::array::codec::impl_pcodec_codec!(NumpyDateTime64DataType, I64, 1);
#[cfg(feature = "bitround")]
crate::array::codec::impl_bitround_codec!(NumpyDateTime64DataType, 8, int64);
#[cfg(feature = "zfp")]
crate::array::codec::impl_zfp_codec!(NumpyDateTime64DataType, Int64);
crate::array::codec::impl_packbits_codec!(NumpyDateTime64DataType, 64, signed, 1);
