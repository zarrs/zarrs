use std::ops::Deref;

use base64::{Engine, prelude::BASE64_STANDARD};
use zarrs_data_type::DataTypePlugin;
use zarrs_data_type::{DataTypeFillValueMetadataError, FillValue};
use zarrs_metadata::{
    ConfigurationSerialize,
    v3::{FillValueMetadataV3, MetadataV3},
};
use zarrs_plugin::{
    ExtensionIdentifier, PluginCreateError, PluginMetadataInvalidError, PluginUnsupportedError,
    ZarrVersions,
};

use crate::array::{
    DataType,
    data_type::{
        BFloat16DataType, BoolDataType, BytesDataType, Complex64DataType, Complex128DataType,
        ComplexBFloat16DataType, ComplexFloat4E2M1FNDataType, ComplexFloat6E2M3FNDataType,
        ComplexFloat6E3M2FNDataType, ComplexFloat8E3M4DataType, ComplexFloat8E4M3B11FNUZDataType,
        ComplexFloat8E4M3DataType, ComplexFloat8E4M3FNUZDataType, ComplexFloat8E5M2DataType,
        ComplexFloat8E5M2FNUZDataType, ComplexFloat8E8M0FNUDataType, ComplexFloat16DataType,
        ComplexFloat32DataType, ComplexFloat64DataType, Float4E2M1FNDataType, Float6E2M3FNDataType,
        Float6E3M2FNDataType, Float8E3M4DataType, Float8E4M3B11FNUZDataType, Float8E4M3DataType,
        Float8E4M3FNUZDataType, Float8E5M2DataType, Float8E5M2FNUZDataType, Float8E8M0FNUDataType,
        Float16DataType, Float32DataType, Float64DataType, Int2DataType, Int4DataType,
        Int8DataType, Int16DataType, Int32DataType, Int64DataType, NumpyDateTime64DataType,
        NumpyTimeDelta64DataType, OptionalDataType, RawBitsDataType, StringDataType, UInt2DataType,
        UInt4DataType, UInt8DataType, UInt16DataType, UInt32DataType, UInt64DataType,
        complex_subfloat_hex_string_to_fill_value, subfloat_hex_string_to_fill_value,
    },
};

/// A named data type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedDataType {
    name: String,
    data_type: DataType,
}

impl NamedDataType {
    /// Create a new [`NamedDataType`].
    #[must_use]
    pub fn new(name: String, data_type: DataType) -> Self {
        Self { name, data_type }
    }

    /// Create a new [`NamedDataType`] with the default name for the data type.
    #[must_use]
    pub fn new_default_name(data_type: DataType) -> Self {
        let name = data_type.default_name().into_owned();
        Self { name, data_type }
    }

    /// The name of the data type.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The underlying data type extension.
    #[must_use]
    pub const fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// Wrap this data type in an [`Optional`](DataType::Optional).
    ///
    /// Can be chained to create nested optional types.
    ///
    /// # Examples
    /// ```
    /// # use zarrs::array::{DataType, data_type::OptionalDataType};
    /// // Single level optional
    /// let opt_u8 = DataType::UInt8.into_optional();
    /// # assert_eq!(opt_u8, DataType::Optional(OptionalDataType::new(DataType::UInt8.into_named())));
    ///
    /// // Nested optional
    /// let opt_opt_u8 = DataType::UInt8.into_optional().into_optional();
    /// ```
    #[must_use]
    pub fn into_optional(self) -> Self {
        let data_type = DataType::Optional(OptionalDataType::new(self));
        let name = data_type.default_name().into_owned();
        Self::new(name, data_type)
    }

    /// Create the data type metadata.
    #[must_use]
    pub fn metadata(&self) -> MetadataV3 {
        let configuration = self.data_type.configuration();
        if configuration.is_empty() {
            MetadataV3::new(self.name.clone())
        } else {
            MetadataV3::new_with_configuration(self.name.clone(), configuration)
        }
    }

    /// Create a fill value from metadata.
    ///
    /// # Errors
    /// Returns [`DataTypeFillValueMetadataError`] if the fill value is incompatible with the data type.
    #[allow(clippy::too_many_lines)]
    pub fn fill_value_from_metadata(
        &self,
        fill_value: &FillValueMetadataV3,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        use FillValue as FV;
        let name = self.name();
        let err0 = || DataTypeFillValueMetadataError::new(name.to_string(), fill_value.clone());
        let err = |_| DataTypeFillValueMetadataError::new(name.to_string(), fill_value.clone());
        match self.data_type() {
            DataType::Bool => Ok(FV::from(fill_value.as_bool().ok_or_else(err0)?)),
            DataType::Int2 => {
                let int = fill_value.as_i64().ok_or_else(err0)?;
                let int = i8::try_from(int).map_err(err)?;
                if (-2..2).contains(&int) {
                    Ok(FV::from(int))
                } else {
                    Err(err0())
                }
            }
            DataType::Int4 => {
                let int = fill_value.as_i64().ok_or_else(err0)?;
                let int = i8::try_from(int).map_err(err)?;
                if (-8..8).contains(&int) {
                    Ok(FV::from(int))
                } else {
                    Err(err0())
                }
            }
            DataType::Int8 => {
                let int = fill_value.as_i64().ok_or_else(err0)?;
                let int = i8::try_from(int).map_err(err)?;
                Ok(FV::from(int))
            }
            DataType::Int16 => {
                let int = fill_value.as_i64().ok_or_else(err0)?;
                let int = i16::try_from(int).map_err(err)?;
                Ok(FV::from(int))
            }
            DataType::Int32 => {
                let int = fill_value.as_i64().ok_or_else(err0)?;
                let int = i32::try_from(int).map_err(err)?;
                Ok(FV::from(int))
            }
            DataType::Int64 => {
                let int = fill_value.as_i64().ok_or_else(err0)?;
                Ok(FV::from(int))
            }
            DataType::UInt2 => {
                let int = fill_value.as_u64().ok_or_else(err0)?;
                let int = u8::try_from(int).map_err(err)?;
                if (0..4).contains(&int) {
                    Ok(FV::from(int))
                } else {
                    Err(err0())
                }
            }
            DataType::UInt4 => {
                let int = fill_value.as_u64().ok_or_else(err0)?;
                let int = u8::try_from(int).map_err(err)?;
                if (0..16).contains(&int) {
                    Ok(FV::from(int))
                } else {
                    Err(err0())
                }
            }
            DataType::UInt8 => {
                let int = fill_value.as_u64().ok_or_else(err0)?;
                let int = u8::try_from(int).map_err(err)?;
                Ok(FV::from(int))
            }
            DataType::UInt16 => {
                let int = fill_value.as_u64().ok_or_else(err0)?;
                let int = u16::try_from(int).map_err(err)?;
                Ok(FV::from(int))
            }
            DataType::UInt32 => {
                let int = fill_value.as_u64().ok_or_else(err0)?;
                let int = u32::try_from(int).map_err(err)?;
                Ok(FV::from(int))
            }
            DataType::UInt64 => {
                let int = fill_value.as_u64().ok_or_else(err0)?;
                Ok(FV::from(int))
            }
            DataType::Float8E4M3 => {
                #[cfg(feature = "float8")]
                {
                    subfloat_hex_string_to_fill_value(fill_value)
                        .or_else(|| {
                            let number = float8::F8E4M3::from_f64(fill_value.as_f64()?);
                            Some(FV::from(number.to_bits()))
                        })
                        .ok_or_else(err0)
                }
                #[cfg(not(feature = "float8"))]
                subfloat_hex_string_to_fill_value(fill_value).ok_or_else(err0)
            }
            DataType::Float8E5M2 => {
                #[cfg(feature = "float8")]
                {
                    subfloat_hex_string_to_fill_value(fill_value)
                        .or_else(|| {
                            let number = float8::F8E5M2::from_f64(fill_value.as_f64()?);
                            Some(FV::from(number.to_bits()))
                        })
                        .ok_or_else(err0)
                }
                #[cfg(not(feature = "float8"))]
                subfloat_hex_string_to_fill_value(fill_value).ok_or_else(err0)
            }
            DataType::Float4E2M1FN
            | DataType::Float6E2M3FN
            | DataType::Float6E3M2FN
            | DataType::Float8E3M4
            | DataType::Float8E4M3B11FNUZ
            | DataType::Float8E4M3FNUZ
            | DataType::Float8E5M2FNUZ
            | DataType::Float8E8M0FNU => {
                // FIXME: Support normal floating point fill value metadata for these data types.
                subfloat_hex_string_to_fill_value(fill_value).ok_or_else(err0)
            }
            DataType::BFloat16 => Ok(FV::from(fill_value.as_bf16().ok_or_else(err0)?)),
            DataType::Float16 => Ok(FV::from(fill_value.as_f16().ok_or_else(err0)?)),
            DataType::Float32 => Ok(FV::from(fill_value.as_f32().ok_or_else(err0)?)),
            DataType::Float64 => Ok(FV::from(fill_value.as_f64().ok_or_else(err0)?)),
            DataType::ComplexBFloat16 => {
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = re.as_bf16().ok_or_else(err0)?;
                    let im = im.as_bf16().ok_or_else(err0)?;
                    Ok(FV::from(num::complex::Complex::<half::bf16>::new(re, im)))
                } else {
                    Err(err0())?
                }
            }
            DataType::ComplexFloat16 => {
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = re.as_f16().ok_or_else(err0)?;
                    let im = im.as_f16().ok_or_else(err0)?;
                    Ok(FV::from(num::complex::Complex::<half::f16>::new(re, im)))
                } else {
                    Err(err0())?
                }
            }
            DataType::Complex64 | DataType::ComplexFloat32 => {
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = re.as_f32().ok_or_else(err0)?;
                    let im = im.as_f32().ok_or_else(err0)?;
                    Ok(FV::from(num::complex::Complex32::new(re, im)))
                } else {
                    Err(err0())?
                }
            }
            DataType::Complex128 | DataType::ComplexFloat64 => {
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = re.as_f64().ok_or_else(err0)?;
                    let im = im.as_f64().ok_or_else(err0)?;
                    Ok(FV::from(num::complex::Complex64::new(re, im)))
                } else {
                    Err(err0())?
                }
            }
            DataType::ComplexFloat8E4M3 => {
                #[cfg(feature = "float8")]
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = subfloat_hex_string_to_fill_value(re)
                        .or_else(|| {
                            let number = float8::F8E4M3::from_f64(re.as_f64()?);
                            Some(FV::from(number.to_bits()))
                        })
                        .ok_or_else(err0)?;
                    let im = subfloat_hex_string_to_fill_value(im)
                        .or_else(|| {
                            let number = float8::F8E4M3::from_f64(im.as_f64()?);
                            Some(FV::from(number.to_bits()))
                        })
                        .ok_or_else(err0)?;
                    Ok(FV::from(num::complex::Complex::new(re, im)))
                } else {
                    Err(err0())?
                }
                #[cfg(not(feature = "float8"))]
                complex_subfloat_hex_string_to_fill_value(fill_value).ok_or_else(err0)
            }
            DataType::ComplexFloat8E5M2 => {
                #[cfg(feature = "float8")]
                if let [re, im] = fill_value.as_array().ok_or_else(err0)? {
                    let re = subfloat_hex_string_to_fill_value(re)
                        .or_else(|| {
                            let number = float8::F8E5M2::from_f64(re.as_f64()?);
                            Some(FV::from(number.to_bits()))
                        })
                        .ok_or_else(err0)?;
                    let im = subfloat_hex_string_to_fill_value(im)
                        .or_else(|| {
                            let number = float8::F8E5M2::from_f64(im.as_f64()?);
                            Some(FV::from(number.to_bits()))
                        })
                        .ok_or_else(err0)?;
                    Ok(FV::from(num::complex::Complex::new(re, im)))
                } else {
                    Err(err0())?
                }
                #[cfg(not(feature = "float8"))]
                complex_subfloat_hex_string_to_fill_value(fill_value).ok_or_else(err0)
            }
            DataType::ComplexFloat4E2M1FN
            | DataType::ComplexFloat6E2M3FN
            | DataType::ComplexFloat6E3M2FN
            | DataType::ComplexFloat8E3M4
            | DataType::ComplexFloat8E4M3B11FNUZ
            | DataType::ComplexFloat8E4M3FNUZ
            | DataType::ComplexFloat8E5M2FNUZ
            | DataType::ComplexFloat8E8M0FNU => {
                // FIXME: Support normal floating point fill value metadata for these data types.
                complex_subfloat_hex_string_to_fill_value(fill_value).ok_or_else(err0)
            }
            DataType::RawBits(size) => {
                let bytes = fill_value.as_bytes().ok_or_else(err0)?;
                if bytes.len() == *size {
                    Ok(FV::from(bytes))
                } else {
                    Err(err0())?
                }
            }
            DataType::Bytes => {
                if let Some(bytes) = fill_value.as_bytes() {
                    // Permit bytes for any data type alias of `bytes`
                    Ok(FV::from(bytes))
                } else if let Some(string) = fill_value.as_str() {
                    Ok(FV::from(
                        BASE64_STANDARD.decode(string).map_err(|_| err0())?,
                    ))
                } else {
                    Err(err0())?
                }
            }
            DataType::String => Ok(FV::from(fill_value.as_str().ok_or_else(err0)?)),
            DataType::NumpyDateTime64 {
                unit: _,
                scale_factor: _,
            }
            | DataType::NumpyTimeDelta64 {
                unit: _,
                scale_factor: _,
            } => {
                if let Some("NaT") = fill_value.as_str() {
                    Ok(FV::from(i64::MIN))
                } else if let Some(i) = fill_value.as_i64() {
                    Ok(FV::from(i))
                } else {
                    Err(err0())?
                }
            }
            DataType::Optional(opt) => {
                if fill_value.is_null() {
                    // Null fill value for optional type: single 0x00 byte
                    Ok(FillValue::new_optional_null())
                } else if let Some([inner]) = fill_value.as_array() {
                    // Wrapped value [inner_metadata] -> Some(inner)
                    let inner_fv = opt.fill_value_from_metadata(inner)?;
                    Ok(inner_fv.into_optional())
                } else {
                    // Invalid format for optional
                    Err(err0())
                }
            }
            DataType::Extension(ext) => ext.fill_value(fill_value).map_err(|_| {
                DataTypeFillValueMetadataError::new(name.to_string(), fill_value.clone())
            }),
        }
    }
}

impl Deref for NamedDataType {
    type Target = DataType;

    fn deref(&self) -> &Self::Target {
        &self.data_type
    }
}

impl From<NamedDataType> for DataType {
    fn from(value: NamedDataType) -> Self {
        value.data_type
    }
}

impl TryFrom<&MetadataV3> for NamedDataType {
    type Error = PluginCreateError;

    /// Create a [`NamedDataType`] from metadata.
    ///
    /// # Errors
    ///
    /// Returns [`PluginCreateError`] if the metadata is invalid or not associated with a registered data type plugin.
    #[allow(clippy::too_many_lines)]
    fn try_from(metadata: &MetadataV3) -> Result<Self, Self::Error> {
        if !metadata.must_understand() {
            return Err(PluginCreateError::Other(
                r#"data type must not have `"must_understand": false`"#.to_string(),
            ));
        }

        let name = metadata.name();

        // Handle data types with configuration
        if let Some(configuration) = metadata.configuration() {
            if NumpyDateTime64DataType::matches_name(name, ZarrVersions::V3) {
                use crate::metadata_ext::data_type::numpy_datetime64::NumpyDateTime64DataTypeConfigurationV1;
                let NumpyDateTime64DataTypeConfigurationV1 { unit, scale_factor } =
                    NumpyDateTime64DataTypeConfigurationV1::try_from_configuration(
                        configuration.clone(),
                    )
                    .map_err(|_| {
                        PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                            NumpyDateTime64DataType::IDENTIFIER,
                            "data_type",
                            metadata.to_string(),
                        ))
                    })?;
                return Ok(Self::new(
                    name.to_string(),
                    DataType::NumpyDateTime64 { unit, scale_factor },
                ));
            }

            if NumpyTimeDelta64DataType::matches_name(name, ZarrVersions::V3) {
                use crate::metadata_ext::data_type::numpy_timedelta64::NumpyTimeDelta64DataTypeConfigurationV1;
                let NumpyTimeDelta64DataTypeConfigurationV1 { unit, scale_factor } =
                    NumpyTimeDelta64DataTypeConfigurationV1::try_from_configuration(
                        configuration.clone(),
                    )
                    .map_err(|_| {
                        PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                            NumpyTimeDelta64DataType::IDENTIFIER,
                            "data_type",
                            metadata.to_string(),
                        ))
                    })?;
                return Ok(Self::new(
                    name.to_string(),
                    DataType::NumpyTimeDelta64 { unit, scale_factor },
                ));
            }

            if OptionalDataType::matches_name(name, ZarrVersions::V3) {
                use crate::metadata_ext::data_type::optional::OptionalDataTypeConfigurationV1;
                let OptionalDataTypeConfigurationV1 {
                    name: inner_name,
                    configuration,
                } = OptionalDataTypeConfigurationV1::try_from_configuration(configuration.clone())
                    .map_err(|_| {
                        PluginCreateError::MetadataInvalid(PluginMetadataInvalidError::new(
                            OptionalDataType::IDENTIFIER,
                            "data_type",
                            metadata.to_string(),
                        ))
                    })?;

                // Create metadata for the inner data type
                let inner_metadata = if configuration.is_empty() {
                    MetadataV3::new(inner_name)
                } else {
                    MetadataV3::new_with_configuration(inner_name, configuration)
                };

                // Recursively parse the inner data type
                let inner_data_type = Self::try_from(&inner_metadata)?;
                let data_type = DataType::Optional(OptionalDataType::new(inner_data_type));
                return Ok(Self::new(name.to_string(), data_type));
            }
        }

        // Handle data types with no configuration
        if metadata.configuration_is_none_or_empty() {
            // Boolean
            if BoolDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Bool));
            }

            // Signed integers
            if Int2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Int2));
            }
            if Int4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Int4));
            }
            if Int8DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Int8));
            }
            if Int16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Int16));
            }
            if Int32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Int32));
            }
            if Int64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Int64));
            }

            // Unsigned integers
            if UInt2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::UInt2));
            }
            if UInt4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::UInt4));
            }
            if UInt8DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::UInt8));
            }
            if UInt16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::UInt16));
            }
            if UInt32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::UInt32));
            }
            if UInt64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::UInt64));
            }

            // Subfloats
            if Float4E2M1FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float4E2M1FN));
            }
            if Float6E2M3FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float6E2M3FN));
            }
            if Float6E3M2FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float6E3M2FN));
            }
            if Float8E3M4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E3M4));
            }
            if Float8E4M3DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E4M3));
            }
            if Float8E4M3B11FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E4M3B11FNUZ));
            }
            if Float8E4M3FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E4M3FNUZ));
            }
            if Float8E5M2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E5M2));
            }
            if Float8E5M2FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E5M2FNUZ));
            }
            if Float8E8M0FNUDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float8E8M0FNU));
            }

            // Standard floats
            if BFloat16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::BFloat16));
            }
            if Float16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float16));
            }
            if Float32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float32));
            }
            if Float64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Float64));
            }

            // Complex subfloats
            if ComplexFloat4E2M1FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat4E2M1FN));
            }
            if ComplexFloat6E2M3FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat6E2M3FN));
            }
            if ComplexFloat6E3M2FNDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat6E3M2FN));
            }
            if ComplexFloat8E3M4DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat8E3M4));
            }
            if ComplexFloat8E4M3DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat8E4M3));
            }
            if ComplexFloat8E4M3B11FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(
                    name.to_string(),
                    DataType::ComplexFloat8E4M3B11FNUZ,
                ));
            }
            if ComplexFloat8E4M3FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat8E4M3FNUZ));
            }
            if ComplexFloat8E5M2DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat8E5M2));
            }
            if ComplexFloat8E5M2FNUZDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat8E5M2FNUZ));
            }
            if ComplexFloat8E8M0FNUDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat8E8M0FNU));
            }

            // Complex floats
            if ComplexBFloat16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexBFloat16));
            }
            if ComplexFloat16DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat16));
            }
            if ComplexFloat32DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat32));
            }
            if ComplexFloat64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::ComplexFloat64));
            }
            if Complex64DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Complex64));
            }
            if Complex128DataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Complex128));
            }

            // Variable-length types
            if StringDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::String));
            }
            if BytesDataType::matches_name(name, ZarrVersions::V3) {
                return Ok(Self::new(name.to_string(), DataType::Bytes));
            }

            // RawBits (r8, r16, r24, etc.)
            if RawBitsDataType::matches_name(name, ZarrVersions::V3) {
                if let Ok(size_bits) = name[1..].parse::<usize>() {
                    if size_bits % 8 == 0 {
                        let size_bytes = size_bits / 8;
                        return Ok(Self::new(name.to_string(), DataType::RawBits(size_bytes)));
                    }
                    return Err(PluginUnsupportedError::new(
                        name.to_string(),
                        "data type".to_string(),
                    )
                    .into());
                }
            }
        }

        // Try an extension plugin
        for plugin in inventory::iter::<DataTypePlugin> {
            if plugin.match_name(name, ZarrVersions::V3) {
                return plugin.create(metadata).map(|dt| {
                    NamedDataType::new(metadata.name().to_string(), DataType::Extension(dt))
                });
            }
        }

        // The data type is not supported
        Err(PluginUnsupportedError::new(name.to_string(), "data type".to_string()).into())
    }
}
