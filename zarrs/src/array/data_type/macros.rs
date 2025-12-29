//! Macros for implementing [`DataTypeExtension`](zarrs_data_type::DataTypeExtension) for data type markers.

/// Helper macro to implement `DataTypeExtension` for simple fixed-size numeric types.
///
/// Usage:
/// - `impl_data_type_extension_numeric!(MarkerType, size, rust_type)` - basic implementation
/// - `impl_data_type_extension_numeric!(MarkerType, size, rust_type; codec_method1, codec_method2, ...)` - with codec overrides
///
/// Available codec methods: `pcodec`, `zfp`, `bitround`, `fixedscaleoffset`, `packbits`
macro_rules! impl_data_type_extension_numeric {
    // Base case: no additional codec methods
    ($marker:ty, $size:tt, $rust_type:tt) => {
        impl_data_type_extension_numeric!($marker, $size, $rust_type;);
    };

    // With optional codec method overrides
    ($marker:ty, $size:tt, $rust_type:tt; $($codec:ident),* $(,)?) => {
        impl zarrs_data_type::DataTypeExtension for $marker {
            fn identifier(&self) -> &'static str {
                <$marker as zarrs_plugin::ExtensionIdentifier>::IDENTIFIER
            }

            fn configuration(&self) -> zarrs_metadata::Configuration {
                zarrs_metadata::Configuration::default()
            }

            fn size(&self) -> zarrs_metadata::DataTypeSize {
                zarrs_metadata::DataTypeSize::Fixed($size)
            }

            fn fill_value(
                &self,
                fill_value_metadata: &zarrs_metadata::v3::FillValueMetadataV3,
            ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
                impl_data_type_extension_numeric!(@fill_value self, fill_value_metadata, $rust_type)
            }

            fn metadata_fill_value(
                &self,
                fill_value: &zarrs_data_type::FillValue,
            ) -> Result<zarrs_metadata::v3::FillValueMetadataV3, zarrs_data_type::DataTypeFillValueError> {
                impl_data_type_extension_numeric!(@metadata_fill_value self, fill_value, $rust_type, $size)
            }

            fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
                Some(self)
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            $(
                impl_data_type_extension_numeric!(@codec_method $codec);
            )*
        }

        #[allow(unused_variables)]
        impl zarrs_data_type::DataTypeExtensionBytesCodec for $marker {
            fn encode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
                impl_data_type_extension_numeric!(@bytes_codec bytes, endianness, $rust_type, $size)
            }

            fn decode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
                impl_data_type_extension_numeric!(@bytes_codec bytes, endianness, $rust_type, $size)
            }
        }
    };

    // Codec method overrides
    (@codec_method pcodec) => {
        fn codec_pcodec(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionPcodecCodec> {
            Some(self)
        }
    };
    (@codec_method zfp) => {
        fn codec_zfp(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionZfpCodec> {
            Some(self)
        }
    };
    (@codec_method bitround) => {
        fn codec_bitround(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBitroundCodec> {
            Some(self)
        }
    };
    (@codec_method fixedscaleoffset) => {
        fn codec_fixedscaleoffset(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionFixedScaleOffsetCodec> {
            Some(self)
        }
    };
    (@codec_method packbits) => {
        fn codec_packbits(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionPackBitsCodec> {
            Some(self)
        }
    };

    // Fill value from metadata for signed integers
    (@fill_value $self:ident, $fill_value_metadata:ident, i8) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError::new(
            $self.identifier().to_string(),
            $fill_value_metadata.clone(),
        );
        let int = $fill_value_metadata.as_i64().ok_or_else(err)?;
        let int = i8::try_from(int).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, i16) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError::new(
            $self.identifier().to_string(),
            $fill_value_metadata.clone(),
        );
        let int = $fill_value_metadata.as_i64().ok_or_else(err)?;
        let int = i16::try_from(int).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, i32) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError::new(
            $self.identifier().to_string(),
            $fill_value_metadata.clone(),
        );
        let int = $fill_value_metadata.as_i64().ok_or_else(err)?;
        let int = i32::try_from(int).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, i64) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError::new(
            $self.identifier().to_string(),
            $fill_value_metadata.clone(),
        );
        let int = $fill_value_metadata.as_i64().ok_or_else(err)?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    // Fill value from metadata for unsigned integers
    (@fill_value $self:ident, $fill_value_metadata:ident, u8) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError::new(
            $self.identifier().to_string(),
            $fill_value_metadata.clone(),
        );
        let int = $fill_value_metadata.as_u64().ok_or_else(err)?;
        let int = u8::try_from(int).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, u16) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError::new(
            $self.identifier().to_string(),
            $fill_value_metadata.clone(),
        );
        let int = $fill_value_metadata.as_u64().ok_or_else(err)?;
        let int = u16::try_from(int).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, u32) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError::new(
            $self.identifier().to_string(),
            $fill_value_metadata.clone(),
        );
        let int = $fill_value_metadata.as_u64().ok_or_else(err)?;
        let int = u32::try_from(int).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, u64) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError::new(
            $self.identifier().to_string(),
            $fill_value_metadata.clone(),
        );
        let int = $fill_value_metadata.as_u64().ok_or_else(err)?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    // Fill value from metadata for floats
    (@fill_value $self:ident, $fill_value_metadata:ident, f16) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError::new(
            $self.identifier().to_string(),
            $fill_value_metadata.clone(),
        );
        Ok(zarrs_data_type::FillValue::from($fill_value_metadata.as_f16().ok_or_else(err)?))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, bf16) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError::new(
            $self.identifier().to_string(),
            $fill_value_metadata.clone(),
        );
        Ok(zarrs_data_type::FillValue::from($fill_value_metadata.as_bf16().ok_or_else(err)?))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, f32) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError::new(
            $self.identifier().to_string(),
            $fill_value_metadata.clone(),
        );
        Ok(zarrs_data_type::FillValue::from($fill_value_metadata.as_f32().ok_or_else(err)?))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, f64) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError::new(
            $self.identifier().to_string(),
            $fill_value_metadata.clone(),
        );
        Ok(zarrs_data_type::FillValue::from($fill_value_metadata.as_f64().ok_or_else(err)?))
    }};

    // Metadata fill value for specific types
    (@metadata_fill_value $self:ident, $fill_value:ident, i8, 1) => {{
        let error = || zarrs_data_type::DataTypeFillValueError::new($self.identifier().to_string(), $fill_value.clone());
        let bytes: [u8; 1] = $fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = i8::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, i16, 2) => {{
        let error = || zarrs_data_type::DataTypeFillValueError::new($self.identifier().to_string(), $fill_value.clone());
        let bytes: [u8; 2] = $fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = i16::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, i32, 4) => {{
        let error = || zarrs_data_type::DataTypeFillValueError::new($self.identifier().to_string(), $fill_value.clone());
        let bytes: [u8; 4] = $fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = i32::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, i64, 8) => {{
        let error = || zarrs_data_type::DataTypeFillValueError::new($self.identifier().to_string(), $fill_value.clone());
        let bytes: [u8; 8] = $fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = i64::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, u8, 1) => {{
        let error = || zarrs_data_type::DataTypeFillValueError::new($self.identifier().to_string(), $fill_value.clone());
        let bytes: [u8; 1] = $fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = u8::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, u16, 2) => {{
        let error = || zarrs_data_type::DataTypeFillValueError::new($self.identifier().to_string(), $fill_value.clone());
        let bytes: [u8; 2] = $fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = u16::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, u32, 4) => {{
        let error = || zarrs_data_type::DataTypeFillValueError::new($self.identifier().to_string(), $fill_value.clone());
        let bytes: [u8; 4] = $fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = u32::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, u64, 8) => {{
        let error = || zarrs_data_type::DataTypeFillValueError::new($self.identifier().to_string(), $fill_value.clone());
        let bytes: [u8; 8] = $fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = u64::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, f16, 2) => {{
        let error = || zarrs_data_type::DataTypeFillValueError::new($self.identifier().to_string(), $fill_value.clone());
        let bytes: [u8; 2] = $fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = half::f16::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, bf16, 2) => {{
        let error = || zarrs_data_type::DataTypeFillValueError::new($self.identifier().to_string(), $fill_value.clone());
        let bytes: [u8; 2] = $fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = half::bf16::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, f32, 4) => {{
        let error = || zarrs_data_type::DataTypeFillValueError::new($self.identifier().to_string(), $fill_value.clone());
        let bytes: [u8; 4] = $fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = f32::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, f64, 8) => {{
        let error = || zarrs_data_type::DataTypeFillValueError::new($self.identifier().to_string(), $fill_value.clone());
        let bytes: [u8; 8] = $fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
        let number = f64::from_ne_bytes(bytes);
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(number))
    }};

    // Encode/decode for single-byte types (passthrough)
    (@bytes_codec $bytes:ident, $_endianness:ident, $rust_type:tt, 1) => {{
        Ok($bytes)
    }};
    // Encode/decode for multi-byte types (endianness swap if needed)
    (@bytes_codec $bytes:ident, $endianness:ident, $rust_type:tt, $size:tt) => {{
        let endianness = $endianness.ok_or(zarrs_data_type::DataTypeExtensionBytesCodecError::EndiannessNotSpecified)?;
        if endianness == zarrs_metadata::Endianness::native() {
            Ok($bytes)
        } else {
            // Swap endianness
            let mut result = $bytes.into_owned();
            for chunk in result.as_chunks_mut::<$size>().0 {
                chunk.reverse();
            }
            Ok(std::borrow::Cow::Owned(result))
        }
    }};
}

pub(crate) use impl_data_type_extension_numeric;

/// Macro to register a data type as a `DataTypePlugin`.
macro_rules! register_data_type_plugin {
    ($marker:ident) => {
        inventory::submit! {
            zarrs_data_type::DataTypePlugin::new(
                <$marker as zarrs_plugin::ExtensionIdentifier>::IDENTIFIER,
                <$marker as zarrs_plugin::ExtensionIdentifier>::matches_name,
                <$marker as zarrs_plugin::ExtensionIdentifier>::default_name,
                |_metadata: &zarrs_metadata::v3::MetadataV3| -> Result<std::sync::Arc<dyn zarrs_data_type::DataTypeExtension>, zarrs_plugin::PluginCreateError> {
                    Ok(std::sync::Arc::new($marker))
                },
            )
        }
    };
}

pub(crate) use register_data_type_plugin;

/// Macro to implement `DataTypeExtensionPackBitsCodec` for data types.
///
/// # Usage
/// ```ignore
/// // For single-component types:
/// impl_packbits_codec!(Int32DataType, 32, signed, 1);
/// impl_packbits_codec!(UInt32DataType, 32, unsigned, 1);
/// impl_packbits_codec!(Float32DataType, 32, float, 1);
///
/// // For complex types (2 components):
/// impl_packbits_codec!(Complex64DataType, 32, float, 2);
/// ```
macro_rules! impl_packbits_codec {
    // Multi-component, signed integer
    ($marker:ty, $bits:expr, signed, $components:expr) => {
        impl zarrs_data_type::DataTypeExtensionPackBitsCodec for $marker {
            fn component_size_bits(&self) -> u64 {
                $bits
            }
            fn num_components(&self) -> u64 {
                $components
            }
            fn sign_extension(&self) -> bool {
                true
            }
        }
    };
    // Multi-component, unsigned integer
    ($marker:ty, $bits:expr, unsigned, $components:expr) => {
        impl zarrs_data_type::DataTypeExtensionPackBitsCodec for $marker {
            fn component_size_bits(&self) -> u64 {
                $bits
            }
            fn num_components(&self) -> u64 {
                $components
            }
            fn sign_extension(&self) -> bool {
                false
            }
        }
    };
    // Multi-component, float (no sign extension)
    ($marker:ty, $bits:expr, float, $components:expr) => {
        impl zarrs_data_type::DataTypeExtensionPackBitsCodec for $marker {
            fn component_size_bits(&self) -> u64 {
                $bits
            }
            fn num_components(&self) -> u64 {
                $components
            }
            fn sign_extension(&self) -> bool {
                false
            }
        }
    };
}

pub(crate) use impl_packbits_codec;

/// Macro to implement `DataTypeExtensionPcodecCodec` for data types.
///
/// # Usage
/// ```ignore
/// // Unsupported type:
/// impl_pcodec_codec!(BFloat16DataType, None);
///
/// // Supported type:
/// impl_pcodec_codec!(Int32DataType, I32);
/// impl_pcodec_codec!(Float32DataType, F32);
///
/// // Complex type (2 elements per element):
/// impl_pcodec_codec!(Complex64DataType, F32, 2);
/// ```
macro_rules! impl_pcodec_codec {
    // Unsupported type
    ($marker:ty, None) => {
        impl zarrs_data_type::DataTypeExtensionPcodecCodec for $marker {
            fn pcodec_element_type(&self) -> Option<zarrs_data_type::PcodecElementType> {
                None
            }
        }
    };
    // Single element type
    ($marker:ty, $element_type:ident) => {
        impl zarrs_data_type::DataTypeExtensionPcodecCodec for $marker {
            fn pcodec_element_type(&self) -> Option<zarrs_data_type::PcodecElementType> {
                Some(zarrs_data_type::PcodecElementType::$element_type)
            }
        }
    };
    // Multi-element type (e.g., complex numbers)
    ($marker:ty, $element_type:ident, $elements_per_element:expr) => {
        impl zarrs_data_type::DataTypeExtensionPcodecCodec for $marker {
            fn pcodec_element_type(&self) -> Option<zarrs_data_type::PcodecElementType> {
                Some(zarrs_data_type::PcodecElementType::$element_type)
            }
            fn pcodec_elements_per_element(&self) -> usize {
                $elements_per_element
            }
        }
    };
}

pub(crate) use impl_pcodec_codec;

/// Macro to implement `DataTypeExtensionFixedScaleOffsetCodec` for data types.
///
/// # Usage
/// ```ignore
/// // Unsupported type:
/// impl_fixedscaleoffset_codec!(BFloat16DataType, None);
///
/// // Supported type:
/// impl_fixedscaleoffset_codec!(Int32DataType, I32);
/// impl_fixedscaleoffset_codec!(Float32DataType, F32);
/// ```
macro_rules! impl_fixedscaleoffset_codec {
    // Unsupported type
    ($marker:ty, None) => {
        impl zarrs_data_type::DataTypeExtensionFixedScaleOffsetCodec for $marker {
            fn fixedscaleoffset_element_type(
                &self,
            ) -> Option<zarrs_data_type::FixedScaleOffsetElementType> {
                None
            }
        }
    };
    // Supported type
    ($marker:ty, $element_type:ident) => {
        impl zarrs_data_type::DataTypeExtensionFixedScaleOffsetCodec for $marker {
            fn fixedscaleoffset_element_type(
                &self,
            ) -> Option<zarrs_data_type::FixedScaleOffsetElementType> {
                Some(zarrs_data_type::FixedScaleOffsetElementType::$element_type)
            }
        }
    };
}

pub(crate) use impl_fixedscaleoffset_codec;

/// Macro to implement `DataTypeExtensionZfpCodec` for data types.
///
/// # Usage
/// ```ignore
/// // Unsupported type:
/// impl_zfp_codec!(BFloat16DataType, None);
///
/// // Native ZFP type (no promotion):
/// impl_zfp_codec!(Int32DataType, Int32);
/// impl_zfp_codec!(Float32DataType, Float);
///
/// // Promoted type:
/// impl_zfp_codec!(Int8DataType, Int32, I8ToI32);
/// impl_zfp_codec!(Int16DataType, Int32, I16ToI32);
/// ```
macro_rules! impl_zfp_codec {
    // Unsupported type
    ($marker:ty, None) => {
        impl zarrs_data_type::DataTypeExtensionZfpCodec for $marker {
            fn zfp_type(&self) -> Option<zarrs_data_type::ZfpType> {
                None
            }
        }
    };
    // Native type (no promotion needed)
    ($marker:ty, $zfp_type:ident) => {
        impl zarrs_data_type::DataTypeExtensionZfpCodec for $marker {
            fn zfp_type(&self) -> Option<zarrs_data_type::ZfpType> {
                Some(zarrs_data_type::ZfpType::$zfp_type)
            }
            fn zfp_promotion(&self) -> zarrs_data_type::ZfpPromotion {
                zarrs_data_type::ZfpPromotion::None
            }
        }
    };
    // Promoted type
    ($marker:ty, $zfp_type:ident, $promotion:ident) => {
        impl zarrs_data_type::DataTypeExtensionZfpCodec for $marker {
            fn zfp_type(&self) -> Option<zarrs_data_type::ZfpType> {
                Some(zarrs_data_type::ZfpType::$zfp_type)
            }
            fn zfp_promotion(&self) -> zarrs_data_type::ZfpPromotion {
                zarrs_data_type::ZfpPromotion::$promotion
            }
        }
    };
}

pub(crate) use impl_zfp_codec;

/// Macro to implement `DataTypeExtensionBitroundCodec` for data types.
///
/// # Usage
/// ```ignore
/// // Float types (have mantissa bits):
/// impl_bitround_codec!(Float32DataType, 4, float32, 23);
/// impl_bitround_codec!(Float64DataType, 8, float64, 52);
/// impl_bitround_codec!(Float16DataType, 2, float16, 10);
/// impl_bitround_codec!(BFloat16DataType, 2, float16, 7);
///
/// // Integer types (no mantissa bits):
/// impl_bitround_codec!(Int32DataType, 4, int32);
/// impl_bitround_codec!(Int64DataType, 8, int64);
/// ```
macro_rules! impl_bitround_codec {
    // Float16/BFloat16 types (use round_bytes_float16 with specified mantissa bits)
    ($marker:ty, 2, float16, $mantissa_bits:expr) => {
        impl zarrs_data_type::DataTypeExtensionBitroundCodec for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                Some($mantissa_bits)
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                zarrs_data_type::round_bytes_float16(bytes, keepbits, $mantissa_bits);
            }
        }
    };
    // Float32 types
    ($marker:ty, 4, float32, $mantissa_bits:expr) => {
        impl zarrs_data_type::DataTypeExtensionBitroundCodec for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                Some($mantissa_bits)
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                zarrs_data_type::round_bytes_float32(bytes, keepbits, $mantissa_bits);
            }
        }
    };
    // Float64 types
    ($marker:ty, 8, float64, $mantissa_bits:expr) => {
        impl zarrs_data_type::DataTypeExtensionBitroundCodec for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                Some($mantissa_bits)
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                zarrs_data_type::round_bytes_float64(bytes, keepbits, $mantissa_bits);
            }
        }
    };
    // Int8 types (no mantissa, round from MSB)
    ($marker:ty, 1, int8) => {
        impl zarrs_data_type::DataTypeExtensionBitroundCodec for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                zarrs_data_type::round_bytes_int8(bytes, keepbits);
            }
        }
    };
    // Int16 types
    ($marker:ty, 2, int16) => {
        impl zarrs_data_type::DataTypeExtensionBitroundCodec for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                zarrs_data_type::round_bytes_int16(bytes, keepbits);
            }
        }
    };
    // Int32 types
    ($marker:ty, 4, int32) => {
        impl zarrs_data_type::DataTypeExtensionBitroundCodec for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                zarrs_data_type::round_bytes_int32(bytes, keepbits);
            }
        }
    };
    // Int64 types
    ($marker:ty, 8, int64) => {
        impl zarrs_data_type::DataTypeExtensionBitroundCodec for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                zarrs_data_type::round_bytes_int64(bytes, keepbits);
            }
        }
    };
    // UInt8 types (use int8 rounding function)
    ($marker:ty, 1, uint8) => {
        impl zarrs_data_type::DataTypeExtensionBitroundCodec for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                zarrs_data_type::round_bytes_int8(bytes, keepbits);
            }
        }
    };
    // UInt16 types (use int16 rounding function)
    ($marker:ty, 2, uint16) => {
        impl zarrs_data_type::DataTypeExtensionBitroundCodec for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                zarrs_data_type::round_bytes_int16(bytes, keepbits);
            }
        }
    };
    // UInt32 types (use int32 rounding function)
    ($marker:ty, 4, uint32) => {
        impl zarrs_data_type::DataTypeExtensionBitroundCodec for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                zarrs_data_type::round_bytes_int32(bytes, keepbits);
            }
        }
    };
    // UInt64 types (use int64 rounding function)
    ($marker:ty, 8, uint64) => {
        impl zarrs_data_type::DataTypeExtensionBitroundCodec for $marker {
            fn mantissa_bits(&self) -> Option<u32> {
                None
            }
            fn round(&self, bytes: &mut [u8], keepbits: u32) {
                zarrs_data_type::round_bytes_int64(bytes, keepbits);
            }
        }
    };
}

pub(crate) use impl_bitround_codec;

/// Macro to implement a passthrough `DataTypeExtensionBytesCodec` for data types.
///
/// This is useful for single-byte types and other types where no byte-swapping
/// or transformation is needed during encoding/decoding.
///
/// # Usage
/// ```ignore
/// impl_bytes_codec_passthrough!(BoolDataType);
/// impl_bytes_codec_passthrough!(UInt4DataType);
/// impl_bytes_codec_passthrough!(Float8E4M3DataType);
/// ```
macro_rules! impl_bytes_codec_passthrough {
    ($marker:ty) => {
        impl zarrs_data_type::DataTypeExtensionBytesCodec for $marker {
            fn encode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                _endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<
                std::borrow::Cow<'a, [u8]>,
                zarrs_data_type::DataTypeExtensionBytesCodecError,
            > {
                Ok(bytes)
            }

            fn decode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                _endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<
                std::borrow::Cow<'a, [u8]>,
                zarrs_data_type::DataTypeExtensionBytesCodecError,
            > {
                Ok(bytes)
            }
        }
    };
}

pub(crate) use impl_bytes_codec_passthrough;
