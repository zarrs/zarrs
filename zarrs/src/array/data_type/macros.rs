//! Macros for implementing [`DataTypeExtension`](zarrs_data_type::DataTypeExtension) for data type markers.

/// Helper macro to implement `DataTypeExtension` for simple fixed-size numeric types.
macro_rules! impl_data_type_extension_numeric {
    ($marker:ty, $size:tt, $rust_type:tt) => {
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
        }

        #[allow(unused_variables)]
        impl zarrs_data_type::DataTypeExtensionBytesCodec for $marker {
            fn encode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
                impl_data_type_extension_numeric!(@encode bytes, endianness, $rust_type, $size)
            }

            fn decode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
                impl_data_type_extension_numeric!(@decode bytes, endianness, $rust_type, $size)
            }
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

    // Encode for single-byte types
    (@encode $bytes:ident, $_endianness:ident, $rust_type:tt, 1) => {{
        Ok($bytes)
    }};
    // Encode for multi-byte types
    (@encode $bytes:ident, $endianness:ident, $rust_type:tt, $size:tt) => {{
        let endianness = $endianness.ok_or(zarrs_data_type::DataTypeExtensionBytesCodecError::EndiannessNotSpecified)?;
        if endianness == zarrs_metadata::Endianness::native() {
            Ok($bytes)
        } else {
            // Swap endianness
            let mut result = $bytes.into_owned();
            for chunk in result.chunks_exact_mut($size) {
                chunk.reverse();
            }
            Ok(std::borrow::Cow::Owned(result))
        }
    }};

    // Decode for single-byte types
    (@decode $bytes:ident, $_endianness:ident, $rust_type:tt, 1) => {{
        Ok($bytes)
    }};
    // Decode for multi-byte types
    (@decode $bytes:ident, $endianness:ident, $rust_type:tt, $size:tt) => {{
        let endianness = $endianness.ok_or(zarrs_data_type::DataTypeExtensionBytesCodecError::EndiannessNotSpecified)?;
        if endianness == zarrs_metadata::Endianness::native() {
            Ok($bytes)
        } else {
            // Swap endianness
            let mut result = $bytes.into_owned();
            for chunk in result.chunks_exact_mut($size) {
                chunk.reverse();
            }
            Ok(std::borrow::Cow::Owned(result))
        }
    }};
}

/// Macro to implement `DataTypeExtension` for complex types.
macro_rules! impl_complex_data_type {
    ($marker:ty, $size:tt, $component_type:tt) => {
        impl zarrs_data_type::DataTypeExtension for $marker {
            fn identifier(&self) -> &'static str {
                <Self as zarrs_plugin::ExtensionIdentifier>::IDENTIFIER
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
                let err = || {
                    zarrs_data_type::DataTypeFillValueMetadataError::new(
                        self.identifier().to_string(),
                        fill_value_metadata.clone(),
                    )
                };
                if let [re, im] = fill_value_metadata.as_array().ok_or_else(err)? {
                    impl_complex_data_type!(@parse_components self, re, im, $component_type, err)
                } else {
                    Err(err())
                }
            }

            fn metadata_fill_value(
                &self,
                fill_value: &zarrs_data_type::FillValue,
            ) -> Result<zarrs_metadata::v3::FillValueMetadataV3, zarrs_data_type::DataTypeFillValueError> {
                let error = || {
                    zarrs_data_type::DataTypeFillValueError::new(self.identifier().to_string(), fill_value.clone())
                };
                impl_complex_data_type!(@to_metadata self, fill_value, $component_type, $size, error)
            }

            fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
                Some(self)
            }
        }

        impl zarrs_data_type::DataTypeExtensionBytesCodec for $marker {
            fn encode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
                let component_size = $size / 2;
                if component_size == 1 {
                    Ok(bytes)
                } else {
                    let endianness = endianness.ok_or(zarrs_data_type::DataTypeExtensionBytesCodecError::EndiannessNotSpecified)?;
                    if endianness == zarrs_metadata::Endianness::native() {
                        Ok(bytes)
                    } else {
                        let mut result = bytes.into_owned();
                        for chunk in result.chunks_exact_mut(component_size) {
                            chunk.reverse();
                        }
                        Ok(std::borrow::Cow::Owned(result))
                    }
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
    };

    (@parse_components $self:ident, $re:ident, $im:ident, f32, $err:ident) => {{
        let re = $re.as_f32().ok_or_else($err)?;
        let im = $im.as_f32().ok_or_else($err)?;
        Ok(zarrs_data_type::FillValue::from(num::complex::Complex32::new(re, im)))
    }};
    (@parse_components $self:ident, $re:ident, $im:ident, f64, $err:ident) => {{
        let re = $re.as_f64().ok_or_else($err)?;
        let im = $im.as_f64().ok_or_else($err)?;
        Ok(zarrs_data_type::FillValue::from(num::complex::Complex64::new(re, im)))
    }};
    (@parse_components $self:ident, $re:ident, $im:ident, f16, $err:ident) => {{
        let re = $re.as_f16().ok_or_else($err)?;
        let im = $im.as_f16().ok_or_else($err)?;
        Ok(zarrs_data_type::FillValue::from(num::complex::Complex::<half::f16>::new(re, im)))
    }};
    (@parse_components $self:ident, $re:ident, $im:ident, bf16, $err:ident) => {{
        let re = $re.as_bf16().ok_or_else($err)?;
        let im = $im.as_bf16().ok_or_else($err)?;
        Ok(zarrs_data_type::FillValue::from(num::complex::Complex::<half::bf16>::new(re, im)))
    }};

    (@to_metadata $self:ident, $fill_value:ident, f32, 8, $error:ident) => {{
        let bytes: [u8; 8] = $fill_value.as_ne_bytes().try_into().map_err(|_| $error())?;
        let re = f32::from_ne_bytes(bytes[0..4].try_into().unwrap());
        let im = f32::from_ne_bytes(bytes[4..8].try_into().unwrap());
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(vec![
            zarrs_metadata::v3::FillValueMetadataV3::from(re),
            zarrs_metadata::v3::FillValueMetadataV3::from(im),
        ]))
    }};
    (@to_metadata $self:ident, $fill_value:ident, f64, 16, $error:ident) => {{
        let bytes: [u8; 16] = $fill_value.as_ne_bytes().try_into().map_err(|_| $error())?;
        let re = f64::from_ne_bytes(bytes[0..8].try_into().unwrap());
        let im = f64::from_ne_bytes(bytes[8..16].try_into().unwrap());
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(vec![
            zarrs_metadata::v3::FillValueMetadataV3::from(re),
            zarrs_metadata::v3::FillValueMetadataV3::from(im),
        ]))
    }};
    (@to_metadata $self:ident, $fill_value:ident, f16, 4, $error:ident) => {{
        let bytes: [u8; 4] = $fill_value.as_ne_bytes().try_into().map_err(|_| $error())?;
        let re = half::f16::from_ne_bytes(bytes[0..2].try_into().unwrap());
        let im = half::f16::from_ne_bytes(bytes[2..4].try_into().unwrap());
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(vec![
            zarrs_metadata::v3::FillValueMetadataV3::from(re),
            zarrs_metadata::v3::FillValueMetadataV3::from(im),
        ]))
    }};
    (@to_metadata $self:ident, $fill_value:ident, bf16, 4, $error:ident) => {{
        let bytes: [u8; 4] = $fill_value.as_ne_bytes().try_into().map_err(|_| $error())?;
        let re = half::bf16::from_ne_bytes(bytes[0..2].try_into().unwrap());
        let im = half::bf16::from_ne_bytes(bytes[2..4].try_into().unwrap());
        Ok(zarrs_metadata::v3::FillValueMetadataV3::from(vec![
            zarrs_metadata::v3::FillValueMetadataV3::from(re),
            zarrs_metadata::v3::FillValueMetadataV3::from(im),
        ]))
    }};
}

/// Macro to implement `DataTypeExtension` for subfloat types (single-byte floating point formats).
macro_rules! impl_subfloat_data_type {
    ($marker:ty) => {
        impl zarrs_data_type::DataTypeExtension for $marker {
            fn identifier(&self) -> &'static str {
                <Self as zarrs_plugin::ExtensionIdentifier>::IDENTIFIER
            }

            fn configuration(&self) -> zarrs_metadata::Configuration {
                zarrs_metadata::Configuration::default()
            }

            fn size(&self) -> zarrs_metadata::DataTypeSize {
                zarrs_metadata::DataTypeSize::Fixed(1)
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
                // Subfloats use hex string representation like "0x00"
                if let Some(s) = fill_value_metadata.as_str() {
                    if let Some(hex) = s.strip_prefix("0x") {
                        if let Ok(byte) = u8::from_str_radix(hex, 16) {
                            return Ok(zarrs_data_type::FillValue::from(byte));
                        }
                    }
                }
                // Also accept integer values in range
                if let Some(int) = fill_value_metadata.as_u64() {
                    if let Ok(byte) = u8::try_from(int) {
                        return Ok(zarrs_data_type::FillValue::from(byte));
                    }
                }
                Err(err())
            }

            fn metadata_fill_value(
                &self,
                fill_value: &zarrs_data_type::FillValue,
            ) -> Result<zarrs_metadata::v3::FillValueMetadataV3, zarrs_data_type::DataTypeFillValueError> {
                let error = || {
                    zarrs_data_type::DataTypeFillValueError::new(self.identifier().to_string(), fill_value.clone())
                };
                let bytes: [u8; 1] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                // Return as hex string
                Ok(zarrs_metadata::v3::FillValueMetadataV3::from(format!("0x{:02x}", bytes[0])))
            }

            fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
                Some(self)
            }
        }

        impl zarrs_data_type::DataTypeExtensionBytesCodec for $marker {
            fn encode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                _endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
                // Single byte, no endianness conversion needed
                Ok(bytes)
            }

            fn decode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                _endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
                // Single byte, no endianness conversion needed
                Ok(bytes)
            }
        }
    };
}

/// Macro to implement `DataTypeExtension` for complex subfloat types (two subfloats packed together).
macro_rules! impl_complex_subfloat_data_type {
    ($marker:ty) => {
        impl zarrs_data_type::DataTypeExtension for $marker {
            fn identifier(&self) -> &'static str {
                <Self as zarrs_plugin::ExtensionIdentifier>::IDENTIFIER
            }

            fn configuration(&self) -> zarrs_metadata::Configuration {
                zarrs_metadata::Configuration::default()
            }

            fn size(&self) -> zarrs_metadata::DataTypeSize {
                zarrs_metadata::DataTypeSize::Fixed(2)
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
                // Complex subfloats use array of two hex strings like ["0x00", "0x00"]
                if let Some([re, im]) = fill_value_metadata.as_array() {
                    let parse_hex = |v: &zarrs_metadata::v3::FillValueMetadataV3| -> Option<u8> {
                        if let Some(s) = v.as_str() {
                            if let Some(hex) = s.strip_prefix("0x") {
                                return u8::from_str_radix(hex, 16).ok();
                            }
                        }
                        if let Some(int) = v.as_u64() {
                            return u8::try_from(int).ok();
                        }
                        None
                    };
                    if let (Some(re_byte), Some(im_byte)) = (parse_hex(re), parse_hex(im)) {
                        return Ok(zarrs_data_type::FillValue::from([re_byte, im_byte]));
                    }
                }
                Err(err())
            }

            fn metadata_fill_value(
                &self,
                fill_value: &zarrs_data_type::FillValue,
            ) -> Result<zarrs_metadata::v3::FillValueMetadataV3, zarrs_data_type::DataTypeFillValueError> {
                let error = || {
                    zarrs_data_type::DataTypeFillValueError::new(self.identifier().to_string(), fill_value.clone())
                };
                let bytes: [u8; 2] = fill_value.as_ne_bytes().try_into().map_err(|_| error())?;
                // Return as array of hex strings
                Ok(zarrs_metadata::v3::FillValueMetadataV3::from(vec![
                    zarrs_metadata::v3::FillValueMetadataV3::from(format!("0x{:02x}", bytes[0])),
                    zarrs_metadata::v3::FillValueMetadataV3::from(format!("0x{:02x}", bytes[1])),
                ]))
            }

            fn codec_bytes(&self) -> Option<&dyn zarrs_data_type::DataTypeExtensionBytesCodec> {
                Some(self)
            }
        }

        impl zarrs_data_type::DataTypeExtensionBytesCodec for $marker {
            fn encode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                _endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
                // Two single-byte components, no endianness conversion needed
                Ok(bytes)
            }

            fn decode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                _endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<std::borrow::Cow<'a, [u8]>, zarrs_data_type::DataTypeExtensionBytesCodecError> {
                // Two single-byte components, no endianness conversion needed
                Ok(bytes)
            }
        }
    };
}

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

pub(crate) use impl_complex_data_type;
pub(crate) use impl_complex_subfloat_data_type;
pub(crate) use impl_data_type_extension_numeric;
pub(crate) use impl_subfloat_data_type;
pub(crate) use register_data_type_plugin;
