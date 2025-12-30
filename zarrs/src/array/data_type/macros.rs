//! Macros for implementing [`DataTypeExtension`](zarrs_data_type::DataTypeExtension) for data type markers.

/// Helper macro to implement `DataTypeExtension` for simple fixed-size numeric types.
///
/// This macro implements the `DataTypeExtension` trait and `BytesCodecDataTypeTraits` trait,
/// and also registers the type with the bytes codec registry.
///
/// Usage:
/// - `impl_data_type_extension_numeric!(MarkerType, size, rust_type)` - basic implementation
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

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        #[allow(unused_variables)]
        impl crate::array::codec::BytesCodecDataTypeTraits for $marker {
            fn encode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<std::borrow::Cow<'a, [u8]>, crate::array::codec::CodecError> {
                impl_data_type_extension_numeric!(@bytes_codec bytes, endianness, $rust_type, $size)
            }

            fn decode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<std::borrow::Cow<'a, [u8]>, crate::array::codec::CodecError> {
                impl_data_type_extension_numeric!(@bytes_codec bytes, endianness, $rust_type, $size)
            }
        }

        crate::register_data_type_extension_codec!(
            $marker,
            crate::array::codec::BytesPlugin,
            crate::array::codec::BytesCodecDataTypeTraits
        );
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
        let endianness = $endianness.ok_or(crate::array::codec::CodecError::from("endianness must be specified for multi-byte data types"))?;
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
