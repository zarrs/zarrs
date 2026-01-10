//! Macros for implementing [`DataTypeTraits`](zarrs_data_type::DataTypeTraits) for data type markers.

/// Helper macro to implement `DataTypeTraits` for simple fixed-size numeric types.
///
/// This macro implements the `DataTypeTraits` trait and `BytesCodecDataTypeTraits` trait,
/// and also registers the type with the bytes codec registry.
///
/// Usage:
/// - `impl_data_type_extension_numeric!(MarkerType, size, rust_type)` - basic implementation
macro_rules! impl_data_type_extension_numeric {
    ($marker:ty, $size:tt, $rust_type:tt) => {
        impl zarrs_data_type::DataTypeTraits for $marker {
            fn configuration(&self) -> zarrs_metadata::Configuration {
                zarrs_metadata::Configuration::default()
            }

            fn size(&self) -> zarrs_metadata::DataTypeSize {
                zarrs_metadata::DataTypeSize::Fixed($size)
            }

            fn fill_value(
                &self,
                fill_value_metadata: &zarrs_metadata::FillValueMetadata,
                version: zarrs_plugin::ZarrVersions,
            ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
                // V2: null fill value means default (0 for numeric types)
                if matches!(version, zarrs_plugin::ZarrVersions::V2)
                    && fill_value_metadata.is_null()
                {
                    return impl_data_type_extension_numeric!(@fill_value_default $rust_type);
                }
                impl_data_type_extension_numeric!(@fill_value self, fill_value_metadata, $rust_type)
            }

            fn metadata_fill_value(
                &self,
                fill_value: &zarrs_data_type::FillValue,
            ) -> Result<zarrs_metadata::FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
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

        crate::array::codec::register_data_type_extension_codec!(
            $marker,
            crate::array::codec::BytesPlugin,
            crate::array::codec::BytesCodecDataTypeTraits
        );
    };

    // Fill value from metadata for signed integers
    (@fill_value $self:ident, $fill_value_metadata:ident, i8) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError;
        let int = $fill_value_metadata.as_i64().ok_or_else(err)?;
        let int = i8::try_from(int).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, i16) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError;
        let int = $fill_value_metadata.as_i64().ok_or_else(err)?;
        let int = i16::try_from(int).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, i32) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError;
        let int = $fill_value_metadata.as_i64().ok_or_else(err)?;
        let int = i32::try_from(int).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, i64) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError;
        let int = $fill_value_metadata.as_i64().ok_or_else(err)?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    // Fill value from metadata for unsigned integers
    (@fill_value $self:ident, $fill_value_metadata:ident, u8) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError;
        let int = $fill_value_metadata.as_u64().ok_or_else(err)?;
        let int = u8::try_from(int).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, u16) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError;
        let int = $fill_value_metadata.as_u64().ok_or_else(err)?;
        let int = u16::try_from(int).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, u32) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError;
        let int = $fill_value_metadata.as_u64().ok_or_else(err)?;
        let int = u32::try_from(int).map_err(|_| err())?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, u64) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError;
        let int = $fill_value_metadata.as_u64().ok_or_else(err)?;
        Ok(zarrs_data_type::FillValue::from(int))
    }};
    // Fill value from metadata for floats
    (@fill_value $self:ident, $fill_value_metadata:ident, f16) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError;
        Ok(zarrs_data_type::FillValue::from($fill_value_metadata.as_f16().ok_or_else(err)?))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, bf16) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError;
        Ok(zarrs_data_type::FillValue::from($fill_value_metadata.as_bf16().ok_or_else(err)?))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, f32) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError;
        Ok(zarrs_data_type::FillValue::from($fill_value_metadata.as_f32().ok_or_else(err)?))
    }};
    (@fill_value $self:ident, $fill_value_metadata:ident, f64) => {{
        let err = || zarrs_data_type::DataTypeFillValueMetadataError;
        Ok(zarrs_data_type::FillValue::from($fill_value_metadata.as_f64().ok_or_else(err)?))
    }};

    // Default fill value for V2 null (0 for numeric types)
    (@fill_value_default i8) => {{ Ok(zarrs_data_type::FillValue::from(0i8)) }};
    (@fill_value_default i16) => {{ Ok(zarrs_data_type::FillValue::from(0i16)) }};
    (@fill_value_default i32) => {{ Ok(zarrs_data_type::FillValue::from(0i32)) }};
    (@fill_value_default i64) => {{ Ok(zarrs_data_type::FillValue::from(0i64)) }};
    (@fill_value_default u8) => {{ Ok(zarrs_data_type::FillValue::from(0u8)) }};
    (@fill_value_default u16) => {{ Ok(zarrs_data_type::FillValue::from(0u16)) }};
    (@fill_value_default u32) => {{ Ok(zarrs_data_type::FillValue::from(0u32)) }};
    (@fill_value_default u64) => {{ Ok(zarrs_data_type::FillValue::from(0u64)) }};
    (@fill_value_default f16) => {{ Ok(zarrs_data_type::FillValue::from(half::f16::ZERO)) }};
    (@fill_value_default bf16) => {{ Ok(zarrs_data_type::FillValue::from(half::bf16::ZERO)) }};
    (@fill_value_default f32) => {{ Ok(zarrs_data_type::FillValue::from(0.0f32)) }};
    (@fill_value_default f64) => {{ Ok(zarrs_data_type::FillValue::from(0.0f64)) }};

    // Metadata fill value for specific types
    (@metadata_fill_value $self:ident, $fill_value:ident, i8, 1) => {{
        let bytes: [u8; 1] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = i8::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, i16, 2) => {{
        let bytes: [u8; 2] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = i16::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, i32, 4) => {{
        let bytes: [u8; 4] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = i32::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, i64, 8) => {{
        let bytes: [u8; 8] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = i64::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, u8, 1) => {{
        let bytes: [u8; 1] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = u8::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, u16, 2) => {{
        let bytes: [u8; 2] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = u16::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, u32, 4) => {{
        let bytes: [u8; 4] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = u32::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, u64, 8) => {{
        let bytes: [u8; 8] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = u64::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, f16, 2) => {{
        let bytes: [u8; 2] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = half::f16::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, bf16, 2) => {{
        let bytes: [u8; 2] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = half::bf16::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, f32, 4) => {{
        let bytes: [u8; 4] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = f32::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }};
    (@metadata_fill_value $self:ident, $fill_value:ident, f64, 8) => {{
        let bytes: [u8; 8] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = f64::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
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

/// Macro to register a configuration free data type as `DataTypePluginV3` and `DataTypePluginV2`.
///
/// These data types must be a marker type (i.e. unit struct).
macro_rules! register_data_type_plugin {
    ($marker:ident) => {
        // Register V3 plugin
        inventory::submit! {
            zarrs_data_type::DataTypePluginV3::new::<$marker>(
                |_metadata: &zarrs_metadata::v3::MetadataV3| -> Result<zarrs_data_type::DataType, zarrs_plugin::PluginCreateError> {
                    Ok(std::sync::Arc::new($marker).into())
                },
            )
        }

        // Register V2 plugin
        inventory::submit! {
            zarrs_data_type::DataTypePluginV2::new::<$marker>(
                |_metadata: &zarrs_metadata::v2::DataTypeMetadataV2| -> Result<zarrs_data_type::DataType, zarrs_plugin::PluginCreateError> {
                    Ok(std::sync::Arc::new($marker).into())
                },
            )
        }
    };
}

pub(crate) use register_data_type_plugin;
