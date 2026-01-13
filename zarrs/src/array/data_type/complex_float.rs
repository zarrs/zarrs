//! Standard complex float data types.

use super::macros::register_data_type_plugin;

/// Macro to implement `DataTypeTraits` for complex types.
macro_rules! impl_complex_data_type {
    ($marker:ty, $size:tt, $component_type:tt) => {
        impl zarrs_data_type::DataTypeTraits for $marker {
            fn configuration(&self, _version: zarrs_plugin::ZarrVersion) -> zarrs_metadata::Configuration {
                zarrs_metadata::Configuration::default()
            }

            fn size(&self) -> zarrs_metadata::DataTypeSize {
                zarrs_metadata::DataTypeSize::Fixed($size)
            }

            fn fill_value(
                &self,
                fill_value_metadata: &zarrs_metadata::FillValueMetadata,
                _version: zarrs_plugin::ZarrVersion,
            ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
                if let [re, im] = fill_value_metadata.as_array().ok_or(zarrs_data_type::DataTypeFillValueMetadataError)? {
                    impl_complex_data_type!(@parse_components self, re, im, $component_type)
                } else {
                    Err(zarrs_data_type::DataTypeFillValueMetadataError)
                }
            }

            fn metadata_fill_value(
                &self,
                fill_value: &zarrs_data_type::FillValue,
            ) -> Result<zarrs_metadata::FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
                impl_complex_data_type!(@to_metadata self, fill_value, $component_type, $size)
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        impl crate::array::codec::BytesCodecDataTypeTraits for $marker {
            fn encode<'a>(
                &self,
                bytes: std::borrow::Cow<'a, [u8]>,
                endianness: Option<zarrs_metadata::Endianness>,
            ) -> Result<std::borrow::Cow<'a, [u8]>, crate::array::codec::CodecError> {
                let component_size = $size / 2;
                if component_size == 1 {
                    Ok(bytes)
                } else {
                    let endianness = endianness.ok_or(crate::array::codec::CodecError::from(
                        "`bytes` codec `endianness` not specified for a multi-byte data type".to_string()
                    ))?;
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
            ) -> Result<std::borrow::Cow<'a, [u8]>, crate::array::codec::CodecError> {
                self.encode(bytes, endianness)
            }
        }

        zarrs_codec::register_data_type_extension_codec!(
            $marker,
            crate::array::codec::BytesPlugin,
            crate::array::codec::BytesCodecDataTypeTraits
        );
    };

    (@parse_components $self:ident, $re:ident, $im:ident, f32) => {{
        let re = $re.as_f32().ok_or(zarrs_data_type::DataTypeFillValueMetadataError)?;
        let im = $im.as_f32().ok_or(zarrs_data_type::DataTypeFillValueMetadataError)?;
        Ok(zarrs_data_type::FillValue::from(num::complex::Complex32::new(re, im)))
    }};
    (@parse_components $self:ident, $re:ident, $im:ident, f64) => {{
        let re = $re.as_f64().ok_or(zarrs_data_type::DataTypeFillValueMetadataError)?;
        let im = $im.as_f64().ok_or(zarrs_data_type::DataTypeFillValueMetadataError)?;
        Ok(zarrs_data_type::FillValue::from(num::complex::Complex64::new(re, im)))
    }};
    (@parse_components $self:ident, $re:ident, $im:ident, f16) => {{
        let re = $re.as_f16().ok_or(zarrs_data_type::DataTypeFillValueMetadataError)?;
        let im = $im.as_f16().ok_or(zarrs_data_type::DataTypeFillValueMetadataError)?;
        Ok(zarrs_data_type::FillValue::from(num::complex::Complex::<half::f16>::new(re, im)))
    }};
    (@parse_components $self:ident, $re:ident, $im:ident, bf16) => {{
        let re = $re.as_bf16().ok_or(zarrs_data_type::DataTypeFillValueMetadataError)?;
        let im = $im.as_bf16().ok_or(zarrs_data_type::DataTypeFillValueMetadataError)?;
        Ok(zarrs_data_type::FillValue::from(num::complex::Complex::<half::bf16>::new(re, im)))
    }};

    (@to_metadata $self:ident, $fill_value:ident, f32, 8) => {{
        let bytes: [u8; 8] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let re = f32::from_ne_bytes(bytes[0..4].try_into().unwrap());
        let im = f32::from_ne_bytes(bytes[4..8].try_into().unwrap());
        Ok(zarrs_metadata::FillValueMetadata::from(vec![
            zarrs_metadata::FillValueMetadata::from(re),
            zarrs_metadata::FillValueMetadata::from(im),
        ]))
    }};
    (@to_metadata $self:ident, $fill_value:ident, f64, 16) => {{
        let bytes: [u8; 16] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let re = f64::from_ne_bytes(bytes[0..8].try_into().unwrap());
        let im = f64::from_ne_bytes(bytes[8..16].try_into().unwrap());
        Ok(zarrs_metadata::FillValueMetadata::from(vec![
            zarrs_metadata::FillValueMetadata::from(re),
            zarrs_metadata::FillValueMetadata::from(im),
        ]))
    }};
    (@to_metadata $self:ident, $fill_value:ident, f16, 4) => {{
        let bytes: [u8; 4] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let re = half::f16::from_ne_bytes(bytes[0..2].try_into().unwrap());
        let im = half::f16::from_ne_bytes(bytes[2..4].try_into().unwrap());
        Ok(zarrs_metadata::FillValueMetadata::from(vec![
            zarrs_metadata::FillValueMetadata::from(re),
            zarrs_metadata::FillValueMetadata::from(im),
        ]))
    }};
    (@to_metadata $self:ident, $fill_value:ident, bf16, 4) => {{
        let bytes: [u8; 4] = $fill_value.as_ne_bytes().try_into().map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let re = half::bf16::from_ne_bytes(bytes[0..2].try_into().unwrap());
        let im = half::bf16::from_ne_bytes(bytes[2..4].try_into().unwrap());
        Ok(zarrs_metadata::FillValueMetadata::from(vec![
            zarrs_metadata::FillValueMetadata::from(re),
            zarrs_metadata::FillValueMetadata::from(im),
        ]))
    }};
}

/// The `complex_bfloat16` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexBFloat16DataType;
register_data_type_plugin!(ComplexBFloat16DataType);
zarrs_plugin::impl_extension_aliases!(ComplexBFloat16DataType, v3: "complex_bfloat16");

/// The `complex_float16` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat16DataType;
register_data_type_plugin!(ComplexFloat16DataType);
zarrs_plugin::impl_extension_aliases!(ComplexFloat16DataType, v3: "complex_float16");

/// The `complex_float32` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat32DataType;
register_data_type_plugin!(ComplexFloat32DataType);
zarrs_plugin::impl_extension_aliases!(ComplexFloat32DataType, v3: "complex_float32");

/// The `complex_float64` data type.
#[derive(Debug, Clone, Copy)]
pub struct ComplexFloat64DataType;
register_data_type_plugin!(ComplexFloat64DataType);
zarrs_plugin::impl_extension_aliases!(ComplexFloat64DataType, v3: "complex_float64");

/// The `complex64` data type.
#[derive(Debug, Clone, Copy)]
pub struct Complex64DataType;
register_data_type_plugin!(Complex64DataType);
zarrs_plugin::impl_extension_aliases!(Complex64DataType,
    v3: "complex64", [],
    v2: "<c8", ["<c8", ">c8"]
);

/// The `complex128` data type.
#[derive(Debug, Clone, Copy)]
pub struct Complex128DataType;
register_data_type_plugin!(Complex128DataType);
zarrs_plugin::impl_extension_aliases!(Complex128DataType,
    v3: "complex128", [],
    v2: "<c16", ["<c16", ">c16"]
);

// DataTypeTraits implementations for standard complex types
impl_complex_data_type!(ComplexBFloat16DataType, 4, bf16);
impl_complex_data_type!(ComplexFloat16DataType, 4, f16);
impl_complex_data_type!(ComplexFloat32DataType, 8, f32);
impl_complex_data_type!(ComplexFloat64DataType, 16, f64);
impl_complex_data_type!(Complex64DataType, 8, f32);
impl_complex_data_type!(Complex128DataType, 16, f64);

// Bitround implementations for standard complex types
#[cfg(feature = "bitround")]
mod bitround_impls {
    use crate::array::codec::array_to_array::bitround::impl_bitround_codec;
    impl_bitround_codec!(super::ComplexBFloat16DataType, 2, float16, 7);
    impl_bitround_codec!(super::ComplexFloat16DataType, 2, float16, 10);
    impl_bitround_codec!(super::ComplexFloat32DataType, 4, float32, 23);
    impl_bitround_codec!(super::ComplexFloat64DataType, 8, float64, 52);
    impl_bitround_codec!(super::Complex64DataType, 4, float32, 23);
    impl_bitround_codec!(super::Complex128DataType, 8, float64, 52);
}

// Pcodec implementations for standard complex types
#[cfg(feature = "pcodec")]
mod pcodec_impls {
    use crate::array::codec::array_to_bytes::pcodec::impl_pcodec_codec;
    // crate::array::codec::array_to_bytes::pcodec::impl_pcodec_codec!(ComplexBFloat16DataType, BF16, 2);
    impl_pcodec_codec!(super::ComplexFloat16DataType, F16, 2);
    impl_pcodec_codec!(super::ComplexFloat32DataType, F32, 2);
    impl_pcodec_codec!(super::ComplexFloat64DataType, F64, 2);
    impl_pcodec_codec!(super::Complex64DataType, F32, 2);
    impl_pcodec_codec!(super::Complex128DataType, F64, 2);
}

// PackBits implementations for standard complex types
use crate::array::codec::array_to_bytes::packbits::impl_packbits_codec;
impl_packbits_codec!(super::ComplexBFloat16DataType, 16, float, 2);
impl_packbits_codec!(super::ComplexFloat16DataType, 16, float, 2);
impl_packbits_codec!(super::ComplexFloat32DataType, 32, float, 2);
impl_packbits_codec!(super::ComplexFloat64DataType, 64, float, 2);
impl_packbits_codec!(super::Complex64DataType, 32, float, 2);
impl_packbits_codec!(super::Complex128DataType, 64, float, 2);
