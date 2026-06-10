//! The `int4` data type.

use super::macros::register_data_type_plugin;

/// The `int4` data type.
#[derive(Debug, Clone, Copy)]
pub struct Int4DataType;
register_data_type_plugin!(Int4DataType);
zarrs_plugin::impl_extension_aliases!(Int4DataType, v3: "int4");

impl zarrs_data_type::DataTypeTraits for Int4DataType {
    fn configuration(&self, _version: zarrs_plugin::ZarrVersion) -> zarrs_metadata::Configuration {
        zarrs_metadata::Configuration::default()
    }

    fn size(&self) -> zarrs_metadata::DataTypeSize {
        zarrs_metadata::DataTypeSize::Fixed(1)
    }

    fn fill_value(
        &self,
        fill_value_metadata: &zarrs_metadata::FillValueMetadata,
        _version: zarrs_plugin::ZarrVersion,
    ) -> Result<zarrs_data_type::FillValue, zarrs_data_type::DataTypeFillValueMetadataError> {
        let int = fill_value_metadata
            .as_i64()
            .ok_or(zarrs_data_type::DataTypeFillValueMetadataError)?;
        // int4 range: -8 to 7
        if !(-8..=7).contains(&int) {
            return Err(zarrs_data_type::DataTypeFillValueMetadataError);
        }
        #[expect(clippy::cast_possible_truncation)]
        Ok(zarrs_data_type::FillValue::from(int as i8))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &zarrs_data_type::FillValue,
    ) -> Result<zarrs_metadata::FillValueMetadata, zarrs_data_type::DataTypeFillValueError> {
        let bytes: [u8; 1] = fill_value
            .as_ne_bytes()
            .try_into()
            .map_err(|_| zarrs_data_type::DataTypeFillValueError)?;
        let number = i8::from_ne_bytes(bytes);
        Ok(zarrs_metadata::FillValueMetadata::from(number))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn compatible_element_types(&self) -> &'static [std::any::TypeId] {
        const TYPES: [std::any::TypeId; 1] = [std::any::TypeId::of::<i8>()];
        &TYPES
    }
}

zarrs_data_type::codec_traits::impl_pack_bits_data_type_traits!(Int4DataType, 4, signed, 1);
zarrs_data_type::codec_traits::impl_bytes_data_type_traits!(Int4DataType, 1);
zarrs_data_type::codec_traits::impl_cast_value_data_type_traits_signed_integer!(
    Int4DataType,
    i8,
    4
);
// ScaleOffset implementation for int4 (stored as i8, range -8..=7)
use zarrs_data_type::codec_traits::impl_scale_offset_data_type_traits;
use zarrs_data_type::codec_traits::scale_offset::{
    ScaleOffsetDataTypeTraits, ScaleOffsetError, scale_offset_encode_int,
};

impl ScaleOffsetDataTypeTraits for Int4DataType {
    fn scale_offset_encode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        let offset: i8 = match offset {
            Some(bytes) => i8::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 0,
        };
        let scale: i8 = match scale {
            Some(bytes) => i8::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 1,
        };
        for chunk in bytes.as_chunks_mut::<1>().0 {
            let value = i8::from_ne_bytes(*chunk);
            let result = scale_offset_encode_int(&value, &offset, &scale)?;
            if !(-8..=7).contains(&result) {
                return Err(ScaleOffsetError::NotRepresentable);
            }
            *chunk = result.to_ne_bytes();
        }
        Ok(())
    }

    fn scale_offset_decode(
        &self,
        bytes: &mut [u8],
        offset: Option<&[u8]>,
        scale: Option<&[u8]>,
    ) -> Result<(), ScaleOffsetError> {
        let offset: i8 = match offset {
            Some(bytes) => i8::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 0,
        };
        let scale: i8 = match scale {
            Some(bytes) => i8::from_ne_bytes(
                bytes
                    .try_into()
                    .map_err(|_| ScaleOffsetError::InvalidElementBytes)?,
            ),
            None => 1,
        };
        if scale == 0 {
            return Err(ScaleOffsetError::DivisionByZero);
        }
        for chunk in bytes.as_chunks_mut::<1>().0 {
            let value = i8::from_ne_bytes(*chunk);
            let result = value
                .checked_div(scale)
                .and_then(|q| q.checked_add(offset))
                .ok_or(ScaleOffsetError::NotRepresentable)?;
            if !(-8..=7).contains(&result) {
                return Err(ScaleOffsetError::NotRepresentable);
            }
            *chunk = result.to_ne_bytes();
        }
        Ok(())
    }
}
impl_scale_offset_data_type_traits!(Int4DataType);
