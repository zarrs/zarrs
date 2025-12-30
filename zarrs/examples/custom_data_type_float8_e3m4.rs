//! A custom data type for `float8_e3m4`.
//!
//! It accepts float compatible fill values.

use core::f32;
use std::{borrow::Cow, sync::Arc};

use serde::Deserialize;
use zarrs::array::codec::{BytesCodecDataTypeTraits, CodecError};
use zarrs::array::{
    ArrayBuilder, ArrayBytes, ArrayError, DataType, DataTypeSize, Element, ElementOwned,
    FillValueMetadataV3,
};
use zarrs::metadata::{Configuration, v3::MetadataV3};
use zarrs::storage::store::MemoryStore;
use zarrs_data_type::{
    DataTypeExtension, DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypePlugin,
    FillValue,
};
use zarrs_plugin::{PluginCreateError, PluginMetadataInvalidError, ZarrVersions};

/// A unique identifier for  the custom data type.
const FLOAT8_E3M4: &str = "zarrs.test.float8_e3m4";

/// The data type for an array of the custom data type.
#[derive(Debug)]
struct CustomDataTypeFloat8e3m4;

zarrs_plugin::impl_extension_aliases!(CustomDataTypeFloat8e3m4, FLOAT8_E3M4);

/// The in-memory representation of the custom data type.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
struct CustomDataTypeFloat8e3m4Element(u8);

// Register the data type so that it can be recognised when opening arrays.
inventory::submit! {
    DataTypePlugin::new(FLOAT8_E3M4, matches_name_custom_dtype, default_name_custom_dtype, create_custom_dtype)
}

fn matches_name_custom_dtype(name: &str, _version: ZarrVersions) -> bool {
    name == FLOAT8_E3M4
}

fn default_name_custom_dtype(_version: ZarrVersions) -> Cow<'static, str> {
    FLOAT8_E3M4.into()
}

fn create_custom_dtype(metadata: &MetadataV3) -> Result<DataType, PluginCreateError> {
    if metadata.configuration_is_none_or_empty() {
        Ok(Arc::new(CustomDataTypeFloat8e3m4))
    } else {
        Err(PluginMetadataInvalidError::new(FLOAT8_E3M4, "codec", metadata.to_string()).into())
    }
}

/// Implement the core data type extension methods
impl DataTypeExtension for CustomDataTypeFloat8e3m4 {
    fn identifier(&self) -> &'static str {
        FLOAT8_E3M4
    }

    fn configuration(&self) -> Configuration {
        Configuration::default()
    }

    fn fill_value(
        &self,
        fill_value_metadata: &FillValueMetadataV3,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        let err = || {
            DataTypeFillValueMetadataError::new(
                self.identifier().to_string(),
                fill_value_metadata.clone(),
            )
        };
        let element_metadata: f32 = fill_value_metadata.as_f32().ok_or_else(err)?;
        let element = CustomDataTypeFloat8e3m4Element::from(element_metadata);
        Ok(FillValue::new(element.to_ne_bytes().to_vec()))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadataV3, DataTypeFillValueError> {
        let element = CustomDataTypeFloat8e3m4Element::from_ne_bytes(
            fill_value.as_ne_bytes().try_into().map_err(|_| {
                DataTypeFillValueError::new(self.identifier().to_string(), fill_value.clone())
            })?,
        );
        Ok(FillValueMetadataV3::from(element.as_f32()))
    }

    fn size(&self) -> zarrs::array::DataTypeSize {
        DataTypeSize::Fixed(1)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Add support for the `bytes` codec. This must be implemented for fixed-size data types, even if they just pass-through the data type.
impl BytesCodecDataTypeTraits for CustomDataTypeFloat8e3m4 {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        _endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, CodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        _endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, CodecError> {
        Ok(bytes)
    }
}

// Register codec support
zarrs::array::codec::register_data_type_extension_codec!(
    CustomDataTypeFloat8e3m4,
    zarrs::array::codec::BytesPlugin,
    zarrs::array::codec::BytesCodecDataTypeTraits
);

// FIXME: Not tested for correctness. Prefer a supporting crate.
fn float32_to_float8_e3m4(val: f32) -> u8 {
    let bits = val.to_bits();
    let sign = ((bits >> 24) & 0x80) as u8;
    let unbiased_exponent = ((bits >> 23) & 0xFF) as i16 - 127;
    let mantissa = ((bits >> 19) & 0x0F) as u8;

    let biased_to_exponent = unbiased_exponent + 3;

    if biased_to_exponent < 0 {
        // Flush denormals and underflowing values to zero
        sign
    } else if biased_to_exponent > 7 {
        // Overflow: return ±Infinity
        sign | 0b01110000
    } else {
        sign | ((biased_to_exponent as u8) << 4) | mantissa
    }
}

// FIXME: Not tested for correctness. Prefer a supporting crate.
fn float8_e3m4_to_float32(val: u8) -> f32 {
    let sign = (val & 0b10000000) as u32;
    let biased_exponent = ((val >> 4) & 0b111) as i16;
    let mantissa = (val & 0b1111) as u32;

    let f32_bits = if biased_exponent == 0 {
        // Subnormal
        return f32::from_bits(sign << 24 | mantissa << 19);
    } else if biased_exponent == 7 {
        // Infinity or NaN
        if mantissa == 0 {
            (sign << 24) | 0x7F800000 // ±Infinity
        } else {
            (sign << 24) | 0x7F800000 | (mantissa << 19) // NaN
        }
    } else {
        let unbiased_exponent = biased_exponent - 3;
        let biased_to_exponent = (unbiased_exponent + 127) as u32;
        let new_mantissa = mantissa << 19;
        (sign << 24) | (biased_to_exponent << 23) | new_mantissa
    };
    f32::from_bits(f32_bits)
}

impl From<f32> for CustomDataTypeFloat8e3m4Element {
    fn from(value: f32) -> Self {
        Self(float32_to_float8_e3m4(value))
    }
}

impl CustomDataTypeFloat8e3m4Element {
    fn to_ne_bytes(&self) -> [u8; 1] {
        [self.0]
    }

    fn from_ne_bytes(bytes: &[u8; 1]) -> Self {
        Self(bytes[0])
    }

    fn as_f32(&self) -> f32 {
        float8_e3m4_to_float32(self.0)
    }
}

/// This defines how an in-memory CustomDataTypeFloat8e3m4Element is converted into ArrayBytes before encoding via the codec pipeline.
impl Element for CustomDataTypeFloat8e3m4Element {
    fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
        (data_type.identifier() == FLOAT8_E3M4)
            .then_some(())
            .ok_or(ArrayError::IncompatibleElementType)
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<zarrs::array::ArrayBytes<'a>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let mut bytes: Vec<u8> = Vec::with_capacity(elements.len());
        for element in elements {
            bytes.push(element.0);
        }
        Ok(ArrayBytes::Fixed(Cow::Owned(bytes)))
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<zarrs::array::ArrayBytes<'static>, ArrayError> {
        Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
    }
}

/// This defines how ArrayBytes are converted into a CustomDataTypeFloat8e3m4Element after decoding via the codec pipeline.
impl ElementOwned for CustomDataTypeFloat8e3m4Element {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let bytes = bytes.into_fixed()?;
        let bytes_len = bytes.len();
        let mut elements = Vec::with_capacity(bytes_len);
        // NOTE: Could memcpy here
        for byte in bytes.iter() {
            elements.push(CustomDataTypeFloat8e3m4Element(*byte))
        }
        Ok(elements)
    }
}

fn main() {
    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let fill_value = CustomDataTypeFloat8e3m4Element::from(1.23);
    let array = ArrayBuilder::new(
        vec![6, 1], // array shape
        vec![5, 1], // regular chunk shape
        Arc::new(CustomDataTypeFloat8e3m4) as DataType,
        FillValue::new(fill_value.to_ne_bytes().to_vec()),
    )
    .array_to_array_codecs(vec![
        #[cfg(feature = "transpose")]
        Arc::new(zarrs::array::codec::TransposeCodec::new(
            zarrs::array::codec::array_to_array::transpose::TransposeOrder::new(&[1, 0]).unwrap(),
        )),
    ])
    .bytes_to_bytes_codecs(vec![
        #[cfg(feature = "gzip")]
        Arc::new(zarrs::array::codec::GzipCodec::new(5).unwrap()),
        #[cfg(feature = "crc32c")]
        Arc::new(zarrs::array::codec::Crc32cCodec::new()),
    ])
    // .storage_transformers(vec![].into())
    .build(store, array_path)
    .unwrap();
    println!("{}", array.metadata().to_string_pretty());

    let data = [
        CustomDataTypeFloat8e3m4Element::from(2.34),
        CustomDataTypeFloat8e3m4Element::from(3.45),
        CustomDataTypeFloat8e3m4Element::from(f32::INFINITY),
        CustomDataTypeFloat8e3m4Element::from(f32::NEG_INFINITY),
        CustomDataTypeFloat8e3m4Element::from(f32::NAN),
    ];
    array.store_chunk(&[0, 0], &data).unwrap();

    let data: Vec<CustomDataTypeFloat8e3m4Element> =
        array.retrieve_array_subset(&array.subset_all()).unwrap();

    for f in &data {
        println!(
            "float8_e3m4: {:08b} f32: {}",
            f.to_ne_bytes()[0],
            f.as_f32()
        );
    }

    assert_eq!(data[0], CustomDataTypeFloat8e3m4Element::from(2.34));
    assert_eq!(data[1], CustomDataTypeFloat8e3m4Element::from(3.45));
    assert_eq!(
        data[2],
        CustomDataTypeFloat8e3m4Element::from(f32::INFINITY)
    );
    assert_eq!(
        data[3],
        CustomDataTypeFloat8e3m4Element::from(f32::NEG_INFINITY)
    );
    assert_eq!(data[4], CustomDataTypeFloat8e3m4Element::from(f32::NAN));
    assert_eq!(data[5], CustomDataTypeFloat8e3m4Element::from(1.23));
}

#[test]
fn custom_data_type_fixed_size() {
    main()
}
