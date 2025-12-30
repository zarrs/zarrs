//! A custom data type for `uint12`.
//!
//! It accepts uint compatible fill values.

use std::{borrow::Cow, sync::Arc};

use serde::Deserialize;
use zarrs::array::codec::{BytesCodecDataTypeTraits, CodecError, PackBitsCodecDataTypeTraits};
use zarrs::metadata::{Configuration, v3::MetadataV3};
use zarrs::storage::store::MemoryStore;
use zarrs::{
    array::{
        ArrayBuilder, ArrayBytes, ArrayError, DataType, DataTypeSize, Element, ElementOwned,
        FillValueMetadataV3,
    },
    array_subset::ArraySubset,
};
use zarrs_data_type::{
    DataTypeExtension, DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypePlugin,
    FillValue,
};
use zarrs_plugin::{PluginCreateError, PluginMetadataInvalidError, ZarrVersions};

/// A unique identifier for  the custom data type.
const UINT12: &str = "zarrs.test.uint12";

/// The data type for an array of the custom data type.
#[derive(Debug)]
struct CustomDataTypeUInt12;

zarrs_plugin::impl_extension_aliases!(CustomDataTypeUInt12, UINT12);

/// The in-memory representation of the custom data type.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
struct CustomDataTypeUInt12Element(u16);

// Register the data type so that it can be recognised when opening arrays.
inventory::submit! {
    DataTypePlugin::new(UINT12, matches_name_custom_dtype, default_name_custom_dtype, create_custom_dtype)
}

fn matches_name_custom_dtype(name: &str, _version: ZarrVersions) -> bool {
    name == UINT12
}

fn default_name_custom_dtype(_version: ZarrVersions) -> Cow<'static, str> {
    UINT12.into()
}

fn create_custom_dtype(metadata: &MetadataV3) -> Result<DataType, PluginCreateError> {
    if metadata.configuration_is_none_or_empty() {
        Ok(Arc::new(CustomDataTypeUInt12))
    } else {
        Err(PluginMetadataInvalidError::new(UINT12, "codec", metadata.to_string()).into())
    }
}

/// Implement the core data type extension methods
impl DataTypeExtension for CustomDataTypeUInt12 {
    fn identifier(&self) -> &'static str {
        UINT12
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
        let element_metadata: u64 = fill_value_metadata.as_u64().ok_or_else(err)?;
        let element = CustomDataTypeUInt12Element::try_from(element_metadata).map_err(|_| {
            DataTypeFillValueMetadataError::new(UINT12.to_string(), fill_value_metadata.clone())
        })?;
        Ok(FillValue::new(element.to_le_bytes().to_vec()))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadataV3, DataTypeFillValueError> {
        let element = CustomDataTypeUInt12Element::from_le_bytes(
            fill_value.as_ne_bytes().try_into().map_err(|_| {
                DataTypeFillValueError::new(self.identifier().to_string(), fill_value.clone())
            })?,
        );
        Ok(FillValueMetadataV3::from(element.as_u16()))
    }

    fn size(&self) -> zarrs::array::DataTypeSize {
        DataTypeSize::Fixed(2)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Add support for the `bytes` codec. This must be implemented for fixed-size data types, even if they just pass-through the data type.
impl BytesCodecDataTypeTraits for CustomDataTypeUInt12 {
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

/// Add support for the `packbits` codec.
impl PackBitsCodecDataTypeTraits for CustomDataTypeUInt12 {
    fn component_size_bits(&self) -> u64 {
        12
    }

    fn num_components(&self) -> u64 {
        1
    }

    fn sign_extension(&self) -> bool {
        false
    }
}

// Register codec support
zarrs::register_data_type_extension_codec!(
    CustomDataTypeUInt12,
    zarrs::array::codec::BytesPlugin,
    zarrs::array::codec::BytesCodecDataTypeTraits
);
zarrs::register_data_type_extension_codec!(
    CustomDataTypeUInt12,
    zarrs::array::codec::PackBitsPlugin,
    zarrs::array::codec::PackBitsCodecDataTypeTraits
);

impl TryFrom<u64> for CustomDataTypeUInt12Element {
    type Error = u64;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if value < 4096 {
            Ok(Self(value as u16))
        } else {
            Err(value)
        }
    }
}

impl CustomDataTypeUInt12Element {
    fn to_le_bytes(&self) -> [u8; 2] {
        self.0.to_le_bytes()
    }

    fn from_le_bytes(bytes: [u8; 2]) -> Self {
        Self(u16::from_le_bytes(bytes))
    }

    fn as_u16(&self) -> u16 {
        self.0
    }
}

/// This defines how an in-memory CustomDataTypeUInt12Element is converted into ArrayBytes before encoding via the codec pipeline.
impl Element for CustomDataTypeUInt12Element {
    fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
        // Check if the data type identifier matches our custom data type
        (data_type.identifier() == UINT12)
            .then_some(())
            .ok_or(ArrayError::IncompatibleElementType)
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<zarrs::array::ArrayBytes<'a>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let mut bytes: Vec<u8> =
            Vec::with_capacity(elements.len() * size_of::<CustomDataTypeUInt12Element>());
        for element in elements {
            bytes.extend_from_slice(&element.to_le_bytes());
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

/// This defines how ArrayBytes are converted into a CustomDataTypeUInt12Element after decoding via the codec pipeline.
impl ElementOwned for CustomDataTypeUInt12Element {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let bytes = bytes.into_fixed()?;
        let bytes_len = bytes.len();
        let mut elements = Vec::with_capacity(bytes_len / size_of::<CustomDataTypeUInt12Element>());
        for chunk in bytes.as_chunks::<2>().0 {
            elements.push(CustomDataTypeUInt12Element::from_le_bytes(*chunk))
        }
        Ok(elements)
    }
}

fn main() {
    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let fill_value = CustomDataTypeUInt12Element::try_from(15).unwrap();
    let array = ArrayBuilder::new(
        vec![4096, 1], // array shape
        vec![5, 1],    // regular chunk shape
        Arc::new(CustomDataTypeUInt12) as DataType,
        FillValue::new(fill_value.to_le_bytes().to_vec()),
    )
    .array_to_array_codecs(vec![
        #[cfg(feature = "transpose")]
        Arc::new(zarrs::array::codec::TransposeCodec::new(
            zarrs::array::codec::array_to_array::transpose::TransposeOrder::new(&[1, 0]).unwrap(),
        )),
    ])
    .array_to_bytes_codec(Arc::new(zarrs::array::codec::PackBitsCodec::default()))
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

    let data: Vec<CustomDataTypeUInt12Element> = (0..4096)
        .into_iter()
        .map(|i| CustomDataTypeUInt12Element::try_from(i).unwrap())
        .collect();

    array
        .store_array_subset(&array.subset_all(), &data)
        .unwrap();

    let data: Vec<CustomDataTypeUInt12Element> =
        array.retrieve_array_subset(&array.subset_all()).unwrap();

    for i in 0usize..4096 {
        let element = CustomDataTypeUInt12Element::try_from(i as u64).unwrap();
        assert_eq!(data[i], element);
        let element_pd: Vec<CustomDataTypeUInt12Element> = array
            .retrieve_array_subset(&ArraySubset::new_with_ranges(&[
                (i as u64)..i as u64 + 1,
                0..1,
            ]))
            .unwrap();
        assert_eq!(element_pd[0], element);
    }
}

#[test]
fn custom_data_type_uint12() {
    main()
}
