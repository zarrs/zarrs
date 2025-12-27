//! A custom data type for `uint4`.
//!
//! It accepts uint compatible fill values.

use std::{borrow::Cow, sync::Arc};

use serde::Deserialize;
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
    DataTypeExtension, DataTypeExtensionBytesCodec, DataTypeExtensionBytesCodecError,
    DataTypeExtensionError, DataTypeExtensionPackBitsCodec, DataTypeFillValueError,
    DataTypeFillValueMetadataError, DataTypePlugin, FillValue,
};
use zarrs_plugin::{PluginCreateError, PluginMetadataInvalidError, ZarrVersions};

/// A unique identifier for  the custom data type.
const UINT4: &'static str = "zarrs.test.uint4";

/// The data type for an array of the custom data type.
#[derive(Debug)]
struct CustomDataTypeUInt4;

/// The in-memory representation of the custom data type.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
struct CustomDataTypeUInt4Element(u8);

// Register the data type so that it can be recognised when opening arrays.
inventory::submit! {
    DataTypePlugin::new(UINT4, matches_name_custom_dtype, default_name_custom_dtype, create_custom_dtype)
}

fn matches_name_custom_dtype(name: &str, _version: ZarrVersions) -> bool {
    name == UINT4
}

fn default_name_custom_dtype(_version: ZarrVersions) -> Cow<'static, str> {
    UINT4.into()
}

fn create_custom_dtype(
    metadata: &MetadataV3,
) -> Result<Arc<dyn DataTypeExtension>, PluginCreateError> {
    if metadata.configuration_is_none_or_empty() {
        Ok(Arc::new(CustomDataTypeUInt4))
    } else {
        Err(PluginMetadataInvalidError::new(UINT4, "codec", metadata.to_string()).into())
    }
}

/// Implement the core data type extension methods
impl DataTypeExtension for CustomDataTypeUInt4 {
    fn identifier(&self) -> &'static str {
        UINT4
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
        let element = CustomDataTypeUInt4Element::try_from(element_metadata).map_err(|_| {
            DataTypeFillValueMetadataError::new(UINT4.to_string(), fill_value_metadata.clone())
        })?;
        Ok(FillValue::new(element.to_ne_bytes().to_vec()))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadataV3, DataTypeFillValueError> {
        let element = CustomDataTypeUInt4Element::from_ne_bytes(
            fill_value.as_ne_bytes().try_into().map_err(|_| {
                DataTypeFillValueError::new(self.identifier().to_string(), fill_value.clone())
            })?,
        );
        Ok(FillValueMetadataV3::from(element.as_u8()))
    }

    fn size(&self) -> zarrs::array::DataTypeSize {
        DataTypeSize::Fixed(1)
    }

    fn codec_bytes(&self) -> Result<&dyn DataTypeExtensionBytesCodec, DataTypeExtensionError> {
        Ok(self)
    }

    fn codec_packbits(
        &self,
    ) -> Result<&dyn DataTypeExtensionPackBitsCodec, DataTypeExtensionError> {
        Ok(self)
    }
}

/// Add support for the `bytes` codec. This must be implemented for fixed-size data types, even if they just pass-through the data type.
impl DataTypeExtensionBytesCodec for CustomDataTypeUInt4 {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        _endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, DataTypeExtensionBytesCodecError> {
        Ok(bytes)
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        _endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, DataTypeExtensionBytesCodecError> {
        Ok(bytes)
    }
}

/// Add support for the `packbits` codec.
impl DataTypeExtensionPackBitsCodec for CustomDataTypeUInt4 {
    fn component_size_bits(&self) -> u64 {
        4
    }

    fn num_components(&self) -> u64 {
        1
    }

    fn sign_extension(&self) -> bool {
        false
    }
}

impl TryFrom<u64> for CustomDataTypeUInt4Element {
    type Error = u64;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        if value < 16 {
            Ok(Self(value as u8))
        } else {
            Err(value)
        }
    }
}

impl CustomDataTypeUInt4Element {
    fn to_ne_bytes(&self) -> [u8; 1] {
        [self.0]
    }

    fn from_ne_bytes(bytes: &[u8; 1]) -> Self {
        Self(bytes[0])
    }

    fn as_u8(&self) -> u8 {
        self.0
    }
}

/// This defines how an in-memory CustomDataTypeUInt4Element is converted into ArrayBytes before encoding via the codec pipeline.
impl Element for CustomDataTypeUInt4Element {
    fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
        (data_type == &DataType::Extension(Arc::new(CustomDataTypeUInt4)))
            .then_some(())
            .ok_or(ArrayError::IncompatibleElementType)
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<zarrs::array::ArrayBytes<'a>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let mut bytes: Vec<u8> =
            Vec::with_capacity(elements.len() * size_of::<CustomDataTypeUInt4Element>());
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

/// This defines how ArrayBytes are converted into a CustomDataTypeUInt4Element after decoding via the codec pipeline.
impl ElementOwned for CustomDataTypeUInt4Element {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let bytes = bytes.into_fixed()?;
        let bytes_len = bytes.len();
        let mut elements = Vec::with_capacity(bytes_len / size_of::<CustomDataTypeUInt4Element>());
        for byte in bytes.iter() {
            elements.push(CustomDataTypeUInt4Element(*byte))
        }
        Ok(elements)
    }
}

fn main() {
    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let fill_value = CustomDataTypeUInt4Element::try_from(15).unwrap();
    let array = ArrayBuilder::new(
        vec![6, 1], // array shape
        vec![5, 1], // regular chunk shape
        DataType::Extension(Arc::new(CustomDataTypeUInt4)),
        FillValue::new(fill_value.to_ne_bytes().to_vec()),
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

    let data = [
        CustomDataTypeUInt4Element::try_from(1).unwrap(),
        CustomDataTypeUInt4Element::try_from(2).unwrap(),
        CustomDataTypeUInt4Element::try_from(3).unwrap(),
        CustomDataTypeUInt4Element::try_from(4).unwrap(),
        CustomDataTypeUInt4Element::try_from(5).unwrap(),
    ];
    array.store_chunk(&[0, 0], &data).unwrap();

    let data: Vec<CustomDataTypeUInt4Element> =
        array.retrieve_array_subset(&array.subset_all()).unwrap();

    for f in &data {
        println!("uint4: {:08b} u8: {}", f.as_u8(), f.as_u8());
    }

    assert_eq!(data[0], CustomDataTypeUInt4Element::try_from(1).unwrap());
    assert_eq!(data[1], CustomDataTypeUInt4Element::try_from(2).unwrap());
    assert_eq!(data[2], CustomDataTypeUInt4Element::try_from(3).unwrap());
    assert_eq!(data[3], CustomDataTypeUInt4Element::try_from(4).unwrap());
    assert_eq!(data[4], CustomDataTypeUInt4Element::try_from(5).unwrap());
    assert_eq!(data[5], CustomDataTypeUInt4Element::try_from(15).unwrap());

    let data: Vec<CustomDataTypeUInt4Element> = array
        .retrieve_array_subset(&ArraySubset::new_with_ranges(&[1..3, 0..1]))
        .unwrap();
    assert_eq!(data[0], CustomDataTypeUInt4Element::try_from(2).unwrap());
    assert_eq!(data[1], CustomDataTypeUInt4Element::try_from(3).unwrap());
}

#[test]
fn custom_data_type_uint4() {
    main()
}
