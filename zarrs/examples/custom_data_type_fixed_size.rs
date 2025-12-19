//! A custom structured data type {x: u64, y:f32}.
//!
//! This structure has 16 bytes in-memory due to padding.
//! It is passed into the codec pieline with padding removed (12 bytes).
//! The bytes codec properly serialises each element in the requested endianness.
//!
//! Fill values are of the form
//! ```json
//! {
//!   "x": 123,
//!   "y" 4.56
//! }
//! ```

use std::{borrow::Cow, collections::HashMap, sync::Arc};

use num::traits::{FromBytes, ToBytes};
use serde::Deserialize;
use zarrs::array::{
    ArrayBuilder, ArrayBytes, ArrayError, DataType, DataTypeSize, Element, ElementOwned,
    FillValueMetadataV3,
};
use zarrs::metadata::{Configuration, Endianness, v3::MetadataV3};
use zarrs::storage::store::MemoryStore;
use zarrs_data_type::{
    DataTypeExtension, DataTypeExtensionBytesCodec, DataTypeExtensionBytesCodecError,
    DataTypeExtensionError, DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypePlugin,
    FillValue,
};
use zarrs_plugin::{PluginCreateError, PluginMetadataInvalidError};

/// The in-memory representation of the custom data type.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
struct CustomDataTypeFixedSizeElement {
    x: u64,
    y: f32,
}

/// Defines the conversion of an element to a fill value
impl From<CustomDataTypeFixedSizeElement> for FillValueMetadataV3 {
    fn from(value: CustomDataTypeFixedSizeElement) -> Self {
        FillValueMetadataV3::from(HashMap::from([
            ("x".to_string(), FillValueMetadataV3::from(value.x)),
            ("y".to_string(), FillValueMetadataV3::from(value.y)),
        ]))
    }
}

/// The metadata is structured the same as the data type element
type CustomDataTypeFixedSizeMetadata = CustomDataTypeFixedSizeElement;

/// The padding bytes of CustomDataTypeFixedSizeElement are not serialised.
/// These are stripped as soon as the data is converted into ArrayBytes *before* it goes into the codec pipeline.
type CustomDataTypeFixedSizeBytes = [u8; size_of::<u64>() + size_of::<f32>()];

/// These defines how the CustomDataTypeFixedSizeBytes are converted TO little/big endian
/// Implementing this particular trait (from num-traits) is not necessary, but it is used in DataTypeExtensionBytesCodec/Element/ElementOwned
impl ToBytes for CustomDataTypeFixedSizeElement {
    type Bytes = CustomDataTypeFixedSizeBytes;

    fn to_be_bytes(&self) -> Self::Bytes {
        let mut bytes = [0; size_of::<CustomDataTypeFixedSizeBytes>()];
        let (x, y) = bytes.split_at_mut(size_of::<u64>());
        x.copy_from_slice(&self.x.to_be_bytes());
        y.copy_from_slice(&self.y.to_be_bytes());
        bytes
    }

    fn to_le_bytes(&self) -> Self::Bytes {
        let mut bytes = [0; size_of::<CustomDataTypeFixedSizeBytes>()];
        let (x, y) = bytes.split_at_mut(size_of::<u64>());
        x.copy_from_slice(&self.x.to_le_bytes());
        y.copy_from_slice(&self.y.to_le_bytes());
        bytes
    }
}

/// These defines how the CustomDataTypeFixedSizeBytes are converted FROM little/big endian
/// Implementing this particular trait (from num-traits) is not necessary, but it is used in DataTypeExtensionBytesCodec/Element/ElementOwned
impl FromBytes for CustomDataTypeFixedSizeElement {
    type Bytes = CustomDataTypeFixedSizeBytes;

    fn from_be_bytes(bytes: &Self::Bytes) -> Self {
        let (x, y) = bytes.split_at(size_of::<u64>());
        CustomDataTypeFixedSizeElement {
            x: u64::from_be_bytes(unsafe { x.try_into().unwrap_unchecked() }),
            y: f32::from_be_bytes(unsafe { y.try_into().unwrap_unchecked() }),
        }
    }

    fn from_le_bytes(bytes: &Self::Bytes) -> Self {
        let (x, y) = bytes.split_at(size_of::<u64>());
        CustomDataTypeFixedSizeElement {
            x: u64::from_le_bytes(unsafe { x.try_into().unwrap_unchecked() }),
            y: f32::from_le_bytes(unsafe { y.try_into().unwrap_unchecked() }),
        }
    }
}

/// This defines how an in-memory CustomDataTypeFixedSizeElement is converted into ArrayBytes before encoding via the codec pipeline.
impl Element for CustomDataTypeFixedSizeElement {
    fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
        (data_type == &DataType::Extension(Arc::new(CustomDataTypeFixedSize)))
            .then_some(())
            .ok_or(ArrayError::IncompatibleElementType)
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<zarrs::array::ArrayBytes<'a>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let mut bytes: Vec<u8> =
            Vec::with_capacity(size_of::<CustomDataTypeFixedSizeBytes>() * elements.len());
        for element in elements {
            bytes.extend_from_slice(&element.to_ne_bytes());
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

/// This defines how ArrayBytes are converted into a CustomDataTypeFixedSizeElement after decoding via the codec pipeline.
impl ElementOwned for CustomDataTypeFixedSizeElement {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let bytes = bytes.into_fixed()?;
        let bytes_len = bytes.len();
        let mut elements =
            Vec::with_capacity(bytes_len / size_of::<CustomDataTypeFixedSizeBytes>());
        for bytes in bytes.chunks_exact(size_of::<CustomDataTypeFixedSizeBytes>()) {
            elements.push(CustomDataTypeFixedSizeElement::from_ne_bytes(unsafe {
                bytes.try_into().unwrap_unchecked()
            }))
        }
        Ok(elements)
    }
}

/// The data type for an array of [`CustomDataTypeFixedSizeElement`].
#[derive(Debug)]
struct CustomDataTypeFixedSize;

/// A custom unique identifier
const CUSTOM_NAME: &'static str = "zarrs.test.CustomDataTypeFixedSize";

fn is_custom_dtype(name: &str) -> bool {
    name == CUSTOM_NAME
}

fn create_custom_dtype(
    metadata: &MetadataV3,
) -> Result<Arc<dyn DataTypeExtension>, PluginCreateError> {
    if metadata.configuration_is_none_or_empty() {
        Ok(Arc::new(CustomDataTypeFixedSize))
    } else {
        Err(PluginMetadataInvalidError::new(CUSTOM_NAME, "codec", metadata.to_string()).into())
    }
}

// Register the data type so that it can be recognised when opening arrays.
inventory::submit! {
    DataTypePlugin::new(CUSTOM_NAME, is_custom_dtype, create_custom_dtype)
}

/// Implement the core data type extension methods
impl DataTypeExtension for CustomDataTypeFixedSize {
    fn name(&self) -> String {
        CUSTOM_NAME.to_string()
    }

    fn configuration(&self) -> Configuration {
        Configuration::default()
    }

    fn fill_value(
        &self,
        fill_value_metadata: &FillValueMetadataV3,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        let err = || DataTypeFillValueMetadataError::new(self.name(), fill_value_metadata.clone());
        let element_metadata: CustomDataTypeFixedSizeMetadata =
            fill_value_metadata.as_custom().ok_or_else(err)?;
        Ok(FillValue::new(element_metadata.to_ne_bytes().to_vec()))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadataV3, DataTypeFillValueError> {
        let element = CustomDataTypeFixedSizeMetadata::from_ne_bytes(
            fill_value
                .as_ne_bytes()
                .try_into()
                .map_err(|_| DataTypeFillValueError::new(self.name(), fill_value.clone()))?,
        );
        Ok(FillValueMetadataV3::from(element))
    }

    fn size(&self) -> zarrs::array::DataTypeSize {
        DataTypeSize::Fixed(size_of::<CustomDataTypeFixedSizeBytes>())
    }

    fn codec_bytes(&self) -> Result<&dyn DataTypeExtensionBytesCodec, DataTypeExtensionError> {
        Ok(self)
    }
}

/// Add support for the `bytes` codec. This must be implemented for fixed-size data types, even if they just pass-through the data type.
impl DataTypeExtensionBytesCodec for CustomDataTypeFixedSize {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, DataTypeExtensionBytesCodecError> {
        if let Some(endianness) = endianness {
            if endianness != Endianness::native() {
                let mut bytes = bytes.into_owned();
                for bytes in bytes.chunks_exact_mut(size_of::<CustomDataTypeFixedSizeBytes>()) {
                    let value = CustomDataTypeFixedSizeElement::from_ne_bytes(&unsafe {
                        bytes.try_into().unwrap_unchecked()
                    });
                    if endianness == Endianness::Little {
                        bytes.copy_from_slice(&value.to_le_bytes());
                    } else {
                        bytes.copy_from_slice(&value.to_be_bytes());
                    }
                }
                Ok(Cow::Owned(bytes))
            } else {
                Ok(bytes)
            }
        } else {
            Err(DataTypeExtensionBytesCodecError::EndiannessNotSpecified)
        }
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, DataTypeExtensionBytesCodecError> {
        if let Some(endianness) = endianness {
            if endianness != Endianness::native() {
                let mut bytes = bytes.into_owned();
                for bytes in bytes.chunks_exact_mut(size_of::<u64>() + size_of::<f32>()) {
                    let value = if endianness == Endianness::Little {
                        CustomDataTypeFixedSizeElement::from_le_bytes(&unsafe {
                            bytes.try_into().unwrap_unchecked()
                        })
                    } else {
                        CustomDataTypeFixedSizeElement::from_be_bytes(&unsafe {
                            bytes.try_into().unwrap_unchecked()
                        })
                    };
                    bytes.copy_from_slice(&value.to_ne_bytes());
                }
                Ok(Cow::Owned(bytes))
            } else {
                Ok(bytes)
            }
        } else {
            Err(DataTypeExtensionBytesCodecError::EndiannessNotSpecified)
        }
    }
}

fn main() {
    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let fill_value = CustomDataTypeFixedSizeElement { x: 1, y: 2.3 };
    let array = ArrayBuilder::new(
        vec![4, 1], // array shape
        vec![2, 1], // regular chunk shape
        DataType::Extension(Arc::new(CustomDataTypeFixedSize)),
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
        CustomDataTypeFixedSizeElement { x: 3, y: 4.5 },
        CustomDataTypeFixedSizeElement { x: 6, y: 7.8 },
    ];
    array.store_chunk(&[0, 0], &data).unwrap();

    let data: Vec<CustomDataTypeFixedSizeElement> =
        array.retrieve_array_subset(&array.subset_all()).unwrap();

    assert_eq!(data[0], CustomDataTypeFixedSizeElement { x: 3, y: 4.5 });
    assert_eq!(data[1], CustomDataTypeFixedSizeElement { x: 6, y: 7.8 });
    assert_eq!(data[2], CustomDataTypeFixedSizeElement { x: 1, y: 2.3 });
    assert_eq!(data[3], CustomDataTypeFixedSizeElement { x: 1, y: 2.3 });

    println!("{data:#?}");
}

#[test]
fn custom_data_type_fixed_size() {
    main()
}
