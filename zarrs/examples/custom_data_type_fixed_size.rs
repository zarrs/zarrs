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

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Arc;

use num::traits::{FromBytes, ToBytes};
use serde::Deserialize;
use zarrs::array::{
    ArrayBuilder, ArrayBytes, DataType, DataTypeSize, Element, ElementError, ElementOwned,
    FillValueMetadata,
};
use zarrs::metadata::v3::MetadataV3;
use zarrs::metadata::{Configuration, Endianness};
use zarrs::storage::store::MemoryStore;
use zarrs_data_type::codec_traits::bytes::{BytesCodecEndiannessMissingError, BytesDataTypeTraits};
use zarrs_data_type::{
    DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypePluginV3, DataTypeTraits,
    FillValue,
};
use zarrs_plugin::{PluginCreateError, ZarrVersion};

/// The in-memory representation of the custom data type.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
struct CustomDataTypeFixedSizeElement {
    x: u64,
    y: f32,
}

/// Defines the conversion of an element to a fill value
impl From<CustomDataTypeFixedSizeElement> for FillValueMetadata {
    fn from(value: CustomDataTypeFixedSizeElement) -> Self {
        FillValueMetadata::from(HashMap::from([
            ("x".to_string(), FillValueMetadata::from(value.x)),
            ("y".to_string(), FillValueMetadata::from(value.y)),
        ]))
    }
}

/// The metadata is structured the same as the data type element
type CustomDataTypeFixedSizeMetadata = CustomDataTypeFixedSizeElement;

/// The padding bytes of CustomDataTypeFixedSizeElement are not serialised.
/// These are stripped as soon as the data is converted into ArrayBytes *before* it goes into the codec pipeline.
type CustomDataTypeFixedSizeBytes = [u8; size_of::<u64>() + size_of::<f32>()];

/// These defines how the CustomDataTypeFixedSizeBytes are converted TO little/big endian
/// Implementing this particular trait (from num-traits) is not necessary, but it is used in DataTypeTraitsBytesCodec/Element/ElementOwned
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
/// Implementing this particular trait (from num-traits) is not necessary, but it is used in DataTypeTraitsBytesCodec/Element/ElementOwned
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
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        data_type
            .is::<CustomDataTypeFixedSize>()
            .then_some(())
            .ok_or(ElementError::IncompatibleElementType)
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<zarrs::array::ArrayBytes<'a>, ElementError> {
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
    ) -> Result<zarrs::array::ArrayBytes<'static>, ElementError> {
        Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
    }
}

/// This defines how ArrayBytes are converted into a CustomDataTypeFixedSizeElement after decoding via the codec pipeline.
impl ElementOwned for CustomDataTypeFixedSizeElement {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
        Self::validate_data_type(data_type)?;
        let bytes = bytes.into_fixed()?;
        let bytes_len = bytes.len();
        let mut elements =
            Vec::with_capacity(bytes_len / size_of::<CustomDataTypeFixedSizeBytes>());
        for bytes in bytes
            .as_chunks::<{ size_of::<CustomDataTypeFixedSizeBytes>() }>()
            .0
        {
            elements.push(CustomDataTypeFixedSizeElement::from_ne_bytes(bytes))
        }
        Ok(elements)
    }
}

/// The data type for an array of [`CustomDataTypeFixedSizeElement`].
#[derive(Debug)]
struct CustomDataTypeFixedSize;

/// A custom name for the data type.
const CUSTOM_NAME: &str = "zarrs.test.CustomDataTypeFixedSize";

zarrs_plugin::impl_extension_aliases!(CustomDataTypeFixedSize, v3: CUSTOM_NAME);

impl zarrs_data_type::DataTypeTraitsV3 for CustomDataTypeFixedSize {
    fn create(metadata: &MetadataV3) -> Result<DataType, PluginCreateError> {
        metadata.to_typed_configuration::<zarrs_metadata::EmptyConfiguration>()?;
        Ok(Arc::new(CustomDataTypeFixedSize).into())
    }
}

// Register the data type so that it can be recognised when opening arrays.
inventory::submit! {
    DataTypePluginV3::new::<CustomDataTypeFixedSize>()
}

/// Implement the core data type extension methods
impl DataTypeTraits for CustomDataTypeFixedSize {
    fn configuration(&self, _version: ZarrVersion) -> Configuration {
        Configuration::default()
    }

    fn fill_value(
        &self,
        fill_value_metadata: &FillValueMetadata,
        _version: ZarrVersion,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        let element_metadata: CustomDataTypeFixedSizeMetadata = fill_value_metadata
            .as_custom()
            .ok_or(DataTypeFillValueMetadataError)?;
        Ok(FillValue::new(element_metadata.to_ne_bytes().to_vec()))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadata, DataTypeFillValueError> {
        let element = CustomDataTypeFixedSizeMetadata::from_ne_bytes(
            fill_value
                .as_ne_bytes()
                .try_into()
                .map_err(|_| DataTypeFillValueError)?,
        );
        Ok(FillValueMetadata::from(element))
    }

    fn size(&self) -> zarrs::array::DataTypeSize {
        DataTypeSize::Fixed(size_of::<CustomDataTypeFixedSizeBytes>())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Add support for the `bytes` codec. This must be implemented for fixed-size data types, even if they just pass-through the data type.
impl BytesDataTypeTraits for CustomDataTypeFixedSize {
    fn encode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, BytesCodecEndiannessMissingError> {
        if let Some(endianness) = endianness {
            if endianness != Endianness::native() {
                let mut bytes = bytes.into_owned();
                for bytes in bytes
                    .as_chunks_mut::<{ size_of::<CustomDataTypeFixedSizeBytes>() }>()
                    .0
                {
                    let value = CustomDataTypeFixedSizeElement::from_ne_bytes(bytes);
                    if endianness == Endianness::Little {
                        *bytes = value.to_le_bytes();
                    } else {
                        *bytes = value.to_be_bytes();
                    }
                }
                Ok(Cow::Owned(bytes))
            } else {
                Ok(bytes)
            }
        } else {
            Err(BytesCodecEndiannessMissingError)
        }
    }

    fn decode<'a>(
        &self,
        bytes: std::borrow::Cow<'a, [u8]>,
        endianness: Option<zarrs_metadata::Endianness>,
    ) -> Result<std::borrow::Cow<'a, [u8]>, BytesCodecEndiannessMissingError> {
        if let Some(endianness) = endianness {
            if endianness != Endianness::native() {
                let mut bytes = bytes.into_owned();
                for bytes in bytes
                    .as_chunks_mut::<{ size_of::<u64>() + size_of::<f32>() }>()
                    .0
                {
                    let value = if endianness == Endianness::Little {
                        CustomDataTypeFixedSizeElement::from_le_bytes(bytes)
                    } else {
                        CustomDataTypeFixedSizeElement::from_be_bytes(bytes)
                    };
                    *bytes = value.to_ne_bytes();
                }
                Ok(Cow::Owned(bytes))
            } else {
                Ok(bytes)
            }
        } else {
            Err(BytesCodecEndiannessMissingError)
        }
    }
}

// Register codec support
zarrs_data_type::register_data_type_extension_codec!(
    CustomDataTypeFixedSize,
    zarrs_data_type::codec_traits::bytes::BytesDataTypePlugin,
    zarrs_data_type::codec_traits::bytes::BytesDataTypeTraits
);

fn main() {
    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let fill_value = CustomDataTypeFixedSizeElement { x: 1, y: 2.3 };
    let array = ArrayBuilder::new(
        vec![4, 1], // array shape
        vec![2, 1], // regular chunk shape
        Arc::new(CustomDataTypeFixedSize),
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
