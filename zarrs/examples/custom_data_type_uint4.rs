//! A custom data type for `uint4`.
//!
//! It accepts uint compatible fill values.

use std::any::Any;
use std::borrow::Cow;
use std::sync::Arc;

use serde::Deserialize;
use zarrs::array::{
    ArrayBuilder, ArrayBytes, DataType, DataTypeSize, Element, ElementError, ElementOwned,
    FillValueMetadata,
};
use zarrs::metadata::Configuration;
use zarrs::metadata::v3::MetadataV3;
use zarrs::storage::store::MemoryStore;
use zarrs_data_type::codec_traits::packbits::PackBitsDataTypeTraits;
use zarrs_data_type::{
    DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypePluginV3, DataTypeTraits,
    FillValue,
};
use zarrs_plugin::{PluginCreateError, ZarrVersion};

/// A name for  the custom data type.
const UINT4: &str = "zarrs.test.uint4";

/// The data type for an array of the custom data type.
#[derive(Debug)]
struct CustomDataTypeUInt4;

zarrs_plugin::impl_extension_aliases!(CustomDataTypeUInt4, v3: UINT4);

impl zarrs_data_type::DataTypeTraitsV3 for CustomDataTypeUInt4 {
    fn create(metadata: &MetadataV3) -> Result<DataType, PluginCreateError> {
        metadata.to_typed_configuration::<zarrs_metadata::EmptyConfiguration>()?;
        Ok(Arc::new(CustomDataTypeUInt4).into())
    }
}

// Register the data type so that it can be recognised when opening arrays.
inventory::submit! {
    DataTypePluginV3::new::<CustomDataTypeUInt4>()
}

/// The in-memory representation of the custom data type.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
struct CustomDataTypeUInt4Element(u8);

/// Implement the core data type extension methods
impl DataTypeTraits for CustomDataTypeUInt4 {
    fn configuration(&self, _version: ZarrVersion) -> Configuration {
        Configuration::default()
    }

    fn fill_value(
        &self,
        fill_value_metadata: &FillValueMetadata,
        _version: ZarrVersion,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        let element_metadata: u64 = fill_value_metadata
            .as_u64()
            .ok_or(DataTypeFillValueMetadataError)?;
        let element = CustomDataTypeUInt4Element::try_from(element_metadata)
            .map_err(|_| DataTypeFillValueMetadataError)?;
        Ok(FillValue::new(element.into_ne_bytes().to_vec()))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadata, DataTypeFillValueError> {
        let element = CustomDataTypeUInt4Element::from_ne_bytes(
            fill_value
                .as_ne_bytes()
                .try_into()
                .map_err(|_| DataTypeFillValueError)?,
        );
        Ok(FillValueMetadata::from(element.into_u8()))
    }

    fn size(&self) -> zarrs::array::DataTypeSize {
        DataTypeSize::Fixed(1)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    /// Allow u8 as compatible element type.
    fn compatible_element_types(&self) -> &'static [std::any::TypeId] {
        const TYPES: [std::any::TypeId; 1] = [std::any::TypeId::of::<u8>()];
        &TYPES
    }
}

// Add support for the `bytes` codec using the helper macro (component size 1 = passthrough).
zarrs_data_type::codec_traits::impl_bytes_data_type_traits!(CustomDataTypeUInt4, 1);

/// Add support for the `packbits` codec.
impl PackBitsDataTypeTraits for CustomDataTypeUInt4 {
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

// Register packbits codec support
zarrs_data_type::register_data_type_extension_codec!(
    CustomDataTypeUInt4,
    zarrs_data_type::codec_traits::packbits::PackBitsDataTypePlugin,
    zarrs_data_type::codec_traits::packbits::PackBitsDataTypeTraits
);

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
    fn into_ne_bytes(self) -> [u8; 1] {
        [self.0]
    }

    fn from_ne_bytes(bytes: [u8; 1]) -> Self {
        Self(bytes[0])
    }

    fn into_u8(self) -> u8 {
        self.0
    }
}

/// This defines how an in-memory CustomDataTypeUInt4Element is converted into ArrayBytes before encoding via the codec pipeline.
impl Element for CustomDataTypeUInt4Element {
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        // Check if the data type matches our custom data type
        data_type
            .is::<CustomDataTypeUInt4>()
            .then_some(())
            .ok_or(ElementError::IncompatibleElementType)
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<zarrs::array::ArrayBytes<'a>, ElementError> {
        Self::validate_data_type(data_type)?;
        let mut bytes: Vec<u8> =
            Vec::with_capacity(std::mem::size_of_val(elements));
        for element in elements {
            bytes.push(element.0);
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

/// This defines how ArrayBytes are converted into a CustomDataTypeUInt4Element after decoding via the codec pipeline.
impl ElementOwned for CustomDataTypeUInt4Element {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
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
        Arc::new(CustomDataTypeUInt4),
        FillValue::new(fill_value.into_ne_bytes().to_vec()),
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
        println!("uint4: {:08b} u8: {}", f.into_u8(), f.into_u8());
    }

    assert_eq!(data[0], CustomDataTypeUInt4Element::try_from(1).unwrap());
    assert_eq!(data[1], CustomDataTypeUInt4Element::try_from(2).unwrap());
    assert_eq!(data[2], CustomDataTypeUInt4Element::try_from(3).unwrap());
    assert_eq!(data[3], CustomDataTypeUInt4Element::try_from(4).unwrap());
    assert_eq!(data[4], CustomDataTypeUInt4Element::try_from(5).unwrap());
    assert_eq!(data[5], CustomDataTypeUInt4Element::try_from(15).unwrap());

    let data: Vec<CustomDataTypeUInt4Element> = array.retrieve_array_subset(&[1..3, 0..1]).unwrap();
    assert_eq!(data[0], CustomDataTypeUInt4Element::try_from(2).unwrap());
    assert_eq!(data[1], CustomDataTypeUInt4Element::try_from(3).unwrap());
}

#[test]
fn custom_data_type_uint4() {
    main()
}
