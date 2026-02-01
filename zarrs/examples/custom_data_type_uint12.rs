//! A custom data type for `uint12`.
//!
//! It accepts uint compatible fill values.

use std::borrow::Cow;
use std::sync::Arc;

use serde::Deserialize;
use zarrs::array::{
    ArrayBuilder, ArrayBytes, DataType, DataTypeSize, Element, ElementError, ElementOwned,
};
use zarrs::metadata::v3::MetadataV3;
use zarrs::metadata::{Configuration, FillValueMetadata};
use zarrs::storage::store::MemoryStore;
use zarrs_data_type::codec_traits::packbits::PackBitsDataTypeTraits;
use zarrs_data_type::{
    DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypePluginV3, DataTypeTraits,
    FillValue,
};
use zarrs_plugin::{PluginCreateError, ZarrVersion};

/// A name for  the custom data type.
const UINT12: &str = "zarrs.test.uint12";

/// The data type for an array of the custom data type.
#[derive(Debug)]
struct CustomDataTypeUInt12;

zarrs_plugin::impl_extension_aliases!(CustomDataTypeUInt12, v3: UINT12);

impl zarrs_data_type::DataTypeTraitsV3 for CustomDataTypeUInt12 {
    fn create(metadata: &MetadataV3) -> Result<DataType, PluginCreateError> {
        metadata.to_typed_configuration::<zarrs_metadata::EmptyConfiguration>()?;
        Ok(Arc::new(CustomDataTypeUInt12).into())
    }
}

// Register the data type so that it can be recognised when opening arrays.
inventory::submit! {
    DataTypePluginV3::new::<CustomDataTypeUInt12>()
}

/// The in-memory representation of the custom data type.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
struct CustomDataTypeUInt12Element(u16);

/// Implement the core data type extension methods
impl DataTypeTraits for CustomDataTypeUInt12 {
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
        let element = CustomDataTypeUInt12Element::try_from(element_metadata)
            .map_err(|_| DataTypeFillValueMetadataError)?;
        Ok(FillValue::new(element.to_le_bytes().to_vec()))
    }

    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadata, DataTypeFillValueError> {
        let element = CustomDataTypeUInt12Element::from_le_bytes(
            fill_value
                .as_ne_bytes()
                .try_into()
                .map_err(|_| DataTypeFillValueError)?,
        );
        Ok(FillValueMetadata::from(element.as_u16()))
    }

    fn size(&self) -> zarrs::array::DataTypeSize {
        DataTypeSize::Fixed(2)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    /// Allow u16 as compatible element type.
    fn compatible_element_types(&self) -> &'static [std::any::TypeId] {
        const TYPES: [std::any::TypeId; 1] = [std::any::TypeId::of::<u16>()];
        &TYPES
    }
}

// Add support for the `bytes` codec using the helper macro (component size 1 = passthrough).
zarrs_data_type::codec_traits::impl_bytes_data_type_traits!(CustomDataTypeUInt12, 1);

/// Add support for the `packbits` codec.
impl PackBitsDataTypeTraits for CustomDataTypeUInt12 {
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

// Register packbits codec support
zarrs_data_type::register_data_type_extension_codec!(
    CustomDataTypeUInt12,
    zarrs_data_type::codec_traits::packbits::PackBitsDataTypePlugin,
    zarrs_data_type::codec_traits::packbits::PackBitsDataTypeTraits
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
    fn validate_data_type(data_type: &DataType) -> Result<(), ElementError> {
        // Check if the data type matches our custom data type
        data_type
            .is::<CustomDataTypeUInt12>()
            .then_some(())
            .ok_or(ElementError::IncompatibleElementType)
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<zarrs::array::ArrayBytes<'a>, ElementError> {
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
    ) -> Result<zarrs::array::ArrayBytes<'static>, ElementError> {
        Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
    }
}

/// This defines how ArrayBytes are converted into a CustomDataTypeUInt12Element after decoding via the codec pipeline.
impl ElementOwned for CustomDataTypeUInt12Element {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ElementError> {
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
        Arc::new(CustomDataTypeUInt12),
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
            .retrieve_array_subset(&[(i as u64)..i as u64 + 1, 0..1])
            .unwrap();
        assert_eq!(element_pd[0], element);
    }
}

#[test]
fn custom_data_type_uint12() {
    main()
}
