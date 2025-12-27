#![allow(missing_docs)]

use std::borrow::Cow;
use std::sync::Arc;

use derive_more::Deref;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use zarrs::array::{
    ArrayBuilder, ArrayBytes, ArrayBytesOffsets, ArrayError, DataType, DataTypeSize, Element,
    ElementOwned, FillValueMetadataV3,
};
use zarrs::metadata::{Configuration, v3::MetadataV3};
use zarrs::storage::store::MemoryStore;
use zarrs_data_type::{
    DataTypeExtension, DataTypeFillValueError, DataTypeFillValueMetadataError, DataTypePlugin,
    FillValue,
};
use zarrs_plugin::{PluginCreateError, PluginMetadataInvalidError};

#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize, Deref)]
struct CustomDataTypeVariableSizeElement(Option<f32>);

impl From<Option<f32>> for CustomDataTypeVariableSizeElement {
    fn from(value: Option<f32>) -> Self {
        Self(value)
    }
}

impl Element for CustomDataTypeVariableSizeElement {
    fn validate_data_type(data_type: &DataType) -> Result<(), ArrayError> {
        (data_type == &DataType::Extension(Arc::new(CustomDataTypeVariableSize)))
            .then_some(())
            .ok_or(ArrayError::IncompatibleElementType)
    }

    fn to_array_bytes<'a>(
        data_type: &DataType,
        elements: &'a [Self],
    ) -> Result<zarrs::array::ArrayBytes<'a>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let mut bytes = Vec::new();
        let mut offsets = Vec::with_capacity(elements.len() + 1);

        for element in elements {
            offsets.push(bytes.len());
            if let Some(value) = element.0 {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        offsets.push(bytes.len());
        let offsets = unsafe {
            // SAFETY: Constructed correctly above
            ArrayBytesOffsets::new_unchecked(offsets)
        };
        unsafe { Ok(ArrayBytes::new_vlen_unchecked(bytes, offsets)) }
    }

    fn into_array_bytes(
        data_type: &DataType,
        elements: Vec<Self>,
    ) -> Result<zarrs::array::ArrayBytes<'static>, ArrayError> {
        Ok(Self::to_array_bytes(data_type, &elements)?.into_owned())
    }
}

impl ElementOwned for CustomDataTypeVariableSizeElement {
    fn from_array_bytes(
        data_type: &DataType,
        bytes: ArrayBytes<'_>,
    ) -> Result<Vec<Self>, ArrayError> {
        Self::validate_data_type(data_type)?;
        let (bytes, offsets) = bytes.into_variable()?.into_parts();

        let mut elements = Vec::with_capacity(offsets.len().saturating_sub(1));
        for (curr, next) in offsets.iter().tuple_windows() {
            let bytes = &bytes[*curr..*next];
            if let Ok(bytes) = <[u8; 4]>::try_from(bytes) {
                let value = f32::from_le_bytes(bytes);
                elements.push(CustomDataTypeVariableSizeElement(Some(value)));
            } else if bytes.len() == 0 {
                elements.push(CustomDataTypeVariableSizeElement(None));
            } else {
                panic!()
            }
        }

        Ok(elements)
    }
}

/// The data type for an array of [`CustomDataTypeVariableSizeElement`].
#[derive(Debug)]
struct CustomDataTypeVariableSize;

const CUSTOM_NAME: &'static str = "zarrs.test.CustomDataTypeVariableSize";

fn matches_name_custom(name: &str, _version: zarrs_plugin::ZarrVersions) -> bool {
    name == CUSTOM_NAME
}

fn default_name_custom(_version: zarrs_plugin::ZarrVersions) -> Cow<'static, str> {
    CUSTOM_NAME.into()
}

fn create_custom_dtype(
    metadata: &MetadataV3,
) -> Result<Arc<dyn DataTypeExtension>, PluginCreateError> {
    if metadata.configuration_is_none_or_empty() {
        Ok(Arc::new(CustomDataTypeVariableSize))
    } else {
        Err(PluginMetadataInvalidError::new(CUSTOM_NAME, "codec", metadata.to_string()).into())
    }
}

inventory::submit! {
    DataTypePlugin::new(CUSTOM_NAME, matches_name_custom, default_name_custom, create_custom_dtype)
}

impl DataTypeExtension for CustomDataTypeVariableSize {
    fn identifier(&self) -> &'static str {
        CUSTOM_NAME
    }

    fn configuration(&self) -> Configuration {
        Configuration::default()
    }

    fn fill_value(
        &self,
        fill_value_metadata: &FillValueMetadataV3,
    ) -> Result<FillValue, DataTypeFillValueMetadataError> {
        if let Some(f) = fill_value_metadata.as_f32() {
            Ok(FillValue::new(f.to_ne_bytes().to_vec()))
        } else if fill_value_metadata.is_null() {
            Ok(FillValue::new(vec![]))
        } else if let Some(bytes) = fill_value_metadata.as_bytes() {
            Ok(FillValue::new(bytes))
        } else {
            Err(DataTypeFillValueMetadataError::new(
                self.identifier().to_string(),
                fill_value_metadata.clone(),
            ))
        }
    }

    fn metadata_fill_value(
        &self,
        fill_value: &FillValue,
    ) -> Result<FillValueMetadataV3, DataTypeFillValueError> {
        let fill_value = fill_value.as_ne_bytes();
        if fill_value.len() == 0 {
            Ok(FillValueMetadataV3::Null)
        } else if fill_value.len() == 4 {
            let value = f32::from_ne_bytes(fill_value.try_into().unwrap());
            Ok(FillValueMetadataV3::from(value))
        } else {
            Err(DataTypeFillValueError::new(
                self.identifier().to_string(),
                fill_value.into(),
            ))
        }
    }

    fn size(&self) -> zarrs::array::DataTypeSize {
        DataTypeSize::Variable
    }
}

fn main() {
    let store = std::sync::Arc::new(MemoryStore::default());
    let array_path = "/array";
    let array = ArrayBuilder::new(
        vec![4, 1], // array shape
        vec![3, 1], // regular chunk shape
        DataType::Extension(Arc::new(CustomDataTypeVariableSize)),
        [],
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
        CustomDataTypeVariableSizeElement::from(Some(1.0)),
        CustomDataTypeVariableSizeElement::from(None),
        CustomDataTypeVariableSizeElement::from(Some(3.0)),
    ];
    array.store_chunk(&[0, 0], &data).unwrap();

    let data: Vec<CustomDataTypeVariableSizeElement> =
        array.retrieve_array_subset(&array.subset_all()).unwrap();

    assert_eq!(data[0], CustomDataTypeVariableSizeElement::from(Some(1.0)));
    assert_eq!(data[1], CustomDataTypeVariableSizeElement::from(None));
    assert_eq!(data[2], CustomDataTypeVariableSizeElement::from(Some(3.0)));
    assert_eq!(data[3], CustomDataTypeVariableSizeElement::from(None));

    println!("{data:#?}");
}

#[test]
fn custom_data_type_variable_size() {
    main()
}
