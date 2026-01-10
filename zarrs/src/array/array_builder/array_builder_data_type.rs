use std::sync::Arc;

use zarrs_data_type::DataTypeTraits;

use crate::array::{ArrayCreateError, DataType};
use crate::metadata::v3::MetadataV3;

/// An input that can be mapped to a data type.
#[derive(Debug, Clone)]
pub struct ArrayBuilderDataType(ArrayBuilderDataTypeImpl);

impl PartialEq for ArrayBuilderDataType {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

#[derive(Debug, Clone)]
enum ArrayBuilderDataTypeImpl {
    DataType(DataType),
    Metadata(MetadataV3),
    MetadataString(String),
}

impl PartialEq for ArrayBuilderDataTypeImpl {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::DataType(a), Self::DataType(b)) => a.eq(b.as_ref()),
            (Self::Metadata(a), Self::Metadata(b)) => a == b,
            (Self::MetadataString(a), Self::MetadataString(b)) => a == b,
            _ => false,
        }
    }
}

impl ArrayBuilderDataType {
    pub(crate) fn to_data_type(&self) -> Result<DataType, ArrayCreateError> {
        match &self.0 {
            ArrayBuilderDataTypeImpl::DataType(data_type) => Ok(data_type.clone()),
            ArrayBuilderDataTypeImpl::Metadata(metadata) => {
                DataType::from_metadata(metadata).map_err(ArrayCreateError::DataTypeCreateError)
            }
            ArrayBuilderDataTypeImpl::MetadataString(metadata) => {
                // assume the metadata corresponds to a "name" if it cannot be parsed as MetadataV3
                // this makes "float32" work for example, where normally r#""float32""# would be required
                let metadata =
                    MetadataV3::try_from(metadata.as_str()).unwrap_or(MetadataV3::new(metadata));
                DataType::from_metadata(&metadata).map_err(ArrayCreateError::DataTypeCreateError)
            }
        }
    }
}

impl From<DataType> for ArrayBuilderDataType {
    fn from(value: DataType) -> Self {
        Self(ArrayBuilderDataTypeImpl::DataType(value))
    }
}

impl<T: DataTypeTraits + 'static> From<Arc<T>> for ArrayBuilderDataType {
    fn from(value: Arc<T>) -> Self {
        Self(ArrayBuilderDataTypeImpl::DataType(value.clone().into()))
    }
}

impl From<MetadataV3> for ArrayBuilderDataType {
    fn from(value: MetadataV3) -> Self {
        Self(ArrayBuilderDataTypeImpl::Metadata(value))
    }
}

impl From<String> for ArrayBuilderDataType {
    fn from(value: String) -> Self {
        Self(ArrayBuilderDataTypeImpl::MetadataString(value))
    }
}

impl From<&str> for ArrayBuilderDataType {
    fn from(value: &str) -> Self {
        Self(ArrayBuilderDataTypeImpl::MetadataString(value.to_string()))
    }
}
