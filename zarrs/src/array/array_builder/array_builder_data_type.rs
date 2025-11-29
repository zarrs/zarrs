use crate::metadata::v3::MetadataV3;
use crate::{
    array::{ArrayCreateError, DataType},
    config::global_config,
};

/// An input that can be mapped to a data type.
#[derive(Debug, PartialEq, Clone)]
pub struct ArrayBuilderDataType(ArrayBuilderDataTypeImpl);

#[derive(Debug, PartialEq, Clone)]
enum ArrayBuilderDataTypeImpl {
    DataType(DataType),
    Metadata(MetadataV3),
    MetadataString(String),
}

impl ArrayBuilderDataType {
    pub(crate) fn to_data_type(&self) -> Result<DataType, ArrayCreateError> {
        match &self.0 {
            ArrayBuilderDataTypeImpl::DataType(data_type) => Ok(data_type.clone()),
            ArrayBuilderDataTypeImpl::Metadata(metadata) => {
                DataType::from_metadata(metadata, global_config().data_type_aliases_v3())
                    .map_err(ArrayCreateError::DataTypeCreateError)
            }
            ArrayBuilderDataTypeImpl::MetadataString(metadata) => {
                // assume the metadata corresponds to a "name" if it cannot be parsed as MetadataV3
                // this makes "float32" work for example, where normally r#""float32""# would be required
                let metadata =
                    MetadataV3::try_from(metadata.as_str()).unwrap_or(MetadataV3::new(metadata));
                DataType::from_metadata(&metadata, global_config().data_type_aliases_v3())
                    .map_err(ArrayCreateError::DataTypeCreateError)
            }
        }
    }
}

impl From<DataType> for ArrayBuilderDataType {
    fn from(value: DataType) -> Self {
        Self(ArrayBuilderDataTypeImpl::DataType(value))
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
