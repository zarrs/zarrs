use derive_more::{derive::From, Display};
use serde::{Deserialize, Serialize, Serializer};
use thiserror::Error;

use crate::{v2::MetadataV2, ArrayShape, ChunkKeySeparator, ChunkShape, Endianness};

/// Zarr V2 array metadata.
///
/// An example `JSON` document for a Zarr V2 array:
/// ```json
/// {
///     "chunks": [
///         1000,
///         1000
///     ],
///     "compressor": {
///         "id": "blosc",
///         "cname": "lz4",
///         "clevel": 5,
///         "shuffle": 1
///     },
///     "dtype": "<f8",
///     "fill_value": "NaN",
///     "filters": [
///         {"id": "delta", "dtype": "<f8", "astype": "<f4"}
///     ],
///     "order": "C",
///     "shape": [
///         10000,
///         10000
///     ],
///     "zarr_format": 2
/// }
/// ```
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, Display)]
#[serde(tag = "node_type", rename = "array")]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct ArrayMetadataV2 {
    /// An integer defining the version of the storage specification to which the array adheres. Must be `2`.
    pub zarr_format: monostate::MustBe!(2u64),
    /// An array of integers providing the length of each dimension of the Zarr array.
    pub shape: ArrayShape,
    /// A list of integers defining the length of each dimension of a chunk of the array.
    pub chunks: ChunkShape,
    /// The data type of the Zarr array.
    pub dtype: DataTypeMetadataV2,
    /// A JSON object identifying the primary compression codec and providing configuration parameters, or null if no compressor is to be used.
    pub compressor: Option<MetadataV2>,
    /// A scalar value providing the default value to use for uninitialized portions of the array, or null if no fill value is to be used.
    pub fill_value: FillValueMetadataV2,
    /// Either “C” or “F”, defining the layout of bytes within each chunk of the array.
    pub order: ArrayMetadataV2Order,
    /// A list of JSON objects providing codec configurations, or null if no filters are to be applied.
    #[serde(default, serialize_with = "serialize_v2_filters")]
    pub filters: Option<Vec<MetadataV2>>,
    /// If present, either the string "." or "/" defining the separator placed between the dimensions of a chunk.
    #[serde(default = "chunk_key_separator_default_zarr_v2")]
    pub dimension_separator: ChunkKeySeparator,
    /// Optional user defined attributes contained in a separate `.zattrs` file.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub attributes: serde_json::Map<String, serde_json::Value>,
}

#[allow(clippy::ref_option)]
fn serialize_v2_filters<S>(
    filters: &Option<Vec<MetadataV2>>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let filters = filters.as_ref().filter(|v| !v.is_empty());
    match filters {
        Some(filters) => serializer.collect_seq(filters),
        None => serializer.serialize_none(),
    }
}

impl ArrayMetadataV2 {
    /// Create Zarr V2 array metadata.
    ///
    /// Defaults to:
    /// - C order,
    /// - empty attributes, and
    /// - no additional fields.
    #[must_use]
    pub fn new(
        shape: ArrayShape,
        chunks: ChunkShape,
        dtype: DataTypeMetadataV2,
        fill_value: FillValueMetadataV2,
        compressor: Option<MetadataV2>,
        filters: Option<Vec<MetadataV2>>,
    ) -> Self {
        let filters = filters.filter(|v| !v.is_empty());
        Self {
            zarr_format: monostate::MustBe!(2u64),
            shape,
            chunks,
            dtype,
            compressor,
            fill_value,
            order: ArrayMetadataV2Order::C,
            filters,
            dimension_separator: ChunkKeySeparator::Dot,
            attributes: serde_json::Map::default(),
        }
    }

    /// Serialize the metadata as a pretty-printed String of JSON.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn to_string_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("array metadata is valid JSON")
    }

    /// Set the dimension separator.
    #[must_use]
    pub fn with_dimension_separator(mut self, dimension_separator: ChunkKeySeparator) -> Self {
        self.dimension_separator = dimension_separator;
        self
    }

    /// Set the order.
    #[must_use]
    pub fn with_order(mut self, order: ArrayMetadataV2Order) -> Self {
        self.order = order;
        self
    }

    /// Set the user attributes.
    #[must_use]
    pub fn with_attributes(
        mut self,
        attributes: serde_json::Map<String, serde_json::Value>,
    ) -> Self {
        self.attributes = attributes;
        self
    }
}

const fn chunk_key_separator_default_zarr_v2() -> ChunkKeySeparator {
    ChunkKeySeparator::Dot
}

/// Zarr V2 structured data type metadata.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug)]
#[serde(
    from = "DataTypeMetadataV2StructuredTuple",
    into = "DataTypeMetadataV2StructuredTuple"
)]
pub struct DataTypeMetadataV2Structured {
    /// Field name.
    fieldname: String,
    /// Data type.
    datatype: String,
    /// Subarray shape.
    shape: Option<Vec<u64>>,
}

#[derive(Serialize, Deserialize)]
struct DataTypeMetadataV2StructuredTuple(
    String,
    String,
    #[serde(skip_serializing_if = "Option::is_none")] Option<Vec<u64>>,
);

impl From<DataTypeMetadataV2StructuredTuple> for DataTypeMetadataV2Structured {
    fn from(value: DataTypeMetadataV2StructuredTuple) -> Self {
        let DataTypeMetadataV2StructuredTuple(fieldname, datatype, shape) = value;
        Self {
            fieldname,
            datatype,
            shape,
        }
    }
}

impl From<DataTypeMetadataV2Structured> for DataTypeMetadataV2StructuredTuple {
    fn from(value: DataTypeMetadataV2Structured) -> Self {
        Self(value.fieldname, value.datatype, value.shape)
    }
}

/// Zarr V2 data type metadata.
#[derive(Serialize, Deserialize, Clone, PartialEq, Eq, Debug, From)]
#[serde(untagged)]
pub enum DataTypeMetadataV2 {
    /// A simple data type.
    #[from(String, &str)]
    Simple(String),
    /// A structured data type.
    Structured(Vec<DataTypeMetadataV2Structured>),
}

/// A Zarr V2 invalid data type endianness error.
#[derive(Clone, Debug, Error)]
#[error("invalid V2 data type for {_0:?} endianness, must begin with |, < or >")]
pub struct DataTypeMetadataV2EndiannessError(DataTypeMetadataV2);

/// Get the endianness of a Zarr V2 data type.
///
/// # Errors
/// Returns a [`DataTypeMetadataV2EndiannessError`] if the data type is not supported or the endianness prefix is invalid.
pub fn data_type_metadata_v2_to_endianness(
    data_type: &DataTypeMetadataV2,
) -> Result<Option<Endianness>, DataTypeMetadataV2EndiannessError> {
    match data_type {
        DataTypeMetadataV2::Simple(data_type_str) => {
            if data_type_str.starts_with('|') {
                Ok(None)
            } else if data_type_str.starts_with('<') {
                Ok(Some(Endianness::Little))
            } else if data_type_str.starts_with('>') {
                Ok(Some(Endianness::Big))
            } else {
                Err(DataTypeMetadataV2EndiannessError(data_type.clone()))
            }
        }
        DataTypeMetadataV2::Structured(_) => {
            Err(DataTypeMetadataV2EndiannessError(data_type.clone()))
        }
    }
}

/// Zarr V2 fill value metadata.
pub type FillValueMetadataV2 = crate::v3::FillValueMetadataV3;

/// Zarr V2 order metadata. Indicates the layout of bytes within a chunk.
#[derive(Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArrayMetadataV2Order {
    /// Row-major order. The last dimension varies fastest.
    C,
    /// Column-major order. The first dimension varies fastest.
    F,
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;

    #[test]
    fn empty_filters() {
        let array = ArrayMetadataV2 {
            zarr_format: monostate::MustBe!(2u64),
            shape: vec![10000, 10000],
            chunks: vec![NonZeroU64::new(1000).unwrap(); 2].try_into().unwrap(),
            dtype: DataTypeMetadataV2::Simple("<f8".to_string()),
            compressor: None,
            fill_value: FillValueMetadataV2::String("NaN".to_string()),
            order: ArrayMetadataV2Order::C,
            filters: Some(vec![]),
            dimension_separator: ChunkKeySeparator::Dot,
            attributes: serde_json::Map::new(),
        };
        let serialized = serde_json::to_string(&array).unwrap();
        assert_eq!(
            serialized,
            r#"{"node_type":"array","zarr_format":2,"shape":[10000,10000],"chunks":[1000,1000],"dtype":"<f8","compressor":null,"fill_value":"NaN","order":"C","filters":null,"dimension_separator":"."}"#
        );
    }
}
