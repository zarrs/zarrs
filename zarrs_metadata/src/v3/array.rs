use derive_more::Display;
use serde::{Deserialize, Serialize};

use super::AdditionalFieldsV3;
use crate::array::IntoDimensionName;
use crate::v3::MetadataV3;
use crate::{ArrayShape, ChunkKeySeparator, DimensionName};

mod fill_value;
mod nan_representations;

pub use fill_value::FillValueMetadataV3;
pub use nan_representations::{ZARR_NAN_BF16, ZARR_NAN_F16, ZARR_NAN_F32, ZARR_NAN_F64};

/// Zarr V3 array metadata.
///
/// An example `JSON` document for a Zarr V3 array:
/// ```json
/// {
///     "zarr_format": 3,
///     "node_type": "array",
///     "shape": [10000, 1000],
///     "dimension_names": ["rows", "columns"],
///     "data_type": "float64",
///     "chunk_grid": {
///         "name": "regular",
///         "configuration": {
///             "chunk_shape": [1000, 100]
///         }
///     },
///     "chunk_key_encoding": {
///         "name": "default",
///         "configuration": {
///             "separator": "/"
///         }
///     },
///     "codecs": [{
///         "name": "gzip",
///         "configuration": {
///             "level": 1
///         }
///     }],
///     "fill_value": "NaN",
///     "attributes": {
///         "foo": 42,
///         "bar": "apples",
///         "baz": [1, 2, 3, 4]
///     }
/// }
/// ```
#[non_exhaustive]
#[allow(clippy::unsafe_derive_deserialize)]
#[derive(Serialize, Deserialize, Clone, PartialEq, Debug, Display)]
#[display("{}", serde_json::to_string(self).unwrap_or_default())]
pub struct ArrayMetadataV3 {
    /// An integer defining the version of the storage specification to which the array adheres. Must be `3`.
    pub zarr_format: monostate::MustBe!(3u64),
    /// A string defining the type of hierarchy node element, must be `array` here.
    pub node_type: monostate::MustBe!("array"),
    /// An array of integers providing the length of each dimension of the Zarr array.
    pub shape: ArrayShape,
    /// The data type of the Zarr array.
    pub data_type: MetadataV3,
    /// The chunk grid of the Zarr array.
    pub chunk_grid: MetadataV3,
    /// The mapping from chunk grid cell coordinates to keys in the underlying store.
    pub chunk_key_encoding: MetadataV3,
    /// Provides an element value to use for uninitialised portions of the Zarr array.
    ///
    /// Suitable values are dependent on the data type.
    ///
    /// Boolean.
    /// *The value must be a JSON boolean (false or true).*
    ///
    /// Signed integers (`int{8,16,32,64}`) or unsigned integers (`uint{8,16,32,64}`).
    /// *The value must be a JSON number with no fraction or exponent part that is within the representable range of the data type.*
    ///
    /// Floating point numbers (`float{16,32,64}`, `bfloat16`).
    ///  - *A JSON number, that will be rounded to the nearest representable value.*
    ///  - *A JSON string of the form:*
    ///  - *`"Infinity"`, denoting positive infinity;*
    ///  - *`"-Infinity"`, denoting negative infinity;*
    ///  - *`"NaN"`, denoting thenot-a-number (NaN) value where the sign bit is 0 (positive), the most significant bit (MSB) of the mantissa is 1, and all other bits of the mantissa are zero;*
    ///  - *`"0xYYYYYYYY"`, specifying the byte representation of the floating point number as an unsigned integer.
    ///
    /// Complex numbers (`complex{64,128}`)
    /// *The value must be a two-element array, specifying the real and imaginary components respectively, where each component is specified as defined above for floating point number.*
    ///
    /// Raw data types (`r<N>`)
    /// *An array of integers, with length equal to `<N>`, where each integer is in the range `[0, 255]`.*
    pub fill_value: FillValueMetadataV3,
    /// Specifies a list of codecs to be used for encoding and decoding chunks.
    pub codecs: Vec<MetadataV3>,
    /// Optional user defined attributes.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub attributes: serde_json::Map<String, serde_json::Value>,
    /// An optional list of storage transformers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub storage_transformers: Vec<MetadataV3>,
    /// An optional list of dimension names.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimension_names: Option<Vec<DimensionName>>,
    // /// Extension definitions (Zarr 3.2, [ZEP0010](https://github.com/zarr-developers/zeps/pull/67)).
    // #[serde(default, skip_serializing_if = "Vec::is_empty")]
    // pub extensions: Vec<MetadataV3>,
    /// Additional fields.
    #[serde(flatten)]
    pub additional_fields: AdditionalFieldsV3,
}

impl ArrayMetadataV3 {
    /// Create new Zarr V3 array metadata.
    ///
    /// Defaults to:
    /// - `default` chunk key encoding with the '/' separator,
    /// - empty attributes,
    /// - no dimension names,
    /// - no storage transformers,
    /// - no extensions, and
    /// - no additional fields.
    #[must_use]
    pub fn new(
        shape: ArrayShape,
        chunk_grid: MetadataV3,
        data_type: MetadataV3,
        fill_value: FillValueMetadataV3,
        codecs: Vec<MetadataV3>,
    ) -> Self {
        #[derive(Serialize)]
        struct DefaultChunkKeyEncodingConfiguration {
            pub separator: ChunkKeySeparator,
        }

        let chunk_key_encoding = unsafe {
            // SAFETY: The default chunk key encoding configuration is valid JSON.
            MetadataV3::new_with_serializable_configuration(
                "default".to_string(),
                &DefaultChunkKeyEncodingConfiguration {
                    separator: crate::ChunkKeySeparator::Slash,
                },
            )
            .unwrap_unchecked()
        };

        Self {
            zarr_format: monostate::MustBe!(3u64),
            node_type: monostate::MustBe!("array"),
            shape,
            data_type,
            chunk_grid,
            chunk_key_encoding,
            fill_value,
            codecs,
            attributes: serde_json::Map::default(),
            storage_transformers: Vec::default(),
            dimension_names: None,
            additional_fields: AdditionalFieldsV3::default(),
            // extensions: Vec::default(),
        }
    }

    /// Serialize the metadata as a pretty-printed String of JSON.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn to_string_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("array metadata is valid JSON")
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

    /// Set the additional fields.
    #[must_use]
    pub fn with_additional_fields(mut self, additional_fields: AdditionalFieldsV3) -> Self {
        self.additional_fields = additional_fields;
        self
    }

    /// Set the chunk key encoding.
    #[must_use]
    pub fn with_chunk_key_encoding(mut self, chunk_key_encoding: MetadataV3) -> Self {
        self.chunk_key_encoding = chunk_key_encoding;
        self
    }

    /// Set the dimension names.
    #[must_use]
    pub fn with_dimension_names<I, D>(mut self, dimension_names: Option<I>) -> Self
    where
        I: IntoIterator<Item = D>,
        D: IntoDimensionName,
    {
        if let Some(dimension_names) = dimension_names {
            self.dimension_names = Some(
                dimension_names
                    .into_iter()
                    .map(IntoDimensionName::into_dimension_name)
                    .collect(),
            );
        } else {
            self.dimension_names = None;
        }
        self
    }

    /// Set the storage transformers.
    #[must_use]
    pub fn with_storage_transformers(mut self, storage_transformers: Vec<MetadataV3>) -> Self {
        self.storage_transformers = storage_transformers;
        self
    }

    // /// Set the extension definitions.
    // #[must_use]
    // pub fn with_extensions(mut self, extensions: Vec<MetadataV3>) -> Self {
    //     self.extensions = extensions;
    //     self
    // }
}
