//! [Zarr](https://zarr-specs.readthedocs.io/) metadata support for the [`zarrs`](https://docs.rs/zarrs/latest/zarrs/index.html) crate.
//!
//! This crate supports serialisation and deserialisation of Zarr V2 and V3 core metadata.
//!
//! [`ArrayMetadata`] and [`GroupMetadata`] can represent any conformant Zarr array/group metadata.
//! The [`zarrs_metadata_ext`](https://docs.rs/zarrs/latest/zarrs_metadata_ext/) crate supports the serialisation and deserialisation of known Zarr extension point metadata into concrete structures.
//!
//! ## Licence
//! `zarrs_metadata` is licensed under either of
//!  - the Apache License, Version 2.0 [LICENSE-APACHE](https://docs.rs/crate/zarrs_metadata/latest/source/LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
//!  - the MIT license [LICENSE-MIT](https://docs.rs/crate/zarrs_metadata/latest/source/LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.
//!
//! Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

use derive_more::derive::{Display, From};
use derive_more::{Deref, Into};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

mod array;

/// An array shape. Dimensions may be zero.
pub type ArrayShape = Vec<u64>;

/// A chunk shape. Dimensions must be non-zero.
pub type ChunkShape = Vec<std::num::NonZeroU64>;

/// Zarr V3 metadata.
pub mod v3;

/// Zarr V2 metadata.
pub mod v2;

pub use array::{ChunkKeySeparator, DimensionName, Endianness, IntoDimensionName};
use thiserror::Error;

/// Zarr array metadata (V2 or V3).
#[derive(Deserialize, Serialize, Clone, PartialEq, Debug, Display, From)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum ArrayMetadata {
    /// Zarr Version 3.
    V3(v3::ArrayMetadataV3),
    /// Zarr Version 2.
    V2(v2::ArrayMetadataV2),
}

impl ArrayMetadata {
    /// Serialize the metadata as a pretty-printed String of JSON.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn to_string_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("array metadata is valid JSON")
    }
}

impl TryFrom<&str> for ArrayMetadata {
    type Error = serde_json::Error;
    fn try_from(metadata_json: &str) -> Result<Self, Self::Error> {
        serde_json::from_str::<Self>(metadata_json)
    }
}

/// Zarr group metadata (V2 or V3).
#[derive(Serialize, Deserialize, Clone, Eq, PartialEq, Debug, Display, From)]
#[serde(untagged)]
pub enum GroupMetadata {
    /// Zarr Version 3.
    V3(v3::GroupMetadataV3),
    /// Zarr Version 2.
    V2(v2::GroupMetadataV2),
}

impl GroupMetadata {
    /// Serialize the metadata as a pretty-printed String of JSON.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn to_string_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("group metadata is valid JSON")
    }
}

impl TryFrom<&str> for GroupMetadata {
    type Error = serde_json::Error;
    fn try_from(metadata_json: &str) -> Result<Self, Self::Error> {
        serde_json::from_str::<Self>(metadata_json)
    }
}

/// Zarr node metadata ([`ArrayMetadata`] or [`GroupMetadata`]).
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(untagged)]
#[allow(clippy::large_enum_variant)]
pub enum NodeMetadata {
    /// Array metadata.
    Array(ArrayMetadata),

    /// Group metadata.
    Group(GroupMetadata),
}

impl NodeMetadata {
    /// Serialize the metadata as a pretty-printed String of JSON.
    #[allow(clippy::missing_panics_doc)]
    #[must_use]
    pub fn to_string_pretty(&self) -> String {
        serde_json::to_string_pretty(self).expect("node metadata is valid JSON")
    }
}

/// A data type size. Fixed or variable.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DataTypeSize {
    /// Fixed size (in bytes).
    Fixed(usize),
    /// Variable sized.
    ///
    /// <https://github.com/zarr-developers/zeps/pull/47>
    Variable,
}

/// Configuration metadata.
#[derive(Default, Serialize, Deserialize, Debug, Clone, Deref, From, Into, Eq, PartialEq)]
pub struct Configuration(serde_json::Map<String, serde_json::Value>);

impl<T: ConfigurationSerialize> From<T> for Configuration {
    fn from(value: T) -> Self {
        match serde_json::to_value(value) {
            Ok(serde_json::Value::Object(configuration)) => configuration.into(),
            _ => {
                panic!("the configuration could not be converted to a JSON object")
            }
        }
    }
}

/// A trait for configurations that are JSON serialisable.
///
/// Implementors of this trait guarantee that the configuration is always serialisable to a JSON object.
pub trait ConfigurationSerialize: Serialize + DeserializeOwned {
    /// Convert from a configuration.
    ///
    /// ### Errors
    /// Returns a [`serde_json::Error`] if `configuration` cannot be deserialised into the concrete implementation.
    fn try_from_configuration(configuration: Configuration) -> Result<Self, serde_json::Error> {
        serde_json::from_value(serde_json::Value::Object(configuration.0))
    }
}

/// An invalid configuration error.
#[derive(Clone, Debug, Error, From)]
#[error("{name} is unsupported, configuration: {configuration:?}")]
pub struct ConfigurationError {
    name: String,
    configuration: Option<Configuration>,
}

impl ConfigurationError {
    /// Create a new invalid configuration error.
    #[must_use]
    pub fn new(name: String, configuration: Option<Configuration>) -> Self {
        Self {
            name,
            configuration,
        }
    }

    /// Return the name of the invalid configuration.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the underlying configuration metadata of the invalid configuration.
    #[must_use]
    pub const fn configuration(&self) -> Option<&Configuration> {
        self.configuration.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use v3::{AdditionalFieldV3, AdditionalFieldsV3, MetadataV3};

    use super::*;

    #[test]
    fn metadata() {
        let metadata = MetadataV3::try_from(r#""bytes""#);
        assert!(metadata.is_ok());
        assert_eq!(metadata.unwrap().to_string(), r"bytes");
        assert!(MetadataV3::try_from(r#"{ "name": "bytes" }"#).is_ok());
        let metadata =
            MetadataV3::try_from(r#"{ "name": "bytes", "configuration": { "endian": "little" } }"#);
        assert!(metadata.is_ok());
        let metadata = metadata.unwrap();
        assert_eq!(metadata.to_string(), r#"bytes {"endian":"little"}"#);
        assert_eq!(metadata.name(), "bytes");
        assert!(metadata.configuration().is_some());
        let configuration = metadata.configuration().unwrap();
        assert!(configuration.contains_key("endian"));
        assert_eq!(
            configuration.get("endian").unwrap().as_str().unwrap(),
            "little"
        );
        assert_eq!(
            MetadataV3::try_from(r#"{ "name": "bytes", "invalid": { "endian": "little" } }"#)
                .unwrap_err()
                .to_string(),
            r#"Expected metadata "<name>" or {"name":"<name>"} or {"name":"<name>","configuration":{}}"#
        );
        let metadata =
            MetadataV3::try_from(r#"{ "name": "bytes", "configuration": { "endian": "little" } }"#)
                .unwrap();
        let mut configuration = serde_json::Map::new();
        configuration.insert("endian".to_string(), "little".into());
        assert_eq!(metadata.configuration(), Some(&configuration.into()));
    }

    #[test]
    fn additional_fields_constructors() {
        let additional_field = serde_json::Map::new();
        let additional_field: AdditionalFieldV3 = additional_field.into();
        assert!(additional_field.must_understand());
        assert!(
            additional_field.as_value() == &serde_json::Value::Object(serde_json::Map::default())
        );
        assert!(serde_json::to_string(&additional_field).unwrap() == r#"{"must_understand":true}"#);

        let additional_field = AdditionalFieldV3::new("test", true);
        assert!(additional_field.must_understand());
        assert!(additional_field.as_value() == &serde_json::Value::String("test".to_string()));
        assert!(serde_json::to_string(&additional_field).unwrap() == r#""test""#);

        let additional_field = AdditionalFieldV3::new(123, false);
        assert!(!additional_field.must_understand());
        assert!(
            additional_field.as_value()
                == &serde_json::Value::Number(serde_json::Number::from(123))
        );
        assert!(serde_json::to_string(&additional_field).unwrap() == "123");
    }

    #[test]
    fn additional_fields_valid() {
        let json = r#"{
            "unknown_field": {
                "key": "value",
                "must_understand": false
            },
            "unsupported_field_1": {
                "key": "value",
                "must_understand": true
            },
            "unsupported_field_2": {
                "key": "value"
            },
            "unsupported_field_3": [],
            "unsupported_field_4": "test"
        }"#;
        let additional_fields = serde_json::from_str::<AdditionalFieldsV3>(json).unwrap();
        assert!(additional_fields.len() == 5);
        assert!(!additional_fields["unknown_field"].must_understand());
        assert!(additional_fields["unsupported_field_1"].must_understand());
        assert!(additional_fields["unsupported_field_2"].must_understand());
        assert!(additional_fields["unsupported_field_3"].must_understand());
        assert!(additional_fields["unsupported_field_4"].must_understand());
    }
}
